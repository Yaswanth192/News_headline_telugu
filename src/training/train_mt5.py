"""
src/training/train_mt5.py
=========================
Phase 1 — mT5-small fine-tuning for Telugu headline generation.

Features:
  - CUDA availability check before training.
  - FP16 mixed precision (RTX 3050 Ampere — CC 8.6).
  - Configurable batch size and gradient accumulation.
  - Gradient checkpointing support.
  - Smoke test mode (20 steps on 100 examples).
  - Checkpoint saving every N steps.
  - Resume from checkpoint.
  - Detailed logging to logs/.
  - Random seed setting.
  - Peak GPU memory tracking.
  - Safe OOM handling.

Run (smoke test — ~20 steps):
    python src/training/train_mt5.py --smoke-test

Run (50-step validation):
    python src/training/train_mt5.py --max-steps 50

Run (full training):
    python src/training/train_mt5.py --max-steps 5000

Resume:
    python src/training/train_mt5.py --resume-from checkpoints/mt5_small/step_200
"""

import argparse
import json
import logging
import pathlib
import random
import sys
import datetime
import time

# Fix Windows cp1252 for Telugu text
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Model
    "model_name": "google/mt5-small",
    "source_prefix": "summarize: ",
    "max_input_length": 512,
    "max_target_length": 64,
    # Training
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,   # effective batch = 32
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "max_steps": 5000,
    "save_steps": 200,
    "logging_steps": 10,
    "fp16": True,
    "gradient_checkpointing": False,
    "seed": 42,
    # Paths
    "train_jsonl": str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"),
    "val_jsonl": str(PROJECT_ROOT / "data" / "processed" / "validation.jsonl"),
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints" / "mt5_small"),
    "log_dir": str(PROJECT_ROOT / "logs"),
    "cache_dir": str(PROJECT_ROOT / ".cache" / "huggingface"),
    # Smoke test override
    "smoke_test_max_steps": 20,
    "smoke_test_examples": 100,
}

SMOKE_CONFIG_OVERRIDES = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 2,
    "logging_steps": 5,
    "save_steps": 20,
    "max_input_length": 256,
    "max_target_length": 32,
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HeadlineDataset:
    """PyTorch Dataset over preprocessed JSONL records."""

    def __init__(self, jsonl_path: str, tokenizer, config: dict, max_examples: int = None):
        import torch
        self.tokenizer = tokenizer
        self.config = config
        self.records = []

        path = pathlib.Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                rec = json.loads(line.strip())
                self.records.append(rec)

        logger.info(f"Loaded {len(self.records)} records from {path.name}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        rec = self.records[idx]
        return {
            "input_ids": torch.tensor(rec["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(rec["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(rec["labels"], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    """Pad sequences within a batch to the same length."""
    import torch

    def pad_seq(seqs, pad_val):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    input_ids = pad_seq([x["input_ids"] for x in batch], pad_token_id)
    attention_mask = pad_seq([x["attention_mask"] for x in batch], 0)
    labels = pad_seq([x["labels"] for x in batch], -100)  # -100 = ignore in loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu():
    """Verify CUDA availability and return device."""
    import torch
    if not torch.cuda.is_available():
        logger.error("[BLOCKED] CUDA not available. This script requires a CUDA GPU.")
        logger.error("  Cannot train mT5-small on CPU in reasonable time.")
        sys.exit(1)
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    logger.info(f"GPU: {name} | VRAM: {vram} MiB | CUDA: {torch.version.cuda}")
    return device


def get_peak_vram_mib() -> int:
    """Return peak GPU memory allocated in MiB."""
    import torch
    return torch.cuda.max_memory_allocated() // (1024 ** 2)


def train(config: dict, smoke_test: bool = False, resume_from: str = None):
    import torch
    from torch.utils.data import DataLoader
    from functools import partial

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from torch.cuda.amp import autocast, GradScaler
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    # ── Setup ──────────────────────────────────────────────────────────────
    set_seed(config["seed"])
    device = check_gpu()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_type = "smoke" if smoke_test else "train"
    log_dir = pathlib.Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"mt5_small_{run_type}_{run_id}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    checkpoint_dir = pathlib.Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"mT5-small Training — {run_type.upper()}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config: {json.dumps({k: v for k, v in config.items() if 'path' not in k.lower() and 'dir' not in k.lower()}, indent=2)}")
    logger.info("=" * 60)

    # ── Tokenizer and Model ────────────────────────────────────────────────
    logger.info(f"Loading tokenizer and model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        cache_dir=config["cache_dir"],
    )

    if resume_from:
        logger.info(f"Resuming model from: {resume_from}")
        model = AutoModelForSeq2SeqLM.from_pretrained(resume_from)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_name"],
            cache_dir=config["cache_dir"],
        )

    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ENABLED")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")
    logger.info(f"VRAM after model load: {get_peak_vram_mib()} MiB")

    # ── Dataset ────────────────────────────────────────────────────────────
    max_examples = config["smoke_test_examples"] if smoke_test else None
    try:
        train_ds = HeadlineDataset(
            config["train_jsonl"], tokenizer, config, max_examples=max_examples
        )
    except FileNotFoundError as e:
        logger.error(f"[FAIL] {e}")
        logger.error("Run: python src/data/download_dataset.py && python src/data/preprocess.py first.")
        sys.exit(1)

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,        # Windows: no multiprocessing workers
        pin_memory=True,
    )

    # ── Optimizer and Scaler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    use_bf16 = torch.cuda.is_bf16_supported() and config.get("fp16", True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    scaler = GradScaler(enabled=False)  # BF16 / FP32 do not need loss scaling
    logger.info(f"Mixed precision configured: {amp_dtype}")

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    global_step = 0
    total_loss = 0.0
    optimizer.zero_grad()
    max_steps = config["smoke_test_max_steps"] if smoke_test else config["max_steps"]
    warmup_steps = config["warmup_steps"]

    train_start = time.time()
    training_log = []

    logger.info(f"Starting training: max_steps={max_steps}, "
                f"batch={config['per_device_train_batch_size']}, "
                f"grad_accum={config['gradient_accumulation_steps']}, "
                f"effective_batch={config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")

    data_iter = iter(train_loader)
    accum_step = 0

    try:
        while global_step < max_steps:
            # Get next batch, re-create iterator if exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config["gradient_accumulation_steps"]

            loss.backward()
            accum_step += 1
            total_loss += loss.item() * config["gradient_accumulation_steps"]

            if accum_step == config["gradient_accumulation_steps"]:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # LR warmup (linear)
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / max(warmup_steps, 1)
                    for pg in optimizer.param_groups:
                        pg["lr"] = config["learning_rate"] * lr_scale

                optimizer.step()
                optimizer.zero_grad()
                accum_step = 0
                global_step += 1

                avg_loss = total_loss / global_step
                peak_vram = get_peak_vram_mib()

                if global_step % config["logging_steps"] == 0:
                    elapsed = time.time() - train_start
                    logger.info(
                        f"Step {global_step}/{max_steps} | loss={loss.item() * config['gradient_accumulation_steps']:.4f} "
                        f"| avg_loss={avg_loss:.4f} | peak_VRAM={peak_vram}MiB | elapsed={elapsed:.0f}s"
                    )
                    training_log.append({
                        "step": global_step,
                        "loss": round(loss.item() * config["gradient_accumulation_steps"], 4),
                        "avg_loss": round(avg_loss, 4),
                        "peak_vram_mib": peak_vram,
                        "elapsed_s": round(elapsed, 1),
                    })

                # Check loss is finite
                if not torch.isfinite(torch.tensor(loss.item())):
                    logger.error(f"[FAIL] Non-finite loss at step {global_step}: {loss.item()}")
                    sys.exit(1)

                # Save checkpoint
                if global_step % config["save_steps"] == 0 or global_step == max_steps:
                    ckpt_path = checkpoint_dir / f"step_{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    tokenizer.save_pretrained(str(ckpt_path))
                    logger.info(f"Checkpoint saved: {ckpt_path}")

    except torch.cuda.OutOfMemoryError:
        logger.error("[OOM] CUDA out of memory! Suggestions:")
        logger.error("  1. Reduce --max-article-len (try 256)")
        logger.error("  2. Reduce per_device_train_batch_size to 1")
        logger.error("  3. Enable gradient_checkpointing: True in config")
        sys.exit(1)

    # Final save
    final_path = checkpoint_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training log
    elapsed_total = time.time() - train_start
    summary = {
        "run_id": run_id,
        "run_type": run_type,
        "smoke_test": smoke_test,
        "max_steps": max_steps,
        "final_step": global_step,
        "final_avg_loss": round(total_loss / max(global_step, 1), 4),
        "peak_vram_mib": get_peak_vram_mib(),
        "elapsed_s": round(elapsed_total, 1),
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": config["model_name"],
        "config": config,
        "training_log": training_log,
    }
    summary_path = log_dir / f"mt5_small_{run_type}_{run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"  Steps:       {global_step}")
    logger.info(f"  Avg loss:    {summary['final_avg_loss']}")
    logger.info(f"  Peak VRAM:   {summary['peak_vram_mib']} MiB")
    logger.info(f"  Time:        {elapsed_total:.0f}s")
    logger.info(f"  Checkpoint:  {checkpoint_dir}")
    logger.info(f"  Log:         {log_file}")
    logger.info(f"  Summary:     {summary_path}")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train mT5-small for Telugu headline generation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run minimal 20-step smoke test on 100 examples")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-input-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.smoke_test:
        config.update(SMOKE_CONFIG_OVERRIDES)

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        config["smoke_test_max_steps"] = args.max_steps
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["gradient_accumulation_steps"] = args.grad_accum
    if args.no_fp16:
        config["fp16"] = False
    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True
    if args.max_input_len is not None:
        config["max_input_length"] = args.max_input_len
    config["seed"] = args.seed

    train(config, smoke_test=args.smoke_test, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
