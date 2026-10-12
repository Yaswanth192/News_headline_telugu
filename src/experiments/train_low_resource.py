"""
src/experiments/train_low_resource.py
=====================================
Phase 4 — Controlled Training Pipeline for Low-Resource Telugu Headline Generation.

Trains mT5-small across different data proportions:
  - 5%   (4,178 samples)
  - 10%  (8,356 samples)
  - 25%  (20,890 samples)
  - 50%  (41,781 samples)
  - 100% (83,562 samples)

Design:
  - Consistent batch size: per_device=2, grad_accum=16 (effective batch = 32)
  - Mixed precision: bfloat16
  - Fixed steps per proportion: 75 steps (smoke-tested, ~2,400 examples processed per run)
  - Checkpointing every 25 steps, with final model saved to checkpoints/low_resource/<pct>pct/final
  - Logs throughput, training duration, and peak VRAM
"""

import argparse
import datetime
import json
import logging
import pathlib
import random
import sys
import time

# Windows UTF-8 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from src.training.train_mt5 import HeadlineDataset, collate_fn


def train_proportion(
    proportion_pct: int,
    max_steps: int = 75,
    seed: int = 42,
    lr: float = 5e-4,
    effective_batch_size: int = 32,
):
    import torch
    from torch.utils.data import DataLoader
    from functools import partial
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not torch.cuda.is_available():
        logger.error("[BLOCKED] CUDA required for Phase 4 low-resource experiments.")
        sys.exit(1)

    device = torch.device("cuda")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_name = "google/mt5-small"
    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    subset_path = PROJECT_ROOT / "data" / "subsets" / f"train_{proportion_pct}pct.jsonl"
    out_dir = PROJECT_ROOT / "checkpoints" / "low_resource" / f"{proportion_pct}pct"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"low_resource_{proportion_pct}pct_{run_id}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"Phase 4 Training: {proportion_pct}% Data Proportion ({subset_path.name})")
    logger.info(f"Max Steps: {max_steps} | Device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir).to(device)

    # 2. Dataset
    config = {
        "max_input_length": 512,
        "max_target_length": 64,
        "source_prefix": "summarize: ",
    }
    dataset = HeadlineDataset(str(subset_path), tokenizer, config)
    logger.info(f"Loaded {len(dataset)} examples from {subset_path.name}")

    per_device_batch = 2
    accum_steps = effective_batch_size // per_device_batch

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    optimizer.zero_grad()
    global_step = 0
    accum_loss = 0.0
    train_start = time.time()
    step_records = []

    # Iterate over dataloader (looping over epochs if necessary)
    data_iter = iter(loader)
    while global_step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / accum_steps

            loss.backward()
            accum_loss += loss.item() * accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if global_step % 10 == 0 or global_step == max_steps:
            elapsed = time.time() - train_start
            step_time = elapsed / global_step
            vram_peak = torch.cuda.max_memory_allocated() // (1024 ** 2)
            logger.info(
                f"[{proportion_pct}%] Step {global_step:3d}/{max_steps} | "
                f"Loss: {accum_loss / accum_steps:7.4f} | "
                f"Step time: {step_time:4.2f}s | "
                f"Peak VRAM: {vram_peak} MiB"
            )
            step_records.append({
                "step": global_step,
                "loss": round(accum_loss / accum_steps, 4),
                "peak_vram_mib": vram_peak,
                "elapsed_s": round(elapsed, 2),
            })

    # Save final model
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final checkpoint to: {final_dir}")

    total_time = round(time.time() - train_start, 2)
    meta = {
        "proportion_pct": proportion_pct,
        "sample_count": len(dataset),
        "steps": global_step,
        "effective_batch_size": effective_batch_size,
        "learning_rate": lr,
        "training_time_s": total_time,
        "peak_vram_mib": torch.cuda.max_memory_allocated() // (1024 ** 2),
        "step_log": step_records,
    }

    with open(final_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training for {proportion_pct}% completed in {total_time}s")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train mT5-small on a specific data proportion")
    parser.add_argument("--proportion", type=int, required=True, choices=[5, 10, 25, 50, 100])
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_proportion(args.proportion, max_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
