"""
src/training/train_indicbart.py
==============================
Phase 3 — IndicBART fine-tuning for Telugu headline generation on RTX 3050.

Uses ai4bharat/IndicBART (BART-based multilingual model for Indic languages).
Telugu language tag: <2te>
Runs on local CUDA GPU with batch=1, gradient accumulation=16 (effective batch 16).
"""

import argparse
import datetime
import json
import logging
import pathlib
import random
import sys
import time

# Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model_name": "ai4bharat/IndicBART",
    "lang_tag": "<2te>",
    "max_input_length": 256,
    "max_target_length": 64,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "max_steps": 50,
    "save_steps": 25,
    "logging_steps": 5,
    "seed": 42,
    "train_jsonl": str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"),
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints" / "indicbart"),
    "log_dir": str(PROJECT_ROOT / "logs"),
    "cache_dir": str(PROJECT_ROOT / ".cache" / "huggingface"),
}


class IndicBARTDataset:
    def __init__(self, jsonl_path: str, tokenizer, config: dict, max_examples: int = None):
        import torch
        self.tokenizer = tokenizer
        self.config = config
        self.records = []

        path = pathlib.Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL not found: {path}")

        lang_tag = config.get("lang_tag", "<2te>")
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                rec = json.loads(line.strip())
                # Format with IndicBART language tag
                art = rec["original_article"] + " " + lang_tag
                hl = rec["original_headline"] + " " + lang_tag

                src_enc = tokenizer(
                    art,
                    max_length=config["max_input_length"],
                    truncation=True,
                    padding=False,
                )
                tgt_enc = tokenizer(
                    hl,
                    max_length=config["max_target_length"],
                    truncation=True,
                    padding=False,
                )
                self.records.append({
                    "input_ids": src_enc["input_ids"],
                    "attention_mask": src_enc["attention_mask"],
                    "labels": tgt_enc["input_ids"],
                })

        logger.info(f"Loaded {len(self.records)} records for IndicBART")

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
    import torch

    def pad_seq(seqs, pad_val):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    input_ids = pad_seq([x["input_ids"] for x in batch], pad_token_id)
    attention_mask = pad_seq([x["attention_mask"] for x in batch], 0)
    labels = pad_seq([x["labels"] for x in batch], -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train(config: dict, max_steps: int = None):
    import torch
    from torch.utils.data import DataLoader
    from functools import partial
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not torch.cuda.is_available():
        logger.error("[BLOCKED] CUDA required for IndicBART training.")
        sys.exit(1)

    device = torch.device("cuda")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = pathlib.Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"indicbart_train_{run_id}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    total_steps = max_steps or config["max_steps"]
    logger.info("=" * 60)
    logger.info(f"IndicBART Fine-Tuning | Steps: {total_steps} | Device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=config["cache_dir"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"], cache_dir=config["cache_dir"]).to(device)

    dataset = IndicBARTDataset(
        config["train_jsonl"],
        tokenizer,
        config,
        max_examples=total_steps * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * 2,
    )

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    model.train()
    optimizer.zero_grad()
    global_step = 0
    accum_loss = 0.0
    accum_steps = config["gradient_accumulation_steps"]
    train_start = time.time()
    training_log = []

    for batch_idx, batch in enumerate(loader):
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

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % config["logging_steps"] == 0 or global_step == total_steps:
                elapsed = time.time() - train_start
                step_time = elapsed / global_step
                vram_peak = torch.cuda.max_memory_allocated() // (1024 ** 2)
                logger.info(
                    f"Step {global_step:4d}/{total_steps} | "
                    f"Loss: {accum_loss / accum_steps:7.4f} | "
                    f"Step time: {step_time:4.2f}s | "
                    f"Peak VRAM: {vram_peak} MiB"
                )
                training_log.append({
                    "step": global_step,
                    "loss": round(accum_loss / accum_steps, 4),
                    "peak_vram_mib": vram_peak,
                    "elapsed_s": round(elapsed, 2),
                })
                accum_loss = 0.0

            if global_step % config["save_steps"] == 0:
                save_dir = checkpoint_dir / f"step_{global_step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                logger.info(f"Saved checkpoint to: {save_dir}")

            if global_step >= total_steps:
                break

    # Save final model
    final_dir = checkpoint_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved final checkpoint to: {final_dir}")

    meta = {
        "model_name": config["model_name"],
        "global_steps": global_step,
        "peak_vram_mib": torch.cuda.max_memory_allocated() // (1024 ** 2),
        "total_training_time_s": round(time.time() - train_start, 2),
        "training_log": training_log,
    }
    with open(final_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicBART for Telugu headline generation")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    train(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
