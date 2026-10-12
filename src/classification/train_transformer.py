"""
src/classification/train_transformer.py
=======================================
Phase 2 — Transformer Classifier on Local RTX 3050 GPU (4 GB VRAM).

Model: `ai4bharat/indic-bert` (IndicBERT), 12-layer multi-lingual transformer
specifically pre-trained on Indian languages including Telugu.
Compact footprint (~33M parameters), perfectly suited for local 4 GB VRAM RTX 3050.

Features:
  - Input: headline [SEP] article (tokenized as sentence pairs)
  - Native Ampere bfloat16 / fp16 training
  - Gradient accumulation to match effective batch size
  - Memory-safe sequence length (256/384 tokens)
  - Checkpoint saving to checkpoints/classifier_bert
  - Comprehensive metric logging (Accuracy, Macro-F1, Weighted-F1, Confusion Matrix)

Run:
    python src/classification/train_transformer.py [--smoke-test] [--max-steps 100]
"""

import argparse
import json
import logging
import pathlib
import sys
import time
import datetime
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "teclass"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "classifier_bert"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
ID2LABEL = {0: "HREL", 1: "MREL", 2: "LREL"}
LABEL2ID = {"HREL": 0, "MREL": 1, "LREL": 2}


class TeClassDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.headlines = df["headline"].tolist()
        self.articles = df["article"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hl = str(self.headlines[idx])
        art = str(self.articles[idx])
        lbl = int(self.labels[idx])

        # Tokenize pair: (headline, article)
        enc = self.tokenizer(
            hl,
            art,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(lbl, dtype=torch.long)
        }


def check_gpu():
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU (training will be slow).")
        return torch.device("cpu")
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    logger.info(f"Using GPU: {name} | VRAM: {vram} MiB | CUDA: {torch.version.cuda}")
    return device


def evaluate_model(model, dataloader, device, amp_dtype):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted"))
    rep = classification_report(all_labels, all_preds, target_names=["HREL", "MREL", "LREL"], output_dict=True)
    conf = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            k: {
                "precision": round(rep[k]["precision"], 4),
                "recall": round(rep[k]["recall"], 4),
                "f1_score": round(rep[k]["f1-score"], 4),
                "support": int(rep[k]["support"])
            } for k in ["HREL", "MREL", "LREL"]
        },
        "confusion_matrix": conf
    }


def train_classifier(smoke_test: bool = False, max_steps: int = None, epochs: int = 1):
    device = check_gpu()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    logger.info(f"Loading IndicBERT tokenizer & model: {MODEL_NAME}")
    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=cache_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        cache_dir=cache_dir
    ).to(device)

    # Load data
    df_train = pd.read_parquet(DATA_DIR / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "validation.parquet")

    if smoke_test:
        logger.info("SMOKE TEST MODE: using subset of 200 training examples.")
        df_train = df_train.sample(n=min(200, len(df_train)), random_state=42).reset_index(drop=True)
        df_val = df_val.sample(n=min(100, len(df_val)), random_state=42).reset_index(drop=True)
        max_steps = max_steps or 25
    else:
        max_steps = max_steps or 200

    train_ds = TeClassDataset(df_train, tokenizer, max_len=256)
    val_ds = TeClassDataset(df_val, tokenizer, max_len=256)

    batch_size = 4
    grad_accum = 4  # Effective batch size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    logger.info(f"Starting training: max_steps={max_steps}, batch_size={batch_size}, grad_accum={grad_accum}")
    model.train()
    global_step = 0
    total_loss = 0.0
    accum_step = 0
    data_iter = iter(train_loader)

    start_time = time.time()
    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum

        loss.backward()
        accum_step += 1
        total_loss += loss.item() * grad_accum

        if accum_step == grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_step = 0
            global_step += 1

            if global_step % 10 == 0 or global_step == max_steps:
                elapsed = time.time() - start_time
                avg_loss = total_loss / global_step
                peak_vram = torch.cuda.max_memory_allocated() // (1024 ** 2) if device.type == "cuda" else 0
                logger.info(f"Step {global_step}/{max_steps} | Loss: {loss.item()*grad_accum:.4f} | Avg Loss: {avg_loss:.4f} | Peak VRAM: {peak_vram}MiB | {elapsed:.0f}s")

    # Save final transformer checkpoint
    final_ckpt = CKPT_DIR / "final"
    model.save_pretrained(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))
    logger.info(f"Saved transformer checkpoint to: {final_ckpt}")

    # Evaluate on validation split
    logger.info("Evaluating transformer on validation set...")
    val_results = evaluate_model(model, val_loader, device, amp_dtype)
    val_results["model_type"] = f"IndicBERT ({MODEL_NAME})"
    val_results["checkpoint"] = str(final_ckpt)
    val_results["smoke_test"] = smoke_test
    val_results["trained_steps"] = global_step
    val_results["peak_vram_mib"] = torch.cuda.max_memory_allocated() // (1024 ** 2) if device.type == "cuda" else 0

    logger.info(f"Validation Accuracy: {val_results['accuracy']}")
    logger.info(f"Validation Macro-F1: {val_results['macro_f1']}")
    logger.info(f"Validation Weighted-F1: {val_results['weighted_f1']}")

    eval_file = OUTPUT_DIR / "transformer_classifier_eval.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation metrics saved to: {eval_file}")

    return val_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    train_classifier(smoke_test=args.smoke_test, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
