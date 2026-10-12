"""
src/experiments/evaluate_low_resource.py
========================================
Phase 4 — Controlled Evaluation of Low-Resource Models.

Evaluates trained checkpoints across data proportions (5%, 10%, 25%, 50%, 100%)
against the identical validation dataset (50 samples).

Computes:
  - ROUGE-1, ROUGE-2, ROUGE-L (IndicTokenizer)
  - chrF
  - Average inference latency
  - Peak VRAM
  - Relevance classification using Phase 2 mBERT classifier

Outputs:
  - outputs/metrics/low_resource_results.json
  - outputs/metrics/low_resource_summary.csv
"""

import argparse
import datetime
import json
import logging
import pathlib
import sys
import time
import pandas as pd

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

from src.evaluation.evaluate import compute_rouge, compute_chrf, generate_headline
from src.evaluation.compare_models import classify_headlines


def evaluate_all(n_samples: int = 50, split: str = "validation"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

    if not torch.cuda.is_available():
        logger.error("[BLOCKED] CUDA required for evaluation.")
        sys.exit(1)

    device = torch.device("cuda")

    # 1. Load validation data
    val_file = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    records = []
    with open(val_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            records.append(json.loads(line.strip()))

    articles = [r["original_article"] for r in records]
    references = [r["original_headline"] for r in records]
    logger.info(f"Loaded {len(records)} evaluation samples from {val_file.name}")

    # 2. Load Phase 2 Classifier
    classifier_bundle = None
    classifier_path = PROJECT_ROOT / "checkpoints" / "classifier_bert" / "final"
    if classifier_path.exists():
        clf_tok = AutoTokenizer.from_pretrained(str(classifier_path))
        clf_mod = AutoModelForSequenceClassification.from_pretrained(str(classifier_path)).to(device)
        clf_mod.eval()
        classifier_bundle = (clf_mod, clf_tok, device, {0: "HREL", 1: "MREL", 2: "LREL"})

    proportions = [5, 10, 25, 50, 100]
    results = {}
    rows = []

    config = {"source_prefix": "summarize: ", "max_input_length": 512, "max_target_length": 64}

    for pct in proportions:
        ckpt_path = PROJECT_ROOT / "checkpoints" / "low_resource" / f"{pct}pct" / "final"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint for {pct}% not found at {ckpt_path}. Skipping.")
            continue

        logger.info(f"\nEvaluating {pct}% model from {ckpt_path}...")
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(str(ckpt_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt_path)).to(device)
        model.eval()

        preds = []
        times = []
        for art in articles:
            t0 = time.time()
            pred = generate_headline(model, tok, art, config, device)
            times.append(time.time() - t0)
            preds.append(pred)

        rouge = compute_rouge(preds, references)
        chrf = compute_chrf(preds, references)
        vram = torch.cuda.max_memory_allocated() // (1024 ** 2)
        rel = classify_headlines(classifier_bundle, articles, preds)

        avg_lat = round(sum(times) / len(times), 4) if times else 0.0

        # Read training meta if exists
        meta_file = ckpt_path / "experiment_meta.json"
        train_s = None
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
                train_s = meta.get("training_time_s")

        results[f"{pct}pct"] = {
            "proportion_pct": pct,
            "rouge": rouge,
            "chrf": chrf,
            "avg_latency_s": avg_lat,
            "peak_vram_mib": vram,
            "training_time_s": train_s,
            "relevance_distribution": rel,
        }

        rows.append({
            "Proportion (%)": pct,
            "ROUGE-1": rouge.get("rouge1", 0.0),
            "ROUGE-2": rouge.get("rouge2", 0.0),
            "ROUGE-L": rouge.get("rougeL", 0.0),
            "chrF": chrf,
            "HREL Count": rel.get("HREL", 0),
            "MREL Count": rel.get("MREL", 0),
            "LREL Count": rel.get("LREL", 0),
            "Latency (s)": avg_lat,
            "Peak VRAM (MiB)": vram,
            "Train Time (s)": train_s,
        })

        del model
        del tok
        torch.cuda.empty_cache()

    # Save outputs
    out_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "low_resource_results.json"
    csv_path = out_dir / "low_resource_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info(f"\nSaved evaluation metrics to: {json_path}")
    logger.info(f"Saved evaluation summary table to: {csv_path}")
    logger.info("\n" + df.to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate low-resource trained models")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    evaluate_all(n_samples=args.n_samples, split=args.split)


if __name__ == "__main__":
    main()
