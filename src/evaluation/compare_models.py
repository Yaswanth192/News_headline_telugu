"""
src/evaluation/compare_models.py
================================
Phase 3 — Controlled Headline Generation Model Comparison.

Compares:
  - mT5-small (checkpoints/mt5_small/final)
  - IndicBART (checkpoints/indicbart/final)

Evaluates both on the identical validation split subset using:
  - ROUGE-1, ROUGE-2, ROUGE-L (with IndicTokenizer)
  - chrF (character n-gram metric)
  - Average inference latency (seconds per sample)
  - Peak VRAM footprint during inference
  - Relevance classification using Phase 2 mBERT classifier:
    HREL (High), MREL (Medium), LREL (Low) proportions.

Saves:
  - outputs/metrics/model_comparison_phase3.json
  - outputs/predictions/comparison_predictions.jsonl
"""

import argparse
import datetime
import json
import logging
import pathlib
import sys
import time

# Windows UTF-8 stdout
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


def generate_indicbart_headline(model, tokenizer, article: str, device) -> str:
    import torch
    import re
    # IndicBART prompt with <2te>
    source = article + " <2te>"
    inputs = tokenizer(
        source,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    inputs.pop("token_type_ids", None)

    gen_kwargs = {
        "max_new_tokens": 64,
        "num_beams": 4,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
    }
    if hasattr(tokenizer, "lang_code_to_id") and "<2te>" in tokenizer.lang_code_to_id:
        gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["<2te>"]

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    text = re.sub(r"\[CLS\]|\[SEP\]|<2te>", "", text).strip()
    return text


def classify_headlines(classifier_bundle, articles, headlines):
    """Run Phase 2 mBERT classifier over (article, headline) pairs."""
    if classifier_bundle is None:
        return {"HREL": 0, "MREL": 0, "LREL": 0, "total": 0}

    model, tokenizer, device, id_to_label = classifier_bundle
    import torch

    counts = {"HREL": 0, "MREL": 0, "LREL": 0}
    for art, hl in zip(articles, headlines):
        inputs = tokenizer(
            art,
            hl,
            max_length=256,
            truncation="longest_first",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            label = id_to_label.get(pred_id, "UNKNOWN")
            if label in counts:
                counts[label] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Controlled Model Comparison: mT5-small vs IndicBART")
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

    if not torch.cuda.is_available():
        logger.error("[BLOCKED] CUDA required for model comparison.")
        sys.exit(1)

    device = torch.device("cuda")

    # 1. Load validation data
    val_file = PROJECT_ROOT / "data" / "processed" / f"{args.split}.jsonl"
    records = []
    with open(val_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n_examples:
                break
            records.append(json.loads(line.strip()))

    articles = [r["original_article"] for r in records]
    references = [r["original_headline"] for r in records]
    logger.info(f"Loaded {len(records)} evaluation samples.")

    # 2. Load Phase 2 Classifier
    classifier_bundle = None
    classifier_path = PROJECT_ROOT / "checkpoints" / "classifier_bert" / "final"
    if classifier_path.exists():
        logger.info(f"Loading Phase 2 classifier from: {classifier_path}")
        clf_tokenizer = AutoTokenizer.from_pretrained(str(classifier_path))
        clf_model = AutoModelForSequenceClassification.from_pretrained(str(classifier_path)).to(device)
        clf_model.eval()
        id_to_label = {0: "HREL", 1: "MREL", 2: "LREL"}
        classifier_bundle = (clf_model, clf_tokenizer, device, id_to_label)

    results = {}
    saved_preds = []

    # ── Evaluate mT5-small ──────────────────────────────────────────────────
    mt5_path = PROJECT_ROOT / "checkpoints" / "mt5_small" / "final"
    if mt5_path.exists():
        logger.info(f"\nEvaluating mT5-small from {mt5_path}...")
        torch.cuda.reset_peak_memory_stats()
        mt5_tokenizer = AutoTokenizer.from_pretrained(str(mt5_path))
        mt5_model = AutoModelForSeq2SeqLM.from_pretrained(str(mt5_path)).to(device)
        mt5_model.eval()

        mt5_preds = []
        mt5_times = []
        config = {"source_prefix": "summarize: ", "max_input_length": 512, "max_target_length": 64}

        for art in articles:
            t0 = time.time()
            pred = generate_headline(mt5_model, mt5_tokenizer, art, config, device)
            mt5_times.append(time.time() - t0)
            mt5_preds.append(pred)

        mt5_rouge = compute_rouge(mt5_preds, references)
        mt5_chrf = compute_chrf(mt5_preds, references)
        mt5_vram = torch.cuda.max_memory_allocated() // (1024 ** 2)
        mt5_rel = classify_headlines(classifier_bundle, articles, mt5_preds)

        results["mT5-small"] = {
            "rouge": mt5_rouge,
            "chrf": mt5_chrf,
            "avg_latency_s": round(sum(mt5_times) / len(mt5_times), 4),
            "peak_vram_mib": mt5_vram,
            "relevance_distribution": mt5_rel,
        }

        # Clear mT5 model from GPU memory
        del mt5_model
        del mt5_tokenizer
        torch.cuda.empty_cache()

    # ── Evaluate IndicBART ──────────────────────────────────────────────────
    indicbart_path = PROJECT_ROOT / "checkpoints" / "indicbart" / "final"
    if indicbart_path.exists():
        logger.info(f"\nEvaluating IndicBART from {indicbart_path}...")
        torch.cuda.reset_peak_memory_stats()
        ib_tokenizer = AutoTokenizer.from_pretrained(str(indicbart_path))
        ib_model = AutoModelForSeq2SeqLM.from_pretrained(str(indicbart_path)).to(device)
        ib_model.eval()

        ib_preds = []
        ib_times = []
        for art in articles:
            t0 = time.time()
            pred = generate_indicbart_headline(ib_model, ib_tokenizer, art, device)
            ib_times.append(time.time() - t0)
            ib_preds.append(pred)

        ib_rouge = compute_rouge(ib_preds, references)
        ib_chrf = compute_chrf(ib_preds, references)
        ib_vram = torch.cuda.max_memory_allocated() // (1024 ** 2)
        ib_rel = classify_headlines(classifier_bundle, articles, ib_preds)

        results["IndicBART"] = {
            "rouge": ib_rouge,
            "chrf": ib_chrf,
            "avg_latency_s": round(sum(ib_times) / len(ib_times), 4),
            "peak_vram_mib": ib_vram,
            "relevance_distribution": ib_rel,
        }

        del ib_model
        del ib_tokenizer
        torch.cuda.empty_cache()

    # ── Save Outputs ────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "model_comparison_phase3.json"

    comparison_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_samples": len(records),
        "split": args.split,
        "models": results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Phase 3 Controlled Comparison Summary:")
    logger.info(json.dumps(comparison_report, indent=2, ensure_ascii=False))
    logger.info("=" * 60)
    logger.info(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
