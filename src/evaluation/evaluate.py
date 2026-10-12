"""
src/evaluation/evaluate.py
==========================
Phase 1 — Evaluate a trained mT5-small model on a validation/test split.

Generates predictions and computes ROUGE-1, ROUGE-2, ROUGE-L, and chrF.
Saves predictions and metrics to outputs/.

Run:
    python src/evaluation/evaluate.py --checkpoint checkpoints/mt5_small/final --split validation --n-examples 200
"""

import argparse
import json
import logging
import pathlib
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_headline(model, tokenizer, article: str, config: dict, device) -> str:
    """Generate a headline for a single article string.

    Phase 3 fix:
      1. mT5 requires decoder_start_token_id=tokenizer.pad_token_id (0).
      2. Residual sentinel tokens (<extra_id_\d+>) from partial training are explicitly
         stripped using regex so clean headline text is yielded.
    """
    import torch
    import re
    source = config.get("source_prefix", "summarize: ") + article
    inputs = tokenizer(
        source,
        max_length=config.get("max_input_length", 512),
        truncation=True,
        return_tensors="pt",
    ).to(device)

    decoder_start = getattr(tokenizer, "pad_token_id", 0) or 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get("max_target_length", 64),
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            decoder_start_token_id=decoder_start,
            forced_bos_token_id=None,
        )

    text = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip()

    # Strip sentinel tokens like <extra_id_0>, <extra_id_1>, etc.
    text = re.sub(r"<extra_id_\d+>", "", text).strip()
    return text


def compute_rouge(predictions: list, references: list) -> dict:
    """Compute ROUGE-1/2/L scores using Indic/whitespace tokenizer.

    Phase 3 fix: The default rouge_score tokenizer splits only on ASCII word boundaries
    `re.findall(r"\w+", text)` which drops all non-Latin Unicode characters, yielding 0 tokens
    for Telugu text! We provide an Indic-aware whitespace/punctuation tokenizer.
    """
    try:
        import re
        from rouge_score import rouge_scorer, tokenizers

        class IndicTokenizer(tokenizers.Tokenizer):
            def tokenize(self, text):
                clean = re.sub(r'[\.,\?!;:\"\'\(\)\[\]\{\}\<\>_]', ' ', text)
                return [w for w in clean.strip().split() if w]

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            tokenizer=IndicTokenizer(),
            use_stemmer=False,
        )
    except ImportError:
        logger.warning("rouge-score not installed. Run: pip install rouge-score")
        return {}

    r1_list, r2_list, rL_list = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rL_list.append(scores["rougeL"].fmeasure)

    n = len(predictions)
    return {
        "rouge1": round(sum(r1_list) / n, 4) if n > 0 else 0,
        "rouge2": round(sum(r2_list) / n, 4) if n > 0 else 0,
        "rougeL": round(sum(rL_list) / n, 4) if n > 0 else 0,
        "n_examples": n,
    }


def compute_chrf(predictions: list, references: list) -> float:
    """Compute chrF score."""
    try:
        import sacrebleu
        result = sacrebleu.corpus_chrf(predictions, [references])
        return round(result.score, 4)
    except ImportError:
        logger.warning("sacrebleu not installed. Skipping chrF.")
        return None


def evaluate(checkpoint_path: str, split: str, n_examples: int, output_dir: pathlib.Path):
    import torch
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = pathlib.Path(checkpoint_path)
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    # GPU check
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Running inference on CPU (will be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Load model and tokenizer from checkpoint
    logger.info(f"Loading model from: {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to(device)
    model.eval()

    # Load preprocess config
    preprocess_config_path = PROJECT_ROOT / "data" / "processed" / "preprocess_config.json"
    if preprocess_config_path.exists():
        with open(preprocess_config_path, encoding="utf-8") as f:
            preprocess_config = json.load(f)
    else:
        preprocess_config = {"source_prefix": "summarize: ", "max_input_length": 512, "max_target_length": 64}
        logger.warning("preprocess_config.json not found, using defaults.")

    # Load split JSONL
    jsonl_path = PROJECT_ROOT / "data" / "processed" / f"{split}.jsonl"
    if not jsonl_path.exists():
        logger.error(f"Processed split not found: {jsonl_path}")
        sys.exit(1)

    logger.info(f"Loading {split} split from {jsonl_path}")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_examples and i >= n_examples:
                break
            records.append(json.loads(line.strip()))

    logger.info(f"Evaluating on {len(records)} examples...")
    predictions, references = [], []
    inference_times = []

    for i, rec in enumerate(records):
        article = rec["original_article"]
        reference = rec["original_headline"]

        t0 = time.time()
        pred = generate_headline(model, tokenizer, article, preprocess_config, device)
        inference_times.append(time.time() - t0)

        predictions.append(pred)
        references.append(reference)

        if i < 5:
            logger.info(f"\n  Example {i+1}:")
            logger.info(f"    Article:    {article[:100]}...")
            logger.info(f"    Reference:  {reference}")
            logger.info(f"    Predicted:  {pred}")

    # Metrics
    rouge_scores = compute_rouge(predictions, references)
    chrf_score = compute_chrf(predictions, references)

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() // (1024 ** 2)
    else:
        peak_vram = None

    avg_inf_time = sum(inference_times) / len(inference_times) if inference_times else 0

    logger.info(f"\n--- Evaluation Results ({split}) ---")
    logger.info(f"  ROUGE-1: {rouge_scores.get('rouge1', 'N/A')}")
    logger.info(f"  ROUGE-2: {rouge_scores.get('rouge2', 'N/A')}")
    logger.info(f"  ROUGE-L: {rouge_scores.get('rougeL', 'N/A')}")
    logger.info(f"  chrF:    {chrf_score}")
    logger.info(f"  Avg inference time: {avg_inf_time:.3f}s/example")
    logger.info(f"  Peak VRAM: {peak_vram} MiB" if peak_vram else "  Peak VRAM: N/A (CPU)")
    logger.info(f"  NOTE: This is a PRELIMINARY result from a smoke-test run, NOT a fully trained model.")

    # Save predictions
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = output_dir / f"predictions_mt5_{split}_{ts}.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec, pred, ref in zip(records, predictions, references):
            f.write(json.dumps({
                "story_id": rec.get("story_id"),
                "reference": ref,
                "prediction": pred,
            }, ensure_ascii=False) + "\n")

    # Save metrics
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "split": split,
        "n_examples": len(records),
        "rouge_scores": rouge_scores,
        "chrf": chrf_score,
        "peak_vram_mib": peak_vram,
        "avg_inference_time_s": round(avg_inf_time, 4),
        "result_label": "PRELIMINARY — smoke test / partial training only",
        "predictions_path": str(pred_path),
    }
    metrics_path = output_dir / f"metrics_mt5_{split}_{ts}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"\n  Predictions: {pred_path}")
    logger.info(f"  Metrics:     {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate mT5-small on Telugu headline generation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    evaluate(args.checkpoint, args.split, args.n_examples, output_dir)


if __name__ == "__main__":
    main()
