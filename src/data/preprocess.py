"""
src/data/preprocess.py
======================
Phase 1 — Preprocessing pipeline for Telugu article-to-headline generation.

- Cleans text (strips whitespace, removes nulls).
- Filters out extremely short articles or headlines.
- Tokenizes with mT5 tokenizer.
- Saves processed splits to data/processed/.

Run:
    python src/data/preprocess.py [--model-name google/mt5-small] [--max-article-len 512] [--max-headline-len 64]
"""

import argparse
import json
import logging
import pathlib
import sys

# Fix Windows cp1252 for Telugu text
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
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Source and target prefix for mT5 seq2seq
SOURCE_PREFIX = "summarize: "

# Minimum character counts to consider a valid example
MIN_ARTICLE_CHARS = 50
MIN_HEADLINE_CHARS = 5


def clean_text(text) -> str:
    """Strip whitespace, return empty string for nulls."""
    if text is None:
        return ""
    return str(text).strip()


def is_valid_example(article: str, headline: str) -> bool:
    """Return True if both article and headline meet minimum quality requirements."""
    return (
        len(article) >= MIN_ARTICLE_CHARS
        and len(headline) >= MIN_HEADLINE_CHARS
    )


def preprocess_split(df, tokenizer, max_input_len: int, max_target_len: int) -> list:
    """
    Tokenize a split and return a list of dicts with:
      input_ids, attention_mask, labels, original_article, original_headline, story_id
    """
    records = []
    skipped = 0

    articles = df["article"].tolist()
    headlines = df["headline"].tolist()
    story_ids = df["story_id"].tolist() if "story_id" in df.columns else [None] * len(articles)

    for story_id, article, headline in zip(story_ids, articles, headlines):
        art = clean_text(article)
        hl = clean_text(headline)

        if not is_valid_example(art, hl):
            skipped += 1
            continue

        # Tokenize input
        source = SOURCE_PREFIX + art
        model_inputs = tokenizer(
            source,
            max_length=max_input_len,
            truncation=True,
            padding=False,
        )

        # Tokenize target
        target_enc = tokenizer(
            text_target=hl,
            max_length=max_target_len,
            truncation=True,
            padding=False,
        )

        labels = target_enc["input_ids"]

        records.append({
            "story_id": story_id,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "original_article": art,
            "original_headline": hl,
        })

    logger.info(f"  Processed: {len(records)} valid, {skipped} skipped")
    return records


def main():
    parser = argparse.ArgumentParser(description="Preprocess telugu_news_dataset for mT5")
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--max-article-len", type=int, default=512)
    parser.add_argument("--max-headline-len", type=int, default=64)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / ".cache" / "huggingface"))
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    processed_dir = pathlib.Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )

    stats = {}
    for split_name in ["train", "validation", "test"]:
        parquet_path = raw_dir / f"{split_name}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Skipping missing split: {parquet_path}")
            continue

        logger.info(f"\nPreprocessing {split_name}...")
        df = pd.read_parquet(parquet_path)

        records = preprocess_split(df, tokenizer, args.max_article_len, args.max_headline_len)

        # Save as JSON lines for easy inspection and random access
        out_path = processed_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stats[split_name] = {"n_processed": len(records), "out_path": str(out_path)}
        logger.info(f"  Saved {len(records)} records to {out_path}")

    # Save preprocessing config
    config = {
        "model_name": args.model_name,
        "max_article_len": args.max_article_len,
        "max_headline_len": args.max_headline_len,
        "source_prefix": SOURCE_PREFIX,
        "min_article_chars": MIN_ARTICLE_CHARS,
        "min_headline_chars": MIN_HEADLINE_CHARS,
        "split_stats": stats,
    }
    config_path = processed_dir / "preprocess_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info(f"\nPreprocessing config saved to: {config_path}")
    logger.info("\n[PASS] Preprocessing complete.")


if __name__ == "__main__":
    main()
