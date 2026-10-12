"""
src/data/download_dataset.py
============================
Phase 1 — Download and validate saidines12/telugu_news_dataset.

Run:
    python src/data/download_dataset.py [--output-dir data/raw] [--cache-dir .cache/huggingface]

This script:
    1. Downloads the dataset from Hugging Face (train/validation/test splits).
    2. Verifies columns: story_id, article, headline.
    3. Saves parquet files to data/raw/.
    4. Writes a dataset_info.json audit file.
    5. Never commits dataset files (see .gitignore).
"""

import argparse
import json
import logging
import pathlib
import sys
import datetime

# Fix Windows cp1252 console: reconfigure stdout/stderr to UTF-8
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
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DATASET_ID = "saidines12/telugu_news_dataset"
EXPECTED_COLUMNS = {"story_id", "article", "headline"}
EXPECTED_SPLITS = {"train", "validation", "test"}


def download_and_validate(output_dir: pathlib.Path, cache_dir: pathlib.Path) -> dict:
    """Download the dataset, validate schema, return audit info."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {DATASET_ID} ...")
    import os
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    try:
        ds = load_dataset(DATASET_ID, cache_dir=str(cache_dir))
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    logger.info(f"Dataset loaded. Splits: {list(ds.keys())}")

    # Validate splits
    actual_splits = set(ds.keys())
    missing_splits = EXPECTED_SPLITS - actual_splits
    if missing_splits:
        logger.warning(f"Expected splits {EXPECTED_SPLITS}, missing: {missing_splits}")

    audit = {
        "dataset_id": DATASET_ID,
        "download_timestamp": datetime.datetime.now().isoformat(),
        "splits": {},
        "columns": None,
        "schema_ok": True,
        "warnings": [],
    }

    for split_name in sorted(actual_splits):
        split_ds = ds[split_name]
        n = len(split_ds)
        cols = set(split_ds.column_names)

        # Validate columns
        missing_cols = EXPECTED_COLUMNS - cols
        extra_cols = cols - EXPECTED_COLUMNS
        if missing_cols:
            msg = f"Split '{split_name}': missing columns {missing_cols}"
            logger.error(msg)
            audit["warnings"].append(msg)
            audit["schema_ok"] = False

        audit["splits"][split_name] = {
            "n_rows": n,
            "columns": sorted(cols),
        }
        logger.info(f"  {split_name}: {n:,} rows, columns: {sorted(cols)}")

        # Set columns from first split
        if audit["columns"] is None:
            audit["columns"] = sorted(cols)

        # Show a sample
        if n > 0:
            sample = split_ds[0]
            logger.info(f"  Sample story_id: {sample.get('story_id', 'N/A')}")
            headline = sample.get("headline", "")
            article = sample.get("article", "")
            logger.info(f"  Sample headline: {str(headline)[:100]}")
            logger.info(f"  Sample article (first 150 chars): {str(article)[:150]}")
            audit["splits"][split_name]["sample_headline"] = str(headline)[:200]
            audit["splits"][split_name]["sample_article"] = str(article)[:200]

        # Save split as parquet
        out_path = output_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        logger.info(f"  Saved to {out_path}")
        audit["splits"][split_name]["parquet_path"] = str(out_path)

    # Save audit
    audit_path = output_dir / "dataset_info.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    logger.info(f"\nDataset audit saved to: {audit_path}")

    return audit


def main():
    parser = argparse.ArgumentParser(description="Download and validate telugu_news_dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    audit = download_and_validate(
        output_dir=pathlib.Path(args.output_dir),
        cache_dir=pathlib.Path(args.cache_dir),
    )

    if audit["schema_ok"]:
        logger.info("\n[PASS] Dataset download and validation complete.")
    else:
        logger.error("\n[FAIL] Dataset validation found schema issues. See warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
