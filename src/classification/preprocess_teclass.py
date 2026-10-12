"""
src/classification/preprocess_teclass.py
========================================
Phase 2 — Preprocessing pipeline for Telugu Headline Relevance Classification.

Features:
  - Validates and filters required columns (article, headline, label).
  - Validates label against verified set: {HREL, MREL, LREL}.
  - Formats inputs for text classification:
      Option A (Feature/TF-IDF): "headline [SEP] article"
      Option B (Transformer): headline and article pair
  - Encodes labels:
      HREL -> 0 (Highly Related)
      MREL -> 1 (Moderately Related)
      LREL -> 2 (Least Related)
  - Prepares clean parquet / jsonl files for both training and evaluation.

Run:
    python src/classification/preprocess_teclass.py
"""

import argparse
import json
import logging
import pathlib
import sys
import pandas as pd

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
TECLASS_DIR = PROJECT_ROOT / "data" / "teclass"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "teclass"

LABEL_TO_ID = {"HREL": 0, "MREL": 1, "LREL": 2}
ID_TO_LABEL = {0: "HREL", 1: "MREL", 2: "LREL"}
LABEL_EXPLANATION = {
    "HREL": "Highly Related — The headline accurately captures the core factual main event (FME) of the news article.",
    "MREL": "Moderately Related — The headline focuses on secondary events (FSE), strong conclusions (STC), or weak takeaways (WKC).",
    "LREL": "Least Related — The headline is sensational (SEN), clickbait (CBT), misleading (MLC), or unrelated (USO)."
}


def clean_str(val) -> str:
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def preprocess_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    records = []
    skipped = 0

    for idx, row in df.iterrows():
        art = clean_str(row.get("article"))
        hl = clean_str(row.get("headline"))
        lbl = clean_str(row.get("label"))
        rec_id = clean_str(row.get("id"))

        if not art or not hl:
            skipped += 1
            continue

        if lbl not in LABEL_TO_ID:
            skipped += 1
            continue

        records.append({
            "id": rec_id,
            "article": art,
            "headline": hl,
            "label_str": lbl,
            "label": LABEL_TO_ID[lbl],
            "combined_text": f"{hl} [SEP] {art}"
        })

    logger.info(f"Split '{split_name}': {len(records)} retained, {skipped} skipped.")
    return pd.DataFrame(records)


def preprocess_all(data_dir: pathlib.Path, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": data_dir / "TeClass_training_data.csv",
        "validation": data_dir / "TeClass_development_data.csv",
        "test": data_dir / "TeClass_testing_data.csv"
    }

    stats = {}
    for split_name, path in splits.items():
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue

        df_raw = pd.read_csv(path)
        df_clean = preprocess_dataframe(df_raw, split_name)

        out_parquet = output_dir / f"{split_name}.parquet"
        out_jsonl = output_dir / f"{split_name}.jsonl"

        df_clean.to_parquet(out_parquet, index=False)
        df_clean.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)

        stats[split_name] = {
            "count": len(df_clean),
            "label_counts": df_clean["label_str"].value_counts().to_dict(),
            "parquet_path": str(out_parquet),
            "jsonl_path": str(out_jsonl),
        }

    config = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "label_explanation": LABEL_EXPLANATION,
        "splits": stats,
    }
    with open(output_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Preprocessing completed. Metadata saved to {output_dir / 'preprocess_config.json'}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(TECLASS_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    preprocess_all(pathlib.Path(args.data_dir), pathlib.Path(args.output_dir))


if __name__ == "__main__":
    main()
