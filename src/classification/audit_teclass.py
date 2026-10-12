"""
src/classification/audit_teclass.py
===================================
Phase 2 — Comprehensive Audit and Preprocessing Validation for TeClass Dataset.

Investigates:
  - Official Splits & Schema (id, url, article, headline, label)
  - Class distribution (HREL, MREL, LREL)
  - Missing and empty values
  - Character and word length distributions
  - Duplicate article-headline pairs
  - Cross-split leakage (ID overlap, article overlap, headline overlap)
  - Documentation of verified research taxonomy:
      HREL: Highly Relevant (Primary category covering FME - Factual Main Event)
      MREL: Moderately Relevant (Primary category covering FSE, STC, WKC)
      LREL: Least Relevant (Primary category covering SEN, CBT, MLC, USO)

Run:
    python src/classification/audit_teclass.py [--data-dir data/teclass] [--output-dir outputs/metrics]
"""

import argparse
import json
import logging
import pathlib
import sys
import datetime

# Fix Windows console UTF-8 output
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

# Verified TeClass Taxonomy
LABEL_TAXONOMY = {
    "HREL": {
        "full_name": "Highly Related",
        "description": "Headline is directly and accurately tied to the core content/main event of the article.",
        "fine_grained_subclasses": ["FME (Factual Main Event)"],
        "paper_relevance": "High Relevance"
    },
    "MREL": {
        "full_name": "Moderately Related",
        "description": "Headline is moderately related, focusing on secondary facts, takeaways, or conclusions.",
        "fine_grained_subclasses": [
            "FSE (Factual Secondary Event)",
            "STC (Strong Conclusion)",
            "WKC (Weak Conclusion)"
        ],
        "paper_relevance": "Medium Relevance"
    },
    "LREL": {
        "full_name": "Least Related",
        "description": "Headline is minimally related, misleading, sensational, clickbait, or unsupported.",
        "fine_grained_subclasses": [
            "SEN (Sensational)",
            "CBT (Clickbait)",
            "MLC (Misleading Conclusion)",
            "USO (Unsupported Opinion)"
        ],
        "paper_relevance": "Low Relevance"
    }
}

VALID_LABELS = set(LABEL_TAXONOMY.keys())


def audit_teclass(data_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
    import pandas as pd
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "paper_citation": "Kanumolu et al., LREC-COLING 2024: TeClass: A Human-Annotated Relevance-based Headline Classification and Generation Dataset for Telugu",
        "paper_url": "https://aclanthology.org/2024.lrec-main.1364/",
        "dataset_source": "https://github.com/gopichandkanumolu/TeClass",
        "license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        "taxonomy": LABEL_TAXONOMY,
        "splits": {},
        "leakage": {},
        "overall_warnings": []
    }

    split_files = {
        "train": data_dir / "TeClass_training_data.csv",
        "validation": data_dir / "TeClass_development_data.csv",
        "test": data_dir / "TeClass_testing_data.csv",
    }

    dfs = {}
    for split_name, path in split_files.items():
        if not path.exists():
            msg = f"Missing split file: {path}"
            logger.error(msg)
            report["overall_warnings"].append(msg)
            continue

        df = pd.read_csv(path)
        dfs[split_name] = df
        n_total = len(df)
        logger.info(f"Auditing {split_name}: {n_total:,} rows from {path.name}")

        # Check required columns
        req_cols = {"id", "article", "headline", "label"}
        missing_cols = req_cols - set(df.columns)
        if missing_cols:
            report["overall_warnings"].append(f"{split_name} missing columns: {missing_cols}")

        # Missing values
        null_article = int(df["article"].isna().sum())
        null_headline = int(df["headline"].isna().sum())
        null_label = int(df["label"].isna().sum())

        # Label validity and distribution
        valid_mask = df["label"].isin(VALID_LABELS)
        invalid_labels = df[~valid_mask]["label"].tolist() if not valid_mask.all() else []
        class_counts = {k: int(v) for k, v in df["label"].value_counts().items()}

        # Lengths
        art_lens = df["article"].dropna().astype(str).str.len()
        hl_lens = df["headline"].dropna().astype(str).str.len()

        # Duplicates
        dup_pairs = int(df.duplicated(subset=["article", "headline"]).sum())

        split_info = {
            "total_samples": n_total,
            "columns": list(df.columns),
            "null_article": null_article,
            "null_headline": null_headline,
            "null_label": null_label,
            "invalid_labels": invalid_labels,
            "class_distribution": class_counts,
            "duplicate_pairs_within_split": dup_pairs,
            "article_char_length": {
                "min": int(art_lens.min()) if len(art_lens) else 0,
                "mean": round(float(art_lens.mean()), 1) if len(art_lens) else 0,
                "median": int(art_lens.median()) if len(art_lens) else 0,
                "max": int(art_lens.max()) if len(art_lens) else 0,
            },
            "headline_char_length": {
                "min": int(hl_lens.min()) if len(hl_lens) else 0,
                "mean": round(float(hl_lens.mean()), 1) if len(hl_lens) else 0,
                "median": int(hl_lens.median()) if len(hl_lens) else 0,
                "max": int(hl_lens.max()) if len(hl_lens) else 0,
            }
        }
        report["splits"][split_name] = split_info

    # Leakage checks
    logger.info("Computing cross-split leakage checks...")
    for s1, s2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if s1 in dfs and s2 in dfs:
            df1, df2 = dfs[s1], dfs[s2]
            id_overlap = len(set(df1["id"].dropna()) & set(df2["id"].dropna()))
            art_overlap = len(set(df1["article"].dropna().str.strip()) & set(df2["article"].dropna().str.strip()))
            hl_overlap = len(set(df1["headline"].dropna().str.strip()) & set(df2["headline"].dropna().str.strip()))

            key = f"{s1}_vs_{s2}"
            report["leakage"][key] = {
                "id_overlap": id_overlap,
                "article_overlap": art_overlap,
                "headline_overlap": hl_overlap
            }
            logger.info(f"  {key}: id_overlap={id_overlap}, article_overlap={art_overlap}, headline_overlap={hl_overlap}")

    out_file = output_dir / "teclass_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"TeClass audit report saved to: {out_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit TeClass Dataset")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "teclass"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    args = parser.parse_args()

    audit_teclass(pathlib.Path(args.data_dir), pathlib.Path(args.output_dir))


if __name__ == "__main__":
    main()
