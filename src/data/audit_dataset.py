"""
src/data/audit_dataset.py
=========================
Phase 1 — Comprehensive dataset audit.

Analyses the downloaded dataset for:
  - Missing values (null article or headline)
  - Exact duplicate article-headline pairs
  - Near-duplicate / leakage between splits (exact headline match)
  - Article and headline length distributions
  - Telugu text quality (Unicode range check)

Run:
    python src/data/audit_dataset.py [--raw-dir data/raw] [--output-dir outputs/metrics]
"""

import argparse
import json
import logging
import pathlib
import sys
import datetime
from collections import defaultdict

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


def is_telugu(text: str, threshold: float = 0.3) -> bool:
    """Return True if >= threshold fraction of chars are Telugu Unicode (0x0C00–0x0C7F)."""
    if not text:
        return False
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    return telugu_chars / len(text) >= threshold


def compute_length_stats(values: list) -> dict:
    """Compute basic length statistics for a list of strings."""
    import numpy as np
    lengths = [len(str(v)) if v else 0 for v in values]
    if not lengths:
        return {}
    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def audit_dataset(raw_dir: pathlib.Path, output_dir: pathlib.Path) -> dict:
    """Load parquet splits and produce a full audit report."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "audit_timestamp": datetime.datetime.now().isoformat(),
        "raw_dir": str(raw_dir),
        "splits": {},
        "leakage": {},
        "overall_warnings": [],
    }

    split_data = {}
    for split_name in ["train", "validation", "test"]:
        parquet_path = raw_dir / f"{split_name}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet not found: {parquet_path} — skipping split '{split_name}'")
            continue

        logger.info(f"\n--- Auditing split: {split_name} ---")
        df = pd.read_parquet(parquet_path)
        split_data[split_name] = df
        n_total = len(df)
        logger.info(f"  Rows: {n_total:,}")
        logger.info(f"  Columns: {list(df.columns)}")

        split_report = {"n_total": n_total, "warnings": []}

        # Missing values
        null_article = df["article"].isna().sum() if "article" in df.columns else n_total
        null_headline = df["headline"].isna().sum() if "headline" in df.columns else n_total
        empty_article = (df["article"].fillna("").str.strip() == "").sum() if "article" in df.columns else 0
        empty_headline = (df["headline"].fillna("").str.strip() == "").sum() if "headline" in df.columns else 0

        split_report["null_article"] = int(null_article)
        split_report["null_headline"] = int(null_headline)
        split_report["empty_article"] = int(empty_article)
        split_report["empty_headline"] = int(empty_headline)
        logger.info(f"  Null articles: {null_article}, Null headlines: {null_headline}")
        logger.info(f"  Empty articles: {empty_article}, Empty headlines: {empty_headline}")
        if null_article + empty_article > 0:
            split_report["warnings"].append(f"{null_article + empty_article} missing/empty articles")
        if null_headline + empty_headline > 0:
            split_report["warnings"].append(f"{null_headline + empty_headline} missing/empty headlines")

        # Exact duplicates within split
        if "article" in df.columns and "headline" in df.columns:
            dup_pairs = df.duplicated(subset=["article", "headline"], keep=False).sum()
            dup_articles = df.duplicated(subset=["article"], keep=False).sum()
            split_report["exact_dup_pairs"] = int(dup_pairs)
            split_report["dup_article_count"] = int(dup_articles)
            logger.info(f"  Exact duplicate (article, headline) pairs: {dup_pairs}")
            logger.info(f"  Duplicate articles (same article, possibly different headline): {dup_articles}")
            if dup_pairs > 0:
                split_report["warnings"].append(f"{dup_pairs} exact duplicate (article, headline) pairs")

        # Length stats
        if "article" in df.columns:
            art_stats = compute_length_stats(df["article"].tolist())
            split_report["article_char_length"] = art_stats
            logger.info(f"  Article char length: min={art_stats.get('min')}, "
                        f"mean={art_stats.get('mean'):.0f}, max={art_stats.get('max')}, "
                        f"p5={art_stats.get('p5'):.0f}, p95={art_stats.get('p95'):.0f}")

        if "headline" in df.columns:
            hl_stats = compute_length_stats(df["headline"].tolist())
            split_report["headline_char_length"] = hl_stats
            logger.info(f"  Headline char length: min={hl_stats.get('min')}, "
                        f"mean={hl_stats.get('mean'):.0f}, max={hl_stats.get('max')}")

        # Telugu quality check (sample 500 rows)
        if "article" in df.columns:
            sample = df["article"].dropna().head(500).tolist()
            telugu_ok = sum(1 for t in sample if is_telugu(t))
            telugu_pct = 100 * telugu_ok / len(sample) if sample else 0
            split_report["telugu_quality_pct_sample500"] = round(telugu_pct, 1)
            logger.info(f"  Telugu text quality (sample 500): {telugu_pct:.1f}% pass threshold")
            if telugu_pct < 70:
                split_report["warnings"].append(f"Low Telugu quality: only {telugu_pct:.1f}% of articles pass threshold")

        report["splits"][split_name] = split_report

    # Cross-split leakage check (headline exact match)
    logger.info("\n--- Leakage checks ---")
    split_names = list(split_data.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1:]:
            if "headline" in split_data[s1].columns and "headline" in split_data[s2].columns:
                set1 = set(split_data[s1]["headline"].dropna().str.strip())
                set2 = set(split_data[s2]["headline"].dropna().str.strip())
                overlap = len(set1 & set2)
                pct = 100 * overlap / max(len(set2), 1)
                key = f"{s1}_vs_{s2}"
                report["leakage"][key] = {
                    "headline_overlap": overlap,
                    "pct_of_smaller_split": round(pct, 2),
                }
                logger.info(f"  {s1} vs {s2}: {overlap} headline overlaps ({pct:.2f}%)")
                if overlap > 0:
                    report["overall_warnings"].append(
                        f"Potential headline leakage: {s1} vs {s2}: {overlap} shared headlines"
                    )

    # Article exact leakage
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1:]:
            if "article" in split_data[s1].columns and "article" in split_data[s2].columns:
                set1 = set(split_data[s1]["article"].dropna().str.strip())
                set2 = set(split_data[s2]["article"].dropna().str.strip())
                overlap = len(set1 & set2)
                key = f"{s1}_vs_{s2}_article"
                report["leakage"][key] = {"article_overlap": overlap}
                logger.info(f"  {s1} vs {s2} article overlap: {overlap}")
                if overlap > 0:
                    report["overall_warnings"].append(
                        f"Potential article leakage: {s1} vs {s2}: {overlap} shared articles"
                    )

    # Save
    report_path = output_dir / "dataset_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"\nAudit report saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit the downloaded telugu_news_dataset")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    args = parser.parse_args()

    report = audit_dataset(
        raw_dir=pathlib.Path(args.raw_dir),
        output_dir=pathlib.Path(args.output_dir),
    )

    warnings = report.get("overall_warnings", [])
    for split_report in report.get("splits", {}).values():
        warnings.extend(split_report.get("warnings", []))

    if warnings:
        logger.warning(f"\n[WARNINGS] {len(warnings)} issue(s) found:")
        for w in warnings:
            logger.warning(f"  - {w}")
    else:
        logger.info("\n[PASS] No critical issues found.")


if __name__ == "__main__":
    main()
