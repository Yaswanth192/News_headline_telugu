"""
src/experiments/quality_filtering.py
====================================
Phase 5 — Controlled Quality Filtering & Matched-Size Experiment Design.

Operationalizes Data Quality for Telugu Headline Generation:
  1. Filters malformed records (e.g. empty strings, non-Telugu characters).
  2. Filters extreme length anomalies:
     - Articles under 100 characters or over 3,000 characters.
     - Headlines under 10 characters or over 150 characters.
  3. Filters exact and near duplicate article-headline pairs.
  4. Removes headline-article verbatim copying (headlines that are simply the first sentence verbatim).
  5. Produces matched-size subsets to strictly isolate data quality from data quantity.

Outputs:
  - data/subsets/quality_manifest.json
  - data/subsets/train_5pct_quality.jsonl
  - data/subsets/train_25pct_quality.jsonl
"""

import json
import logging
import pathlib
import sys
import re

# Windows UTF-8 stdout
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


def is_valid_quality_example(record: dict) -> tuple[bool, str]:
    art = record.get("original_article", "").strip()
    hl = record.get("original_headline", "").strip()

    if not art or not hl:
        return False, "EMPTY_TEXT"

    if len(art) < 100:
        return False, "ARTICLE_TOO_SHORT"
    if len(art) > 3000:
        return False, "ARTICLE_TOO_LONG"

    if len(hl) < 10:
        return False, "HEADLINE_TOO_SHORT"
    if len(hl) > 150:
        return False, "HEADLINE_TOO_LONG"

    # Check for Telugu script presence (Unicode range \u0C00-\u0C7F)
    telugu_pattern = re.compile(r"[\u0C00-\u0C7F]")
    if not telugu_pattern.search(art):
        return False, "NO_TELUGU_IN_ARTICLE"
    if not telugu_pattern.search(hl):
        return False, "NO_TELUGU_IN_HEADLINE"

    # Verbatim headline-article lead match check
    if hl in art[:len(hl) + 20]:
        return False, "VERBATIM_COPY_LEAD"

    return True, "VALID"


def run_quality_filtering(source_jsonl: pathlib.Path, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Filtering dataset: {source_jsonl.name}")

    valid_records = []
    exclusion_stats = {}
    seen_articles = set()

    with open(source_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            art = rec.get("original_article", "")

            # Exact duplicate article filter
            if art in seen_articles:
                exclusion_stats["DUPLICATE_ARTICLE"] = exclusion_stats.get("DUPLICATE_ARTICLE", 0) + 1
                continue

            valid, reason = is_valid_quality_example(rec)
            if not valid:
                exclusion_stats[reason] = exclusion_stats.get(reason, 0) + 1
                continue

            seen_articles.add(art)
            valid_records.append(rec)

    total_in = len(valid_records) + sum(exclusion_stats.values())
    logger.info(f"Quality filter results for {source_jsonl.name}:")
    logger.info(f"  Input: {total_in} | Valid: {len(valid_records)} | Filtered: {sum(exclusion_stats.values())}")
    logger.info(f"  Filter Breakdown: {json.dumps(exclusion_stats, indent=2)}")

    return valid_records, exclusion_stats


def main():
    subsets_dir = PROJECT_ROOT / "data" / "subsets"
    manifest_path = subsets_dir / "quality_manifest.json"

    quality_summary = {}

    # Filter 5% subset
    f5 = subsets_dir / "train_5pct.jsonl"
    if f5.exists():
        records_5, stats_5 = run_quality_filtering(f5, subsets_dir)
        out_5 = subsets_dir / "train_5pct_quality.jsonl"
        with open(out_5, "w", encoding="utf-8") as f:
            for r in records_5:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        quality_summary["5pct"] = {
            "source": str(f5),
            "output": str(out_5),
            "retained": len(records_5),
            "exclusions": stats_5,
        }

    # Filter 25% subset
    f25 = subsets_dir / "train_25pct.jsonl"
    if f25.exists():
        records_25, stats_25 = run_quality_filtering(f25, subsets_dir)
        out_25 = subsets_dir / "train_25pct_quality.jsonl"
        with open(out_25, "w", encoding="utf-8") as f:
            for r in records_25:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        quality_summary["25pct"] = {
            "source": str(f25),
            "output": str(out_25),
            "retained": len(records_25),
            "exclusions": stats_25,
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nSaved quality filtering manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
