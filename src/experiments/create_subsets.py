"""
src/experiments/create_subsets.py
=================================
Phase 4 — Deterministic, Nested Subset Creation for Low-Resource Telugu Headline Generation.

Creates subsets for:
  - 5%   (4,178 examples)
  - 10%  (8,356 examples)
  - 25%  (20,890 examples)
  - 50%  (41,781 examples)
  - 100% (83,562 examples)

Guarantees:
  1. Nested hierarchy: 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%.
  2. Fixed seed (default 42) ensures deterministic reproducibility.
  3. Validation and test sets remain untouched and fixed.
  4. Metadata manifest saved with exact hashes and sample counts.
"""

import argparse
import datetime
import hashlib
import json
import logging
import pathlib
import random
import sys

# Ensure UTF-8 on Windows
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

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]


def sha256_file(filepath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_subsets(train_path: pathlib.Path, output_dir: pathlib.Path, seed: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading full train dataset from: {train_path}")
    with open(train_path, encoding="utf-8") as f:
        all_records = [line.strip() for line in f if line.strip()]

    total = len(all_records)
    logger.info(f"Total training examples: {total}")

    # Deterministic permutation for nesting:
    # Any prefix [:k] is guaranteed to be contained in any larger prefix [:m] where k <= m.
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "source_file": str(train_path),
        "source_count": total,
        "subsets": {},
    }

    for frac in FRACTIONS:
        pct = int(round(frac * 100))
        n_samples = int(round(total * frac)) if frac < 1.0 else total
        subset_indices = indices[:n_samples]

        out_file = output_dir / f"train_{pct}pct.jsonl"
        logger.info(f"Writing {pct}% subset ({n_samples} samples) -> {out_file.name}")

        with open(out_file, "w", encoding="utf-8") as f:
            for idx in subset_indices:
                f.write(all_records[idx] + "\n")

        # Record story_ids sample for quick verification
        first_few_story_ids = [
            json.loads(all_records[idx])["story_id"] for idx in subset_indices[:5]
        ]

        manifest["subsets"][f"{pct}pct"] = {
            "fraction": frac,
            "sample_count": n_samples,
            "file_path": str(out_file),
            "sha256": sha256_file(out_file),
            "first_story_ids": first_few_story_ids,
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Subset creation complete. Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate nested subsets for low-resource experiments")
    parser.add_argument("--train-jsonl", default=str(PROJECT_ROOT / "data" / "processed" / "train.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "subsets"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_subsets(pathlib.Path(args.train_jsonl), pathlib.Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
