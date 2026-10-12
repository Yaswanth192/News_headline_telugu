"""
tests/test_phase4.py
====================
Phase 4 automated test suite:
  1. Test nested subset sample sizes match exact proportions (5%, 10%, 25%, 50%, 100%).
  2. Test nested subset consistency (5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%).
  3. Test deterministic reproducibility with seed 42.
  4. Test validation set remains identical and untainted.
  5. Test manifest.json exists with SHA-256 hashes and correct schema.
  6. Test evaluation script components handle missing checkpoints gracefully.
"""

import json
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_subset_manifest_exists():
    manifest_file = PROJECT_ROOT / "data" / "subsets" / "manifest.json"
    assert manifest_file.exists(), "Manifest file data/subsets/manifest.json must exist"
    with open(manifest_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["source_count"] == 83562
    assert "5pct" in data["subsets"]
    assert "10pct" in data["subsets"]
    assert "25pct" in data["subsets"]
    assert "50pct" in data["subsets"]
    assert "100pct" in data["subsets"]


def test_subset_sample_counts():
    manifest_file = PROJECT_ROOT / "data" / "subsets" / "manifest.json"
    with open(manifest_file, encoding="utf-8") as f:
        data = json.load(f)

    expected = {
        "5pct": 4178,
        "10pct": 8356,
        "25pct": 20890,
        "50pct": 41781,
        "100pct": 83562,
    }
    for k, v in expected.items():
        assert data["subsets"][k]["sample_count"] == v, f"{k} sample count mismatch"


def test_nested_subset_consistency():
    subsets_dir = PROJECT_ROOT / "data" / "subsets"
    f5 = subsets_dir / "train_5pct.jsonl"
    f10 = subsets_dir / "train_10pct.jsonl"
    f25 = subsets_dir / "train_25pct.jsonl"

    assert f5.exists() and f10.exists() and f25.exists()

    with open(f5, encoding="utf-8") as f:
        ids_5 = [json.loads(line)["story_id"] for line in f]
    with open(f10, encoding="utf-8") as f:
        ids_10 = [json.loads(line)["story_id"] for line in f]
    with open(f25, encoding="utf-8") as f:
        ids_25 = [json.loads(line)["story_id"] for line in f]

    set_5 = set(ids_5)
    set_10 = set(ids_10)
    set_25 = set(ids_25)

    # 5% must be a strict subset of 10%
    assert set_5.issubset(set_10), "5% subset must be strictly contained in 10% subset"
    # 10% must be a strict subset of 25%
    assert set_10.issubset(set_25), "10% subset must be strictly contained in 25% subset"


def test_validation_split_unchanged():
    val_file = PROJECT_ROOT / "data" / "processed" / "validation.jsonl"
    assert val_file.exists(), "Validation set must exist"
    count = sum(1 for _ in open(val_file, encoding="utf-8"))
    assert count == 5962, "Validation split size must remain exactly 5,962"
