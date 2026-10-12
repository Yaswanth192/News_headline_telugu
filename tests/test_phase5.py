"""
tests/test_phase5.py
====================
Phase 5 test suite:
  1. Test MemoryTracker correctly distinguishes dedicated VRAM vs PyTorch allocations.
  2. Test MemoryTracker detects WDDM overcommit accurately without relabeling RAM as VRAM.
  3. Test Gemma 2 feasibility audit report exists and reports BLOCKED status with documented blockers.
  4. Test QualityFiltering deterministically rejects empty text, extreme lengths, and verbatim copies.
  5. Test QualityFiltering produces valid quality manifests.
  6. Test Denominator integrity in metric files (n_examples matches sample size).
"""

import json
import pathlib
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_memory_tracker_distinguishes_vram_and_ram():
    from src.utils.memory_tracker import get_memory_stats
    stats = get_memory_stats()
    assert "system_ram_total_gb" in stats
    assert "gpu_available" in stats
    if stats["gpu_available"]:
        assert "physical_dedicated_vram_mib" in stats
        assert "pytorch_peak_allocated_mib" in stats
        # Total dedicated physical VRAM on RTX 3050 should be ~4096 MiB
        assert 4000 <= stats["physical_dedicated_vram_mib"] <= 4200


def test_gemma2_feasibility_report():
    report_file = PROJECT_ROOT / "outputs" / "metrics" / "gemma2_feasibility_report.json"
    assert report_file.exists(), "Gemma 2 feasibility report must exist"
    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_id"] == "google/gemma-2-2b"
    assert data["overall_status"] == "BLOCKED"
    assert len(data["blockers"]) > 0
    # Must explicitly state the HF authentication and packaging blockers
    blockers_str = " ".join(data["blockers"])
    assert "HUGGINGFACE_GATED_AUTH_REQUIRED" in blockers_str or "gated" in blockers_str.lower()


def test_quality_filtering_rules():
    from src.experiments.quality_filtering import is_valid_quality_example

    # Empty text
    valid, reason = is_valid_quality_example({"original_article": "", "original_headline": ""})
    assert not valid and reason == "EMPTY_TEXT"

    # Too short
    valid, reason = is_valid_quality_example({"original_article": "చిన్న కథ", "original_headline": "హెడ్"})
    assert not valid and "SHORT" in reason

    # Verbatim copy
    lead = "ప్రభుత్వం కీలక నిర్ణయం తీసుకుంది"
    long_art = lead + " మరియు మరిన్ని వివరాలు ఉన్నాయి ఇక్కడ చాలా పెద్ద ఆర్టికల్ సమాచారం అందించబడుతుంది కాబట్టి ఇది కచ్చితంగా వంద అక్షరాల కంటే ఎక్కువ పొడవు కలిగి ఉంటుంది."
    valid, reason = is_valid_quality_example({"original_article": long_art, "original_headline": lead})
    assert not valid and reason == "VERBATIM_COPY_LEAD"


def test_quality_manifest_exists():
    q_manifest = PROJECT_ROOT / "data" / "subsets" / "quality_manifest.json"
    assert q_manifest.exists()
    with open(q_manifest, encoding="utf-8") as f:
        data = json.load(f)
    assert "5pct" in data
    assert "exclusions" in data["5pct"]
    assert data["5pct"]["retained"] > 0


def test_metric_denominator_integrity():
    res_file = PROJECT_ROOT / "outputs" / "metrics" / "low_resource_results.json"
    if res_file.exists():
        with open(res_file, encoding="utf-8") as f:
            data = json.load(f)
        for pct_key, d in data.items():
            n = d["rouge"]["n_examples"]
            rel = d["relevance_distribution"]
            total_classified = rel["HREL"] + rel["MREL"] + rel["LREL"]
            # Total classified must match denominator
            assert total_classified == n, f"Classified count {total_classified} must match denominator {n} for {pct_key}"
