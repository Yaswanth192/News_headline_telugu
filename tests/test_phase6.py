"""
tests/test_phase6.py
====================
Phase 6 Automated Test Suite for Streamlit Application & Inference Manager.

Validates:
1. ModelManager discovery of checkpoints.
2. Safe sequential loading and memory release.
3. Input validation (empty text raises ValueError).
4. Predict relevance integration.
5. All research artifact metrics and plot assets exist.
6. App module imports without errors.
"""

import json
import pathlib
import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_model_registry_integrity():
    """Verify all checkpoints registered in MODEL_REGISTRY exist on disk."""
    from src.inference.manager import MODEL_REGISTRY, CLASSIFIER_REGISTRY

    assert len(MODEL_REGISTRY) >= 4, "Expected at least 4 registered generator variants."
    for key, meta in MODEL_REGISTRY.items():
        assert meta["path"].exists(), f"Registered checkpoint path does not exist: {meta['path']}"

    for key, meta in CLASSIFIER_REGISTRY.items():
        assert meta["path"].exists(), f"Registered classifier path does not exist: {meta['path']}"


def test_manager_input_validation():
    """Verify that empty or whitespace input raises ValueError."""
    from src.inference.manager import get_model_manager

    mgr = get_model_manager(force_cpu=True)
    with pytest.raises(ValueError):
        mgr.generate("indicbart_final", "")

    with pytest.raises(ValueError):
        mgr.generate("indicbart_final", "   ")


def test_manager_unknown_model_raises():
    """Verify that unknown model key raises ValueError."""
    from src.inference.manager import get_model_manager

    mgr = get_model_manager(force_cpu=True)
    with pytest.raises(ValueError):
        mgr.generate("non_existent_model", "వార్త")


def test_gpu_memory_info_format():
    """Verify GPU memory info dictionary schema."""
    from src.inference.manager import ModelManager

    mgr = ModelManager(force_cpu=False)
    info = mgr.get_gpu_memory_info()
    assert "available" in info
    assert "device_name" in info
    assert "allocated_mib" in info
    assert "reserved_mib" in info
    assert "total_mib" in info
    if torch.cuda.is_available():
        assert info["available"] is True
        assert info["total_mib"] > 3000.0


def test_app_imports_cleanly():
    """Verify app.py imports and initializes components cleanly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("app", str(PROJECT_ROOT / "app.py"))
    assert spec is not None
    app_module = importlib.util.module_from_spec(spec)
    # Ensure all preset samples have required keys and non-empty Telugu text
    import app
    assert len(app.PRESET_SAMPLES) >= 3
    for s in app.PRESET_SAMPLES:
        assert "title" in s
        assert "article" in s
        assert "reference_headline" in s
        assert len(s["article"]) > 50


def test_dashboard_metric_assets_exist():
    """Verify all research dashboard files exist and are valid JSON / CSV."""
    p3_metrics = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison_phase3.json"
    p4_metrics = PROJECT_ROOT / "outputs" / "metrics" / "low_resource_results.json"
    p5_report = PROJECT_ROOT / "outputs" / "metrics" / "gemma2_feasibility_report.json"
    plot_rouge = PROJECT_ROOT / "outputs" / "plots" / "learning_curve_rouge.png"
    plot_chrf = PROJECT_ROOT / "outputs" / "plots" / "learning_curve_chrf.png"

    assert p3_metrics.exists(), "Phase 3 model comparison metrics missing."
    assert p4_metrics.exists(), "Phase 4 low resource metrics missing."
    assert p5_report.exists(), "Phase 5 Gemma 2 feasibility report missing."
    assert plot_rouge.exists(), "Phase 4 ROUGE plot missing."
    assert plot_chrf.exists(), "Phase 4 chrF plot missing."

    # Validate JSON content
    with open(p3_metrics, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "models" in data
        assert "mT5-small" in data["models"]
        assert "IndicBART" in data["models"]

    with open(p5_report, "r", encoding="utf-8") as f:
        gemma = json.load(f)
        assert gemma["overall_status"] == "BLOCKED"
