"""
tests/test_environment.py
=========================
Phase 0 — Environment sanity checks (pytest).

Run:
    python -m pytest tests/test_environment.py -v
"""

import sys
import importlib
import pathlib
import subprocess
import json
import pytest


PROJECT_ROOT = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Directory structure tests
# ---------------------------------------------------------------------------

def test_project_root_exists():
    """Project root directory must exist."""
    assert PROJECT_ROOT.exists(), f"Project root not found: {PROJECT_ROOT}"


@pytest.mark.parametrize("subdir", [
    "src",
    "data/raw",
    "data/processed",
    "data/splits",
    "logs",
    "checkpoints",
    "outputs/predictions",
    "outputs/metrics",
    "outputs/plots",
    "tests",
    "docs",
])
def test_required_directory_exists(subdir):
    """All required project directories must exist."""
    d = PROJECT_ROOT / subdir
    assert d.exists(), f"Missing directory: {d}"


@pytest.mark.parametrize("fname", [
    ".gitignore",
    "README.md",
    "PROJECT_STATUS.md",
    "requirements.txt",
    "docs/PROJECT_PLAN.md",
    "tests/test_gpu.py",
    "tests/test_environment.py",
])
def test_required_file_exists(fname):
    """All Phase 0 required files must exist."""
    f = PROJECT_ROOT / fname
    assert f.exists(), f"Missing file: {f}"


# ---------------------------------------------------------------------------
# Python version
# ---------------------------------------------------------------------------

def test_python_version():
    """Python version must be 3.9 or higher."""
    assert sys.version_info >= (3, 9), (
        f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}"
    )


# ---------------------------------------------------------------------------
# Core non-ML package imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", [
    "json",
    "pathlib",
    "logging",
    "datetime",
    "argparse",
    "numpy",
    "pandas",
])
def test_core_package_importable(package):
    """Core packages that must be importable."""
    mod = importlib.import_module(package)
    assert mod is not None, f"Cannot import {package}"


# ---------------------------------------------------------------------------
# PyTorch (soft check — may not be installed yet)
# ---------------------------------------------------------------------------

def test_torch_importable():
    """PyTorch must be importable. (Install: pip install torch --index-url https://download.pytorch.org/whl/cu121)"""
    try:
        import torch
        assert torch.__version__, "torch.__version__ is empty"
    except ImportError:
        pytest.fail(
            "PyTorch is not installed.\n"
            "Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )


def test_cuda_available():
    """CUDA must be available for GPU training."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    if not torch.cuda.is_available():
        pytest.fail(
            "CUDA is not available.\n"
            "Ensure you installed the CUDA build of PyTorch and your GPU driver is current."
        )


# ---------------------------------------------------------------------------
# GPU report check
# ---------------------------------------------------------------------------

def test_gpu_report_exists():
    """GPU test report JSON must exist (run python tests/test_gpu.py first)."""
    report = PROJECT_ROOT / "logs" / "phase0_gpu_test.json"
    if not report.exists():
        pytest.skip(
            "GPU test report not found. Run: python tests/test_gpu.py"
        )
    with open(report) as f:
        data = json.load(f)
    assert data.get("all_passed") is True, (
        f"GPU test did not pass. See {report} for details."
    )


# ---------------------------------------------------------------------------
# Git availability
# ---------------------------------------------------------------------------

def test_git_available():
    """Git must be available on PATH."""
    result = subprocess.run(
        ["git", "--version"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, "git command not found on PATH"
    assert "git version" in result.stdout.lower()
