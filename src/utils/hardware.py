"""
src/utils/hardware.py
=====================
Hardware detection utilities.

Used by training scripts to auto-configure batch size, gradient accumulation,
mixed precision, and gradient checkpointing based on detected VRAM.
"""

import logging
import json
import pathlib
from typing import Dict, Any

logger = logging.getLogger(__name__)

# VRAM thresholds (MiB) → training configuration recommendations
VRAM_CONFIGS = {
    # < 4 GB: tiny models only
    3072: {
        "description": "< 4 GB VRAM (very constrained)",
        "recommended_models": ["mT5-small"],
        "mt5_small": {"batch_size": 2, "grad_accum": 16, "fp16": True, "grad_checkpoint": True},
        "gemma2_qlora": None,  # BLOCKED
    },
    # 4 GB (RTX 3050 laptop)
    4096: {
        "description": "4 GB VRAM (RTX 3050)",
        "recommended_models": ["mT5-small", "IndicBART"],
        "mt5_small": {"batch_size": 4, "grad_accum": 8, "fp16": True, "grad_checkpoint": False},
        "indicbart": {"batch_size": 4, "grad_accum": 8, "fp16": True, "grad_checkpoint": False},
        "gemma2_qlora": {"batch_size": 1, "grad_accum": 32, "fp16": True, "grad_checkpoint": True,
                         "note": "MARGINAL — smoke test required"},
    },
    # 6 GB
    6144: {
        "description": "6 GB VRAM",
        "recommended_models": ["mT5-small", "IndicBART", "Gemma2-QLoRA"],
        "mt5_small": {"batch_size": 8, "grad_accum": 4, "fp16": True, "grad_checkpoint": False},
        "indicbart": {"batch_size": 8, "grad_accum": 4, "fp16": True, "grad_checkpoint": False},
        "gemma2_qlora": {"batch_size": 2, "grad_accum": 16, "fp16": True, "grad_checkpoint": True},
    },
    # 8 GB
    8192: {
        "description": "8 GB VRAM",
        "recommended_models": ["mT5-small", "IndicBART", "Gemma2-QLoRA"],
        "mt5_small": {"batch_size": 16, "grad_accum": 2, "fp16": True, "grad_checkpoint": False},
        "indicbart": {"batch_size": 16, "grad_accum": 2, "fp16": True, "grad_checkpoint": False},
        "gemma2_qlora": {"batch_size": 4, "grad_accum": 8, "fp16": True, "grad_checkpoint": False},
    },
}


def get_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU information using PyTorch.

    Returns a dict with:
        cuda_available (bool)
        device_name (str)
        vram_mib (int)
        cuda_version (str)
        compute_capability (str)
        fp16_supported (bool)
    """
    info: Dict[str, Any] = {"cuda_available": False}

    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Cannot detect GPU.")
        return info

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Will use CPU (training will be very slow).")
        return info

    props = torch.cuda.get_device_properties(0)
    info = {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "vram_mib": props.total_memory // (1024 ** 2),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{props.major}.{props.minor}",
        "fp16_supported": props.major >= 7,
        "num_devices": torch.cuda.device_count(),
    }
    return info


def get_training_config(model_key: str, vram_mib: int) -> Dict[str, Any]:
    """
    Return recommended training hyper-parameters for a given model and VRAM.

    Args:
        model_key:  one of 'mt5_small', 'indicbart', 'gemma2_qlora'
        vram_mib:   total VRAM in MiB

    Returns:
        dict with batch_size, grad_accum, fp16, grad_checkpoint, and optional note
    """
    # Find the matching tier (largest tier that doesn't exceed vram_mib)
    tier = None
    for threshold in sorted(VRAM_CONFIGS.keys()):
        if vram_mib >= threshold:
            tier = VRAM_CONFIGS[threshold]

    if tier is None:
        logger.error(f"VRAM {vram_mib} MiB is below all known tiers.")
        return {}

    config = tier.get(model_key)
    if config is None:
        logger.warning(f"Model '{model_key}' is BLOCKED for {vram_mib} MiB VRAM.")
        return {"blocked": True, "reason": f"Insufficient VRAM ({vram_mib} MiB) for {model_key}"}

    return config


def print_hardware_summary():
    """Print a human-readable hardware summary to stdout."""
    info = get_gpu_info()
    print("=" * 60)
    print("HARDWARE SUMMARY")
    print("=" * 60)
    if info["cuda_available"]:
        print(f"  GPU:               {info['device_name']}")
        print(f"  VRAM:              {info['vram_mib']} MiB")
        print(f"  CUDA version:      {info['cuda_version']}")
        print(f"  Compute cap.:      {info['compute_capability']}")
        print(f"  FP16 Tensor Cores: {'Yes' if info['fp16_supported'] else 'No'}")
        print(f"  Devices:           {info['num_devices']}")
        print()
        print("  Recommended configs:")
        for model_key in ["mt5_small", "indicbart", "gemma2_qlora"]:
            cfg = get_training_config(model_key, info["vram_mib"])
            if cfg.get("blocked"):
                print(f"    {model_key:20s}: BLOCKED — {cfg.get('reason', '')}")
            elif cfg:
                note = f"  [{cfg.get('note', '')}]" if cfg.get("note") else ""
                print(f"    {model_key:20s}: batch={cfg['batch_size']}, "
                      f"accum={cfg['grad_accum']}, fp16={cfg['fp16']}{note}")
    else:
        print("  GPU: NOT DETECTED (CPU only)")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_summary()
