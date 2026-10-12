"""
src/utils/memory_tracker.py
===========================
Comprehensive memory tracker distinguishing:
  1. PyTorch Dedicated GPU Memory Allocated (`torch.cuda.memory_allocated()`)
  2. PyTorch Dedicated GPU Memory Reserved (`torch.cuda.memory_reserved()`)
  3. Hardware Dedicated VRAM Limit (`torch.cuda.get_device_properties(0).total_memory`)
  4. System RAM Used and Available (`psutil.virtual_memory()`)
  5. Windows WDDM Shared GPU Memory Overcommit Indicator

Never conflates PyTorch virtual GPU allocation with physical dedicated VRAM.
"""

import torch
import psutil


def get_memory_stats(device: torch.device = None) -> dict:
    stats = {
        "system_ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "system_ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "system_ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "gpu_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        dev_idx = device.index if device and device.index is not None else 0
        props = torch.cuda.get_device_properties(dev_idx)
        total_vram_mib = round(props.total_memory / (1024 ** 2), 1)
        allocated_mib = round(torch.cuda.memory_allocated(dev_idx) / (1024 ** 2), 1)
        reserved_mib = round(torch.cuda.memory_reserved(dev_idx) / (1024 ** 2), 1)
        max_allocated_mib = round(torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2), 1)
        max_reserved_mib = round(torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 2), 1)

        is_overcommitted = max_allocated_mib > total_vram_mib

        stats.update({
            "gpu_name": props.name,
            "physical_dedicated_vram_mib": total_vram_mib,
            "pytorch_allocated_mib": allocated_mib,
            "pytorch_reserved_mib": reserved_mib,
            "pytorch_peak_allocated_mib": max_allocated_mib,
            "pytorch_peak_reserved_mib": max_reserved_mib,
            "wddm_shared_memory_overcommit": is_overcommitted,
            "overcommit_spill_mib": max(0.0, round(max_allocated_mib - total_vram_mib, 1)),
        })

    return stats


def format_memory_summary(stats: dict) -> str:
    if not stats.get("gpu_available"):
        return f"CPU/RAM: {stats['system_ram_used_gb']} GB used / {stats['system_ram_total_gb']} GB total"

    overcommit_str = (
        f" [WDDM Overcommit: {stats['overcommit_spill_mib']} MiB spilled to Shared RAM]"
        if stats["wddm_shared_memory_overcommit"]
        else " [Within Dedicated VRAM]"
    )

    return (
        f"Dedicated VRAM: {stats['physical_dedicated_vram_mib']} MiB | "
        f"PyTorch Peak Alloc: {stats['pytorch_peak_allocated_mib']} MiB{overcommit_str} | "
        f"System RAM: {stats['system_ram_used_gb']}/{stats['system_ram_total_gb']} GB"
    )
