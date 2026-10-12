"""
src/experiments/gemma2_feasibility.py
====================================
Phase 5 — Multi-Stage Gemma 2 2B Feasibility Investigation on RTX 3050 (4 GB VRAM).

Executes the staged feasibility protocol:
  Stage 1: Environment & Dependency Audit (OS, Python, Torch, Transformers, BitsAndBytes, PEFT, Accelerate)
  Stage 2: Model Gating & Access Verification (HuggingFace authentication check)
  Stage 3: Architectural Memory Estimation (Model weights FP16/4-bit, KV cache, activation headroom)
  Stage 4: Quantization Engine Verification (bitsandbytes availability on Windows)
  Stage 5: Verdict & Blocker Classification
"""

import json
import logging
import pathlib
import sys
import datetime
import torch
import psutil

# Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from src.utils.memory_tracker import get_memory_stats


def investigate_feasibility():
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_id": "google/gemma-2-2b",
        "stages": {},
        "blockers": [],
        "overall_status": "UNKNOWN",
    }

    # ── Stage 1: Environment & Hardware Audit ────────────────────────────────
    logger.info("Stage 1: Auditing local execution environment...")
    mem_stats = get_memory_stats()
    stage1 = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": mem_stats.get("gpu_name", "N/A"),
        "dedicated_vram_mib": mem_stats.get("physical_dedicated_vram_mib", 0),
        "system_ram_total_gb": mem_stats.get("system_ram_total_gb", 0),
        "system_ram_available_gb": mem_stats.get("system_ram_available_gb", 0),
    }

    # Check package dependencies
    pkg_status = {}
    for pkg in ["transformers", "accelerate", "peft", "bitsandbytes"]:
        try:
            mod = __import__(pkg)
            pkg_status[pkg] = getattr(mod, "__version__", "INSTALLED")
        except ImportError:
            pkg_status[pkg] = "NOT_INSTALLED"
        except Exception as e:
            pkg_status[pkg] = f"ERROR: {e}"

    stage1["packages"] = pkg_status
    report["stages"]["stage1_environment"] = stage1

    logger.info(f"  GPU Dedicated VRAM: {stage1['dedicated_vram_mib']} MiB")
    logger.info(f"  System RAM: {stage1['system_ram_available_gb']} GB available")
    logger.info(f"  Package audit: {json.dumps(pkg_status)}")

    # ── Stage 2: HuggingFace Gated Access Verification ────────────────────────
    logger.info("\nStage 2: Verifying HuggingFace repository access for google/gemma-2-2b...")
    from huggingface_hub import HfApi
    api = HfApi()
    gated_status = "UNKNOWN"
    access_granted = False
    try:
        info = api.model_info("google/gemma-2-2b")
        gated_status = getattr(info, "gated", "Unknown")
        # Attempt to access a config file to verify user token authentication
        from transformers import AutoTokenizer
        try:
            AutoTokenizer.from_pretrained("google/gemma-2-2b")
            access_granted = True
            logger.info("  HF Token authentication: VALID")
        except Exception as e:
            err_msg = str(e)
            if "gated repo" in err_msg.lower() or "401" in err_msg:
                access_granted = False
                logger.warning(f"  HF Access BLOCKED: Gated model access requires HF login token.")
                report["blockers"].append("HUGGINGFACE_GATED_AUTH_REQUIRED: 'google/gemma-2-2b' requires accepting terms on HuggingFace and running 'huggingface-cli login'.")
            else:
                access_granted = False
                report["blockers"].append(f"TOKENIZER_ACCESS_ERROR: {err_msg}")
    except Exception as e:
        report["blockers"].append(f"HF_API_ERROR: {e}")

    report["stages"]["stage2_access"] = {
        "is_gated": gated_status,
        "authenticated_access_granted": access_granted,
    }

    # ── Stage 3: Mathematical Memory Budget Estimation ──────────────────────
    logger.info("\nStage 3: Computing Gemma 2 2B memory footprint requirements...")
    # Gemma 2 2B parameters: ~2.6 Billion parameters
    # Parameter weights in FP16/BF16: 2.6B * 2 bytes = ~5.2 GB (Exceeds 4 GB VRAM)
    # Parameter weights in INT4: 2.6B * 0.5 bytes = ~1.3 GB
    # Optimizer states (AdamW FP32): 2.6B * 8 bytes = ~20.8 GB (Full fine-tune impossible)
    # LoRA trainable params (~1%): ~26M params -> LoRA weights + AdamW states = ~250 MB
    # Activations & KV cache (batch=1, seq=512): ~1.2 - 1.8 GB
    # Base CUDA context overhead: ~400 - 600 MB
    # Total INT4 QLoRA minimum footprint: ~1.3 GB (weights) + ~1.5 GB (activations) + ~0.5 GB (CUDA context) = ~3.3 - 3.8 GB
    mem_model = {
        "parameter_count": "2.61 Billion",
        "full_fp16_weights_gb": 5.22,
        "full_adamw_states_gb": 20.88,
        "int4_weights_gb": 1.31,
        "estimated_qlora_training_footprint_gb": 3.6,
        "dedicated_vram_limit_gb": round(stage1["dedicated_vram_mib"] / 1024, 2),
        "full_finetune_feasible": False,
        "theoretical_qlora_feasible_on_paper": True,
    }
    report["stages"]["stage3_memory_budget"] = mem_model
    logger.info(f"  Full FP16 weights: {mem_model['full_fp16_weights_gb']} GB (Exceeds dedicated VRAM)")
    logger.info(f"  INT4 QLoRA theoretical minimum: ~{mem_model['estimated_qlora_training_footprint_gb']} GB (Marginal on 4 GB)")

    # ── Stage 4: Quantization Engine Verification ────────────────────────────
    logger.info("\nStage 4: Auditing 4-bit quantization software stack...")
    bnb_available = pkg_status.get("bitsandbytes") != "NOT_INSTALLED"
    peft_available = pkg_status.get("peft") != "NOT_INSTALLED"

    stage4 = {
        "bitsandbytes_installed": bnb_available,
        "peft_installed": peft_available,
        "windows_cuda_bnb_supported": False,  # Standard bitsandbytes requires compiled Windows CUDA binaries
    }

    if not bnb_available:
        report["blockers"].append("BITSANDBYTES_MISSING: 'bitsandbytes' is not installed in the virtual environment. 4-bit NF4 quantization requires bitsandbytes.")
    if not peft_available:
        report["blockers"].append("PEFT_MISSING: 'peft' package is not installed in the virtual environment for parameter-efficient LoRA adapters.")

    report["stages"]["stage4_quantization_stack"] = stage4

    # ── Stage 5: Verdict & Classification ────────────────────────────────────
    logger.info("\nStage 5: Determining overall feasibility verdict...")
    if len(report["blockers"]) > 0:
        report["overall_status"] = "BLOCKED"
    else:
        report["overall_status"] = "FEASIBLE"

    logger.info(f"Overall Status: {report['overall_status']}")
    for b in report["blockers"]:
        logger.warning(f"  [BLOCKER] {b}")

    out_file = PROJECT_ROOT / "outputs" / "metrics" / "gemma2_feasibility_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\nFeasibility audit saved to: {out_file}")

    return report


if __name__ == "__main__":
    investigate_feasibility()
