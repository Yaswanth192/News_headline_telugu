"""
src/evaluation/assess_indicbart.py
===================================
Phase 3 — IndicBART feasibility assessment on RTX 3050 (4 GB VRAM).

Checks:
  1. Can ai4bharat/IndicBART be downloaded from HuggingFace?
  2. How much VRAM does loading the model consume?
  3. Can a single forward pass (inference) complete without OOM?
  4. Can model.generate() produce output on a Telugu sentence?

Run:
    python src/evaluation/assess_indicbart.py

Output:
    outputs/metrics/indicbart_feasibility.json
"""

import json
import logging
import pathlib
import sys
import datetime

# Fix Windows cp1252 for Telugu text
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

INDICBART_MODEL_ID = "ai4bharat/IndicBART"
CACHE_DIR = str(PROJECT_ROOT / ".cache" / "huggingface")
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "metrics" / "indicbart_feasibility.json"


def assess():
    import torch

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_id": INDICBART_MODEL_ID,
        "status": "UNKNOWN",
        "details": {},
        "errors": [],
    }

    # ── Step 1: GPU check ────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        msg = "CUDA not available — cannot assess VRAM feasibility."
        logger.error(msg)
        result["status"] = "BLOCKED"
        result["errors"].append(msg)
        return result

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    logger.info(f"GPU: {gpu_name} | Total VRAM: {total_vram} MiB")
    result["details"]["gpu"] = gpu_name
    result["details"]["total_vram_mib"] = total_vram

    # ── Step 2: HuggingFace access check ─────────────────────────────────────
    logger.info(f"Checking HuggingFace access for: {INDICBART_MODEL_ID}")
    try:
        from huggingface_hub import model_info
        info = model_info(INDICBART_MODEL_ID)
        result["details"]["hf_model_id"] = info.id
        result["details"]["hf_access"] = "OK"
        logger.info(f"  Model found on HuggingFace: {info.id}")
    except Exception as e:
        err = str(e)
        logger.error(f"  HuggingFace access FAILED: {err}")
        result["status"] = "BLOCKED"
        result["details"]["hf_access"] = "FAILED"
        result["errors"].append(f"HF access error: {err}")
        return result

    # ── Step 3: Tokenizer download ────────────────────────────────────────────
    logger.info("Loading IndicBART tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            INDICBART_MODEL_ID, cache_dir=CACHE_DIR
        )
        result["details"]["tokenizer"] = "OK"
        logger.info(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        result["details"]["vocab_size"] = tokenizer.vocab_size
    except Exception as e:
        err = str(e)
        logger.error(f"  Tokenizer load FAILED: {err}")
        result["status"] = "BLOCKED"
        result["errors"].append(f"Tokenizer error: {err}")
        return result

    # ── Step 4: Model load + VRAM measurement ────────────────────────────────
    logger.info("Loading IndicBART model (this may take 1-2 minutes on first run)...")
    torch.cuda.reset_peak_memory_stats()
    try:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICBART_MODEL_ID, cache_dir=CACHE_DIR
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model loaded. Parameters: {n_params / 1e6:.1f}M")
        result["details"]["n_params_millions"] = round(n_params / 1e6, 1)

        model = model.to(device)
        vram_after_load = torch.cuda.memory_allocated() // (1024 ** 2)
        logger.info(f"  VRAM after model load: {vram_after_load} MiB / {total_vram} MiB")
        result["details"]["vram_after_load_mib"] = vram_after_load
        result["details"]["vram_headroom_mib"] = total_vram - vram_after_load

    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM during model load: {e}"
        logger.error(f"  {err}")
        result["status"] = "BLOCKED_OOM"
        result["errors"].append(err)
        return result
    except Exception as e:
        err = str(e)
        logger.error(f"  Model load FAILED: {err}")
        result["status"] = "BLOCKED"
        result["errors"].append(f"Model load error: {err}")
        return result

    # ── Step 5: Inference smoke test ─────────────────────────────────────────
    logger.info("Running inference smoke test on a Telugu sentence...")
    test_article = (
        "తెలంగాణ రాష్ట్రంలో నేడు భారీ వర్షాలు కురిసే అవకాశం ఉందని వాతావరణ శాఖ హెచ్చరించింది. "
        "హైదరాబాద్ నగరం మరియు పరిసర ప్రాంతాల్లో ఆరెంజ్ అలర్ట్ జారీ చేశారు."
    )
    try:
        model.eval()
        # IndicBART uses language tags: <2te> for Telugu
        # Check what language tags the tokenizer supports
        if hasattr(tokenizer, "lang_code_to_id"):
            lang_codes = list(tokenizer.lang_code_to_id.keys())
            result["details"]["supported_lang_codes"] = lang_codes[:20]
            logger.info(f"  Supported lang codes: {lang_codes[:10]}...")
            # Find Telugu tag
            te_tag = None
            for code in lang_codes:
                if "te" in code.lower() and "telugu" in code.lower() or code == "<2te>":
                    te_tag = code
                    break
            if te_tag is None:
                for code in lang_codes:
                    if "te" in code:
                        te_tag = code
                        break
            result["details"]["telugu_lang_tag"] = te_tag
            logger.info(f"  Telugu language tag: {te_tag}")

        # Tokenize with truncation
        inputs = tokenizer(
            test_article,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to(device)

        # IndicBART is BART-based and does not accept token_type_ids
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            # Attempt generation
            gen_kwargs = {
                "max_new_tokens": 32,
                "num_beams": 2,
                "early_stopping": True,
            }
            # If forced_bos_token_id is available for Telugu
            if hasattr(tokenizer, "lang_code_to_id") and "<2te>" in tokenizer.lang_code_to_id:
                gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["<2te>"]
            outputs = model.generate(**inputs, **gen_kwargs)

        vram_peak_inference = torch.cuda.max_memory_allocated() // (1024 ** 2)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  Generated: {generated}")
        logger.info(f"  Peak VRAM during inference: {vram_peak_inference} MiB")

        result["details"]["inference_smoke"] = "PASSED"
        result["details"]["sample_generated"] = generated
        result["details"]["vram_peak_inference_mib"] = vram_peak_inference

    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM during inference: {e}"
        logger.error(f"  {err}")
        result["details"]["inference_smoke"] = "FAILED_OOM"
        result["errors"].append(err)
    except Exception as e:
        err = str(e)
        logger.error(f"  Inference FAILED: {err}")
        result["details"]["inference_smoke"] = f"FAILED: {err}"
        result["errors"].append(f"Inference error: {err}")

    # ── Step 6: Training feasibility estimate ─────────────────────────────────
    vram_load = result["details"].get("vram_after_load_mib", total_vram)
    headroom = total_vram - vram_load
    logger.info(f"\n=== IndicBART Feasibility Summary ===")
    logger.info(f"  Total VRAM:        {total_vram} MiB")
    logger.info(f"  VRAM after load:   {vram_load} MiB")
    logger.info(f"  Headroom:          {headroom} MiB")

    if headroom < 200:
        feasibility = "INFEASIBLE — insufficient VRAM headroom for any batch training"
    elif headroom < 600:
        feasibility = "MARGINAL — inference only; training requires batch=1 + gradient_checkpointing"
    else:
        feasibility = "FEASIBLE — inference and minimal training (batch=1, accum=16+) viable"

    result["details"]["training_feasibility"] = feasibility
    result["details"]["recommendation"] = feasibility
    logger.info(f"  Feasibility:       {feasibility}")

    if not result["errors"]:
        result["status"] = "FEASIBLE" if headroom >= 200 else "BLOCKED_OOM"
    else:
        result["status"] = "BLOCKED"

    return result


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = assess()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"\nFeasibility report saved to: {OUTPUT_PATH}")

    if result["status"] in ("BLOCKED", "BLOCKED_OOM"):
        logger.error(f"\n[BLOCKED] IndicBART is NOT feasible on this GPU.")
        for err in result["errors"]:
            logger.error(f"  - {err}")
        sys.exit(1)
    else:
        logger.info(f"\n[OK] IndicBART status: {result['status']}")


if __name__ == "__main__":
    main()
