# PROJECT STATUS — Telugu News Headline Generator

> Last updated: 2026-10-12 18:07 IST

---

## Hardware Audit (Phase 0 Finding)

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | **4096 MiB (4 GB)** |
| Driver | 592.82 |
| CUDA (driver-reported) | 13.1 |
| GPU Util at audit | 0% (idle) |
| Python | 3.11.0 (`C:\Users\VICTUS\AppData\Local\Programs\Python\Python311\python.exe`) |
| Git | 2.47.0.windows.2 |
| Drive C: Free | 18.34 GB |
| Drive D: Free | 88.36 GB |

### VRAM Implications

| Model | Estimated VRAM | Verdict |
|-------|---------------|---------|
| mT5-small (300M params) | ~1–1.5 GB (FP16) | ✅ Feasible |
| IndicBART (~244M params) | ~0.6–1 GB (FP16) | ✅ Likely feasible |
| Gemma 2 2B (full fine-tune) | ~16 GB | ❌ Exceeds 4 GB |
| Gemma 2 2B (QLoRA 4-bit) | ~3–5 GB | ⚠️ Marginal — requires smoke test |

---

## Phase Checklist

| Phase | Name | Status | Updated |
|-------|------|--------|---------|
| 0 | Hardware Audit & Foundation | ✅ PASSED | 2026-10-12 |
| 1 | Dataset & Baseline | ✅ PASSED | 2026-10-12 |
| 2 | Relevance Classification | ✅ PASSED | 2026-10-12 |
| 3 | Model Comparison | ✅ PASSED | 2026-10-12 |
| 4 | Low-Resource Learning | ⬜ NOT STARTED | — |
| 5 | Gemma 2 Quantity vs Quality | ⬜ NOT STARTED | — |
| 6 | Streamlit Application | ⬜ NOT STARTED | — |
| 7 | Final Research Report | ⬜ NOT STARTED | — |

---

## Phase 2 — Relevance Classification

**Status:** ✅ PASSED

### Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Original paper & dataset audit | ✅ PASSED | Kanumolu et al. (LREC-COLING 2024), 26,178 pairs |
| Taxonomy & label verification | ✅ PASSED | HREL (FME), MREL (FSE/STC/WKC), LREL (SEN/CBT/MLC/USO) |
| Data preprocessing & split integrity | ✅ PASSED | Clean parquet/jsonl splits, 0 ID cross-leakage |
| Classical ML baseline | ✅ PASSED | TF-IDF + Logistic Regression (Macro-F1: 0.4775) |
| Local GPU transformer classifier | ✅ PASSED | mBERT fine-tuned on RTX 3050 (Macro-F1: 0.5114, VRAM: 3.4GB) |
| Checkpoints saved & reproducible | ✅ PASSED | `checkpoints/classifier_tfidf/`, `checkpoints/classifier_bert/` |
| Evaluation on validation split | ✅ PASSED | Saved to `outputs/metrics/*_classifier_eval.json` |
| Standalone prediction interface | ✅ PASSED | `src/classification/predict.py` (CLI + Python API) |
| Phase 1 pipeline integrity | ✅ PASSED | All Phase 1 tests untouched and passing |
| Full test suite green | ✅ PASSED | 61/61 passing (`pytest tests/ -v`) |

---

## Phase 1 — Dataset & Baseline

**Status:** ✅ PASSED

### Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Dataset download & schema validation | ✅ PASSED | `saidines12/telugu_news_dataset` (102,332 rows) |
| Missing values, duplicates & leakage audited | ✅ PASSED | `outputs/metrics/dataset_audit.json` |
| Text preprocessing & tokenization | ✅ PASSED | 101,965 examples in `data/processed/*.jsonl` |
| Local GPU training smoke test | ✅ PASSED | 20 steps, loss: 30.68 -> 12.61, RTX 3050 |
| Baseline 50-step training run | ✅ PASSED | 50 steps, loss: 12.34 -> 4.01, RTX 3050 |
| Checkpoint saved and reloaded | ✅ PASSED | `checkpoints/mt5_small/final` verified |
| Evaluation on validation set | ✅ PASSED | chrF = 11.80, avg 0.294s/ex, saved in `outputs/` |
| Test suite expanded & green | ✅ PASSED | 52/52 tests passing (`pytest tests/ -v`) |

---

## Phase 0 — Hardware Audit & Clean Foundation

**Status:** ✅ PASSED

### Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Actual hardware findings recorded | ✅ PASSED | nvidia-smi: RTX 3050 Laptop, 4095 MiB, driver 592.82 |
| GPU VRAM identified (not assumed) | ✅ PASSED | 4095 MiB confirmed — NOT 8 GB |
| GPU visibility tested (PyTorch CUDA) | ✅ PASSED | `test_gpu.py` → CUDA 12.1, Compute 8.6, tensor op = 12.0 |
| Python version checked | ✅ PASSED | Python 3.11.0 |
| Git version checked | ✅ PASSED | git 2.47.0.windows.2 |
| Disk space recorded | ✅ PASSED | D: 88 GB free (used for data/checkpoints) |
| Virtual environment created | ✅ PASSED | `venv/` at project root |
| PyTorch CUDA build installed | ✅ PASSED | torch-2.5.1+cu121 installed |
| Minimal GPU tensor test passes | ✅ PASSED | Result: 12.0, log: `logs/phase0_gpu_test.json` |
| Mixed precision (FP16) check | ✅ PASSED | Compute capability 8.6 (Ampere) — FP16 Tensor Cores confirmed |
| .gitignore created | ✅ PASSED | Covers venv, checkpoints, .cache, .env |
| README.md created | ✅ PASSED | With hardware table, structure, quick start |
| PROJECT_STATUS.md created | ✅ PASSED | This document |
| docs/PROJECT_PLAN.md created | ✅ PASSED | 8-phase plan with commands |
| Automated test suite passes | ✅ PASSED | 31/31 tests pass (`python -m pytest tests/ -v`) |
| Git initialized + initial commit | ✅ PASSED | commit e43948c — 14 files |
| No model training performed | ✅ PASSED | No training scripts invoked |
| No existing user files deleted | ✅ PASSED | Directory was empty at start |

### Files Created in Phase 0

- `.gitignore`
- `README.md`
- `PROJECT_STATUS.md` (this file)
- `docs/PROJECT_PLAN.md`
- `requirements.txt`
- `tests/test_gpu.py`
- `tests/test_environment.py`
- `src/__init__.py`
- `src/utils/hardware.py`

---

## Phase 1 — Dataset & Baseline

**Status:** ✅ PASSED (2026-10-12)

---

## Phase 2 — Relevance Classification

**Status:** ✅ PASSED (2026-10-12)

---

## Phase 3 — Controlled Generation Model Comparison

**Status:** ✅ PASSED (2026-10-12)

- Diagnosed & fixed zero ROUGE score bug by implementing `IndicTokenizer` for Telugu script.
- Eliminated `<extra_id_0>` sentinel token leakage via `decoder_start_token_id=0` and regex stripping.
- Verified `ai4bharat/IndicBART` feasibility on local RTX 3050 (4 GB VRAM) — 931 MiB load, 3,164 MiB headroom.
- Trained `IndicBART` (50 steps, bfloat16, grad accum 16) locally and saved checkpoints.
- Conducted controlled model comparison between `mT5-small` and `IndicBART` on identical validation samples across ROUGE-1/2/L, chrF, latency, VRAM, and relevance classification via Phase 2 `mBERT` classifier.
- Full test suite passes: 67/67 automated tests passing. Detailed in `docs/PHASE3_REPORT.md`.

---

## Blocked Items

| Item | Reason | Resolution |
|------|--------|------------|
| Gemma 2 2B QLoRA | 4 GB VRAM — marginal, needs smoke test | Will test after Phase 1 model comparison |
| TeClass dataset access | Not yet verified | Phase 2 investigation required |

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-10-12 | Use D: drive for data/checkpoints | 88 GB free vs 18 GB on C: |
| 2026-10-12 | PyTorch CUDA 12.x build targeting CUDA 12.1 | RTX 3050 driver supports CUDA 13.1 (backward compatible) |
| 2026-10-12 | mT5-small as first generation model | Fits 4 GB VRAM comfortably |
| 2026-10-12 | Mark Gemma 2 QLoRA as marginal | 4 GB VRAM is at the lower limit for 4-bit 2B model |
