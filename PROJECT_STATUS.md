# PROJECT STATUS — Telugu News Headline Generator

> Last updated: 2026-10-12

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
| 0 | Hardware Audit & Foundation | 🟡 IN PROGRESS | 2026-10-12 |
| 1 | Dataset & Baseline | ⬜ NOT STARTED | — |
| 2 | Relevance Classification | ⬜ NOT STARTED | — |
| 3 | Model Comparison | ⬜ NOT STARTED | — |
| 4 | Low-Resource Learning | ⬜ NOT STARTED | — |
| 5 | Gemma 2 Quantity vs Quality | ⬜ NOT STARTED | — |
| 6 | Streamlit Application | ⬜ NOT STARTED | — |
| 7 | Final Research Report | ⬜ NOT STARTED | — |

---

## Phase 0 — Hardware Audit & Clean Foundation

**Status:** 🟡 IN PROGRESS

### Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Actual hardware findings recorded | ✅ PASSED | nvidia-smi output above |
| GPU VRAM identified (not assumed) | ✅ PASSED | 4096 MiB confirmed |
| GPU visibility test (PyTorch CUDA) | 🟡 PENDING | PyTorch not yet installed in venv |
| Python version checked | ✅ PASSED | Python 3.11.0 |
| Git version checked | ✅ PASSED | git 2.47.0.windows.2 |
| Disk space recorded | ✅ PASSED | D: 88.36 GB free |
| Virtual environment created | ✅ PASSED | `venv/` created |
| PyTorch CUDA build installed | 🟡 PENDING | Installation in progress |
| Minimal GPU tensor test passes | 🟡 PENDING | Awaiting PyTorch install |
| .gitignore created | ✅ PASSED | `.gitignore` committed |
| README.md created | ✅ PASSED | `README.md` committed |
| PROJECT_STATUS.md created | ✅ PASSED | This file |
| docs/PROJECT_PLAN.md created | 🟡 PENDING | In progress |
| Test suite created and passes | 🟡 PENDING | `tests/` in progress |
| No model training performed | ✅ PASSED | Not started |
| No existing user files deleted | ✅ PASSED | Verified directory was empty |

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

**Status:** ⬜ NOT STARTED

*Waiting for Phase 0 approval.*

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
