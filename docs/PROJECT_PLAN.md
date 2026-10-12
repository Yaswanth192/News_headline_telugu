# PROJECT PLAN — Telugu News Headline Generator

> Version: 1.0 | Date: 2026-10-12 | Status: ACTIVE

---

## Overview

This document records the full implementation plan, phase commands, design decisions, and reproducibility notes for the Telugu News Headline Generator academic research project.

---

## Hardware Configuration

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| Python | 3.11.0 |
| OS | Windows 11 |
| Project Drive | D:\ (88 GB free) |

---

## Environment Setup

### Create and Activate Virtual Environment

```powershell
# From project root
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install PyTorch with CUDA 12.1 support

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> CUDA 12.1 wheel is compatible with driver 592.82 (CUDA 13.1 driver is backward compatible).

### Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### Verify GPU

```powershell
python tests/test_gpu.py
```

---

## Phase 0 — Hardware Audit & Foundation

**Goal:** Establish clean, verified environment. No model training.

### Commands Run

```powershell
nvidia-smi
python --version
git --version
Get-PSDrive -PSProvider FileSystem  # disk space
python -m venv venv
python tests/test_gpu.py            # after pip install torch
python -m pytest tests/ -v
git init
git add .
git commit -m "Phase 0: project foundation"
```

### Artifacts

- `.gitignore`
- `README.md`
- `PROJECT_STATUS.md`
- `docs/PROJECT_PLAN.md` (this file)
- `requirements.txt`
- `tests/test_gpu.py`
- `tests/test_environment.py`
- `src/utils/hardware.py`

---

## Phase 1 — Dataset & Baseline

**Goal:** Download and validate dataset; mT5-small smoke test.

### Dataset

- **Source:** `saidines12/telugu_news_dataset` (Hugging Face)
- **Task:** article → headline (seq2seq)
- **Expected columns:** `article`, `headline` (to be verified)

### Commands

```powershell
# Download and validate
python src/data/download_dataset.py

# Preprocessing
python src/data/preprocess.py

# Baseline smoke test (mT5-small, 50 steps)
python src/training/train_mt5.py --smoke-test

# Full training
python src/training/train_mt5.py --config configs/mt5_small.yaml

# Evaluate
python src/evaluation/evaluate.py --checkpoint checkpoints/mt5_small/best --split test
```

### Evaluation Metrics

- ROUGE-1, ROUGE-2, ROUGE-L
- chrF
- BERTScore (optional, if memory allows)

---

## Phase 2 — Relevance Classification

**Goal:** TeClass-based classifier for Telugu headline relevance categories.

### Verified Label Taxonomy (Kanumolu et al., LREC-COLING 2024)

| Primary Category | Full Name | Associated Sub-Classes | Description |
|---|---|---|---|
| **HREL** | Highly Related | **FME** (Factual Main Event) | Headline directly and accurately states the primary factual event of the article. |
| **MREL** | Moderately Related | **FSE** (Factual Secondary Event), **STC** (Strong Conclusion), **WKC** (Weak Conclusion) | Headline focuses on secondary details, inferences, or partial takeaways. |
| **LREL** | Least Related | **SEN** (Sensational), **CBT** (Clickbait), **MLC** (Misleading Conclusion), **USO** (Unsupported Opinion) | Headline is sensationalized, clickbait, misleading, or out of scope. |

> **Finding:** HREL, MREL, and LREL form the 3 primary relevance tiers. HREL corresponds directly to headlines covering the Factual Main Event (FME). In binary relevance, Relevant = {FME, STC, FSE} and Less Relevant = {WKC, MLC, SEN, CBT}.

### Commands

```powershell
# Audit TeClass dataset
python src/classification/audit_teclass.py

# Preprocess TeClass splits
python src/classification/preprocess_teclass.py

# Train classical baseline (TF-IDF + Logistic Regression)
python src/classification/train_baseline.py

# Train transformer classifier (mBERT on local RTX 3050 GPU)
python src/classification/train_transformer.py --max-steps 150

# Run standalone inference CLI
python src/classification/predict.py --headline "<headline>" --article "<article>" --model-type bert
```

### Metrics

- Macro-F1
- Weighted-F1
- Per-class precision, recall, F1
- Confusion matrix (saved to `outputs/plots/`)

---

## Phase 3 — Model Comparison

**Goal:** Compare mT5-small, IndicBART (if feasible), and Gemma 2 (if feasible) on identical test set.

### Models

| Model | Parameters | VRAM Est. (FP16) | Status |
|-------|-----------|------------------|--------|
| mT5-small | 300M | ~1.2 GB | ✅ Planned |
| IndicBART | 244M | ~0.8 GB | ⚠️ Verify compatibility |
| Gemma 2 2B (QLoRA 4-bit) | 2B | ~3–4 GB | ⚠️ Smoke test required |

### Commands

```powershell
python src/evaluation/compare_models.py --models mt5_small indicbart gemma2 --split test
```

---

## Phase 4 — Low-Resource Learning

**Goal:** Train on 5%, 10%, 25%, 50%, 100% of training data.

### Design

- Validation and test sets are **fixed** across all runs.
- Seeds: 42, 123, 456 (report mean ± std where runs permit).
- Results saved per fraction to `outputs/metrics/low_resource/`.

### Commands

```powershell
python src/experiments/low_resource.py --fractions 0.05 0.10 0.25 0.50 1.00 --seed 42
python src/experiments/plot_learning_curve.py
```

---

## Phase 5 — Gemma 2 Quantity vs Quality

**Goal:** Isolate the effect of data quantity vs quality for Gemma 2 Telugu generation.

### Prerequisite

- Gemma 2 QLoRA smoke test must pass on 4 GB VRAM before full run.
- If smoke test fails: **BLOCKED**, document and report.

### Commands

```powershell
# Memory feasibility test
python src/experiments/gemma2_memory_test.py

# Full experiment (only if feasibility confirmed)
python src/experiments/gemma2_qty_vs_quality.py
```

---

## Phase 6 — Streamlit Application

**Goal:** Modern UI for headline generation and classification.

### Commands

```powershell
streamlit run app/streamlit_app.py
```

### Features

- Telugu article input
- Headline generation (from best trained model)
- Relevance classification with confidence
- Model comparison table (from saved results)
- Error handling for missing checkpoints

---

## Phase 7 — Final Report

**Goal:** Full academic report with reproducible results.

### Commands

```powershell
python src/reporting/generate_tables.py
python src/reporting/generate_plots.py
```

---

## Reproducibility Notes

- All seeds: `SEED = 42` (primary), documented per experiment
- HuggingFace cache: `D:\large documents\Large Projects\News_Headline\.cache\huggingface`
- Checkpoints: `checkpoints/<model_name>/`
- All runs log to `logs/<run_name>_<timestamp>.log`
- Resume: all training scripts accept `--resume-from <checkpoint_path>`

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gemma 2 OOM on 4 GB VRAM | High | Medium | Use QLoRA 4-bit; smoke test first; mark BLOCKED if fails |
| TeClass not publicly accessible | Medium | High | Document limitation; use alternative annotation scheme if approved |
| mT5-small poor ROUGE on Telugu | Medium | Low | Expected; IndicBART likely better |
| C: drive full (18 GB) | High | High | All large files on D: drive |
