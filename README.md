# Telugu News Headline Generator

## Academic Research Project — Telugu NLP

An end-to-end research project for **Telugu article-to-headline generation** and **headline relevance classification**, trained locally on an NVIDIA RTX 3050.

---

## Research Questions

| ID  | Question |
|-----|----------|
| RQ1 | Can a relevance classifier identify Telugu headlines that represent the article's main factual event versus other annotated categories? |
| RQ2 | Which feasible generation model performs best under a controlled Telugu headline-generation evaluation? |
| RQ3 | How does headline-generation performance change with 5%, 10%, 25%, 50%, and 100% of the training data? |
| RQ4 | Does training-data quantity or quality have a greater impact on Gemma 2 Telugu headline generation? |

---

## Hardware Target

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 3050 (Laptop) |
| VRAM | **4 GB** |
| Driver | 592.82 |
| CUDA | 13.1 (driver-reported) |
| OS | Windows 11 |

> **Note:** All training is designed for local GPU execution. No paid cloud GPU is required.

---

## Candidate Models

| Model | Role | Status |
|---|---|---|
| mT5-small (300M) | Generation baseline 1 | ✅ Fine-Tuned & Evaluated (Phase 1, 3, 4) |
| TF-IDF + Logistic Regression | Relevance Classifier baseline | ✅ Trained & Evaluated (Phase 2) |
| mBERT (`bert-base-multilingual-cased`) | Relevance Classifier Transformer | ✅ Trained & Evaluated on RTX 3050 (Phase 2) |
| IndicBART (~244M) | Generation baseline 2 | ✅ Fine-Tuned & Evaluated (Phase 3) |
| Gemma 2 2B + QLoRA | Advanced generation | 🚫 BLOCKED — Gated HF Auth & Windows BnB (Phase 5) |

---

## Project Structure

```
News_Headline/
├── src/                    # Source modules
├── data/
│   ├── raw/                # Downloaded datasets (gitignored)
│   ├── processed/          # Cleaned & tokenized data
│   └── splits/             # Train/val/test splits
├── checkpoints/            # Model checkpoints (gitignored)
├── outputs/
│   ├── predictions/        # Model predictions
│   ├── metrics/            # Evaluation JSON reports
│   └── plots/              # Learning curves, confusion matrices
├── logs/                   # Training logs
├── tests/                  # Automated test suite
├── docs/                   # Documentation and plans
├── venv/                   # Virtual environment (gitignored)
├── README.md
├── PROJECT_STATUS.md
└── requirements.txt
```

---

## Quick Start

### 1. Activate environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify GPU
```powershell
python tests/test_gpu.py
```

### 4. Launch Telugu Headline Studio (Streamlit)
```powershell
python -m streamlit run app.py
```

### 5. Run complete test suite
```powershell
python -m pytest tests/ -v
```

---

## Documentation & Research Reports

- Final Comprehensive Audit: [`docs/FINAL_AUDIT_REPORT.md`](docs/FINAL_AUDIT_REPORT.md)
- Complete Research Paper & Submission Report: [`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md)
- Phase-by-phase status tracking: [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
- Engineering reproduction commands: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)

---

## Datasets

| Dataset | Purpose | Source | License |
|---------|---------|--------|---------|
| `saidines12/telugu_news_dataset` | Headline generation | Hugging Face Hub (Andhra Jyothi & Eenadu) | Public access |
| TeClass (Kanumolu et al., LREC-COLING 2024) | Relevance classification | GitHub / ACL Anthology | CC BY-NC-SA 4.0 |

---

## Reproducibility

- All random seeds are documented and fixed.
- All experiments log to `logs/`.
- All results are saved to `outputs/metrics/`.
- Commands to reproduce each phase are in `docs/PROJECT_PLAN.md`.

---

## License

Academic research use only. See individual dataset licenses.
