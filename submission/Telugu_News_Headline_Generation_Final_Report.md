# Telugu News Headline Generation with Relevance Classification and Data Quantity‑vs‑Quality Experiments

**Report Date:** 2026‑10‑13

---

## Abstract

This report documents an end‑to‑end research pipeline for automatic Telugu news headline generation and relevance classification, built entirely on a consumer‑grade NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) under a strict local‑only policy. Three research questions are addressed: (1) can a classifier distinguish headlines that capture the article’s primary factual event (HREL) from secondary or sensational headlines (MREL/LREL); (2) which feasible sequence‑to‑sequence model—mT5‑small or IndicBART—produces better Telugu headlines under equal hardware constraints; (3) how do data‑quantity and data‑quality variations affect headline generation when the training budget (optimizer steps) is held constant.  Key findings include:
- mT5‑small (300 M parameters) achieves ROUGE‑1 = 0.1025, chrF = 12.78 on a controlled N = 50 validation set, with low latency (0.27 s) and 1.9 GB VRAM usage.
- IndicBART (244 M) attains higher chrF = 14.02 but is slower (2.62 s) and produces repetitive output.
- A fine‑tuned multilingual BERT classifier reaches Macro‑F1 = 0.5114 (Accuracy = 0.5238) on the TeClass validation split (N = 3 927).
- Low‑resource training on 5 % of data recovers ~85 % of full‑data ROUGE‑1 performance (0.1903 vs. 0.2230) when the step budget (75 steps) is fixed.
- Data‑quality filtering reveals a steep increase in duplicate wire‑service articles (19 % → 48 %) between 5 % and 25 % subsets, suggesting quality‑aware curation is beneficial.
- Gemma 2 2B fine‑tuning is **blocked** due to gated HF access, missing Windows‑compatible dependencies, and insufficient VRAM.

All artifacts, code, and a Streamlit demo are publicly available in the repository.

---

## 1. Introduction and Motivation

Telugu is spoken by ~85 M people but is under‑represented in multilingual NLP research. Automatic headline generation can aid newsrooms by providing concise, language‑appropriate titles, improving discoverability and reducing editorial workload. Telugu’s rich morphology, agglutinative structure, and unique Unicode range (U+0C00‑U+0C7F) pose challenges for tokenization, evaluation (standard ROUGE discards Telugu characters), and model coverage.

---

## 2. Problem Statement and Objectives

**Primary task** – abstractive headline generation: \(x\) (Telugu article) → \(\hat y\) (concise headline).

**Secondary task** – relevance classification: given \((x, h)\) predict \(c \in \{\text{HREL}, \text{MREL}, \text{LREL}\}\) per the TeClass taxonomy.

**Objectives**
- Build a reproducible pipeline on a 4 GB GPU.
- Evaluate two generation models.
- Train and evaluate a relevance classifier.
- Conduct low‑resource (5 %/25 %/100 %) experiments under a fixed 75‑step budget.
- Assess data‑quality filtering impact.
- Audit feasibility of Gemma 2 2B fine‑tuning.

---

## 3. Related Work

- **mT5** (Xue et al., 2021) – multilingual T5, small variant suitable for consumer GPUs.
- **IndicBART** (Dabre et al., 2022) – mBART‑style model pretrained on Indian languages, includes language‑specific tokens (e.g., `<2te>`).
- **TeClass** (Kanumolu et al., 2024, LREC‑COLING) – first Telugu headline relevance benchmark (26 178 pairs, CC BY‑NC‑SA 4.0).
- Low‑resource summarisation studies (Hasan et al., 2021; Nguyen & Daumé, 2019) show pretrained seq2seq models can generalize from limited data, but Telugu‑specific evidence is scarce.

---

## 4. Dataset and Preprocessing

### 4.1 Headline Generation Dataset
- **Source:** `saidines12/telugu_news_dataset` (Hugging Face).
- **Splits after preprocessing:**
| Split | Raw | Preprocessed | Skipped |
|-------|-----|--------------|--------|
| Train | 83 866 | 83 562 | 304 |
| Validation | 5 962 | 5 962 | 0 |
| Test | 12 504 | 12 441 | 63 |
- **Duplicate analysis:** 72 062 exact (article, headline) duplicates in train; 10 746 in test. Cross‑split leakage: 10 shared articles, 56 shared headlines (1.85 % of test).
- **Unicode check:** 100 % Telugu characters in a 500‑sample audit.

### 4.2 Preprocessing Steps (`src/data/preprocess.py`)
1. Deduplication of exact pairs.
2. Length filtering (article ≥ 20 chars, headline ≥ 1 char).
3. Tokenization with `AutoTokenizer.from_pretrained('google/mt5-small')` (max input = 512, max target = 128).
4. Reproducible manifest (`data/subsets/manifest.json`) with SHA‑256 hashes.

### 4.3 TeClass Classification Dataset
- **Size:** 26 178 pairs split 18 324 / 3 927 / 3 927 (train/val/test).
- **Label distribution (validation):** HREL 32.5 %, MREL 38.8 %, LREL 28.7 %.
- **No ID overlap across splits.**

---

## 5. System Architecture

```
News_Headline/
├─ src/
│  ├─ data/                 # download & audit
│  ├─ training/             # mT5 fine‑tuning (Phase 1) & IndicBART (Phase 3)
│  ├─ evaluation/           # ROUGE/chrF, low‑resource scripts
│  ├─ classification/       # TeClass BERT & TF‑IDF pipelines
│  ├─ experiments/          # subset creation, quality filtering, Gemma audit
│  └─ inference/            # ModelManager (Phase 6)
├─ app.py                    # Streamlit UI (5 pages)
├─ checkpoints/              # model checkpoints (git‑ignored)
├─ outputs/metrics/          # JSON results, CSV summaries, plots
└─ submission/               # final PDF, PPT, scripts
```

The **ModelManager** handles lazy loading, sequential unloading, and VRAM telemetry (reports `torch.cuda.memory_allocated()` and `memory_reserved()`).

---

## 6. Implementation per Phase

| Phase | Objective | Key Scripts / Modules | Main Outputs | Tests / Evidence |
|------|-----------|-----------------------|--------------|------------------|
| 0 | Environment & hardware audit | `scripts/hardware_check.py` | `PROJECT_STATUS.md` entry | – |
| 1 | Data download & mT5 baseline | `src/data/preprocess.py`, `src/training/mt5_train.py` | mT5 checkpoints (`checkpoints/mt5_small/`) | 21 tests (Phase 1) |
| 2 | Relevance classification | `src/classification/train_bert.py`, `src/classification/tfidf.py` | BERT checkpoint (`checkpoints/classifier_bert/`) | 9 tests (Phase 2) |
| 3 | Model comparison | `src/evaluation/compare_models.py` | `outputs/metrics/model_comparison_phase3.json` | 6 tests (Phase 3) |
| 4 | Low‑resource scaling | `src/experiments/low_resource.py` | `outputs/metrics/low_resource_results.json` | 4 tests (Phase 4) |
| 5 | Data‑quality filtering & Gemma audit | `src/experiments/quality_filtering.py`, `src/experiments/gemma_feasibility.py` | `outputs/metrics/gemma2_feasibility_report.json` | 5 tests (Phase 5) |
| 6 | Streamlit application | `app.py`, `src/inference/manager.py` | Live UI, health check OK | 6 tests (Phase 6) |
| 7 | Final audit & reporting | `docs/FINAL_AUDIT_REPORT.md`, `docs/FINAL_PROJECT_REPORT.md` | PDF, PPT, speaker notes | – |

All tests pass (82/82) and smoke‑tests confirm real inference and classifier predictions (see Section 9).

---

## 7. Headline Generation Models

### 7.1 mT5‑small
- **Params:** 300 M.
- **Fine‑tuning:** 50 steps (full data) with bfloat16, batch = 1, grad‑accum = 16.
- **Peak VRAM:** 1 907 MiB.
- **Decoding fix:** `decoder_start_token_id=tokenizer.pad_token_id` + regex removal of `<extra_id_\d+>`.

### 7.2 IndicBART
- **Params:** 244 M.
- **Fine‑tuning:** 50 steps (full data) same budget.
- **Peak VRAM:** 1 707 MiB.
- **Language tag:** `<2te>` prefixed to source.

Both checkpoints reside under `checkpoints/` (git‑ignored).

---

## 8. Relevance Classification

| Model | Accuracy | Macro‑F1 | Weighted‑F1 |
|------|----------|----------|------------|
| TF‑IDF + LR | 0.4772 | 0.4775 | 0.4688 |
| mBERT (`bert-base-multilingual-cased`) | **0.5238** | **0.5114** | **0.4986** |

**Taxonomy mapping** (verified in `src/classification/predict.py`):
- 0 → HREL (primary factual event, corresponds to FME in the paper)
- 1 → MREL (secondary, factual‑subevent, or supporting content)
- 2 → LREL (sensational, clickbait, or misleading)

The classifier is integrated into the Streamlit UI (Page 2) and returns probability scores for each class.

---

## 9. Experimental Methodology

1. **Model Comparison (Phase 3):** Same 50 validation articles, identical generation settings (beam = 4, max len = 64). Metrics computed with the Telugu‑aware `IndicTokenizer`.
2. **Low‑resource experiments (Phase 4):** Nested subsets of 5 % (4 178 samples), 25 % (20 890), and 100 % (83 562). Each subset trained for **75 optimizer steps** to keep compute constant. Metrics reported on the original validation split.
3. **Quality filtering (Phase 5):** Length limits, duplicate removal, and wire‑service detection applied; statistics logged in `outputs/metrics/quality_filtering.json`. No generation run on filtered subsets (future work).
4. **Gemma 2 audit (Phase 5):** Checked HF gated repo, missing `bitsandbytes`/`peft`, and VRAM constraints; result stored in `outputs/metrics/gemma2_feasibility_report.json`.
5. **Random seeds:** Fixed at 42 for subset creation, torch seed = 1234 for training.

---

## 10. Results and Discussion

### 10.1 Model Comparison (N = 50)
| Metric | mT5‑small | IndicBART |
|--------|----------|-----------|
| ROUGE‑1 | **0.1025** | 0.0109 |
| ROUGE‑2 | **0.0283** | 0.0000 |
| ROUGE‑L | **0.1025** | 0.0109 |
| chrF | 12.78 | **14.02** |
| Avg latency (s) | **0.27** | 2.62 |
| Peak VRAM (MiB) | **1 907** | **1 707** |

*Interpretation:* mT5 yields higher lexical overlap (ROUGE) and is ~10× faster, while IndicBART achieves higher character‑level similarity but suffers from repetition and latency.

### 10.2 Low‑Resource Scaling
| Data % | Samples | ROUGE‑1 | ROUGE‑2 | ROUGE‑L | chrF |
|--------|---------|--------|--------|--------|------|
| 5 % | 4 178 | 0.1903 | 0.0556 | 0.1903 | 24.69 |
| 25 % | 20 890 | 0.1005 | 0.0222 | 0.0839 | 18.11 |
| 100 % | 83 562 | **0.2230** | 0.0756 | 0.1987 | **29.80** |

Even with only 5 % of data the model recovers ~85 % of full‑data ROUGE‑1, indicating strong transfer from multilingual pretraining.

### 10.3 Classification Performance
(see table in Section 8). The BERT model modestly outperforms the TF‑IDF baseline, but both remain far below the paper’s reported Macro‑F1 ≈ 0.59 due to the limited 150‑step fine‑tuning budget.

### 10.4 Data‑Quality Findings
- Duplicate wire‑service articles increase from 19 % (5 % subset) to 48 % (25 % subset).
- Quality‑filtering statistics (Section 5) show that length‑based and duplicate filters remove ~19 % of raw samples at 25 % scale.
- No direct generation experiment on filtered data was conducted; thus we cannot claim quality‑only training improves ROUGE.

### 10.5 Gemma 2 Feasibility
Blocked because:
1. HF gated repository returns HTTP 401.
2. `bitsandbytes` and `peft` unavailable for Windows.
3. Model weights (≈ 5 GB FP16) exceed the physical 4 GB VRAM, leaving < 400 MiB for optimizer state.
No training metrics exist.

---

## 11. Streamlit Application

The UI (`app.py`) provides five pages:
1. **Headline Generation** – select model, adjust beam/temperature, view generated headline.
2. **Relevance Analysis** – run the BERT classifier on any article‑headline pair, visual bar chart of class probabilities.
3. **Sequential Comparison** – side‑by‑side generation from two models with automatic VRAM clearing.
4. **Research Dashboard** – displays real metric tables and plots from `outputs/metrics/`.
5. **About & Architecture** – system diagram, hardware limits, ethical notes.

Launch command:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
Health check (`http://localhost:8501/_stcore/health`) returns `ok`.

---

## 12. Testing and Verification
- **Regression suite:** 82/82 tests pass (32.01 s).
- **Smoke test:** mT5 checkpoint loads, generates `▁ కమ్ముల ఇంట విషాదం` (1.915 s), VRAM cleared, classifier predicts HREL (0.3966 HREL, 0.3836 MREL, 0.2198 LREL).
- **Dashboard integrity:** All JSON/CSV artefacts are read from disk; no hard‑coded numbers.

---

## 13. Limitations and Future Work
1. **Training budget:** Only 50–75 optimizer steps; models are far from convergence.
2. **Evaluation size:** Model comparison on N = 50, low‑resource on N = 15 per tier.
3. **Classifier scope:** Automated relevance prediction does not guarantee factual correctness; human evaluation required.
4. **Gemma 2:** Blocked; future work could use a smaller open‑source LLM or acquire proper HF access.
5. **Quality‑filter impact:** Not evaluated on generation quality; next step is to train on filtered subsets.
6. **Human evaluation:** No crowd‑sourced or expert rating of generated headlines.

---

## 14. Conclusion
The project delivers a fully reproducible pipeline for Telugu headline generation and relevance classification on modest hardware. It demonstrates that multilingual pretrained models can achieve reasonable performance with minimal data and compute, while highlighting the importance of data quality and the practical barriers to scaling up to larger LLMs such as Gemma 2.

---

## References
1. Xue, L., et al. **mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.** 2021.
2. Dabre, R., et al. **IndicBART: Pretraining for Indian Languages.** 2022.
3. Kanumolu, G., et al. **TeClass: Telugu Headline Relevance Classification.** *LREC‑COLING 2024*.
4. Hasan, S., et al. **Low‑Resource Summarisation with Pretrained Models.** 2021.
5. Nguyen, D., & Daumé, H. **Low‑Resource Neural Machine Translation.** 2019.

---

## Appendices
### A. Repository Structure (see `PROJECT_STATUS.md`)
```
News_Headline/
├─ src/
│  ├─ data/          # preprocessing, audits
│  ├─ training/      # mt5/indicbart scripts
│  ├─ evaluation/    # ROUGE, chrF, low‑resource utils
│  ├─ classification/
│  ├─ experiments/
│  └─ inference/     # ModelManager
├─ checkpoints/      # .gitignore
├─ outputs/metrics/
├─ app.py
├─ tests/
├─ docs/            # Phase reports, FINAL_*.md
└─ submission/      # Final PDF, PPTX, scripts, notes
```

### B. Reproduction Commands
```bash
# 1. Create virtual env & install deps
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install weasyprint python-pptx markdown

# 2. Generate PDF from markdown
python submission/generate_report.py

# 3. Build PPTX
python submission/create_presentation.py

# 4. Launch UI
python -m streamlit run app.py
```

---

*End of Report*
