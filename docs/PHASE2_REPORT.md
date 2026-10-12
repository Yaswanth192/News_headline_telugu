# Phase 2 Report — Telugu Headline Relevance Classification

> **Date:** 2026-10-12  
> **Status:** ✅ PASSED  
> **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4095 MiB VRAM), Local Windows 11  

---

## 1. Executive Summary

Phase 2 investigates, validates, and implements the complete **Telugu Headline Relevance Classification** research pipeline based on the peer-reviewed TeClass benchmark:
- Successfully located and cloned the official research repository and dataset from LREC-COLING 2024: *TeClass: A Human-Annotated Relevance-based Headline Classification and Generation Dataset for Telugu* (Kanumolu et al.).
- Verified the complete research taxonomy and relationship between relevance categories (`HREL`, `MREL`, `LREL`) and semantic sub-classes (`FME`, `FSE`, `STC`, `WKC`, `SEN`, `CBT`, `MLC`, `USO`).
- Preprocessed all 26,178 article-headline pairs across official train, validation (development), and test splits.
- Implemented and evaluated two classification paradigms:
  1. **Baseline Classifier:** TF-IDF n-grams + Logistic Regression (Macro-F1: 0.4775).
  2. **Transformer Classifier:** Multilingual BERT (`google-bert/bert-base-multilingual-cased`) fine-tuned on the local RTX 3050 in bfloat16 (Macro-F1: 0.5114, Accuracy: 0.5238, Peak VRAM: 3,436 MiB).
- Developed a standalone inference pipeline [`src/classification/predict.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/src/classification/predict.py) returning class predictions, probability distributions, model checkpoints, and human-interpretable taxonomy explanations.
- All 61 tests across the entire repository (Phase 0, Phase 1, Phase 2) pass (100% green).

---

## 2. Research Verification & Annotation Taxonomy

### Paper Information & Source Verification
- **Title:** *TeClass: A Human-Annotated Relevance-based Headline Classification and Generation Dataset for Telugu*
- **Authors:** Gopichand Kanumolu, Lokesh Madasu, Nirmal Surange, Manish Shrivastava (LTRC, IIIT Hyderabad)
- **Publication:** Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 15711–15720, Torino, Italia.
- **ACL Anthology URL:** [https://aclanthology.org/2024.lrec-main.1364/](https://aclanthology.org/2024.lrec-main.1364/)
- **Official Code & Dataset Repository:** [https://github.com/gopichandkanumolu/TeClass](https://github.com/gopichandkanumolu/TeClass)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

### Taxonomy Architecture & Relationship Between HREL and FME

Our research reveals the formal hierarchy used by human annotators:
`HREL`, `MREL`, and `LREL` are **not** disjoint parallel classes to `FME`, `FSE`, etc.; rather, they form the **primary 3-class relevance partition**, under which the fine-grained semantic sub-categories reside:

```
Relevance Classification (TeClass Taxonomy)
├── HREL (Highly Relevant / High Relevance)
│   └── FME: Factual Main Event (The headline explicitly and accurately captures the primary event of the article)
├── MREL (Moderately Relevant / Medium Relevance)
│   ├── FSE: Factual Secondary Event (Captures a secondary event/sub-event factually present in the article)
│   ├── STC: Strong Conclusion (Logically sound conclusion/inference capturing the majority of the article)
│   └── WKC: Weak Conclusion (Conclusion inferred from only a minor, isolated portion of the article)
└── LREL (Least Relevant / Low Relevance)
    ├── SEN: Sensational (Biased, emotionally loaded framing that distorts the story to attract attention)
    ├── CBT: Clickbait (Extreme gap between promise and substance; creates suspense/withholds information)
    ├── MLC: Misleading Conclusion (Vague, false, or distorted conclusion unsupported by the article)
    └── USO: Unsupported Opinion (Subjective opinions stated without backing in the article text)
```

**Key Finding on HREL vs. FME:**  
`HREL` is the primary macro-class that represents headline-article pairs where the headline reflects the **Factual Main Event (FME)**. When binary relevance classification is evaluated in the paper, it groups `Relevant` = {FME + STC + FSE} vs. `Less Relevant` = {WKC + MLC + SEN + CBT}. In the official dataset release files, the ground-truth target is provided at the verified 3-class tier: `HREL`, `MREL`, and `LREL`.

---

## 3. Dataset Audit & Preprocessing Statistics

The TeClass dataset provides human annotations for **26,178** article-headline pairs with majority-vote consensus from 3 native Telugu annotators.

### Split Distribution

| Split | Raw Rows | Preprocessed Rows | HREL Count | MREL Count | LREL Count | Columns |
|---|---|---|---|---|---|---|
| **Train** | 18,324 | 18,324 | 5,962 (32.5%) | 7,105 (38.8%) | 5,257 (28.7%) | `id`, `url`, `article`, `headline`, `label` |
| **Validation (Dev)** | 3,927 | 3,927 | 1,277 (32.5%) | 1,523 (38.8%) | 1,127 (28.7%) | `id`, `url`, `article`, `headline`, `label` |
| **Test** | 3,927 | 3,927 | 1,278 (32.5%) | 1,523 (38.8%) | 1,126 (28.7%) | `id`, `url`, `article`, `headline`, `label` |
| **Total** | **26,178** | **26,178** | **8,517 (32.5%)** | **10,151 (38.8%)** | **7,510 (28.7%)** | — |

- **Missing / Malformed Records:** 0 null values across all three splits.
- **Split Ratio:** Exactly 70% Train, 15% Validation, 15% Test.
- **Cross-Split Leakage:**
  - Train vs. Validation ID overlap = 0.
  - Train vs. Test ID overlap = 0.
  - Validation vs. Test ID overlap = 0.
  - Article exact overlap between train and validation is minimal (4 articles out of 3,927, <0.1%), representing recurring wire reports with distinct headlines.
- **Audit Artifact:** [`outputs/metrics/teclass_audit.json`](file:///D:/large%20documents/Large%20Projects/News_Headline/outputs/metrics/teclass_audit.json)

---

## 4. Models and Training Configurations

All models were trained locally on the RTX 3050 Laptop GPU and CPU without external APIs.

### Model 1: Baseline Classifier (TF-IDF + Logistic Regression)
- **Features:** Word and subword n-grams (1, 2) up to 30,000 features.
- **Input representation:** `f"{headline} [SEP] {article}"`
- **Classifier:** Scikit-Learn Logistic Regression with balanced class weighting.
- **Execution:** CPU, completed in 5.4 seconds.
- **Checkpoint:** [`checkpoints/classifier_tfidf/tfidf_logreg.pkl`](file:///D:/large%20documents/Large%20Projects/News_Headline/checkpoints/classifier_tfidf/tfidf_logreg.pkl)

### Model 2: Transformer Classifier (Multilingual BERT)
- **Pretrained Architecture:** `google-bert/bert-base-multilingual-cased` (177.8M parameters).
- **Access Status:** Public, ungated open checkpoint (replaces gated `ai4bharat/indic-bert`).
- **Precision:** `torch.bfloat16` on RTX 3050 (Compute 8.6).
- **Batch Size:** 4 per device, gradient accumulation 4 (effective batch size = 16).
- **Sequence Length:** 256 tokens.
- **Learning Rate:** 2e-5 with AdamW and linear weight decay (0.01).
- **Peak VRAM:** **3,436 MiB** (Safely fits within the 4,095 MiB hardware budget; zero CUDA OOM).
- **Checkpoint:** [`checkpoints/classifier_bert/final`](file:///D:/large%20documents/Large%20Projects/News_Headline/checkpoints/classifier_bert/final)

---

## 5. Evaluation Metrics & Comparison

Evaluated on the official held-out **Validation Split** (3,927 examples):

| Metric | TF-IDF + Logistic Regression | mBERT (Transformer) | Published TeClass Paper (mBERT) |
|---|---|---|---|
| **Accuracy** | 0.4772 | **0.5238** | — |
| **Macro-F1** | 0.4775 | **0.5114** | ~0.5900 |
| **Weighted-F1** | 0.4688 | **0.4986** | ~0.5900 |
| **HREL F1-Score** | 0.5050 | **0.6225** | 0.6600 |
| **MREL F1-Score** | 0.4357 | 0.3423 | 0.5000 |
| **LREL F1-Score** | 0.4918 | **0.5693** | 0.6200 |

### Per-Class Detailed Performance (mBERT)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **HREL** (High Relevance / FME) | 0.5002 | **0.8238** | **0.6225** | 1,277 |
| **MREL** (Medium Relevance / FSE, STC, WKC) | 0.4804 | 0.2659 | 0.3423 | 1,523 |
| **LREL** (Low Relevance / Clickbait, Sensational) | **0.6116** | 0.5324 | **0.5693** | 1,127 |

### Confusion Matrix (mBERT Validation Set)
Rows = Ground Truth, Columns = Predicted:
```
           Pred: HREL    Pred: MREL    Pred: LREL
True HREL:     1052          147           78
True MREL:      815          405          303
True LREL:      236          291          600
```

### Error Analysis & Research Insights
1. **Strong Main-Event (HREL) Recall:** mBERT achieves 82.4% recall on HREL, identifying headlines that represent factual main events.
2. **Boundary Confusion between HREL and MREL:** The primary source of error is false positives where MREL examples are classified as HREL (815 cases). In Telugu journalism, differentiating between a "main event" (FME) and a "salient secondary event" (FSE) or "strong conclusion" (STC) involves nuanced semantic scope that challenges sentence-pair cross-attention at length 256.
3. **Robust Clickbait / Irrelevant Detection:** LREL has the highest precision (61.2%), making it suitable as a filter for noisy web-scraped headlines.

---

## 6. Integration and Prediction API

The standalone interface is located at [`src/classification/predict.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/src/classification/predict.py).

### CLI Usage
```powershell
python src/classification/predict.py `
  --headline "శేఖర్ కమ్ముల తండ్రి శేషయ్య కన్నుమూత" `
  --article "ప్రముఖ దర్శకులు శేఖర్ కమ్ముల ఇంట విషాదం నెలకొంది . ఆయన తండ్రి కమ్ముల శేషయ్య ( 89 ) కన్నుమూశారు ." `
  --model-type bert
```

### Python API
```python
from src.classification.predict import predict_relevance

result = predict_relevance(
    article="ప్రముఖ దర్శకులు శేఖర్ కమ్ముల ఇంట విషాదం నెలకొంది...",
    headline="శేఖర్ కమ్ముల తండ్రి శేషయ్య కన్నుమూత",
    model_type="bert"
)
# Returns:
# {
#   "predicted_class": "HREL",
#   "probabilities": {"HREL": 0.4533, "MREL": 0.3565, "LREL": 0.1902},
#   "model_type": "mBERT (google-bert/bert-base-multilingual-cased)",
#   "checkpoint": "checkpoints/classifier_bert/final",
#   "explanation": "Highly Related: Headline directly and factually captures the article's core main event (FME - Factual Main Event)."
# }
```

---

## 7. Automated Test Suite Results

Ran complete test suite across the codebase:
```powershell
python -m pytest tests/ -v
```
**Results:** **61 passed out of 61 tests (100% passing in 31.12s)**:
- Phase 0: 31/31 passing (Hardware audit, environment, structure)
- Phase 1: 21/21 passing (Dataset validation, preprocessing, mT5 training and evaluation)
- Phase 2: 9/9 passing (TeClass schemas, labels, preprocessing, TF-IDF baseline, mBERT inference, API contracts)

---

## 8. Acceptance Criteria Verification

| Criterion | Status | Evidence |
|---|---|---|
| Original paper and dataset sources documented | ✅ PASSED | LREC-COLING 2024 paper cited; repo cloned and verified |
| Annotation definitions traceable to source | ✅ PASSED | HREL, MREL, LREL taxonomy mapped to fine-grained classes |
| Label mapping matches verified taxonomy | ✅ PASSED | {0: HREL, 1: MREL, 2: LREL} |
| Dataset audit report saved | ✅ PASSED | [`outputs/metrics/teclass_audit.json`](file:///D:/large%20documents/Large%20Projects/News_Headline/outputs/metrics/teclass_audit.json) |
| Preprocessing tested and split integrity verified | ✅ PASSED | 26,178 records partitioned; zero ID overlap |
| Classical ML baseline trained | ✅ PASSED | TF-IDF + Logistic Regression saved to disk |
| Local GPU transformer classifier trained | ✅ PASSED | mBERT fine-tuned on RTX 3050 (Peak VRAM: 3,436 MiB) |
| Validation metrics and confusion matrix reported | ✅ PASSED | Saved in [`outputs/metrics/`](file:///D:/large%20documents/Large%20Projects/News_Headline/outputs/metrics/) |
| Standalone inference API works | ✅ PASSED | [`src/classification/predict.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/src/classification/predict.py) tested via CLI and tests |
| Phase 1 generation pipeline unmodified | ✅ PASSED | mT5 codebase and tests untouched and passing |
| 100% tests green | ✅ PASSED | 61/61 passing |
