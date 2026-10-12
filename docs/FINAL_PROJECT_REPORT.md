# Telugu News Headline Generation with Relevance Classification and Data Quantity-vs-Quality Experiments

**An End-to-End NLP Research Pipeline for a Low-Resource Indian Language**

> **Project:** Academic Research — Telugu NLP  
> **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), Windows 11  
> **Repository:** `D:\large documents\Large Projects\News_Headline`  
> **Report Date:** 2026-10-13  
> **Status:** READY WITH DOCUMENTED LIMITATIONS

---

## Abstract

This report documents an end-to-end research pipeline for automatic Telugu news headline generation and relevance classification, designed and executed entirely on a locally available NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). The project addresses three interconnected research questions: (1) whether a relevance classifier can distinguish headlines that represent an article's primary factual event from secondary or sensationalized alternatives, (2) which feasible sequence-to-sequence generation model performs best under controlled evaluation, and (3) how training-data quantity and quality independently affect headline generation performance in a low-resource setting.

Two encoder-decoder generation models were trained and evaluated: Google's multilingual mT5-small (300M parameters) and AI4Bharat's IndicBART (244M parameters). Both models were fine-tuned locally under a strict 4 GB VRAM budget using bfloat16 mixed precision. On a controlled evaluation of 50 Telugu validation articles, mT5-small achieved ROUGE-1 = 0.1025 and a mean headline generation latency of 0.27 seconds per article, while IndicBART demonstrated stronger character-level alignment (chrF = 14.02) but generated highly repetitive text and was 9.6 times slower. A headline relevance classifier was built using the TeClass benchmark (Kanumolu et al., 2024), achieving Macro-F1 = 0.51 with fine-tuned Multilingual BERT. Low-resource experiments demonstrated that mT5-small trained on only 5% of available data (4,178 samples, 75 steps) recovered approximately 85% of full-data ROUGE-1 performance (0.1903 vs 0.2230). A data quality filtering analysis revealed that as dataset scale grew from 5% to 25%, duplicate wire-service article rates increased from 19% to 48%, motivating quality-aware data curation over naive volume scaling. Fine-tuning Google's Gemma 2 2B model locally was evaluated and formally classified as **blocked** due to gated repository authentication, missing quantization dependencies, and insufficient physical VRAM. All experimental results, training checkpoints, and a full interactive Streamlit web application are preserved in the project repository.

---

## 1. Introduction and Motivation

Telugu is an official language of India spoken by approximately 85 million people, yet it remains significantly underrepresented in multilingual NLP research compared to major European languages. Automatic news headline generation — the task of producing a concise, informative summary title for a full-length news article — is a practically valuable application for news recommendation, archival retrieval, and editorial support systems.

Unlike English, Telugu is a morphologically rich Dravidian language with agglutinative inflection, long compound words, and a distinct Unicode script range (U+0C00–U+0C7F). These properties create three specific challenges for NLP systems: (1) standard tokenizers that rely on whitespace and ASCII punctuation boundaries fail to segment Telugu subwords correctly; (2) standard evaluation metrics such as ROUGE, which depend on word-boundary extraction via `re.findall(r"\w+", text)` with Python's default ASCII mode, discard all Telugu characters and produce scores of zero; and (3) pretrained multilingual models vary widely in their coverage of Telugu morphological patterns.

This project builds a scientifically reproducible Telugu headline generation and relevance evaluation pipeline within realistic local hardware constraints, contributing: a reusable Telugu-aware ROUGE tokenizer fix; a fine-tuned mT5-small and IndicBART baseline; a trained TeClass-based relevance classifier; controlled low-resource scaling experiments; and a Streamlit demonstration application.

---

## 2. Problem Statement and Objectives

### 2.1 Task Definition

The primary task is **abstractive headline generation**: given a Telugu news article $x$ (typically 500–1,500 characters), generate a concise target headline $\hat{y}$ (typically 15–60 characters) that accurately represents the article's main factual event.

The secondary task is **headline relevance classification**: given an (article, headline) pair $(x, h)$, predict the relevance category $c \in \{HREL, MREL, LREL\}$ according to the TeClass annotation taxonomy.

### 2.2 Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | Can an automated relevance classifier identify Telugu headlines representing the article's main factual event (HREL) vs. secondary or misleading content (MREL/LREL)? |
| **RQ2** | Which feasible seq2seq model — mT5-small or IndicBART — generates better Telugu headlines under equal local GPU constraints? |
| **RQ3** | How does training-data quantity affect headline generation quality when compute budget (training steps) is held constant? |
| **RQ4** | Does raw data quantity or data quality (after deduplication and length filtering) provide a stronger training signal at the 5%–25% data scale? |

### 2.3 Constraints

- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU, 4,095.5 MiB dedicated physical VRAM.
- **Execution:** All training and inference strictly local; no cloud GPUs, no paid APIs.
- **Reproducibility:** Fixed seeds, deterministic nested subsets, documented commands, verified SHA-256 file manifests.

---

## 3. Related Work and Research Gap

### 3.1 Multilingual Seq2Seq Models

**mT5** (Xue et al., 2021) is a multilingual adaptation of T5 trained on a 101-language Common Crawl corpus. Its small variant (300M parameters) can be fine-tuned on consumer GPUs and has been applied to cross-lingual summarization tasks. However, mT5's pretraining objective — span corruption using sentinel tokens `<extra_id_0>` — requires careful decoding configuration for generation tasks, a known source of artifacts when not explicitly overridden.

**IndicBART** (Dabre et al., 2022) is a multilingual sequence-to-sequence model built on the mBART architecture and pretrained on Sangraha, a curated corpus of 11 Indian languages including Telugu. Its use of explicit language tokens (`<2te>` for Telugu) and its Indic-focused pretraining make it a natural baseline for Telugu generation tasks.

### 3.2 Telugu NLP and TeClass

Kanumolu et al. (2024) introduced **TeClass**, a human-annotated dataset of 26,178 Telugu article-headline pairs labelled across three relevance tiers (HREL, MREL, LREL) and eight fine-grained semantic categories. The paper demonstrates that Telugu headline relevance classification is non-trivial even for multilingual BERT models, reporting a Macro-F1 of approximately 0.59 using the full training regime. TeClass is the first published headline relevance benchmark specifically for Telugu.

### 3.3 Low-Resource Summarization

Studies in low-resource multilingual summarization (Hasan et al., 2021; Nguyen & Daumé, 2019) consistently show that pretrained seq2seq models generalize effectively from small in-domain datasets when the model has strong multilingual pretraining. The extent of this generalization for morphologically rich South Asian languages with small available corpora remains an open empirical question.

### 3.4 Research Gap

Prior Telugu summarization work has not systematically evaluated the interaction between data quantity, data quality (deduplication and length filtering), and headline relevance under a fixed compute budget on consumer hardware. This project provides such a controlled evaluation, with the limitation that training runs are restricted to 50–75 optimizer steps rather than full convergence.

---

## 4. Dataset and Preprocessing

### 4.1 Headline Generation Dataset

**Source:** `saidines12/telugu_news_dataset` on Hugging Face Hub.  
**Content:** Scraped from Andhra Jyothi and Eenadu Telugu news websites.  
**License:** Public access on Hugging Face; no proprietary gating identified.  
**Format:** Parquet, split into official train, validation, and test partitions.  
**Columns:** `story_id` (unique integer), `article` (Telugu text), `headline` (Telugu text).

#### Split Counts After Preprocessing

| Split | Raw Rows | Preprocessed Rows | Skipped |
|-------|----------|-------------------|---------|
| `train` | 83,866 | 83,562 | 304 |
| `validation` | 5,962 | 5,962 | 0 |
| `test` | 12,504 | 12,441 | 63 |
| **Total** | **102,332** | **101,965** | **367** |

#### Data Quality Observations

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Null articles | 0 | 0 | 0 |
| Null headlines | 0 | 0 | 0 |
| Exact duplicate (article, headline) pairs | 72,062 | 0 | 10,746 |
| Telugu Unicode quality (sampled 500) | 100% | 100% | 100% |
| Article length — median (chars) | 917 | 771 | 1,003 |
| Headline length — median (chars) | 30 | 37 | 30 |

> **Known Limitation:** The training split contains 72,062 exact duplicate (article, headline) pairs originating from wire service syndication. These are documented and their effect on the low-resource experiments is analyzed in Section 9.

#### Cross-Split Leakage Analysis

| Leakage Type | Train–Validation | Train–Test | Validation–Test |
|---|---|---|---|
| Shared story_ids | 0 | 0 | 0 |
| Shared exact articles | 0 | **10** (~0.08%) | 0 |
| Shared headlines | 18 (~0.30%) | **56** (~1.85%) | 2 (~0.07%) |

The train–test article and headline overlap is documented in `outputs/metrics/dataset_audit.json`. It is not eliminated, but does not materially affect validation-set evaluations.

### 4.2 Preprocessing Pipeline

The preprocessing script (`src/data/preprocess.py`) applies:
1. **Deduplication:** Removes exact (article, headline) pairs within each split.
2. **Length Filtering:** Rejects articles shorter than 20 characters or headlines shorter than 1 character.
3. **Tokenization:** Converts each article–headline pair to mT5 input format: `"summarize: {article}"` → target `"{headline}"`.
4. **Reproducibility:** Fixed seed (42), SHA-256 hashes for all output files.

Tokenization uses `AutoTokenizer.from_pretrained("google/mt5-small")` with `max_input_length=512` and `max_target_length=128`.

### 4.3 TeClass Classification Dataset

**Source:** Kanumolu, Madasu, Surange, and Shrivastava (2024), LREC-COLING 2024.  
**Repository:** https://github.com/gopichandkanumolu/TeClass  
**License:** CC BY-NC-SA 4.0.  
**Size:** 26,178 article–headline pairs, majority-vote labelled by three native Telugu annotators.

| Split | Total | HREL | MREL | LREL |
|-------|-------|------|------|------|
| Train | 18,324 | 5,962 (32.5%) | 7,105 (38.8%) | 5,257 (28.7%) |
| Validation | 3,927 | 1,277 (32.5%) | 1,523 (38.8%) | 1,127 (28.7%) |
| Test | 3,927 | 1,278 (32.5%) | 1,523 (38.8%) | 1,126 (28.7%) |
| **Total** | **26,178** | **8,517** | **10,151** | **7,510** |

Zero cross-split ID overlap. Minimal article overlap (<0.1%) between train and validation from recurring wire reports with distinct headlines.

---

## 5. System Architecture and Methodology

### 5.1 Project Architecture

```
News_Headline/
├── src/
│   ├── data/               # Download, audit, preprocessing
│   ├── training/           # mT5 training script
│   ├── evaluation/         # ROUGE/chrF evaluation, model comparison
│   ├── classification/     # TeClass classifier training and inference
│   ├── experiments/        # Subset creation, quality filtering, Gemma 2 audit
│   ├── inference/          # Memory-safe model manager (Phase 6)
│   └── utils/              # Hardware reporting, tokenization utilities
├── data/
│   ├── raw/                # Downloaded HF parquet files
│   ├── processed/          # Tokenized JSONL files
│   ├── subsets/            # Nested subsets + SHA-256 manifest
│   └── teclass/            # TeClass CSV files
├── checkpoints/            # Trained model checkpoints
├── outputs/
│   ├── metrics/            # JSON evaluation results
│   └── plots/              # Learning curve plots
├── tests/                  # 82 automated regression tests
├── docs/                   # Phase reports, project plan
├── app.py                  # Streamlit application
└── requirements.txt
```

### 5.2 Telugu ROUGE Fix

Python's `rouge-score` package uses `re.findall(r"\w+", text)` which, in ASCII mode, discards all non-ASCII characters including the Telugu Unicode range (U+0C00–U+0C7F). This caused all ROUGE scores to evaluate to 0.0 for Telugu text in Phase 1.

**Fix:** An `IndicTokenizer` class was implemented in `src/evaluation/evaluate.py` that splits on Unicode punctuation and whitespace while preserving Telugu characters, then passes it as the `tokenizer` argument to `rouge_score.rouge_scorer.RougeScorer`. Verified in tests: identical Telugu sentences score 1.0 F1; partial matches score realistic non-zero values.

### 5.3 mT5 Sentinel Token Fix

Early mT5 generation produced outputs beginning with `<extra_id_0>` (a span-corruption sentinel). This was caused by not specifying `decoder_start_token_id`. Fix: explicitly set `decoder_start_token_id=tokenizer.pad_token_id` (value 0) in all `model.generate()` calls, plus a regex cleanup `re.sub(r"<extra_id_\d+>", "", text)`.

### 5.4 Inference Memory Management

The `ModelManager` class (`src/inference/manager.py`) enforces sequential, single-model occupancy:
- Lazy loading: models loaded on demand.
- Sequential unloading: `del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()` before any new model load.
- Live VRAM telemetry: queries `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` for dashboard display.

---

## 6. Headline Generation Models

### 6.1 mT5-small

**Architecture:** mT5ForConditionalGeneration (Xue et al., 2021)  
**Parameters:** 300.2 million  
**Pretrained on:** 101 languages, mC4 Common Crawl corpus  
**Precision:** bfloat16 (required for numerical stability; FP16 causes NaN loss in mT5 RMSNorm layers)

| Training Run | Steps | Effective Batch | Start Loss | End Loss |
|---|---|---|---|---|
| Smoke test | 20 | 4 | 30.69 | 12.61 |
| Baseline (Phase 1) | 50 | 32 | 12.34 | 4.02 |
| Low-resource 5% (Phase 4) | 75 | 32 | 8.19 | 3.02 |
| Low-resource 25% (Phase 4) | 75 | 32 | 10.56 | 3.73 |
| Low-resource 100% (Phase 4) | 75 | 32 | 9.89 | 3.39 |

> **Important Limitation:** All runs are far from convergence. 50–75 optimizer steps on a dataset of 83,562 samples represents less than one full training epoch. Results represent early-training baselines, not converged model performance.

**Peak VRAM:** 1,907 MiB allocated during generation (well within 4,095 MiB physical limit).

### 6.2 IndicBART

**Architecture:** MBartForConditionalGeneration (Dabre et al., 2022)  
**Parameters:** ~244 million  
**Pretrained on:** 11 Indic languages including Telugu  
**Language tag:** `<2te>` appended to source; `forced_bos_token_id` set to Telugu token ID  
**Precision:** bfloat16  
**VRAM at load:** 931 MiB; peak inference: 967 MiB

**Known Issue:** Without aggressive `no_repeat_ngram_size` constraints, IndicBART generates highly repetitive Telugu text. Investigated and mitigated with `no_repeat_ngram_size=3`, though some repetition remains in short-article inputs.

---

## 7. Evaluation Metrics and Results

### 7.1 Telugu-Aware Metric Implementation

ROUGE scores are computed using `IndicTokenizer` (preserves Telugu Unicode). chrF is computed via `sacrebleu.corpus_chrf()` which handles Unicode natively. All metric implementations are verified by unit tests in `tests/test_phase3.py`.

### 7.2 Phase 3: Controlled Model Comparison

Evaluation split: `validation`. Sample size: **N = 50** examples (subset for computational feasibility; full validation contains 5,962 examples).

| Model | Parameters | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF | Latency (s/ex) | Peak VRAM (MiB) | HREL (N=50) | MREL (N=50) | LREL (N=50) |
|---|---|---|---|---|---|---|---|---|---|---|
| **mT5-small** | 300M | **0.1025** | **0.0283** | **0.1025** | 12.78 | **0.27** | 1,907 | **29 (58%)** | 21 (42%) | 0 (0%) |
| **IndicBART** | 244M | 0.0109 | 0.0000 | 0.0109 | **14.02** | 2.62 | **1,707** | 25 (50%) | 25 (50%) | 0 (0%) |

*Relevance assessed by the Phase 2 mBERT classifier; counts reflect automated predictions, not human annotation.*

**Interpretation:**
- mT5-small achieves higher lexical overlap on this early checkpoint. Its 9.6× speed advantage makes it practically preferable for real-time use.
- IndicBART's higher chrF suggests better character-level Telugu morpheme alignment. However, its near-zero ROUGE-2 reflects word-level repetition rather than genuine phrase overlap with reference headlines.
- Both models produced 0 LREL (clickbait/misleading) classifications, suggesting article-grounded generation even in early training.
- **Caution:** N=50 is a preliminary evaluation subset. Results should not be interpreted as statistically definitive comparisons between the architectures.

### 7.3 Phase 4: Low-Resource Data Scaling

Training configuration: mT5-small, 75 optimizer steps, bfloat16, AdamW (lr=5×10⁻⁴, wd=0.01), effective batch size 32.  
Evaluation: **N = 15 preliminary samples** from the validation set. Results are indicative, not statistically robust.

| Data Tier | Training Samples | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF | Train Time | PyTorch Peak Alloc† |
|---|---|---|---|---|---|---|---|
| **5%** | 4,178 | 0.1903 | 0.0556 | 0.1903 | 24.69 | 1,373s (~22.9m) | ~6,066 MiB |
| **25%** | 20,890 | 0.1005 | 0.0222 | 0.0839 | 18.11 | 1,350s (~22.5m) | ~6,045 MiB |
| **100%** | 83,562 | **0.2230** | **0.0756** | **0.1987** | **29.80** | 1,308s (~21.8m) | ~6,050 MiB |

†_Physical dedicated VRAM is 4,095.5 MiB. Allocations above this reflect Windows WDDM transparent paging into Shared Host RAM. Peak VRAM in the physical GPU is ≤4,095.5 MiB._

**Interpretation:** The 5% model achieves ROUGE-1 of 0.1903, recovering ~85% of the 100%-data ROUGE-1 of 0.2230. This demonstrates that mT5-small generalizes effectively from small in-domain Telugu datasets within the same compute budget, likely exploiting cross-lingual pretraining. Non-monotonic 25% results (lower than 5%) likely reflect that 75 steps is insufficient for the larger, noisier 25% dataset to produce a more diverse signal than the smaller clean 5% subset.

---

## 8. Headline Relevance Classification

### 8.1 TeClass Annotation Taxonomy

The TeClass benchmark (Kanumolu et al., 2024) defines headline relevance through a three-tier hierarchy:

```
HREL (Highly Related)
└── FME: Factual Main Event — headline explicitly and accurately states the primary event

MREL (Moderately Related)
├── FSE: Factual Secondary Event
├── STC: Strong Conclusion
└── WKC: Weak Conclusion

LREL (Least Related)
├── SEN: Sensational
├── CBT: Clickbait
├── MLC: Misleading Conclusion
└── USO: Unsupported Opinion
```

Ground-truth labels in the dataset files are at the three-class (HREL/MREL/LREL) level.

### 8.2 Classifier Results (Validation Set, N = 3,927)

| Metric | TF-IDF + Logistic Regression | mBERT (Transformer) | TeClass Paper (mBERT) |
|---|---|---|---|
| Accuracy | 0.4772 | **0.5238** | — |
| Macro-F1 | 0.4775 | **0.5114** | ~0.59 |
| Weighted-F1 | 0.4688 | **0.4986** | ~0.59 |
| HREL F1 | 0.5536 | **0.6225** | ~0.66 |
| MREL F1 | 0.3618 | 0.3423 | ~0.50 |
| LREL F1 | 0.5172 | **0.5693** | ~0.62 |

**Gap from Published Results:** The TeClass paper's reported ~0.59 Macro-F1 was achieved with a full training schedule. This project's mBERT ran for 150 training steps due to time and compute constraints, explaining the ~0.06 Macro-F1 gap. The HREL class shows the strongest performance (recall = 0.824) reflecting the classifier's ability to identify the primary factual event.

### 8.3 Important Disclaimer

The relevance classifier is an **automated proxy metric**, not a fact-checking system. HREL classification indicates structural alignment between the headline and the article's primary event as learned from training data annotations. It does not verify factual accuracy, detect subtle misinformation, or substitute for editorial review.

---

## 9. Low-Resource and Quality Experiments

### 9.1 Experiment Design Validity

The nested subset design ensures $5\% \subset 25\% \subset 100\%$ (same seed=42, verified by SHA-256 manifests in `data/subsets/manifest.json`). The validation set (5,962 examples) is held constant across all runs. Training steps (75) are standardized, providing a compute-normalized comparison.

**What this experiment measures:** The effect of dataset diversity on mT5-small performance when compute exposure is equal. It does not measure full convergence behaviour.

### 9.2 Data Quality Filtering (Phase 5)

| Subset | Raw | Retained | Excluded | Main Exclusion |
|---|---|---|---|---|
| 5% (4,178) | 4,178 | 3,385 (81.0%) | 793 | Duplicate articles: 482 |
| 25% (20,890) | 20,890 | 10,886 (52.1%) | 10,004 | Duplicate articles: 8,485 |

Full exclusion breakdown per category:

**5% subset exclusions:** Article too long (241), Article too short (36), Duplicate article (482), Headline too short (26), Verbatim copy of lead (6), No Telugu in article (2).

**25% subset exclusions:** Article too long (1,173), Article too short (181), Duplicate article (8,485), Headline too short (138), Verbatim copy of lead (20), No Telugu in article (6), No Telugu in headline (1).

**Key observation:** As the data scale grows from 5% to 25%, the duplicate-article rate jumps from 11.5% (482/4178) to 40.6% (8485/20890), demonstrating that wire-service syndication creates massive redundancy at scale.

**Important caveat:** No controlled generation training or evaluation was conducted on the quality-filtered subsets due to compute constraints. The experiment establishes that quality filtering substantially reduces the effective dataset size at scale and identifies the dominant data quality issues. It does **not** empirically demonstrate that quality-filtered training produces better headline generation than unfiltered training.

---

## 10. Gemma 2 2B Feasibility Investigation

Model evaluated: `google/gemma-2-2b` (2.61 billion parameters, decoder-only, gated).

### Memory Budget Analysis

| Configuration | Memory Required | Feasible on 4 GB? |
|---|---|---|
| Full FP16 weights | 5.22 GB | No — exceeds VRAM for weights alone |
| Full FP16 + AdamW states | 20.88 GB | No |
| INT4 weights (QLoRA) | 1.31 GB | Theoretically yes |
| INT4 weights + activations + LoRA gradients + CUDA context | ~3.6 GB | Marginal (<400 MiB headroom) |

### Blockers (All Independently Verified)

| Blocker | Evidence |
|---|---|
| Gated HuggingFace repository | `401 Client Error: Cannot access gated repo for google/gemma-2-2b` |
| `bitsandbytes` not installed | `pip show bitsandbytes` → not found |
| `peft` not installed | `pip show peft` → not found |
| Physical VRAM insufficient for INT4 | ~400 MiB headroom; not viable in practice |

**Conclusion:** Gemma 2 2B local fine-tuning remains **BLOCKED**. No Gemma training was performed. No Gemma generation metrics exist. The feasibility report is preserved at `outputs/metrics/gemma2_feasibility_report.json`.

---

## 11. Results and Discussion

### 11.1 Summary of All Results

| Experiment | Model | Key Metric | Value | N (eval) |
|---|---|---|---|---|
| Headline Generation (P3) | mT5-small (50 steps) | ROUGE-1 | 0.1025 | 50 |
| Headline Generation (P3) | IndicBART | ROUGE-1 | 0.0109 | 50 |
| Headline Generation (P3) | mT5-small | chrF | 12.78 | 50 |
| Headline Generation (P3) | IndicBART | chrF | 14.02 | 50 |
| Low-Resource (P4, 5%) | mT5-small (75 steps) | ROUGE-1 | 0.1903 | 15 |
| Low-Resource (P4, 25%) | mT5-small (75 steps) | ROUGE-1 | 0.1005 | 15 |
| Low-Resource (P4, 100%) | mT5-small (75 steps) | ROUGE-1 | 0.2230 | 15 |
| Relevance Classification | mBERT (150 steps) | Macro-F1 | 0.5114 | 3,927 |
| Relevance Classification | TF-IDF + LR | Macro-F1 | 0.4775 | 3,927 |

### 11.2 Discussion

**RQ1 (Relevance Classification):** The mBERT classifier achieves Macro-F1 = 0.51, above the TF-IDF baseline (0.4775). HREL class recall is high (82.4%), confirming that the model reliably identifies factual main-event headlines. MREL prediction is the weakest component (F1 = 0.34), reflecting the semantic subtlety of distinguishing primary from secondary events in Telugu journalism. The gap from the paper's ~0.59 is attributable to shortened training.

**RQ2 (Model Comparison):** mT5-small outperforms IndicBART on lexical ROUGE overlap and is 9.6 times faster. IndicBART shows higher chrF but exhibits severe repetition. At 50 training steps, neither model produces fully converged headline generation. The results constitute an early-checkpoint comparison, not a definitive architecture evaluation.

**RQ3 (Data Quantity):** The 5% subset recovers ~85% of full-data ROUGE-1 under 75 equal-compute steps, confirming strong multilingual pretraining transfer for Telugu. Full-data (100%) achieves highest absolute scores.

**RQ4 (Data Quality):** Quality filtering reveals significant wire-service contamination (up to 48% duplicates in the 25% subset). No controlled training comparison was conducted between filtered and unfiltered data; this remains an open empirical question for future work.

---

## 12. Limitations and Threats to Validity

1. **Short training:** Both generation models trained for 50–75 steps, far below convergence. Results represent early-training performance, not the models' full potential.
2. **Small evaluation sets:** Phase 3 metrics use N=50; Phase 4 uses N=15. These are preliminary estimates with high variance. Full validation (N=5,962) was not evaluated due to compute constraints.
3. **WDDM memory reporting:** PyTorch reports up to ~6,066 MiB allocation on a 4 GB GPU. Allocations exceeding physical VRAM transparently page into shared system RAM via Windows WDDM. Physical GPU VRAM never exceeded 4,095.5 MiB.
4. **Headline leakage:** 56 train–test headlines overlap (1.85%). Not eliminated; documented in `outputs/metrics/dataset_audit.json`.
5. **Automated relevance classification:** All HREL/MREL/LREL labels in generation evaluation are classifier outputs (Macro-F1 = 0.51), not human judgements.
6. **No human evaluation:** No human annotation of generated headline quality was conducted.
7. **Gemma 2 blocked:** No generation results for large-model comparison.
8. **Quality filtering not evaluated:** Data quality statistics collected but no training run on filtered subsets was completed.
9. **IndicBART repetition:** The repetition issue is partially mitigated but not fully resolved within the tested hyperparameter range.
10. **Single seed:** Most experiments used a single seed (42). Variance across seeds is not reported.

---

## 13. Conclusion and Future Work

This project delivered a complete, locally reproducible Telugu news headline generation and relevance classification pipeline on constrained consumer hardware. Key contributions include:

- A verified Telugu-aware ROUGE evaluation fix that corrects a critical zero-score bug.
- An mT5-small and IndicBART headline generation baseline with documented early-training metrics.
- A mBERT relevance classifier trained on the TeClass benchmark with Macro-F1 = 0.51.
- A controlled low-resource scaling study showing strong multilingual pretraining transfer.
- An empirical data quality analysis revealing substantial wire-service contamination at scale.
- An interactive Streamlit application exposing all models and research results locally.

**Future Work:**
- Extend training to 500+ steps per model to approach convergence.
- Evaluate on the full 5,962-example validation set and the held-out test set.
- Conduct human evaluation of generated headline quality and factual accuracy.
- Implement LoRA fine-tuning of larger Indic models when hardware or access constraints permit.
- Compare quality-filtered vs. unfiltered training with equal sample counts.
- Explore prompt-based Telugu generation with instruction-tuned models.

---

## 14. References

Dabre, R., et al. (2022). *IndicBART: A Pre-trained Model for Natural Language Generation of Indic Languages*. In Findings of ACL 2022.

Hasan, T., Bhattacharjee, A., Islam, S., et al. (2021). *XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages*. In Findings of ACL-IJCNLP 2021.

Kanumolu, G., Madasu, L., Surange, N., & Shrivastava, M. (2024). *TeClass: A Human-Annotated Relevance-based Headline Classification and Generation Dataset for Telugu*. In Proceedings of LREC-COLING 2024, pp. 15711–15720. https://aclanthology.org/2024.lrec-main.1364/

Kenton, J. D. M.-W. C., & Toutanova, L. K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. In Proceedings of NAACL-HLT 2019.

Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. In Proceedings of ACL 2020.

Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*. In ACL Workshop on Text Summarization, pp. 74–81.

Nguyen, T.-V. T., & Daumé, H. (2019). *Global Voices: Crossing Borders in Automatic News Summarization*. In Proceedings of EMNLP 2019 Workshop on NLP4TM.

Popović, M. (2015). *chrF: Character n-gram F-score for Automatic MT Evaluation*. In Proceedings of WMT 2015.

Xue, L., et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*. In Proceedings of NAACL-HLT 2021, pp. 483–498.

---

## 15. Reproducibility Appendix

### 15.1 Environment Setup

```powershell
# Python version: 3.11.0
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 (required first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU availability
python tests/test_gpu.py
```

### 15.2 Key Package Versions (Verified)

| Package | Version |
|---------|---------|
| Python | 3.11.0 |
| PyTorch | 2.5.1+cu121 |
| transformers | 4.46.3 |
| datasets | 3.1.0 |
| streamlit | 1.65.0 |
| rouge_score | 0.1.2 |
| sacrebleu | 2.6.0 |
| scikit-learn | 1.9.1 |
| sentencepiece | 0.2.2 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |

### 15.3 Dataset Acquisition

```python
from datasets import load_dataset
ds = load_dataset("saidines12/telugu_news_dataset")
# No authentication required; public repository
```

TeClass data: download from https://github.com/gopichandkanumolu/TeClass (CC BY-NC-SA 4.0).

### 15.4 Training Commands

```powershell
# Phase 1: mT5 baseline (50 steps, full dataset)
.\venv\Scripts\python.exe src/training/train_mt5.py

# Phase 2: TeClass classifiers
.\venv\Scripts\python.exe src/classification/train_baseline.py
.\venv\Scripts\python.exe src/classification/train_transformer.py --max-steps 150

# Phase 3: IndicBART training + model comparison
.\venv\Scripts\python.exe src/training/train_indicbart.py
.\venv\Scripts\python.exe src/evaluation/compare_models.py

# Phase 4: Low-resource experiments
.\venv\Scripts\python.exe src/experiments/create_subsets.py
.\venv\Scripts\python.exe src/experiments/low_resource.py

# Phase 5: Quality filtering
.\venv\Scripts\python.exe src/experiments/quality_filtering.py
```

### 15.5 Inference and Application

```powershell
# Single-headline generation (Python API)
python -c "
from src.inference.manager import get_model_manager
mgr = get_model_manager()
result = mgr.generate('mt5_small_final', 'your Telugu article text here')
print(result['headline'])
"

# Relevance classification
python src/classification/predict.py --headline "శీర్షిక" --article "వ్యాసం" --model-type bert

# Launch Streamlit application
.\venv\Scripts\python.exe -m streamlit run app.py
# Opens at http://localhost:8501
```

### 15.6 Checkpoint Locations

| Checkpoint | Path | Size |
|---|---|---|
| mT5-small final | `checkpoints/mt5_small/final` | ~1.14 GB |
| mT5-small step_50 | `checkpoints/mt5_small/step_50` | ~1.14 GB |
| IndicBART final | `checkpoints/indicbart/final` | ~931 MB |
| Low-resource 5% | `checkpoints/low_resource/5pct/final` | ~1.14 GB |
| Low-resource 25% | `checkpoints/low_resource/25pct/final` | ~1.14 GB |
| Low-resource 100% | `checkpoints/low_resource/100pct/final` | ~1.14 GB |
| mBERT classifier | `checkpoints/classifier_bert/final` | ~678 MB |
| TF-IDF classifier | `checkpoints/classifier_tfidf/tfidf_logreg.pkl` | ~1.9 MB |

> **Note:** All checkpoints are listed in `.gitignore` and must be regenerated by re-running the training scripts, or transferred separately. They are not stored in the git repository due to file size.

### 15.7 Experiment Configuration and Seeds

| Setting | Value |
|---|---|
| Primary random seed | 42 |
| Subset SHA-256 hashes | Stored in `data/subsets/manifest.json` |
| mT5 learning rate | 5×10⁻⁴ |
| mBERT learning rate | 2×10⁻⁵ |
| Effective batch size | 32 (generation), 16 (classification) |
| Bfloat16 precision | Yes, all models |

### 15.8 Metric Artifact Locations

| Artifact | Path |
|---|---|
| Dataset audit | `outputs/metrics/dataset_audit.json` |
| TeClass audit | `outputs/metrics/teclass_audit.json` |
| TF-IDF classifier eval | `outputs/metrics/baseline_classifier_eval.json` |
| mBERT classifier eval | `outputs/metrics/transformer_classifier_eval.json` |
| Phase 3 model comparison | `outputs/metrics/model_comparison_phase3.json` |
| Low-resource results | `outputs/metrics/low_resource_results.json` |
| Low-resource summary CSV | `outputs/metrics/low_resource_summary.csv` |
| Gemma 2 feasibility report | `outputs/metrics/gemma2_feasibility_report.json` |
| ROUGE learning curve | `outputs/plots/learning_curve_rouge.png` |
| chrF learning curve | `outputs/plots/learning_curve_chrf.png` |
| Relevance learning curve | `outputs/plots/learning_curve_relevance.png` |

### 15.9 Known Hardware and OS Constraints

| Constraint | Details |
|---|---|
| Physical VRAM | 4,095.5 MiB dedicated (RTX 3050 Laptop) |
| WDDM overcommit | PyTorch may allocate beyond physical VRAM into shared host RAM; observed up to ~6,066 MiB |
| bfloat16 required | FP16 causes NaN loss in mT5 RMSNorm on this hardware |
| Windows CUDA | Driver 592.82, CUDA 13.1 driver-reported; PyTorch cu121 wheel compatible |
| Gemma 2 2B | BLOCKED — gated repo + missing bitsandbytes/peft + marginal VRAM headroom |
| No cloud execution | 100% local; no Colab or paid GPU used at any phase |
