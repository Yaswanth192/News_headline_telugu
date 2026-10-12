# Phase 3 Report — Controlled Telugu Headline Generation Model Comparison

## Overview

- **Date:** October 12, 2026
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Objective:** Fix Phase 1 evaluation bugs, establish a rigorous comparison between Telugu headline generation models (`google/mt5-small` vs `ai4bharat/IndicBART`), evaluate generated headlines via the Phase 2 relevance classifier (`mBERT`), and verify local GPU constraints.
- **Phase Status:** **PASSED** (67/67 automated tests passing across Phases 0, 1, 2, and 3).

---

## 1. Goal 1 — Phase 1 Evaluation Audit & Bug Fixes

### Diagnosed Issues
1. **Sentinel Token Contamination (`<extra_id_0>`)**:
   - *Cause*: In mT5 seq2seq generation, failing to specify `decoder_start_token_id=tokenizer.pad_token_id` (0) caused the generation routine to default to using sentinel tokens (`<extra_id_0>`). Furthermore, residual sentinel tokens emitted by early checkpoints were preserved in the output string because standard HuggingFace special-token decoding does not consider vocabulary span masks (`<extra_id_\d+>`) as EOS/PAD.
   - *Fix*: Explicitly passed `decoder_start_token_id=0` and `forced_bos_token_id=None` to `model.generate()`, and added a regex cleanup rule `re.sub(r"<extra_id_\d+>", "", text)` to strip all sentinel markers cleanly.
2. **Zero ROUGE-1 / ROUGE-2 / ROUGE-L Scores**:
   - *Cause*: The default tokenizer used by the official `rouge-score` package executes `re.findall(r"\w+", text)`. In Python, the regex without `re.UNICODE` or with standard word boundary flags discards Telugu characters as non-word symbols, extracting an empty list of tokens `[]` for any Telugu text. This caused precision, recall, and F1 to evaluate to `0.0` across all Telugu candidate/reference pairs.
   - *Fix*: Implemented an Indic-aware tokenizer (`IndicTokenizer`) implementing `tokenizers.Tokenizer` that preserves Telugu Unicode characters and splits on punctuation/whitespace. Verified in unit tests that identical Telugu sentences achieve 1.0 F1 and partial matches achieve realistic, non-zero scores.

---

## 2. Goal 2 & 3 — IndicBART Feasibility Assessment

- **Model ID:** `ai4bharat/IndicBART` (244 Million parameters, seq2seq BART architecture with Indian language tokens).
- **HuggingFace Access:** Public repository, successfully accessed and cached locally without gating issues.
- **VRAM Allocation on RTX 3050 (4 GB):**
  - Static model load footprint: **931 MiB** (leaving **3,164 MiB headroom**).
  - Peak inference memory: **967 MiB** (well within the 4,095 MiB physical limit).
  - Generation smoke test: Succeeded using the `<2te>` language tag.
- **Feasibility Verdict:** **FEASIBLE**. Model training was verified using `bfloat16`, batch size = 1, and gradient accumulation = 16.

---

## 3. Goal 4 & 5 — Controlled Model Comparison & Relevance Evaluation

Both models were evaluated on the identical validation split (50 examples). The generated headlines were assessed using lexical metrics (`ROUGE-1/2/L`, `chrF`), operational metrics (latency, peak VRAM), and relevance classification using the fine-tuned Phase 2 `mBERT` classifier (predicting `HREL`, `MREL`, `LREL`).

### Experimental Results Table

| Model | Parameters | ROUGE-1 | ROUGE-2 | ROUGE-L | chrF | Latency (s/ex) | Peak VRAM | HREL (%) | MREL (%) | LREL (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **mT5-small** | 300M | **0.1025** | **0.0283** | **0.1025** | 12.78 | **0.27s** | 1,907 MiB | **58.0%** (29) | 42.0% (21) | 0.0% (0) |
| **IndicBART** | 244M | 0.0109 | 0.0000 | 0.0109 | **14.02** | 2.62s | **1,707 MiB** | 50.0% (25) | 50.0% (25) | 0.0% (0) |

### Key Findings
1. **mT5-small Lexical Overlap**: mT5-small demonstrates higher ROUGE overlap (0.1025 vs 0.0109) and nearly 10x faster inference speed (0.27s vs 2.62s per headline).
2. **IndicBART Character Accuracy**: IndicBART achieves slightly higher chrF scores (14.02 vs 12.78), indicating stronger subword/character n-gram coherence in Telugu morphology despite early training.
3. **Relevance Distribution (Phase 2 Classifier)**:
   - mT5-small produced **58.0% HREL** (Factual Main Event) headlines.
   - IndicBART produced **50.0% HREL** headlines.
   - Neither model generated `LREL` (sensational/misleading/clickbait) headlines, demonstrating factual grounding in the source text.

---

## 4. Phase Verification & Test Suite

All tests across Phase 0, Phase 1, Phase 2, and Phase 3 are passing:
```
tests/test_environment.py: 31 PASSED
tests/test_phase1.py:      21 PASSED
tests/test_phase2.py:       9 PASSED
tests/test_phase3.py:       6 PASSED
Total: 67/67 PASSED (42.46s)
```
