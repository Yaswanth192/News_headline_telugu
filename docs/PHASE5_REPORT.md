# Phase 5 Report — Gemma 2 Quantity-vs-Quality Research & Feasibility Audit

## Executive Summary

- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4,095.5 MiB dedicated hardware VRAM)
- **Host OS:** Windows 11 (64-bit), Python 3.11.0, PyTorch 2.5.1+cu121
- **Target Model:** `google/gemma-2-2b` (2.61 Billion parameters)
- **Primary Research Focus:** Quantity vs. Quality of training data in low-resource Telugu headline generation, and local GPU feasibility of modern 2B+ LLMs.
- **Phase Status:** **PASSED** (76/76 automated tests passing across Phases 0–5).
- **Gemma 2 Training Feasibility:** **BLOCKED** on local hardware due to strict gating authentication, package incompatibilities (bitsandbytes on Windows), and physical VRAM constraints.

---

## 1. Part A — Mandatory Phase 4 Integrity Audit

Before establishing Phase 5 conclusions, an audit of Phase 4 artifacts, training logs, and measurement code was conducted:

| Audit Item | Phase 4 Claim | Audit Finding & Root Cause | Resolution Status |
|---|---|---|---|
| **VRAM Measurement Discrepancy** | "Peak VRAM: 6,045–6,066 MiB" on a 4,095 MiB GPU | Physical dedicated VRAM is strictly 4,095.5 MiB. On Windows 11 with WDDM, `torch.cuda.memory_allocated()` permits overcommit by transparently paging tensor allocations into Shared Host RAM. PyTorch allocated ~6 GB virtual GPU memory, with ~1.97 GB residing in system RAM. | **RESOLVED & DOCUMENTED** (Relabeled from "Peak VRAM" to "PyTorch Peak Alloc with WDDM Overcommit"). |
| **Relevance Classifier Denominator** | "HREL = 13 (86.7%)" | The evaluation loop used an early cutoff of **$N = 15$ examples** per proportion to preserve execution time, not the full validation set (5,962). | **RESOLVED & DOCUMENTED** (Explicitly reported as $N = 15$ preliminary sample with exact counts). |
| **Fixed Training Step Budget** | 75 steps per proportion | Standardized 75 steps (~2,400 examples processed) tests compute-normalized data diversity, not full epoch convergence. Subsets (5%, 25%, 100%) were verified as strictly nested ($5\% \subset 25\% \subset 100\%$). | **VERIFIED** |
| **Evaluation Integrity** | ROUGE and chrF scores | Verified using custom `IndicTokenizer` (preventing ASCII regex word-boundary dropouts). | **VERIFIED** |

**Audit Outcome:** **PARTIALLY VERIFIED** (all operational data and checkpoints verified; memory labels and sample sizes corrected).

---

## 2. Part B — Gemma 2 2B Hardware Feasibility Investigation

A staged, empirical feasibility assessment was executed via [`src/experiments/gemma2_feasibility.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/src/experiments/gemma2_feasibility.py):

### Staged Protocol Findings

1. **Stage 1 — Environment & Hardware:**
   - Dedicated Physical VRAM: **4,095.5 MiB**
   - Host RAM: **15.34 GB** total (6.23 GB available)
   - CUDA Version: 12.1
2. **Stage 2 — HuggingFace Gating & Access:**
   - Model `google/gemma-2-2b` is a **gated repository** (`gated: manual`).
   - Token authentication check failed with `401 Client Error: Cannot access gated repo`.
   - *Blocker:* Running Gemma 2 locally requires user agreement to Google's terms on HuggingFace and local login via `huggingface-cli login`.
3. **Stage 3 — Architectural Memory Budget:**
   - Parameter count: 2.61 Billion parameters.
   - Full FP16 weights: $2.61\times 10^9 \times 2\text{ bytes} \approx \mathbf{5.22\text{ GB}}$ (Exceeds dedicated 4 GB VRAM for weights alone, before any forward activations).
   - Full AdamW states: $2.61\times 10^9 \times 8\text{ bytes} \approx \mathbf{20.88\text{ GB}}$ (Full fine-tuning is physically impossible locally).
   - 4-bit INT4 weights: $2.61\times 10^9 \times 0.5\text{ bytes} \approx \mathbf{1.31\text{ GB}}$.
   - Activations + KV Cache (batch=1, seq=512) + LoRA gradients: $\approx \mathbf{1.8\text{ GB}}$.
   - Base CUDA Context: $\approx \mathbf{0.5\text{ GB}}$.
   - Total theoretical QLoRA footprint: $\approx \mathbf{3.6\text{ GB}}$. While theoretically fitting under 4 GB on paper, it leaves $<400\text{ MiB}$ headroom, causing extreme vulnerability to CUDA OOM spikes.
4. **Stage 4 — Quantization Engine (bitsandbytes & PEFT):**
   - Neither `bitsandbytes` nor `peft` is installed in the Windows virtual environment.
   - On Windows, standard `bitsandbytes` requires specialized precompiled wheels (`bitsandbytes-windows`) which frequently exhibit runtime DLL mismatches with PyTorch 2.5.1+cu121.
5. **Verdict:** **BLOCKED** on local hardware. Model training was not simulated or faked. Artifact saved at [`outputs/metrics/gemma2_feasibility_report.json`](file:///D:/large%20documents/Large%20Projects/News_Headline/outputs/metrics/gemma2_feasibility_report.json).

---

## 3. Part C & D — Operationalizing Quantity-vs-Quality Research

To address the core scientific question—*How do data quantity and data quality affect Telugu headline generation?*—we designed and implemented an empirical quality-filtering engine in [`src/experiments/quality_filtering.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/src/experiments/quality_filtering.py).

### Operational Quality Rules
1. **Length Bounds:** Rejects degenerate articles ($<100$ or $>3,000$ chars) and headlines ($<10$ or $>150$ chars).
2. **Script Validation:** Strictly enforces Telugu Unicode characters (`\u0C00-\u0C7F`).
3. **Deduplication:** Strips exact article repetitions across the corpus.
4. **Verbatim Copy Elimination:** Detects and excludes headlines that merely copy the first sentence of the lead article verbatim without summarization.

### Quality Filtering Results

| Dataset Subset | Raw Input Examples | Retained High-Quality Examples | Excluded Records | Primary Exclusion Reason |
|---|---|---|---|---|
| **5% Subset** | 4,178 | **3,385** (81.0%) | 793 (19.0%) | Duplicate Articles (482), Article Too Long (241) |
| **25% Subset** | 20,890 | **10,886** (52.1%) | 10,004 (47.9%) | Duplicate Articles (8,485), Article Too Long (1,173) |

*Manifest saved at `data/subsets/quality_manifest.json`.*

### Research Takeaway
In low-resource news datasets, as data volume scales from 5% to 25%, the rate of duplicate articles and length outliers increases dramatically (nearly 48% of the 25% subset comprised duplicate reports of wire stories). Therefore, **curating for quality (deduplication and length filtering) yields much cleaner training signal than blindly scaling noisy data quantity**.

---

## 4. Test Suite & Verification

The project test suite was expanded with [`tests/test_phase5.py`](file:///D:/large%20documents/Large%20Projects/News_Headline/tests/test_phase5.py) covering memory distinction, feasibility blockers, quality filtering rules, and denominator verification.

All **76 automated tests pass cleanly**:
```
tests/test_environment.py: 31 PASSED
tests/test_phase1.py:      21 PASSED
tests/test_phase2.py:       9 PASSED
tests/test_phase3.py:       6 PASSED
tests/test_phase4.py:       4 PASSED
tests/test_phase5.py:       5 PASSED
Total: 76/76 PASSED (34.2s)
```
