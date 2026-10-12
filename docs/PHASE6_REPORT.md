# Phase 6 Implementation & Verification Report: Streamlit Application

## 1. Overview & Objectives
Phase 6 establishes a full-featured, interactive, local **Streamlit Application** (`Telugu News Headline Studio` via `app.py`) for the Telugu News Headline Generation project. 
The system integrates all checkpoints, generation pipelines, relevance classification models, and experimental analytics produced across Phases 0 through 5 into a cohesive UI.

**Hardware & Runtime Environment:**
- **Host GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (Dedicated physical VRAM: 4,095.5 MiB).
- **Host RAM:** 15.34 GB total (6.2 GB available).
- **Execution Mode:** Strictly local, Windows 11. No cloud APIs, no external compute.
- **Python:** 3.11.0 in `venv`.
- **Streamlit:** 1.65.0.

---

## 2. Key Architecture & Memory Safety Safeguards

### 2.1 The Inference Manager (`src/inference/manager.py`)
To prevent GPU Out-of-Memory (OOM) crashes under Windows WDDM on the 4 GB VRAM RTX 3050:
- **Sequential Lazy Loading:** Only one large transformer model is loaded into CUDA memory at any given time.
- **Explicit VRAM Garbage Collection:** Switching models triggers `del self.active_gen_model`, followed by `gc.collect()` and `torch.cuda.empty_cache()`.
- **Real-Time Memory Telemetry:** The UI sidebar actively queries `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`, presenting real-time VRAM allocation and a one-click cache purge button.

### 2.2 Model Registry & Checkpoints
The application dynamically discovers and loads all verified checkpoints from disk:
1. `mT5-small (Phase 1/3 Final)` (`checkpoints/mt5_small/final`)
2. `mT5-small (Step 50 Intermediate)` (`checkpoints/mt5_small/step_50`)
3. `IndicBART (Phase 3 Final)` (`checkpoints/indicbart/final`)
4. `mT5-small (5% Data - 75 Steps)` (`checkpoints/low_resource/5pct/final`)
5. `mT5-small (25% Data - 75 Steps)` (`checkpoints/low_resource/25pct/final`)
6. `mT5-small (100% Data - 75 Steps)` (`checkpoints/low_resource/100pct/final`)
7. `mBERT Relevance Classifier (Phase 2)` (`checkpoints/classifier_bert/final`)
8. `TF-IDF Relevance Classifier (Phase 2)` (`checkpoints/classifier_tfidf/tfidf_logreg.pkl`)

---

## 3. Application Pages & Features

The application provides a multi-page interface via `app.py`:

### Page 1: Headline Generation
- **Input:** Article text area supporting Telugu Unicode fonts (`Mandali`, `Gautami`, `Nirmala UI`), character and word counters, and preset real news articles from `data/processed/validation.jsonl`.
- **Controls:** Model checkpoint selection, beam search width (`num_beams` 1–8), max sequence length, and sampling temperature.
- **Outputs:** High-prominence headline display, latency metric in seconds, word length count, copy-to-clipboard field, and structured JSON prediction download.

### Page 2: Relevance Analysis
- **Classifier Selection:** Choice between fine-tuned cross-attention `mBERT` (Macro-F1 ~0.51) and fast linear baseline `TF-IDF + Logistic Regression`.
- **Assessment:** Categorizes input pair into TeClass taxonomy (`HREL`, `MREL`, `LREL`) with confidence probabilities and bar chart visualization.
- **Plain-Language Definitions:** Explicitly defines `HREL` (FME — Factual Main Event), `MREL` (Secondary/Conclusion), and `LREL` (Clickbait/Misleading), accompanied by clear editorial disclaimers.

### Page 3: Sequential Model Comparison
- **Head-to-Head Evaluation:** Enables side-by-side generation for any two models on the same article.
- **VRAM Safety:** Executes Model A, unloads Model A, runs Model B, unloads Model B, and evaluates relevance of both generated outputs.
- **Comparative Metrics:** Side-by-side headline text, generation latency, word counts, and relevance confidence.

### Page 4: Research Dashboard
- **Authentic Metrics:** Direct visualization of repository experiment outputs:
  - Phase 3 mT5-small vs IndicBART comparison table ($N=50$ samples).
  - Phase 4 Low-Resource data scaling table ($5\%$, $25\%$, $100\%$ subsets).
  - High-resolution learning curve trajectories (`learning_curve_rouge.png`, `learning_curve_chrf.png`).
  - Gated access, quantization dependencies, and parameter budget report for Gemma 2 2B (`gemma2_feasibility_report.json`).

### Page 5: About & Architecture
- Details the system design, hardware limits, Telugu NLP preprocessing, TeClass relevance taxonomy, and research limitations.

---

## 4. Verification & Testing

### 4.1 Automated Test Suite (`tests/test_phase6.py`)
A dedicated automated test suite was constructed and verified:
- `test_model_registry_integrity`: Confirms all registered generator and classifier checkpoints exist on disk.
- `test_manager_input_validation`: Ensures whitespace/empty input raises `ValueError`.
- `test_manager_unknown_model_raises`: Rejects non-existent model keys.
- `test_gpu_memory_info_format`: Validates GPU VRAM dictionary schema.
- `test_app_imports_cleanly`: Confirms `app.py` and preset validation samples load correctly.
- `test_dashboard_metric_assets_exist`: Confirms all plots and JSON metrics files exist and contain valid data.

### 4.2 Full Regression Suite
Running the full test suite across all project phases (Phases 0 through 6):
```bash
.\venv\Scripts\python.exe -m pytest tests/
```
**Result:** **82 passed in 33.69s** (100% pass rate).

### 4.3 Server Smoke Test
The Streamlit server was launched headlessly and verified via HTTP probe:
```
http://localhost:8501/_stcore/health -> 200 OK
```

---

## 5. How to Run Locally

From the project root:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.
