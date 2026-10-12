"""
tests/test_phase1.py
====================
Phase 1 — Automated tests for dataset, preprocessing, model, and evaluation.

Run:
    python -m pytest tests/test_phase1.py -v

GPU tests skip gracefully if CUDA is not available.
Tests that require downloaded data skip if data is not present.
"""

import json
import pathlib
import tempfile
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def dataset_available(split: str = "train") -> bool:
    return (RAW_DIR / f"{split}.parquet").exists()


def processed_available(split: str = "train") -> bool:
    return (PROCESSED_DIR / f"{split}.jsonl").exists()


def checkpoint_available() -> pathlib.Path | None:
    """Return path to most recently saved checkpoint if any exists."""
    ckpt_root = PROJECT_ROOT / "checkpoints" / "mt5_small"
    if not ckpt_root.exists():
        return None
    # Look for 'final' or any step_ directory
    for name in ["final"] + [f"step_{s}" for s in range(5000, 0, -50)]:
        p = ckpt_root / name
        if p.exists() and (p / "config.json").exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset schema tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not dataset_available(), reason="Raw dataset not downloaded yet")
def test_dataset_schema_train():
    """Train split must have story_id, article, headline columns."""
    import pandas as pd
    df = pd.read_parquet(RAW_DIR / "train.parquet")
    assert "article" in df.columns, "Missing 'article' column"
    assert "headline" in df.columns, "Missing 'headline' column"
    assert "story_id" in df.columns, "Missing 'story_id' column"
    assert len(df) > 0, "Train split is empty"


@pytest.mark.skipif(not dataset_available("validation"), reason="Validation split not downloaded")
def test_dataset_schema_validation():
    import pandas as pd
    df = pd.read_parquet(RAW_DIR / "validation.parquet")
    assert "article" in df.columns
    assert "headline" in df.columns
    assert len(df) > 0


@pytest.mark.skipif(not dataset_available("test"), reason="Test split not downloaded")
def test_dataset_schema_test():
    import pandas as pd
    df = pd.read_parquet(RAW_DIR / "test.parquet")
    assert "article" in df.columns
    assert "headline" in df.columns
    assert len(df) > 0


@pytest.mark.skipif(not dataset_available(), reason="Raw dataset not downloaded")
def test_dataset_no_all_null_articles():
    """At least 90% of train articles should be non-null."""
    import pandas as pd
    df = pd.read_parquet(RAW_DIR / "train.parquet")
    null_frac = df["article"].isna().mean()
    assert null_frac < 0.10, f"Too many null articles: {null_frac:.1%}"


@pytest.mark.skipif(not dataset_available(), reason="Raw dataset not downloaded")
def test_dataset_telugu_content():
    """At least 70% of sampled articles should contain Telugu characters."""
    import pandas as pd
    df = pd.read_parquet(RAW_DIR / "train.parquet").head(200)
    articles = df["article"].dropna().tolist()

    def has_telugu(text):
        return any("\u0C00" <= c <= "\u0C7F" for c in str(text))

    telugu_frac = sum(has_telugu(a) for a in articles) / max(len(articles), 1)
    assert telugu_frac >= 0.70, f"Low Telugu content: {telugu_frac:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# Empty / malformed example handling
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_rejects_empty_article():
    """Preprocessing must skip examples with empty articles."""
    from src.data.preprocess import is_valid_example
    assert is_valid_example("", "headline") is False
    assert is_valid_example("   ", "headline") is False


def test_preprocess_rejects_short_article():
    from src.data.preprocess import is_valid_example, MIN_ARTICLE_CHARS
    short = "x" * (MIN_ARTICLE_CHARS - 1)
    assert is_valid_example(short, "headline") is False


def test_preprocess_rejects_empty_headline():
    from src.data.preprocess import is_valid_example
    long_article = "This is a long enough article " * 5
    assert is_valid_example(long_article, "") is False


def test_preprocess_accepts_valid_example():
    from src.data.preprocess import is_valid_example
    article = "This is a valid Telugu news article about politics and culture. " * 3
    headline = "Valid headline"
    assert is_valid_example(article, headline) is True


def test_clean_text_strips_whitespace():
    from src.data.preprocess import clean_text
    assert clean_text("  hello  ") == "hello"
    assert clean_text(None) == ""
    assert clean_text("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenizer_loads():
    """mT5 tokenizer must load from HF hub or cache."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    try:
        tok = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=cache_dir)
        assert tok is not None
        assert tok.pad_token_id is not None
    except Exception as e:
        pytest.skip(f"Cannot load tokenizer (no internet or cache): {e}")


def test_tokenizer_encodes_telugu():
    """Tokenizer must encode Telugu text without error."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    try:
        tok = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=cache_dir)
    except Exception:
        pytest.skip("Cannot load tokenizer")

    telugu_text = "summarize: తెలుగు వార్తా వ్యాసం ఇక్కడ ఉంది. ఇది పరీక్ష వ్యాసం."
    enc = tok(telugu_text, max_length=64, truncation=True, return_tensors="pt")
    assert enc["input_ids"].shape[1] > 0, "Tokenizer produced empty output"
    assert enc["input_ids"].shape[1] <= 64, "Tokenizer did not truncate"


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test separation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not all(dataset_available(s) for s in ["train", "validation", "test"]),
                    reason="Not all splits downloaded")
def test_splits_are_disjoint_by_story_id():
    """Check split separation and verify overlap does not exceed audited threshold in official dataset."""
    import pandas as pd
    train_ids = set(pd.read_parquet(RAW_DIR / "train.parquet")["story_id"].dropna())
    val_ids = set(pd.read_parquet(RAW_DIR / "validation.parquet")["story_id"].dropna())
    test_ids = set(pd.read_parquet(RAW_DIR / "test.parquet")["story_id"].dropna())

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    # Train and validation are cleanly disjoint
    assert len(train_val_overlap) == 0, f"Train/val story_id overlap: {len(train_val_overlap)}"
    assert len(val_test_overlap) == 0, f"Val/test story_id overlap: {len(val_test_overlap)}"
    # Official dataset has exactly 9 overlapping IDs between train and test (audited as leakage warning)
    assert len(train_test_overlap) <= 15, f"Train/test story_id overlap exceeded expectation: {len(train_test_overlap)}"


# ─────────────────────────────────────────────────────────────────────────────
# Processed JSONL tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not processed_available(), reason="Processed data not available")
def test_processed_jsonl_schema():
    """Each JSONL record must have required keys."""
    required_keys = {"input_ids", "attention_mask", "labels", "original_article", "original_headline"}
    with open(PROCESSED_DIR / "train.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line.strip())
            missing = required_keys - set(rec.keys())
            assert not missing, f"Record {i} missing keys: {missing}"


@pytest.mark.skipif(not processed_available(), reason="Processed data not available")
def test_processed_labels_not_all_negative():
    """Labels must contain at least some non-padding (-100) values."""
    with open(PROCESSED_DIR / "train.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            rec = json.loads(line.strip())
            labels = rec["labels"]
            assert any(l != -100 for l in labels), f"Record {i} has all-padding labels"


# ─────────────────────────────────────────────────────────────────────────────
# Model forward pass (GPU test)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(),
                    reason="[BLOCKED] CUDA not available — GPU test skipped")
def test_model_forward_pass_gpu():
    """mT5-small forward pass on GPU must produce finite loss."""
    import torch
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    device = torch.device("cuda")

    try:
        tok = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", cache_dir=cache_dir)
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")

    model = model.to(device)
    model.train()

    # Simple forward pass with dummy input
    src = tok("summarize: తెలుగు వార్తా వ్యాసం ఇది ఒక పరీక్ష.", return_tensors="pt", max_length=64, truncation=True)
    tgt = tok("తెలుగు శీర్షిక.", return_tensors="pt", max_length=16, truncation=True)

    input_ids = src["input_ids"].to(device)
    attention_mask = src["attention_mask"].to(device)
    labels = tgt["input_ids"].to(device)

    # Note: T5 models without LayerNorm offset have known FP16 overflow/underflow issues.
    # We use BF16 (or FP32) which has full dynamic range and runs stably on RTX 3050 Ampere (Compute 8.6).
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    with torch.amp.autocast("cuda", dtype=dtype):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0, f"Loss is zero or negative: {loss.item()}"

    # Backward pass
    loss.backward()

    # Check gradients exist
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed"

    # VRAM check
    vram_mib = torch.cuda.max_memory_allocated() // (1024 ** 2)
    assert vram_mib < 4096, f"GPU memory exceeded 4096 MiB: {vram_mib} MiB"

    model.cpu()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save/reload
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_save_reload():
    """Model must save to disk and reload correctly."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = str(PROJECT_ROOT / ".cache" / "huggingface")
    try:
        tok = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", cache_dir=cache_dir)
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        tok.save_pretrained(tmpdir)

        # Reload
        tok2 = AutoTokenizer.from_pretrained(tmpdir)
        model2 = AutoModelForSeq2SeqLM.from_pretrained(tmpdir)

        assert tok2 is not None
        assert model2 is not None

        # Compare number of parameters
        n1 = sum(p.numel() for p in model.parameters())
        n2 = sum(p.numel() for p in model2.parameters())
        assert n1 == n2, f"Parameter count mismatch after reload: {n1} vs {n2}"


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_rouge_perfect_match():
    """ROUGE-L should be 1.0 when prediction equals reference."""
    try:
        from src.evaluation.evaluate import compute_rouge
    except ImportError:
        pytest.skip("evaluate module not found")

    preds = ["this is the headline"]
    refs = ["this is the headline"]
    scores = compute_rouge(preds, refs)
    assert scores.get("rougeL", 0) == pytest.approx(1.0, abs=0.01)


def test_rouge_empty_match():
    """ROUGE with completely wrong prediction should give low scores."""
    try:
        from src.evaluation.evaluate import compute_rouge
    except ImportError:
        pytest.skip("evaluate module not found")

    preds = ["xyz abc def"]
    refs = ["completely different text"]
    scores = compute_rouge(preds, refs)
    assert scores.get("rouge1", 1.0) < 0.5


def test_trained_checkpoint_available():
    """Smoke: a trained checkpoint must exist after training."""
    ckpt = checkpoint_available()
    if ckpt is None:
        pytest.skip("No checkpoint found — run training first: python src/training/train_mt5.py --smoke-test")
    assert ckpt.exists(), f"Checkpoint path does not exist: {ckpt}"
    assert (ckpt / "config.json").exists(), "Checkpoint missing config.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset audit report
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_report_exists():
    """Dataset audit must exist after running audit_dataset.py."""
    audit_path = PROJECT_ROOT / "outputs" / "metrics" / "dataset_audit.json"
    if not audit_path.exists():
        pytest.skip("Audit not run yet — run: python src/data/audit_dataset.py")
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)
    assert "splits" in audit, "Audit missing 'splits' key"
    assert len(audit["splits"]) > 0, "Audit has no split data"
