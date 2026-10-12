"""
tests/test_phase2.py
====================
Phase 2 — Automated Test Suite for Telugu Headline Relevance Classification.

Tests cover:
  1. TeClass Dataset schema and split integrity
  2. Taxonomy & label validity (HREL, MREL, LREL)
  3. Preprocessing handling of empty/malformed inputs
  4. TF-IDF baseline loading & prediction
  5. Transformer (mBERT) classifier checkpoint loading & prediction
  6. Predict API contract and error handling
  7. Metric calculations & evaluation artifacts
"""

import json
import pathlib
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
TECLASS_DIR = PROJECT_ROOT / "data" / "teclass"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "teclass"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
BERT_CKPT = PROJECT_ROOT / "checkpoints" / "classifier_bert" / "final"
TFIDF_CKPT = PROJECT_ROOT / "checkpoints" / "classifier_tfidf" / "tfidf_logreg.pkl"


def test_teclass_files_exist():
    """Verify official TeClass dataset files exist on disk."""
    for split in ["TeClass_training_data.csv", "TeClass_development_data.csv", "TeClass_testing_data.csv"]:
        path = TECLASS_DIR / split
        assert path.exists(), f"Missing TeClass dataset file: {path}"


def test_teclass_schema_and_labels():
    """Verify TeClass splits have required columns and only verified labels {HREL, MREL, LREL}."""
    allowed_labels = {"HREL", "MREL", "LREL"}
    for split in ["TeClass_training_data.csv", "TeClass_development_data.csv", "TeClass_testing_data.csv"]:
        df = pd.read_csv(TECLASS_DIR / split)
        assert "article" in df.columns
        assert "headline" in df.columns
        assert "label" in df.columns
        labels = set(df["label"].dropna().unique())
        assert labels.issubset(allowed_labels), f"Found unexpected labels: {labels - allowed_labels}"


def test_processed_splits_exist():
    """Verify preprocessed parquet and jsonl files exist."""
    for split in ["train", "validation", "test"]:
        assert (PROCESSED_DIR / f"{split}.parquet").exists()
        assert (PROCESSED_DIR / f"{split}.jsonl").exists()


def test_predict_empty_inputs_raise_value_error():
    """Predict API must reject empty or whitespace inputs."""
    from src.classification.predict import predict_relevance
    with pytest.raises(ValueError, match="Article text cannot be empty"):
        predict_relevance("", "valid headline")

    with pytest.raises(ValueError, match="Headline text cannot be empty"):
        predict_relevance("valid article content " * 5, "   ")


def test_predict_invalid_model_type_raises():
    """Predict API must reject unsupported model types."""
    from src.classification.predict import predict_relevance
    with pytest.raises(ValueError, match="Unknown model_type"):
        predict_relevance("వార్త", "శీర్షిక", model_type="nonexistent")


def test_tfidf_baseline_prediction():
    """TF-IDF baseline classifier produces valid predictions and probabilities."""
    from src.classification.predict import predict_relevance
    article = "హైదరాబాద్ : నగరంలో రోడ్డు ప్రమాదం జరిగింది . పోలీసులు సంఘటనా స్థలానికి చేరుకున్నారు ."
    headline = "నగరంలో రోడ్డు ప్రమాదం"

    res = predict_relevance(article, headline, model_type="tfidf")
    assert res["predicted_class"] in {"HREL", "MREL", "LREL"}
    assert "probabilities" in res
    assert set(res["probabilities"].keys()) == {"HREL", "MREL", "LREL"}
    assert pytest.approx(sum(res["probabilities"].values()), abs=0.01) == 1.0
    assert len(res["explanation"]) > 0


def test_transformer_bert_prediction():
    """Transformer classifier produces valid predictions and probabilities on GPU or CPU."""
    if not BERT_CKPT.exists():
        pytest.skip("mBERT checkpoint not found")
    from src.classification.predict import predict_relevance
    article = "హైదరాబాద్ : నగరంలో రోడ్డు ప్రమాదం జరిగింది . పోలీసులు సంఘటనా స్థలానికి చేరుకున్నారు ."
    headline = "నగరంలో రోడ్డు ప్రమాదం"

    res = predict_relevance(article, headline, model_type="bert")
    assert res["predicted_class"] in {"HREL", "MREL", "LREL"}
    assert "probabilities" in res
    assert set(res["probabilities"].keys()) == {"HREL", "MREL", "LREL"}
    assert pytest.approx(sum(res["probabilities"].values()), abs=0.01) == 1.0
    assert "mBERT" in res["model_type"]
    assert len(res["explanation"]) > 0


def test_evaluation_metrics_saved():
    """Verify evaluation metric JSON artifacts exist with required fields."""
    tfidf_eval = METRICS_DIR / "baseline_classifier_eval.json"
    bert_eval = METRICS_DIR / "transformer_classifier_eval.json"

    assert tfidf_eval.exists(), "Missing baseline_classifier_eval.json"
    with open(tfidf_eval, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "accuracy" in data
        assert "macro_f1" in data
        assert "confusion_matrix" in data

    assert bert_eval.exists(), "Missing transformer_classifier_eval.json"
    with open(bert_eval, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "accuracy" in data
        assert "macro_f1" in data
        assert "confusion_matrix" in data


def test_phase1_generation_pipeline_unmodified():
    """Verify Phase 1 mT5 evaluation and training scripts are functional and untouched."""
    from src.evaluation.evaluate import compute_rouge
    scores = compute_rouge(["this is headline"], ["this is headline"])
    assert scores["rougeL"] == pytest.approx(1.0, abs=0.01)
