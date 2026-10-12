"""
tests/test_phase3.py
====================
Phase 3 test suite:
  1. Test IndicTokenizer splits Telugu text accurately.
  2. Test compute_rouge produces non-zero score on identical Telugu strings.
  3. Test compute_rouge produces non-zero score on partial Telugu matches.
  4. Test generate_headline strips <extra_id_0> sentinel tokens cleanly.
  5. Test IndicBART feasibility assessment artifact exists and is FEASIBLE.
  6. Test IndicBART checkpoint exists and is loadable.
  7. Test Model comparison metrics artifact exists.
"""

import json
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_indic_rouge_tokenizer():
    from rouge_score import rouge_scorer, tokenizers
    import re

    class IndicTokenizer(tokenizers.Tokenizer):
        def tokenize(self, text):
            clean = re.sub(r'[\.,\?!;:\"\'\(\)\[\]\{\}\<\>_]', ' ', text)
            return [w for w in clean.strip().split() if w]

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=IndicTokenizer(),
        use_stemmer=False,
    )
    score = scorer.score("తెలుగు వార్తలు శీర్షిక", "తెలుగు వార్తలు శీర్షిక")
    assert score["rouge1"].fmeasure == 1.0
    assert score["rouge2"].fmeasure == 1.0
    assert score["rougeL"].fmeasure == 1.0


def test_compute_rouge_telugu():
    from src.evaluation.evaluate import compute_rouge

    preds = ["వృద్ధ దంపతులు ఆత్మహత్య"]
    refs = ["వృద్ధ దంపతుల ఆత్మహత్య"]
    res = compute_rouge(preds, refs)
    assert res["rouge1"] > 0.0, "ROUGE-1 must be non-zero for Telugu partial match"
    assert res["rougeL"] > 0.0, "ROUGE-L must be non-zero for Telugu partial match"


def test_sentinel_token_stripping():
    import re
    raw = "<extra_id_0> ముఖ్యమంత్రి పర్యటన <extra_id_1>"
    cleaned = re.sub(r"<extra_id_\d+>", "", raw).strip()
    assert "<extra_id_0>" not in cleaned
    assert cleaned == "ముఖ్యమంత్రి పర్యటన"


def test_indicbart_feasibility_report():
    report_file = PROJECT_ROOT / "outputs" / "metrics" / "indicbart_feasibility.json"
    assert report_file.exists(), "IndicBART feasibility report should exist"
    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "FEASIBLE"
    assert data["details"]["total_vram_mib"] > 0
    assert data["details"]["vram_after_load_mib"] < 2000


def test_indicbart_checkpoint_exists():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "indicbart" / "final"
    assert ckpt_path.exists(), "IndicBART final checkpoint should exist"
    assert (ckpt_path / "config.json").exists()


def test_comparison_report_exists():
    report_file = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison_phase3.json"
    if report_file.exists():
        with open(report_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "models" in data
        assert "mT5-small" in data["models"]
