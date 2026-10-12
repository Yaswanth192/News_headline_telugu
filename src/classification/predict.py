"""
src/classification/predict.py
=============================
Phase 2 — Unified Inference Pipeline for Telugu Headline Relevance Classification.

Usage:
  python src/classification/predict.py --headline "తెలుగు శీర్షిక" --article "తెలుగు వార్తా వ్యాసం..." [--model-type bert|tfidf]

Interface:
  predict_relevance(article, headline, model_type="bert") -> dict:
    - predicted_class: 'HREL' | 'MREL' | 'LREL'
    - probabilities: {'HREL': p1, 'MREL': p2, 'LREL': p3}
    - model_checkpoint: str
    - explanation: str (Traceable to TeClass annotation definitions)
"""

import argparse
import json
import logging
import pathlib
import sys
import pickle
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
BERT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "classifier_bert" / "final"
TFIDF_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "classifier_tfidf" / "tfidf_logreg.pkl"

LABEL_MAPPING = {
    0: "HREL",
    1: "MREL",
    2: "LREL"
}

LABEL_EXPLANATIONS = {
    "HREL": "Highly Related: Headline directly and factually captures the article's core main event (FME - Factual Main Event).",
    "MREL": "Moderately Related: Headline focuses on secondary events (FSE), logical conclusions (STC), or minor takeaways (WKC).",
    "LREL": "Least Related: Headline is sensationalized (SEN), clickbait (CBT), misleading (MLC), or out of scope (USO)."
}

_BERT_MODEL = None
_BERT_TOKENIZER = None
_TFIDF_ARTIFACT = None


def get_bert_model():
    global _BERT_MODEL, _BERT_TOKENIZER
    if _BERT_MODEL is None:
        if not BERT_CKPT_DIR.exists():
            raise FileNotFoundError(f"BERT checkpoint not found at: {BERT_CKPT_DIR}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _BERT_TOKENIZER = AutoTokenizer.from_pretrained(str(BERT_CKPT_DIR))
        _BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(str(BERT_CKPT_DIR)).to(device)
        _BERT_MODEL.eval()
    return _BERT_MODEL, _BERT_TOKENIZER


def get_tfidf_model():
    global _TFIDF_ARTIFACT
    if _TFIDF_ARTIFACT is None:
        if not TFIDF_CKPT_PATH.exists():
            raise FileNotFoundError(f"TF-IDF model artifact not found at: {TFIDF_CKPT_PATH}")
        with open(TFIDF_CKPT_PATH, "rb") as f:
            _TFIDF_ARTIFACT = pickle.load(f)
    return _TFIDF_ARTIFACT


def predict_relevance(article: str, headline: str, model_type: str = "bert") -> dict:
    art = str(article or "").strip()
    hl = str(headline or "").strip()

    if not art:
        raise ValueError("Article text cannot be empty.")
    if not hl:
        raise ValueError("Headline text cannot be empty.")

    if model_type.lower() == "bert":
        model, tokenizer = get_bert_model()
        device = next(model.parameters()).device
        inputs = tokenizer(
            hl,
            art,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().tolist()

        pred_idx = int(torch.argmax(outputs.logits, dim=1).item())
        pred_label = LABEL_MAPPING.get(pred_idx, "UNKNOWN")

        return {
            "predicted_class": pred_label,
            "probabilities": {
                "HREL": round(probs[0], 4),
                "MREL": round(probs[1], 4),
                "LREL": round(probs[2], 4)
            },
            "model_type": "mBERT (google-bert/bert-base-multilingual-cased)",
            "checkpoint": str(BERT_CKPT_DIR),
            "explanation": LABEL_EXPLANATIONS.get(pred_label, "")
        }

    elif model_type.lower() == "tfidf":
        artifact = get_tfidf_model()
        vec = artifact["vectorizer"]
        clf = artifact["classifier"]

        combined = f"{hl} [SEP] {art}"
        X_vec = vec.transform([combined])
        pred_idx = int(clf.predict(X_vec)[0])
        probs = clf.predict_proba(X_vec)[0].tolist()
        pred_label = LABEL_MAPPING.get(pred_idx, "UNKNOWN")

        return {
            "predicted_class": pred_label,
            "probabilities": {
                "HREL": round(probs[0], 4),
                "MREL": round(probs[1], 4),
                "LREL": round(probs[2], 4)
            },
            "model_type": "TF-IDF + Logistic Regression",
            "checkpoint": str(TFIDF_CKPT_PATH),
            "explanation": LABEL_EXPLANATIONS.get(pred_label, "")
        }

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'bert' or 'tfidf'.")


def main():
    parser = argparse.ArgumentParser(description="Classify Telugu headline relevance against article")
    parser.add_argument("--headline", required=True, help="Telugu headline text")
    parser.add_argument("--article", required=True, help="Telugu news article text")
    parser.add_argument("--model-type", default="bert", choices=["bert", "tfidf"], help="Model type to use")
    args = parser.parse_args()

    result = predict_relevance(args.article, args.headline, model_type=args.model_type)
    print("\n" + "=" * 60)
    print("TELUGU HEADLINE RELEVANCE CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"Headline:        {args.headline}")
    print(f"Article (snip):  {args.article[:120]}...")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Probabilities:   {result['probabilities']}")
    print(f"Model:           {result['model_type']}")
    print(f"Explanation:     {result['explanation']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
