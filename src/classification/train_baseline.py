"""
src/classification/train_baseline.py
====================================
Phase 2 — Baseline Classifier (TF-IDF + Logistic Regression).

Trains a lightweight, fast, reproducible classical ML baseline on the 
Telugu article-headline classification task.
Input: f"{headline} [SEP] {article}"
Target: {0: HREL, 1: MREL, 2: LREL}

Saves:
  - Trained model pipeline (TfidfVectorizer + LogisticRegression)
  - Validation metrics (Macro-F1, Weighted-F1, Per-class, Accuracy, Confusion Matrix)

Run:
    python src/classification/train_baseline.py
"""

import argparse
import json
import logging
import pathlib
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

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
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "teclass"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "classifier_tfidf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"

TARGET_NAMES = ["HREL", "MREL", "LREL"]


def train_baseline(data_dir: pathlib.Path, ckpt_dir: pathlib.Path, output_dir: pathlib.Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"

    if not train_path.exists() or not val_path.exists():
        logger.error("Preprocessed data not found. Run src/classification/preprocess_teclass.py first.")
        sys.exit(1)

    logger.info(f"Loading training data from {train_path.name}...")
    df_train = pd.read_parquet(train_path)
    logger.info(f"Loading validation data from {val_path.name}...")
    df_val = pd.read_parquet(val_path)

    X_train = df_train["combined_text"].tolist()
    y_train = df_train["label"].tolist()

    X_val = df_val["combined_text"].tolist()
    y_val = df_val["label"].tolist()

    logger.info(f"Training set: {len(X_train)} samples. Validation set: {len(X_val)} samples.")

    # 1. Feature Extraction: Character + Word n-gram TF-IDF
    logger.info("Extracting TF-IDF features (subword n-grams 1 to 3, max 30,000 features)...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # 2. Classifier: Logistic Regression with balanced class weights
    logger.info("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)

    # 3. Evaluation on Validation Set
    y_pred = clf.predict(X_val_vec)
    y_proba = clf.predict_proba(X_val_vec)

    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_val, y_pred, average="weighted"))
    report_dict = classification_report(y_val, y_pred, target_names=TARGET_NAMES, output_dict=True)
    conf_mat = confusion_matrix(y_val, y_pred).tolist()

    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation Macro-F1: {macro_f1:.4f}")
    logger.info(f"Validation Weighted-F1: {weighted_f1:.4f}")

    # 4. Save Checkpoint
    model_artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "target_names": TARGET_NAMES,
        "id_to_label": {0: "HREL", 1: "MREL", 2: "LREL"}
    }
    model_path = ckpt_dir / "tfidf_logreg.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    logger.info(f"Model artifact saved to: {model_path}")

    # 5. Save Evaluation Metrics
    eval_results = {
        "model_type": "TF-IDF + LogisticRegression",
        "split": "validation",
        "num_samples": len(y_val),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            k: {
                "precision": round(report_dict[k]["precision"], 4),
                "recall": round(report_dict[k]["recall"], 4),
                "f1_score": round(report_dict[k]["f1-score"], 4),
                "support": int(report_dict[k]["support"])
            } for k in TARGET_NAMES
        },
        "confusion_matrix": conf_mat,
        "classes": TARGET_NAMES
    }

    eval_path = output_dir / "baseline_classifier_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation report saved to: {eval_path}")

    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    train_baseline(pathlib.Path(args.data_dir), pathlib.Path(args.ckpt_dir), pathlib.Path(args.output_dir))


if __name__ == "__main__":
    main()
