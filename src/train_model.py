"""Reproducible Model Training and Evaluation Pipeline for PhishGuard AI.

Fits a TF-IDF + Logistic Regression pipeline strictly on training data and computes
genuine, un-hardcoded evaluation metrics on the held-out test split.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline

from src.data_loader import load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard.trainer")


def build_pipeline() -> Pipeline:
    """Build the TF-IDF and Logistic Regression scikit-learn Pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
        ),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                solver="lbfgs"
            )
        )
    ])


def train_and_evaluate(
    data_path: str = "data/phishing_dataset.csv",
    model_output_path: str = "models/phishguard_model.joblib",
    metadata_output_path: str = "models/model_metadata.json",
    provenance_path: Optional[str] = "data/dataset_provenance.json",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train the model pipeline and evaluate strictly on held-out test data."""
    logger.info("Starting PhishGuard AI training pipeline...")

    # 1. Load and split data
    X_train, X_test, y_train, y_test, data_meta = load_and_preprocess_data(
        csv_path=data_path,
        test_size=test_size,
        random_state=random_state
    )

    # 2. Build Pipeline
    pipeline = build_pipeline()

    # 3. Fit pipeline strictly on X_train (TF-IDF fitted ONLY on training split)
    logger.info("Fitting TF-IDF and Logistic Regression pipeline on %d training samples...", len(X_train))
    pipeline.fit(X_train, y_train)

    # 4. Predict on held-out X_test
    logger.info("Evaluating on %d held-out test samples...", len(X_test))
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # 5. Compute genuine evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_binary = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_binary = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1_binary = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()
    # cm is [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        output_dict=True,
        zero_division=0
    )

    # Extract top influential TF-IDF n-grams for transparency
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    top_phish_indices = np.argsort(coefs)[-15:][::-1]
    top_legit_indices = np.argsort(coefs)[:15]

    top_phishing_features = [
        {"feature": str(feature_names[i]), "weight": float(round(coefs[i], 4))}
        for i in top_phish_indices
    ]
    top_legitimate_features = [
        {"feature": str(feature_names[i]), "weight": float(round(coefs[i], 4))}
        for i in top_legit_indices
    ]

    # Load dataset provenance if available
    dataset_provenance = {}
    if provenance_path and Path(provenance_path).is_file():
        try:
            with open(provenance_path, "r", encoding="utf-8") as f:
                dataset_provenance = json.load(f)
        except Exception as e:
            logger.warning("Failed to load provenance file: %s", e)

    evaluation_metadata = {
        "model_name": "PhishGuard-LR-TFIDF",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_state,
        "pipeline_architecture": {
            "vectorizer": "TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)",
            "classifier": "LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')"
        },
        "dataset_split": data_meta,
        "dataset_provenance": dataset_provenance,
        "evaluation_metrics": {
            "accuracy": round(acc, 4),
            "precision_phishing": round(prec_binary, 4),
            "recall_phishing": round(rec_binary, 4),
            "f1_phishing": round(f1_binary, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix": {
                "true_negatives_legit_correct": tn,
                "false_positives_legit_as_phish": fp,
                "false_negatives_phish_missed": fn,
                "true_positives_phish_caught": tp,
                "raw_matrix": cm
            },
            "per_class_report": report_dict
        },
        "explainability": {
            "top_phishing_indicators_tfidf": top_phishing_features,
            "top_legitimate_indicators_tfidf": top_legitimate_features
        },
        "limitations_and_ethics": {
            "calibrated_probabilities": False,
            "calibration_note": "Logistic regression sigmoid probabilities are uncalibrated and reflect linear score confidence.",
            "known_limitations": [
                "Model relies on lexical and structural n-gram patterns in text and cannot verify actual cryptographic email signatures (SPF/DKIM/DMARC).",
                "Novel zero-day phishing lures that use unfamiliar phrasing may evade bag-of-words classifiers (false negatives).",
                "Legitimate emails that contain urgent business deadlines or invoice discussions may trigger false alarms (false positives).",
                "Designed as a decision-support demonstration tool, not an autonomous email gateway."
            ]
        }
    }

    # 6. Save model and metadata
    model_path = Path(model_output_path)
    meta_path = Path(metadata_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    logger.info("Saved fitted model pipeline to %s", model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_metadata, f, indent=2)
    logger.info("Saved model evaluation metadata to %s", meta_path)

    logger.info(
        "Evaluation summary: Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | TestSize=%d",
        acc, prec_binary, rec_binary, f1_binary, len(X_test)
    )

    return evaluation_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhishGuard AI classification pipeline")
    parser.add_argument("--data-path", default="data/phishing_dataset.csv", help="Path to input CSV dataset")
    parser.add_argument("--output-model", default="models/phishguard_model.joblib", help="Output model joblib path")
    parser.add_argument("--output-metadata", default="models/model_metadata.json", help="Output metadata JSON path")
    parser.add_argument("--provenance-file", default="data/dataset_provenance.json", help="Path to dataset provenance JSON")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for held-out test split")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data_path,
        model_output_path=args.output_model,
        metadata_output_path=args.output_metadata,
        provenance_path=args.provenance_file,
        test_size=args.test_size,
        random_state=args.seed
    )
