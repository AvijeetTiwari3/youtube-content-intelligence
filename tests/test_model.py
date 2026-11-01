"""
Unit Tests for Machine Learning Virality Classifier
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_model_artifact_exists():
    model_path = ROOT_DIR / "models" / "virality_classifier.pkl"
    metrics_path = ROOT_DIR / "models" / "model_metrics.json"

    assert model_path.exists(), "virality_classifier.pkl missing"
    assert metrics_path.exists(), "model_metrics.json missing"


def test_model_inference():
    model_path = ROOT_DIR / "models" / "virality_classifier.pkl"
    model = joblib.load(model_path)

    # Synthetic test input
    sample_features = pd.DataFrame([{
        "category_id": 24,
        "publish_hour": 15,
        "publish_month": 11,
        "tag_count": 18,
        "title_word_count": 8,
        "title_char_length": 55,
        "title_caps_ratio": 0.25,
        "title_sentiment_polarity": 0.35,
        "like_to_view_ratio": 0.045,
        "comment_to_view_ratio": 0.005,
    }])

    pred = model.predict(sample_features)
    proba = model.predict_proba(sample_features)

    assert pred[0] in [0, 1], "Prediction must be 0 or 1"
    assert 0.0 <= proba[0, 1] <= 1.0, "Probability must be between 0 and 1"
    assert proba.shape == (1, 2), "Probability output must have shape (1, 2)"


def test_model_evaluation_metrics():
    metrics_path = ROOT_DIR / "models" / "model_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["accuracy_pct"] >= 80.0, f"Expected accuracy >= 80%, got {metrics['accuracy_pct']}"
    assert metrics["roc_auc_score"] >= 0.85, f"Expected ROC-AUC >= 0.85, got {metrics['roc_auc_score']}"
    assert "feature_importances" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
