"""
Machine Learning Virality Prediction Engine
Trains Random Forest & Gradient Boosting ensembles to classify high-virality content (>=1M views).
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def train_virality_model(
    parquet_path: str = "youtube-content-intelligence/data/processed/youtube_transformed.parquet",
    model_dir: str = "youtube-content-intelligence/models",
) -> dict:
    print("Training Machine Learning Virality Classifier...")
    df = pd.read_parquet(parquet_path)

    # Features selected for pre-publish and early-signal prediction:
    # 1. Metadata: category_id, publish_hour, publish_month, tag_count
    # 2. Title NLP: title_word_count, title_char_length, title_caps_ratio, title_sentiment_polarity
    # 3. Early Engagement Ratio signals: like_to_view_ratio, comment_to_view_ratio
    feature_cols = [
        "category_id",
        "publish_hour",
        "publish_month",
        "tag_count",
        "title_word_count",
        "title_char_length",
        "title_caps_ratio",
        "title_sentiment_polarity",
        "like_to_view_ratio",
        "comment_to_view_ratio",
    ]

    X = df[feature_cols].fillna(0)
    y = df["is_viral"]

    # Stratified Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1,
    )

    # 2. Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
    )

    # Ensemble: Soft Voting Classifier combining RF and Gradient Boosting
    ensemble_clf = VotingClassifier(
        estimators=[("rf", rf_model), ("gb", gb_model)],
        voting="soft",
    )

    print("Fitting Ensemble Classifier on Training Data...")
    ensemble_clf.fit(X_train, y_train)

    # Predict Probabilities and Classes on Test Set
    y_pred = ensemble_clf.predict(X_test)
    y_pred_proba = ensemble_clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_pred_proba))
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract Feature Importances from RF base estimator
    rf_fitted = ensemble_clf.named_estimators_["rf"]
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(feature_cols, rf_fitted.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    metrics = {
        "accuracy_pct": round(acc * 100, 2),
        "roc_auc_score": round(auc, 4),
        "precision_viral": round(report["1"]["precision"], 4),
        "recall_viral": round(report["1"]["recall"], 4),
        "f1_viral": round(report["1"]["f1-score"], 4),
        "test_records": len(X_test),
        "feature_importances": importances,
        "feature_names": feature_cols,
    }

    # Save Serialized Model Artifact
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "virality_classifier.pkl"
    joblib.dump(ensemble_clf, model_path)

    metrics_path = out_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model Training & Validation Complete!")
    print(f"Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f} | F1 Score: {report['1']['f1-score']:.4f}")
    print(f"Top 3 Predictors: {list(importances.items())[:3]}")
    print(f"Serialized model saved to: {model_path}")
    return metrics


if __name__ == "__main__":
    train_virality_model()
