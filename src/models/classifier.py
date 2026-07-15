"""
classifier.py — Resume Category Classifier (Supervised — MODEL A).

Pipeline:
    clean text → TF-IDF → Logistic Regression / SVM / Random Forest
    (best model selected via cross-validation)

Usage:
    from src.models.classifier import train_classifier, predict_category, load_classifier
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    CLASSIFIER_PATH,
    MAX_TFIDF_FEATURES,
    NGRAM_RANGE,
    MIN_DF,
    TEST_SIZE,
    CV_FOLDS,
    RANDOM_STATE,
)


# ── Build pipelines ────────────────────────────────────────────────────────────

def _build_pipelines() -> dict[str, Pipeline]:
    """Return candidate classification pipelines."""
    tfidf = dict(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
    )
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=5.0)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   LinearSVC(max_iter=2000, random_state=RANDOM_STATE, C=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


# ── Train ─────────────────────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    text_col: str = "clean_resume",
    label_col: str = "Category",
    save: bool = True,
) -> tuple[Pipeline, dict]:
    """
    Train resume category classifier.

    Parameters
    ----------
    df        : preprocessed DataFrame (output of preprocess_resumes)
    text_col  : column containing cleaned text
    label_col : column containing category string labels
    save      : persist best model to CLASSIFIER_PATH

    Returns
    -------
    (best_pipeline, results_dict)
    """
    X = df[text_col].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = _build_pipelines()
    results   = {}

    print(f"\n[train_classifier] Evaluating {len(pipelines)} models via {CV_FOLDS}-fold CV …\n")
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        results[name] = {"cv_mean": scores.mean(), "cv_std": scores.std()}
        print(f"  {name:25s}  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Select best model
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best_pipe = pipelines[best_name]
    print(f"\n[train_classifier] Best model: {best_name}")

    # Final train on full training set, evaluate on held-out test set
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))

    results["best_model"]  = best_name
    results["test_report"] = report
    results["confusion_matrix"] = cm
    results["classes"]     = sorted(set(y))

    print("\n[train_classifier] Test-set Classification Report:")
    print(classification_report(y_test, y_pred))

    if save:
        joblib.dump(best_pipe, CLASSIFIER_PATH)
        print(f"[train_classifier] Model saved → {CLASSIFIER_PATH}")

    return best_pipe, results


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_category(text: str, pipeline: Pipeline | None = None) -> str:
    """
    Predict the job category for a single resume text string.

    Parameters
    ----------
    text     : cleaned resume text
    pipeline : fitted pipeline (loaded automatically if None)

    Returns
    -------
    Predicted category string (e.g. 'Data Science').
    """
    if pipeline is None:
        pipeline = load_classifier()
    return pipeline.predict([text])[0]


def predict_proba(text: str, pipeline: Pipeline | None = None) -> dict[str, float]:
    """
    Return class probabilities for a single resume text.
    Only works for models that support predict_proba (e.g. Logistic Regression).
    """
    if pipeline is None:
        pipeline = load_classifier()
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba.")
    proba  = pipeline.predict_proba([text])[0]
    labels = pipeline.classes_
    return dict(sorted(zip(labels, proba), key=lambda x: -x[1]))


# ── Persist / Load ────────────────────────────────────────────────────────────

def load_classifier(path=CLASSIFIER_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Classifier not found at {path}. Run train_classifier() first."
        )
    pipeline = joblib.load(path)
    print(f"[load_classifier] Loaded from {path}")
    return pipeline
