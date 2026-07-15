"""
fit_predictor.py — Shortlisting / Fit Predictor (Supervised — MODEL B, optional).

Predicts whether a resume is a good fit for a specific job using
engineered features (cosine sim + skill overlap + experience match).

Models compared: Logistic Regression → XGBoost

Usage:
    from src.models.fit_predictor import train_fit_predictor, predict_fit_score
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[fit_predictor] XGBoost not installed. Only Logistic Regression will be used.")

from src.config import FIT_PREDICTOR_PATH, RANDOM_STATE, TEST_SIZE, CV_FOLDS


# ── Train ─────────────────────────────────────────────────────────────────────

def train_fit_predictor(
    X: pd.DataFrame,
    y: np.ndarray | list,
    save: bool = True,
) -> tuple[object, dict]:
    """
    Train the fit predictor.

    Parameters
    ----------
    X    : feature DataFrame from build_match_features() — shape (n_samples, 3)
    y    : binary labels (1 = good fit, 0 = not fit)
    save : persist best model to FIT_PREDICTOR_PATH

    Returns
    -------
    (best_model, results_dict)
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_arr
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]),
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    results = {}
    print(f"\n[train_fit_predictor] Evaluating {len(candidates)} models …\n")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc")
        results[name] = {"cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()}
        print(f"  {name:25s}  ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name  = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    best_model = candidates[best_name]
    print(f"\n[train_fit_predictor] Best model: {best_name}")

    best_model.fit(X_train, y_train)
    y_pred  = best_model.predict(X_test)
    y_proba = (
        best_model.predict_proba(X_test)[:, 1]
        if hasattr(best_model, "predict_proba")
        else best_model.decision_function(X_test)
    )

    results["best_model"]  = best_name
    results["test_report"] = classification_report(y_test, y_pred, output_dict=True)
    results["roc_auc"]     = roc_auc_score(y_test, y_proba)
    results["conf_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nTest ROC-AUC: {results['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred))

    if save:
        joblib.dump(best_model, FIT_PREDICTOR_PATH)
        print(f"[train_fit_predictor] Model saved → {FIT_PREDICTOR_PATH}")

    return best_model, results


# ── Predict ───────────────────────────────────────────────────────────────────

def predict_fit_score(features: np.ndarray | pd.DataFrame, model=None) -> float:
    """
    Predict the probability that a candidate is a good fit.

    Parameters
    ----------
    features : single row of match features (shape (1, 3) or (3,))
    model    : fitted model (loaded if None)

    Returns
    -------
    Float in [0, 1] — fit probability.
    """
    if model is None:
        model = load_fit_predictor()

    if isinstance(features, pd.DataFrame):
        features = features.values
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0][1])
    return float(model.predict(features)[0])


# ── Load ──────────────────────────────────────────────────────────────────────

def load_fit_predictor(path=FIT_PREDICTOR_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Fit predictor not found at {path}. Run train_fit_predictor() first."
        )
    return joblib.load(path)
