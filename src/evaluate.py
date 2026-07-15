"""
evaluate.py — Shared evaluation utilities for all SmartHire models.

Usage:
    from src.evaluate import plot_confusion_matrix, precision_at_k, print_metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    silhouette_score,
)

from src.config import FIGURES_DIR


# ── Classification ────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return a dict of standard classification metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict."""
    print("\n── Evaluation Metrics ──────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("────────────────────────────────────────────\n")


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list | None = None,
    title: str = "Confusion Matrix",
    save_name: str = "confusion_matrix.png",
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(max(8, len(cm)), max(6, len(cm) - 1)))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=labels or "auto",
        yticklabels=labels or "auto",
        cmap="Blues",
        linewidths=0.5,
    )
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = FIGURES_DIR / save_name
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot_confusion_matrix] Saved → {fig_path}")


# ── Clustering ────────────────────────────────────────────────────────────────

def clustering_metrics(matrix, labels) -> dict:
    """Return silhouette score and inertia (if KMeans labels)."""
    sil = silhouette_score(matrix, labels, sample_size=min(5000, matrix.shape[0]))
    return {"silhouette_score": sil}


# ── Recommender ───────────────────────────────────────────────────────────────

def precision_at_k(
    recommended_titles: list[str],
    relevant_titles: list[str],
    k: int = 10,
) -> float:
    """
    Precision@K for a single query.

    Parameters
    ----------
    recommended_titles : ordered list of recommended job titles
    relevant_titles    : set of ground-truth relevant job titles
    k                  : cutoff

    Returns
    -------
    Float in [0, 1]
    """
    top_k    = recommended_titles[:k]
    relevant = set(relevant_titles)
    hits     = sum(1 for t in top_k if t in relevant)
    return hits / k


def mean_precision_at_k(
    all_recommended: list[list[str]],
    all_relevant: list[list[str]],
    k: int = 10,
) -> float:
    """Mean Precision@K over multiple queries."""
    scores = [
        precision_at_k(rec, rel, k)
        for rec, rel in zip(all_recommended, all_relevant)
    ]
    return float(np.mean(scores))
