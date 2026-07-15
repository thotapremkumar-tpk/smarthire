"""
text_features.py — TF-IDF vectorizer helpers.

Usage:
    from src.features.text_features import fit_tfidf, transform_tfidf, load_vectorizer
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from src.config import (
    TFIDF_PATH,
    MAX_TFIDF_FEATURES,
    NGRAM_RANGE,
    MIN_DF,
)


# ── Fit ────────────────────────────────────────────────────────────────────────

def fit_tfidf(
    corpus: list[str] | pd.Series,
    max_features: int = MAX_TFIDF_FEATURES,
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int = MIN_DF,
    save: bool = True,
) -> tuple[TfidfVectorizer, spmatrix]:
    """
    Fit a TF-IDF vectorizer on the given corpus and return
    (fitted_vectorizer, sparse_matrix).

    Parameters
    ----------
    corpus       : iterable of cleaned strings
    max_features : vocabulary size cap
    ngram_range  : (min_n, max_n) e.g. (1, 2) for unigrams + bigrams
    min_df       : ignore terms that appear in fewer than min_df documents
    save         : persist vectorizer to TFIDF_PATH

    Returns
    -------
    (TfidfVectorizer, sparse TF-IDF matrix)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,       # apply 1+log(tf) — helps with long texts
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(corpus)
    print(f"[fit_tfidf] Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"[fit_tfidf] Matrix shape: {matrix.shape}")

    if save:
        joblib.dump(vectorizer, TFIDF_PATH)
        print(f"[fit_tfidf] Vectorizer saved → {TFIDF_PATH}")

    return vectorizer, matrix


# ── Transform ─────────────────────────────────────────────────────────────────

def transform_tfidf(
    vectorizer: TfidfVectorizer,
    texts: list[str] | pd.Series,
) -> spmatrix:
    """
    Transform new texts using a previously fitted vectorizer.
    """
    return vectorizer.transform(texts)


# ── Persist / Load ────────────────────────────────────────────────────────────

def save_vectorizer(vectorizer: TfidfVectorizer, path=TFIDF_PATH) -> None:
    joblib.dump(vectorizer, path)
    print(f"[save_vectorizer] Saved → {path}")


def load_vectorizer(path=TFIDF_PATH) -> TfidfVectorizer:
    if not path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {path}. Train the model first."
        )
    vectorizer = joblib.load(path)
    print(f"[load_vectorizer] Loaded from {path}")
    return vectorizer


# ── Top-terms helper ──────────────────────────────────────────────────────────

def top_terms(
    vectorizer: TfidfVectorizer,
    matrix: spmatrix,
    row_idx: int,
    n: int = 20,
) -> list[str]:
    """
    Return the top-n TF-IDF terms for a given row (document) in the matrix.
    Useful for interpreting what a resume / job is 'about'.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = matrix[row_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:n]
    return feature_names[top_indices].tolist()


def vocabulary_skills(vectorizer: TfidfVectorizer) -> list[str]:
    """Return the full vocabulary list (sorted alphabetically)."""
    return sorted(vectorizer.vocabulary_.keys())
