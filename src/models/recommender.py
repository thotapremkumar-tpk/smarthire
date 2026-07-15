"""
recommender.py — Content-based Job Recommender (Unsupervised — CORE ENGINE).

Algorithm:
    1. Vectorize all job descriptions with TF-IDF (fit once, save to disk).
    2. At inference: vectorize the resume, compute cosine similarity with every job.
    3. Return the top-N matching jobs with their scores.

Usage:
    from src.models.recommender import build_job_index, recommend_jobs
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix, save_npz, load_npz

from src.config import (
    JOB_VECTORS_PATH,
    TFIDF_PATH,
    TOP_N_JOBS,
    MODELS_DIR,
    JOBS_CLEAN_CSV,
)
from src.features.text_features import fit_tfidf, load_vectorizer, transform_tfidf

_JOB_VECTORS_NPZ = MODELS_DIR / "job_tfidf_matrix.npz"


# ── Build job index ───────────────────────────────────────────────────────────

def build_job_index(
    job_corpus: pd.DataFrame | None = None,
    text_col: str = "clean_text",
    save: bool = True,
):
    """
    Fit a TF-IDF vectorizer on all job descriptions and save the matrix.

    Parameters
    ----------
    job_corpus : output of build_job_corpus() — must have `text_col`
    text_col   : column with cleaned job text
    save       : persist vectorizer + matrix

    Returns
    -------
    (vectorizer, job_matrix)
    """
    # Load from disk if not provided
    if job_corpus is None:
        if not JOBS_CLEAN_CSV.exists():
            raise FileNotFoundError(
                f"Job corpus not found at {JOBS_CLEAN_CSV}. "
                "Run: python -m src.data.preprocess"
            )
        job_corpus = pd.read_csv(JOBS_CLEAN_CSV)

    print("[build_job_index] Fitting TF-IDF on job corpus …")
    vectorizer, job_matrix = fit_tfidf(job_corpus[text_col], save=save)

    if save:
        save_npz(str(_JOB_VECTORS_NPZ), job_matrix)
        print(f"[build_job_index] Job matrix saved → {_JOB_VECTORS_NPZ}")

    return vectorizer, job_matrix


def load_job_index():
    """Load pre-built vectorizer and job matrix from disk."""
    vectorizer  = load_vectorizer(TFIDF_PATH)
    job_matrix  = load_npz(str(_JOB_VECTORS_NPZ))
    print(f"[load_job_index] Loaded job matrix {job_matrix.shape}")
    return vectorizer, job_matrix


# ── Recommend ─────────────────────────────────────────────────────────────────

def recommend_jobs(
    resume_text: str,
    job_corpus: pd.DataFrame,
    vectorizer=None,
    job_matrix: spmatrix | None = None,
    top_n: int = TOP_N_JOBS,
) -> pd.DataFrame:
    """
    Return the top-N jobs most similar to `resume_text`.

    Parameters
    ----------
    resume_text : cleaned resume text string
    job_corpus  : full job DataFrame (output of build_job_corpus)
    vectorizer  : fitted TfidfVectorizer (loaded from disk if None)
    job_matrix  : pre-computed (n_jobs, vocab) sparse matrix (loaded if None)
    top_n       : number of results to return

    Returns
    -------
    pd.DataFrame with columns:
        rank, title, company, location, experience, salary, match_score, description
    """
    if vectorizer is None or job_matrix is None:
        vectorizer, job_matrix = load_job_index()

    # Vectorize resume
    resume_vector = transform_tfidf(vectorizer, [resume_text])

    # Cosine similarities
    scores = cosine_similarity(resume_vector, job_matrix).flatten()

    # Top-N indices
    top_idx = np.argsort(scores)[::-1][:top_n]

    results = job_corpus.iloc[top_idx].copy()
    results["match_score"] = (scores[top_idx] * 100).round(2)
    results["rank"]        = range(1, len(top_idx) + 1)

    cols = ["rank", "title", "company", "location", "experience",
            "salary", "match_score", "description"]
    available_cols = [c for c in cols if c in results.columns]

    return results[available_cols].reset_index(drop=True)
