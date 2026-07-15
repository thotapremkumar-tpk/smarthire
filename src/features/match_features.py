"""
match_features.py — engineered features for the fit/shortlisting predictor.

Computes numeric features that describe how well a resume matches a job:
  - skill_overlap        : Jaccard similarity of skill sets
  - experience_match     : 1 if resume experience >= job requirement, else 0
  - cosine_sim           : TF-IDF cosine similarity score
  - title_match          : 1 if predicted resume category matches job title words

Usage:
    from src.features.match_features import build_match_features
    X = build_match_features(resume_text, job_rows, vectorizer, resume_vector)
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


# ── Skill extraction ──────────────────────────────────────────────────────────

# A lightweight hard-coded skill list; replace / extend as needed.
COMMON_SKILLS = {
    "python", "java", "sql", "javascript", "typescript", "html", "css",
    "react", "angular", "nodejs", "django", "flask", "fastapi",
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "data analysis", "data science", "nlp", "computer vision",
    "docker", "kubernetes", "aws", "azure", "gcp", "git",
    "excel", "tableau", "power bi",
    "communication", "leadership", "project management", "agile", "scrum",
}


def extract_skills(text: str) -> set[str]:
    """Return the set of COMMON_SKILLS found in `text`."""
    text_lower = text.lower()
    return {skill for skill in COMMON_SKILLS if skill in text_lower}


def skill_overlap(resume_text: str, job_text: str) -> float:
    """
    Jaccard similarity between skill sets of resume and job.
    Returns a float in [0, 1].
    """
    resume_skills = extract_skills(resume_text)
    job_skills    = extract_skills(job_text)
    if not resume_skills and not job_skills:
        return 0.0
    intersection = resume_skills & job_skills
    union        = resume_skills | job_skills
    return len(intersection) / len(union)


# ── Experience matching ────────────────────────────────────────────────────────

def _parse_years(text: str) -> int | None:
    """Extract the first integer year-count from a string (e.g. '3+ years' → 3)."""
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else None


def experience_match(resume_text: str, job_experience_str: str) -> float:
    """
    Simple heuristic:
      - parse years required from job description
      - parse years claimed in resume
      - return 1.0 if resume >= requirement, 0.5 if unknown, 0.0 if under
    """
    req = _parse_years(job_experience_str)
    if req is None:
        return 0.5  # unknown requirement

    # Look for "X years" pattern in resume
    matches = re.findall(r"(\d+)\s*(?:\+)?\s*years?", resume_text.lower())
    if not matches:
        return 0.5  # can't parse resume experience
    resume_yrs = max(int(m) for m in matches)

    return 1.0 if resume_yrs >= req else 0.0


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_match_features(
    resume_text: str,
    job_corpus: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    resume_vector: spmatrix,
    job_vectors: spmatrix,
) -> pd.DataFrame:
    """
    Build a feature DataFrame with one row per job in `job_corpus`.

    Features (columns):
        cosine_sim        — TF-IDF cosine similarity
        skill_overlap     — Jaccard skill similarity
        experience_match  — experience heuristic score

    Parameters
    ----------
    resume_text   : raw / cleaned resume string
    job_corpus    : DataFrame with columns [title, description, skills, experience]
    vectorizer    : fitted TfidfVectorizer
    resume_vector : (1, vocab) sparse matrix of the resume
    job_vectors   : (n_jobs, vocab) sparse matrix of all jobs

    Returns
    -------
    pd.DataFrame of shape (n_jobs, 3)
    """
    # Cosine similarity between resume and every job
    cos_sims = cosine_similarity(resume_vector, job_vectors).flatten()

    # Per-job skill overlap and experience match
    skill_scores = []
    exp_scores   = []

    for _, row in job_corpus.iterrows():
        job_text = f"{row.get('title', '')} {row.get('description', '')} {row.get('skills', '')}"
        skill_scores.append(skill_overlap(resume_text, job_text))
        exp_scores.append(experience_match(resume_text, str(row.get("experience", ""))))

    features = pd.DataFrame({
        "cosine_sim":       cos_sims,
        "skill_overlap":    skill_scores,
        "experience_match": exp_scores,
    })

    return features
