"""
clustering.py — Job Clustering & Skill-Gap Report (Unsupervised — DISCOVERY + INSIGHT).

Steps:
    1. K-Means clustering on job TF-IDF vectors.
    2. Elbow + Silhouette analysis to pick optimal k.
    3. PCA / t-SNE for 2D visualisation.
    4. Skill-gap report: compare candidate's skills vs target cluster's top skills.

Usage:
    from src.models.clustering import (
        train_kmeans, assign_cluster,
        skill_gap_report, plot_clusters
    )
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.sparse import spmatrix

from src.config import (
    CLUSTER_MODEL_PATH,
    FIGURES_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
    TOP_SKILLS_PER_CLUSTER,
)


# ── Elbow + Silhouette analysis ───────────────────────────────────────────────

def elbow_analysis(
    job_matrix: spmatrix,
    k_range: range = range(2, 20),
    save_fig: bool = True,
) -> dict:
    """
    Compute inertia and silhouette scores for a range of k values.

    Returns dict {'k': [...], 'inertia': [...], 'silhouette': [...]}
    """
    inertias    = []
    silhouettes = []

    print("[elbow_analysis] Testing k values …")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(job_matrix)
        inertias.append(km.inertia_)
        sil = silhouette_score(job_matrix, labels, sample_size=min(5000, job_matrix.shape[0]))
        silhouettes.append(sil)
        print(f"  k={k:2d}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

    results = {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes}

    if save_fig:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(list(k_range), inertias, "bo-")
        ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")
        ax2.plot(list(k_range), silhouettes, "rs-")
        ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Scores")
        plt.tight_layout()
        fig_path = FIGURES_DIR / "elbow_silhouette.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[elbow_analysis] Figure saved → {fig_path}")

    return results


# ── Train K-Means ─────────────────────────────────────────────────────────────

def train_kmeans(
    job_matrix: spmatrix,
    n_clusters: int = N_CLUSTERS,
    save: bool = True,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-Means and return (model, cluster_labels).
    """
    print(f"[train_kmeans] Fitting KMeans with k={n_clusters} …")
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto", max_iter=500)
    labels = km.fit_predict(job_matrix)

    sil = silhouette_score(job_matrix, labels, sample_size=min(5000, job_matrix.shape[0]))
    print(f"[train_kmeans] Silhouette score: {sil:.4f}")

    if save:
        joblib.dump(km, CLUSTER_MODEL_PATH)
        print(f"[train_kmeans] Model saved → {CLUSTER_MODEL_PATH}")

    return km, labels


# ── Assign cluster ────────────────────────────────────────────────────────────

def assign_cluster(resume_vector: spmatrix, km: KMeans) -> int:
    """Return the cluster id closest to `resume_vector`."""
    return int(km.predict(resume_vector)[0])


# ── Top skills per cluster ────────────────────────────────────────────────────

def cluster_top_skills(
    job_matrix: spmatrix,
    labels: np.ndarray,
    vectorizer,
    n_top: int = TOP_SKILLS_PER_CLUSTER,
) -> dict[int, list[str]]:
    """
    For each cluster, compute the mean TF-IDF vector and extract top terms.

    Returns {cluster_id: [skill1, skill2, …]}
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_skills = {}

    dense = job_matrix.toarray()  # may be large; fine for <= 50k jobs
    for cluster_id in np.unique(labels):
        mask        = labels == cluster_id
        mean_vector = dense[mask].mean(axis=0)
        top_idx     = mean_vector.argsort()[::-1][:n_top]
        cluster_skills[int(cluster_id)] = feature_names[top_idx].tolist()

    return cluster_skills


# ── Skill-gap report ──────────────────────────────────────────────────────────

def skill_gap_report(
    resume_text: str,
    target_cluster_id: int,
    cluster_skills: dict[int, list[str]],
) -> dict:
    """
    Compare resume skills vs target cluster's top skills.

    Returns:
        {
          'target_cluster': int,
          'cluster_top_skills': [...],
          'resume_skills_found': [...],
          'missing_skills': [...],
          'match_pct': float,
        }
    """
    from src.features.match_features import extract_skills

    target_skills    = cluster_skills.get(target_cluster_id, [])
    resume_skills    = extract_skills(resume_text)

    found   = [s for s in target_skills if s in resume_skills]
    missing = [s for s in target_skills if s not in resume_skills]
    match_pct = (len(found) / len(target_skills) * 100) if target_skills else 0.0

    return {
        "target_cluster":       target_cluster_id,
        "cluster_top_skills":   target_skills,
        "resume_skills_found":  found,
        "missing_skills":       missing,
        "match_pct":            round(match_pct, 1),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_clusters(
    job_matrix: spmatrix,
    labels: np.ndarray,
    method: str = "pca",
    save: bool = True,
) -> None:
    """
    2D visualisation of job clusters using PCA or t-SNE.

    Parameters
    ----------
    job_matrix : TF-IDF matrix of jobs
    labels     : K-Means cluster labels
    method     : 'pca' (fast) or 'tsne' (slow but better separation)
    save       : save figure to reports/figures/
    """
    print(f"[plot_clusters] Running {method.upper()} …")
    dense = job_matrix.toarray()

    if method == "tsne":
        # PCA first to speed up t-SNE
        pca   = PCA(n_components=50, random_state=RANDOM_STATE)
        dense = pca.fit_transform(dense)
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=RANDOM_STATE)

    coords = reducer.fit_transform(dense)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Job Clusters ({method.upper()})")
    plt.xlabel("Component 1"); plt.ylabel("Component 2")
    plt.tight_layout()

    if save:
        fig_path = FIGURES_DIR / f"clusters_{method}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"[plot_clusters] Figure saved → {fig_path}")
    plt.close()


# ── Load ──────────────────────────────────────────────────────────────────────

def load_cluster_model(path=CLUSTER_MODEL_PATH) -> KMeans:
    if not path.exists():
        raise FileNotFoundError(f"Cluster model not found at {path}. Run train_kmeans() first.")
    return joblib.load(path)
