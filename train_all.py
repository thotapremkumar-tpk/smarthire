"""
train_all.py — SmartHire Master Training Script (Optimised)
============================================================
Run from project root:
    python train_all.py
"""

import sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import nltk
for r in ["stopwords", "punkt", "punkt_tab"]:
    nltk.download(r, quiet=True)

from src.config import (
    RESUMES_CLEAN_CSV, JOBS_CLEAN_CSV, MODELS_DIR, FIGURES_DIR,
    CLASSIFIER_PATH, TFIDF_PATH, CLUSTER_MODEL_PATH, FIT_PREDICTOR_PATH,
    RANDOM_STATE,
)

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}\n")
def save_fig(name):
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✅  Saved → reports/figures/{name}")

# ══════════════════════════════════════════════════════════════
# STEP 0 – Load data
# ══════════════════════════════════════════════════════════════
section("STEP 0 — Load Preprocessed Data")

resumes_df = pd.read_csv(RESUMES_CLEAN_CSV)
jobs_df    = pd.read_csv(JOBS_CLEAN_CSV)

print(f"  Resumes : {resumes_df.shape}  |  categories : {resumes_df['Category'].nunique()}")
print(f"  Jobs    : {jobs_df.shape}")

# ══════════════════════════════════════════════════════════════
# STEP 1 – EDA Plots
# ══════════════════════════════════════════════════════════════
section("STEP 1 — EDA Visualizations")

# 1a. Resume category distribution
plt.figure(figsize=(12, 6))
counts = resumes_df["Category"].value_counts()
sns.barplot(x=counts.values, y=counts.index, palette="viridis")
plt.xlabel("Number of Resumes")
plt.title("Resume Category Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("01a_resume_category_distribution.png")

# 1b. Top job titles
plt.figure(figsize=(10, 6))
top_titles = jobs_df["title"].fillna("Unknown").value_counts().head(20)
sns.barplot(x=top_titles.values, y=top_titles.index, palette="coolwarm")
plt.xlabel("Count")
plt.title("Top 20 Job Titles in Corpus", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("01b_top_job_titles.png")

# 1c. Resume word count distribution
resumes_df["resume_len"] = resumes_df["clean_resume"].str.split().str.len()
plt.figure(figsize=(10, 4))
plt.hist(resumes_df["resume_len"].clip(upper=600), bins=40, color="#6C63FF", edgecolor="white")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.title("Resume Text Length Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("01c_resume_length_distribution.png")

# 1d. Experience level distribution (if available)
if "experience" in jobs_df.columns:
    plt.figure(figsize=(10, 5))
    exp_counts = jobs_df["experience"].fillna("Unknown").value_counts().head(12)
    sns.barplot(x=exp_counts.values, y=exp_counts.index, palette="magma")
    plt.xlabel("Number of Jobs")
    plt.title("Experience Levels in Job Corpus", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("01d_job_experience_levels.png")

print("  EDA plots done.")

# ══════════════════════════════════════════════════════════════
# STEP 2 – Classifier  (LR + SVM only — RF skipped for speed)
# ══════════════════════════════════════════════════════════════
section("STEP 2 — Resume Category Classifier (MODEL A)")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

X = resumes_df["clean_resume"].fillna("").values
y = resumes_df["Category"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

tfidf_p = dict(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

candidates = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_p)),
        ("clf",   LogisticRegression(max_iter=1000, C=5.0, random_state=RANDOM_STATE)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_p)),
        ("clf",   LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ]),
}

cv_results = {}
print(f"  5-fold CV on {len(X_tr)} samples …\n")
for name, pipe in candidates.items():
    t0 = time.time()
    scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"  {name:25s}  CV Acc: {scores.mean():.4f} ± {scores.std():.4f}  ({time.time()-t0:.1f}s)")

best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
best_pipe  = candidates[best_name]
best_pipe.fit(X_tr, y_tr)
y_pred = best_pipe.predict(X_te)

acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, average="weighted")
print(f"\n  ➡  Best: {best_name}  Test Acc={acc:.4f}  F1={f1:.4f}")
print(f"\n{classification_report(y_te, y_pred)}")

# Save
joblib.dump(best_pipe, CLASSIFIER_PATH)
joblib.dump(best_pipe.named_steps["tfidf"], TFIDF_PATH)
print(f"  ✅  Saved → models/classifier.pkl  ({best_name})")
print(f"  ✅  Saved → models/tfidf_vectorizer.pkl")

# 2a. CV comparison bar chart
plt.figure(figsize=(7, 4))
names  = list(cv_results.keys())
means  = [cv_results[n]["mean"] for n in names]
stds   = [cv_results[n]["std"]  for n in names]
colors = ["#6C63FF" if n == best_name else "#B0B0B0" for n in names]
plt.bar(names, means, yerr=stds, color=colors, capsize=6, edgecolor="white", width=0.4)
plt.ylabel("CV Accuracy")
plt.ylim(0.7, 1.0)
plt.title("Classifier CV Accuracy Comparison", fontsize=13, fontweight="bold")
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=11)
plt.tight_layout()
save_fig("02a_classifier_cv_comparison.png")

# 2b. Confusion matrix
classes_sorted = sorted(set(y))
cm = confusion_matrix(y_te, y_pred, labels=classes_sorted)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes_sorted, yticklabels=classes_sorted,
            linewidths=0.5)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
save_fig("02b_confusion_matrix.png")

# 2c. Per-class F1
rep = classification_report(y_te, y_pred, output_dict=True)
cls_f1 = {k: v["f1-score"] for k, v in rep.items()
          if k not in ["accuracy","macro avg","weighted avg"]}
sc = sorted(cls_f1, key=cls_f1.get)
plt.figure(figsize=(10, 8))
plt.barh(sc, [cls_f1[c] for c in sc], color="#6C63FF", edgecolor="white")
plt.xlabel("F1 Score")
plt.xlim(0, 1.05)
plt.axvline(x=f1, color="red", linestyle="--", alpha=0.7, label=f"Weighted avg F1={f1:.3f}")
plt.title("Per-Class F1 Score", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
save_fig("02c_per_class_f1.png")

# ══════════════════════════════════════════════════════════════
# STEP 3 – Job Recommender (TF-IDF index, sub-sampled)
# ══════════════════════════════════════════════════════════════
section("STEP 3 — Job Recommender (TF-IDF Index)")

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz

# Sub-sample to 30K for manageable index size
IDX_SIZE = min(30000, len(jobs_df))
job_sample = jobs_df.sample(IDX_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"  Fitting TF-IDF on {IDX_SIZE:,} jobs …")
job_vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
job_matrix = job_vec.fit_transform(job_sample["clean_text"].fillna(""))
print(f"  Job TF-IDF matrix : {job_matrix.shape}")

save_npz(str(MODELS_DIR / "job_tfidf_matrix.npz"), job_matrix)
joblib.dump(job_vec,      MODELS_DIR / "job_tfidf_vectorizer.pkl")
joblib.dump(job_sample,   MODELS_DIR / "job_corpus_sample.pkl")
print(f"  ✅  Saved → models/job_tfidf_matrix.npz")
print(f"  ✅  Saved → models/job_tfidf_vectorizer.pkl")

# Qualitative check
sample_res  = resumes_df["clean_resume"].iloc[0]
sample_cat  = resumes_df["Category"].iloc[0]
res_vec     = job_vec.transform([sample_res])
scores_all  = cosine_similarity(res_vec, job_matrix).flatten()
top10_idx   = np.argsort(scores_all)[::-1][:10]

print(f"\n  Sample resume: {sample_cat}")
print("  Top 5 recommended jobs:")
for rank, idx in enumerate(top10_idx[:5], 1):
    row = job_sample.iloc[idx]
    print(f"  {rank}. [{scores_all[idx]*100:.1f}%] {row.get('title','N/A')} @ {row.get('company','N/A')}")

# 3a. Match score distribution
plt.figure(figsize=(10, 4))
plt.hist(scores_all * 100, bins=60, color="#FF6584", edgecolor="white")
plt.xlabel("Cosine Similarity Score (%)")
plt.ylabel("Number of Jobs")
plt.title("Distribution of Match Scores (Sample Resume)", fontsize=13, fontweight="bold")
best_score = scores_all[top10_idx[0]] * 100
plt.axvline(x=best_score, color="gold", linestyle="--", lw=2, label=f"Best: {best_score:.1f}%")
plt.legend()
plt.tight_layout()
save_fig("03a_match_score_distribution.png")

# 3b. Top 10 matches bar chart
top10_scores = scores_all[top10_idx] * 100
top10_titles = [str(job_sample.iloc[i].get("title", "N/A"))[:35] for i in top10_idx]
bar_colors   = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
plt.figure(figsize=(10, 6))
plt.barh(top10_titles[::-1], top10_scores[::-1], color=bar_colors, edgecolor="white")
plt.xlabel("Match Score (%)")
plt.title(f"Top 10 Job Recommendations\n(Resume category: {sample_cat})", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("03b_top10_recommendations.png")

# ══════════════════════════════════════════════════════════════
# STEP 4 – Clustering (sub-sampled, k=2..10)
# ══════════════════════════════════════════════════════════════
section("STEP 4 — Job Clustering (K-Means)")

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

CLUST_N = min(8000, job_matrix.shape[0])
rng     = np.random.RandomState(RANDOM_STATE)
cidx    = rng.choice(job_matrix.shape[0], CLUST_N, replace=False)
mat_c   = job_matrix[cidx]
df_c    = job_sample.iloc[cidx].reset_index(drop=True)

print(f"  Clustering on {CLUST_N:,} jobs, k=2..10 …")

k_range    = range(2, 11)
inertias   = []
silhouettes = []

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5, max_iter=100)
    lbl = km.fit_predict(mat_c)
    inertias.append(km.inertia_)
    sil_n = min(2000, CLUST_N)
    sil_i = rng.choice(CLUST_N, sil_n, replace=False)
    sil   = silhouette_score(mat_c[sil_i], lbl[sil_i])
    silhouettes.append(sil)
    print(f"    k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

# Elbow + silhouette plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, "bo-", lw=2)
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method", fontsize=13, fontweight="bold")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), silhouettes, "rs-", lw=2)
best_k = list(k_range)[int(np.argmax(silhouettes))]
axes[1].axvline(x=best_k, color="green", linestyle="--", label=f"Best k={best_k}")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs k", fontsize=13, fontweight="bold")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Choosing Optimal Number of Clusters", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("04a_elbow_silhouette.png")

# Final clustering with best_k
print(f"\n  Final K-Means with k={best_k} …")
final_km  = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
final_lbl = final_km.fit_predict(mat_c)
df_c      = df_c.copy()
df_c["cluster"] = final_lbl
joblib.dump(final_km, CLUSTER_MODEL_PATH)
print(f"  ✅  Saved → models/cluster_model.pkl  (k={best_k})")

# PCA 2D scatter
svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
coords = svd.fit_transform(mat_c)
plt.figure(figsize=(12, 8))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=final_lbl,
                 cmap="tab10", alpha=0.35, s=6)
plt.colorbar(sc, label="Cluster")
plt.xlabel("SVD Component 1"); plt.ylabel("SVD Component 2")
plt.title(f"Job Clusters — 2D Projection (k={best_k})", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("04b_cluster_pca_2d.png")

# Cluster sizes
sizes = pd.Series(final_lbl).value_counts().sort_index()
plt.figure(figsize=(9, 4))
colors_c = plt.cm.tab10(np.linspace(0, 1, best_k))
plt.bar(sizes.index.astype(str), sizes.values, color=colors_c, edgecolor="white")
plt.xlabel("Cluster ID"); plt.ylabel("Number of Jobs")
plt.title(f"Cluster Size Distribution (k={best_k})", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("04c_cluster_sizes.png")

# Print top titles per cluster
print("\n  Top titles per cluster:")
for cid in range(best_k):
    top = df_c[df_c["cluster"] == cid]["title"].dropna().value_counts().head(3).index.tolist()
    print(f"  Cluster {cid}: {', '.join(top)}")

# ══════════════════════════════════════════════════════════════
# STEP 5 – Fit Predictor
# ══════════════════════════════════════════════════════════════
section("STEP 5 — Fit Predictor / Shortlisting (MODEL B)")

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score, roc_curve, classification_report as cr

print("  Building resume-job pair features …")
np.random.seed(RANDOM_STATE)
N_PAIRS = 4000
pairs   = []

for i in range(N_PAIRS):
    ri   = i % len(resumes_df)
    rrow = resumes_df.iloc[ri]
    rtxt = rrow["clean_resume"]
    rcat = rrow["Category"]

    cat_kw  = rcat.lower().split()[0]
    pos_pool = job_sample[job_sample["title"].str.lower().str.contains(cat_kw, na=False)]
    if len(pos_pool) == 0:
        pos_pool = job_sample

    pos_job = pos_pool.sample(1, random_state=i).iloc[0]
    neg_job = job_sample.sample(1, random_state=i + 50000).iloc[0]

    rwords = set(str(rtxt).lower().split())
    for job, lbl in [(pos_job, 1), (neg_job, 0)]:
        jwords = set(str(job.get("clean_text","")).lower().split())
        skwords= set(str(job.get("skills","")).lower().split())
        pairs.append({
            "jaccard":        len(rwords & jwords) / (len(rwords | jwords) + 1e-9),
            "skill_overlap":  len(rwords & skwords) / (len(skwords) + 1e-9),
            "resume_len":     len(rtxt.split()) / 1000.0,
            "has_exp_kw":     int(any(w in rtxt.lower() for w in ["years","experience","worked"])),
            "title_match":    int(cat_kw in str(job.get("title","")).lower()),
            "label":          lbl,
        })

fit_df = pd.DataFrame(pairs)
print(f"  {len(fit_df):,} pairs  |  class balance: {fit_df['label'].mean():.2f}")

feats  = ["jaccard","skill_overlap","resume_len","has_exp_kw","title_match"]
Xf     = fit_df[feats].values
yf     = fit_df["label"].values

Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.2,
                                                random_state=RANDOM_STATE, stratify=yf)
fit_results = {}

# Logistic Regression
lr = LR(max_iter=500, random_state=RANDOM_STATE)
lr.fit(Xf_tr, yf_tr)
lr_p  = lr.predict_proba(Xf_te)[:, 1]
lr_auc = roc_auc_score(yf_te, lr_p)
fit_results["Logistic Regression"] = {"model": lr, "proba": lr_p, "auc": lr_auc}
print(f"  Logistic Regression  ROC-AUC: {lr_auc:.4f}")

# XGBoost
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                         random_state=RANDOM_STATE, eval_metric="logloss",
                         verbosity=0)
    xgb.fit(Xf_tr, yf_tr)
    xgb_p  = xgb.predict_proba(Xf_te)[:, 1]
    xgb_auc = roc_auc_score(yf_te, xgb_p)
    fit_results["XGBoost"] = {"model": xgb, "proba": xgb_p, "auc": xgb_auc}
    print(f"  XGBoost              ROC-AUC: {xgb_auc:.4f}")
except ImportError:
    print("  XGBoost not available — skipping.")

best_fit = max(fit_results, key=lambda k: fit_results[k]["auc"])
joblib.dump(fit_results[best_fit]["model"], FIT_PREDICTOR_PATH)
print(f"  ✅  Saved → models/fit_predictor.pkl  ({best_fit})")

# ROC curves
plt.figure(figsize=(8, 6))
for name, info in fit_results.items():
    fpr, tpr, _ = roc_curve(yf_te, info["proba"])
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={info['auc']:.3f})")
plt.plot([0,1],[0,1],"k--",alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Fit Predictor", fontsize=13, fontweight="bold")
plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout()
save_fig("05a_fit_predictor_roc.png")

# Feature importance
plt.figure(figsize=(8, 4))
coefs = np.abs(lr.coef_[0])
si    = np.argsort(coefs)
plt.barh([feats[i] for i in si], coefs[si], color="#6C63FF", edgecolor="white")
plt.xlabel("|Coefficient|")
plt.title("Fit Predictor Feature Importance (LR)", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("05b_fit_feature_importance.png")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("✅  ALL DONE — SUMMARY")

best_sil = max(silhouettes)
print(f"  MODELS")
print(f"    classifier.pkl          ← {best_name}  Acc={acc:.4f}  F1={f1:.4f}")
print(f"    tfidf_vectorizer.pkl    ← TF-IDF (resumes)")
print(f"    job_tfidf_matrix.npz    ← {job_matrix.shape[0]:,} jobs × {job_matrix.shape[1]:,} features")
print(f"    job_tfidf_vectorizer.pkl")
print(f"    cluster_model.pkl       ← KMeans k={best_k}  silhouette={best_sil:.4f}")
print(f"    fit_predictor.pkl       ← {best_fit}  AUC={fit_results[best_fit]['auc']:.4f}")
print()
print(f"  FIGURES ({len(list(FIGURES_DIR.glob('*.png')))} charts in reports/figures/)")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"    {f.name}")
print()
print("  🚀  Run the app:  streamlit run app/streamlit_app.py")
