"""
streamlit_app.py — SmartHire Web Portal (Error-Free Version)

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import (
    CLASSIFIER_PATH, CLUSTER_MODEL_PATH,
    JOBS_CLEAN_CSV, MODELS_DIR, TOP_N_JOBS,
)
from src.parsing.resume_parser import parse_resume
from src.data.preprocess import clean_text as clean_text_fn

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartHire — AI Job Portal",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
        border-radius: 20px;
        border: 1px solid rgba(168,85,247,0.3);
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p { font-size: 1.1rem; color: #94a3b8; margin-top: 0.5rem; }

    .score-badge {
        display: inline-block; padding: 0.3rem 0.8rem;
        border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .score-high   { background:rgba(52,211,153,0.2);  color:#34d399; border:1px solid #34d399; }
    .score-medium { background:rgba(251,191,36,0.2);  color:#fbbf24; border:1px solid #fbbf24; }
    .score-low    { background:rgba(239,68,68,0.2);   color:#ef4444; border:1px solid #ef4444; }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.5rem; font-weight: 600; font-size: 1rem;
        transition: all 0.3s ease; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99,102,241,0.4);
    }
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a855f7); }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05); border-radius: 12px;
        padding: 1rem; border: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 8px 8px 0 0; color: white;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(99,102,241,0.4);
        border-radius: 12px; padding: 1rem;
    }
    h2, h3 { color: #e2e8f0; }
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ── Load all models once (cached) ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all pre-trained models. Returns a dict with None for missing files."""
    m = {}

    # 1. Resume category classifier
    m["classifier"] = joblib.load(CLASSIFIER_PATH) if CLASSIFIER_PATH.exists() else None

    # 2. Job TF-IDF vectorizer (8000 features — used for recommender + clustering)
    job_vec_path    = MODELS_DIR / "job_tfidf_vectorizer.pkl"
    job_npz_path    = MODELS_DIR / "job_tfidf_matrix.npz"
    job_corpus_path = MODELS_DIR / "job_corpus_sample.pkl"

    if job_vec_path.exists() and job_npz_path.exists():
        from scipy.sparse import load_npz
        m["job_vectorizer"] = joblib.load(job_vec_path)          # vocab=8000
        m["job_matrix"]     = load_npz(str(job_npz_path))        # shape (30000, 8000)
        m["job_corpus"]     = (joblib.load(job_corpus_path)
                               if job_corpus_path.exists() else None)
    else:
        m["job_vectorizer"] = None
        m["job_matrix"]     = None
        m["job_corpus"]     = None

    # 3. Cluster model
    m["cluster_model"] = (joblib.load(CLUSTER_MODEL_PATH)
                          if CLUSTER_MODEL_PATH.exists() else None)

    return m


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 60:   return "score-high"
    if score >= 35:   return "score-medium"
    return "score-low"


def _top_skills_for_cluster(cluster_id: int, job_corpus_df: pd.DataFrame,
                             cluster_labels: np.ndarray,
                             vectorizer, n_top: int = 15) -> list[str]:
    """
    Extract top TF-IDF terms for a given cluster from the corpus text.
    Safe: operates only on the sub-sample that aligns with cluster_labels.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer as TV

    mask = cluster_labels == cluster_id
    texts = job_corpus_df["clean_text"].fillna("").values[mask]
    if len(texts) == 0:
        return []

    # Re-vectorize just the cluster's texts to get top terms
    tv = TV(max_features=500, stop_words="english", ngram_range=(1, 1))
    try:
        tv.fit_transform(texts)
        names = tv.get_feature_names_out()
        return names[:n_top].tolist()
    except Exception:
        return []


# ── UI: Hero ──────────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🎯 SmartHire</h1>
        <p>Upload your CV — get matched jobs, a fit score, and a personalised skill-gap report.</p>
    </div>
    """, unsafe_allow_html=True)


# ── UI: Upload ────────────────────────────────────────────────────────────────

def render_upload() -> str | None:
    st.markdown("### 📄 Upload Your Resume")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drop your CV (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("""
        <div class="glass-card" style="padding:1rem;text-align:center;">
            <p style="color:#94a3b8;font-size:0.85rem;margin:0;">
            ✅ PDF supported<br>✅ DOCX supported<br>✅ TXT supported<br>🔒 Processed locally
            </p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is None:
        return None

    with st.spinner("📖 Parsing resume …"):
        try:
            raw_bytes = uploaded.read()
            ext       = uploaded.name.rsplit(".", 1)[-1].lower()
            raw_text  = parse_resume(raw_bytes, file_type=ext)
            cleaned   = clean_text_fn(raw_text)

            if not cleaned.strip():
                st.error("⚠️ Could not extract text — please try a different file.")
                return None

            st.success(f"✅ Resume parsed — {len(raw_text.split()):,} words extracted.")
            with st.expander("📝 Preview extracted text"):
                st.text_area("", raw_text[:2000] + ("…" if len(raw_text) > 2000 else ""),
                             height=200, disabled=True)
            return cleaned

        except Exception as e:
            st.error(f"❌ Error parsing file: {e}")
            return None


# ── UI: Category Tab ──────────────────────────────────────────────────────────

def render_category(clean_resume: str, models: dict):
    clf = models.get("classifier")
    if clf is None:
        st.warning("⚠️ Classifier not found. Run `python train_all.py` first.")
        return None

    try:
        predicted = clf.predict([clean_resume])[0]
    except Exception as e:
        st.error(f"❌ Classifier error: {e}")
        return None

    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin:0 0 0.5rem 0;">🏷️ Predicted Job Category</h3>
        <span style="font-size:1.5rem;font-weight:700;color:#a78bfa;">{predicted}</span>
    </div>
    """, unsafe_allow_html=True)

    # Probability chart (only for models with predict_proba)
    if hasattr(clf, "predict_proba"):
        try:
            proba  = clf.predict_proba([clean_resume])[0]
            labels = clf.classes_
            top5   = sorted(zip(labels, proba), key=lambda x: -x[1])[:5]

            fig = go.Figure(go.Bar(
                x=[p * 100 for _, p in top5],
                y=[l for l, _ in top5],
                orientation="h",
                marker=dict(color=[p * 100 for _, p in top5],
                            colorscale=[[0, "#6366f1"], [1, "#a855f7"]]),
                text=[f"{p*100:.1f}%" for _, p in top5],
                textposition="outside",
            ))
            fig.update_layout(
                title="Top-5 Category Probabilities",
                xaxis_title="Probability (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass  # Chart is optional — skip silently

    return predicted


# ── UI: Job Matches Tab ───────────────────────────────────────────────────────

def render_recommendations(clean_resume: str, models: dict, top_n: int = TOP_N_JOBS):
    st.markdown("### 💼 Top Job Matches")

    vectorizer = models.get("job_vectorizer")
    job_matrix = models.get("job_matrix")
    job_corpus = models.get("job_corpus")

    if vectorizer is None or job_matrix is None:
        st.warning("⚠️ Job index not found. Run `python train_all.py` first.")
        return

    if job_corpus is None:
        st.warning("⚠️ Job corpus sample not found. Run `python train_all.py` first.")
        return

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Both resume_vec and job_matrix now use the SAME 8000-feature vocabulary
        resume_vec = vectorizer.transform([clean_resume])          # (1, 8000)
        scores     = cosine_similarity(resume_vec, job_matrix).flatten()  # (30000,)
        top_idx    = np.argsort(scores)[::-1][:top_n]

        # job_corpus has exactly 30K rows — same as job_matrix
        results = job_corpus.iloc[top_idx].copy().reset_index(drop=True)
        results["match_score"] = (scores[top_idx] * 100).round(1)
        results["rank"]        = range(1, len(top_idx) + 1)

    except Exception as e:
        st.error(f"❌ Recommender error: {e}")
        return

    for _, row in results.iterrows():
        sc      = float(row.get("match_score", 0))
        badge   = score_class(sc)
        title   = str(row.get("title",       "Unknown Role"))
        company = str(row.get("company",     ""))
        loc     = str(row.get("location",    ""))
        exp     = str(row.get("experience",  ""))
        desc    = str(row.get("description", ""))[:300]

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                <div>
                    <span style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">
                        #{int(row['rank'])}&nbsp; {title}
                    </span><br/>
                    <span style="color:#94a3b8;font-size:0.85rem;">
                        {"🏢 " + company if company and company != "nan" else ""}
                        {"&nbsp;|&nbsp; 📍 " + loc if loc and loc != "nan" else ""}
                        {"&nbsp;|&nbsp; ⏱ " + exp if exp and exp != "nan" else ""}
                    </span>
                </div>
                <span class="score-badge {badge}">{sc}% Match</span>
            </div>
            <p style="margin-top:0.75rem;color:#94a3b8;font-size:0.9rem;line-height:1.5;">
                {desc}{"…" if len(str(row.get("description", ""))) > 300 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── UI: Skill Gap Tab ─────────────────────────────────────────────────────────

def render_skill_gap(clean_resume: str, models: dict):
    st.markdown("### 🛠️ Skill-Gap Report")

    cluster_model = models.get("cluster_model")
    vectorizer    = models.get("job_vectorizer")
    job_corpus    = models.get("job_corpus")

    if cluster_model is None or vectorizer is None:
        st.warning("⚠️ Cluster model not found. Run `python train_all.py` first.")
        return

    try:
        # Transform resume with the JOB vectorizer (8000 features)
        resume_vec = vectorizer.transform([clean_resume])  # (1, 8000)

        # Predict which cluster this resume belongs to
        cluster_id = int(cluster_model.predict(resume_vec)[0])

        # Get cluster labels by predicting on the job_matrix
        # This is safe because cluster_model.predict works on any sized matrix
        if job_corpus is not None:
            job_matrix  = models.get("job_matrix")
            # Predict labels for the 30K job matrix (not stored labels from 8K training)
            cluster_labels = cluster_model.predict(job_matrix)  # shape (30000,)
            cluster_skills = _top_skills_for_cluster(
                cluster_id, job_corpus, cluster_labels, vectorizer
            )
        else:
            cluster_skills = []

    except Exception as e:
        st.error(f"❌ Skill-gap error: {e}")
        return

    # Get resume skills (simple word extraction)
    resume_words = set(clean_resume.lower().split())
    found   = [s for s in cluster_skills if s.lower() in resume_words]
    missing = [s for s in cluster_skills if s.lower() not in resume_words]
    match_pct = round(len(found) / len(cluster_skills) * 100, 1) if cluster_skills else 0.0

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Cluster",  f"#{cluster_id}")
    col2.metric("Skills Found",    f"{len(found)}/{len(cluster_skills)}")
    col3.metric("Skill Match",     f"{match_pct}%")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=match_pct,
        title={"text": "Skill Match %", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis":    {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":     {"color": "#a855f7"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,0.2)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.2)"},
                {"range": [70,100], "color": "rgba(52,211,153,0.2)"},
            ],
        },
        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 40}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        height=280, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Skills You Have")
        if found:
            badges = " ".join(
                f'<span style="background:rgba(52,211,153,0.15);color:#34d399;'
                f'padding:0.2rem 0.6rem;border-radius:999px;margin:0.2rem;'
                f'display:inline-block;font-size:0.85rem;">{s}</span>'
                for s in found
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No matching skills detected — try adding more technical keywords to your CV.")

    with col_b:
        st.markdown("#### ❌ Skills to Develop")
        if missing:
            badges = " ".join(
                f'<span style="background:rgba(239,68,68,0.15);color:#ef4444;'
                f'padding:0.2rem 0.6rem;border-radius:999px;margin:0.2rem;'
                f'display:inline-block;font-size:0.85rem;">{s}</span>'
                for s in missing
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.success("🎉 You have all the top skills for this job cluster!")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> int:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
            <span style="font-size:2rem;">🎯</span>
            <h2 style="margin:0.5rem 0 0;color:#a78bfa;">SmartHire</h2>
            <p style="color:#64748b;font-size:0.8rem;margin:0;">ML Job Portal</p>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1);">
        """, unsafe_allow_html=True)

        st.markdown("#### 🤖 Model Status")
        checks = {
            "Resume Classifier":  CLASSIFIER_PATH.exists(),
            "Job Recommender":    (MODELS_DIR / "job_tfidf_matrix.npz").exists(),
            "Job Vectorizer":     (MODELS_DIR / "job_tfidf_vectorizer.pkl").exists(),
            "Cluster Model":      CLUSTER_MODEL_PATH.exists(),
            "Job Corpus":         (MODELS_DIR / "job_corpus_sample.pkl").exists(),
        }
        for name, ready in checks.items():
            icon  = "✅" if ready else "⚠️"
            color = "#34d399" if ready else "#fbbf24"
            st.markdown(f'<p style="color:{color};margin:0.3rem 0;">{icon} {name}</p>',
                        unsafe_allow_html=True)

        if not all(checks.values()):
            st.markdown("""
            <div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);
                        border-radius:10px;padding:0.8rem;margin-top:1rem;
                        font-size:0.83rem;color:#fbbf24;">
            ⚠️ Missing models.<br><br>
            Run from project root:<br>
            <code>python train_all.py</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Settings")
        top_n = st.slider("Top N Jobs", min_value=5, max_value=20, value=TOP_N_JOBS)
        return top_n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    top_n = render_sidebar()
    render_hero()

    with st.spinner("⏳ Loading models …"):
        models = load_models()

    clean_resume = render_upload()

    if clean_resume:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["🏷️ Category", "💼 Job Matches", "🛠️ Skill Gap"])

        with tab1:
            render_category(clean_resume, models)

        with tab2:
            render_recommendations(clean_resume, models, top_n=top_n)

        with tab3:
            render_skill_gap(clean_resume, models)

    else:
        st.markdown("### ✨ What SmartHire Can Do For You")
        cols = st.columns(3)
        features = [
            ("🏷️", "Resume Classifier",   "Automatically identifies your job domain using ML."),
            ("💼", "Job Recommender",     "Finds the top matching jobs from 30,000+ listings."),
            ("🛠️", "Skill-Gap Report",    "Shows exactly which skills you need to develop next."),
        ]
        for col, (icon, title, desc) in zip(cols, features):
            col.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:2rem 1rem;">
                <div style="font-size:2.5rem;">{icon}</div>
                <h3 style="margin:0.5rem 0;color:#e2e8f0;">{title}</h3>
                <p style="color:#94a3b8;font-size:0.9rem;line-height:1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
