"""
streamlit_app.py — SmartHire Web Portal

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.config import (
    CLASSIFIER_PATH, TFIDF_PATH, CLUSTER_MODEL_PATH,
    JOBS_CLEAN_CSV, MODELS_DIR, TOP_N_JOBS,
)
from src.parsing.resume_parser import parse_resume
from src.data.preprocess import clean_text

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

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* Hero banner */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
        border-radius: 20px;
        border: 1px solid rgba(168,85,247,0.3);
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    /* Match score badge */
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-high   { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
    .score-medium { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
    .score-low    { background: rgba(239,68,68,0.2);   color: #ef4444; border: 1px solid #ef4444; }

    /* Sidebar */
    .css-1d391kg { background: rgba(15,12,41,0.8) !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99,102,241,0.4);
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a855f7); }

    /* Metric */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Tab active */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 8px 8px 0 0;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(99,102,241,0.4);
        border-radius: 12px;
        padding: 1rem;
    }

    h2, h3 { color: #e2e8f0; }
    hr { border-color: rgba(255,255,255,0.1); }
</style>
""", unsafe_allow_html=True)


# ── Helper: load models ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all pre-trained models and job corpus."""
    models = {}

    # Classifier
    if CLASSIFIER_PATH.exists():
        models["classifier"] = joblib.load(CLASSIFIER_PATH)
    else:
        models["classifier"] = None

    # TF-IDF vectorizer + job matrix
    npz_path = MODELS_DIR / "job_tfidf_matrix.npz"
    if TFIDF_PATH.exists() and npz_path.exists():
        from scipy.sparse import load_npz
        models["vectorizer"]  = joblib.load(TFIDF_PATH)
        models["job_matrix"]  = load_npz(str(npz_path))
    else:
        models["vectorizer"]  = None
        models["job_matrix"]  = None

    # Cluster model
    if CLUSTER_MODEL_PATH.exists():
        models["cluster_model"] = joblib.load(CLUSTER_MODEL_PATH)
    else:
        models["cluster_model"] = None

    return models


@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame | None:
    """Load the processed job corpus CSV."""
    if JOBS_CLEAN_CSV.exists():
        return pd.read_csv(JOBS_CLEAN_CSV)
    return None


# ── Score colour helper ────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 60:
        return "score-high"
    elif score >= 35:
        return "score-medium"
    return "score-low"


# ── UI Sections ───────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero">
        <h1>🎯 SmartHire</h1>
        <p>Upload your CV — get matched jobs, a fit score, and a personalised skill-gap report powered by ML.</p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_section() -> str | None:
    """File uploader — returns cleaned resume text or None."""
    st.markdown("### 📄 Upload Your Resume")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Drop your CV here (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("""
        <div class="glass-card" style="padding:1rem; text-align:center;">
            <p style="color:#94a3b8; font-size:0.85rem; margin:0;">
            ✅ PDF supported<br>
            ✅ DOCX supported<br>
            ✅ TXT supported<br>
            🔒 Processed locally
            </p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("📖 Parsing your resume …"):
            try:
                raw_bytes = uploaded_file.read()
                file_type = uploaded_file.name.split(".")[-1]
                raw_text  = parse_resume(raw_bytes, file_type=file_type)
                clean     = clean_text(raw_text)

                if not clean.strip():
                    st.error("⚠️ Could not extract text. Please try a different file.")
                    return None

                st.success(f"✅ Resume parsed — {len(raw_text.split()):,} words extracted.")
                with st.expander("📝 Preview extracted text"):
                    st.text_area("", raw_text[:2000] + ("…" if len(raw_text) > 2000 else ""),
                                 height=200, disabled=True)

                return clean

            except Exception as e:
                st.error(f"❌ Error parsing file: {e}")
                return None
    return None


def render_category(clean_text_val: str, models: dict):
    """Predict and display resume category."""
    classifier = models.get("classifier")
    if classifier is None:
        st.warning("⚠️ Classifier not trained yet. Run notebook 02 first.")
        return

    with st.spinner("🔍 Classifying resume …"):
        predicted = classifier.predict([clean_text_val])[0]

    st.markdown(f"""
    <div class="glass-card">
        <h3 style="margin:0 0 0.5rem 0;">🏷️ Predicted Job Category</h3>
        <span style="font-size:1.5rem; font-weight:700; color:#a78bfa;">{predicted}</span>
    </div>
    """, unsafe_allow_html=True)

    # Show probabilities if available
    if hasattr(classifier, "predict_proba"):
        try:
            proba  = classifier.predict_proba([clean_text_val])[0]
            labels = classifier.classes_
            top5   = sorted(zip(labels, proba), key=lambda x: -x[1])[:5]

            fig = go.Figure(go.Bar(
                x=[p * 100 for _, p in top5],
                y=[l for l, _ in top5],
                orientation="h",
                marker=dict(
                    color=[p * 100 for _, p in top5],
                    colorscale=[[0, "#6366f1"], [1, "#a855f7"]],
                ),
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
            pass

    return predicted


def render_recommendations(clean_resume: str, models: dict, corpus: pd.DataFrame | None):
    """Display top-N job recommendations."""
    st.markdown("### 💼 Top Job Matches")

    vectorizer = models.get("vectorizer")
    job_matrix = models.get("job_matrix")

    if vectorizer is None or job_matrix is None or corpus is None:
        st.warning("⚠️ Job index not built yet. Run notebook 03 first.")
        return

    with st.spinner("🔎 Finding best matches …"):
        from sklearn.metrics.pairwise import cosine_similarity
        resume_vec = vectorizer.transform([clean_resume])
        scores     = cosine_similarity(resume_vec, job_matrix).flatten()
        top_idx    = np.argsort(scores)[::-1][:TOP_N_JOBS]
        results    = corpus.iloc[top_idx].copy()
        results["match_score"] = (scores[top_idx] * 100).round(1)
        results["rank"]        = range(1, len(top_idx) + 1)

    for _, row in results.iterrows():
        sc    = float(row.get("match_score", 0))
        badge = score_class(sc)
        title   = row.get("title", "Unknown Role")
        company = row.get("company", "")
        loc     = row.get("location", "")
        exp     = row.get("experience", "")
        desc    = str(row.get("description", ""))[:300]

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <span style="font-size:1.1rem; font-weight:600; color:#e2e8f0;">
                        #{int(row['rank'])} &nbsp; {title}
                    </span>
                    <br/>
                    <span style="color:#94a3b8; font-size:0.85rem;">
                        {f"🏢 {company}" if company else ""}
                        {f"&nbsp;|&nbsp; 📍 {loc}" if loc else ""}
                        {f"&nbsp;|&nbsp; ⏱ {exp}" if exp else ""}
                    </span>
                </div>
                <span class="score-badge {badge}">{sc}% Match</span>
            </div>
            <p style="margin-top:0.75rem; color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                {desc}{"…" if len(str(row.get("description",""))) > 300 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_skill_gap(clean_resume: str, models: dict, corpus: pd.DataFrame | None):
    """Show skill-gap report."""
    st.markdown("### 🛠️ Skill-Gap Report")

    cluster_model = models.get("cluster_model")
    vectorizer    = models.get("vectorizer")
    job_matrix    = models.get("job_matrix")

    if any(v is None for v in [cluster_model, vectorizer, job_matrix, corpus]):
        st.warning("⚠️ Cluster model not trained yet. Run notebook 04 first.")
        return

    with st.spinner("📊 Generating skill-gap report …"):
        resume_vec     = vectorizer.transform([clean_resume])
        cluster_id     = int(cluster_model.predict(resume_vec)[0])
        cluster_labels = cluster_model.labels_

        # Top skills for the target cluster
        from src.models.clustering import cluster_top_skills, skill_gap_report
        c_skills = cluster_top_skills(job_matrix, cluster_labels, vectorizer)
        gap      = skill_gap_report(clean_resume, cluster_id, c_skills)

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Cluster", f"#{gap['target_cluster']}")
    col2.metric("Skills Found",   f"{len(gap['resume_skills_found'])}/{len(gap['cluster_top_skills'])}")
    col3.metric("Skill Match",    f"{gap['match_pct']}%")

    # Radial progress
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gap["match_pct"],
        title={"text": "Skill Match %", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":       {"color": "#a855f7"},
            "bgcolor":   "rgba(0,0,0,0)",
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
        height=280,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Skills You Have")
        if gap["resume_skills_found"]:
            for s in gap["resume_skills_found"]:
                st.markdown(f'<span style="background:rgba(52,211,153,0.15); color:#34d399; '
                            f'padding:0.2rem 0.6rem; border-radius:999px; '
                            f'margin:0.2rem; display:inline-block; font-size:0.85rem;">{s}</span>',
                            unsafe_allow_html=True)
        else:
            st.info("No common skills detected in the target cluster.")

    with col_b:
        st.markdown("#### ❌ Skills to Develop")
        if gap["missing_skills"]:
            for s in gap["missing_skills"]:
                st.markdown(f'<span style="background:rgba(239,68,68,0.15); color:#ef4444; '
                            f'padding:0.2rem 0.6rem; border-radius:999px; '
                            f'margin:0.2rem; display:inline-block; font-size:0.85rem;">{s}</span>',
                            unsafe_allow_html=True)
        else:
            st.success("You have all the top skills for this job cluster! 🎉")


def render_sidebar():
    """Render sidebar with model status."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <span style="font-size:2rem;">🎯</span>
            <h2 style="margin:0.5rem 0 0; color:#a78bfa;">SmartHire</h2>
            <p style="color:#64748b; font-size:0.8rem; margin:0;">ML Job Portal</p>
        </div>
        <hr style="border-color:rgba(255,255,255,0.1);">
        """, unsafe_allow_html=True)

        st.markdown("#### 🤖 Model Status")
        checks = {
            "Resume Classifier":  CLASSIFIER_PATH.exists(),
            "Job Recommender":    (MODELS_DIR / "job_tfidf_matrix.npz").exists(),
            "Cluster Model":      CLUSTER_MODEL_PATH.exists(),
            "Job Corpus":         JOBS_CLEAN_CSV.exists(),
        }
        for name, ready in checks.items():
            icon = "✅" if ready else "⚠️"
            color = "#34d399" if ready else "#fbbf24"
            st.markdown(f'<p style="color:{color}; margin:0.3rem 0;">{icon} {name}</p>',
                        unsafe_allow_html=True)

        all_ready = all(checks.values())
        if not all_ready:
            st.markdown("""
            <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3);
                        border-radius:10px; padding:0.8rem; margin-top:1rem; font-size:0.83rem;
                        color:#fbbf24;">
            ⚠️ Some models are not trained.<br><br>
            Run the notebooks in order:<br>
            1️⃣ 01_eda.ipynb<br>
            2️⃣ 02_resume_classifier.ipynb<br>
            3️⃣ 03_recommender.ipynb<br>
            4️⃣ 04_clustering_topics.ipynb
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

    with st.spinner("Loading models …"):
        models = load_models()
        corpus = load_corpus()

    # Upload
    clean_resume = render_upload_section()

    if clean_resume:
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🏷️ Category", "💼 Job Matches", "🛠️ Skill Gap"])

        with tab1:
            render_category(clean_resume, models)

        with tab2:
            render_recommendations(clean_resume, models, corpus)

        with tab3:
            render_skill_gap(clean_resume, models, corpus)

    else:
        # Landing — show feature cards
        st.markdown("### ✨ What SmartHire Can Do For You")
        cols = st.columns(3)
        features = [
            ("🏷️", "Resume Classifier",   "Automatically identifies your job domain using ML."),
            ("💼", "Job Recommender",     "Finds the top matching jobs from thousands of listings."),
            ("🛠️", "Skill-Gap Report",    "Shows exactly which skills you need to develop next."),
        ]
        for col, (icon, title, desc) in zip(cols, features):
            col.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:2rem 1rem;">
                <div style="font-size:2.5rem;">{icon}</div>
                <h3 style="margin:0.5rem 0; color:#e2e8f0;">{title}</h3>
                <p style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
