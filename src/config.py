"""
config.py — central configuration: paths, constants, and model hyperparameters.
All other modules import from here so changing a value here propagates everywhere.
"""

from pathlib import Path

# ── Root of the repository ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
INTERIM_DIR     = DATA_DIR / "interim"
PROCESSED_DIR   = DATA_DIR / "processed"

# ── Raw dataset filenames ─────────────────────────────────────────────────────
# Resumes
RESUME_CSV      = RAW_DIR / "UpdatedResumeDataSet.csv"

# Naukri
NAUKRI_CSV      = RAW_DIR / "naukri_com-job_sample.csv"

# LinkedIn (three files inside the Linkedin/ subfolder)
LINKEDIN_DIR          = RAW_DIR / "Linkedin"
LINKEDIN_POSTINGS     = LINKEDIN_DIR / "postings.csv"
LINKEDIN_JOB_SKILLS   = LINKEDIN_DIR / "jobs" / "job_skills.csv"
LINKEDIN_SKILLS_MAP   = LINKEDIN_DIR / "mappings" / "skills.csv"

# ── Processed / interim artefacts ─────────────────────────────────────────────
JOB_CORPUS_CSV   = INTERIM_DIR  / "job_corpus.csv"        # merged job table (interim)
JOBS_CLEAN_CSV   = PROCESSED_DIR / "jobs_clean.csv"        # + clean_text column (recommender input)
RESUMES_CLEAN_CSV = PROCESSED_DIR / "resumes_clean.csv"    # cleaned resume + Category (classifier input)

# Legacy alias kept so notebooks that import RESUME_PROC_CSV still work
RESUME_PROC_CSV  = RESUMES_CLEAN_CSV

# ── Model / artefact paths ────────────────────────────────────────────────────
MODELS_DIR          = ROOT_DIR / "models"
CLASSIFIER_PATH     = MODELS_DIR / "classifier.pkl"
TFIDF_PATH          = MODELS_DIR / "tfidf_vectorizer.pkl"
FIT_PREDICTOR_PATH  = MODELS_DIR / "fit_predictor.pkl"
CLUSTER_MODEL_PATH  = MODELS_DIR / "cluster_model.pkl"
JOB_VECTORS_PATH    = MODELS_DIR / "job_tfidf_matrix.pkl"

# ── Reports ───────────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Text-processing constants ─────────────────────────────────────────────────
MAX_TFIDF_FEATURES  = 5_000          # vocabulary size for TF-IDF
NGRAM_RANGE         = (1, 2)         # unigrams + bigrams
MIN_DF              = 2              # ignore tokens appearing in < 2 docs

# ── Recommender ───────────────────────────────────────────────────────────────
TOP_N_JOBS = 10                      # number of job recommendations returned

# ── Clustering ────────────────────────────────────────────────────────────────
N_CLUSTERS      = 10                 # K-Means k (tune via elbow / silhouette)
RANDOM_STATE    = 42

# ── Classifier ───────────────────────────────────────────────────────────────
TEST_SIZE       = 0.2
CV_FOLDS        = 5

# ── Skill-gap ─────────────────────────────────────────────────────────────────
TOP_SKILLS_PER_CLUSTER = 15          # skills shown in gap report

# ── Ensure required directories exist ─────────────────────────────────────────
for _dir in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
