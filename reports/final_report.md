# SmartHire — Final Project Report

**Course:** Machine Learning Industrial Project  
**Team:** SmartHire ML Team  
**Date:** July 2026  
**Repository:** https://github.com/thotapremkumar-tpk/smarthire

---

## 1. Introduction

SmartHire is an ML-powered job recommendation portal that bridges the gap between job seekers and employers. Given a candidate's CV (PDF, DOCX, or plain text), the system:

1. **Classifies** the resume into a job domain (e.g., Data Science, Java Developer, HR)
2. **Recommends** the top matching job postings using cosine similarity
3. **Predicts a fit score** indicating how well the candidate matches a specific role
4. **Generates a skill-gap report** comparing the candidate's skills against target job clusters

---

## 2. Datasets

| Dataset | Source | Size | Description |
|---|---|---|---|
| `UpdatedResumeDataSet.csv` | Kaggle | 962 resumes, 25 categories | Labelled resumes for classifier training |
| `naukri_com-job_sample.csv` | Kaggle | 21,848 jobs | Indian job postings with title, skills, experience |
| `LinkedIn Job Postings 2023–24` | Kaggle | 110,898 jobs | Large set of detailed job descriptions |

**After merging and cleaning:** 132,746 total jobs → 75,000 deduplicated jobs in `jobs_clean.csv`.

---

## 3. Data Preprocessing

### 3.1 Resume Preprocessing
- Removed HTML tags, special characters, and extra whitespace
- Lowercased all text; removed English stopwords (NLTK)
- Lemmatized tokens using WordNet
- Encoded category labels as integers for the classifier

### 3.2 Job Corpus Preprocessing
- Merged Naukri and LinkedIn datasets with a common schema:  
  `title, company, location, skills, description, experience`
- Joined LinkedIn `job_skills.csv` with `mappings/skills.csv` on `skill_abr`
- Created a unified `text` field: `title + " " + skills + " " + description`
- Applied same text cleaning pipeline as resumes → stored in `clean_text`
- Removed 57,746 duplicate entries after union

---

## 4. Machine Learning Models

### 4.1 MODEL A — Resume Category Classifier (Supervised)

**Task:** Multi-class text classification — 25 job domain categories  
**Pipeline:** `Raw text → TF-IDF (3,000 features, unigrams+bigrams) → Classifier`

| Model | CV Accuracy (5-fold) | Std Dev |
|---|---|---|
| **Logistic Regression** ✓ | **0.9948** | ±0.0049 |
| Linear SVM | 0.9922 | ±0.0076 |

**Best Model:** Logistic Regression (C=5.0, max_iter=1000)  
**Test Set Performance:** Accuracy = **1.0000**, F1 = **1.0000** (193 test samples, 25 classes)

The classifier achieves near-perfect accuracy because the resume dataset has highly domain-specific vocabulary — each category uses distinct technical keywords (e.g., "Hadoop" → Hadoop Engineer, "blockchain" → Blockchain Developer).

**Saved:** `models/classifier.pkl`, `models/tfidf_vectorizer.pkl`

---

### 4.2 CORE ENGINE — Job Recommender (Unsupervised)

**Task:** Content-based top-N job ranking using cosine similarity  
**Algorithm:**
1. All 30,000 job descriptions vectorised with TF-IDF (8,000 features)
2. Uploaded resume vectorised with the same vocabulary
3. Cosine similarity computed between resume vector and all job vectors
4. Top-N results returned ranked by similarity score

**Sample output for a "Data Science" resume:**
| Rank | Match Score | Job Title | Company |
|---|---|---|---|
| 1 | 27.6% | Content Development \| Data Science & NLP | Way2Class |
| 2 | 26.5% | Palantir Developer | Tata Consultancy Services |
| 3 | 25.3% | Sr Data Scientist | Karvy Analytics Limited |
| 4 | 23.9% | Senior Data Scientist | Optimus Solutions |
| 5 | 22.9% | Lead Machine Learning Engineer | Confidential |

Results are semantically relevant — all top 5 jobs are data/ML roles, confirming the recommender works correctly.

**Saved:** `models/job_tfidf_matrix.npz`, `models/job_tfidf_vectorizer.pkl`

---

### 4.3 DISCOVERY — Job Clustering (Unsupervised)

**Task:** Discover natural job families using K-Means  
**Method:** K-Means on TF-IDF job vectors; k selected via Elbow Method + Silhouette Score

| k | Inertia | Silhouette Score |
|---|---|---|
| **2** | 7,256 | **0.0264** ← Best |
| 3 | 7,194 | 0.0219 |
| 4 | 7,143 | 0.0202 |
| 5–10 | decreasing | 0.0117–0.0134 |

**Selected k=2** (highest silhouette score). The low silhouette values (typical for high-dimensional text data) indicate overlapping job categories — expected given the diverse nature of job descriptions.

**Cluster Profiles:**
- **Cluster 0:** Tech-oriented (Android Developer, Project Manager, Sales)
- **Cluster 1:** Finance/Admin-oriented (Project Manager, Staff Accountant, Senior Accountant)

**Saved:** `models/cluster_model.pkl`

---

### 4.4 MODEL B — Fit Predictor / Shortlisting (Supervised — Optional)

**Task:** Binary classification — does this candidate fit this job? (0/1)  
**Features engineered from resume-job pairs:**

| Feature | Description |
|---|---|
| `jaccard` | Word overlap (Jaccard similarity) between resume and job text |
| `skill_overlap` | Fraction of job skills present in resume |
| `resume_len` | Normalised resume word count |
| `has_exp_kw` | Binary — resume mentions "years", "experience", "worked" |
| `title_match` | Binary — resume category keyword appears in job title |

**Training data:** 8,000 synthetic resume-job pairs (balanced 50/50 positive/negative)

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** ✓ | **0.9971** |
| XGBoost | 0.9962 |

**Best Model:** Logistic Regression (ROC-AUC = 0.9971)  
**Most important feature:** `title_match` and `jaccard` similarity

**Saved:** `models/fit_predictor.pkl`

---

## 5. System Architecture

```
Resume Upload (PDF / DOCX / text)
            │
            ▼
    Text Extraction → Preprocessing (clean, tokenize, vectorize)
            │
            ├──► Classifier (supervised)    → predicts target role (25 categories)
            ├──► Similarity engine (unsup.) → top-N matching jobs + match scores
            ├──► Fit predictor (supervised) → shortlisting probability (0–100%)
            └──► Skill-gap module (unsup.)  → CV improvement report
            │
            ▼
    Streamlit Web Portal
    (shows recommended jobs, fit scores, and skill gaps)
```

---

## 6. Evaluation Metrics

| Component | Metric | Value |
|---|---|---|
| Classifier | Accuracy | 1.0000 |
| Classifier | Weighted F1 | 1.0000 |
| Classifier | CV Accuracy | 0.9948 ± 0.0049 |
| Clustering | Silhouette Score (k=2) | 0.0264 |
| Clustering | Inertia | 7,256 |
| Fit Predictor | ROC-AUC (LR) | 0.9971 |
| Fit Predictor | ROC-AUC (XGBoost) | 0.9962 |
| Recommender | Qualitative check | ✅ Top 5 jobs semantically relevant |

---

## 7. Web Portal (Streamlit App)

The portal (`app/streamlit_app.py`) provides a glassmorphism dark-themed UI with:
- **CV Upload** — drag-and-drop PDF/DOCX/TXT
- **Detected Category** — classifier prediction
- **Job Recommendations** — ranked table with match scores
- **Fit Score** — percentage likelihood of shortlisting
- **Skill Gap Report** — missing skills vs target cluster

To run locally:
```bash
streamlit run app/streamlit_app.py
```

---

## 8. Results Discussion

### Strengths
- **Classifier** achieves 99.5% CV accuracy — excellent for 25 categories
- **Recommender** returns semantically meaningful results (validated qualitatively)
- **Fit Predictor** achieves 99.7% ROC-AUC — strong binary classification
- Complete end-to-end pipeline from raw CSV → trained models → web app

### Limitations
- Clustering silhouette scores are low (0.02–0.03), typical for sparse high-dimensional text vectors; t-SNE would give better visual separation but was excluded for runtime
- Fit Predictor training data is **synthetic** (automatically generated pairs) — a real-world dataset of recruiter decisions would improve generalisability
- Recommender uses TF-IDF (bag-of-words) — replacing with sentence embeddings (BERT/SBERT) would capture semantic similarity better
- Dataset limited to Indian (Naukri) and global (LinkedIn) sources — may not generalise to all markets

### Future Improvements (Stretch Goals)
1. Replace TF-IDF with sentence embeddings (e.g., `sentence-transformers`)
2. Add a learning-to-rank model for ordering results
3. Build a rule-based "mentor" chatbot using skill-gap output
4. Deploy on Streamlit Community Cloud with a public URL
5. Collect real recruiter feedback data to retrain the fit predictor

---

## 9. Repository Structure

```
smarthire/
├── README.md                    # project intro & setup
├── requirements.txt             # pinned dependencies
├── train_all.py                 # master training script
├── data/
│   ├── raw/                     # original datasets (never edited)
│   ├── interim/job_corpus.csv   # merged 132,746 jobs
│   └── processed/               # model-ready data
├── notebooks/                   # 5 exploration notebooks
├── src/                         # reusable Python modules
│   ├── config.py
│   ├── data/{load_data, preprocess}.py
│   ├── features/{text_features, match_features}.py
│   ├── models/{classifier, recommender, clustering, fit_predictor}.py
│   ├── parsing/resume_parser.py
│   └── evaluate.py
├── models/                      # saved .pkl model files
├── app/streamlit_app.py         # web portal
├── reports/figures/             # 18 generated charts
└── tests/test_features.py       # unit tests
```

---

## 10. Conclusion

SmartHire successfully implements all **core deliverables** from the project brief:
- ✅ Resume Category Classifier (supervised) with metrics
- ✅ Job Recommender (unsupervised, cosine similarity)
- ✅ Job Clustering with Elbow + Silhouette analysis
- ✅ Fit Predictor (optional — LR vs XGBoost comparison)
- ✅ Working Streamlit demo portal
- ✅ Full Git repository with correct structure
- ✅ Written report (this document)

The system demonstrates the full ML pipeline from data collection → preprocessing → feature engineering → model training → evaluation → deployment, covering both supervised and unsupervised learning paradigms.
