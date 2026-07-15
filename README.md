# SmartHire 🎯

> An ML-powered job recommendation portal — upload your CV and instantly get matched jobs, a fit score, and a personalised skill-gap report.

---

## Features
| Feature | ML Approach |
|---|---|
| **Resume Category Classifier** | TF-IDF + Logistic Regression / SVM |
| **Job Recommender** | Cosine Similarity (TF-IDF vectors) |
| **Fit / Shortlisting Predictor** | XGBoost (optional) |
| **Job Clustering** | K-Means + PCA/t-SNE |
| **Skill-Gap Report** | Cluster centroid skill diff |
| **Web Portal** | Streamlit |

---

## Project Structure
```
smarthire/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/            # original downloaded datasets (never edit)
│   ├── interim/        # merged / partially cleaned data
│   └── processed/      # final model-ready data
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_resume_classifier.ipynb
│   ├── 03_recommender.ipynb
│   ├── 04_clustering_topics.ipynb
│   └── 05_fit_predictor.ipynb
│
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── text_features.py
│   │   └── match_features.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── recommender.py
│   │   ├── clustering.py
│   │   └── fit_predictor.py
│   ├── parsing/
│   │   └── resume_parser.py
│   └── evaluate.py
│
├── models/             # saved .pkl model files
├── app/
│   └── streamlit_app.py
├── reports/
│   └── figures/
└── tests/
    └── test_features.py
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/thotapremkumar-tpk/smarthire.git
cd smarthire

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Place datasets in data/raw/
#    - UpdatedResumeDataSet.csv
#    - jobs_dataset.csv

# 6. Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Datasets
- [UpdatedResumeDataSet](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) — resumes with job categories
- [LinkedIn Job Postings 2023–2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) — detailed job descriptions
- [Naukri Job Listings](https://www.kaggle.com/) — Indian job postings

---

## Milestones
- **Week 1** — Data cleaning, EDA, Resume Classifier
- **Week 2** — Job Recommender, Clustering, Skill-Gap Report
- **Week 3** — Streamlit Portal, Final Report

---

## Team
SmartHire ML Industrial Project