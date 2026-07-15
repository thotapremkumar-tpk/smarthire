"""
tests/test_features.py — Basic unit tests for SmartHire feature modules.

Run with:
    pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestCleanText:
    from src.data.preprocess import clean_text

    def test_lowercases(self):
        from src.data.preprocess import clean_text
        assert clean_text("Python JAVA SQL") == clean_text("python java sql")

    def test_removes_urls(self):
        from src.data.preprocess import clean_text
        result = clean_text("visit http://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_removes_numbers(self):
        from src.data.preprocess import clean_text
        result = clean_text("5 years of experience in 2023")
        assert "5" not in result
        assert "2023" not in result

    def test_removes_stop_words(self):
        from src.data.preprocess import clean_text
        result = clean_text("I am a software engineer")
        assert " i " not in f" {result} "
        assert " am " not in f" {result} "

    def test_returns_string(self):
        from src.data.preprocess import clean_text
        assert isinstance(clean_text("hello world"), str)

    def test_handles_empty(self):
        from src.data.preprocess import clean_text
        assert clean_text("") == ""

    def test_handles_none(self):
        from src.data.preprocess import clean_text
        assert clean_text(None) == ""


# ── Match features ────────────────────────────────────────────────────────────

class TestSkillOverlap:
    def test_perfect_overlap(self):
        from src.features.match_features import skill_overlap
        text = "python sql machine learning"
        score = skill_overlap(text, text)
        assert score == 1.0

    def test_no_overlap(self):
        from src.features.match_features import skill_overlap
        score = skill_overlap("python developer", "excel tableau")
        # Might not be exactly 0 depending on common skills found, but low
        assert score < 0.5

    def test_partial_overlap(self):
        from src.features.match_features import skill_overlap
        score = skill_overlap("python sql react", "python java sql docker")
        assert 0 < score < 1

    def test_returns_float(self):
        from src.features.match_features import skill_overlap
        score = skill_overlap("python", "python java")
        assert isinstance(score, float)


class TestExperienceMatch:
    def test_meets_requirement(self):
        from src.features.match_features import experience_match
        score = experience_match("5 years of experience in data science", "3 years")
        assert score == 1.0

    def test_below_requirement(self):
        from src.features.match_features import experience_match
        score = experience_match("1 year of experience", "5 years required")
        assert score == 0.0

    def test_unknown_requirement(self):
        from src.features.match_features import experience_match
        score = experience_match("2 years experience", "experience preferred")
        assert score == 0.5

    def test_unknown_resume(self):
        from src.features.match_features import experience_match
        score = experience_match("recent graduate", "3 years")
        assert score == 0.5


# ── TF-IDF features ───────────────────────────────────────────────────────────

class TestTfidf:
    CORPUS = [
        "python machine learning data science",
        "java spring boot microservices backend",
        "react angular javascript frontend developer",
        "sql database postgresql data analyst",
        "docker kubernetes devops cloud aws",
    ]

    def test_fit_returns_tuple(self):
        from src.features.text_features import fit_tfidf
        vec, mat = fit_tfidf(self.CORPUS, save=False)
        assert vec is not None
        assert mat is not None

    def test_matrix_shape(self):
        from src.features.text_features import fit_tfidf
        vec, mat = fit_tfidf(self.CORPUS, save=False)
        assert mat.shape[0] == len(self.CORPUS)

    def test_transform_single(self):
        from src.features.text_features import fit_tfidf, transform_tfidf
        vec, _ = fit_tfidf(self.CORPUS, save=False)
        result = transform_tfidf(vec, ["python developer"])
        assert result.shape[0] == 1

    def test_top_terms_returns_list(self):
        from src.features.text_features import fit_tfidf, top_terms
        vec, mat = fit_tfidf(self.CORPUS, save=False)
        terms = top_terms(vec, mat, row_idx=0, n=5)
        assert isinstance(terms, list)
        assert len(terms) == 5


# ── Resume parser ─────────────────────────────────────────────────────────────

class TestResumeParser:
    def test_txt_parsing(self):
        from src.parsing.resume_parser import parse_resume
        text = "John Doe\nPython Developer\n5 years experience"
        result = parse_resume(text.encode(), file_type="txt")
        assert "Python" in result or "python" in result.lower()

    def test_unsupported_type_raises(self):
        from src.parsing.resume_parser import parse_resume
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_resume(b"content", file_type="xlsx")


# ── Evaluate ──────────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect(self):
        from src.evaluate import precision_at_k
        recs = ["Data Scientist", "ML Engineer", "AI Researcher"]
        rels = ["Data Scientist", "ML Engineer", "AI Researcher"]
        assert precision_at_k(recs, rels, k=3) == 1.0

    def test_zero(self):
        from src.evaluate import precision_at_k
        assert precision_at_k(["Java Dev"], ["Python Dev"], k=1) == 0.0

    def test_partial(self):
        from src.evaluate import precision_at_k
        recs = ["Data Scientist", "Java Dev", "ML Engineer"]
        rels = ["Data Scientist", "ML Engineer"]
        score = precision_at_k(recs, rels, k=3)
        assert abs(score - 2/3) < 1e-9
