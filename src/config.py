"""
Central configuration for AniMate.
This module contains all the hardcoded values, paths, and constants used across the project.
"""

from pathlib import Path

# Project Root
# Assuming this file is in src/config.py, project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Files
MASTER_DB_FILENAME = "anime_master_db.csv"
RAW_DATA_PATH = RAW_DATA_DIR / MASTER_DB_FILENAME
PROCESSED_DATA_FILENAME = "anime_data_processed.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
MODEL_CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# Model Artifacts
KNN_MODEL_FILE = "knn_model.joblib"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.joblib"
PROCESSED_DATA_PKL = "processed_data.pkl"

# Data Processing
REQUIRED_COLUMNS = ["title", "synopsis"]
CATEGORY_COLUMNS = ["genres", "studio", "demographic", "source", "status"]

# NLTK Resources
NLTK_RESOURCES = ["corpora/stopwords", "tokenizers/punkt_tab", "tokenizers/punkt"]
