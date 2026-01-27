"""
Module for training the recommendation model.
This script loads processed data, vectorizes it, trains a KNN model,
and saves the artifacts to the models/ directory.
"""

import os
import sys

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_text


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
             # Return empty dict if config is invalid or empty to avoid AttributeError later
             # or we could raise an error. A valid config is usually required, but fail-safe 
             # empty dict allows get() calls to return defaults.
             return {}
        return cfg


def train():
    """
    Train the model and save artifacts.
    """
    print("Loading configuration...")
    config = load_config()

    print("Loading processed data...")
    # We load from data/processed/anime_data_processed.csv
    # Ideally this filename matches what process.py outputs.
    processed_data_path = os.path.join(
        project_root, "data", "processed", "anime_data_processed.csv"
    )

    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data not found at {processed_data_path}")
        print("Please run src/pipeline/process.py first.")
        return

    # Load data. We assume types are inferred correctly or we can use dtypes from config if applicable.
    # But since it's processed CSV, we just load it.
    df = pd.read_csv(processed_data_path)

    # Check if 'stemmed_synopsis' exists, if not create it (backward compatibility or safety)
    if "stemmed_synopsis" not in df.columns:
        print("stemmed_synopsis column missing, generating it...")
        # basic fillna to avoid errors
        df["Synopsis"] = df["Synopsis"].fillna("")
        df["stemmed_synopsis"] = df["Synopsis"].apply(preprocess_text)

    # Ensure no float/NaN values in text column
    df["stemmed_synopsis"] = df["stemmed_synopsis"].fillna("")

    print("Vectorizing and building model...")
    model_cfg = config.get("model", {})

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english", max_features=model_cfg.get("max_features", 5000)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["stemmed_synopsis"])

    knn_model = NearestNeighbors(
        n_neighbors=model_cfg.get("n_neighbors", 5),
        metric=model_cfg.get("metric", "cosine"),
    ).fit(tfidf_matrix)

    print("Saving artifacts...")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(knn_model, os.path.join(models_dir, "knn_model.joblib"))
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    # Save the dataframe as pickle for fast loading in app
    # The app needs the full dataframe to display results
    df.to_pickle(os.path.join(models_dir, "processed_data.pkl"))

    print("Training complete. Artifacts saved in 'models/'.")


if __name__ == "__main__":
    train()
