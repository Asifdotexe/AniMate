"""
This script trains the recommendation model using the anime data.
"""

import os
import sys

import joblib
import pandas as pd
import yaml

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import engine


def train():
    """
    Train using config.yaml parameters and save artifacts to models/ directory.
    """
    print("Loading configuration...")
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loading data...")
    data_cfg = config["data"]
    # Construct absolute path for data
    data_path = os.path.join(project_root, data_cfg["file_path"].replace("/", os.sep))
    df = engine.load_data(data_path, data_cfg["dtypes"])

    print("Vectorizing and building model...")
    knn_model, tfidf_vectorizer = engine.vectorize_and_build_model(df, config["model"])

    print("Saving artifacts...")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(knn_model, os.path.join(models_dir, "knn_model.joblib"))
    joblib.dump(tfidf_vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    # Save the processed dataframe (with stemmed_synopsis if needed, though typically we just need the source data)
    # engine.vectorize_and_build_model modifies df in-place to add 'stemmed_synopsis'.
    # We should save this processed DF so we don't have to re-process it at runtime?
    # Actually, runtime 'get_recommendations' doesn't use 'stemmed_synopsis' of the target data,
    # it only uses the indices from KNN to pick rows.
    # However, if we want to add new items efficiently later, retaining the stemmed version helps.
    # For now, let's strictly save what's needed.
    # The 'df' now has 'stemmed_synopsis'.
    # Let's save it as a pickle for fast loading.
    df.to_pickle(os.path.join(models_dir, "processed_data.pkl"))

    print("Training complete. Artifacts saved in 'models/'.")


if __name__ == "__main__":
    train()
