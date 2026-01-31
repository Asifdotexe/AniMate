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
from src import config


def load_config() -> dict:
    """Load configuration from config.yaml.

    :return: Configuration dictionary.
    """
    config_path = config.MODEL_CONFIG_FILE
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
             # Return empty dict if config is invalid or empty to avoid AttributeError later
             # or we could raise an error. A valid config is usually required, but fail-safe 
             # empty dict allows get() calls to return defaults.
             return {}
        return cfg


def load_processed_data(data_path: str) -> pd.DataFrame:
    """Load and optimize processed data.

    :param data_path: Path to processed data.
    :return: DataFrame with processed data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run src/pipeline/process.py first.")

    df = pd.read_csv(data_path)

    # Memory Optimization: explicit cast to category
    for col in config.CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure text column exists
    if "stemmed_synopsis" not in df.columns:
        print("Regenerating stemmed_synopsis...")
        df["synopsis"] = df["synopsis"].fillna("")
        df["stemmed_synopsis"] = df["synopsis"].apply(preprocess_text)
    
    df["stemmed_synopsis"] = df["stemmed_synopsis"].fillna("")
    return df


def train_knn_model(df: pd.DataFrame, config: dict) -> tuple[NearestNeighbors, TfidfVectorizer]:
    """Vectorize text and train K-NN model.

    :param df: DataFrame with processed data.
    :param config: Configuration dictionary.
    :return: Tuple of K-NN model and TfidfVectorizer.
    """
    model_cfg = config.get("model", {})
    
    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_features=model_cfg.get("max_features", 5000)
    )
    tfidf_matrix = vectorizer.fit_transform(df["stemmed_synopsis"])

    print("Training model...")
    knn = NearestNeighbors(
        n_neighbors=model_cfg.get("n_neighbors", 5),
        metric=model_cfg.get("metric", "cosine"),
    )
    knn.fit(tfidf_matrix)
    
    return knn, vectorizer


def save_artifacts(models_dir: str, knn: NearestNeighbors, vectorizer: TfidfVectorizer, df: pd.DataFrame):
    """Save model artifacts and processed data.

    :param models_dir: Directory to save artifacts.
    :param knn: K-NN model.
    :param vectorizer: TfidfVectorizer.
    :param df: DataFrame with processed data.
    """
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(knn, os.path.join(models_dir, config.KNN_MODEL_FILE))
    joblib.dump(vectorizer, os.path.join(models_dir, config.TFIDF_VECTORIZER_FILE))
    df.to_pickle(os.path.join(models_dir, config.PROCESSED_DATA_PKL))
    print(f"Artifacts saved to {models_dir}")


def train():
    """Main training execution."""
    print("Loading configuration...")
    
    try:
        model_config = load_config()

        data_path = config.PROCESSED_DATA_PATH
        df = load_processed_data(data_path)

        knn_model, vectorizer = train_knn_model(df, model_config)

        models_dir = config.MODELS_DIR
        save_artifacts(models_dir, knn_model, vectorizer, df)
        
    except Exception:
        import traceback
        print(f"Training failed:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    train()
