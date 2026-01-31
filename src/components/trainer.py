"""
Module for training the recommendation model.
This script loads processed data, vectorizes it, trains a KNN model,
and saves the artifacts to the models/ directory.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.components.transformation import preprocess_text
from src.config import config
from src.logger import setup_logging

logger = setup_logging("trainer")


def load_processed_data(data_path: Path) -> pd.DataFrame:
    """Load and optimize processed data.

    :param data_path: Path to processed data.
    :return: DataFrame with processed data.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. Run src/components/transformation.py first."
        )

    df = pd.read_csv(data_path)

    # Import CATEGORY_COLUMNS from transformation to ensure categorical columns are available for downstream processing.
    from src.components.transformation import CATEGORY_COLUMNS

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure text column exists
    if "stemmed_synopsis" not in df.columns:
        logger.info("Regenerating stemmed_synopsis...")
        df["synopsis"] = df["synopsis"].fillna("")
        df["stemmed_synopsis"] = df["synopsis"].apply(preprocess_text)

    df["stemmed_synopsis"] = df["stemmed_synopsis"].fillna("")
    return df


def train_knn_model(df: pd.DataFrame) -> tuple[NearestNeighbors, TfidfVectorizer]:
    """Vectorize text and train K-NN model.

    :param df: DataFrame with processed data.
    :return: Tuple of K-NN model and TfidfVectorizer.
    """
    model_cfg = config.model

    logger.info("Vectorizing data...")
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=model_cfg.vectorizer_max_features
    )
    # Ensure stemmed_synopsis is string
    tfidf_matrix = vectorizer.fit_transform(df["stemmed_synopsis"].astype(str))

    logger.info("Training model...")
    knn = NearestNeighbors(
        n_neighbors=model_cfg.top_k_recommendations,  # use top_k_recommendations for n_neighbors
        metric="cosine",
    )
    knn.fit(tfidf_matrix)

    return knn, vectorizer


def save_artifacts(
    knn: NearestNeighbors, vectorizer: TfidfVectorizer, df: pd.DataFrame
):
    """Save model artifacts and processed data.

    :param knn: K-NN model.
    :param vectorizer: TfidfVectorizer.
    :param df: DataFrame with processed data.
    """
    knn_model_path = Path(config.paths.knn_model)
    vectorizer_path = Path(config.paths.vectorizer)
    embeddings_path = Path(config.paths.vector_embeddings)

    # Ensure directories exist
    knn_model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(knn, knn_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    df.to_pickle(embeddings_path)
    logger.info(f"Artifacts saved to {knn_model_path.parent}")


def train():
    """Main training execution."""
    logger.info("Starting model training...")

    try:
        data_path = Path(config.paths.processed_data)
        df = load_processed_data(data_path)

        knn_model, vectorizer = train_knn_model(df)

        save_artifacts(knn_model, vectorizer, df)

    except Exception:
        import traceback

        logger.error(f"Training failed:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    train()
