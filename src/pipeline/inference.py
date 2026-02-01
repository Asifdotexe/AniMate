"""
Module that contains the engine for the recommendation system (inference only).
"""

import gc
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.components.transformation import preprocess_text
from src.config import config
from src.logger import setup_logging

logger = setup_logging("inference")


def load_models() -> tuple[NearestNeighbors, TfidfVectorizer]:
    """
    Load pre-trained model and vectorizer from the configured paths.

    :return: A tuple containing the k-NN model and the TF-IDF vectorizer.
    """
    knn_path = Path(config.paths.knn_model)
    tfidf_path = Path(config.paths.vectorizer)

    if not knn_path.exists() or not tfidf_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Please run src/components/trainer.py first."
        )

    logger.info("Loading models...")
    knn_model = joblib.load(knn_path)
    tfidf_vectorizer = joblib.load(tfidf_path)

    return knn_model, tfidf_vectorizer


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed dataframe from the configured path.

    :return: The processed dataframe.
    """
    data_path = Path(config.paths.vector_embeddings)
    # Note: user instruction said vector_embeddings.pkl, old code loaded processed_data.pkl.
    # We renamed processed_data.pkl to vector_embeddings.pkl in step 1.

    if not data_path.exists():
        raise FileNotFoundError(
            f"Embeddings data not found at {data_path}. Please run src/components/trainer.py first."
        )
    logger.info(f"Loading data from {data_path}...")
    return pd.read_pickle(data_path)


def _filter_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter out exact query matches from recommendations.

    :param df: DataFrame with recommendations.
    :param query: User query.
    :return: Filtered DataFrame.
    """
    # Normalize title column if needed (handling legacy case)
    if "title" not in df.columns and "Title" in df.columns:
        df = df.rename(columns={"Title": "title"})

    if "title" not in df.columns:
        return df

    # Exclude the query itself if it appears in results
    mask = ~df["title"].str.contains(query, case=False, na=False, regex=False)
    filtered = df[mask]

    return filtered if not filtered.empty else df


def get_recommendations(
    query: str,
    tfidf_vectorizer: TfidfVectorizer,
    knn_model: NearestNeighbors,
    data: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend anime based on a user query using the k-NN model.

    :param query: The user input query.
    :param tfidf_vectorizer: The fitted TF-IDF vectorizer.
    :param knn_model: The fitted k-NN model.
    :param data: The full anime DataFrame.
    :param top_n: Number of recommendations.

    :returns: A DataFrame containing the recommended anime sorted by score.
    """
    query_processed = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])

    # Query more neighbors to allow for filtering
    n_neighbors_query = top_n + 5
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=n_neighbors_query)

    # Use iloc to get rows, copy to avoid SettingWithCopyWarning
    recommendations = data.iloc[indices[0]].copy()
    
    # Add distance column (smaller distance = better match)
    if distances is None:
        # Fallback if kneighbors returns None for distances
        # Create a dummy distance array
        import numpy as np
        recommendations["distance"] = np.zeros(len(recommendations))
    else:
        recommendations["distance"] = distances[0]

    # Filter and sort
    final_recommendations = _filter_by_query(recommendations, query)

    if "score" not in final_recommendations.columns:
        # Fallback
        final_recommendations["score"] = 0

    # Sort primarily by distance (asc), secondarily by score (desc)
    final_recommendations = final_recommendations.sort_values(
        by=["distance", "score"], ascending=[True, False]
    ).head(top_n)

    # Cleanup memory
    gc.collect()

    return final_recommendations
