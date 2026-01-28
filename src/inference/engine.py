"""
Module that contains the engine for the recommendation system (inference only).
"""

import gc
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import preprocess_text


def load_models(model_dir: str) -> tuple[NearestNeighbors, TfidfVectorizer]:
    """
    Load pre-trained model and vectorizer from the specified directory.

    :param model_dir: Directory containing the .joblib files.
    :return: A tuple containing the k-NN model and the TF-IDF vectorizer.
    """
    knn_path = os.path.join(model_dir, "knn_model.joblib")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

    if not os.path.exists(knn_path) or not os.path.exists(tfidf_path):
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. Please run src/pipeline/train.py first."
        )

    knn_model = joblib.load(knn_path)
    tfidf_vectorizer = joblib.load(tfidf_path)

    return knn_model, tfidf_vectorizer


def load_processed_data(model_dir: str) -> pd.DataFrame:
    """
    Load the processed dataframe from the model directory.

    :param model_dir: Directory containing the .pkl file.
    :return: The processed dataframe.
    """
    data_path = os.path.join(model_dir, "processed_data.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found in {model_dir}. Please run src/pipeline/train.py first."
        )
    return pd.read_pickle(data_path)


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
    _, indices = knn_model.kneighbors(query_tfidf, n_neighbors=n_neighbors_query)

    recommendations = data.iloc[indices[0]].copy()

    # Normalize title column
    if "title" not in recommendations.columns and "Title" in recommendations.columns:
        recommendations = recommendations.rename(columns={"Title": "title"})
        
    # Filter out the query itself if it matches a title exactly (basic self-exclusion)
    if "title" in recommendations.columns:
         filtered_recommendations = recommendations[
            ~recommendations["title"].str.contains(query, case=False, na=False)
        ]
    else:
        # If we can't find title column even after check, just skip filtering
        filtered_recommendations = recommendations

    if filtered_recommendations.empty:
        filtered_recommendations = recommendations

    if "score" not in filtered_recommendations.columns:
        raise ValueError(
            "The 'score' column is missing from the DataFrame "
            "'filtered_recommendations'. Cannot sort recommendations."
        )

    final_recommendations = filtered_recommendations.sort_values(
        by="score", ascending=False
    ).head(top_n)

    # Cleanup memory
    del recommendations
    gc.collect()

    return final_recommendations
