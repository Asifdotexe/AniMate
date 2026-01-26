"""
Module that contains the k-NN model for the recommendation system.
"""

import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import preprocess_text

# Create path to project root to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def load_data() -> pd.DataFrame:
    """
    Load the anime data from the CSV file.

    :return: DataFrame containing the anime data.
    """
    data_path = os.path.join(
        project_root, "data", "final", "processed_data_02092024.csv"
    )
    return pd.read_csv(data_path)


def vectorize(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Vectorize the text data using TF-IDF.

    :param df: DataFrame containing text data.
    :return: TF-IDF DataFrame and the vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )
    return tfidf_df, tfidf_vectorizer


def build_knn_model(
    tfidf_features_df: pd.DataFrame, n_neighbors: int = 10
) -> NearestNeighbors:
    """
    Build and fit a k-NN model using the TF-IDF features.

    :param tfidf_features_df: DataFrame with TF-IDF features.
    :param n_neighbors: Number of neighbors to use for k-NN (default is 10).
    :return: Fitted k-NN model.
    """
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn_model.fit(tfidf_features_df)
    return knn_model


def recommend_anime_knn(
    query: str,
    tfidf_vectorizer: TfidfVectorizer,
    knn_model: NearestNeighbors,
    data_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend anime titles based on a user query using the k-NN model.

    :param query: The user's input query.
    :param tfidf_vectorizer: The TF-IDF vectorizer used for the anime data.
    :param knn_model: The k-NN model for finding similar animes.
    :param data_df: The DataFrame containing anime data.
    :param top_n: Number of recommendations to return (default is 10).
    :return: DataFrame containing the top recommended anime titles.
    """
    query_processed = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    _, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n)
    return data_df.iloc[indices[0]][["title", "genres"]]


def anime_recommendation_pipeline(user_query: str, top_n: int = 10) -> pd.DataFrame:
    """
    Full pipeline to process data, build the k-NN model, and recommend animes based on the user query.

    :param user_query: The user's input query for anime recommendation.
    :param top_n: Number of recommendations to return (default is 10).
    :return: DataFrame containing the top recommended anime titles.
    """
    data_df = load_data()
    tfidf_features_df, tfidf_vectorizer = vectorize(data_df)
    knn_model = build_knn_model(tfidf_features_df)
    recommended_animes = recommend_anime_knn(
        user_query, tfidf_vectorizer, knn_model, data_df, top_n
    )
    return recommended_animes
