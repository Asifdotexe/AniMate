"""
Module that contains the engine for the recommendation system.
"""

import gc
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.preprocessing import preprocess_text


def load_data(path: str, dtypes: dict) -> pd.DataFrame:
    """
    Load anime data from a CSV file and optimize data types.

    :param path: Path to the CSV file.
    :returns: A DataFrame containing the anime data.
    """
    return pd.read_csv(path, usecols=dtypes.keys(), dtype=dtypes)


def vectorize_and_build_model(
    df: pd.DataFrame, config: dict
) -> tuple[NearestNeighbors, TfidfVectorizer]:
    """
    Vectorize the synopsis of the anime DataFrame and build a k-NN model.

    :param df: The DataFrame containing anime data.
    :param config: Model configuration dictionary.

    :return: A tuple containing the k-NN model and the TF-IDF vectorizer.
    """
    # Note: Modifying df in place for efficiency, but valid concern for side effects.
    # Given the previous code did this on the global 'data' df, we continue the pattern
    # but strictly speaking this column is needed for the vectorizer.
    if "stemmed_synopsis" not in df.columns:
        df["stemmed_synopsis"] = df["synopsis"].apply(
            lambda x: preprocess_text(x) if pd.notna(x) else ""
        )

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english", max_features=config.get("max_features", 5000)
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["stemmed_synopsis"])

    knn_model = NearestNeighbors(
        n_neighbors=config.get("n_neighbors", 5), metric=config.get("metric", "cosine")
    ).fit(tfidf_matrix)

    return knn_model, tfidf_vectorizer


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
            f"Model artifacts not found in {model_dir}. Please run scripts/train_model.py first."
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
            f"Processed data not found in {model_dir}. Please run scripts/train_model.py first."
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

    # Filter out the query itself if it matches a title exactly (basic self-exclusion)
    # The original logic filtered by title contains query, which might be aggressive if query is generic
    # But we preserve original logic behavior:
    filtered_recommendations = recommendations[
        ~recommendations["title"].str.contains(query, case=False, na=False)
    ]

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
