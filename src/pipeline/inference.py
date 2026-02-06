"""
Module that contains the engine for the recommendation system (inference only).
"""

import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import vstack
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


def search_anime_titles(query: str, df: pd.DataFrame, limit: int = 10) -> list[dict]:
    """
    Search for anime titles matching the query.
    Returns a list of dictionaries with title, image, and year.

    :param query: The search query.
    :param df: The DataFrame containing the anime data.
    :param limit: The maximum number of results to return.
    :return: A list of dictionaries containing the search results.
    """
    if not query or len(query) < 2:
        return []

    # Case-insensitive search
    mask = df["title"].str.contains(query, case=False, na=False) | df[
        "english title"
    ].str.contains(query, case=False, na=False)

    matches = df[mask].head(limit)

    results = []
    for _, row in matches.iterrows():
        results.append(
            {
                "title": row["title"],
                "english_title": (
                    row["english title"] if pd.notna(row["english title"]) else ""
                ),
                "image_url": row["image url"] if pd.notna(row["image url"]) else "",
                "year": (
                    int(row["release year"]) if pd.notna(row["release year"]) else "N/A"
                ),
            }
        )

    return results


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed dataframe from the configured path.

    :return: The processed dataframe.
    """
    data_path = Path(config.paths.vector_embeddings)

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


def recommend_by_description(
    query: str,
    tfidf_vectorizer: TfidfVectorizer,
    knn_model: NearestNeighbors,
    data: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend anime based on a user query using the k-NN model.
    Renamed from get_recommendations.

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
    distances, indices = knn_model.kneighbors(
        query_tfidf, n_neighbors=n_neighbors_query
    )

    # Use iloc to get rows, copy to avoid SettingWithCopyWarning
    recommendations = data.iloc[indices[0]].copy()

    # Add distance column (smaller distance = better match)
    if distances is None:
        # Fallback if kneighbors returns None for distances
        # Create a dummy distance array
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


def get_anime_vector(
    title_query: str, dataframe: pd.DataFrame, vectorizer: TfidfVectorizer
) -> tuple[np.ndarray, str]:
    """
    Finds an anime by title and returns its TF-IDF vector.
    :param title_query: The title to search for.
    :param dataframe: The full anime DataFrame.
    :param vectorizer: The fitted TF-IDF vectorizer.
    :return: A tuple containing the TF-IDF vector and the true title.
    """
    # Case-insensitive search
    match = dataframe[
        dataframe["title"].str.contains(title_query, case=False, na=False, regex=False)
    ]

    if match.empty:
        # Try English title
        match = dataframe[
            dataframe["english title"].str.contains(
                title_query, case=False, na=False, regex=False
            )
        ]

    if match.empty:
        logger.warning(f"'{title_query}' not found in database.")
        return None, None

    # Take the first match
    row = match.iloc[0]
    text_feature = row["combined_features"]

    # Vectorize
    vector = vectorizer.transform([text_feature])
    return vector, row["title"]


def get_user_history_vectors(
    history_titles: list[str], df: pd.DataFrame, vectorizer: TfidfVectorizer
) -> tuple[np.ndarray, list[str]]:
    """
    Converts a list of titles into a matrix of vectors.
    :param history_titles: List of titles to convert.
    :param df: The full anime DataFrame.
    :param vectorizer: The fitted TF-IDF vectorizer.
    :return: A tuple containing the matrix of vectors and the list of found titles.
    """
    vectors = []
    found_titles = []

    for title in history_titles:
        vec, true_title = get_anime_vector(title, df, vectorizer)
        if vec is not None:
            vectors.append(vec)
            found_titles.append(true_title)

    if not vectors:
        return None, []

    return vstack(vectors), found_titles


def calculate_hybrid_recommendation_score(
    candidates_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates a final weighted score for recommendation candidates based on Similarity, Popularity, and Quality.
    :param candidates_dataframe: DataFrame containing the candidates to score.
    :return: DataFrame with the candidates sorted by score.
    """
    df_scored = candidates_dataframe.copy()

    # 1. Similarity Score (0 to 1)
    if "average_similarity_score" not in df_scored.columns:
        if (
            "similarity_sum" in df_scored.columns
            and "frequency_count" in df_scored.columns
        ):
            df_scored["average_similarity_score"] = (
                df_scored["similarity_sum"] / df_scored["frequency_count"]
            )
        elif "distance" in df_scored.columns:
            df_scored["average_similarity_score"] = 1.0 - df_scored["distance"].clip(
                0, 1
            )
        else:
            df_scored["average_similarity_score"] = 0.0

    # 2. Popularity Score (Log-normalized Favorites)
    raw_favorites_count = df_scored["favorites_count"].fillna(0)
    log_transformed_favorites = np.log1p(raw_favorites_count)
    max_log_favorites = log_transformed_favorites.max()

    if max_log_favorites == 0:
        max_log_favorites = 1.0

    df_scored["normalized_popularity_score"] = (
        log_transformed_favorites / max_log_favorites
    )

    # 3. Quality Score (Normalized MAL Score)
    raw_mal_score = df_scored["myanimelist_score"].fillna(0)
    df_scored["normalized_quality_score"] = raw_mal_score / 10.0

    # 4. Frequency Bonus (Only for Multi-Query)
    if "frequency_count" in df_scored.columns:
        max_frequency = df_scored["frequency_count"].max()
        if max_frequency == 0:
            max_frequency = 1
        df_scored["normalized_frequency_bonus"] = (
            df_scored["frequency_count"] / max_frequency
        )

        # Weighted Sum for Multi-Query
        # 40% Sim + 20% Freq + 20% Pop + 20% Quality
        df_scored["final_hybrid_score"] = (
            0.4 * df_scored["average_similarity_score"]
            + 0.2 * df_scored["normalized_frequency_bonus"]
            + 0.2 * df_scored["normalized_popularity_score"]
            + 0.2 * df_scored["normalized_quality_score"]
        )
    else:
        # Fallback / Centroid logic
        df_scored["final_hybrid_score"] = (
            0.5 * df_scored["average_similarity_score"]
            + 0.3 * df_scored["normalized_popularity_score"]
            + 0.2 * df_scored["normalized_quality_score"]
        )

    return df_scored.sort_values("final_hybrid_score", ascending=False)


def recommend_by_history(
    history_titles: list[str],
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    knn: NearestNeighbors,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend anime based on the user's watch history using Multi-Query Aggregation.
    """
    vectors, found_titles = get_user_history_vectors(history_titles, df, vectorizer)

    if vectors is None:
        return pd.DataFrame()

    candidates_map = {}

    # Iterate over each history item's vector
    for i in range(vectors.shape[0]):
        vec = vectors.getrow(i)
        dists, idxs = knn.kneighbors(vec, n_neighbors=min(20, knn.n_samples_fit_))

        for dist, idx in zip(dists[0], idxs[0]):
            row = df.iloc[idx]
            anime_title = row["title"]

            # Franchise Filtering: Exclude if title is too similar to any history item
            is_franchise_duplicate = any(
                history_item.lower() in anime_title.lower()
                or anime_title.lower() in history_item.lower()
                for history_item in found_titles
            )

            if anime_title in found_titles or is_franchise_duplicate:
                continue

            similarity_score = 1.0 - dist

            if anime_title not in candidates_map:
                candidates_map[anime_title] = {
                    "row": row,
                    "similarity_sum": 0,
                    "frequency_count": 0,
                    "min_distance": 1.0,
                }

            candidates_map[anime_title]["similarity_sum"] += similarity_score
            candidates_map[anime_title]["frequency_count"] += 1
            candidates_map[anime_title]["min_distance"] = min(
                candidates_map[anime_title]["min_distance"], dist
            )

    if not candidates_map:
        return pd.DataFrame()

    candidate_anime_list = []
    for title, data in candidates_map.items():
        candidate_anime_list.append(
            {
                "title": title,
                "genres": data["row"].get("genres"),
                "themes": data["row"].get("themes"),
                "similarity_sum": data["similarity_sum"],
                "frequency_count": data["frequency_count"],
                "best_distance": data["min_distance"],
                "favorites_count": data["row"].get("favorites", 0),
                "myanimelist_score": data["row"].get("score", 0),
                "strategy": "Multi-Query",
            }
        )

    df_candidates = pd.DataFrame(candidate_anime_list)
    df_ranked = calculate_hybrid_recommendation_score(df_candidates)

    # Cleanup memory
    gc.collect()

    return df_ranked.head(top_k)
