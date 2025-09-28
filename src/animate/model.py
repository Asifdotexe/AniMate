# src/animate/model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import psutil
import gc

from config import (
    PROCESSED_DATA_PATH,
    VECTORIZER_PATH,
    KNN_MODEL_PATH,
    MAX_FEATURES,
    N_NEIGHBORS
)
from data_processing import preprocess_text


def train_model():
    """Train the TF-IDF vectorizer and k-NN model and save them to disk."""
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print("Training TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english', max_features=MAX_FEATURES
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_synopsis'])

    print("Training k-NN model...")
    knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine')
    knn_model.fit(tfidf_matrix)

    # Ensure the models directory exists
    MODELS_DIR = VECTORIZER_PATH.parent
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)

    print(f"Saving k-NN model to {KNN_MODEL_PATH}...")
    joblib.dump(knn_model, KNN_MODEL_PATH)

    print("Model training complete!")


class RecommendationEngine:
    """Handles loading the model and making predictions."""

    def __init__(self):
        """Load the dataset and pre-trained models into memory."""
        try:
            self.data = pd.read_csv(PROCESSED_DATA_PATH).set_index('title')
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.model = joblib.load(KNN_MODEL_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Model files not found. Please run `poetry run python src/animate/model.py` to train the model."
            )

    def get_recommendations(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """Get anime recommendations based on a user query.

        :param query: The user input query describing the desired anime.
        :param top_n: The number of recommendations to return.
        :return: A DataFrame of the top N recommended anime.
        """
        if not query.strip():
            return pd.DataFrame()

        # Preprocess the user's query
        processed_query = preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])

        # Find the nearest neighbors
        distances, indices = self.model.kneighbors(query_vector, n_neighbors=top_n + 5)

        # Get the recommended anime titles
        recommended_titles = self.data.index[indices[0]]

        # Filter out any anime that might be too similar to the query itself
        filtered_titles = [
            title for title in recommended_titles if query.lower() not in title.lower()
        ]

        # Ensure we have enough recommendations
        if not filtered_titles:
            filtered_titles = recommended_titles

        # Retrieve full data for recommendations and sort by score
        recommendations = self.data.loc[filtered_titles].head(top_n)
        return recommendations.sort_values(by='score', ascending=False)


if __name__ == "__main__":
    train_model()