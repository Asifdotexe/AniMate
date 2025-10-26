"""
This module contains the core logic for the recommendation engine.
It handles data loading, preprocessing, model building, and recommendations.
"""
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from animate.config import FINAL_DATA_PATH, MAX_FEATURES, N_NEIGHBORS

# --- Download NLTK data if not present ---
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)

class RecommendationEngine:
    """
    The core recommendation engine for AniMate.
    Encapsulates all logic for loading, processing, and recommending anime.
    """

    def __init__(self):
        """Initializes the engine by loading data and building the model."""
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))
        self.data = self._load_data()
        self.knn_model, self.tfidf_vectorizer = self._build_model()

    @st.cache_data
    def _load_data(_self) -> pd.DataFrame:
        """
        Loads the final anime dataset from the specified path.
        Uses Streamlit's caching to avoid reloading from disk on every run.
        The `_self` parameter is used because this is a method inside a class.
        """
        dtypes = {
            "title": "category", "other_name": "category", "genres": "category",
            "synopsis": "string", "studio": "category", "demographic": "category",
            "source": "category", "duration_category": "category",
            "total_duration_hours": "float32", "score": "float32", "image_url": "string",
        }
        return pd.read_parquet(FINAL_DATA_PATH, usecols=dtypes.keys(), dtype=dtypes)

    def _stemming_tokenizer(self, text: str) -> list[str]:
        """Custom analyzer for TfidfVectorizer."""
        tokens = word_tokenize(text.lower())
        return [
            self.stemmer.stem(word)
            for word in tokens
            if word.isalpha() and word not in self.stop_words
        ]

    @st.cache_resource
    def _build_model(_self) -> tuple[NearestNeighbors, TfidfVectorizer]:
        """
        Builds the TF-IDF vectorizer and k-NN model.
        Uses Streamlit's resource caching to keep the trained models in memory.
        """
        st.write("First time setup: Building recommendation model...")
        tfidf_vectorizer = TfidfVectorizer(
            analyzer=_self._stemming_tokenizer, max_features=MAX_FEATURES
        )

        with st.spinner("Processing thousands of anime synopses... this might take a moment."):
            tfidf_matrix = tfidf_vectorizer.fit_transform(_self.data["synopsis"].fillna(""))

        knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="cosine").fit(tfidf_matrix)
        st.success("Model built successfully!")
        return knn_model, tfidf_vectorizer

    def get_recommendations(self, query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Generates anime recommendations based on a user's query.

        :param query: The user's input query.
        :param top_n: The number of recommendations to return.
        :return: A DataFrame of recommended anime.
        """
        # Preprocess the user's query
        processed_query = " ".join(self._stemming_tokenizer(query))
        query_vector = self.tfidf_vectorizer.transform([processed_query])

        # Find the nearest neighbors
        _, indices = self.knn_model.kneighbors(query_vector, n_neighbors=top_n + 5)

        # Filter and return recommendations
        recommendations = self.data.iloc[indices[0]]
        filtered_recs = recommendations[
            ~recommendations["title"].str.contains(query, case=False, na=False, regex=False)
        ]

        if filtered_recs.empty:
            filtered_recs = recommendations

        return filtered_recs.head(top_n).sort_values(by="score", ascending=False)
