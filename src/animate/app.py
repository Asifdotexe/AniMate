"""
This module contains the Streamlit UI and application logic for AniMate.
It acts as the main entry point for the web app.
"""

import cProfile
import io
import pstats
import random

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from animate.config import FINAL_DATA_PATH

# Download necessary NLTK data
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)

# Reason to pick Snowball over PorterStemer:
# https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg
stemmer = SnowballStemmer("english")

# Allows us to have progress bar for pandas .apply()
tqdm.pandas()


# Cache data loading and optimize dtypes
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load anime data from a CSV file and optimize data types.

    :returns: A DataFrame containing the anime data with optimized dtypes.
    """
    dtypes = {
        "title": "category",
        "other_name": "category",
        "genres": "category",
        "synopsis": "string",
        "studio": "category",
        "demographic": "category",
        "source": "category",
        "duration_category": "category",
        "total_duration_hours": "float32",
        "score": "float32",
        "image_url": "string",
    }
    # Corrected the file path to be relative to the app.py location
    return pd.read_csv(
        FINAL_DATA_PATH, usecols=dtypes.keys(), dtype=dtypes
    )


stop_words = set(stopwords.words("english"))


def stemming_tokenizer(text: str) -> list[str]:
    """
    A custom tokenizer that tokenizes, removes stopwords, and stems the text.
    This will be passed to TfidfVectorizer.

    :param text: The input text to process.
    :return: A list of processed (stemmed) tokens.
    """
    tokens = word_tokenize(text.lower())
    return [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]


def preprocess_query_text(text: str) -> str:
    """
    Preprocess the user's query text. This is separate from the main
    vectorizer's analyzer to handle single string inputs efficiently.

    :param text: Ingests user's query
    :returns: Processed query
    """
    return " ".join(stemming_tokenizer(text))


# Cache the TF-IDF vectorization and k-NN model to avoid recomputation
@st.cache_resource
def vectorize_and_build_model(
    df: pd.DataFrame,
) -> tuple[NearestNeighbors, TfidfVectorizer]:
    """
    Vectorize the synopsis of the anime DataFrame and build a k-NN model.
    This is now highly optimized to preprocess text inside the vectorizer.

    :param df: The DataFrame containing anime data.
    :return: A tuple containing the k-NN model and the TF-IDF vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(analyzer=stemming_tokenizer, max_features=5000)

    with st.spinner(
        "Processing thousands of anime synopses... this might take a moment."
    ):
        tfidf_matrix = tfidf_vectorizer.fit_transform(df["synopsis"].fillna(""))

    knn_model = NearestNeighbors(n_neighbors=5, metric="cosine").fit(tfidf_matrix)
    return knn_model, tfidf_vectorizer


# Recommend anime using the k-NN model and TF-IDF vectorizer
def recommend_anime_knn(
    query: str,
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    knn_model: NearestNeighbors,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Recommend anime based on a user query using the k-NN model and TF-IDF vectorization.

    :param query: The user input query describing the desired anime.
    :param df: The DataFrame containing anime data.
    :param tfidf_vectorizer: The fitted TF-IDF vectorizer.
    :param knn_model: The fitted k-NN model.
    :param top_n: The number of recommendations to return.

    :return: A DataFrame containing the recommended anime titles and their attributes.
    """
    query_processed = preprocess_query_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    _, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n + 5)

    recommendations = df.iloc[indices[0]]
    filtered_recommendations = recommendations[
        ~recommendations["title"].str.contains(query, case=False, na=False)
    ]

    if filtered_recommendations.empty:
        filtered_recommendations = recommendations

    return filtered_recommendations.head(top_n)


# Full pipeline to get recommendations
def anime_recommendation_pipeline(user_query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Execute the full pipeline to recommend anime based on user input.

    :param user_query: The user input query describing the desired anime.
    :returns: A DataFrame containing the sorted recommended anime titles based on their score.
    """
    knn_model, tfidf_vectorizer = vectorize_and_build_model(data)
    anime_recommendations = recommend_anime_knn(
        user_query, data, tfidf_vectorizer, knn_model, top_n
    )
    return anime_recommendations.sort_values(by="score", ascending=False)


# Streamlit app
st.set_page_config(page_title="AniMate", layout="wide")

# Load custom styles
try:
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.css not found. The app will run with default styling.")


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "landing"  # Default to landing page

# Load data
data = load_data()

# Define loading phrases
loading_phrases = [
    "🔍 Searching for hidden gems in the anime universe...",
    "✨ Summoning the perfect anime recommendations...",
    "🎉 Gathering the coolest anime just for you...",
    "📚 Digging through the anime archives for you...",
    "🚀 Launching into the world of anime to find your match...",
    "🌟 Fetching the ultimate anime experience...",
    "🌀 Sifting through dimensions for the best recommendations...",
    "💫 Scouring the anime cosmos for your next favorite...",
]

# Landing Page
if st.session_state.page == "landing":
    st.title("Welcome to AniMate!")

    st.caption(
        """AniMate is a Python-based anime recommendation system that utilizes natural language processing (NLP)
        to suggest anime based on user preferences."""
    )

    st.caption(
        """
        If you enjoy our recommendations, please consider starring our repository on GitHub ⭐!
        """
    )

    if st.button("Recommend Me Something"):
        st.session_state.page = "recommendations"
        st.rerun()  # Use rerun to immediately switch pages

    st.subheader("Contributors")
    contributors = [
        {
            "github": "https://github.com/Asifdotexe",
            "image": "https://avatars.githubusercontent.com/u/115421661?v=4",
            "alt_name": "Asif Sayyed",
        },
        {
            "github": "https://github.com/PranjalDhamane",
            "image": "https://avatars.githubusercontent.com/u/131870182?v=4",
            "alt_name": "Pranjal Dhamane",
        },
        {
            "github": "https://github.com/tanvisivaraj",
            "image": "https://avatars.githubusercontent.com/u/132070958?v=4",
            "alt_name": "Tanvi Sivaraj",
        },
        {
            "github": "https://github.com/str04",
            "image": "https://avatars.githubusercontent.com/u/123924840?v=4",
            "alt_name": "Shrawani Thakur",
        },
        {
            "github": "https://github.com/aditimane07",
            "image": "https://avatars.githubusercontent.com/u/129670339?v=4",
            "alt_name": "Aditi Mane",
        },
    ]

    cols = st.columns(len(contributors))
    for col, contributor in zip(cols, contributors, strict=True):
        with col:
            st.markdown(
                f"[![Contributor Icon]({contributor['image']})]({contributor['github']})"
            )
            st.caption(contributor["alt_name"])

# Recommendations Page
else:
    st.title("AniMate")
    st.caption(
        """AniMate is a Python-based anime recommendation system that utilizes natural language processing (NLP)
        to suggest anime based on user preferences"""
    )

    query_section, number = st.columns([4, 1])
    with query_section:
        user_prompt = st.text_input(
            "Describe a plot! Let's see if we can find something that matches that."
        )
    with number:
        num_recommendations = st.number_input(
            "No. of results:", min_value=1, max_value=20, value=5
        )

    if st.button("Get Recommendations"):
        if user_prompt.strip():

            # --- PROFILING CODE START ---
            # 1. Initialize the profiler
            profiler = cProfile.Profile()
            profiler.enable()

            # 2. Run the recommendation pipeline
            with st.spinner(random.choice(loading_phrases)):
                recommended_animes = anime_recommendation_pipeline(
                    user_prompt, num_recommendations
                )

            # 3. Stop the profiler
            profiler.disable()
            # --- PROFILING CODE END ---

            st.write("### Recommendations based on your input:")

            if recommended_animes.empty:
                st.warning("No recommendations found. Please try a different query.")
            else:
                for index, row in recommended_animes.iterrows():
                    with st.expander(f"**{row['title'].title()}**"):
                        image_column, text_column = st.columns([1, 3])
                        with image_column:
                            if pd.notna(row["image_url"]):
                                st.image(
                                    row["image_url"],
                                    caption=row["title"].title(),
                                    width=100,
                                )
                        with text_column:
                            for column in [
                                "other_name",
                                "genres",
                                "synopsis",
                                "studio",
                                "demographic",
                                "source",
                                "duration_category",
                                "total_duration_hours",
                            ]:
                                value = row[column]
                                if pd.notna(value):
                                    st.write(
                                        f"**{column.replace('_', ' ').title()}:** {value}"
                                    )

            # --- PROFILING DISPLAY START ---
            # 4. Format and display the profiling statistics
            st.write("---")
            st.subheader("📈 Performance Profile")

            # Create a string stream to capture pstats output
            s = io.StringIO()
            # Sort stats by cumulative time
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats()

            # Display the stats in an expander
            with st.expander("Click to see profiling details"):
                st.code(s.getvalue())
            # --- PROFILING DISPLAY END ---

        else:
            st.warning("Please enter a valid query to get recommendations.")
