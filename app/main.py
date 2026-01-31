"""
RecommendationHaki (見聞色) is an intelligent anime discovery engine
that uses Observation Haki (Matrix Factorization & KNN) to predict your next favorite show.
"""

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to sys.path for streamlit to find modules if run from app/
# Assuming running via `streamlit run app/main.py` from root, or `streamlit run main.py` from app/
# Best to ensure root is in path.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src import logger
from src.config import config
from src.pipeline import inference as engine

# Streamlit app setup
st.set_page_config(page_title=config.app.name, page_icon="🎬")

# Load custom styles
css_path = Path(config.paths.css)
if css_path.exists():
    with open(file=css_path, mode="r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state for navigation
if "page" not in st.session_state:
    # Default to landing page
    st.session_state.page = "landing"


# Cache triggers
@st.cache_resource
def load_resources():
    """
    Load the pre-trained model, vectorizer, and processed data.
    """
    try:
        data = engine.load_processed_data()
        knn_model, tfidf_vectorizer = engine.load_models()
        return data, knn_model, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.info(
            "Run 'python src/pipeline/training_pipeline.py' to generate model artifacts."
        )
        st.stop()


# Load data and model
data, knn_model, tfidf_vectorizer = load_resources()


def display_recommendations(recommendations: pd.DataFrame):
    """Display the list of recommended anime."""
    if recommendations.empty:
        st.warning("No recommendations found. Please try a different query.")
        return

    columns_to_show = [
        "genres",
        "synopsis",
        "studio",
        "demographic",
        "source",
        "score",
        "episodes",
        "release year",
    ]

    for _, row in recommendations.iterrows():
        title = row["title"]
        with st.expander(f"**{title}**"):
            # Dynamic column display
            for col in columns_to_show:
                if col in row.index and pd.notna(row[col]):
                    label = col.replace("_", " ").title()
                    st.write(f"**{label}:** {row[col]}")


# Landing Page
if st.session_state.page == "landing":
    st.title(f"Welcome to {config.app.name}")
    st.markdown("### *Predicting your next favorite anime before you even know it.*")

    st.caption(
        """
        **RecommendationHaki (見聞色)** is an intelligent anime discovery engine designed to cure the modern plague of decision paralysis.
        
        Powered by high-dimensional mathematics and the **Jikan API**, it analyzes the hidden patterns in your viewing history to sense exactly what you're craving next.
        """
    )
    
    # Display logo if available
    logo_path = Path(config.paths.logo)
    if logo_path.exists():
        st.image(str(logo_path), width=200)

    st.caption(
        """
        *Don't just watch anime. Sense it.*
        """
    )
    
    if st.button("Activate Haki (Start Recommendations)"):
        st.session_state.page = "recommendations"

    # Contributors section removed as it's not in the new config.

# Recommendations Page
else:
    st.title(config.app.name)

    query_col, num_col = st.columns([4, 1])
    with query_col:
        user_query = st.text_input(
            "Describe a plot! Let's see if we can find something that matches that."
        ).strip()
    with num_col:
        num_recommendations = st.number_input(
            "No. of results:",
            min_value=1,
            max_value=20,
            value=config.model.top_k_recommendations,
        )

    if st.button("Get Recommendations"):
        if not user_query:
            st.warning("Please enter a valid query to get recommendations.")
        else:
            st.write("### Recommendations based on your input:")

            with st.spinner("Finding anime for you..."):
                results = engine.get_recommendations(
                    user_query,
                    tfidf_vectorizer,
                    knn_model,
                    data,
                    top_n=num_recommendations,
                )

            display_recommendations(results)
