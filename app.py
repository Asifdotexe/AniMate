"""
AniMate is a Python-based anime recommendation system
that utilizes natural language processing (NLP) to suggest anime based on user preferences.
"""

import random

import pandas as pd
import streamlit as st
import yaml

from src import utils
from src.inference import engine

# Load configuration
with open(file="config.yaml", mode="r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Streamlit app setup
st.set_page_config(page_title=config["app"]["page_title"])

# Load custom styles
with open(file="styles.css", mode="r", encoding="utf-8") as f:
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
    model_dir = "models"
    try:
        data = engine.load_processed_data(model_dir)
        knn_model, tfidf_vectorizer = engine.load_models(model_dir)
        return data, knn_model, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.info("Run 'python src/pipeline/train.py' to generate model artifacts.")
        st.stop()


# Load data and model
data, knn_model, tfidf_vectorizer = load_resources()


def display_memory():
    """
    Display memory usage information.
    """
    st.write(f"Memory usage: {utils.get_memory_usage()}%")


def display_recommendations(recommendations: pd.DataFrame):
    """Display the list of recommended anime."""
    if recommendations.empty:
        st.warning("No recommendations found. Please try a different query.")
        return

    columns_to_show = [
        "genres", "synopsis", "studio", "demographic", 
        "source", "score", "episodes", "release year"
    ]

    for _, row in recommendations.iterrows():
        title = row['title']
        with st.expander(f"**{title}**"):
            # Dynamic column display
            for col in columns_to_show:
                if col in row.index and pd.notna(row[col]):
                    label = col.replace("_", " ").title()
                    st.write(f"**{label}:** {row[col]}")


# Landing Page
if st.session_state.page == "landing":
    st.title(f"Welcome to {config['app']['title']}!")

    st.caption(
        """AniMate is a Python-based anime recommendation system
        that utilizes natural language processing (NLP) to suggest anime based on user preferences."""
    )

    st.caption(
        """
        If you enjoy our recommendations, please consider starring our repository on GitHub ⭐!
        """
    )

    if st.button("Recommend Me Something"):
        st.session_state.page = "recommendations"

    display_memory()

    st.subheader("Contributors")
    contributors = config["contributors"]

    cols = st.columns(len(contributors))
    for col, contributor in zip(cols, contributors):
        with col:
            st.markdown(
                f"[![Contributor Icon]({contributor['image']})]({contributor['github']})"
            )
            st.caption(contributor["name"])

# Recommendations Page
else:
    st.title(config["app"]["title"])

    display_memory()

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
            value=config["model"].get("default_top_n", 5),
        )

    if st.button("Get Recommendations"):
        if not user_query:
            st.warning("Please enter a valid query to get recommendations.")
        else:
            st.write("### Recommendations based on your input:")
            
            with st.spinner(random.choice(config["app"]["loading_phrases"])):
                results = engine.get_recommendations(
                    user_query,
                    tfidf_vectorizer,
                    knn_model,
                    data,
                    top_n=num_recommendations,
                )
            
            display_recommendations(results)
