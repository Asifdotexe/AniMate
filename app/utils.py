"""
Module that contains the utils for the recommendation system (inference only).
"""

import html
from pathlib import Path
from string import Template

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.config import config
from src.pipeline import inference as engine


# Helper to load templates
def load_html_template(filename: str) -> str:
    """
    Loads a template file from the templates directory.

    :param filename: The name of the template file to load.
    :return: The content of the template file.
    """
    # Assuming utils.py is in app/utils.py, templates is in app/templates
    template_path = Path(__file__).parent / "templates" / filename
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def format_number(num: int | float) -> str:
    """Formats a number with K/M suffixes.

    :param num: The number to format.
    :return: The formatted number as a string.
    """
    if pd.isna(num):
        return "0"
    try:
        num = int(num)
    except (ValueError, TypeError):
        return "0"

    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}k"
    return str(num)


# Cache triggers
@st.cache_resource
def load_resources() -> tuple[pd.DataFrame, NearestNeighbors, TfidfVectorizer]:
    """
    Load the pre-trained model, vectorizer, and processed data.

    :return: A tuple containing the processed data, k-NN model, and TF-IDF vectorizer.
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


def display_recommendations(recommendations: pd.DataFrame) -> None:
    """Display the list of recommended anime as a grid of cards.

    :param recommendations: DataFrame containing the recommended anime.
    """
    if recommendations.empty:
        st.warning("No recommendations found. Please try a different query.")
        return

    # Load card template
    try:
        card_html_template = Template(load_html_template("card.html"))
    except FileNotFoundError:
        st.error("Template 'card.html' not found!")
        return

    # Render all cards in a single string to let CSS Grid handle the layout
    cards_html = ""
    for _, row in recommendations.iterrows():

        synopsis = row.get("synopsis")
        if pd.isna(synopsis) or synopsis is None:
            synopsis = "No synopsis available."
        else:
            synopsis = str(synopsis)

        # Prepared variables for template
        image_url = row.get("image url", "")
        if not image_url or not str(image_url).startswith(("http://", "https://")):
            image_url = "https://via.placeholder.com/300x450?text=No+Image"

        # Prepare context variables
        context = {
            "title": html.escape(str(row.get("title", "Unknown Title"))),
            "title_japanese": (
                html.escape(str(row.get("japanese title")))
                if pd.notna(row.get("japanese title"))
                else ""
            ),
            "score": html.escape(f"{row.get('score', 'N/A')}"),
            "rating": html.escape(str(row.get("content rating", "N/A"))),
            "image": html.escape(str(image_url)),
            "synopsis": html.escape(synopsis[:200] + "..."),
            "genres": html.escape(str(row.get("genres", "Anime"))),
            "episodes": html.escape(str(row.get("episodes", "?")).replace(".0", "")),
            "duration": html.escape(
                str(row.get("duration", "Unknown"))
                .replace(" min per ep", "m")
                .replace(" hr", "h")
                .replace(" min", "m")
            ),
            "favorites": format_number(row.get("favorites", 0)),
        }

        cards_html += card_html_template.substitute(context)

    # Wrap in grid container
    full_html = f'<div class="anime-grid">{cards_html}</div>'
    st.markdown(full_html, unsafe_allow_html=True)


def inject_custom_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    css_path = Path(config.paths.css)
    if css_path.exists():
        with open(file=css_path, mode="r", encoding="utf-8") as f:
            custom_css = f.read()

        st.markdown(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/ionicons/5.5.2/collection/components/icon/icon.min.css">',
            unsafe_allow_html=True,
        )
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    # Hide Sidebar globally
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
