"""
RecommendationHaki (見聞色) is an intelligent anime discovery engine
that uses Observation Haki (Matrix Factorization & KNN) to predict your next favorite show.
"""

import sys
from pathlib import Path
from string import Template

import pandas as pd
import streamlit as st
import html
import subprocess

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


# Helper to load templates
def load_html_template(filename: str) -> str:
    """
    Loads a template file from the templates directory.

    :param filename: The name of the template file to load.
    :return: The content of the template file.
    """
    template_path = Path(__file__).parent / "templates" / filename
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def format_number(num):
    """Formats a number with K/M suffixes."""
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


# Streamlit app setup
st.set_page_config(page_title=config.app.name, page_icon=config.paths.get("favicon", "app/assets/favicon.png"), layout="wide")

# Load custom styles
css_path = Path(config.paths.css)
if css_path.exists():
    with open(file=css_path, mode="r", encoding="utf-8") as f:
        custom_css = f.read()
        
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/ionicons/5.5.2/collection/components/icon/icon.min.css">',
        unsafe_allow_html=True
    )
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

# Initialize session state for navigation
if "page" not in st.session_state:
    # Default to landing page
    st.session_state.page = "landing"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_contribution_stats():
    """
    Fetches contribution statistics from git history.
    Returns a dictionary mapping author names (lowercase) to commit counts.
    """
    try:
        # Get shortlog with commit counts
        result = subprocess.check_output(
            ["git", "shortlog", "-sn", "--all"], 
            stderr=subprocess.STDOUT,
            text=True
        )
        stats = {}
        for line in result.strip().split('\n'):
            if not line.strip():
                continue
            # Format is usually: "   25  Author Name"
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                count, name = parts
                stats[name.lower()] = int(count)
        return stats
    except Exception as e:
        print(f"Error fetching git stats: {e}")
        return {}


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
    """Display the list of recommended anime as a grid of cards."""
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
    for idx, row in recommendations.iterrows():

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
            "title_japanese": html.escape(str(row.get("japanese title"))) if pd.notna(row.get("japanese title")) else "",
            "score": html.escape(f"{row.get('score', 'N/A')}"),
            "rating": html.escape(str(row.get("content rating", "N/A"))),
            "image": html.escape(str(image_url)),
            "synopsis": html.escape(synopsis[:200] + "..."),
            "genres": html.escape(str(row.get("genres", "Anime"))),
            "episodes": html.escape(str(row.get("episodes", "?")).replace(".0", "")),
            "duration": html.escape(str(row.get("duration", "Unknown")).replace(" min per ep", "m").replace(" hr", "h").replace(" min", "m")),
            "favorites": format_number(row.get("favorites", 0))
        }
        
        cards_html += card_html_template.substitute(context)

    # Wrap in grid container
    full_html = f'<div class="anime-grid">{cards_html}</div>'
    st.markdown(full_html, unsafe_allow_html=True)


# Landing Page
if st.session_state.page == "landing":
    import base64

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Inject Background Image
    bg_path = Path(__file__).parent / "assets" / "background.png"
    if bg_path.exists():
        bin_str = get_base64_image(bg_path)
        try:
             bg_template = Template(load_html_template("background_style.html"))
             page_bg_img = bg_template.substitute(bin_str=bin_str)
             st.markdown(page_bg_img, unsafe_allow_html=True)
        except FileNotFoundError:
             st.warning("Background template not found.")

    try:
        hero_html = load_html_template("hero.html")
        st.markdown(hero_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Hero template not found.")

    # Display logo and button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("ACTIVATE HAKI", use_container_width=True):
             st.session_state.page = "recommendations"
    
    # Contributors Section
    contribution_stats = get_contribution_stats()
    
    # Load contributors from config
    contributors_data = config.contributors

    try:
        contributor_item_template = Template(load_html_template("contributor_item.html"))
        contributors_wrapper_template = Template(load_html_template("contributors_wrapper.html"))
        
        items_html = ""
        for c in contributors_data:
            # Match name dynamically (case-insensitive)
            commit_count = contribution_stats.get(c['name'].lower(), 0)
            
            # Fallback for known variations if needed, or rely on normalization
            if commit_count == 0 and c['name'].lower() == "aditi mane":
                 commit_count = contribution_stats.get("aditi mane", 0)

            tooltip_text = f"{c['name']} • {commit_count} contributions"
            
            items_html += contributor_item_template.substitute(
                github=c["github"],
                tooltip_text=tooltip_text,
                image=c["image"],
                name=c["name"]
            )
            
        contributors_html = contributors_wrapper_template.substitute(content=items_html)
        st.markdown(contributors_html, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Contributor templates not found.")

# Recommendations Page
else:
    if st.button("Home", icon=":material/home:"):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown('<h1 style="margin-bottom: 0;">RECOMMENDATION HAKI</h1>', unsafe_allow_html=True)

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

    if st.button("Get Recommendations", use_container_width=True):
        if not user_query:
            st.warning("Please enter a valid query to get recommendations.")
        else:
            st.write("### Recommendations based on your input:")

            with st.spinner("Finding anime for you..."):
                results = engine.recommend_by_description(
                    user_query,
                    tfidf_vectorizer,
                    knn_model,
                    data,
                    top_n=num_recommendations,
                )

            display_recommendations(results)
