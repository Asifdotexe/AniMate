"""
RecommendationHaki (見聞色) is an intelligent anime discovery engine
that uses Observation Haki (Matrix Factorization & KNN) to predict your next favorite show.
"""

import sys
from pathlib import Path
from string import Template

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


# Streamlit app setup
st.set_page_config(page_title=config.app.name, page_icon="🎬", layout="wide")

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

    st.markdown('<div class="anime-grid">', unsafe_allow_html=True)
    
    # Load card template
    try:
        card_html_template = Template(load_html_template("card.html"))
    except FileNotFoundError:
        st.error("Template 'card.html' not found!")
        return

    
    # Render all cards in a single string to let CSS Grid handle the layout
    cards_html = ""
    for idx, row in recommendations.iterrows():
        # Prepared variables for template
        context = {
            "title": row.get("title", "Unknown Title"),
            "title_japanese": row.get("japanese title") if pd.notna(row.get("japanese title")) else "",
            "score": f"{row.get('score', 'N/A')}",
            "rating": row.get("content rating", "N/A"),
            "image": row.get("image url", "https://via.placeholder.com/300x450?text=No+Image"),
            "synopsis": (row.get("synopsis", "No synopsis available.") or "")[:200] + "...",
            "genres": row.get("genres", "Anime")
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
    bg_path = Path("app/assets/background.png")
    if bg_path.exists():
        bin_str = get_base64_image(bg_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background: transparent;
        }}
        .background-layer {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(1.5px); /* Slight blur */
            opacity: 0.25;     /* Low opacity */
            z-index: -1;
        }}
        .background-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.8));
            z-index: -1;
        }}
        /* Remove pseudo-element approach if it conflicts */
        .stApp::before {{ content: none; }}
        </style>
        <div class="background-layer"></div>
        <div class="background-overlay"></div>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

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
                results = engine.get_recommendations(
                    user_query,
                    tfidf_vectorizer,
                    knn_model,
                    data,
                    top_n=num_recommendations,
                )

            display_recommendations(results)
