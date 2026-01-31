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
    
    # Grid Layout using Streamlit Columns (since generic HTML grid is hard to inject fully with clickables)
    # Actually, for pure aesthetics, st.markdown HTML is better.
    # Note: Streamlit buttons inside HTML are tricky. We will use a visual card only for now.
    
    cols = st.columns(3) # 3 columns for grid
    
    for idx, row in recommendations.iterrows():
        with cols[idx % 3]:
            title = row["title"]
            score = row.get("score", "N/A")
            synopsis = row.get("synopsis", "No synopsis available.")[:150] + "..."
            genres = row.get("genres", "Anime")
            
            # Using st.container for card styling wrapper
            with st.container():
                st.markdown(f"""
                <div class="anime-card">
                    <div class="card-content">
                        <div class="card-title">{title}</div>
                        <div class="card-meta">
                            <span>{genres}</span>
                            <span class="match-score">{score} Match</span>
                        </div>
                        <p style="font-size: 0.8rem; color: #ccc; margin-top: 10px;">{synopsis}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable details
                with st.expander("Explore"):
                     st.write(f"**Studio:** {row.get('studio', 'N/A')}")
                     st.write(f"**Episodes:** {row.get('episodes', 'N/A')}")
                     st.write(f"**Year:** {row.get('release year', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)


# Landing Page
if st.session_state.page == "landing":
    st.markdown("""
<div class="hero-container">
<h1>Recommendation Haki</h1>
<div class="hero-subtitle">見聞色 (Observation Haki)</div>
<p class="pitch-text">
<br/>
"Predicting your next favorite anime before you even know it."
</p>
</div>
""", unsafe_allow_html=True)

    # Display logo and button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("ACTIVATE HAKI", use_container_width=True):
             st.session_state.page = "recommendations"

# Recommendations Page
else:
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
