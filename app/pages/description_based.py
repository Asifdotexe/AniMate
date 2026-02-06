"""
Description based anime recommendation page.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils import config, display_recommendations, inject_custom_css, load_resources
from src.pipeline import inference as engine

# Page Setup
st.set_page_config(
    page_title=f"Description Based - {config.app.name}",
    page_icon=config.paths.get("favicon", "app/assets/favicon.png"),
    layout="wide",
)

# Inject Styles
inject_custom_css()

# Load Resources
data, knn_model, tfidf_vectorizer = load_resources()

# Navigation Button (Back to Home)
if st.button("Home", icon=":material/home:"):
    st.switch_page("main.py")

st.markdown(
    '<h1 style="margin-bottom: 0;">Description Based Recommendation</h1>',
    unsafe_allow_html=True,
)
st.markdown("_Describe a plot or feeling, and we'll find the anime that matches._")

query_col, num_col = st.columns([4, 1])
with query_col:
    user_query = st.text_input(
        "Enter your description:",
        placeholder="e.g. A psychological thriller about a brilliant student with a god complex...",
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
