"""
History based anime recommendation page.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils import (config, display_recommendations, inject_custom_css,
                       load_resources)
from src.pipeline import inference as engine

# Page Setup
st.set_page_config(
    page_title=f"History Based - {config.app.name}",
    page_icon=config.paths.get("favicon", "app/assets/favicon.png"),
    layout="wide",
)

# Inject Styles
inject_custom_css()

# Load Resources
data, knn_model, tfidf_vectorizer = load_resources()

# Initialize Session State for History
if "history_list" not in st.session_state:
    st.session_state.history_list = []

# Navigation Button (Back to Home)
if st.button("Home", icon=":material/home:"):
    st.switch_page("main.py")

st.markdown(
    '<h1 style="margin-bottom: 0;">History Based Recommendation</h1>',
    unsafe_allow_html=True,
)
st.markdown("_Build your watch history to get personalized recommendations._")

# Search & add section
st.write("### 1. Build your history")
search_col, add_col = st.columns([4, 1])

with search_col:
    search_query = st.text_input(
        "Search for an anime to add:", placeholder="e.g. Naruto"
    ).strip()

if search_query:
    results = engine.search_anime_titles(search_query, data, limit=5)

    if results:
        # Display search results nicely
        for item in results:
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1:
                    if item["image_url"]:
                        st.image(item["image_url"], width=60)
                with c2:
                    st.markdown(f"**{item['title']}**")
                    if item["english_title"]:
                        st.caption(f"English: {item['english_title']}")
                    st.caption(f"Year: {item['year']}")
                with c3:
                    # Check if already in list
                    if item["title"] in st.session_state.history_list:
                        st.success("Added")
                    else:
                        if st.button("Add (+)", key=f"add_{item['title']}"):
                            st.session_state.history_list.append(item["title"])
                            st.rerun()
    else:
        st.info("No anime description found matching that title.")

# Manage list section
if st.session_state.history_list:
    st.write("### 2. Your Watch History")
    st.caption("We will recommend anime similar to the combination of these titles.")

    # Display chips/tags with remove button
    # Using columns for a wrapped list effect (simplified)
    for i, title in enumerate(st.session_state.history_list):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.info(title)
        with col2:
            if st.button("Remove (x)", key=f"rem_{i}"):
                st.session_state.history_list.pop(i)
                st.rerun()

    # Recommendations section
    st.divider()
    if st.button(
        "Get Recommendations Based on History", type="primary", use_container_width=True
    ):
        with st.spinner("Analyzing your history..."):
            recs = engine.recommend_by_history(
                st.session_state.history_list, data, tfidf_vectorizer, knn_model
            )

        st.write("### Recommendations for you:")
        display_recommendations(recs)

else:
    st.info("Search and add anime above to start building your history.")
