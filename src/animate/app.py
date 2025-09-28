# src/animate/app.py
import streamlit as st
import pandas as pd
import random
import psutil

from model import RecommendationEngine


st.set_page_config(
    page_title="AniMate",
    page_icon="✨",
    layout="wide"
)


@st.cache_resource
def load_engine():
    """Load the recommendation engine once and cache it."""
    return RecommendationEngine()


def monitor_memory():
    """Display the current memory usage in the sidebar."""
    with st.sidebar:
        st.header("System Status")
        st.write(f"Memory Usage: {psutil.virtual_memory().percent}%")

def display_recommendations(recommendations: pd.DataFrame):
    """Display the recommended anime in expanders.

    :param recommendations: DataFrame of recommended anime.
    """
    if recommendations.empty:
        st.warning("No recommendations found. Please try a different query.")
        return

    st.write("### Here are your recommendations:")
    for title, row in recommendations.iterrows():
        with st.expander(f"**{title.title()}**"):
            col1, col2 = st.columns([1, 3])

            with col1:
                if pd.notna(row.get('image_url')):
                    st.image(row['image_url'], caption=title.title(), width=150)

            with col2:
                details = {
                    "Other Name": row.get('other_name'),
                    "Genres": row.get('genres'),
                    "Synopsis": row.get('synopsis'),
                    "Studio": row.get('studio'),
                    "Score": row.get('score'),
                    "Duration": row.get('duration_category')
                }
                for key, value in details.items():
                    if pd.notna(value):
                        st.write(f"**{key}:** {value}")


def main():
    """The main function to run the Streamlit app."""
    st.title("Welcome to AniMate! 🎬")
    st.caption("Your personal guide to the world of anime.")

    monitor_memory()

    engine = load_engine()

    user_query = st.text_input(
        "Describe a plot or an anime you like, and I'll find something similar!",
        placeholder="e.g., a high school student gets magical powers and fights demons"
    )

    num_recommendations = st.sidebar.number_input(
        "Number of recommendations:", min_value=1, max_value=20, value=5, step=1
    )

    if st.button("Get Recommendations", type="primary"):
        if user_query.strip():
            loading_phrases = [
                "🔍 Searching the anime universe...",
                "✨ Summoning perfect recommendations...",
                "🚀 Launching into the world of anime to find your match...",
            ]
            with st.spinner(random.choice(loading_phrases)):
                recommendations = engine.get_recommendations(
                    user_query, num_recommendations
                )

            display_recommendations(recommendations)
        else:
            st.warning("Please enter a query to get recommendations.")

if __name__ == "__main__":
    main()