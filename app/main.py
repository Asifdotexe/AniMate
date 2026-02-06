"""
RecommendationHaki (見聞色) is an intelligent anime discovery engine.
Main Entry Point / Landing Page.
"""

import base64
import subprocess
import sys
from pathlib import Path
from string import Template

# Add project root to sys.path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st

from app.utils import config, inject_custom_css, load_html_template

# Streamlit app setup
st.set_page_config(
    page_title=config.app.name,
    page_icon=config.paths.get("favicon", "app/assets/favicon.png"),
    layout="wide",
)

# Inject Styles
inject_custom_css()


# Helper for git stats (kept local as it's specific to landing)
@st.cache_data(ttl=3600)
def get_contribution_stats():
    """Fetches contribution statistics from git history."""
    try:
        result = subprocess.check_output(
            ["git", "shortlog", "-sn", "--all"], stderr=subprocess.STDOUT, text=True
        )
        stats = {}
        for line in result.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                count, name = parts
                stats[name.lower()] = int(count)
        return stats
    except Exception as e:
        print(f"Error fetching git stats: {e}")
        return {}


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

# Hero Section
try:
    hero_html = load_html_template("hero.html")
    st.markdown(hero_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("Hero template not found.")

# Navigation Buttons
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("Description Based", use_container_width=True, type="primary"):
        st.switch_page("pages/description_based.py")

with col2:
    if st.button("History Based", use_container_width=True, type="primary"):
        st.switch_page("pages/history_based.py")

# Contributors Section
contribution_stats = get_contribution_stats()
contributors_data = config.contributors

try:
    contributor_item_template = Template(load_html_template("contributor_item.html"))
    contributors_wrapper_template = Template(
        load_html_template("contributors_wrapper.html")
    )

    items_html = ""
    for c in contributors_data:
        commit_count = contribution_stats.get(c["name"].lower(), 0)
        # Fallback for known variations if needed
        if commit_count == 0 and c["name"].lower() == "aditi mane":
            commit_count = contribution_stats.get("aditi mane", 0)

        tooltip_text = f"{c['name']} • {commit_count} contributions"

        items_html += contributor_item_template.substitute(
            github=c["github"],
            tooltip_text=tooltip_text,
            image=c["image"],
            name=c["name"],
        )

    contributors_html = contributors_wrapper_template.substitute(content=items_html)
    st.markdown(contributors_html, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Contributor templates not found.")
