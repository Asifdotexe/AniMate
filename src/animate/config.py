"""
This module centralizes configuration for the AniMate application,
including file paths and constants.
"""

from pathlib import Path

# Define the root directory of the project.
# Path(__file__) is this file's path.
# .parent is src/animate/
# .parent.parent is src/
# .parent.parent.parent is the project root (animate/).
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define other key directories relative to the project root.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Base URL from webscraping
MYANIMELIST_BASE_URL = "https://myanimelist.net/anime/genre/"

# used in src/animate/app.py
# This is the maxiumum amount of features for TFIDF vectorzor
MAX_FEATURES = 5000
# Number of neighbors
N_NEIGHBORS = 5

# used in scripts/web_scraper.py
SCRAPER_REQUEST_TIMEOUT = 10
SCRAPER_MAX_WORKERS = 10

# used in scripts/data_cleaner.py
# Assumed average duration for calculation
AVG_EPISODE_DURATION_MINS = 23
# Case-insensitive list of genres that we remove that contain explicit content
MATURE_GENRES = frozenset(['hentai', 'ecchi', 'erotica', 'adult themes'])
# These columns are essential for the current recommendation engine,
# hence this is the bare minimum data to make the datapoint useful.
REQ_RAW_COLUMNS = frozenset([
    'title', 'release_year', 'synopsis', 'episodes', 'genres', 'studio', 'source', 'demographic', 'rating'
])

FINAL_APP_COLUMNS = frozenset([
        'title', 'genres', 'synopsis', 'studio', 'demographic',
        'source', 'duration_category', 'total_duration_hours', 'score'
        # FIXME: Add 'other_name', 'image_url' here if present in available_cols
    ])