"""
This module centralizes configuration for the AniMate application,
including file paths and constants.
"""

import os
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
FINAL_DATA_DIR = DATA_DIR / "final"

# Base URL from webscraping
MYANIMELIST_BASE_URL = "https://myanimelist.net/anime/genre/"


def genre_url(genre_id: int) -> str:
    """Constructs a MyAnimeList genre URL with trailing slash.
    :param genre_id: Ingests the genre id
    :returns: URL to scrape
    """
    return f"{MYANIMELIST_BASE_URL}{genre_id}/"


# Resolve the dataset path
def _latest_final_csv() -> Path:
    files = sorted(
        FINAL_DATA_DIR.glob("anime_dump_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else FINAL_DATA_DIR / "anime_dump_25092024.csv"


FINAL_DATA_PATH = Path(os.getenv("ANIMATE_FINAL_DATA_PATH", _latest_final_csv()))

# used in src/animate/app.py
MAX_FEATURES = 5000
N_NEIGHBORS = 5

# used in scripts/web_scraper.py
SCRAPER_REQUEST_TIMEOUT = 10
SCRAPER_MAX_WORKERS = 10