"""
Configuration settings for the Jikan API data collection pipeline.

This file centralizes hardcoded values to make the collector easy to configure
and understand.

References:
    - Jikan API v4 Documentation: https://docs.api.jikan.moe/
    - Rate Limits: 3 requests/second (per IP) and 60 requests/minute.
      See: https://jikan.moe/help
"""

import os

# ==========================================
# API Connection Settings
# ==========================================

# Base URL for Jikan v4
BASE_URL = "https://api.jikan.moe/v4"

# Rate Limiting
# Jikan allows ~3 requests/second. 
# We use a 1.0s delay (1 request/sec) to be extremely safe and avoid 429s.
RATE_LIMIT_DELAY = 1.0 

# Time to sleep (in seconds) if we hit a 429 (Too Many Requests) error.
# Jikan bans are usually temporary, but a longer backoff is polite.
ERROR_SLEEP_TIME = 5.0

# ==========================================
# Pagination & Fetching Settings
# ==========================================

# Maximum number of items Jikan returns per page.
ITEMS_PER_PAGE = 25

# Standard Jikan filter for top anime (e.g. 'bypopularity', 'favorite', 'airing')
# 'bypopularity' corresponds to "Top Anime" sorted by member count.
DEFAULT_FILTER_TYPE = "bypopularity"

# Default fetch settings
# 20 pages * 25 items = 500 records.
# Increase this to fetch more of the catalogue.
DEFAULT_START_PAGE = 1
DEFAULT_PAGE_LIMIT = 200

# ==========================================
# File System Paths
# ==========================================

# Directory for raw data storage relative to the project root
# We assume the script is run from project root, but we can make this absolute if needed.
DATA_RAW_DIR = os.path.join("data", "raw")

# Name of the persistent master database file
MASTER_DB_FILENAME = "anime_master_db.csv"
MASTER_DB_PATH = os.path.join(DATA_RAW_DIR, MASTER_DB_FILENAME)
