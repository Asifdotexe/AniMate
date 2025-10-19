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
FINAL_DATA_DIR = DATA_DIR / "final"

# Define the full path to the final dataset used by the application.
FINAL_DATA_PATH = FINAL_DATA_DIR / "AnimeData_25092024.csv"

# Centralize other settings
MAX_FEATURES = 5000
N_NEIGHBORS = 5
