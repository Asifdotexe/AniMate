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


# Resolve the dataset path
def _latest_final_csv() -> Path:
    files = sorted(
        FINAL_DATA_DIR.glob("anime_dump_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else FINAL_DATA_DIR / "anime_dump_25072024.csv"


FINAL_DATA_PATH = Path(os.getenv("ANIMATE_FINAL_DATA_PATH", _latest_final_csv()))

# Centralize other settings
MAX_FEATURES = 5000
N_NEIGHBORS = 5
