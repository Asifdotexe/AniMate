"""
Module for processing raw anime data.
This script loads the latest raw data, cleans it, applies text preprocessing,
and saves the result to data/processed.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to sys.path to import src modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.preprocessing import preprocess_text
from src import config
from src.utils import setup_logging

logger = setup_logging("process_data")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.

    :param df: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    # Validate required columns
    required_columns = config.REQUIRED_COLUMNS
    # Be case-insensitive for columns
    df.columns = df.columns.str.strip().str.lower()
    
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Deduplicate and clean
    df = df.drop_duplicates(subset=["title"])
    df = df.dropna(subset=["synopsis"])

    if "episodes" in df.columns:
        df["episodes"] = df["episodes"].fillna("N/A")

    # Normalize Score/Rating
    # Prioritize 'rating', fallback to 'score', default to 0
    if "rating" in df.columns:
        df["score"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    elif "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    else:
        df["score"] = 0

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to create features for the model.

    :param df: Cleaned DataFrame.
    :return: DataFrame with additional feature columns.
    """
    tqdm.pandas(desc="Preprocessing Synopsis")
    df["stemmed_synopsis"] = df["synopsis"].progress_apply(preprocess_text)
    return df


def main():
    """
    Main execution flow for data processing.
    """
    raw_file = config.RAW_DATA_PATH
    processed_dir = config.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        logger.error(f"Error: Master DB not found at {raw_file}")
        return

    logger.info(f"Processing master database: {raw_file}")
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} rows.")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    df_clean = clean_data(df)
    logger.info(f"Cleaned data: {len(df_clean)} rows.")

    df_processed = process_features(df_clean)

    # Memory Optimization: Convert object columns to category
    category_columns = config.CATEGORY_COLUMNS
    for col in category_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype("category")

    output_path = config.PROCESSED_DATA_PATH
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()