"""
Module for processing raw anime data.
This script loads the latest raw data, cleans it, applies text preprocessing,
and saves the result to data/processed.
"""

import glob
import os
import sys

import pandas as pd
from tqdm import tqdm

# Add project root to sys.path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_text


def get_latest_raw_file(raw_dir: str) -> str:
    """
    Get the path to the latest CSV file in the raw directory.

    :param raw_dir: Directory containing raw CSV files.
    :return: Path to the latest file.
    :raises FileNotFoundError: If no CSV files are found.
    """
    list_of_files = glob.glob(os.path.join(raw_dir, "AnimeData_*.csv"))
    if not list_of_files:
        raise FileNotFoundError(f"No AnimeData_*.csv files found in {raw_dir}")
    return max(list_of_files, key=os.path.getctime)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.

    :param df: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    # Validate required columns
    # Validate required columns
    required_columns = ["title", "synopsis"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Drop duplicates
    df = df.drop_duplicates(subset=["title"])

    # Drop rows with missing synopsis as it's critical for recommendation
    df = df.dropna(subset=["synopsis"])

    # Handle episodes if it exists
    # Handle episodes if it exists
    if "episodes" in df.columns:
        df["episodes"] = df["episodes"].fillna("N/A")

    # Normalize Score/Rating
    if "score" not in df.columns and "rating" not in df.columns:
        # Create default if neither exists
        df["score"] = 0
    elif "score" not in df.columns and "rating" in df.columns:
         # Use Rating as Score source
         df["score"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    elif "score" in df.columns:
        # Ensure Score is numeric
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to create features for the model.

    :param df: Cleaned DataFrame.
    :return: DataFrame with additional feature columns.
    """
    tqdm.pandas(desc="Preprocessing Synopsis")
    tqdm.pandas(desc="Preprocessing Synopsis")
    df["stemmed_synopsis"] = df["synopsis"].progress_apply(preprocess_text)
    return df


def main():
    """
    Main execution flow for data processing.
    """
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    try:
        raw_file = get_latest_raw_file(raw_dir)
        print(f"Processing latest raw file: {raw_file}")
    except FileNotFoundError as e:
        print(e)
        # Fallback for dev/testing if raw is empty but we have existing data
        # check if there is a 'final' data to use as base?
        # For now, just exit if no raw data.
        return

    df = pd.read_csv(raw_file)
    print(f"Loaded {len(df)} rows.")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    df_clean = clean_data(df)
    print(f"Cleaned data: {len(df_clean)} rows.")

    df_processed = process_features(df_clean)

    # Memory Optimization: Convert object columns to category
    category_columns = ["genres", "studio", "demographic", "source", "status"]
    for col in category_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype("category")

    output_path = os.path.join(processed_dir, "anime_data_processed.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()