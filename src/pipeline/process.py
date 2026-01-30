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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.

    :param df: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    # Validate required columns
    required_columns = ["title", "synopsis"]
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
    raw_file = os.path.join(project_root, "data", "raw", "anime_master_db.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_file):
        print(f"Error: Master DB not found at {raw_file}")
        return

    print(f"Processing master database: {raw_file}")
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