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

from src.preprocessing import preprocess_text


# Add project root to sys.path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


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
    # Drop duplicates
    df = df.drop_duplicates(subset=["Title"])

    # Drop rows with missing synopsis as it's critical for recommendation
    df = df.dropna(subset=["Synopsis"])

    # Fill other NaNs with defaults
    df["Episodes"] = df["Episodes"].fillna("N/A")
    df["Score"] = df["Rating"].fillna(
        0
    )  # Rename Rating to Score if needed or keep consistent

    # Ensure Score is numeric
    df["Score"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to create features for the model.

    :param df: Cleaned DataFrame.
    :return: DataFrame with additional feature columns.
    """
    tqdm.pandas(desc="Preprocessing Synopsis")
    df["stemmed_synopsis"] = df["Synopsis"].progress_apply(preprocess_text)
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
    df.columns = df.columns.str.strip()

    df_clean = clean_data(df)
    print(f"Cleaned data: {len(df_clean)} rows.")

    df_processed = process_features(df_clean)

    output_path = os.path.join(processed_dir, "anime_data_processed.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
