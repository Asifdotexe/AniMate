"""
Clean and preprocesses the raw scraped data for use in the AniMate app
"""

import numpy as np
import pandas as pd

from animate.config import (AVG_EPISODE_DURATION_MINS, PROCESSED_DATA_DIR,
                            RAW_DATA_DIR, REQ_RAW_COLUMNS)
from animate.util import fetch_latest_final_csv_path


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column name to lowercase and underscore.

    :param df: Pandas dataframe containing the raw data
    :return: Pandas dataframe with updated column name
    """
    print("Cleaning column...")
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    return df


def clean_numeric_columns(series: pd.Series, target_type=float) -> pd.Series:
    """
    Converts a Series to a numeric type, coercing errors to NaN.

    :param series: Pandas series that needs to be cleaned
    :param target: The datatype of the column we want to convert the existing column to
    :return: Cleaned pandas series
    """
    print("Cleaning numeric columns...")
    # FIXME: Understand the scraper's old cold to see if this can be done during the scraping process
    # Converting the 'N/A' to np.nan
    series = series.replace("N/A", np.nan)
    # Converting the errors to np.nan
    numeric_series = pd.to_numeric(series, errors="coerce")
    # Convert to the desired integer type if possible (e.g., float -> Int64)
    if pd.api.types.is_integer_dtype(target_type) and not numeric_series.isnull().any():
        # Use nullable integer type Int64 to handle potential NaNs introduced by coerce
        # If no NaNs remain after coerce, can convert safely.
        # However, if NaNs *might* exist, better to keep as float or use pd.Int64Dtype()
        try:
            # Use nullable integer type Int64 to handle potential NaNs
            return numeric_series.astype(pd.Int64Dtype())
        except TypeError:
            # Keep as float if conversion fails (e.g., due to remaining NaNs)
            pass

    elif pd.api.types.is_float_dtype(target_type):
        # Ensure it's standard float
        return numeric_series.astype(float)
    return numeric_series


def calculate_duration_features(
    no_of_episodes: pd.Series, avg_duration: int
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates total duration in hours and assigns a duration category

    :param no_of_episodes: Number of episodes for a given anime
    :param avg_duration: The average anime duration (based on assumption)
    :return: _description_
    """
    # Ensure episodes is numeric, fill NaN with 0 for calculation
    episodes_numeric = clean_numeric_columns(no_of_episodes, target_type=float).fillna(0)

    total_duration_minutes = episodes_numeric * avg_duration
    total_duration_hours = (total_duration_minutes / 60).round(2)

    # Define bins and labels for duration categories
    bins = [-np.inf, 0, 2, 6, 12, 30, 100, np.inf]
    labels = [
        "Unknown",  # Corresponds to 0 episodes or NaN
        "Very Short (<2h)",  # Movie/Special
        "Short (2-6h)",
        "Medium (6-12h)",
        "Long (12-30h)",
        "Very Long (30-100h)",
        "Epic (>100h)",
    ]

    # Use pd.cut to categorize
    duration_category = pd.cut(
        total_duration_hours, bins=bins, labels=labels, right=True
    )

    if "Unknown" not in duration_category.cat.categories:
        # This case should ideally not happen if labels are defined correctly
        duration_category = duration_category.cat.add_categories("Unknown")
    duration_category[episodes_numeric <= 0] = "Unknown"

    # Convert hours back to NaN where episodes were originally NaN or 0
    total_duration_hours = total_duration_hours.where(episodes_numeric > 0, np.nan)

    # Return duration_category as pandas Categorical dtype
    return total_duration_hours, duration_category.astype("category")


def clean_synopsis(series: pd.Series) -> pd.Series:
    """
    Cleans the synopsis text, removing placeholders.

    :param series: Pandas series that needs to be cleaned
    :return: Cleaned pandas synopsis series
    """
    placeholder_patterns = [
        r'^\(No synopsis yet\.\)$', # Match exact string
        r'^No synopsis has been added.*', # Match start of string
        r'^N/A$' # Match exact 'N/A'
    ]
    # Apply replacements iteratively or combine regex carefully
    cleaned_series = series.astype(str).replace(placeholder_patterns, np.nan, regex=True)
    return cleaned_series.fillna('').str.strip()


def main() -> None:
    """
    Orcastrates the data cleaning process
    """

    # Defining the input and output file dynamically by picking the file with the latest suffix data in it's filename
    # example: anime_dump_*.csv, where * will be a date string.
    input_path = fetch_latest_final_csv_path(RAW_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / input_path.name

    if not input_path.exists():
        raise FileExistsError(f"Error: Input file not found at {input_path}")

    # Setting low_memory as there are come columns with mix-type data
    df = pd.read_csv(input_path, low_memory=False)
    df = clean_column_names(df)
    available_cols = [col for col in REQ_RAW_COLUMNS if col in df.columns]

    missing_cols = set(REQ_RAW_COLUMNS) - set(available_cols)
    if missing_cols:
        print(
            f"Warning: Raw data missing expected columns: {missing_cols}. They will not be included."
        )

    # Ensure essential columns exist before proceeding
    if "title" not in available_cols or "synopsis" not in available_cols:
        raise ValueError(
            "Error: Essential 'title' or 'synopsis' column missing from input data."
        )

    df = df[available_cols].copy()

    df['synopsis'] = clean_synopsis(df['synopsis'])

    # Drop rows with missing essential info (after cleaning synopsis)
    initial_rows = len(df)
    # Check for NaN/None in title as well
    df.dropna(subset=['title'], inplace=True)
    # Ensure title is not empty string
    df = df[df['title'].astype(str).str.strip().str.len() > 0]
    df.dropna(subset=['synopsis'], inplace=True)
    # Ensure synopsis not empty after cleaning
    df = df[df['synopsis'].str.len() > 0]

    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing/empty title or synopsis.")
    if df.empty:
        # Use ValueError for condition that prevents script's purpose
        raise ValueError("Error: No valid data remaining after dropping missing titles/synopses.")


    if "release_year" in df.columns:
        df["release_year"] = clean_numeric_columns(df["release_year"], target_type=int)
    if "episodes" in df.columns:
        df["episodes"] = clean_numeric_columns(df["episodes"], target_type=float)
    if "rating" in df.columns:
        df["score"] = clean_numeric_columns(df["rating"], target_type=float)
    else:
        df["score"] = np.nan
    if "rating" in available_cols and "score" in df.columns:
        df.drop(columns=["rating"], inplace=True)

    if "episodes" in df.columns:
        df["total_duration_hours"], df["duration_category"] = (
            calculate_duration_features(df["episodes"], AVG_EPISODE_DURATION_MINS)
        )
    else:
        df["total_duration_hours"] = np.nan
        df["duration_category"] = "Unknown"
    # Ensure duration_category is category type
    df["duration_category"] = df["duration_category"].astype("category")

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
