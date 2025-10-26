"""
Clean and preprocesses the raw scraped data for use in the AniMate app
"""

import argparse
import cProfile
import io
import pstats
import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from animate.config import (AVG_EPISODE_DURATION_MINS, FINAL_APP_COLUMNS,
                            MATURE_GENRES, PROCESSED_DATA_DIR, RAW_DATA_DIR,
                            REQ_RAW_COLUMNS)
from animate.util import fetch_latest_final_csv_path


# Pre-compiled Regex for Mature Genres
_MATURE_PATTERN_RE: Optional[re.Pattern] = None
# Pre-compiled Regex for Synopsis Placeholders
_SYNOPSIS_PLACEHOLDER_PATTERN: Optional[re.Pattern] = None

if MATURE_GENRES:
    pattern_str = r'\b(?:' + '|'.join(re.escape(g) for g in MATURE_GENRES) + r')\b'
    try:
        _MATURE_PATTERN_RE = re.compile(pattern_str, re.IGNORECASE) # Compile case-insensitively
    except re.error as e:
        print(f"Error compiling mature genre regex: {e}. Filtering will be skipped.")

# Compile synopsis patterns as well
_synopsis_patterns_list = [
    r'^\(No synopsis yet\.\)$', # Match exact string
    r'^No synopsis has been added.*', # Match start of string
    r'^N/A$' # Match exact 'N/A'
]
try:
    # Combine patterns into one regex for single replacement call
    # Ensure each original pattern is treated as a group for the OR |
    _SYNOPSIS_PLACEHOLDER_PATTERN = re.compile('|'.join(f"(?:{p})" for p in _synopsis_patterns_list))
except re.error as e:
     print(f"Error compiling synopsis placeholder regex: {e}. Cleaning might be incomplete.")


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
    :param target_type: Target dtype hint (e.g., int or float)
    :return: Cleaned pandas series
    """
    # Replace 'N/A' string with actual NaN before conversion
    # Ensure it's string type first to use .replace reliably
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        series = series.replace("N/A", np.nan)

    # Convert, turning errors into NaN. Result is usually float64 if NaNs occur.
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Check if the target type is specifically integer.
    if target_type is int:
        # Get the non-NaN values directly
        finite_values = numeric_series.dropna()

        if finite_values.empty:
            try:
                return numeric_series.astype(pd.Int64Dtype())
            except Exception:
                return numeric_series

        # Check if all existing (finite) values are integral (whole numbers)
        # Using modulo is generally safer than direct comparison for floats
        is_integral = (finite_values % 1 == 0).all()

        # If all non-NaN values are integral, cast to nullable Int64.
        if is_integral:
            try:
                # Use nullable integer type Int64, works even if NaNs are present.
                return numeric_series.astype(pd.Int64Dtype())
            except Exception as e:
                print(
                    f"Warning: Could not cast to Int64 despite integral check: {e}. Keeping as float."
                )
                pass

    if pd.api.types.is_float_dtype(target_type) or target_type is int:
        try:
            # Attempt to cast to standard float
            return numeric_series.astype(float)
        except Exception as e:
            # If float cast fails, return the result of pd.to_numeric
            print(
                f"Warning: Could not cast to float: {e}. Returning original numeric series."
            )
            return numeric_series  # Explicit return

    # If target_type was neither int nor float initially, return the numeric series as is.
    return numeric_series  # Explicit final return


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
    episodes_numeric = clean_numeric_columns(no_of_episodes, target_type=float).fillna(
        0
    )

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
    if _SYNOPSIS_PLACEHOLDER_PATTERN is None:
        print("Warning: Synopsis placeholder regex not compiled. Skipping synopsis cleaning.")
        # Return series as string, fillna just in case
        return series.astype(str).fillna("").str.strip()

    # Use StringDtype to preserve pd.NA instead of converting to "nan"
    cleaned_series = series.astype(str).str.replace(
        _SYNOPSIS_PLACEHOLDER_PATTERN,
        "",  # Replace with empty string
        regex=True
    ).str.strip()

    return cleaned_series.fillna("")


def filter_mature(df: pd.DataFrame, genre_column: str = "genres") -> pd.DataFrame:
    """
    Removes rows containing mature genres.

    :param df: Pandas dataframe containing genre column
    :param genre_column: Optional field if there is any additional genre columns, defaults to 'genres'
    :return: Pandas dataframe post mature content filtering
    """
    if not MATURE_GENRES or _MATURE_PATTERN_RE is None:
        return df

    if genre_column not in df.columns:
        print(
            f"Warning: Genre column '{genre_column}' not found. Skipping mature content filtering."
        )
        return df

    # Ensure genres are strings and handle NaN
    genres_str = df[genre_column].fillna("").astype(str)

    # If no mature genres defined, skip filtering
    if not MATURE_GENRES:
         return df

    mask = ~genres_str.str.contains(_MATURE_PATTERN_RE, regex=True, na=False)

    removed_count = len(df) - mask.sum()
    if removed_count > 0:
        print(f"Removed {removed_count} entries with mature genres.")
    return df[mask]

def load_and_validate_data(input_path: Path) -> pd.DataFrame:
    """Loads raw data, validates path, cleans columns, selects initial columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Error: Input file not found at {input_path}")
    print(f"Loading raw data from {input_path}...")
    try:
        df = pd.read_csv(input_path, low_memory=False, na_values=["N/A"])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    df = clean_column_names(df)
    available_cols = [col for col in REQ_RAW_COLUMNS if col in df.columns]
    missing_cols = set(REQ_RAW_COLUMNS) - set(available_cols)
    if missing_cols:
        print(f"Warning: Raw data missing expected columns: {missing_cols}.")
    if "title" not in available_cols or "synopsis" not in available_cols:
        raise ValueError("Error: Essential 'title' or 'synopsis' column missing.")
    return df[available_cols].copy()


def apply_initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans synopsis and drops rows with missing essential info."""
    df["synopsis"] = clean_synopsis(df["synopsis"])
    initial_rows = len(df)
    df.dropna(subset=["title"], inplace=True)
    df = df[df["title"].astype(str).str.strip().str.len() > 0]
    df.dropna(subset=["synopsis"], inplace=True)
    df = df[df["synopsis"].str.len() > 0]
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing/empty title/synopsis.")
    if df.empty:
        raise ValueError("Error: No valid data after dropping missing essentials.")
    return df


def process_numeric_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans numeric columns and calculates duration features."""
    if "release_year" in df.columns:
        df["release_year"] = clean_numeric_columns(df["release_year"], target_type=int)
    if "episodes" in df.columns:
        df["episodes"] = clean_numeric_columns(df["episodes"], target_type=float)
    if "rating" in df.columns:
        df["score"] = clean_numeric_columns(df["rating"], target_type=float)
        df.drop(
            columns=["rating"], inplace=True, errors="ignore"
        )  # Ignore error if 'rating' somehow gone
    else:
        df["score"] = np.nan

    if "episodes" in df.columns:
        df["total_duration_hours"], df["duration_category"] = (
            calculate_duration_features(df["episodes"], AVG_EPISODE_DURATION_MINS)
        )
    else:
        df["total_duration_hours"] = np.nan
        df["duration_category"] = "Unknown"
    df["duration_category"] = df["duration_category"].astype("category")
    return df


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filters mature content, selects final columns, converts types, deduplicates."""
    df = filter_mature(df, genre_column="genres")
    if df.empty:
        raise ValueError("Error: No data remaining after filtering mature content.")

    final_columns_present = [col for col in FINAL_APP_COLUMNS if col in df.columns]
    df_final = df[final_columns_present].copy()

    for col in ["genres", "studio", "demographic", "source"]:
        if col in df_final.columns and df_final[col].dtype == "object":
            df_final[col] = df_final[col].astype("category")

    initial_rows = len(df_final)
    if "title" in df_final.columns:
        df_final.drop_duplicates(subset=["title"], keep="last", inplace=True)
        rows_dropped = initial_rows - len(df_final)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} duplicate entries based on title.")
    else:
        print("Warning: 'title' column not found, cannot deduplicate.")
    return df_final


def run_cleaning_pipeline() -> None:
    """
    Orchestrates the data cleaning process
    """

    # Defining the input and output file dynamically by picking the file with the latest suffix data in it's filename
    # example: anime_dump_*.csv, where * will be a date string.
    input_path = fetch_latest_final_csv_path(RAW_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / input_path.name

    try:
        df = load_and_validate_data(input_path)
        df = apply_initial_cleaning(df)
        df = process_numeric_and_features(df)
        df_final = finalize_dataframe(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Successfully cleaned data saved to {output_path}")
        print(f"Final dataset shape: {df_final.shape}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw scraped anime data.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for the cleaning process.",
    )
    args = parser.parse_args()

    if args.profile:
        print("--- Running Cleaning Script with Profiling ---")
        profiler = cProfile.Profile()
        profiler.enable()
        success = False
        try:
            run_cleaning_pipeline()
            success = True  # Mark as successful if no exceptions
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")
            traceback.print_exc()
        finally:
            profiler.disable()
            print("\n--- PERFORMANCE PROFILE ---")
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(30)  # Show top 30 lines
            print(s.getvalue())
    else:
        try:
            run_cleaning_pipeline()
            print("\n--- Cleaning Complete ---")
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
