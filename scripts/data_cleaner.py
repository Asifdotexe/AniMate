"""
Clean and preprocesses the raw scraped data for use in the AniMate app
"""

import pandas as pd

from animate.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from animate.util import fetch_latest_final_csv_path


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column name to lowercase and underscore.

    :param df: Pandas dataframe containing the raw data
    :return: Pandas dataframe with updated column name
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False)
    return df


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

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
