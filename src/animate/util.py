"""
Utility functions
"""

from pathlib import Path


def latest_final_csv(data_directory: Path) -> Path:
    """
    Fetches the latest file from the provided data directory based on the suffix timestamp in the CSV filenames

    :param data_directory: Path to the directory containing file in name_timestamp.csv format
    :return: Path to the latest CSV file based on the timestamp
    """
    file_path = sorted(
        data_directory.glob("anime_dump_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return file_path[0]
