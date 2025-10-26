"""
Utility functions
"""

from pathlib import Path


def fetch_latest_final_parquet_path(data_directory: Path) -> Path:
    """
    Fetches the latest file from the provided data directory based on the suffix timestamp in the parquet filenames

    :param data_directory: Path to the directory containing file in name_timestamp.parquet format
    :return: Path to the latest parquet file based on the timestamp
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Directory not found: {data_directory}")
    candidates = sorted(
        data_directory.glob("anime_dump_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No anime_dump_*.parquet found in {data_directory}")
    return candidates[0]
