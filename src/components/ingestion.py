"""
Module for collecting anime data using the Jikan API (v4).
This script builds a local dataset of anime, supporting incremental updates.
It fetches data from the /top/anime endpoint and merges it with an existing master dataset.
"""

import time
from typing import Dict, List, Any
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import config
from src.logger import setup_logging

logger = setup_logging("ingestion")


class JikanAPI:
    """
    Wrapper for Jikan API v4 interactions.
    """
    
    def __init__(self):
        """
        Initialize the API wrapper.
        """
        self.delay = config.api.delay_seconds
        self.base_url = config.api.jikan_base_url

    def get_top_anime(self, page: int = 1, filter_type: str = "bypopularity") -> Dict[str, Any]:
        """
        Fetches a specific page of top anime.
        
        :param page: Page number to fetch.
        :param filter_type: Jikan filter parameter (e.g., 'bypopularity', 'favorite').
        :return: JSON response dictionary.
        """
        url = f"{self.base_url}/top/anime"
        params = {
            "page": page,
            "filter": filter_type,
            "limit": config.api.batch_size
        }
        
        retries = 0
        while retries <= config.api.max_retries:
            try:
                response = requests.get(url, params=params, timeout=10) # Timeout hardcoded or add to config
                
                # Rate limiting handling
                if response.status_code == 429:
                    retries += 1
                    if retries > config.api.max_retries:
                        logger.error(f"Max retries ({config.api.max_retries}) exceeded for page {page}.")
                        return {}
                        
                    logger.warning(f"Rate limited (429). Retrying {retries}/{config.api.max_retries} in {self.delay}s...")
                    time.sleep(self.delay)
                    continue
                
                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching page {page}: {e}")
                return {}
        return {}

def extract_names(data_list: List[Dict[str, Any]]) -> str:
    """Helper to extract comma-separated names from a list of dicts."""
    if not data_list:
        return "N/A"
    return ", ".join([item.get("name", "") for item in data_list])

def parse_anime_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single anime item from Jikan API response into a flat dictionary.
    """
    # Safe extraction helpers
    aired_from = item.get("aired", {}).get("from", "")
    release_year = aired_from[:4] if aired_from else "N/A"
    
    genres_list = item.get("genres", [])
    explicit_genres = item.get("explicit_genres", [])
    themes = item.get("themes", [])
    demographics = item.get("demographics", [])
    
    # Legacy 'Genres' column often included demographics and themes in MAL scraping
    full_genre_string = extract_names(genres_list + explicit_genres)
    
    return {
        "mal_id": item.get("mal_id"), # Primary Key
        "Title": item.get("title", "N/A"),
        "English Title": item.get("title_english", "N/A"),
        "Japanese Title": item.get("title_japanese", "N/A"),
        "Episodes": item.get("episodes") if item.get("episodes") is not None else "N/A",
        "Release Year": release_year,
        "Status": item.get("status", "N/A"),
        "Air Date": item.get("aired", {}).get("string", "N/A"),
        "Genres": full_genre_string,
        "Themes": extract_names(themes),
        "Demographics": extract_names(demographics),
        "Studio": extract_names(item.get("studios", [])),
        "Producers": extract_names(item.get("producers", [])),
        "Synopsis": item.get("synopsis", "N/A"),
        "Rating": item.get("score", 0),  # MAL Score 1-10
        "Voters": item.get("scored_by", 0),
        "Popularity": item.get("popularity", 0),
        "Rank": item.get("rank", 0),
        "Members": item.get("members", 0),
        "Favorites": item.get("favorites", 0),
        "Content Rating": item.get("rating", "N/A"), # PG-13, R-17+, etc.
        "Source": item.get("source", "N/A"),
        "Duration": item.get("duration", "N/A"),
        "URL": item.get("url", "N/A"),
        "Image URL": item.get("images", {}).get("jpg", {}).get("large_image_url", "N/A")
    }

def build_anime_dataset(start_page: int = 1, limit_pages: int = 5) -> pd.DataFrame:
    """
    Crawls Jikan to build a local dataset of popular anime.
    
    :param start_page: Page to start fetching from.
    :param limit_pages: Number of pages to fetch.
    :return: A Pandas DataFrame.
    """
    api = JikanAPI()
    all_anime = []
    
    # Default filter
    filter_type = "bypopularity"

    logger.info(f"Starting ingestion of Top Anime ({filter_type})...")
    logger.info(f"Target: Pages {start_page} to {start_page + limit_pages - 1}")

    for page in tqdm(range(start_page, start_page + limit_pages), desc="Fetching Pages"):
        data = api.get_top_anime(page, filter_type=filter_type)
        
        items = data.get("data", [])
        if not items:
            logger.info(f"No data found on page {page}. Stopping early.")
            break
            
        for item in items:
            parsed = parse_anime_item(item)
            all_anime.append(parsed)
        
        # Pagination check
        pagination = data.get("pagination", {})
        if not pagination.get("has_next_page", False):
            logger.info("Reached last page of results.")
            break

    df = pd.DataFrame(all_anime)
    logger.info(f"Fetched {len(df)} anime records from API.")
    return df

def merge_datasets(master_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges new fetched data into the master dataset (Upsert logic).
    - Updates existing records (based on mal_id) with new data (ratings, popularity, etc.)
    - Appends completely new records.
    """
    if master_df.empty:
        return new_df
    
    if new_df.empty:
        return master_df

    # Ensure mal_id is integer for clean merging
    # Handle NaNs safely by coercing to numeric and using nullable Int64
    master_df['mal_id'] = pd.to_numeric(master_df['mal_id'], errors='coerce').astype('Int64')
    new_df['mal_id'] = pd.to_numeric(new_df['mal_id'], errors='coerce').astype('Int64')

    # Separation
    new_ids = new_df['mal_id'].unique()
    
    # Filter out rows from Master that are present in New (we will replace them)
    master_preserved = master_df[~master_df['mal_id'].isin(new_ids)]
    
    # Concatenate preserved Master rows with New rows
    merged_df = pd.concat([master_preserved, new_df], ignore_index=True)
    
    # Sort for tidiness (optional, by popularity or rank if available, otherwise index)
    if 'Popularity' in merged_df.columns:
        merged_df = merged_df.sort_values('Popularity', ascending=True)

    logger.info(f"Merged: {len(master_preserved)} retained + {len(new_df)} updated/new = {len(merged_df)} total records.")
    return merged_df

def load_master_dataset(filepath: Path) -> pd.DataFrame:
    """Loads the existing master dataset if it exists."""
    if filepath.exists():
        logger.info(f"Loading existing master dataset from {filepath}...")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error loading master dataset: {e}. Starting fresh.")
            return pd.DataFrame()
    return pd.DataFrame()

def save_dataset(df: pd.DataFrame, filepath: Path):
    """Saves the dataframe to CSV."""
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to save.")
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Master dataset updated at: {filepath}")

def main():
    """
    Main function to run the pipeline.
    """
    # Load existing data
    master_path = Path(config.paths.raw_data)
    master_df = load_master_dataset(master_path)
    
    # Fetch new data
    new_data_df = build_anime_dataset(
        start_page=1, 
        limit_pages=5
    )
    
    # Merge (Upsert)
    final_df = merge_datasets(master_df, new_data_df)
    
    # Save
    save_dataset(final_df, master_path)
    
    logger.info(f"Pipeline complete. Master database updated with {len(new_data_df)} fetched records.")

if __name__ == "__main__":
    main()
