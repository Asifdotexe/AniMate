"""
Module for collecting anime data using the Jikan API (v4).
This script builds a local dataset of anime, supporting incremental updates.
It fetches data from the /top/anime endpoint and merges it with an existing master dataset.
"""

import os
import time
from typing import Dict, List, Any

import pandas as pd
import requests
from tqdm import tqdm

import sys
# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.pipeline import api_config as config
except ImportError:
    import api_config as config


class JikanAPI:
    """
    Wrapper for Jikan API v4 interactions.
    """
    
    def __init__(self, rate_limit_delay: float = config.RATE_LIMIT_DELAY):
        """
        Initialize the API wrapper.
        
        :param rate_limit_delay: Time in seconds to sleep between requests.
        """
        self.delay = rate_limit_delay
        self.base_url = config.BASE_URL

    def get_top_anime(self, page: int = 1, filter_type: str = config.DEFAULT_FILTER_TYPE) -> Dict[str, Any]:
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
            "limit": config.ITEMS_PER_PAGE
        }
        
        retries = 0
        while retries <= config.MAX_RETRIES:
            try:
                response = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                
                # Rate limiting handling
                if response.status_code == 429:
                    retries += 1
                    if retries > config.MAX_RETRIES:
                        print(f"Max retries ({config.MAX_RETRIES}) exceeded for page {page}.")
                        return {}
                        
                    print(f"Rate limited (429). Retrying {retries}/{config.MAX_RETRIES} in {config.ERROR_SLEEP_TIME}s...")
                    time.sleep(config.ERROR_SLEEP_TIME)
                    continue
                
                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
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

def build_anime_dataset(start_page: int = config.DEFAULT_START_PAGE, limit_pages: int = config.DEFAULT_PAGE_LIMIT) -> pd.DataFrame:
    """
    Crawls Jikan to build a local dataset of popular anime.
    
    :param start_page: Page to start fetching from.
    :param limit_pages: Number of pages to fetch.
    :return: A Pandas DataFrame.
    """
    api = JikanAPI()
    all_anime = []

    print(f"Starting ingestion of Top Anime ({config.DEFAULT_FILTER_TYPE})...")
    print(f"Target: Pages {start_page} to {start_page + limit_pages - 1}")

    for page in tqdm(range(start_page, start_page + limit_pages), desc="Fetching Pages"):
        data = api.get_top_anime(page, filter_type=config.DEFAULT_FILTER_TYPE)
        
        items = data.get("data", [])
        if not items:
            print(f"No data found on page {page}. Stopping early.")
            break
            
        for item in items:
            parsed = parse_anime_item(item)
            all_anime.append(parsed)
        
        # Pagination check
        pagination = data.get("pagination", {})
        if not pagination.get("has_next_page", False):
            print("Reached last page of results.")
            break

    df = pd.DataFrame(all_anime)
    print(f"Fetched {len(df)} anime records from API.")
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

    print(f"Merged: {len(master_preserved)} retained + {len(new_df)} updated/new = {len(merged_df)} total records.")
    return merged_df

def load_master_dataset(filepath: str) -> pd.DataFrame:
    """Loads the existing master dataset if it exists."""
    if os.path.exists(filepath):
        print(f"Loading existing master dataset from {filepath}...")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading master dataset: {e}. Starting fresh.")
            return pd.DataFrame()
    return pd.DataFrame()

def save_dataset(df: pd.DataFrame, filepath: str):
    """Saves the dataframe to CSV."""
    if df.empty:
        print("DataFrame is empty. Nothing to save.")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Master dataset updated at: {filepath}")

def main():
    """
    Main function to run the pipeline.
    """
    # Load existing data
    master_df = load_master_dataset(config.MASTER_DB_PATH)
    
    # Fetch new data
    new_data_df = build_anime_dataset(
        start_page=config.DEFAULT_START_PAGE, 
        limit_pages=config.DEFAULT_PAGE_LIMIT
    )
    
    # Merge (Upsert)
    final_df = merge_datasets(master_df, new_data_df)
    
    # Save
    save_dataset(final_df, config.MASTER_DB_PATH)
    
    print(f"Pipeline complete. Master database updated with {len(new_data_df)} fetched records.")

if __name__ == "__main__":
    main()
