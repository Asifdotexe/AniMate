"""
This script contains functions to scrape anime data from MyAnimeList.net.
It is intended to be run as a standalone script to generate the raw dataset.
"""

import io
import argparse
import cProfile
import pstats
import re
import time
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.animate.config import MYANIMELIST_BASE_URL

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
REQUEST_TIMEOUT = 10

logger = logging.getLogger(__name__)


def get_current_date() -> str:
    """Gets the current date formatted as 'DDMMYYYY'."""
    return datetime.now().strftime("%d%m%Y")


def safe_text(element, default: str = "N/A") -> str:
    """Safely extracts text from a BeautifulSoup element."""
    return element.text.strip() if element else default


def safe_int(element, default: str = "N/A") -> int | str:
    """Safely extracts an integer from a BeautifulSoup element."""
    try:
        return int(element.text.strip().replace(",", "")) if element else default
    except (ValueError, AttributeError):
        return default


def safe_float(element, default: str = "N/A") -> float | str:
    """Safely extracts a float from a BeautifulSoup element."""
    try:
        return float(element.text.strip()) if element else default
    except (ValueError, AttributeError):
        return default


def _get_basic_info(soup: BeautifulSoup) -> dict:
    """Extracts title, year, and synopsis."""
    start_date_text = safe_text(soup.find("span", class_="js-start_date"))
    title_h2 = soup.find("h2", class_="h2_anime_title")
    title_a = title_h2.find("a") if title_h2 else None
    synopsis_div = soup.find("div", class_="synopsis js-synopsis")
    synopsis_p = synopsis_div.p if synopsis_div else None
    return {
        "Title": safe_text(title_a),
        "Release Year": start_date_text[:4] if start_date_text != "N/A" else "N/A",
        "Synopsis": safe_text(synopsis_p),
    }


def _get_episode_info(info_div: BeautifulSoup) -> dict:
    """Extracts episode count and airing status."""
    if not info_div:
        return {"Episodes": "N/A", "Status": "N/A"}

    eps_text = info_div.get_text()
    match = re.search(r"(\d+)\s*eps", eps_text)
    # class_=["item finished", "item airing"] looks for a single class string;
    # MAL uses multiple classes. Match by tokens or CSS selectors.
    status_el = (
        info_div.select_one("span.item.finished")
        or info_div.select_one("span.item.airing")
    )
    return {
        "Episodes": match.group(1) if match else "N/A",
        "Status": safe_text(status_el),
    }


def _get_production_info(soup: BeautifulSoup) -> dict:
    """Extracts genres, studio, source, and demographic."""
    properties_div = soup.find("div", class_="properties")
    genres_div = soup.find("div", class_="genres-inner")
    return {
        "Genres": ", ".join(g.text for g in genres_div.find_all("a")) if genres_div else "N/A",
        "Studio": _extract_property(properties_div, "Studio"),
        "Source": _extract_property(properties_div, "Source"),
        "Demographic": _extract_property(properties_div, "Demographic"),
    }


def _extract_property(properties_div: BeautifulSoup, caption: str) -> str:
    """Helper to extract a specific property (like Studio, Source) from its div."""
    if not properties_div:
        return "N/A"
    for div in properties_div.find_all("div", class_="property"):
        caption_span = div.find("span", class_="caption")
        if caption_span and caption_span.text.strip() == caption:
            item_spans = div.find_all("span", class_="item")
            return ", ".join(item.get_text(strip=True) for item in item_spans) or "N/A"
    return "N/A"


def scrape_anime_data(anime_item_html: str) -> dict:
    """
    Extracts structured data from the HTML of a single anime item by delegating
    to specialized helper functions.
    """
    soup = BeautifulSoup(anime_item_html, "html.parser")

    # Delegate extraction to helpers
    basic_info = _get_basic_info(soup)
    episode_info = _get_episode_info(soup.find("div", class_="info"))
    production_info = _get_production_info(soup)

    # Combine results from helpers
    anime_data = {
        **basic_info,
        **episode_info,
        **production_info,
        "Voters": safe_int(soup.find("div", class_="member")),
        "Rating": safe_float(soup.find("div", class_="score")),
    }
    return anime_data

def fetch_and_scrape(url: str, page_limit: int, retries: int = 3, delay: int = 5) -> list[dict]:
    """Fetches and scrapes anime data from a given URL with multiple pages."""
    all_data = []
    consecutive_404s = 0

    for page in tqdm(range(1, page_limit + 1), desc="Pages", leave=False):
        page_url = f"{url}?page={page}"
        for _ in range(retries):
            try:
                response = requests.get(page_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                anime_list = soup.find_all("div", class_="js-anime-category-producer")

                for anime_item in anime_list:
                    all_data.append(scrape_anime_data(str(anime_item)))

                consecutive_404s = 0  # Reset on success
                time.sleep(1)
                break
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    consecutive_404s += 1
                    if consecutive_404s >= 3:
                        print(f"Stopping {url} after 3 consecutive 404s.")
                        return all_data
                    print(f"Page {page_url} not found (404). Skipping.")
                    break  # Don't retry 404s
                print(f"HTTP Error on {page_url}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except requests.RequestException as e:
                print(f"Request Error on {page_url}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        else:
            print(f"Failed to fetch {page_url} after {retries} retries. Skipping page.")
            continue

    return all_data

def save_data(data: list[dict], date_str: str) -> None:
    """Saves the scraped data to a CSV file."""
    if not data:
        print("No data was scraped. Nothing to save.")
        return

    df = pd.DataFrame(data).drop_duplicates()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_path = OUTPUT_DIR / f"AnimeData_{date_str}.csv"
    df.to_csv(file_path, index=False)
    print(f"Data successfully saved to {file_path}")

def main():
    """Main function to run the web scraper for all genres."""
    parser = argparse.ArgumentParser(description='Run the web scraper to extract data from MyAnimeList website')
    parser.add_argument(
        "--profile",
        action='store_true',
        help="Enable profiling on the script to clock the efficiency"
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        type=int,
        default=[1],  # Default to genre 1 if --profile is used without specific genres
        help="A list of genre IDs to scrape when profiling. Defaults to [1].",
    )
    args = parser.parse_args()


    def _run_scraper(genre_list: list[int]) -> None:
        """
        helper function to perform the scraping based on the given genre list.

        :param genre_list: List containing the specific genres to scrape.
        """
        logger.info('Scraping MyAnimeList')
        url_list = [f"{MYANIMELIST_BASE_URL}{genre_id}/" for genre_id in genre_list]

        current_date = get_current_date()
        all_anime_data = []

        for url in tqdm(url_list, desc="Scraping Genres"):
            scraped_data = fetch_and_scrape(url, page_limit=100)
            all_anime_data.extend(scraped_data)

        save_data(all_anime_data, current_date)

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

        _run_scraper(args.genres)

        profiler.disable()
        print("\n--- SCRAPING COMPLETE ---")
        print("\n--- PERFORMANCE PROFILE ---")
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        print(s.getvalue())

    else:
        all_genres = list(range(1, 44))
        _run_scraper(all_genres)  # Use all genres for normal run
        print("\n--- SCRAPING COMPLETE ---")

if __name__ == "__main__":
    main()

