"""
This script contains functions to scrape anime data from MyAnimeList.net.
It is intended to be run as a standalone script to generate the raw dataset.
"""

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
REQUEST_TIMEOUT = 10


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
    Extracts structured data from the HTML of a single anime item.
    Refactored to reduce complexity and improve maintainability.
    """
    soup = BeautifulSoup(anime_item_html, "html.parser")

    # Basic Info
    title = safe_text(soup.find("h2", class_="h2_anime_title").find("a"))
    start_date_text = safe_text(soup.find("span", class_="js-start_date"))
    release_year = start_date_text[:4] if start_date_text != "N/A" else "N/A"
    synopsis = safe_text(soup.find("div", class_="synopsis js-synopsis").p)

    # Episodes and Status
    info_div = soup.find("div", class_="info")
    number_of_episodes = "N/A"
    if info_div:
        eps_text = info_div.get_text()
        match = re.search(r"(\d+)\s*eps", eps_text)
        if match:
            number_of_episodes = match.group(1)
    status = safe_text(info_div.find("span", class_=["item finished", "item airing"]))

    # Genres, Properties, Scores
    genres_div = soup.find("div", class_="genres-inner")
    genres = (
        ", ".join(g.text for g in genres_div.find_all("a")) if genres_div else "N/A"
    )

    properties_div = soup.find("div", class_="properties")
    studio = _extract_property(properties_div, "Studio")
    source = _extract_property(properties_div, "Source")
    demographic = _extract_property(properties_div, "Demographic")

    rating = safe_float(soup.find("div", class_="score"))
    voters = safe_int(soup.find("div", class_="member"))

    return {
        "Title": title,
        "Episodes": number_of_episodes,
        "Release Year": release_year,
        "Status": status,
        "Genres": genres,
        "Studio": studio,
        "Source": source,
        "Demographic": demographic,
        "Synopsis": synopsis,
        "Voters": voters,
        "Rating": rating,
    }


def fetch_and_scrape(
    url: str, page_limit: int = 1, retries: int = 3, delay: int = 5
) -> list[dict]:
    """
    Fetches and scrapes anime data from a given URL with multiple pages.
    """
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
                break  # Exit retry loop
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    consecutive_404s += 1
                    if consecutive_404s >= 3:
                        print(f"Stopping {url} after 3 consecutive 404 errors.")
                        return all_data  # Genre likely has no more pages
                print(f"HTTP Error on {page_url}: {e}. Retrying...")
                time.sleep(delay)
            except requests.RequestException as e:
                print(f"Request Error on {page_url}: {e}. Retrying...")
                time.sleep(delay)
        else:  # This else belongs to the for-loop, runs if `break` is not hit
            print(f"Failed to fetch {page_url} after {retries} retries. Skipping.")
            continue

    return all_data


def save_data(data: list[dict], date_str: str) -> None:
    """
    Saves the scraped data to a CSV file in the configured raw data directory.
    """
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
    print("Starting the AniMate web scraper...")
    base_url = "https://myanimelist.net/anime/genre/"
    genre_ids = range(1, 44)  # Genre IDs from 1 to 43
    url_list = [f"{base_url}{genre_id}/" for genre_id in genre_ids]

    current_date = get_current_date()
    all_anime_data = []

    for url in tqdm(url_list, desc="Scraping Genres"):
        scraped_data = fetch_and_scrape(url, page_limit=100)
        all_anime_data.extend(scraped_data)

    save_data(all_anime_data, current_date)
    print("Scraping complete.")


if __name__ == "__main__":
    main()
