"""
Module for collecting anime data from MyAnimeList.
This script scrapes data and saves it to the data/raw directory.
"""

import os
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


def get_current_date() -> str:
    """
    Get the current date formatted as 'DDMMYY'.

    :return: Current date formatted as 'DDMMYY'.
    """
    return datetime.now().strftime("%d%m%y")


def safe_text(element: bs, default: str = "N/A") -> str:
    """
    Extract the text content from a given BeautifulSoup element.

    :param element: A BeautifulSoup element containing the text to be parsed.
    :param default: A default value to return if the element is not found.
    :return: The text content of the element, or the default value.
    """
    return element.text.strip() if element else default


def safe_parse(element: bs, cast_type: type = str, default="N/A"):
    """Safely extract and cast text from an element."""
    if not element:
        return default
    text = element.text.strip().replace(",", "")
    try:
        return cast_type(text)
    except (ValueError, TypeError):
        return default


def _extract_title(soup: bs) -> str:
    """Extract anime title."""
    return safe_text(soup.find("h2", class_="h2_anime_title").find("a"))


def _extract_release_year(soup: bs) -> str:
    """Extract release year."""
    start_date_text = safe_text(soup.find("span", class_="js-start_date"), "N/A")
    return start_date_text[:4] if start_date_text != "N/A" else "N/A"


def _extract_episodes(info_div: bs) -> str:
    """Extract number of episodes."""
    if not info_div: return "N/A"
    match = re.search(r"(\d+)\s*eps", info_div.get_text())
    return match.group(1) if match else "N/A"


def _extract_status(info_div: bs) -> str:
    """Extract airing status."""
    if not info_div: return "N/A"
    return safe_text(
        info_div.find("span", class_="item finished") or info_div.find("span", class_="item airing")
    )


def _extract_genres(soup: bs) -> str:
    """Extract genres."""
    div = soup.find("div", class_="genres-inner js-genre-inner")
    return ", ".join(t.get_text(strip=True) for t in div.find_all("span", class_="genre")) if div else "N/A"


def _extract_property(properties_div: bs, caption: str) -> str:
    """Extract specific property by caption."""
    if not properties_div: return "N/A"
    for div in properties_div.find_all("div", class_="property"):
        cap = div.find("span", class_="caption")
        if cap and cap.text.strip() == caption:
            items = div.find_all("span", class_="item")
            return ", ".join(i.get_text(strip=True) for i in items) if items else "N/A"
    return "N/A"


def _extract_themes(properties_div: bs) -> str:
    """Extract anime themes."""
    if not properties_div: return "N/A"
    themes_div = properties_div.find("div", class_="property")
    if themes_div:
        matches = re.findall(r'<span class="item"><a.*?>(.*?)</a></span>', str(themes_div))
        return ", ".join(matches) if matches else "N/A"
    return "N/A"


def _extract_rating(soup: bs) -> float | str:
    """Extract anime rating."""
    div = soup.find("div", class_=re.compile(r"scormem-item score score-label score-\d+"))
    return safe_parse(div, float, "N/A")


def _extract_voters(soup: bs) -> int | str:
    """Extract number of voters."""
    return safe_parse(soup.find("div", class_="scormem-item member"), int, "N/A")


def _extract_synopsis(soup: bs) -> str:
    """Extract anime synopsis."""
    div = soup.find("div", class_="synopsis js-synopsis")
    return safe_text(div.find("p", class_="preline"), "N/A") if div else "N/A"


def scrape_anime_item(anime_item_html: str) -> Dict[str, str]:
    """
    Extract data from the HTML content of an anime item.

    :param anime_item_html: A string containing the HTML of an anime item.
    :return: A dictionary with the extracted anime details.
    """
    soup = bs(anime_item_html, "html.parser")
    info_div = soup.find("div", class_="info")
    properties_div = soup.find("div", class_="properties")

    return {
        "Title": _extract_title(soup),
        "Episodes": _extract_episodes(info_div),
        "Release Year": _extract_release_year(soup),
        "Status": _extract_status(info_div),
        "Genres": _extract_genres(soup),
        "Studio": _extract_property(properties_div, "Studio"),
        "Source": _extract_property(properties_div, "Source"),
        "Demographic": _extract_property(properties_div, "Demographic"),
        "Themes": _extract_themes(properties_div),
        "Synopsis": _extract_synopsis(soup),
        "Voters": _extract_voters(soup),
        "Rating": _extract_rating(soup),
    }


def _fetch_page_data(
    url: str, page: int, retries: int, delay: int, timeout: int
) -> Tuple[List[Dict[str, str]], bool]:
    """
    Fetch and parse a single page of anime data.

    :param url: The base URL.
    :param page: The page number to fetch.
    :param retries: Number of retries.
    :param delay: Delay between retries.
    :param timeout: Request timeout.
    :return: A tuple containing a list of scraped data and a boolean indicating if a 404 error occurred.
    """
    page_url = f"{url}?page={page}"
    for _ in range(retries):
        try:
            response = requests.get(page_url, timeout=timeout)
            response.raise_for_status()
            soup = bs(response.content, "html.parser")
            anime_list = soup.find_all("div", class_="js-anime-category-producer")
            return [scrape_anime_item(str(item)) for item in anime_list], False
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 404:
                return [], True
            print(f"Error fetching {page_url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.RequestException as e:
            print(f"Error fetching {page_url}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return [], False


def fetch_and_scrape(
    url: str, page_limit: int = 1, retries: int = 3, delay: int = 5, timeout: int = 10
) -> List[Dict[str, str]]:
    """
    Fetch and scrape anime data from a given URL.

    :param url: The URL to fetch and scrape.
    :param page_limit: The number of pages to scrape.
    :param retries: The number of retries in case of request failure.
    :param delay: The delay between retries in seconds.
    :param timeout: The timeout for the request in seconds.
    :return: A list of dictionaries containing scraped anime data.
    """
    all_data = []
    consecutive_404_errors = 0

    for page in tqdm(range(1, page_limit + 1), desc="Pages"):
        page_data, is_404 = _fetch_page_data(url, page, retries, delay, timeout)

        if is_404:
            consecutive_404_errors += 1
            if consecutive_404_errors > 3:
                print(
                    f"Encountered 404 error for more than 3 pages in a row for {url}. Moving to the next genre."
                )
                return all_data
        else:
            consecutive_404_errors = 0
            all_data.extend(page_data)

    return all_data


def save_data(date: str, data: List[Dict[str, str]]) -> None:
    """
    Processes and saves anime data to a CSV file.

    :param date: The date string used to name the CSV file.
    :param data: A list of dictionaries containing anime data.
    """
    if not data:
        print("No data collected to save.")
        return

    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)

    # Ensure data/raw exists
    # Assuming script is run from project root
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"AnimeData_{date}.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


def main():
    """
    Main function to scrape anime data and save it to a CSV file.
    """
    base_url = "https://myanimelist.net/anime/genre/"
    genre_ids = range(1, 44)  # Genre IDs from 1 to 43
    url_list = [f"{base_url}{genre_id}/" for genre_id in genre_ids]

    current_date = get_current_date()
    all_anime_data = []

    # Check if we should LIMIT the scrape for testing purposes
    # Ideally passed via args, but for now we default to the original behavior
    # For a quick test, we might want to reduce this, but let's keep original behavior.

    print(f"Starting scrape for {len(url_list)} genres...")

    for url in tqdm(url_list, desc="Genres"):
        # Limited page limit for safety in automated run context,
        # but user likely wants full scrape.
        # However, 100 pages * 43 genres is HUGE.
        # I will stick to the original code's limit of 100.
        scraped_data = fetch_and_scrape(url, page_limit=100)
        all_anime_data.extend(scraped_data)

    save_data(current_date, all_anime_data)


if __name__ == "__main__":
    main()
