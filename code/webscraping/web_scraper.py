"""
Module that houses the web scraper to scrape information from MyAnimeList
"""

import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm


def get_current_date() -> str:
    """
    Get the current date formatted as 'DDMMYY'.

    :returns: Current date formatted as 'DDMMYY'.
    """
    return datetime.now().strftime("%d%m%y")


def anime_season(month: str) -> str:
    """
    Converts a given month (as a string) into its corresponding season.

    :param month: A string representing the month in the format 'MM'. The valid values are '01' to '12'.

    :returns: A string representing the season.
    """
    month_num = int(month)
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return seasons[(month_num - 1) // 3] if 1 <= month_num <= 12 else "Unspecified"


def safe_text(element: bs, default: str = "N/A") -> str:
    """
    Extract the text content from a given BeautifulSoup element.

    :param element: A BeautifulSoup element containing the text to be parsed.
    :param default A default value to return if the element is not found or the text cannot be parsed.

    :returns: The text content of the element, or the default value.
    """
    return element.text.strip() if element else default


def safe_int(element: bs, default: str = "N/A") -> int:
    """
    Extract an integer value from the text of an element.

    :param element: A BeautifulSoup element containing the text to be parsed.
    :param default: A default value to return if the element is not found or the text cannot be parsed as an integer.

    :returns: An integer value extracted from the text of the element, or the default value.
    """
    try:
        return int(element.text.strip().replace(",", "")) if element else default
    except ValueError:
        return default


def safe_float(element: bs, default: str = "N/A") -> float:
    """
    Extract a float value from the text of an element.

    :param element: A BeautifulSoup element containing the text to be parsed.
    :default: A default value to return if the element is not found or the text cannot be parsed as a float.

    :returns: A float value extracted from the text of the element, or the default value.
    """
    try:
        return float(element.text.strip()) if element else default
    except ValueError:
        return default


def _extract_title(soup: bs) -> str:
    return safe_text(soup.find("h2", class_="h2_anime_title").find("a"))


def _extract_release_year(soup: bs) -> str:
    start_date_text = safe_text(soup.find("span", class_="js-start_date"), "N/A")
    return start_date_text[:4] if start_date_text != "N/A" else "N/A"


def _extract_episodes(info_div: bs) -> str:
    if not info_div:
        return "N/A"
    eps_text = info_div.get_text()
    match = re.search(r"(\d+)\s*eps", eps_text)
    return match.group(1) if match else "N/A"


def _extract_status(info_div: bs) -> str:
    if not info_div:
        return "N/A"
    return safe_text(
        info_div.find("span", class_="item finished")
        or info_div.find("span", class_="item airing")
    )


def _extract_genres(soup: bs) -> str:
    genres_div = soup.find("div", class_="genres-inner js-genre-inner")
    return (
        ", ".join(
            [
                genre.find("a").text.strip()
                for genre in genres_div.find_all("span", class_="genre")
            ]
        )
        if genres_div
        else "N/A"
    )


def _extract_property(properties_div: bs, caption: str) -> str:
    if not properties_div:
        return "N/A"
    property_divs = properties_div.find_all("div", class_="property")
    for div in property_divs:
        caption_span = div.find("span", class_="caption")
        if caption_span and caption_span.text.strip() == caption:
            item_spans = div.find_all("span", class_="item")
            return (
                ", ".join(item.get_text(strip=True) for item in item_spans)
                if item_spans
                else "N/A"
            )
    return "N/A"


def _extract_themes(properties_div: bs) -> str:
    if not properties_div:
        return "N/A"
    themes_div = properties_div.find("div", class_="property")
    if themes_div:
        themes_html = str(themes_div)
        theme_matches = re.findall(
            r'<span class="item"><a href="/anime/genre/\d+/[^"]*" title="[^"]*">([^<]*)</a></span>',
            themes_html,
        )
        return ", ".join(theme_matches) if theme_matches else "N/A"
    return "N/A"


def _extract_rating(soup: bs) -> float:
    # Find div with class matching "scormem-item score score-label score-X"
    # We use regex to match the pattern because the trailing number varies
    rating_div = soup.find(
        "div", class_=re.compile(r"scormem-item score score-label score-\d+")
    )
    return safe_float(rating_div, "N/A")


def _extract_voters(soup: bs) -> int:
    return safe_int(soup.find("div", class_="scormem-item member"), "N/A")


def _extract_synopsis(soup: bs) -> str:
    synopsis_div = soup.find("div", class_="synopsis js-synopsis")
    return safe_text(
        synopsis_div.find("p", class_="preline") if synopsis_div else None,
        "N/A",
    )


def scrape_anime_item(anime_item_html: str) -> dict[str, str]:
    """
    Extract data from the HTML content of an anime item.

    :param anime_item_html: A string containing the HTML of an anime item.

    :returns: A dictionary with the number of episodes and the release year.
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
) -> tuple[list[dict[str, str]], bool]:
    """
    Fetch and parse a single page of anime data.

    :param url: The base URL.
    :param page: The page number to fetch.
    :param retries: Number of retries.
    :param delay: Delay between retries.
    :param timeout: Request timeout.

    :returns: A tuple containing a list of scraped data and a boolean indicating if a 404 error occurred.
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
            # Safely access response from exception if available
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
) -> list[dict[str, str]]:
    """
    Fetch and scrape anime data from a given URL.

    :param url: The URL to fetch and scrape.
    :param page_limit: The number of pages to scrape.
    :param retries: The number of retries in case of request failure.
    :param delay: The delay between retries in seconds.
    :param timeout: The timeout for the request in seconds.

    :returns: A list of dictionaries containing scraped anime data.
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


def modeler(date: str, data: list[dict[str, str]]) -> None:
    """
    Processes and saves anime data to a CSV file.

    :param: date: The date string used to name the CSV file.
    :param: data: A list of dictionaries containing anime data.
    """
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    file_path = "data/raw"
    df.to_csv(f"../../{file_path}/AnimeData_{date}.csv", index=False)
    print(f"Data saved to {file_path}/AnimeData_{date}.csv")


def main():
    """
    Main function to scrape anime data and save it to a CSV file.
    """
    base_url = "https://myanimelist.net/anime/genre/"
    genre_ids = range(1, 44)  # Genre IDs from 1 to 43
    url_list = [f"{base_url}{genre_id}/" for genre_id in genre_ids]

    current_date = get_current_date()
    all_anime_data = []

    for url in tqdm(url_list, desc="Genres"):
        scraped_data = fetch_and_scrape(url, page_limit=100)
        all_anime_data.extend(scraped_data)

    modeler(current_date, all_anime_data)


if __name__ == "__main__":
    main()
