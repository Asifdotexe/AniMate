import re
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup as bs
import time
from tqdm import tqdm

def get_current_date() -> str:
    """
    Get the current date formatted as 'DDMMYY'.

    Returns:
    - str: Current date formatted as 'DDMMYY'.
    """
    return datetime.now().strftime('%d%m%y')


def anime_season(month: str) -> str:
    """
    Converts a given month (as a string) into its corresponding season.

    Parameters:
    - month (str): A string representing the month in the format 'MM'. The valid values are '01' to '12'.

    Returns:
    - str: A string representing the season.
    """
    month_num = int(month)
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return seasons[(month_num - 1) // 3] if 1 <= month_num <= 12 else "Unspecified"


def safe_text(element, default='N/A') -> str:
    """
    Extract the text content from a given BeautifulSoup element.

    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or the text cannot be parsed.

    Returns:
    - str: The text content of the element, or the default value.
    """
    return element.text.strip() if element else default


def safe_int(element, default='N/A') -> int:
    """
    Extract an integer value from the text of an element.

    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or the text cannot be parsed as an integer.

    Returns:
    - int or str: An integer value extracted from the text of the element, or the default value.
    """
    try:
        return int(element.text.strip().replace(',', '')) if element else default
    except ValueError:
        return default


def safe_float(element, default='N/A') -> float:
    """
    Extract a float value from the text of an element.

    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or the text cannot be parsed as a float.

    Returns:
    - float or str: A float value extracted from the text of the element, or the default value.
    """
    try:
        return float(element.text.strip()) if element else default
    except ValueError:
        return default


def fetch_and_scrape(url: str, page_limit: int = 1, retries: int = 3, delay: int = 5) -> list[dict[str, str]]:
    """
    Fetch and scrape anime data from a given URL.

    Parameters:
    - url (str): The URL to fetch and scrape.
    - page_limit (int): The number of pages to scrape.
    - retries (int): The number of retries in case of request failure.
    - delay (int): The delay between retries in seconds.

    Returns:
    - list: A list of dictionaries containing scraped anime data.
    """
    def scrape_anime_data(anime_item) -> dict[str, str]:
        """
        Extract data from the HTML content of an anime item.

        Parameters:
        - anime_item (BeautifulSoup): A BeautifulSoup object containing the HTML of an anime item.

        Returns:
        - dict: A dictionary with the number of episodes and the release year.
        """
        soup = bs(anime_item, 'html.parser')

        title = safe_text(soup.find('h2', class_='h2_anime_title').find('a'))

        start_date_text = safe_text(soup.find('span', class_='js-start_date'), 'N/A')
        release_year = start_date_text[:4] if start_date_text != 'N/A' else 'N/A'

        info_div = soup.find('div', class_='info')
        number_of_episodes = 'N/A'
        if info_div:
            eps_text = info_div.get_text()
            match = re.search(r'(\d+)\s*eps', eps_text)
            number_of_episodes = match.group(1) if match else 'N/A'

        status = safe_text(info_div.find('span', class_='item finished') or info_div.find('span', class_='item airing'))

        genres_div = soup.find('div', class_='genres-inner js-genre-inner')
        genres = ', '.join([genre.find('a').text.strip() for genre in genres_div.find_all('span', class_='genre')]) if genres_div else 'N/A'

        properties_div = soup.find('div', class_='properties')

        def extract_property(caption):
            """Helper function to extract property values based on the caption"""
            if not properties_div:
                return 'N/A'
            property_divs = properties_div.find_all('div', class_='property')
            for div in property_divs:
                caption_span = div.find('span', class_='caption')
                if caption_span and caption_span.text.strip() == caption:
                    item_spans = div.find_all('span', class_='item')
                    return ', '.join(item.get_text(strip=True) for item in item_spans) if item_spans else 'N/A'
            return 'N/A'

        studio = extract_property('Studio')
        source = extract_property('Source')
        demographic = extract_property('Demographic')

        # Extract themes using regex pattern for URLs
        themes = 'N/A'
        if properties_div:
            themes_div = properties_div.find('div', class_='property')
            if themes_div:
                themes_html = str(themes_div)
                theme_matches = re.findall(
                    r'<span class="item"><a href="/anime/genre/\d+/[^"]*" title="[^"]*">([^<]*)</a></span>', themes_html)
                themes = ', '.join(theme_matches) if theme_matches else 'N/A'

        # Extract the rating
        rating = safe_float(soup.find('div', class_='scormem-item score score-label score-8'), 'N/A')

        # Extract the voter count
        voters = safe_int(soup.find('div', class_='scormem-item member'), 'N/A')

        # Extract synopsis
        synopsis = safe_text(soup.find('div', class_='synopsis js-synopsis').find('p', class_='preline'), 'N/A')

        return {
            'Title': title,
            'Episodes': number_of_episodes,
            'Release Year': release_year,
            'Status': status,
            'Genres': genres,
            'Studio': studio,
            'Source': source,
            'Demographic': demographic,
            'Themes': themes,
            'Synopsis': synopsis,
            'Voters': voters,
            'Rating': rating,
        }

    all_data = []
    consecutive_404_errors = 0  # Counter for consecutive 404 errors

    for page in tqdm(range(1, page_limit + 1), desc="Pages"):
        page_url = f"{url}?page={page}"
        for attempt in range(retries):
            try:
                response = requests.get(page_url)
                response.raise_for_status()
                soup = bs(response.content, 'html.parser')
                anime_list = soup.find_all('div', class_='js-anime-category-producer')
                for anime_item in anime_list:
                    anime_data = scrape_anime_data(str(anime_item))
                    all_data.append(anime_data)
                consecutive_404_errors = 0  # Reset counter on successful fetch
                break
            except requests.HTTPError as e:
                if response.status_code == 404:
                    consecutive_404_errors += 1
                    if consecutive_404_errors > 3:
                        print(f"Encountered 404 error for more than 3 pages in a row for {url}. Moving to the next genre.")
                        return all_data
                print(f"Error fetching {page_url}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            except requests.RequestException as e:
                print(f"Error fetching {page_url}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
    return all_data

def modeler(date: str, data: list[dict[str, str]]) -> None:
    """
    Processes and saves anime data to a CSV file.

    Parameters:
    - date (str): The date string used to name the CSV file.
    - data (list[dict[str, str]]): A list of dictionaries containing anime data.

    The function converts the list of dictionaries to a DataFrame, removes duplicate entries,
    and saves the DataFrame to a CSV file in the 'data/raw' directory.
    """
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    file_path = 'data/raw'
    df.to_csv(f'../../{file_path}/AnimeData_{date}.csv', index=False)
    print(f'Data saved to {file_path}/AnimeData_{date}.csv')


def main():
    base_url = 'https://myanimelist.net/anime/genre/'
    genre_ids = range(1, 44)  # Genre IDs from 1 to 43
    url_list = [f'{base_url}{genre_id}/' for genre_id in genre_ids]

    current_date = get_current_date()
    all_anime_data = []

    for url in tqdm(url_list, desc="Genres"):
        scraped_data = fetch_and_scrape(url, page_limit=100)
        all_anime_data.extend(scraped_data)

    modeler(current_date, all_anime_data)

if __name__ == '__main__':
    main()