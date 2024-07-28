import re
import pandas as pd
import requests
from datetime import datetime
from bs4 import BeautifulSoup as bs

date = datetime.now().strftime('%d%m%y')

def anime_season(month: str) -> str:
    """
    This function converts a given month (as a string) into its corresponding season.

    Parameters:
    - month (str): A string representing the month in the format 'MM'. The valid values are '01' to '12'.

    Returns:
    - str: A string representing the season. The possible values are 'Winter', 'Spring', 'Summer', 'Fall', or 'Unspecified' if the input month is not within the range of 1 to 12.
    """
    month_num = int(month)
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    return seasons[(month_num - 1) // 3] if 1 <= month_num <= 12 else "Unspecified"

def safe_text(element, default='N/A'):
    """
    This function attempts to extract the text content from a given BeautifulSoup element.
    
    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element representing the HTML element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or if the text cannot be parsed.
    
    Returns:
    - str: The text content of the element, or the default value if the element is not found or if the text cannot be parsed.
    """
    return element.text.strip() if element else default

def safe_int(element, default='N/A'):
    """
    This function attempts to extract an integer value from the text of an element.
    
    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element representing the HTML element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or if the text cannot be parsed as an integer.
    
    Returns:
    - int or str: An integer value extracted from the text of the element, or the default value if the text cannot be parsed as an integer.
    """
    try:
        return int(element.text.strip().replace(',', '')) if element else default
    except ValueError:
        return default

def safe_float(element, default='N/A'):
    """
    This function attempts to extract a float value from the text of an element.
    
    Parameters:
    - element (bs4.element.Tag): A BeautifulSoup element representing the HTML element containing the text to be parsed.
    - default (str): A default value to return if the element is not found or if the text cannot be parsed as a float.
    
    Returns:
    - float or str: A float value extracted from the text of the element, or the default value if the text cannot be parsed as a float.
    """
    try:
        return float(element.text.strip()) if element else default
    except ValueError:
        return default

def scrape_anime_data(anime_item) -> dict[str, str]:
    """
    Extract data from the HTML content of an anime item.
    
    Parameters:
    - anime_item (BeautifulSoup): A BeautifulSoup object containing the HTML of an anime item.
    
    Returns:
    - dict: A dictionary with the number of episodes and the release year.
    """
    soup = bs(anime_item, 'html.parser')
    
    start_date_text = soup.find('span', class_='item')
    release_year = start_date_text.text.strip().split(', ')[-1] if start_date_text else 'N/A'
    
    info_div = soup.find('div', class_='info')
    if info_div:
        eps_text = info_div.get_text()
        match = re.search(r'(\d+)\s*eps', eps_text)
        number_of_episodes = match.group(1) if match else 'N/A'
    else:
        number_of_episodes = 'N/A'
    
    status_span = soup.find('span', class_='item finished') or soup.find('span', class_='item airing')
    status = status_span.text.strip() if status_span else 'N/A'
    
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

def fetch_and_scrape(url: str, page_limit: int) -> list[dict[str, str]]:
    """
    Fetches and scrapes anime data from a given URL and page limit.

    Parameters:
    - url (str): The URL of the anime list to scrape.
    - page_limit (int): The maximum number of pages to scrape from the given URL.

    Returns:
    - list[dict[str, str]]: A list of dictionaries containing the scraped anime data. Each dictionary contains the number of episodes and the release year of an anime item.

    The function sends HTTP requests to the specified URL and retrieves the HTML content. It then uses BeautifulSoup to parse the HTML and extract the required data from each anime item. The extracted data is stored in a list of dictionaries and returned as the result.

    If an error occurs during the fetching or scraping process, the function prints an error message and stops further processing.
    """
    all_data = []
    for page_num in range(1, page_limit + 1):
        page_url = f'{url}?page={page_num}'
        print(f'Scraping {page_url}...')

        try:
            response = requests.get(page_url)
            response.raise_for_status()  # Raise an error for bad responses
            soup = bs(response.text, 'html.parser')

            anime_list = soup.find_all('div', class_='anime-item')  # Update with the actual class for anime items
            for anime_item in anime_list:
                anime_data = scrape_anime_data(anime_item)
                all_data.append(anime_data)

        except Exception as e:
            print(f'Error fetching {page_url}: {e}')
            break

    return all_data

def modeler(date: str, data: list[dict[str, str]]) -> None:
    """
    Processes and saves anime data to a CSV file.

    -----
    Parameters:
    - date (str): The date string used to name the CSV file.
    - data (List[Dict[str, str]]): A list of dictionaries containing anime data.

    -----
    The function converts the list of dictionaries to a DataFrame, removes duplicate entries,
    and saves the DataFrame to a CSV file in the 'data/processed' directory.
    """
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    file_path = 'data/raw'    
    df.to_csv(f'../../{file_path}/AnimeData_{date}.csv', index=False)
    print(f'Data saved to {file_path}/AnimeData_{date}.csv')

# EXTRACT AND TRANSFORM
url_list = [   
    'https://myanimelist.net/anime/genre/1/',  # Action
    'https://myanimelist.net/anime/genre/2/',  # Adventure
    'https://myanimelist.net/anime/genre/5/',  # Avant Garde
    'https://myanimelist.net/anime/genre/4/',  # Comedy
    'https://myanimelist.net/anime/genre/8/',  # Drama
    'https://myanimelist.net/anime/genre/10/', # Fantasy
    'https://myanimelist.net/anime/genre/47/', # Gourmet
    'https://myanimelist.net/anime/genre/14/', # Horror
    'https://myanimelist.net/anime/genre/7/',  # Mystery
    'https://myanimelist.net/anime/genre/22/', # Romance
    'https://myanimelist.net/anime/genre/24/', # Sci-fi
    'https://myanimelist.net/anime/genre/36/', # Slice-of-life
    'https://myanimelist.net/anime/genre/30/', # Sport
    'https://myanimelist.net/anime/genre/37/', # Supernatural
    'https://myanimelist.net/anime/genre/41/'  # Suspense
]

if __name__ == '__main__':
    all_data = []
    for url in url_list:
        all_data.extend(fetch_and_scrape(url, 100))
    
    modeler(date, all_data)