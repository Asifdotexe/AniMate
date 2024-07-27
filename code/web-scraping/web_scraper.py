from datetime import datetime

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
    This function extracts and returns the scraped data of an anime from the given anime_item.
   
    Parameters:
    - anime_item (bs4.element.Tag): A BeautifulSoup element representing the anime item to scrape.

    Returns:
    - dict[str, str]: A dictionary containing the scraped data of the anime, with keys representing the data categories and values representing the corresponding data.

    The dictionary contains the following keys and their respective data categories:
    - 'Title': The title of the anime.
    - 'Voters': The number of voters for the anime.
    - 'Avg Score': The average score of the anime.
    - 'Start Date': The start date of the anime.
    - 'Status': The status of the anime (either 'finished' or 'airing').
    - 'Studio': The studio that produced the anime.
    - 'Genres': A comma-separated list of genres for the anime.
    - 'Media': The type of media the anime belongs to.
    - 'Eps': The number of episodes in the anime.
    - 'Duration': The duration of each episode in the anime.
    - 'Synopsis': The synopsis or summary of the anime.
    """
    return {
        'Title': safe_text(anime_item.find('a', class_='link-title')),
        'Voters': safe_int(anime_item.find('div', class_='scormem-item member')),
        'Avg Score': safe_float(anime_item.find('div', title='Score')),
        'Start Date': safe_text(anime_item.find('span', class_='item')),
        'Status': safe_text(
            anime_item.find('span', class_='item finished') or anime_item.find('span', class_='item airing')),
        'Studio': safe_text(anime_item.find('span', class_='producer')),
        'Genres': ', '.join(genre.text.strip() for genre in anime_item.find_all('span', class_='genre')) or 'N/A',
        'Media': safe_text(anime_item.find('span', class_='type')),
        'Eps': safe_text(anime_item.find('span', class_='eps')).split()[0],
        'Duration': safe_text(anime_item.find('span', class_='duration')).split()[0],
        'Synopsis': safe_text(anime_item.find('p', class_='preline'))
    }

def fetch_and_scrape(url: str, page_limit: int) -> list[dict[str, str]]:
    """Fetch and scrape anime data from the given URL."""
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

date = datetime.now().strftime('%d%m%y')

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
        all_data.extend(playwright_scraper(url, 100))
    
    modeler(date, all_data)