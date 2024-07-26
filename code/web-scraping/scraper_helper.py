import re
from tqdm import tqdm
from time import sleep
from bs4 import BeautifulSoup as bs
from playwright.sync_api import sync_playwright

def anime_season(month: str) -> str:
    """
    This function takes a month as input and returns the corresponding season.
    
    Reason for including this:
    Users might want recommendations for anime from a particular season, like 
    "I want to watch anime from the latest season" or 
    "Show me the best anime from Fall 2022."
    ----- 
    Parameters:
    - month (str): The month for which the season needs to be determined.
    -----
    Returns:
    - str: The season corresponding to the input month. If the month is not recognized, it returns 'Unspecified'.
    -----
    Example:
    >>> anime_season('3')
    'Winter'
    >>> anime_season('7')
    'Summer'
    >>> anime_season('13')
    'Unspecified'
    """
    # Define a dictionary mapping months to seasons
    month_to_season = {
        1: 'Winter', 2: 'Winter', 3: 'Winter',
        4: 'Spring', 5: 'Spring', 6: 'Spring',
        7: 'Summer', 8: 'Summer', 9: 'Summer',
        10: 'Fall', 11: 'Fall', 12: 'Fall'
    }

    # Convert month to integer and get the season
    try:
        month_num = int(month)
        return month_to_season.get(month_num, 'Unspecified')
    except ValueError:
        return 'Unspecified'

def scrape_anime_data(anime_item):
    """
    This function takes an HTML element representing an anime item and returns a dictionary containing various information about the anime.

    ----
    Parameters:
    - anime_item (bs4.element.Tag): An HTML element representing an anime item.
    ----
    Returns:
    - dict: A dictionary containing the following keys and their corresponding values:
        - 'Title': The title of the anime.
        - 'Voters': The number of voters for the anime.
        - 'Avg Score': The average score of the anime.
        - 'Year': The year the anime started.
        - 'Season': The season the anime started in.
        - 'Studio': The studio that produced the anime.
        - 'Genre(s)': A comma-separated string of the genres of the anime.
        - 'Media': The type of media the anime is (e.g., TV, movie, OVA).
        - 'Status': The status of the anime (e.g., airing, finished, on-hold).
        - 'Eps': The number of episodes in the anime.
        - 'Duration(min)': The duration of each episode in minutes.
    """
    anime = bs(anime_item.inner_html(), 'html.parser')
    return {
        'Title': anime.find('span', class_='js-title').text,
        'Voters': int(anime.find('span', class_='js-members').text),
        'Avg Score': float(anime.find('span', class_='js-score').text),
        'Year': anime.find('span', class_='js-start_date').text[:4],
        'Season': anime_season(anime.find('span', class_='js-start_date').text[4:6]),
        'Studio': [studio.text.strip() for studio in anime.find('div', class_='properties')][1].replace('Studio', ''),
        'Genre(s)': ', '.join([data.text.strip() for data in anime.find('div', class_='genres-inner js-genre-inner').select('span')]),
        'Media': re.sub(r'[\W+\d]', '', [data.text for data in anime.find('div', class_='info').select('span')][0]),
        'Status': [data.text for data in anime.find('div', class_='info').select('span')][1],
        'Eps': [data.text.split()[0] for data in anime.find('div', class_='info').select('span')][2],
        'Duration(min)': [data.text.split()[0] for data in anime.find('div', class_='info').select('span')][-1],
    }

def playwright_scraper(url: str, last: int) -> list[dict[str, str]]:
    """
    This function uses Playwright to scrape anime data from a specified URL and its subsequent pages.
    
    -----
    Parameters:
    - url (str): The URL of the anime list to scrape.
    - last (int): The last page number to scrape.
    ----
    Returns:
    - list: A list of dictionaries, where each dictionary represents the scraped data of an anime.
    ----
    The function first launches a Chromium browser using Playwright and navigates to the specified URL. It then iterates through the pages from 1 to the specified last page, scraping anime data from each page. The scraped data is appended to a container list. Finally, the function closes the browser and returns the container list.
    """
    container = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)  # Run in headless mode for efficiency
            page = browser.new_page()
            page.goto(url)

            data_name = page.inner_text('.h1').split()[0]
            print(f'Scraping data from {data_name}...')

            # Use tqdm to display progress bar for page processing
            for page_num in tqdm(range(1, last + 1), desc='Processing Pages', unit='page'):
                page_url = f'{url}?page={page_num}'
                
                try:
                    page.goto(page_url, wait_until='networkidle')
                    if page.query_selector('.error404'):
                        print(f'Page {page_num} of {data_name} does not exist.')
                        break

                    anime_list = page.query_selector_all('.js-anime-category-producer')
                    for anime_item in anime_list:
                        container.append(scrape_anime_data(anime_item))
                
                except Exception as e:
                    print(f'Error processing page {page_num}: {e}')
                    break

        except Exception as e:
            print(f'Error initializing Playwright: {e}')
        
        finally:
            browser.close()

    return container