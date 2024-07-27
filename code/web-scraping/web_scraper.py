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

# Preparation
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