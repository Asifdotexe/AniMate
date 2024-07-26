from datetime import datetime
from data_modeler import modeler
from scraper_helper import playwright_scraper

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