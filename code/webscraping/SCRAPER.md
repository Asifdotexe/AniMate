# Anime Data Scraper

## Overview

The Anime Data Scraper is a Python script that scrapes anime data from a specified list of URLs and saves the data to a CSV file. It uses BeautifulSoup for parsing HTML, Requests for HTTP requests, and tqdm for progress tracking. This document explains the script’s components, assumptions, and flow.

## Components

### 1. Functions

#### `get_current_date()`

- **Purpose**: Retrieves the current date formatted as 'DDMMYY'.
- **Returns**: A string representing the current date.

#### `anime_season(month: str) -> str`

- **Purpose**: Converts a given month (in 'MM' format) into its corresponding season.
- **Parameters**:
  - `month`: A string representing the month.
- **Returns**: A string representing the season.

#### `safe_text(element, default='N/A') -> str`

- **Purpose**: Extracts the text content from a BeautifulSoup element.
- **Parameters**:
  - `element`: A BeautifulSoup element.
  - `default`: A default value if the element is not found.
- **Returns**: The text content or the default value.

#### `safe_int(element, default='N/A') -> int`

- **Purpose**: Extracts an integer value from the text of a BeautifulSoup element.
- **Parameters**:
  - `element`: A BeautifulSoup element.
  - `default`: A default value if the text cannot be parsed as an integer.
- **Returns**: An integer value or the default value.

#### `safe_float(element, default='N/A') -> float`

- **Purpose**: Extracts a float value from the text of a BeautifulSoup element.
- **Parameters**:
  - `element`: A BeautifulSoup element.
  - `default`: A default value if the text cannot be parsed as a float.
- **Returns**: A float value or the default value.

#### `fetch_and_scrape(url: str, page_limit: int = 1, retries: int = 3, delay: int = 5) -> List[Dict[str, str]]`

- **Purpose**: Fetches and scrapes anime data from a given URL.
- **Parameters**:
  - `url`: The URL to fetch and scrape.
  - `page_limit`: Number of pages to scrape.
  - `retries`: Number of retries in case of request failure.
  - `delay`: Delay between retries in seconds.
- **Returns**: A list of dictionaries containing scraped anime data.

#### `modeler(date: str, data: List[Dict[str, str]]) -> None`

- **Purpose**: Processes and saves anime data to a CSV file.
- **Parameters**:
  - `date`: The date string used to name the CSV file.
  - `data`: A list of dictionaries containing anime data.

#### `main()`

- **Purpose**: The main function that initiates the scraping process for a list of URLs and saves the data.
- **Details**: Iterates over a list of URLs, calls `fetch_and_scrape` for each, and then calls `modeler` to save the data.

## Assumptions

1. The structure of the HTML pages on the target website follows the expected patterns.
2. The URL list provided covers all the desired genres.
3. The website's structure and classes used for scraping (e.g., `h2_anime_title`, `js-start_date`) are consistent.

## Flow of the Script

1. **Start**: The script starts by calling the `main()` function.
2. **Current Date**: `get_current_date()` is called to get the current date.
3. **URL Iteration**: For each URL in the list, `fetch_and_scrape()` is called to fetch and scrape data.
4. **Scraping**:
   - **Page Loop**: Iterates over pages within the page limit.
   - **Request Handling**: Makes an HTTP request to fetch page content.
   - **Error Handling**: Handles HTTP errors and retries if necessary.
   - **Data Extraction**: Uses BeautifulSoup to parse and extract data from each anime item.
5. **Data Processing**: `modeler()` is called to process and save the scraped data into a CSV file.
6. **End**: The script ends after processing all URLs.

## Structure

+----------------------------------+
|            main()                |
+----------------------------------+
|                                  |
|  +----------------------------+  |
|  |   get_current_date()       |  |
|  +----------------------------+  |
|                                  |
|  +----------------------------+  |
|  |   fetch_and_scrape()       |  |
|  +----------------------------+  |
|  |                            |  |
|  |  +----------------------+  |  |
|  |  |   requests.get()     |  |  |
|  |  +----------------------+  |  |
|  |  |   BeautifulSoup()    |  |  |
|  |  +----------------------+  |  |
|  |  |   scrape_anime_data()|  |  |
|  |  +----------------------+  |  |
|  |                            |  |
|  +----------------------------+  |
|                                  |
|  +----------------------------+  |
|  |         modeler()          |  |
|  +----------------------------+  |
|                                  |
+----------------------------------+


## Flow of the function

Start
  |
  v
get_current_date()
  |
  v
For each URL in URL list
  |
  v
fetch_and_scrape(url)
  |
  +--------------------------+
  |                          |
  |  Page Loop               |
  |    |                     |
  |    v                     |
  |  Request Handling        |
  |    |                     |
  |    v                     |
  |  Error Handling          |
  |    |                     |
  |    v                     |
  |  Data Extraction         |
  +--------------------------+
  |
  v
modeler(date, data)
  |
  v
End

## Conclusion
This script provides an automated way to scrape anime data from a set of URLs and store the data in a CSV file. It handles errors and retries fetching data as necessary, ensuring that it can process a large amount of data efficiently.
