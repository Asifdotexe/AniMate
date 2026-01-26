"""
Module that contains the test cases for the web scraper module.
"""

from unittest.mock import MagicMock, patch

import pytest
from requests import HTTPError
from web_scraper import _fetch_page_data, scrape_anime_item

sample_html = """
<div class="js-anime-category-producer">
    <h2 class="h2_anime_title"><a href="#">Test Anime</a></h2>
    <span class="js-start_date">2023-01-01</span>
    <div class="info">
        12 eps
        <span class="item finished">Finished Airing</span>
    </div>
    <div class="genres-inner js-genre-inner">
        <span class="genre"><a href="#">Action</a></span>
    </div>
    <div class="properties">
        <div class="property">
            <span class="caption">Studio</span>
            <span class="item">Mappa</span>
        </div>
    </div>
    <div class="scormem-item score score-label score-8">8.52</div>
    <div class="scormem-item member">1,234</div>
    <div class="synopsis js-synopsis">
        <p class="preline">Synopsis here.</p>
    </div>
</div>
"""


def test_scrape_anime_item():
    data = scrape_anime_item(sample_html)
    assert data["Title"] == "Test Anime"
    assert data["Episodes"] == "12"
    assert data["Studio"] == "Mappa"
    assert data["Rating"] == 8.52


@patch("web_scraper.requests.get")
def test_fetch_page_data_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = sample_html
    mock_get.return_value = mock_response

    data, is_404 = _fetch_page_data("http://test.com", 1, 1, 1, 10)

    assert not is_404
    assert len(data) == 1
    assert data[0]["Title"] == "Test Anime"


@patch("web_scraper.requests.get")
def test_fetch_page_data_404(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    # Ensure the exception has the response attached
    error = HTTPError("404 Error", response=mock_response)
    mock_response.raise_for_status.side_effect = error
    mock_get.return_value = mock_response

    data, is_404 = _fetch_page_data("http://test.com", 1, 1, 0, 10)

    assert is_404
    assert len(data) == 0
