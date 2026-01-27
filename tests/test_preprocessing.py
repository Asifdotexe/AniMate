"""
Module that contains the test cases for the preprocessing module.
"""

from src.preprocessing import preprocess_text


def test_preprocess_text_basic():
    """Test basic tokenization, stemming, and stopword removal."""
    text = "This is a Test Sentence."
    expected = "test sentenc"  # "this", "is", "a" are stopwords. "test" -> "test", "sentence" -> "sentenc" (stemming)
    assert preprocess_text(text) == expected


def test_preprocess_text_special_chars():
    """Test handling of special characters."""
    text = "Hello!!! World???"
    expected = "hello world"
    assert preprocess_text(text) == expected


def test_preprocess_text_stopwords_only():
    """Test text containing only stopwords."""
    text = "the and is a"
    expected = ""  # All stopwords
    assert preprocess_text(text) == expected


def test_preprocess_text_empty():
    """Test handling of empty string."""
    text = ""
    expected = ""
    assert preprocess_text(text) == expected


def test_preprocess_text_numbers():
    """Test handling of numeric characters."""
    text = "Anime 123"
    expected = "anim"  # "123" is non-alpha
    assert preprocess_text(text) == expected
