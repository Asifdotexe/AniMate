"""
Module that contains the test cases for the preprocessing module.
"""
import pytest
from src.preprocessing import preprocess_text

def test_preprocess_text_basic():
    text = "This is a Test Sentence."
    expected = "test sentenc" # "this", "is", "a" are stopwords. "test" -> "test", "sentence" -> "sentenc" (stemming)
    assert preprocess_text(text) == expected

def test_preprocess_text_special_chars():
    text = "Hello!!! World???"
    expected = "hello world"
    assert preprocess_text(text) == expected

def test_preprocess_text_stopwords_only():
    text = "the and is a"
    expected = "" # All stopwords
    assert preprocess_text(text) == expected

def test_preprocess_text_empty():
    text = ""
    expected = ""
    assert preprocess_text(text) == expected

def test_preprocess_text_numbers():
    text = "Anime 123"
    expected = "anim" # "123" is non-alpha
    assert preprocess_text(text) == expected
