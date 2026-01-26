"""
This module provides preprocessing functions for AniMate.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data packages are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by tokenizing, stemming, and removing stopwords.

    :param text: The input text to preprocess.

    :returns: The processed text as a single string after tokenization, stemming, and stopword removal.
    """
    tokens = word_tokenize(text.lower())
    processed = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(processed)
