"""
This module provides preprocessing functions for AniMate.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src import config

# Ensure necessary NLTK data packages are downloaded
for resource in config.NLTK_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by tokenizing, stemming, and removing stopwords.

    :param text: The input text to preprocess.

    :returns: The processed text as a single string after tokenization, stemming, and stopword removal.
    """
    if not text or not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    processed = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(processed)
