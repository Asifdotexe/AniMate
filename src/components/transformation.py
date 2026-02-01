"""
Module for processing raw anime data.
This script loads the latest raw data, cleans it, applies text preprocessing,
and saves the result to data/processed.
"""

from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from src.config import config
from src.logger import setup_logging

logger = setup_logging("transformation")

# Constants
REQUIRED_COLUMNS = ["title", "synopsis"]
CATEGORY_COLUMNS = [
    "type",
    "source",
    "rating",
    "status",
    "premiered",
    "status",
    "premiered",
    "genres",
    "themes",
    "studio",
    "producer",
    "content rating",
]
NLTK_RESOURCES = ["corpora/stopwords", "tokenizers/punkt_tab", "tokenizers/punkt"]


def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1])


# Initialize lazily
stemmer = PorterStemmer()
stop_words = None


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by tokenizing, stemming, and removing stopwords.

    :param text: The input text to preprocess.
    :returns: The processed text as a single string.
    """
    global stop_words
    
    if not text or not isinstance(text, str):
        return ""
        
    ensure_nltk_resources()
    
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
        
    tokens = word_tokenize(text.lower())
    processed = [
        stemmer.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(processed)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.

    :param df: Raw DataFrame.
    :return: Cleaned DataFrame.
    """
    # Validate required columns
    # Be case-insensitive for columns
    df.columns = df.columns.str.strip().str.lower()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Deduplicate and clean
    df = df.drop_duplicates(subset=["title"])
    df = df.dropna(subset=["synopsis"])

    if "episodes" in df.columns:
        df["episodes"] = df["episodes"].fillna("N/A")

    # Normalize Score/Rating
    if "rating" in df.columns:
        df["score"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    elif "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    else:
        df["score"] = 0

    return df


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to create features for the model.

    :param df: Cleaned DataFrame.
    :return: DataFrame with additional feature columns.
    """
    tqdm.pandas(desc="Preprocessing Synopsis")
    df["stemmed_synopsis"] = df["synopsis"].progress_apply(preprocess_text)

    # create combined features for vectorization
    feature_cols = [
        "title",
        "english title",
        "japanese title",
        "genres",
        "themes",
        "studio",
        "producer",
        "source",
        "content rating",
        "stemmed_synopsis"
    ]
    
    # Fill NAs with empty string for all feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = "" # Handle missing columns gracefully
        else:
            df[col] = df[col].astype(str).fillna("")

    df["combined_features"] = df[feature_cols].agg(" ".join, axis=1)
    
    return df


def main():
    """
    Main execution flow for data processing.
    """
    raw_file = Path(config.paths.raw_data)
    processed_path = Path(config.paths.processed_data)
    processed_dir = processed_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        logger.error(f"Error: Master DB not found at {raw_file}")
        return

    logger.info(f"Processing master database: {raw_file}")
    df = pd.read_csv(raw_file)
    logger.info(f"Loaded {len(df)} rows.")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    df_clean = clean_data(df)
    logger.info(f"Cleaned data: {len(df_clean)} rows.")

    df_processed = process_features(df_clean)

    # Memory Optimization: Convert object columns to category
    for col in CATEGORY_COLUMNS:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype("category")

    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")


if __name__ == "__main__":
    main()
