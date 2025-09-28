# src/animate/data_processing.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def preprocess_text(text: str) -> str:
    """Tokenize, stem, and remove stopwords from text.

    :param text: The input text to preprocess.
    :return: The processed text as a single string.
    """
    if not isinstance(text, str):
        return ""

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words
    ]

    return ' '.join(processed_tokens)


def run_processing():
    """Load raw data, process it, and save the result.

    This function reads the raw anime data, applies text preprocessing to the
    synopsis, and saves the cleaned DataFrame to the processed data directory.
    """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Preprocessing text data...")
    df['processed_synopsis'] = df['synopsis'].apply(
        lambda x: preprocess_text(x) if pd.notna(x) else ''
    )

    # FIXME: Redo the preprocessing steps based on the new inputs

    # Ensure the output directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Data processing complete!")


if __name__ == "__main__":
    run_processing()