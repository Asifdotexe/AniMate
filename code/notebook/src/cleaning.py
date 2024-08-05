import re
import string

import re
import string

def nlp_cleaning(text):
    """
    Perform various text cleaning operations on the input text.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text after applying the following operations:
         
    - Convert the text to lowercase.
    - Remove URLs from the text.
    - Remove non-ASCII characters.
    - Remove numbers.
    - Remove punctuation.
    - Remove extra whitespaces.
    """
    if not isinstance(text, str):
        return ''  # Handle non-text values

    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    
    return text