import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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
    text = re.sub(r'http[s]?://\S+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def nlp_preprocessing(text, method='lemmatize'):
    """
    Perform NLP preprocessing on the input text.

    Parameters:
    text (str): The input text to be preprocessed.
    method (str, optional): The method to be applied for text preprocessing. Default is 'lemmatize'.
    
    Returns:
    str: The preprocessed text.
    """
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords and non-alphabetic tokens
    
    if method == 'stem':
        tokens = [stemmer.stem(word) for word in tokens]
    elif method == 'lemmatize':
        tokens = lemmatize_with_pos(tokens)
    else:
        raise ValueError("Method must be either 'stem' or 'lemmatize'")
    
    processed_text = ' '.join(tokens)
    return processed_text

def lemmatize_with_pos(tokens):
    """
    Lemmatize tokens with part-of-speech tagging.

    Parameters:
    tokens (list): A list of tokens to be lemmatized.

    Returns:
    list: A list of lemmatized tokens.
    """
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        if tag.startswith('J'):  # Adjective
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='a'))
        elif tag.startswith('V'):  # Verb
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='v'))
        elif tag.startswith('N'):  # Noun
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='n'))
        elif tag.startswith('R'):  # Adverb
            lemmatized_tokens.append(lemmatizer.lemmatize(token, pos='r'))
        else:
            lemmatized_tokens.append(lemmatizer.lemmatize(token))
    return lemmatized_tokens