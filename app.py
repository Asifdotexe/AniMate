import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
data = pd.read_csv(r'data\final\processed_data_02092024.csv')
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

def preprocess_text(text: str) -> str:
    """
    Tokenize, remove stopwords, and apply stemming to the text.

    :param text: Input text to preprocess.
    :return: Preprocessed text as a single string.
    :rtype: str
    """
    tokens = word_tokenize(text.lower())
    processed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(processed)

def vectorize(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Vectorize the text data using TF-IDF.

    :param df: DataFrame containing text data.
    :return: TF-IDF DataFrame and the vectorizer.
    :rtype: tuple
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_df, tfidf_vectorizer

def build_knn_model(tfidf_features_df: pd.DataFrame, n_neighbors: int = 5) -> NearestNeighbors:
    """
    Build and fit a k-NN model using the TF-IDF features.

    :param tfidf_features_df: DataFrame with TF-IDF features.
    :param n_neighbors: Number of neighbors to use for k-NN (default is 5).
    :return: Fitted k-NN model.
    :rtype: NearestNeighbors
    """
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn_model.fit(tfidf_features_df)
    return knn_model

def recommend_anime_knn(
        query: str, 
        tfidf_vectorizer: TfidfVectorizer, 
        knn_model: NearestNeighbors, 
        top_n: int = 5
    ) -> pd.DataFrame:
    """
    Recommend anime titles based on a user query using the k-NN model.

    :param query: The user's input query.
    :param tfidf_vectorizer: The TF-IDF vectorizer used for the anime data.
    :param knn_model: The k-NN model for finding similar animes.
    :param top_n: Number of recommendations to return (default is 10).
    :return: DataFrame containing the top recommended anime titles.
    :rtype: pd.DataFrame
    """
    query_processed = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n)
    return data.iloc[indices[0]][['title', 'genres']]

def anime_recommendation_pipeline(user_query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Full pipeline to process data, build the k-NN model, and recommend animes based on the user query.
    
    :param user_query: The user's input query for anime recommendation.
    :param top_n: Number of recommendations to return (default is 5).
    :return: DataFrame containing the top recommended anime titles.
    :rtype: pd.DataFrame
    """
    tfidf_features_df, tfidf_vectorizer = vectorize(data)
    knn_model = build_knn_model(tfidf_features_df)
    recommended_animes = recommend_anime_knn(user_query, tfidf_vectorizer, knn_model, top_n)
    return recommended_animes

# Streamlit app
st.set_page_config(page_title="Anime Recommendation System", layout="wide")

# Custom CSS for background and text styling
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background: url(r"data\raw\anime.jpg");
        background-size: cover;
    }
    .stSidebar {
        background-color: #f8f8f8;
    }
    h1 {
        color: #3a3a3a;
        font-family: 'Helvetica', sans-serif;
    }
    .recommendation-title {
        font-size: 20px;
        color: #ff6f61;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Anime Recommendation System")

# Add a logo to the sidebar
st.sidebar.image(r"data\raw\animate.png", use_column_width=True)

st.sidebar.header("About")
st.sidebar.info(
    """
    This app recommends anime titles based on a k-NN model.
    Enter your favorite anime title or a brief description,
    and the system will suggest similar anime for you to enjoy!
    """
)

# User Input
user_query = st.text_input("Enter an anime title or description to get recommendations:")

if user_query:
    st.write("### Recommendations based on your input:")
    
    # Run the recommendation pipeline
    recommended_animes = anime_recommendation_pipeline(user_query)
    
    # Display the results in a better UI format
    cols = st.columns(len(recommended_animes))
    for index, row in recommended_animes.iterrows():
        with cols[index]:
            st.write(f"<div class='recommendation-title'><strong>Title:</strong> {row['title']}</div>", unsafe_allow_html=True)
            st.write(f"**Genres:** {row['genres']}")