import nltk
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt_tab')

stemmer = PorterStemmer()
data = pd.read_csv('data/final/processed_data_02092024.csv')
stop_words = set(stopwords.words('english'))

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
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['synopsis_processed'])
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
    Ensures that any anime containing the query string in its title is not recommended.

    :param query: The user's input query.
    :param tfidf_vectorizer: The TF-IDF vectorizer used for the anime data.
    :param knn_model: The k-NN model for finding similar animes.
    :param top_n: Number of recommendations to return (default is 5).
    :return: DataFrame containing the top recommended anime titles.
    :rtype: pd.DataFrame
    """
    query_processed = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n + 5)
    recommendations = data.iloc[indices[0]][['title', 'genres', 'synopsis', 'studio', 'demographic', 'source']]
    
    filtered_recommendations = recommendations[~recommendations['title'].str.contains(query, case=False, na=False)]
    if filtered_recommendations.empty:
        filtered_recommendations = recommendations[~recommendations['title'].str.contains(query, case=False, na=False)]
        
    return filtered_recommendations.head(top_n)
        
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

# Code for Streamlit app begins here
st.set_page_config(page_title="Anime Recommendation System")

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Anime Recommendation System")
st.caption("AniMate is a Python-based anime recommendation system that utilizes natural language processing (NLP) to suggest anime based on user preferences")

query, number = st.columns([4, 1])
with query:
    user_query = st.text_input("Enter an anime title or description:")
with number:
    num_recommendations = st.number_input("No. of results:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    if user_query.strip():  # Check if the query is not just empty or whitespace
        st.write("### Recommendations based on your input:")

        with st.spinner('Generating recommendations...'):  # Spinner shows while waiting for results
            recommended_animes = anime_recommendation_pipeline(user_query, num_recommendations)

        if recommended_animes.empty:
            st.warning("No recommendations found. Please try a different query.")
        else:
            for index, row in recommended_animes.iterrows():
                with st.expander(f"**{row['title']}**"):
                    for column in ['genres', 'synopsis', 'studio', 'demographic', 'source']:
                        value = row[column]
                        if pd.notna(value):
                            st.write(f"**{column.replace('_', ' ').title()}:** {value}")
    else:
        st.warning("Please enter a valid query to get recommendations.")