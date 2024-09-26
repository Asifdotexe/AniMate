import nltk
import gc
import random
import psutil
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize the stemmer
stemmer = PorterStemmer()

# Cache data loading and optimize dtypes
@st.cache_data
def load_data():
    dtypes = {
        'title': 'category',
        'other_name': 'category',
        'genres': 'category',
        'synopsis': 'string',
        'studio': 'category',
        'demographic': 'category',
        'source': 'category',
        'duration_category': 'category',
        'total_duration_hours': 'float32',
        'score': 'float32',
        'image_url': 'string'
    }
    return pd.read_csv('data/final/AnimeData_25092024.csv', usecols=dtypes.keys(), dtype=dtypes)

# Stop words for preprocessing
stop_words = set(stopwords.words('english'))

# Preprocess text: tokenize, remove stopwords, and apply stemming
def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text.lower())
    processed = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(processed)

# Cache the TF-IDF vectorization and k-NN model to avoid recomputation
@st.cache_resource
def vectorize_and_build_model(df: pd.DataFrame) -> tuple[NearestNeighbors, TfidfVectorizer]:
    df['stemmed_synopsis'] = df['synopsis'].apply(lambda x: preprocess_text(x) if pd.notna(x) else '')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['stemmed_synopsis'])
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(tfidf_matrix)
    return knn_model, tfidf_vectorizer

# Recommend anime using the k-NN model and TF-IDF vectorizer
def recommend_anime_knn(query: str, tfidf_vectorizer: TfidfVectorizer, knn_model: NearestNeighbors, top_n: int = 5) -> pd.DataFrame:
    query_processed = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=top_n + 5)
    
    recommendations = data.iloc[indices[0]][['title', 'other_name', 'genres', 'synopsis', 'studio', 'demographic', 'source', 'duration_category', 'total_duration_hours', 'score']]
    filtered_recommendations = recommendations[~recommendations['title'].str.contains(query, case=False, na=False)]
    
    if filtered_recommendations.empty:
        filtered_recommendations = recommendations

    return filtered_recommendations.head(top_n)

# Full pipeline to get recommendations
def anime_recommendation_pipeline(user_query: str, top_n: int = 5) -> pd.DataFrame:
    knn_model, tfidf_vectorizer = vectorize_and_build_model(data)
    recommended_animes = recommend_anime_knn(user_query, tfidf_vectorizer, knn_model, top_n)
    recommended_titles = recommended_animes['title']
    recommendations = data.loc[data['title'].isin(recommended_titles)].sort_values(by='score', ascending=False)
    
    # Free memory after processing
    del recommended_animes
    gc.collect()
    
    return recommendations

# Monitor memory usage
def monitor_memory():
    st.write(f"Memory usage: {psutil.virtual_memory().percent}%")
    gc.collect()

# Streamlit app
st.set_page_config(page_title="AniMate")

# Load custom styles
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'  # Default to landing page

# Load data
data = load_data()

# Define loading phrases
loading_phrases = [
    "🔍 Searching for hidden gems in the anime universe...",
    "✨ Summoning the perfect anime recommendations...",
    "🎉 Gathering the coolest anime just for you...",
    "📚 Digging through the anime archives for you...",
    "🚀 Launching into the world of anime to find your match...",
    "🌟 Fetching the ultimate anime experience...",
    "🌀 Sifting through dimensions for the best recommendations...",
    "💫 Scouring the anime cosmos for your next favorite..."
]

# Landing Page
if st.session_state.page == 'landing':
    st.title("Welcome to AniMate!")

    st.caption("AniMate is a Python-based anime recommendation system that utilizes natural language processing (NLP) to suggest anime based on user preferences.")
    
    st.caption(
        """
        If you enjoy our recommendations, please consider starring our repository on GitHub ⭐!
        """
    )

    if st.button("Recommend Me Something"):
        st.session_state.page = 'recommendations'
    
    monitor_memory()

    st.subheader("Contributors")
    contributors = [
        {"github": "https://github.com/Asifdotexe", "image": "https://avatars.githubusercontent.com/u/115421661?v=4", "alt_name": "Asif Sayyed"},
        {"github": "https://github.com/PranjalDhamane", "image": "https://avatars.githubusercontent.com/u/131870182?v=4", "alt_name": "Pranjal Dhamane"},
        {"github": "https://github.com/tanvisivaraj", "image": "https://avatars.githubusercontent.com/u/132070958?v=4", "alt_name": "Tanvi Sivaraj"},
        {"github": "https://github.com/str04", "image": "https://avatars.githubusercontent.com/u/123924840?v=4", "alt_name": "Shrawani Thakur"},
        {"github": "https://github.com/aditimane07", "image": "https://avatars.githubusercontent.com/u/129670339?v=4", "alt_name": "Aditi Mane"},
    ]
    
    cols = st.columns(len(contributors))
    for col, contributor in zip(cols, contributors):
        with col:
            st.markdown(f"[![Contributor Icon]({contributor['image']})]({contributor['github']})")
            st.caption(contributor['alt_name'])

# Recommendations Page
else:
    st.title("AniMate")
    st.caption("AniMate is a Python-based anime recommendation system that utilizes natural language processing (NLP) to suggest anime based on user preferences")

    monitor_memory()
    
    query, number = st.columns([4, 1])
    with query:
        user_query = st.text_input("Describe a plot! Let's see if we can find something that matches that.")
    with number:
        num_recommendations = st.number_input("No. of results:", min_value=1, max_value=20, value=5)

    if st.button("Get Recommendations"):
        if user_query.strip():
            st.write("### Recommendations based on your input:")

            with st.spinner(random.choice(loading_phrases)):
                recommended_animes = anime_recommendation_pipeline(user_query, num_recommendations)

            if recommended_animes.empty:
                st.warning("No recommendations found. Please try a different query.")
            else:
                for index, row in recommended_animes.iterrows():
                    with st.expander(f"**{row['title'].title()}**"):
                        image_column, text_column = st.columns([1, 3])
                        with image_column:
                            if pd.notna(row['image_url']):
                                st.image(row['image_url'], caption=row['title'].title(), width=100)
                        with text_column:
                            for column in ['other_name', 'genres', 'synopsis', 'studio', 'demographic', 'source', 'duration_category', 'total_duration_hours']:
                                value = row[column]
                                if pd.notna(value):
                                    st.write(f"**{column.replace('_', ' ').title()}:** {value}")
        else:
            st.warning("Please enter a valid query to get recommendations.")