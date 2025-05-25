# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies['genres'] = movies['genres'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)

    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return movies, nn, tfidf_matrix, indices

def get_recommendations(title, movies, nn, tfidf_matrix, indices, n_recommendations=10):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found!"]
    distances, indices_list = nn.kneighbors(tfidf_matrix[idx], n_neighbors=n_recommendations+1)
    recommended_indices = indices_list.flatten()[1:]  # skip the input movie itself
    return movies['title'].iloc[recommended_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommender System (Content-Based)")

movies, nn, tfidf_matrix, indices = load_data()
selected_movie = st.selectbox("Choose a movie", sorted(movies['title'].unique()))

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie, movies, nn, tfidf_matrix, indices)
    st.write("### Recommended Movies:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")

