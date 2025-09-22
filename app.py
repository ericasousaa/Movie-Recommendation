import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# Config
# ======================
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# ======================
# Funções auxiliares
# ======================
@st.cache_data
def load_data(path="data/processed/movies_cleaned.csv"):
    return pd.read_csv(path)

@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["soup"].fillna(""))
    sim = cosine_similarity(matrix, matrix)
    return sim, df

def clean_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return ", ".join([g["name"] for g in genres])
    except:
        return str(genre_str)

def recommend(title, df, sim, n=10):
    if title not in df["original_title"].values:
        return pd.DataFrame()
    idx = df[df["original_title"] == title].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recs = df.iloc[[i[0] for i in scores]][
        ["original_title", "genres", "vote_average", "release_date"]
    ].copy()
    recs["score"] = [i[1] for i in scores]
    return recs

# ======================
# App Streamlit
# ======================
st.title("Movie Recommendation System")
st.markdown("Enter a movie title or pick one from the list to get personalized recommendations!")

# Load data & model
df = load_data()
sim, df = build_model(df)

# Inputs
user_input = st.text_input("🔹 Type a movie title:", "")
movie_select = st.selectbox("Or select from the dataset:", df["original_title"].values)
top_n = st.slider("Number of recommendations", 5, 20, 10)

# Botão
if st.button("Recommend"):
    movie = user_input if user_input.strip() else movie_select
    results = recommend(movie, df, sim, top_n)

    if results is not None and not results.empty:
        for _, row in results.iterrows():
            # Título + ano + ícone de câmera 🎥
            st.markdown(f"## 🎥 {row['original_title']} ({row['release_date'][:4]})")

            # Linha com rating e gêneros
            st.markdown(
                f"⭐ {row['vote_average']} | 🎭 {clean_genres(row['genres'])}"
            )

            # Barra de similaridade
            st.progress(float(row["score"]))
            st.markdown("---")
    else:
        st.warning("Movie not found in dataset.")
