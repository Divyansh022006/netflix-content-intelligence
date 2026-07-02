import streamlit as st
import joblib
import pandas as pd
import requests
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AI Netflix Recommender", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

TFIDF_SIM_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"
BERT_SIM_PATH = PROJECT_ROOT / "models" / "bert_similarity.pkl"

# 🔥 YOUR REAL API KEY (IMPORTANT)
TMDB_API_KEY = "479fbff40534ba023225868357a44760"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

tfidf_similarity = joblib.load(TFIDF_SIM_PATH)
bert_similarity = joblib.load(BERT_SIM_PATH)

# =========================
# SAFE CLEANING
# =========================

df["title"] = df["title"].fillna("Unknown")
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
df["overview"] = df["overview"].fillna("") if "overview" in df.columns else ""
df["listed_in"] = df["listed_in"].fillna("Unknown") if "listed_in" in df.columns else "Unknown"

# =========================
# POSTER FUNCTION (FIXED 100%)
# =========================

@st.cache_data
def get_poster(title):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        clean_title = str(title).split("(")[0].strip()

        url = "https://api.themoviedb.org/3/search/multi"

        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": "false"
        }

        res = requests.get(url, params=params, headers=headers, timeout=8)

        if res.status_code != 200:
            return "https://via.placeholder.com/300x450?text=No+Image"

        data = res.json()
        results = data.get("results", [])

        if not results:
            return "https://via.placeholder.com/300x450?text=No+Image"

        # 🔥 BEST MATCH (most popular + has poster)
        best = None
        best_score = 0

        for r in results:
            if r.get("poster_path"):
                score = r.get("popularity", 0)
                if score > best_score:
                    best_score = score
                    best = r

        if best:
            return "https://image.tmdb.org/t/p/w500" + best["poster_path"]

    except Exception as e:
        print("Poster error:", e)

    return "https://via.placeholder.com/300x450?text=No+Image"

# =========================
# SEARCH
# =========================

def search_title(query):
    return df[df["title"].str.contains(query, case=False, na=False)]

# =========================
# RECOMMENDATION ENGINE
# =========================

def recommend(index, similarity_matrix, top_n=5):
    scores = list(enumerate(similarity_matrix[index]))
    scores = [(i, float(s)) for i, s in scores]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for idx, score in scores[1:200]:
        row = df.iloc[idx]

        if pd.isna(row["title"]):
            continue

        results.append({
            "title": row["title"],
            "year": row["release_year"],
            "genre": row["listed_in"],
            "score": round(score, 3)
        })

        if len(results) == top_n:
            break

    return results

# =========================
# UI
# =========================

st.title("🤖 AI Netflix Recommender")
st.markdown("Smart recommendations with working posters")

query = st.text_input("🔍 Search movie/show")

model_choice = st.radio(
    "Choose Model",
    ["TF-IDF", "BERT Semantic AI"],
    horizontal=True
)

# =========================
# MAIN LOGIC
# =========================

if st.button("🚀 Recommend"):

    if not query.strip():
        st.warning("Enter a title")
        st.stop()

    results = search_title(query)

    if results.empty:
        st.error("No match found")
        st.stop()

    st.subheader("Matches")

    for i, (_, row) in enumerate(results.head(5).iterrows()):
        st.write(f"{i+1}. {row['title']} ({row['release_year']})")

    selected_index = results.index[0]

    recs = (
        recommend(selected_index, tfidf_similarity)
        if model_choice == "TF-IDF"
        else recommend(selected_index, bert_similarity)
    )

    st.subheader("Top Recommendations")

    cols = st.columns(3)

    for i, r in enumerate(recs):

        poster = get_poster(r["title"])

        with cols[i % 3]:
            st.image(poster, width=200)
            st.markdown(f"### 🎬 {r['title']}")
            st.write(f"📅 {r['year']}")
            st.write(f"🎭 {r['genre']}")
            st.write(f"⭐ Score: {r['score']}")
            st.markdown("---")