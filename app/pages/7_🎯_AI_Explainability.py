import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AI Explainability", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
TFIDF_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"
BERT_PATH = PROJECT_ROOT / "models" / "bert_similarity.pkl"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

# IMPORTANT FIX: reset index so it matches similarity matrix
df = df.reset_index(drop=True)

tfidf_sim = joblib.load(TFIDF_PATH)
bert_sim = joblib.load(BERT_PATH)

# =========================
# SAFE CLEANING
# =========================

for col in ["title", "listed_in", "overview"]:
    if col in df.columns:
        df[col] = df[col].fillna("")
    else:
        df[col] = ""

# =========================
# UI
# =========================

st.title("🎯 AI Explainability Engine")
st.markdown("Understand WHY a movie/show was recommended")

model_choice = st.radio(
    "Choose Model",
    ["TF-IDF (Keyword)", "BERT (Semantic)"],
    horizontal=True
)

similarity_matrix = tfidf_sim if "TF-IDF" in model_choice else bert_sim

# =========================
# MOVIE LIST
# =========================

movie_list = df["title"].unique().tolist()
selected_movie = st.selectbox("Select a movie/show", movie_list)

# FIX: correct index mapping
selected_index = df.index[df["title"] == selected_movie][0]

# =========================
# RECOMMENDATIONS
# =========================

def get_recommendations(index, sim_matrix, top_n=5):
    scores = list(enumerate(sim_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recs = []
    for i, score in scores[1:200]:
        recs.append((i, float(score)))
        if len(recs) == top_n:
            break
    return recs


def extract_keywords(text):
    words = str(text).lower().split()
    stopwords = set(["the", "a", "and", "is", "in", "of", "to", "with", "on", "for"])
    return set([w for w in words if w not in stopwords and len(w) > 2])


recs = get_recommendations(selected_index, similarity_matrix)

# ❌ REMOVED THIS LINE (this was your black stuck header issue)
# st.subheader(f"🎬 Selected: {selected_movie}")

st.markdown("## 🧠 Why these were recommended")

base_genre = str(df.loc[selected_index, "listed_in"])
base_keywords = extract_keywords(df.loc[selected_index, "overview"])

# =========================
# OUTPUT
# =========================

for idx, score in recs:

    title = df.loc[idx, "title"]
    genre = str(df.loc[idx, "listed_in"])
    overview = df.loc[idx, "overview"]

    rec_keywords = extract_keywords(overview)

    genre_match = len(set(base_genre.split(",")) & set(genre.split(",")))
    keyword_match = len(base_keywords & rec_keywords)

    st.markdown(f"""
---

## 🎬 {title}

- ⭐ **Similarity Score:** `{round(score, 3)}`
- 🎭 **Genre Match:** {genre_match}
- 🧠 **Keyword Overlap:** {keyword_match}

### 💡 Why this was recommended:
- Genre similarity
- Semantic similarity (AI embeddings)
- Keyword/story overlap

---
""")

# Footer (your request)
st.markdown("---")
st.markdown("🚀 Developed by **Divyansh Agarwal**")