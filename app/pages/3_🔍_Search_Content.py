import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Semantic Search", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
TFIDF_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"
BERT_PATH = PROJECT_ROOT / "models" / "bert_similarity.pkl"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

tfidf_sim = joblib.load(TFIDF_PATH)
bert_sim = joblib.load(BERT_PATH)

# =========================
# SAFE CLEANING (FIXED)
# =========================

def safe_col(df, col, default=""):
    if col in df.columns:
        return df[col].fillna(default)
    else:
        return pd.Series([default] * len(df))

df["title"] = safe_col(df, "title", "Unknown")
df["release_year"] = pd.to_numeric(safe_col(df, "release_year", 0), errors="coerce").fillna(0).astype(int)
df["overview"] = safe_col(df, "overview", "")
df["listed_in"] = safe_col(df, "listed_in", "Unknown")

# =========================
# CREATE TEXT FIELD (IMPORTANT)
# =========================

df["text"] = (
    df["title"].astype(str) + " " +
    df["overview"].astype(str)
).str.lower()

# =========================
# SEARCH FUNCTIONS
# =========================

def title_search(query):
    return df[df["title"].str.contains(query, case=False, na=False)]

def semantic_search(query, similarity_matrix, top_n=10):

    query = query.lower()

    # fallback: find best matching row using text similarity (simple)
    match_idx = df[df["text"].str.contains(query, na=False)].index

    if len(match_idx) == 0:
        # ultra fallback → use first row
        idx = 0
    else:
        idx = match_idx[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = [(i, float(s)) for i, s in scores]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, score in scores[1:200]:

        row = df.iloc[i]

        if str(row["title"]).strip() == "":
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

st.title("🔍 AI Semantic Search Engine")
st.markdown("Search Netflix content using AI understanding")

query = st.text_input("Enter search query")

mode = st.radio(
    "Search Mode",
    ["🏷️ Title Search", "🧠 Semantic Search (AI)"],
    horizontal=True
)

model_choice = st.radio(
    "Model",
    ["TF-IDF", "BERT"],
    horizontal=True
)

# =========================
# MAIN LOGIC
# =========================

if st.button("Search 🚀"):

    if not query.strip():
        st.warning("Enter a query")
        st.stop()

    # -------------------------
    # TITLE SEARCH
    # -------------------------
    if mode.startswith("🏷️"):

        results = title_search(query)

        if results.empty:
            st.error("No results found")
        else:
            st.subheader("Title Matches")

            for i, (_, row) in enumerate(results.head(10).iterrows()):
                st.write(f"{i+1}. {row['title']} ({row['release_year']})")

    # -------------------------
    # SEMANTIC SEARCH
    # -------------------------
    else:

        sim = tfidf_sim if model_choice == "TF-IDF" else bert_sim

        results = semantic_search(query, sim)

        if not results:
            st.error("No semantic results found")
        else:
            st.subheader("AI Ranked Results")

            cols = st.columns(3)

            for i, r in enumerate(results):

                with cols[i % 3]:
                    st.markdown(f"### 🎬 {r['title']}")
                    st.write(f"📅 {r['year']}")
                    st.write(f"🎭 {r['genre']}")
                    st.write(f"⭐ Score: {r['score']}")
                    st.markdown("---")