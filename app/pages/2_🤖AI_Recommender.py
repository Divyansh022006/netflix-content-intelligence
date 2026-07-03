import streamlit as st
import pandas as pd
from utils import get_data_path

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Netflix Recommender",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD DATA (CLOUD SAFE)
# =========================

DATA_PATH = get_data_path("netflix_with_posters.csv")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# =========================
# SAFE CLEANING
# =========================

df["title"] = df.get("title", "Unknown").fillna("Unknown")

df["release_year"] = pd.to_numeric(
    df.get("release_year", 0),
    errors="coerce"
).fillna(0).astype(int)

df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")

# =========================
# SEARCH FUNCTION
# =========================

def search_title(query):
    return df[
        df["title"].str.contains(query, case=False, na=False)
    ]

# =========================
# RECOMMENDER (FAST + STABLE)
# =========================

def recommend(index, top_n=6):
    base = df.iloc[index]

    similar = df[
        (df["listed_in"] == base["listed_in"]) |
        (abs(df["release_year"] - base["release_year"]) <= 3)
    ]

    similar = similar.drop_duplicates(subset="title")
    similar = similar[similar.index != index]

    results = []

    for _, row in similar.head(top_n).iterrows():
        results.append({
            "title": row["title"],
            "year": int(row["release_year"]),
            "genre": row["listed_in"],
            "score": 0.65
        })

    return results

# =========================
# UI
# =========================

st.title("🤖 AI Netflix Recommender")
st.caption("⚡ Fast • Cloud Safe • Clean UI")

query = st.text_input("🔍 Search movie or TV show")

# =========================
# MAIN LOGIC
# =========================

if st.button("🚀 Recommend"):

    if not query.strip():
        st.warning("Please enter a title")
        st.stop()

    results = search_title(query)

    if results.empty:
        st.error("No match found")
        st.stop()

    st.subheader("Matches")

    for i, (_, row) in enumerate(results.head(5).iterrows()):
        st.write(f"{i+1}. {row['title']} ({row['release_year']})")

    selected_index = results.index[0]

    recs = recommend(selected_index)

    st.subheader("🎯 Top Recommendations")

    cols = st.columns(3)

    for i, r in enumerate(recs):

        with cols[i % 3]:
            st.markdown(f"### 🎬 {r['title']}")
            st.write(f"📅 **Year:** {r['year']}")
            st.write(f"🎭 **Genre:** {r['genre']}")
            st.progress(r["score"])
            st.caption(f"Similarity Score: {r['score']}")
            st.markdown("---")