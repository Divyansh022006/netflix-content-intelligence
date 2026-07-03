import streamlit as st
import pandas as pd
from utils import get_data_path

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Netflix Smart Search",
    page_icon="🔍",
    layout="wide"
)

# =========================
# LOAD DATA (SAFE)
# =========================

DATA_PATH = get_data_path("netflix_with_posters.csv")

if not DATA_PATH.exists():
    st.error(f"Dataset not found:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# =========================
# SAFE CLEANING
# =========================

# Title
if "title" in df.columns:
    df["title"] = df["title"].fillna("Unknown")
else:
    df["title"] = "Unknown"

# Release Year
if "release_year" in df.columns:
    df["release_year"] = (
        pd.to_numeric(df["release_year"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
else:
    df["release_year"] = 0

# Genre
if "listed_in" in df.columns:
    df["listed_in"] = df["listed_in"].fillna("Unknown")
else:
    df["listed_in"] = "Unknown"

# Poster URL
if "poster_url" in df.columns:
    df["poster_url"] = df["poster_url"].fillna(
        "https://placehold.co/300x450?text=No+Poster"
    )
else:
    df["poster_url"] = "https://placehold.co/300x450?text=No+Poster"

# =========================
# SEARCH FUNCTION
# =========================

@st.cache_data(show_spinner=False)
def search_movies(query):
    query = query.strip()

    if query == "":
        return df.head(0)

    return (
        df[df["title"].str.contains(query, case=False, na=False)]
        .drop_duplicates(subset="title")
        .sort_values("title")
    )

# =========================
# UI
# =========================

st.title("🔍 Netflix Smart Search")
st.caption("Search movies and TV shows with posters")

query = st.text_input(
    "Search movies or TV shows",
    placeholder="e.g. Stranger Things"
)

# =========================
# MAIN
# =========================

if st.button("🔍 Search", use_container_width=True):

    if query.strip() == "":
        st.warning("Please enter a movie or TV show.")
        st.stop()

    results = search_movies(query)

    if results.empty:
        st.warning("No results found.")
        st.stop()

    st.success(f"Found {len(results)} result(s)")

    cols = st.columns(3)

    for i, (_, row) in enumerate(results.head(15).iterrows()):

        with cols[i % 3]:

            st.image(
                row["poster_url"],
                use_container_width=True
            )

            st.markdown(f"### 🎬 {row['title']}")

            st.write(f"📅 **Year:** {row['release_year']}")
            st.write(f"🎭 **Genre:** {row['listed_in']}")

            st.markdown("---")