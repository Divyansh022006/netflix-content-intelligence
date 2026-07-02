import streamlit as st
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Content Explorer", layout="wide")

st.title("🌍 Netflix Content Explorer")
st.markdown("Advanced filtering like real OTT platforms")

# =========================
# LOAD DATA
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

df = pd.read_csv(DATA_PATH)

# =========================
# SAFE CLEANING (FIXED)
# =========================

df["title"] = df["title"].fillna("Unknown")

df["release_year"] = pd.to_numeric(
    df.get("release_year", 0),
    errors="coerce"
).fillna(0).astype(int)

df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")
df["type"] = df.get("type", "Unknown").fillna("Unknown")
df["country"] = df.get("country", "Unknown").fillna("Unknown")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("🎛 Filters")

genre_list = sorted(df["listed_in"].dropna().unique())
type_list = sorted(df["type"].dropna().unique())
country_list = sorted(df["country"].dropna().unique())

selected_genre = st.sidebar.selectbox("Genre", ["All"] + list(genre_list))
selected_type = st.sidebar.selectbox("Type", ["All"] + list(type_list))
selected_country = st.sidebar.selectbox("Country", ["All"] + list(country_list))

min_year = int(df["release_year"].min())
max_year = int(df["release_year"].max())

year_range = st.sidebar.slider(
    "Year Range",
    min_year,
    max_year,
    (min_year, max_year)
)

# =========================
# FILTER DATA
# =========================

filtered = df.copy()

if selected_genre != "All":
    filtered = filtered[filtered["listed_in"] == selected_genre]

if selected_type != "All":
    filtered = filtered[filtered["type"] == selected_type]

if selected_country != "All":
    filtered = filtered[filtered["country"] == selected_country]

filtered = filtered[
    (filtered["release_year"] >= year_range[0]) &
    (filtered["release_year"] <= year_range[1])
]

# =========================
# RESULTS
# =========================

st.subheader(f"🎬 Results: {len(filtered)} titles found")

if len(filtered) == 0:
    st.warning("No results found. Try changing filters.")
else:
    for _, row in filtered.head(50).iterrows():

        st.markdown(f"""
        ---
        ### 🎬 {row['title']}
        - 📅 Year: {row['release_year']}
        - 🎭 Genre: {row['listed_in']}
        - 🌍 Country: {row['country']}
        - 🎞 Type: {row['type']}
        """)