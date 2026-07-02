import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Netflix Insights Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

# =========================
# CLEAN DATA (SAFE)
# =========================

df["title"] = df.get("title", "Unknown").fillna("Unknown")
df["release_year"] = pd.to_numeric(df.get("release_year", 0), errors="coerce").fillna(0).astype(int)
df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")
df["type"] = df.get("type", "Unknown").fillna("Unknown")   # Movie / TV Show

# =========================
# TITLE
# =========================

st.title("📊 Netflix Insights Dashboard")
st.markdown("Understand content trends like a data scientist 📈")

# =========================
# KPI SECTION
# =========================

total_titles = len(df)
total_movies = len(df[df["type"].str.contains("Movie", case=False, na=False)])
total_tv = len(df[df["type"].str.contains("TV", case=False, na=False)])
latest_year = df["release_year"].max()

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎬 Total Content", total_titles)
col2.metric("🎥 Movies", total_movies)
col3.metric("📺 TV Shows", total_tv)
col4.metric("📅 Latest Year", int(latest_year))

# =========================
# CONTENT TYPE DISTRIBUTION
# =========================

st.subheader("🎭 Content Type Distribution")

fig1, ax1 = plt.subplots()

type_counts = df["type"].value_counts()

ax1.bar(type_counts.index, type_counts.values)

ax1.set_ylabel("Count")
ax1.set_title("Movies vs TV Shows")

st.pyplot(fig1)

# =========================
# RELEASE YEAR TREND
# =========================

st.subheader("📈 Content Growth Over Years")

year_counts = df[df["release_year"] > 0]["release_year"].value_counts().sort_index()

fig2, ax2 = plt.subplots()

ax2.plot(year_counts.index, year_counts.values)

ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Titles")
ax2.set_title("Netflix Content Growth Trend")

st.pyplot(fig2)

# =========================
# TOP GENRES
# =========================

st.subheader("🎭 Top Genres")

all_genres = df["listed_in"].dropna().str.split(",")

flat_genres = []

for genres in all_genres:
    for g in genres:
        flat_genres.append(g.strip())

genre_series = pd.Series(flat_genres).value_counts().head(10)

fig3, ax3 = plt.subplots()

ax3.barh(genre_series.index[::-1], genre_series.values[::-1])

ax3.set_title("Top 10 Genres")

st.pyplot(fig3)

# =========================
# YEAR FILTER EXPLORER
# =========================

st.subheader("🔍 Explore by Year")

selected_year = st.slider(
    "Select Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    int(df["release_year"].max())
)

filtered = df[df["release_year"] == selected_year]

st.write(f"Total titles in {selected_year}: {len(filtered)}")

st.dataframe(filtered[["title", "type", "listed_in"]].head(20))

# =========================
# INSIGHT BOX
# =========================

st.subheader("🧠 Key Insights")

st.markdown(f"""
- Most common content type: **{type_counts.idxmax()}**
- Total unique genres: **{df['listed_in'].nunique()}**
- Peak content year: **{year_counts.idxmax() if len(year_counts)>0 else 'N/A'}**
""")