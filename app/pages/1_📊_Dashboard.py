import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_extended_2020_2026.csv"

# =========================
# LOAD DATA (SAFE)
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# CLEANING (IMPORTANT FIX)
# =========================
for col in ["country", "listed_in", "rating", "language"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
df = df.dropna(subset=["release_year"])
df["release_year"] = df["release_year"].astype(int)

# =========================
# TITLE
# =========================
st.title("🎬 Netflix Intelligence Dashboard")
st.markdown("### 2020–2026 Content Analytics")

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("📺 Total Titles", len(df))
col2.metric("🎬 Movies", len(df[df["type"] == "Movie"]))
col3.metric("📺 TV Shows", len(df[df["type"] == "TV Show"]))

st.divider()

# =========================
# TYPE DISTRIBUTION
# =========================
fig1 = px.pie(df, names="type", title="Movies vs TV Shows")
st.plotly_chart(fig1, use_container_width=True)

# =========================
# RELEASE YEAR TREND (FIXED)
# =========================
st.subheader("📅 Release Trend (2020–2026)")

year_df = df.groupby("release_year").size().reset_index(name="count")
year_df = year_df.sort_values("release_year")

fig2 = px.line(
    year_df,
    x="release_year",
    y="count",
    markers=True,
    title="Content Growth Over Time"
)

fig2.update_xaxes(dtick=1)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# TOP GENRES (FIXED SAFE)
# =========================
st.subheader("🎭 Top Genres")

if "listed_in" in df.columns:
    genres = (
        df["listed_in"]
        .astype(str)
        .str.split(",")
        .explode()
        .value_counts()
        .head(10)
        .reset_index()
    )

    genres.columns = ["Genre", "Count"]

    fig3 = px.bar(genres, x="Count", y="Genre", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# TOP COUNTRIES (ONLY IF EXISTS)
# =========================
if "country" in df.columns:
    st.subheader("🌍 Top Countries")

    countries = (
        df["country"]
        .astype(str)
        .str.split(",")
        .explode()
        .value_counts()
        .head(10)
        .reset_index()
    )

    countries.columns = ["Country", "Count"]

    fig4 = px.bar(countries, x="Count", y="Country", orientation="h")
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# RATING DISTRIBUTION
# =========================
st.subheader("⭐ Rating Distribution")

fig5 = px.histogram(df, x="rating", nbins=20)
st.plotly_chart(fig5, use_container_width=True)

# =========================
# PREVIEW
# =========================
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)