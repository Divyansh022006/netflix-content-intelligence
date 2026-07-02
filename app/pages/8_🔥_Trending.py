import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Trending", layout="wide")

st.title("🔥 Netflix Trending Page")
st.markdown("Discover Popular, Recent & AI-Clustered Content")

# =========================
# LOAD DATA
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

df = pd.read_csv(DATA_PATH)

# =========================
# SAFE CLEANING
# =========================

if "title" in df.columns:
    df["title"] = df["title"].fillna("Unknown")
else:
    df["title"] = "Unknown"

if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
else:
    df["release_year"] = 0

if "listed_in" in df.columns:
    df["listed_in"] = df["listed_in"].fillna("Unknown")
else:
    df["listed_in"] = "Unknown"

# =========================
# TREND SCORE (SIMULATED NETFLIX POPULARITY)
# =========================

np.random.seed(42)

df["trend_score"] = (
    df["release_year"] * 0.6 +
    np.random.randint(1, 100, len(df)) * 0.4
)

# =========================
# POPULAR CONTENT
# =========================

st.subheader("⭐ Most Popular Content")

popular = df.sort_values("trend_score", ascending=False).head(10)

for _, row in popular.iterrows():
    st.write(f"🎬 {row['title']} ({row['release_year']})")

st.divider()

# =========================
# RECENT RELEASES
# =========================

st.subheader("🆕 Latest Releases")

recent = df.sort_values("release_year", ascending=False).head(10)

for _, row in recent.iterrows():
    st.write(f"🎬 {row['title']} ({row['release_year']})")

st.divider()

# =========================
# AI CLUSTERING SECTION
# =========================

st.subheader("🧠 AI Cluster Trends (ML-Based)")

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["title"].astype(str))

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

# =========================
# SHOW CLUSTERS
# =========================

for c in sorted(df["cluster"].unique()):

    st.markdown(f"### 🎯 Cluster {c}")

    cluster_df = df[df["cluster"] == c].head(5)

    for _, row in cluster_df.iterrows():
        st.write(f"🎬 {row['title']} ({row['release_year']})")

st.success("Trending Page Loaded Successfully 🚀")