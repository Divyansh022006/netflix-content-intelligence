import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Cluster Explorer", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
TFIDF_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)
tfidf_matrix = joblib.load(TFIDF_PATH)

# =========================
# CLEAN DATA
# =========================

df["title"] = df.get("title", "Unknown").fillna("Unknown")
df["release_year"] = pd.to_numeric(df.get("release_year", 0), errors="coerce").fillna(0).astype(int)
df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")

# =========================
# SIDEBAR CONTROLS
# =========================

st.title("🧠 AI Cluster Explorer")
st.markdown("Discover how Netflix content is grouped using ML clustering")

num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

sample_size = st.sidebar.slider("Sample Size (speed control)", 500, 3000, 1500)

# =========================
# SAMPLE DATA (for speed)
# =========================

X = tfidf_matrix[:sample_size]

# =========================
# KMEANS CLUSTERING
# =========================

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df_sample = df.iloc[:sample_size].copy()
df_sample["cluster"] = clusters

# =========================
# PCA VISUALIZATION
# =========================

pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)

df_sample["pca1"] = reduced[:, 0]
df_sample["pca2"] = reduced[:, 1]

# =========================
# PLOT
# =========================

fig, ax = plt.subplots()

scatter = ax.scatter(
    df_sample["pca1"],
    df_sample["pca2"],
    c=df_sample["cluster"],
    cmap="tab10",
    s=10
)

ax.set_title("Netflix Content Clusters (PCA Reduced)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

st.pyplot(fig)

# =========================
# CLUSTER INSIGHTS
# =========================

st.subheader("📊 Cluster Insights")

selected_cluster = st.selectbox("Select Cluster", sorted(df_sample["cluster"].unique()))

cluster_data = df_sample[df_sample["cluster"] == selected_cluster]

st.write(f"Total items in cluster: {len(cluster_data)}")

st.markdown("### 🎬 Sample Titles")

for i, row in cluster_data.head(10).iterrows():
    st.write(f"- {row['title']} ({row['release_year']})")

# =========================
# GENRE BREAKDOWN
# =========================

st.markdown("### 🎭 Genre Distribution in Cluster")

st.bar_chart(cluster_data["listed_in"].value_counts().head(10))