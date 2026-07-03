import streamlit as st
import pandas as pd
from utils import get_data_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Netflix Clustering Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

DATA_PATH = get_data_path("netflix_with_posters.csv")

if not DATA_PATH.exists():
    st.error("Dataset not found")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df["title"] = df.get("title", "Unknown").fillna("Unknown")
df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")

# IMPORTANT FIX: safe description handling
if "description" not in df.columns:
    df["description"] = df["title"]

df["description"] = df["description"].fillna(df["title"])

# =========================
# LIMIT DATA (IMPORTANT FIX FOR SPEED)
# =========================

df = df.head(3000)   # prevents cloud crash + speeds up PCA

# =========================
# CLUSTERING (CACHED)
# =========================

@st.cache_data(show_spinner=True)
def run_clustering(data):

    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    X = tfidf.fit_transform(data["description"])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # PCA SAFE (convert sparse → dense but limited data prevents crash)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())

    return clusters, reduced

clusters, reduced = run_clustering(df)
df["cluster"] = clusters

# =========================
# UI
# =========================

st.title("🧠 Netflix Content Clustering")
st.caption("TF-IDF + KMeans + PCA (Cloud Optimized)")

# =========================
# VISUALIZATION
# =========================

fig, ax = plt.subplots()

ax.scatter(
    reduced[:, 0],
    reduced[:, 1],
    c=clusters,
    cmap="tab10",
    s=8
)

ax.set_title("Netflix Clusters (PCA View)")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

st.pyplot(fig)

# =========================
# INSIGHT PANEL
# =========================

st.subheader("📊 Cluster Explorer")

cluster_id = st.selectbox("Select Cluster", sorted(df["cluster"].unique()))

cluster_df = df[df["cluster"] == cluster_id]

st.write(f"Items in cluster: {len(cluster_df)}")

st.dataframe(
    cluster_df[["title", "listed_in"]].head(20),
    use_container_width=True
)