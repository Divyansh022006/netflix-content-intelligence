import streamlit as st
import pandas as pd
from utils import get_data_path, safe_load_model

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AI Explainability",
    page_icon="🧠",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

DATA_PATH = get_data_path("netflix_with_posters.csv")

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.reset_index(drop=True)

# =========================
# SAFE CLEANING
# =========================

if "title" not in df.columns:
    st.error("Dataset must contain a title column.")
    st.stop()

df["title"] = df["title"].fillna("Unknown")

if "listed_in" not in df.columns:
    df["listed_in"] = "Unknown"

df["listed_in"] = df["listed_in"].fillna("Unknown")

# support both overview and description
if "overview" in df.columns:
    df["overview"] = df["overview"].fillna("")
elif "description" in df.columns:
    df["overview"] = df["description"].fillna("")
else:
    df["overview"] = ""

# =========================
# LOAD MODELS SAFELY
# =========================

tfidf_sim = safe_load_model("similarity_matrix.pkl")
bert_sim = safe_load_model("bert_similarity.pkl")

# =========================
# UI
# =========================

st.title("🧠 AI Explainability Dashboard")
st.caption("Understand why the recommendation engine selected similar content.")

if tfidf_sim is None or bert_sim is None:
    st.warning("ML similarity models were not found. Running in explanation-only mode.")

model_choice = st.radio(
    "Recommendation Model",
    ["TF-IDF", "BERT Semantic"],
    horizontal=True
)

similarity_matrix = tfidf_sim if model_choice == "TF-IDF" else bert_sim

movie = st.selectbox(
    "Select a Movie / TV Show",
    sorted(df["title"].unique())
)

selected_index = df[df["title"] == movie].index[0]

# =========================
# HELPERS
# =========================

def extract_keywords(text):
    stopwords = {
        "the","and","for","with","this","that",
        "from","into","their","they","have",
        "been","will","your","about","after",
        "before","where","when","while","what",
        "which","whose","there","here","over",
        "under","into","than","then","them",
        "movie","film","show"
    }

    words = str(text).lower().split()

    return {
        w.strip(".,!?()[]")
        for w in words
        if len(w) > 3 and w not in stopwords
    }


def fallback(index, top_n=5):
    base = df.iloc[index]

    similar = df[
        (df["listed_in"] == base["listed_in"])
    ]

    similar = similar[similar.index != index]

    return [(i, 0.60) for i in similar.index[:top_n]]


def get_recommendations(index, matrix, top_n=5):

    if matrix is None:
        return fallback(index, top_n)

    scores = list(enumerate(matrix[index]))
    scores.sort(key=lambda x: x[1], reverse=True)

    return [
        (i, float(score))
        for i, score in scores[1:top_n+1]
    ]

# =========================
# GENERATE
# =========================

recommendations = get_recommendations(
    selected_index,
    similarity_matrix
)

base_genres = set(
    str(df.loc[selected_index, "listed_in"]).split(",")
)

base_keywords = extract_keywords(
    df.loc[selected_index, "overview"]
)

st.markdown("## 🎯 Why These Titles Were Recommended")

for idx, score in recommendations:

    row = df.iloc[idx]

    title = row["title"]
    genres = set(str(row["listed_in"]).split(","))
    overview = row["overview"]

    overlap = base_keywords & extract_keywords(overview)

    genre_overlap = len(base_genres & genres)

    with st.container(border=True):

        st.subheader(f"🎬 {title}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Similarity",
            f"{score:.3f}"
        )

        col2.metric(
            "Genre Match",
            genre_overlap
        )

        col3.metric(
            "Keyword Match",
            len(overlap)
        )

        st.markdown("**Why recommended**")

        st.write("• Similar genres")

        st.write("• Similar story themes")

        st.write("• Similar keywords")

        if overlap:
            st.write(
                "**Shared keywords:** "
                + ", ".join(sorted(list(overlap))[:10])
            )

st.divider()

st.caption("🚀 Developed by Divyansh Agarwal")