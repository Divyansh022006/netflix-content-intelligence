import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import get_data_path, safe_load_model

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="A/B Testing - Netflix ML",
    page_icon="🧪",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

DATA_PATH = get_data_path("netflix_text.csv")

if not DATA_PATH.exists():
    st.error(f"Dataset not found:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.reset_index(drop=True, inplace=True)

# =========================
# SAFE CLEANING
# =========================

if "title" in df.columns:
    df["title"] = df["title"].fillna("Unknown")
else:
    df["title"] = "Unknown"

if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    ).fillna(0).astype(int)
else:
    df["release_year"] = 0

if "listed_in" in df.columns:
    df["listed_in"] = df["listed_in"].fillna("Unknown")
else:
    df["listed_in"] = "Unknown"

if "overview" in df.columns:
    df["overview"] = df["overview"].fillna("")
else:
    df["overview"] = ""

# =========================
# LOAD MODELS
# =========================

tfidf_sim = safe_load_model("similarity_matrix.pkl")
bert_sim = safe_load_model("bert_similarity.pkl")

# =========================
# SEARCH
# =========================

def find_index(query):

    matches = df[
        df["title"].str.contains(
            query,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        return None

    return matches.index[0]

# =========================
# RECOMMENDER
# =========================

def recommend(index, sim_matrix, top_n=5):

    if sim_matrix is None:
        return []

    try:

        scores = list(enumerate(sim_matrix[index]))
        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = []

        for movie_index, score in scores[1:]:

            if movie_index >= len(df):
                continue

            row = df.iloc[movie_index]

            recommendations.append({
                "title": row["title"],
                "year": row["release_year"],
                "genre": row["listed_in"],
                "score": float(score)
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    except Exception:
        return []

# =========================
# SIMULATED CLICK MODEL
# =========================

def simulate_clicks(recommendations):

    if len(recommendations) == 0:
        return 0

    clicks = 0

    for movie in recommendations:

        if movie["score"] >= 0.20:
            clicks += 1

    return clicks

# =========================
# UI
# =========================

st.title("🧪 A/B Testing Dashboard")

st.caption(
    "Compare recommendation quality between TF-IDF and BERT"
)

query = st.text_input(
    "🔍 Enter a movie or TV show"
)

# =========================
# RUN
# =========================

if st.button("🚀 Run A/B Test"):

    if query.strip() == "":
        st.warning("Please enter a movie name.")
        st.stop()

    index = find_index(query)

    if index is None:
        st.error("Movie not found.")
        st.stop()

    tfidf_results = recommend(index, tfidf_sim)
    bert_results = recommend(index, bert_sim)

    if not tfidf_results and not bert_results:
        st.error("Recommendation models could not be loaded.")
        st.stop()

    tfidf_clicks = simulate_clicks(tfidf_results)
    bert_clicks = simulate_clicks(bert_results)

    tfidf_avg = (
        np.mean([x["score"] for x in tfidf_results])
        if tfidf_results else 0
    )

    bert_avg = (
        np.mean([x["score"] for x in bert_results])
        if bert_results else 0
    )

    winner = (
        "🧠 BERT"
        if bert_avg > tfidf_avg
        else "📊 TF-IDF"
    )

    st.success(f"🏆 Winner: {winner}")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 TF-IDF")

        if tfidf_results:

            for movie in tfidf_results:

                st.write(
                    f"🎬 {movie['title']} "
                    f"({movie['year']}) "
                    f"⭐ {movie['score']:.3f}"
                )

        else:
            st.info("Model unavailable.")

    with col2:

        st.subheader("🧠 BERT")

        if bert_results:

            for movie in bert_results:

                st.write(
                    f"🎬 {movie['title']} "
                    f"({movie['year']}) "
                    f"⭐ {movie['score']:.3f}"
                )

        else:
            st.info("Model unavailable.")

    st.subheader("📈 Metrics")

    metrics = pd.DataFrame({
        "Model": ["TF-IDF", "BERT"],
        "Simulated Clicks": [tfidf_clicks, bert_clicks],
        "Average Similarity": [
            round(tfidf_avg, 3),
            round(bert_avg, 3)
        ]
    })

    st.dataframe(metrics, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["TF-IDF", "BERT"],
        [tfidf_clicks, bert_clicks]
    )

    ax.set_ylabel("Simulated Clicks")
    ax.set_title("A/B Test Comparison")

    st.pyplot(fig)

st.markdown("---")
st.caption("🚀 Developed by Divyansh Agarwal")