import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from pathlib import Path
from utils import get_data_path, safe_load_model

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Model Comparison", layout="wide")

# =========================
# SAFE PATHS (CLOUD + LOCAL)
# =========================

DATA_PATH = get_data_path("netflix_text.csv")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# SAFE CLEANING
df["title"] = df.get("title", pd.Series(["Unknown"] * len(df))).fillna("Unknown")
df["release_year"] = pd.to_numeric(
    df.get("release_year", 0),
    errors="coerce"
).fillna(0).astype(int)

# reset index (VERY IMPORTANT for similarity matrix alignment)
df = df.reset_index(drop=True)

# =========================
# LOAD MODELS SAFELY
# =========================

tfidf_sim = safe_load_model("similarity_matrix.pkl")
bert_sim = safe_load_model("bert_similarity.pkl")

if tfidf_sim is None or bert_sim is None:
    st.error("❌ Recommendation models could not be loaded.")
    st.stop()

# =========================
# HELPERS
# =========================

def recommend(sim_matrix, idx, top_n=10):
    try:
        scores = list(enumerate(sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        results = []

        for i, score in scores[1:200]:
            if i >= len(df):
                continue

            row = df.iloc[i]

            results.append({
                "title": row["title"],
                "year": int(row["release_year"]),
                "score": float(score)
            })

            if len(results) == top_n:
                break

        return results

    except Exception:
        return []


def diversity_score(recs):
    if not recs:
        return 0
    years = [r["year"] for r in recs]
    return len(set(years)) / len(years)

# =========================
# UI
# =========================

st.title("📊 Model Comparison: TF-IDF vs BERT")
st.caption("Compare recommendation quality like a research system")

query = st.text_input("🔍 Enter a movie/show title")
top_k = st.slider("Top K recommendations", 5, 15, 10)

# =========================
# MAIN
# =========================

if st.button("Compare Models 🚀"):

    if not query.strip():
        st.warning("Enter a title")
        st.stop()

    matches = df[df["title"].str.contains(query, case=False, na=False)]

    if matches.empty:
        st.error("No match found")
        st.stop()

    idx = matches.index[0]

    # =========================
    # TF-IDF
    # =========================
    start = time.time()
    tfidf_recs = recommend(tfidf_sim, idx, top_k)
    tfidf_time = time.time() - start

    # =========================
    # BERT
    # =========================
    start = time.time()
    bert_recs = recommend(bert_sim, idx, top_k)
    bert_time = time.time() - start

    # =========================
    # METRICS
    # =========================
    tfidf_div = diversity_score(tfidf_recs)
    bert_div = diversity_score(bert_recs)

    # =========================
    # SUMMARY
    # =========================

    st.subheader("⚡ Performance Comparison")

    st.table(pd.DataFrame([
        {
            "Model": "TF-IDF",
            "Time (sec)": round(tfidf_time, 4),
            "Diversity": round(tfidf_div, 3)
        },
        {
            "Model": "BERT",
            "Time (sec)": round(bert_time, 4),
            "Diversity": round(bert_div, 3)
        }
    ]))

    # =========================
    # RESULTS
    # =========================

    st.subheader("🎬 Recommendation Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### TF-IDF Results")
        if tfidf_recs:
            for r in tfidf_recs:
                st.write(f"🎬 {r['title']} ({r['year']}) - ⭐ {r['score']:.3f}")
        else:
            st.warning("No TF-IDF results")

    with col2:
        st.markdown("### BERT Results")
        if bert_recs:
            for r in bert_recs:
                st.write(f"🎬 {r['title']} ({r['year']}) - ⭐ {r['score']:.3f}")
        else:
            st.warning("No BERT results")

    # =========================
    # OVERLAP
    # =========================

    st.subheader("🔁 Overlap Analysis")

    overlap = len(
        set(r["title"] for r in tfidf_recs)
        & set(r["title"] for r in bert_recs)
    )

    st.metric("Common Recommendations", overlap)

    # =========================
    # CHART
    # =========================

    st.subheader("📈 Diversity Comparison")

    st.bar_chart(pd.DataFrame({
        "Model": ["TF-IDF", "BERT"],
        "Diversity": [tfidf_div, bert_div]
    }).set_index("Model"))