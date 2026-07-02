import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Model Comparison", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
TFIDF_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"
BERT_PATH = PROJECT_ROOT / "models" / "bert_similarity.pkl"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

tfidf_sim = joblib.load(TFIDF_PATH)
bert_sim = joblib.load(BERT_PATH)

# =========================
# CLEAN DATA (SAFE)
# =========================

df["title"] = df.get("title", "Unknown").fillna("Unknown")
df["release_year"] = pd.to_numeric(df.get("release_year", 0), errors="coerce").fillna(0).astype(int)

# =========================
# HELPERS
# =========================

def recommend(sim_matrix, idx, top_n=10):
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores[1:200]:
        row = df.iloc[i]

        results.append({
            "title": row["title"],
            "year": row["release_year"],
            "score": float(score)
        })

        if len(results) == top_n:
            break

    return results


def diversity_score(recs):
    # simple diversity: unique years spread
    years = [r["year"] for r in recs]
    return len(set(years)) / len(years)


# =========================
# UI
# =========================

st.title("📊 Model Comparison: TF-IDF vs BERT")
st.markdown("Compare recommendation quality like a research paper")

query = st.text_input("🔍 Enter a movie/show title")

top_k = st.slider("Top K recommendations", 5, 15, 10)

# =========================
# MAIN LOGIC
# =========================

if st.button("Compare Models 🚀"):

    if not query.strip():
        st.warning("Enter a title")
        st.stop()

    # find index
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
    # SUMMARY TABLE
    # =========================
    st.subheader("⚡ Performance Comparison")

    st.table(pd.DataFrame([
        {"Model": "TF-IDF", "Time (sec)": round(tfidf_time, 4), "Diversity": round(tfidf_div, 3)},
        {"Model": "BERT", "Time (sec)": round(bert_time, 4), "Diversity": round(bert_div, 3)},
    ]))

    # =========================
    # SIDE-BY-SIDE RESULTS
    # =========================
    st.subheader("🎬 Recommendation Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### TF-IDF Results")
        for r in tfidf_recs:
            st.write(f"🎬 {r['title']} ({r['year']}) - ⭐ {round(r['score'],3)}")

    with col2:
        st.markdown("### BERT Results")
        for r in bert_recs:
            st.write(f"🎬 {r['title']} ({r['year']}) - ⭐ {round(r['score'],3)}")

    # =========================
    # OVERLAP ANALYSIS
    # =========================
    st.subheader("🔁 Overlap Analysis")

    tfidf_titles = set(r["title"] for r in tfidf_recs)
    bert_titles = set(r["title"] for r in bert_recs)

    overlap = len(tfidf_titles & bert_titles)

    st.metric("Common Recommendations", overlap)

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📈 Diversity Comparison")

    st.bar_chart(pd.DataFrame({
        "Model": ["TF-IDF", "BERT"],
        "Diversity": [tfidf_div, bert_div]
    }).set_index("Model"))