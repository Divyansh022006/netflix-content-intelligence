import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="A/B Testing - Netflix ML", layout="wide")

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
# SAFE CLEANING (FIXED)
# =========================

df["title"] = df["title"].fillna("Unknown") if "title" in df.columns else "Unknown"
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

df["overview"] = df["overview"].fillna("") if "overview" in df.columns else ""
df["listed_in"] = df["listed_in"].fillna("Unknown") if "listed_in" in df.columns else "Unknown"

# =========================
# RECOMMENDER
# =========================

def recommend(index, sim_matrix, top_n=5):
    scores = list(enumerate(sim_matrix[index]))
    scores = [(i, float(s)) for i, s in scores]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, score in scores[1:200]:
        row = df.iloc[i]

        results.append({
            "title": row["title"],
            "year": row["release_year"],
            "genre": row.get("listed_in", "Unknown"),
            "score": score
        })

        if len(results) == top_n:
            break

    return results

# =========================
# SIMULATED CLICK MODEL
# =========================

def simulate_clicks(recs):
    # simple heuristic: higher similarity = higher chance of click
    clicks = 0
    for r in recs:
        if r["score"] > 0.20:
            clicks += 1
    return clicks

# =========================
# UI
# =========================

st.title("🧪 A/B Testing: TF-IDF vs BERT")
st.markdown("Compare recommendation quality like a real ML engineer")

query = st.text_input("🔍 Enter any movie/show")

# =========================
# RUN TEST
# =========================

if st.button("Run A/B Test 🚀"):

    if not query.strip():
        st.warning("Please enter a query")
        st.stop()

    # pick index safely
    idx = df.sample(1).index[0]

    tfidf_recs = recommend(idx, tfidf_sim)
    bert_recs = recommend(idx, bert_sim)

    # =========================
    # METRICS
    # =========================

    tfidf_clicks = simulate_clicks(tfidf_recs)
    bert_clicks = simulate_clicks(bert_recs)

    tfidf_avg = np.mean([r["score"] for r in tfidf_recs])
    bert_avg = np.mean([r["score"] for r in bert_recs])

    winner = "🧠 BERT Wins" if bert_clicks > tfidf_clicks else "📊 TF-IDF Wins"

    st.subheader(f"🏆 Winner: {winner}")

    # =========================
    # SIDE BY SIDE
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📊 TF-IDF Results")
        for r in tfidf_recs:
            st.write(f"🎬 {r['title']} | ⭐ {r['score']:.3f}")

    with col2:
        st.markdown("## 🧠 BERT Results")
        for r in bert_recs:
            st.write(f"🎬 {r['title']} | ⭐ {r['score']:.3f}")

    # =========================
    # METRICS TABLE
    # =========================

    st.subheader("📈 Comparison Metrics")

    metrics = pd.DataFrame({
        "Model": ["TF-IDF", "BERT"],
        "Clicks (Simulated)": [tfidf_clicks, bert_clicks],
        "Avg Similarity": [tfidf_avg, bert_avg]
    })

    st.dataframe(metrics)

    # =========================
    # VISUALIZATION
    # =========================

    fig, ax = plt.subplots()

    ax.bar(
        ["TF-IDF", "BERT"],
        [tfidf_clicks, bert_clicks]
    )

    ax.set_title("Simulated Click Performance")

    st.pyplot(fig)