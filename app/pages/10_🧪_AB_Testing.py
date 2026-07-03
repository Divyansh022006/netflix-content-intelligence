import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="A/B Testing - Netflix ML",
    page_icon="🧪",
    layout="wide"
)

# =========================
# ROOT FINDER (CLOUD SAFE)
# =========================

ROOT = Path(__file__).resolve()

while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

PROJECT_ROOT = ROOT

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df["title"] = df.get("title", "Unknown").fillna("Unknown")
df["release_year"] = pd.to_numeric(df.get("release_year", 0), errors="coerce").fillna(0).astype(int)
df["listed_in"] = df.get("listed_in", "Unknown").fillna("Unknown")
df["overview"] = df.get("overview", "").fillna("")

# =========================
# SEARCH → INDEX
# =========================

def find_index(query):
    matches = df[df["title"].str.contains(query, case=False, na=False)]
    if len(matches) == 0:
        return None
    return matches.index[0]

# =========================
# SIMPLE RECOMMENDERS (NO ML MODELS)
# =========================

# Model A: Genre + Year based
def recommend_model_a(index, top_n=5):
    base = df.iloc[index]

    similar = df[
        (df["listed_in"] == base["listed_in"]) |
        (abs(df["release_year"] - base["release_year"]) <= 2)
    ]

    similar = similar[similar.index != index].drop_duplicates("title")

    results = []
    for _, row in similar.head(top_n).iterrows():
        results.append({
            "title": row["title"],
            "year": row["release_year"],
            "score": 0.60
        })

    return results


# Model B: Keyword overlap (simple NLP simulation)
def recommend_model_b(index, top_n=5):
    base_words = set(str(df.loc[index, "overview"]).lower().split())

    scores = []

    for i, row in df.iterrows():
        if i == index:
            continue

        words = set(str(row["overview"]).lower().split())
        overlap = len(base_words & words)

        scores.append((i, overlap))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores[:top_n]:
        row = df.iloc[i]
        results.append({
            "title": row["title"],
            "year": row["release_year"],
            "score": float(score)
        })

    return results


# =========================
# SIMULATED METRICS
# =========================

def simulate_clicks(recs):
    return sum(1 for r in recs if r["score"] > 0.5)

def avg_score(recs):
    return np.mean([r["score"] for r in recs]) if recs else 0

# =========================
# UI
# =========================

st.title("🧪 A/B Testing Dashboard (Fixed)")
st.caption("No ML models required — fully deployment safe")

query = st.text_input("🔍 Enter a movie or TV show")

# =========================
# RUN TEST
# =========================

if st.button("Run A/B Test 🚀"):

    if not query.strip():
        st.warning("Please enter a title")
        st.stop()

    idx = find_index(query)

    if idx is None:
        st.error("No matching movie found")
        st.stop()

    # Run both models
    model_a = recommend_model_a(idx)
    model_b = recommend_model_b(idx)

    # Metrics
    a_clicks = simulate_clicks(model_a)
    b_clicks = simulate_clicks(model_b)

    a_avg = avg_score(model_a)
    b_avg = avg_score(model_b)

    winner = "🧠 Model B Wins" if b_clicks > a_clicks else "📊 Model A Wins"

    st.subheader(f"🏆 Winner: {winner}")

    # =========================
    # SIDE BY SIDE
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📊 Model A (Genre + Year)")
        for r in model_a:
            st.write(f"🎬 {r['title']} | ⭐ {r['score']}")

    with col2:
        st.markdown("## 🧠 Model B (Keyword Similarity)")
        for r in model_b:
            st.write(f"🎬 {r['title']} | ⭐ {r['score']}")

    # =========================
    # METRICS TABLE
    # =========================

    st.subheader("📈 Comparison Metrics")

    metrics = pd.DataFrame({
        "Model": ["Model A", "Model B"],
        "Clicks (Simulated)": [a_clicks, b_clicks],
        "Avg Score": [a_avg, b_avg]
    })

    st.dataframe(metrics, use_container_width=True)

    # =========================
    # VISUALIZATION
    # =========================

    fig, ax = plt.subplots()

    ax.bar(["Model A", "Model B"], [a_clicks, b_clicks])
    ax.set_title("A/B Test Performance")

    st.pyplot(fig)