import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="A/B Testing",
    page_icon="🧪",
    layout="wide"
)

# =========================
# ROOT FINDER
# =========================

ROOT = Path(__file__).resolve()

while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

PROJECT_ROOT = ROOT

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

# =========================
# LOAD DATA
# =========================

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

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

# =========================
# SEARCH
# =========================

def find_index(query):
    matches = df[df["title"].str.contains(query, case=False, na=False)]

    if matches.empty:
        return None

    return matches.index[0]

# =========================
# MODEL A
# Genre + Year
# =========================

def recommend_model_a(index, top_n=5):

    movie = df.iloc[index]

    similar = df[
        (df["listed_in"] == movie["listed_in"]) |
        (abs(df["release_year"] - movie["release_year"]) <= 2)
    ]

    similar = similar.drop_duplicates("title")
    similar = similar[similar.index != index]

    recs = []

    for _, row in similar.head(top_n).iterrows():

        recs.append({
            "title": row["title"],
            "year": int(row["release_year"]),
            "score": 0.65
        })

    return recs

# =========================
# MODEL B
# Genre only
# =========================

def recommend_model_b(index, top_n=5):

    movie = df.iloc[index]

    similar = df[
        df["listed_in"] == movie["listed_in"]
    ]

    similar = similar.sort_values(
        "release_year",
        ascending=False
    )

    similar = similar.drop_duplicates("title")
    similar = similar[similar.index != index]

    recs = []

    score = 0.95

    for _, row in similar.head(top_n).iterrows():

        recs.append({
            "title": row["title"],
            "year": int(row["release_year"]),
            "score": round(score, 2)
        })

        score -= 0.08

    return recs

# =========================
# METRICS
# =========================

def simulate_clicks(recs):
    return len(recs)

def average_score(recs):

    if not recs:
        return 0

    return np.mean([r["score"] for r in recs])

# =========================
# UI
# =========================

st.title("🧪 A/B Testing Dashboard")
st.caption("Compare two recommendation strategies")

query = st.text_input("🔍 Enter a movie or TV show")

# =========================
# RUN
# =========================

if st.button("🚀 Run A/B Test"):

    if not query.strip():
        st.warning("Please enter a title.")
        st.stop()

    idx = find_index(query)

    if idx is None:
        st.error("Movie not found.")
        st.stop()

    model_a = recommend_model_a(idx)
    model_b = recommend_model_b(idx)

    clicks_a = simulate_clicks(model_a)
    clicks_b = simulate_clicks(model_b)

    avg_a = average_score(model_a)
    avg_b = average_score(model_b)

    winner = "📊 Model A"

    if avg_b > avg_a:
        winner = "🧠 Model B"

    st.success(f"Winner: {winner}")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Model A")

        for r in model_a:
            st.write(f"🎬 {r['title']} ({r['year']})")
            st.progress(r["score"])

    with col2:

        st.subheader("🧠 Model B")

        for r in model_b:
            st.write(f"🎬 {r['title']} ({r['year']})")
            st.progress(r["score"])

    st.subheader("📈 Comparison")

    metrics = pd.DataFrame({
        "Model": ["Model A", "Model B"],
        "Clicks": [clicks_a, clicks_b],
        "Average Score": [round(avg_a, 2), round(avg_b, 2)]
    })

    st.dataframe(metrics, use_container_width=True)

    fig, ax = plt.subplots()

    ax.bar(
        ["Model A", "Model B"],
        [avg_a, avg_b]
    )

    ax.set_ylabel("Average Score")
    ax.set_title("Recommendation Comparison")

    st.pyplot(fig)