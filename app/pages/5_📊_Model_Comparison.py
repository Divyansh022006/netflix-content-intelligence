import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

# =====================================
# FIND PROJECT ROOT (Cloud Safe)
# =====================================

ROOT = Path(__file__).resolve()

while ROOT != ROOT.parent and not (ROOT / "data").exists():
    ROOT = ROOT.parent

PROJECT_ROOT = ROOT

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    # ---------- Safe Columns ----------

    if "title" not in df.columns:
        df["title"] = "Unknown"

    if "listed_in" not in df.columns:
        df["listed_in"] = "Unknown"

    if "release_year" not in df.columns:
        df["release_year"] = 0

    # Some datasets use description instead of overview
    if "overview" not in df.columns:

        if "description" in df.columns:
            df["overview"] = df["description"]
        else:
            df["overview"] = ""

    df["title"] = df["title"].fillna("Unknown")

    df["listed_in"] = df["listed_in"].fillna("Unknown")

    df["overview"] = df["overview"].fillna("")

    df["release_year"] = (
        pd.to_numeric(
            df["release_year"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    return df


df = load_data()

# =====================================
# SEARCH FUNCTION
# =====================================

def find_title(query):

    matches = df[
        df["title"].str.contains(
            query,
            case=False,
            na=False
        )
    ]

    if len(matches) == 0:
        return None

    return matches.index[0]


# =====================================
# METRIC FUNCTIONS
# =====================================

def average_score(results):

    if len(results) == 0:
        return 0

    return round(
        np.mean([r["score"] for r in results]),
        3
    )


def diversity(results):

    if len(results) == 0:
        return 0

    years = [r["year"] for r in results]

    return round(
        len(set(years)) / len(years),
        3
    )


def overlap(model_a, model_b):

    a = set(r["title"] for r in model_a)
    b = set(r["title"] for r in model_b)

    return len(a & b)
# =====================================
# MODEL A
# Genre + Year Recommendation
# =====================================

def recommend_model_a(index, top_n=10):

    movie = df.iloc[index]

    genre = str(movie["listed_in"])
    year = movie["release_year"]

    candidates = df.copy()

    candidates = candidates[candidates.index != index]

    scores = []

    for i, row in candidates.iterrows():

        score = 0

        # Same genre
        if str(row["listed_in"]) == genre:
            score += 0.70

        # Similar release year
        diff = abs(row["release_year"] - year)

        if diff <= 1:
            score += 0.30

        elif diff <= 3:
            score += 0.20

        elif diff <= 5:
            score += 0.10

        scores.append((i, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for idx, score in scores:

        row = df.iloc[idx]

        recommendations.append({

            "title": row["title"],

            "year": row["release_year"],

            "genre": row["listed_in"],

            "score": round(score,3)

        })

        if len(recommendations) == top_n:
            break

    return recommendations


# =====================================
# MODEL B
# Keyword Similarity
# =====================================

def recommend_model_b(index, top_n=10):

    base_text = str(
        df.loc[index,"overview"]
    ).lower()

    base_words = set(base_text.split())

    scores = []

    for i,row in df.iterrows():

        if i == index:
            continue

        words = set(
            str(row["overview"]).lower().split()
        )

        common = len(
            base_words.intersection(words)
        )

        total = len(
            base_words.union(words)
        )

        similarity = 0

        if total > 0:
            similarity = common / total

        scores.append((i, similarity))

    scores.sort(
        key=lambda x:x[1],
        reverse=True
    )

    recommendations=[]

    for idx,score in scores[:top_n]:

        row=df.iloc[idx]

        recommendations.append({

            "title":row["title"],

            "year":row["release_year"],

            "genre":row["listed_in"],

            "score":round(float(score),3)

        })

    return recommendations


# =====================================
# WINNER FUNCTION
# =====================================

def choose_winner(model_a, model_b):

    score_a = average_score(model_a)

    score_b = average_score(model_b)

    div_a = diversity(model_a)

    div_b = diversity(model_b)

    total_a = score_a + div_a

    total_b = score_b + div_b

    if total_b > total_a:

        return "🧠 Keyword Similarity Model"

    return "🎬 Genre + Year Model"
# =====================================
# PAGE HEADER
# =====================================

st.title("📊 Recommendation Model Comparison")
st.caption(
    "Compare two recommendation strategies without requiring heavy ML models."
)

st.markdown("---")

# =====================================
# USER INPUT
# =====================================

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "🔍 Search for a Movie or TV Show"
    )

with col2:
    top_k = st.selectbox(
        "Top K",
        [5, 10, 15],
        index=1
    )

compare = st.button(
    "🚀 Compare Models",
    use_container_width=True
)

# =====================================
# MAIN
# =====================================

if compare:

    if query.strip() == "":
        st.warning("Please enter a movie title.")
        st.stop()

    idx = find_title(query)

    if idx is None:
        st.error("Movie not found.")
        st.stop()

    model_a = recommend_model_a(idx, top_k)
    model_b = recommend_model_b(idx, top_k)

    winner = choose_winner(model_a, model_b)

    # =====================================
    # KPI CARDS
    # =====================================

    score_a = average_score(model_a)
    score_b = average_score(model_b)

    div_a = diversity(model_a)
    div_b = diversity(model_b)

    common = overlap(model_a, model_b)

    st.markdown("## 🏆 Overall Winner")

    st.success(winner)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Genre+Year Score",
            score_a
        )

    with c2:
        st.metric(
            "Keyword Score",
            score_b
        )

    with c3:
        st.metric(
            "Genre Diversity",
            div_a
        )

    with c4:
        st.metric(
            "Keyword Diversity",
            div_b
        )

    with c5:
        st.metric(
            "Common Movies",
            common
        )

    st.markdown("---")

    # =====================================
    # SIDE BY SIDE RESULTS
    # =====================================

    left, right = st.columns(2)

    with left:

        st.subheader("🎬 Model A")
        st.caption("Genre + Release Year")

        for movie in model_a:

            st.markdown(
                f"""
### 🎬 {movie['title']}

📅 **Year:** {movie['year']}

🎭 **Genre:** {movie['genre']}
"""
            )

            st.progress(movie["score"])

            st.caption(
                f"Similarity : {movie['score']:.3f}"
            )

            st.markdown("---")

    with right:

        st.subheader("🧠 Model B")
        st.caption("Keyword Similarity")

        for movie in model_b:

            st.markdown(
                f"""
### 🎬 {movie['title']}

📅 **Year:** {movie['year']}

🎭 **Genre:** {movie['genre']}
"""
            )

            st.progress(movie["score"])

            st.caption(
                f"Similarity : {movie['score']:.3f}"
            )

            st.markdown("---")

        # =====================================
    # FINAL ANALYTICS SECTION
    # =====================================

    st.markdown("---")
    st.subheader("📈 Model Performance Analytics")

    # Prepare chart data
    labels = ["Genre+Year", "Keyword Model"]

    scores = [score_a, score_b]
    diversities = [div_a, div_b]

    # =====================================
    # SCORE COMPARISON CHART
    # =====================================

    st.markdown("### ⭐ Average Score Comparison")

    fig1, ax1 = plt.subplots()

    ax1.bar(labels, scores)

    ax1.set_ylabel("Average Score")
    ax1.set_title("Model Accuracy Comparison")

    st.pyplot(fig1)

    # =====================================
    # DIVERSITY CHART
    # =====================================

    st.markdown("### 🌍 Diversity Comparison")

    fig2, ax2 = plt.subplots()

    ax2.bar(labels, diversities)

    ax2.set_ylabel("Diversity Score")
    ax2.set_title("Recommendation Diversity")

    st.pyplot(fig2)

    # =====================================
    # OVERLAP INSIGHT
    # =====================================

    st.markdown("### 🔁 Model Agreement")

    if common == 0:

        st.info(
            "Models are highly diverse — they recommend different content."
        )

    elif common < 3:

        st.warning(
            "Moderate overlap between models."
        )

    else:

        st.error(
            "High overlap — models behave similarly."
        )

    # =====================================
    # SUMMARY TABLE
    # =====================================

    st.markdown("### 📋 Final Summary Table")

    summary_df = pd.DataFrame({

        "Metric": [
            "Average Score",
            "Diversity",
            "Top-K",
            "Common Recommendations"
        ],

        "Genre+Year": [
            score_a,
            div_a,
            top_k,
            common
        ],

        "Keyword Model": [
            score_b,
            div_b,
            top_k,
            common
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    # =====================================
    # FINAL WINNER BANNER
    # =====================================

    st.markdown("---")

    st.success(f"🏆 FINAL WINNER: {winner}")

    st.balloons()