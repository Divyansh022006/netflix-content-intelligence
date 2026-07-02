import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
SIMILARITY_PATH = PROJECT_ROOT / "models" / "similarity_matrix.pkl"

# ==========================================
# Load Data
# ==========================================

df = pd.read_csv(DATA_PATH)
similarity_matrix = joblib.load(SIMILARITY_PATH)


# ==========================================
# Search Title
# ==========================================

def search_title(query):

    results = df[df["title"].str.contains(query, case=False, na=False)]

    return results


# ==========================================
# Recommendation Function
# ==========================================

def recommend(movie_index, top_n=5):

    scores = list(enumerate(similarity_matrix[movie_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = scores[1:top_n+1]

    print("\nTop Recommendations\n")

    for i, (idx, score) in enumerate(recommendations, start=1):

        row = df.iloc[idx]

        print(
            f"{i}. {row['title']} "
            f"({row['release_year']}) "
            f"- Score: {score:.2f}"
        )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("NETFLIX CONTENT RECOMMENDER")
    print("=" * 60)

    query = input("\nSearch a title: ")

    results = search_title(query)

    if results.empty:

        print("\nNo title found.")

    else:

        print("\nMatching Titles:\n")

        for idx, row in results.head(10).iterrows():

            print(f"{idx}: {row['title']} ({row['release_year']})")

        movie_index = int(input("\nSelect index: "))

        recommend(movie_index)