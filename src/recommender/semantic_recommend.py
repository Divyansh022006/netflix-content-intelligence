import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"
SIMILARITY_PATH = PROJECT_ROOT / "models" / "bert_similarity.pkl"

df = pd.read_csv(DATA_PATH)
similarity = joblib.load(SIMILARITY_PATH)


def search(query):
    return df[df["title"].str.contains(query, case=False, na=False)]


def recommend(index, top_n=5):

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nTop Recommendations\n")

    for idx, score in scores[1:top_n + 1]:

        row = df.iloc[idx]

        print(f"🎬 {row['title']}")
        print(f"📅 {row['release_year']}")
        print(f"🎭 {row['listed_in']}")
        print(f"⭐ Similarity : {score:.2%}")
        print("-" * 50)


if __name__ == "__main__":

    query = input("Search title: ")

    results = search(query)

    if results.empty:
        print("No titles found.")
        exit()

    print("\nMatches:\n")

    for i, (_, row) in enumerate(results.head(10).iterrows(), start=1):
        print(f"{i}. {row['title']} ({row['release_year']})")

    choice = int(input("\nChoose a title (1-10): "))

    selected_index = results.head(10).index[choice - 1]

    recommend(selected_index)