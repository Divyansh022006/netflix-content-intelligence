import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

MODELS_PATH = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_PATH / "tfidf_vectorizer.pkl"
SIMILARITY_PATH = MODELS_PATH / "similarity_matrix.pkl"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():
    return pd.read_csv(TEXT_DATA_PATH)


# ==========================================
# Build TF-IDF Model
# ==========================================

def build_tfidf(df):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = vectorizer.fit_transform(df["search_text"])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return vectorizer, similarity_matrix


# ==========================================
# Save Model
# ==========================================

def save_models(vectorizer, similarity_matrix):

    MODELS_PATH.mkdir(exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(similarity_matrix, SIMILARITY_PATH)

    print("\n✅ TF-IDF Vectorizer Saved")
    print("✅ Similarity Matrix Saved")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    vectorizer, similarity_matrix = build_tfidf(df)

    print("=" * 60)
    print("TF-IDF ENGINE")
    print("=" * 60)

    print(f"Dataset Shape      : {df.shape}")
    print(f"Similarity Matrix  : {similarity_matrix.shape}")

    save_models(vectorizer, similarity_matrix)