import joblib
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_DATA = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"

MODELS = PROJECT_ROOT / "models"

EMBEDDING_PATH = MODELS / "bert_embeddings.pkl"

SIMILARITY_PATH = MODELS / "bert_similarity.pkl"

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(TEXT_DATA)

# ==========================================
# Load Model
# ==========================================

print("Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# Create Embeddings
# ==========================================

embeddings = model.encode(
    df["search_text"].tolist(),
    show_progress_bar=True
)

# ==========================================
# Similarity
# ==========================================

similarity = cosine_similarity(embeddings)

# ==========================================
# Save
# ==========================================

joblib.dump(embeddings, EMBEDDING_PATH)

joblib.dump(similarity, SIMILARITY_PATH)

print("\n✅ Embeddings Saved")

print("✅ Similarity Matrix Saved")