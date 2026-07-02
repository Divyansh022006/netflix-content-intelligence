import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_featured.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():
    """Load the featured Netflix dataset."""
    return pd.read_csv(DATA_PATH)


# ==========================================
# Dataset Overview
# ==========================================

def dataset_overview(df):

    print("=" * 60)
    print("NETFLIX CONTENT INTELLIGENCE PLATFORM")
    print("=" * 60)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nMovies :", (df["type"] == "Movie").sum())
    print("TV Shows :", (df["type"] == "TV Show").sum())


# ==========================================
# Chart : Movies vs TV Shows
# ==========================================

def plot_content_type(df):

    counts = df["type"].value_counts().reset_index()

    counts.columns = ["Content Type", "Count"]

    fig = px.bar(
        counts,
        x="Content Type",
        y="Count",
        text="Count",
        title="Movies vs TV Shows on Netflix",
    )

    fig.show()


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    dataset_overview(df)

    plot_content_type(df)