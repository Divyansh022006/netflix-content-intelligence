import pandas as pd
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_featured.csv"
TEXT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_text.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():
    return pd.read_csv(FEATURED_DATA_PATH)


# ==========================================
# Create Search Text
# ==========================================

def create_search_text(df):

    columns = [
        "title",
        "listed_in",
        "description",
        "director",
        "cast",
        "country",
    ]

    df[columns] = df[columns].fillna("")

    df["search_text"] = (
        df["title"] + " "
        + df["listed_in"] + " "
        + df["description"] + " "
        + df["director"] + " "
        + df["cast"] + " "
        + df["country"]
    )

    # Convert to lowercase
    df["search_text"] = df["search_text"].str.lower()

    return df


# ==========================================
# Save Dataset
# ==========================================

def save_dataset(df):

    df.to_csv(TEXT_DATA_PATH, index=False)

    print(f"\n✅ Text dataset saved to:\n{TEXT_DATA_PATH}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    df = create_search_text(df)

    print("=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)

    print(df[["title", "search_text"]].head())

    save_dataset(df)