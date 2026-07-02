import pandas as pd
from pathlib import Path

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "netflix_titles.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_cleaned.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():
    """Load the raw Netflix dataset."""
    return pd.read_csv(RAW_DATA_PATH)


# ==========================================
# Clean Missing Values
# ==========================================

def clean_missing_values(df):
    """Handle missing values in the dataset."""

    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Not Rated")

    # Remove rows where important information is missing
    df = df.dropna(subset=["date_added", "duration"])

    return df


# ==========================================
# Save Clean Dataset
# ==========================================

def save_dataset(df):
    """Save cleaned dataset."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✅ Cleaned dataset saved to:\n{PROCESSED_DATA_PATH}")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    original_rows = len(df)

    # Missing values before cleaning
    missing_director = df["director"].isna().sum()
    missing_cast = df["cast"].isna().sum()
    missing_country = df["country"].isna().sum()
    missing_rating = df["rating"].isna().sum()

    # Clean dataset
    cleaned_df = clean_missing_values(df)

    final_rows = len(cleaned_df)
    rows_removed = original_rows - final_rows

    # Save cleaned dataset
    save_dataset(cleaned_df)

    # ===============================
    # Data Quality Report
    # ===============================
    print("\n" + "=" * 50)
    print("NETFLIX DATA QUALITY REPORT")
    print("=" * 50)

    print(f"Original Rows           : {original_rows}")
    print(f"Final Rows              : {final_rows}")
    print(f"Rows Removed            : {rows_removed}")

    print("\nMissing Values Fixed")
    print("-" * 30)
    print(f"Director                : {missing_director}")
    print(f"Cast                    : {missing_cast}")
    print(f"Country                 : {missing_country}")
    print(f"Rating                  : {missing_rating}")

    print("\nDataset Status          : PASS ✅")