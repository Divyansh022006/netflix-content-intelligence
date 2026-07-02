import pandas as pd
from pathlib import Path
from datetime import datetime

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_cleaned.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_featured.csv"


# ==========================================
# Load Dataset
# ==========================================

def load_dataset():
    return pd.read_csv(CLEAN_DATA_PATH)


# ==========================================
# Feature 1 : Content Age
# ==========================================

def create_content_age(df):
    current_year = datetime.now().year
    df["content_age"] = current_year - df["release_year"]
    return df


# ==========================================
# Feature 2 : Movie / TV Show
# ==========================================

def create_is_movie(df):
    df["is_movie"] = df["type"].apply(lambda x: 1 if x == "Movie" else 0)
    return df


# ==========================================
# Feature 3 : Duration Minutes
# ==========================================

def extract_duration_minutes(df):

    def duration(value):

        if pd.isna(value):
            return 0

        value = str(value)

        if "min" in value:
            return int(value.split()[0])

        return 0

    df["duration_minutes"] = df["duration"].apply(duration)

    return df


# ==========================================
# Feature 4 : Season Count
# ==========================================

def extract_season_count(df):

    def seasons(value):

        if pd.isna(value):
            return 0

        value = str(value)

        if "Season" in value:
            return int(value.split()[0])

        return 0

    df["season_count"] = df["duration"].apply(seasons)

    return df


# ==========================================
# Feature 5 : Genre Count
# ==========================================

def create_genre_count(df):

    df["genre_count"] = df["listed_in"].apply(
        lambda x: len(str(x).split(","))
    )

    return df


# ==========================================
# Feature 6 : Country Count
# ==========================================

def create_country_count(df):

    df["country_count"] = df["country"].apply(
        lambda x: len(str(x).split(","))
    )

    return df


# ==========================================
# Feature 7 : Cast Count
# ==========================================

def create_cast_count(df):

    df["cast_count"] = df["cast"].apply(
        lambda x: len(str(x).split(","))
    )

    return df


# ==========================================
# Feature 8 : Description Length
# ==========================================

def create_description_length(df):

    df["description_length"] = df["description"].apply(
        lambda x: len(str(x).split())
    )

    return df


# ==========================================
# Save Dataset
# ==========================================

def save_dataset(df):

    df.to_csv(FEATURED_DATA_PATH, index=False)

    print("\n✅ Featured dataset saved successfully.")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    df = load_dataset()

    df = create_content_age(df)
    df = create_is_movie(df)
    df = extract_duration_minutes(df)
    df = extract_season_count(df)
    df = create_genre_count(df)
    df = create_country_count(df)
    df = create_cast_count(df)
    df = create_description_length(df)

    print("=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)

    print(f"\nDataset Shape : {df.shape}")

    print("\nNew Features")

    print("--------------------------")

    print("content_age")
    print("is_movie")
    print("duration_minutes")
    print("season_count")
    print("genre_count")
    print("country_count")
    print("cast_count")
    print("description_length")

    print("\nPreview")

    print(
        df[
            [
                "title",
                "type",
                "content_age",
                "duration_minutes",
                "season_count",
                "genre_count",
                "country_count",
                "cast_count",
                "description_length",
            ]
        ].head()
    )

    save_dataset(df)