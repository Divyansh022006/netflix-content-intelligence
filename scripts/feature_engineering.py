import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/netflix_extended_2020_2026.csv")
OUTPUT_PATH = Path("data/processed/netflix_text.csv")

df = pd.read_csv(DATA_PATH)

# safe fallback columns
df["title"] = df.get("title", "Unknown")
df["overview"] = df.get("overview", "No description")
df["type"] = df.get("type", "Unknown")
df["release_year"] = df.get("release_year", 0)

# 🔥 BUILD AI TEXT (MOST IMPORTANT PART)
df["tags"] = (
    df["title"].astype(str) + " " +
    df["type"].astype(str) + " " +
    df["overview"].astype(str)
)

df["tags"] = df["tags"].str.lower()

df.to_csv(OUTPUT_PATH, index=False)

print("✅ dataset fixed: no listed_in dependency anymore")