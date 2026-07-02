import pandas as pd
from pathlib import Path

# ==========================
# Project Paths
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "netflix_titles.csv"

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv(DATA_PATH)

# ==========================
# Basic Information
# ==========================
print("=" * 60)
print("NETFLIX DATASET REPORT")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())