import pandas as pd

df = pd.read_csv("data/processed/netflix_extended_2020_2026.csv")

print(df.head())
print(df.columns)
print(df.isnull().sum())