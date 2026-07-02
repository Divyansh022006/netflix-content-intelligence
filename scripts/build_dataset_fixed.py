import requests
import pandas as pd
from pathlib import Path
import time

API_KEY = "479fbff40534ba023225868357a44760"

BASE_URL = "https://api.themoviedb.org/3"

all_data = []

# =========================
# FETCH MOVIES + TV
# =========================

def fetch_data(endpoint, content_type, start_year=2020, end_year=2026):

    for year in range(start_year, end_year + 1):
        print(f"\nFetching {content_type} year: {year}")

        for page in range(1, 6):  # enough data

            url = f"{BASE_URL}/discover/{endpoint}"

            params = {
                "api_key": API_KEY,
                "sort_by": "popularity.desc",
                "page": page,
                "primary_release_date.gte" if endpoint == "movie" else "first_air_date.gte": f"{year}-01-01",
                "primary_release_date.lte" if endpoint == "movie" else "first_air_date.lte": f"{year}-12-31",
            }

            res = requests.get(url, params=params)

            if res.status_code != 200:
                print("API error:", res.text)
                break

            data = res.json().get("results", [])

            print(f"{content_type} {year} page {page}: {len(data)} items")

            if not data:
                break

            for item in data:
                all_data.append({
                    "title": item.get("title") or item.get("name"),
                    "type": content_type,
                    "release_year": year,
                    "overview": item.get("overview"),
                    "rating": item.get("vote_average", 0),
                    "language": item.get("original_language")
                })

            time.sleep(0.2)


# =========================
# RUN
# =========================

fetch_data("movie", "Movie")
fetch_data("tv", "TV Show")

df = pd.DataFrame(all_data)

print("\nTOTAL RECORDS:", len(df))

# save
output_path = Path("data/processed/netflix_extended_2020_2026.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("DONE SAVED ✔")