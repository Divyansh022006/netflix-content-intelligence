import requests
import pandas as pd
import time
import random
from pathlib import Path

# =========================
# CONFIG
# =========================

API_KEY = "479fbff40534ba023225868357a44760"
BASE_URL = "https://api.themoviedb.org/3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "netflix_extended_2020_2026.csv"

all_data = []

# =========================
# SAFE FETCH FUNCTION
# =========================

def fetch(page, year, media_type="movie"):
    url = f"{BASE_URL}/discover/{media_type}"

    params = {
        "api_key": API_KEY,
        "page": page,
        "primary_release_year" if media_type == "movie" else "first_air_date_year": year,
        "sort_by": "popularity.desc",
    }

    try:
        r = requests.get(url, params=params, timeout=10)

        if r.status_code != 200:
            print("API Error:", r.status_code)
            return []

        data = r.json()
        return data.get("results", [])

    except Exception as e:
        print("Request failed:", e)
        return []

# =========================
# BUILD DATASET
# =========================

for year in range(2020, 2027):

    print(f"\nFetching year: {year}")

    # 🔥 randomization = avoids flat line problem
    movie_pages = random.randint(3, 7)
    tv_pages = random.randint(2, 5)

    for media, max_pages in [("movie", movie_pages), ("tv", tv_pages)]:

        for page in range(1, max_pages + 1):

            results = fetch(page, year, media)

            print(f"{media} {year} page {page}: {len(results)}")

            time.sleep(0.2)  # avoid rate limit

            for item in results:

                title = item.get("title") or item.get("name")

                if not title:
                    continue

                all_data.append({
                    "title": title,
                    "type": "Movie" if media == "movie" else "TV Show",
                    "release_year": year,
                    "overview": item.get("overview", ""),
                    "rating": item.get("vote_average", 0),
                    "vote_count": item.get("vote_count", 0),
                    "popularity": item.get("popularity", 0),
                    "language": item.get("original_language", "unknown"),
                    "poster_path": item.get("poster_path", "")
                })

# =========================
# SAVE DATASET
# =========================

df = pd.DataFrame(all_data)

if df.empty:
    raise Exception("Dataset is empty — check API or key")

df.drop_duplicates(subset=["title", "release_year"], inplace=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nDONE!")
print("Total records:", len(df))
print("Saved to:", OUTPUT_PATH)