import requests
import pandas as pd
from tqdm import tqdm
import time

API_KEY = "479fbff40534ba023225868357a44760"
BASE_URL = "https://api.themoviedb.org/3"

data = []

# ----------------------------
# GET GENRE MAP (IMPORTANT)
# ----------------------------
def get_genres():
    url = f"{BASE_URL}/genre/movie/list"
    res = requests.get(url, params={"api_key": API_KEY}).json()

    genre_map = {}
    for g in res.get("genres", []):
        genre_map[g["id"]] = g["name"]

    return genre_map

genre_map = get_genres()

# ----------------------------
# FETCH MOVIES
# ----------------------------
def fetch_movies():
    for year in range(2020, 2027):
        print(f"\nFetching Movies Year: {year}")

        for page in range(1, 11):
            url = f"{BASE_URL}/discover/movie"
            params = {
                "api_key": API_KEY,
                "sort_by": "popularity.desc",
                "primary_release_year": year,
                "page": page
            }

            res = requests.get(url, params=params).json()

            for m in res.get("results", []):
                genres = [genre_map.get(i, "") for i in m.get("genre_ids", [])]

                data.append({
                    "title": m.get("title"),
                    "type": "Movie",
                    "release_year": year,
                    "rating": m.get("vote_average", 0),
                    "popularity": m.get("popularity", 0),
                    "language": m.get("original_language"),
                    "genre": ", ".join(genres),
                    "overview": m.get("overview", "")
                })

            time.sleep(0.2)

# ----------------------------
# FETCH TV SHOWS
# ----------------------------
def fetch_tv():
    for year in range(2020, 2027):
        print(f"\nFetching TV Year: {year}")

        for page in range(1, 11):
            url = f"{BASE_URL}/discover/tv"
            params = {
                "api_key": API_KEY,
                "sort_by": "popularity.desc",
                "first_air_date_year": year,
                "page": page
            }

            res = requests.get(url, params=params).json()

            for m in res.get("results", []):
                data.append({
                    "title": m.get("name"),
                    "type": "TV Show",
                    "release_year": year,
                    "rating": m.get("vote_average", 0),
                    "popularity": m.get("popularity", 0),
                    "language": m.get("original_language"),
                    "genre": "",
                    "overview": m.get("overview", "")
                })

            time.sleep(0.2)

# ----------------------------
# BUILD DATASET
# ----------------------------
def build():
    fetch_movies()
    fetch_tv()

    df = pd.DataFrame(data)

    # cleanup
    df = df.dropna(subset=["title"])
    df["release_year"] = df["release_year"].astype(int)

    # save
    output_path = "data/processed/netflix_extended_2020_2026.csv"
    df.to_csv(output_path, index=False)

    print("\nDONE ✅")
    print("Total rows:", len(df))
    print("Saved to:", output_path)

if __name__ == "__main__":
    build()