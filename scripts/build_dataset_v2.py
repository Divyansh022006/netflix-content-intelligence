import requests
import pandas as pd
import time

API_KEY = "479fbff40534ba023225868357a44760"

MOVIE_URL = "https://api.themoviedb.org/3/discover/movie"
TV_URL = "https://api.themoviedb.org/3/discover/tv"

def fetch_movies():
    data = []

    for page in range(1, 15):  # more pages = more data
        res = requests.get(MOVIE_URL, params={
            "api_key": API_KEY,
            "sort_by": "popularity.desc",
            "page": page
        }).json()

        results = res.get("results", [])

        print("Movies page", page, ":", len(results))

        for m in results:
            data.append({
                "title": m.get("title"),
                "type": "Movie",
                "release_year": (m.get("release_date") or "2023")[:4],
                "rating": m.get("vote_average", 0),
                "popularity": m.get("popularity", 0),
                "overview": m.get("overview", "")
            })

        time.sleep(0.2)

    return data


def fetch_tv():
    data = []

    for page in range(1, 15):
        res = requests.get(TV_URL, params={
            "api_key": API_KEY,
            "sort_by": "popularity.desc",
            "page": page
        }).json()

        results = res.get("results", [])

        print("TV page", page, ":", len(results))

        for m in results:
            data.append({
                "title": m.get("name"),
                "type": "TV Show",
                "release_year": (m.get("first_air_date") or "2023")[:4],
                "rating": m.get("vote_average", 0),
                "popularity": m.get("popularity", 0),
                "overview": m.get("overview", "")
            })

        time.sleep(0.2)

    return data


def build():
    movies = fetch_movies()
    tv = fetch_tv()

    print("\nMOVIES:", len(movies))
    print("TV:", len(tv))

    df = pd.DataFrame(movies + tv)

    # 🔥 CRITICAL CHECK
    if df.empty:
        raise Exception("Dataset is empty — API failed or key invalid")

    path = "data/processed/netflix_extended_2020_2026.csv"
    df.to_csv(path, index=False)

    print("\nSAVED:", path)
    print("TOTAL ROWS:", len(df))


if __name__ == "__main__":
    build()