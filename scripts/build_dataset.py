import requests
import pandas as pd

API_KEY = "479fbff40534ba023225868357a44760"

url = "https://api.themoviedb.org/3/discover/movie"

all_movies = []

for year in range(2020, 2027):

    print(f"\nFetching year: {year}")

    for page in range(1, 3):

        params = {
            "api_key": API_KEY,
            "primary_release_year": year,
            "page": page,
            "language": "en-US"
        }

        response = requests.get(url, params=params)

        print("Status Code:", response.status_code)

        data = response.json()

        results = data.get("results", [])

        print("Results found:", len(results))

        for movie in results:
            all_movies.append({
                "title": movie.get("title"),
                "release_year": year,
                "rating": movie.get("vote_average"),
                "popularity": movie.get("popularity")
            })

df = pd.DataFrame(all_movies)

print("\nTOTAL MOVIES:", len(df))

df.to_csv("data/processed/netflix_extended_2020_2026.csv", index=False)

print("DONE")