import pandas as pd
import requests
import time

TMDB_API_KEY = "479fbff40534ba023225868357a44760"

INPUT_PATH = "data/processed/netflix_text.csv"
OUTPUT_PATH = "data/processed/netflix_with_posters.csv"

df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

poster_urls = []

for i, row in df.iterrows():
    title = str(row["title"]).split("(")[0].strip()

    try:
        url = "https://api.themoviedb.org/3/search/multi"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "include_adult": False
        }

        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        results = data.get("results", [])

        poster = None

        if results:
            best = max(results, key=lambda x: x.get("popularity", 0))
            if best.get("poster_path"):
                poster = "https://image.tmdb.org/t/p/w500" + best["poster_path"]

        if not poster:
            poster = "https://placehold.co/300x450?text=No+Poster"

        poster_urls.append(poster)

        print(f"{i+1}/{len(df)} OK: {title}")

        time.sleep(0.25)  # avoids API rate limit

    except Exception as e:
        print("Error:", e)
        poster_urls.append("https://placehold.co/300x450?text=Error")

df["poster_url"] = poster_urls

df.to_csv(OUTPUT_PATH, index=False)

print("DONE → Saved dataset with posters")