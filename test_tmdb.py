import requests

API_KEY = "479fbff40534ba023225868357a44760"

url = "https://api.themoviedb.org/3/movie/popular"

res = requests.get(url, params={"api_key": API_KEY})

print("STATUS:", res.status_code)
print(res.json()["results"][0]["title"])