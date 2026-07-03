import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVE_API_KEY")
BASE_URL = "https://hydapi.nve.no/api/v1"

headers = {
    "Accept": "application/json",
    "X-API-Key": API_KEY,
}

def get_json(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, headers=headers, params=params, timeout=60)

    if response.status_code != 200:
        print("URL:", response.url)
        print("Status:", response.status_code)
        print(response.text[:1000])
        response.raise_for_status()

    return response.json()

# Get station/series metadata
data = get_json("Stations")

print(type(data))

# HydAPI may return either a list or an object with data inside.
if isinstance(data, dict):
    print(data.keys())

# Save raw response preview
print(str(data)[:2000])