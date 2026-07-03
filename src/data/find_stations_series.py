import os
import json
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

    print("Request URL:", response.url)
    print("Status:", response.status_code)

    if response.status_code != 200:
        print(response.text[:2000])
        response.raise_for_status()

    return response.json()

station_id = "1.15.0"

# Try getting time-series metadata for this station
data = get_json("Series", params={"StationId": station_id})

print("\nRESPONSE KEYS:")
print(data.keys())

print("\nRAW PREVIEW:")
print(str(data)[:3000])

os.makedirs("data/raw", exist_ok=True)

with open("data/raw/nve_timeseries_1_15_0.json", "w") as f:
    json.dump(data, f, indent=2)

items = data.get("data", [])

rows = []
for item in items:
    rows.append({
        "stationId": item.get("stationId"),
        "stationName": item.get("stationName"),
        "parameter": item.get("parameter"),
        "parameterName": item.get("parameterName"),
        "resolutionTime": item.get("resolutionTime"),
        "timeSeriesId": item.get("timeSeriesId"),
        "unit": item.get("unit"),
        "fromDate": item.get("fromDate"),
        "toDate": item.get("toDate"),
    })

df = pd.DataFrame(rows)
print("\nTIME SERIES TABLE:")
print(df.head(50).to_string(index=False))

df.to_csv("data/raw/nve_timeseries_1_15_0.csv", index=False)
print("\nSaved:")
print("data/raw/nve_timeseries_1_15_0.json")
print("data/raw/nve_timeseries_1_15_0.csv")