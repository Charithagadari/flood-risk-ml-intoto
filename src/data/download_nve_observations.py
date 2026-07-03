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

def fetch_observations(
    station_id: str,
    parameter: str,
    resolution_time: int,
    start_date: str,
    end_date: str,
):
    """
    Downloads observations from NVE HydAPI.

    station_id: NVE station id, for example "2.11.0" or similar
    parameter: parameter id/name from HydAPI metadata
    resolution_time: time resolution from metadata
    start_date/end_date: YYYY-MM-DD
    """

    endpoint = f"{BASE_URL}/Observations"

    params = {
        "StationId": station_id,
        "Parameter": parameter,
        "ResolutionTime": resolution_time,
        "ReferenceTime": f"{start_date}/{end_date}",
    }

    response = requests.get(endpoint, headers=headers, params=params, timeout=60)

    print("Request URL:", response.url)
    print("Status:", response.status_code)

    if response.status_code != 200:
        print(response.text[:2000])
        response.raise_for_status()

    return response.json()

def parse_observations_to_dataframe(data):
    rows = []

    # HydAPI response structure can contain nested series/observations.
    # This parser is defensive so you can inspect and adapt if needed.
    if isinstance(data, dict):
        items = data.get("data", data.get("Data", []))
    else:
        items = data

    for item in items:
        station_id = item.get("stationId") or item.get("StationId")
        parameter = item.get("parameter") or item.get("Parameter")
        resolution_time = item.get("resolutionTime") or item.get("ResolutionTime")

        observations = (
            item.get("observations")
            or item.get("Observations")
            or item.get("values")
            or item.get("Values")
            or []
        )

        for obs in observations:
            rows.append({
                "station_id": station_id,
                "parameter": parameter,
                "resolution_time": resolution_time,
                "timestamp": obs.get("time") or obs.get("Time") or obs.get("referenceTime") or obs.get("ReferenceTime"),
                "value": obs.get("value") or obs.get("Value"),
                "quality": obs.get("quality") or obs.get("Quality"),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.sort_values("timestamp")

    return df

if __name__ == "__main__":

    STATION_ID = "1.15.0"

    PARAMETERS = {
        "water_level": "1000",
        "reservoir_volume": "1004",
    }

    RESOLUTION_TIME = 60

    START_DATE = "2020-01-01"
    END_DATE = "2026-01-01"

    all_dfs = []

    for feature_name, parameter_id in PARAMETERS.items():
        print(f"\nDownloading {feature_name} | parameter={parameter_id}")

        data = fetch_observations(
            station_id=STATION_ID,
            parameter=parameter_id,
            resolution_time=RESOLUTION_TIME,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        temp_df = parse_observations_to_dataframe(data)

        if temp_df.empty:
            print(f"No data found for {feature_name}")
            continue

        temp_df["feature_name"] = feature_name
        all_dfs.append(temp_df)

    if not all_dfs:
        raise ValueError("No observations downloaded. Check station, parameters, dates, and API key.")

    long_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs("raw", exist_ok=True)
    long_df.to_csv("/Users/charithagadari/Library/Mobile Documents/com~apple~CloudDocs/flood-risk-ml-intoto/src/data/raw/nve_observations_long.csv", index=False)

    wide_df = (
        long_df
        .pivot_table(
            index="timestamp",
            columns="feature_name",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
        .sort_values("timestamp")
    )

    wide_df.to_csv("/Users/charithagadari/Library/Mobile Documents/com~apple~CloudDocs/flood-risk-ml-intoto/src/data/raw/nve_observations_wide.csv", index=False)

    print("\nSaved:")
    print("/Users/charithagadari/Library/Mobile Documents/com~apple~CloudDocs/flood-risk-ml-intoto/src/data/raw/nve_observations_long.csv")
    print("/Users/charithagadari/Library/Mobile Documents/com~apple~CloudDocs/flood-risk-ml-intoto/src/data/raw/nve_observations_wide.csv")
    print(wide_df.head())