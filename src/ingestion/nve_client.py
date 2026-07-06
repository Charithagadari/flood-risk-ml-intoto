from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("NVE_API_KEY")

BASE_URL = "https://hydapi.nve.no/api/v1"

STATION_ID = "1.15.0"

PARAMETERS = {
    "water_level": "1000",
    "reservoir_volume": "1004",
}

RESOLUTION_TIME = 60


# ============================================================
# API VALIDATION
# ============================================================

def validate_api_key() -> None:
    """
    Ensure the NVE API key is available.
    """

    if not API_KEY:
        raise ValueError(
            "NVE_API_KEY was not found.\n"
            "Add it to your .env file:\n\n"
            "NVE_API_KEY=your_api_key"
        )


# ============================================================
# FETCH ONE PARAMETER
# ============================================================

def fetch_observations(
    station_id: str,
    parameter: str,
    resolution_time: int,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Download observations from NVE HydAPI.
    """

    validate_api_key()

    endpoint = f"{BASE_URL}/Observations"

    headers = {
        "Accept": "application/json",
        "X-API-Key": API_KEY,
    }

    params = {
        "StationId": station_id,
        "Parameter": parameter,
        "ResolutionTime": resolution_time,
        "ReferenceTime": (
            f"{start_date}/{end_date}"
        ),
    }

    response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=60,
    )

    print(
        "Request URL:",
        response.url,
    )

    print(
        "Status:",
        response.status_code,
    )

    if response.status_code != 200:

        print(
            response.text[:2000]
        )

        response.raise_for_status()

    return response.json()


# ============================================================
# PARSE NVE RESPONSE
# ============================================================

def parse_observations_to_dataframe(
    data: dict,
) -> pd.DataFrame:
    """
    Convert NVE HydAPI response into a dataframe.
    """

    rows = []

    if isinstance(data, dict):

        items = data.get(
            "data",
            data.get(
                "Data",
                [],
            ),
        )

    else:

        items = data

    for item in items:

        station_id = (
            item.get("stationId")
            or item.get("StationId")
        )

        parameter = (
            item.get("parameter")
            or item.get("Parameter")
        )

        resolution_time = (
            item.get("resolutionTime")
            or item.get("ResolutionTime")
        )

        observations = (
            item.get("observations")
            or item.get("Observations")
            or item.get("values")
            or item.get("Values")
            or []
        )

        for observation in observations:

            timestamp = (
                observation.get("time")
                or observation.get("Time")
                or observation.get(
                    "referenceTime"
                )
                or observation.get(
                    "ReferenceTime"
                )
            )

            # Do not use:
            #
            # observation.get("value")
            # or observation.get("Value")
            #
            # because value=0 is a valid observation.

            value = observation.get("value")

            if value is None:

                value = observation.get(
                    "Value"
                )

            quality = observation.get(
                "quality"
            )

            if quality is None:

                quality = observation.get(
                    "Quality"
                )

            rows.append(
                {
                    "station_id": station_id,
                    "parameter": parameter,
                    "resolution_time": (
                        resolution_time
                    ),
                    "timestamp": timestamp,
                    "value": value,
                    "quality": quality,
                }
            )

    data_frame = pd.DataFrame(rows)

    if data_frame.empty:

        return data_frame

    data_frame["timestamp"] = (
        pd.to_datetime(
            data_frame["timestamp"],
            errors="coerce",
            utc=True,
        )
    )

    data_frame["value"] = (
        pd.to_numeric(
            data_frame["value"],
            errors="coerce",
        )
    )

    data_frame = (
        data_frame
        .dropna(
            subset=[
                "timestamp",
                "value",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return data_frame


# ============================================================
# FETCH LATEST NVE DATA
# ============================================================
# ============================================================
# FETCH WATER-LEVEL OBSERVATIONS FOR A TIME RANGE
# ============================================================

def fetch_water_level_range(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
    station_id: str = STATION_ID,
) -> pd.DataFrame:
    """
    Fetch hourly water-level observations from NVE
    for a requested UTC time range.

    Returns:
        timestamp
        actual_water_level
    """

    start_timestamp = pd.Timestamp(
        start_datetime
    )

    end_timestamp = pd.Timestamp(
        end_datetime
    )

    # --------------------------------------------------------
    # Normalize to UTC
    # --------------------------------------------------------

    if start_timestamp.tzinfo is None:

        start_timestamp = (
            start_timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        start_timestamp = (
            start_timestamp.tz_convert(
                "UTC"
            )
        )

    if end_timestamp.tzinfo is None:

        end_timestamp = (
            end_timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        end_timestamp = (
            end_timestamp.tz_convert(
                "UTC"
            )
        )

    if end_timestamp < start_timestamp:

        raise ValueError(
            "end_datetime must be greater than "
            "or equal to start_datetime."
        )

    # --------------------------------------------------------
    # HydAPI request uses date intervals in this project.
    #
    # Add one day around the query boundaries so that
    # timestamp-level filtering is done locally.
    # --------------------------------------------------------

    query_start_date = (
        start_timestamp
        - pd.Timedelta(days=1)
    ).date().isoformat()

    query_end_date = (
        end_timestamp
        + pd.Timedelta(days=1)
    ).date().isoformat()

    print()

    print(
        "Fetching actual water levels:"
    )

    print(
        "Requested range:",
        start_timestamp,
        "to",
        end_timestamp,
    )

    response_data = fetch_observations(
        station_id=station_id,
        parameter=PARAMETERS[
            "water_level"
        ],
        resolution_time=RESOLUTION_TIME,
        start_date=query_start_date,
        end_date=query_end_date,
    )

    water_level_data = (
        parse_observations_to_dataframe(
            response_data
        )
    )

    if water_level_data.empty:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "actual_water_level",
            ]
        )

    water_level_data = (
        water_level_data[
            [
                "timestamp",
                "value",
            ]
        ]
        .rename(
            columns={
                "value": (
                    "actual_water_level"
                )
            }
        )
        .copy()
    )

    # --------------------------------------------------------
    # Exact local UTC filtering
    # --------------------------------------------------------

    water_level_data = (
        water_level_data[
            (
                water_level_data[
                    "timestamp"
                ]
                >= start_timestamp
            )
            &
            (
                water_level_data[
                    "timestamp"
                ]
                <= end_timestamp
            )
        ]
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        "Actual observations received:",
        len(water_level_data),
    )

    if not water_level_data.empty:

        print(
            "Actual data range:",
            water_level_data[
                "timestamp"
            ].min(),
            "to",
            water_level_data[
                "timestamp"
            ].max(),
        )

    return water_level_data

def fetch_latest_nve_data(
    days: int = 14,
    station_id: str = STATION_ID,
) -> pd.DataFrame:
    """
    Fetch recent water-level and reservoir-volume
    observations required for live forecasting.

    Returns:
        timestamp
        reservoir_volume
        water_level
    """

    if days <= 0:

        raise ValueError(
            "days must be greater than zero."
        )

    end_datetime = datetime.now(
        timezone.utc
    )

    start_datetime = (
        end_datetime
        - timedelta(days=days)
    )

    start_date = (
        start_datetime
        .date()
        .isoformat()
    )

    query_end_datetime = (
        end_datetime
        + timedelta(days=1)
    )

    end_date = (
        query_end_datetime
        .date()
        .isoformat()
    )
    print("=" * 70)

    print(
        "FETCHING LIVE NVE OBSERVATIONS"
    )

    print("=" * 70)

    print(
        "Station ID :",
        station_id,
    )

    print(
        "Start date :",
        start_date,
    )

    print(
        "End date   :",
        end_date,
    )

    all_dataframes = []

    for (
        feature_name,
        parameter_id,
    ) in PARAMETERS.items():

        print()

        print(
            f"Fetching {feature_name}"
        )

        print(
            "Parameter ID:",
            parameter_id,
        )

        response_data = fetch_observations(
            station_id=station_id,
            parameter=parameter_id,
            resolution_time=(
                RESOLUTION_TIME
            ),
            start_date=start_date,
            end_date=end_date,
        )

        parameter_data = (
            parse_observations_to_dataframe(
                response_data
            )
        )

        if parameter_data.empty:

            raise ValueError(
                "No observations returned for "
                f"{feature_name} "
                f"(parameter={parameter_id})."
            )

        print(
            "Rows received:",
            len(parameter_data),
        )

        print(
            "Parameter range:",
            parameter_data[
                "timestamp"
            ].min(),
            "to",
            parameter_data[
                "timestamp"
            ].max(),
        )

        parameter_data[
            "feature_name"
        ] = feature_name

        all_dataframes.append(
            parameter_data
        )

    # ========================================================
    # COMBINE PARAMETERS
    # ========================================================

    long_data = pd.concat(
        all_dataframes,
        ignore_index=True,
    )

    # ========================================================
    # LONG -> WIDE
    # ========================================================

    wide_data = (
        long_data
        .pivot_table(
            index="timestamp",
            columns="feature_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("timestamp")
    )

    wide_data.columns.name = None

    # ========================================================
    # VALIDATE EXPECTED FEATURES
    # ========================================================

    required_columns = [
        "timestamp",
        "water_level",
        "reservoir_volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in wide_data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Live NVE dataframe is missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    # ========================================================
    # NUMERIC CLEANING
    # ========================================================

    numeric_columns = [
        "water_level",
        "reservoir_volume",
    ]

    wide_data[numeric_columns] = (
        wide_data[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # Use only observations where both model inputs
    # can eventually be reconstructed.

    wide_data[numeric_columns] = (
        wide_data[numeric_columns]
        .ffill()
        .bfill()
    )

    wide_data = (
        wide_data
        .dropna(
            subset=required_columns
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if wide_data.empty:

        raise ValueError(
            "Live NVE dataframe is empty "
            "after cleaning."
        )

    print()

    print("=" * 70)

    print(
        "LIVE NVE DATA READY"
    )

    print("=" * 70)

    print(
        "Rows:",
        len(wide_data),
    )

    print(
        "Columns:",
        wide_data.columns.tolist(),
    )

    print(
        "Data range:",
        wide_data["timestamp"].min(),
        "to",
        wide_data["timestamp"].max(),
    )

    print()

    print(
        "Latest observations:"
    )

    print(
        wide_data.tail().to_string(
            index=False
        )
    )

    return wide_data


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    live_data = fetch_latest_nve_data(
        days=14
    )

    print()

    print(
        "Final dataframe shape:",
        live_data.shape,
    )