from __future__ import annotations

import os

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.ingestion.nve_client import (
    fetch_latest_nve_data,
)


# ============================================================
# 1. FORECAST CONFIGURATION
# ============================================================

DATE_COL = "timestamp"

TARGET_COL = "water_level"

RESERVOIR_COL = "reservoir_volume"

STATION_ID = "1.15.0"

HORIZONS = [
    6,
    24,
    72,
]


MODEL_NAMES = {
    6: "flood_forecast_6h",
    24: "flood_forecast_24h",
    72: "flood_forecast_72h",
}


# ============================================================
# 2. FEATURE ENGINEERING CONFIGURATION
# ============================================================

# IMPORTANT:
# These settings must stay consistent with the training notebook.

LAGS = [
    1,
    3,
    6,
    12,
    24,
    48,
    72,
]


ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    72,
]


CHANGE_SLOPE_LAGS = [
    1,
    3,
    6,
    12,
    24,
    48,
    72,
]


# ============================================================
# 3. MLFLOW CONFIGURATION
# ============================================================

MLFLOW_HOME = (
    Path.home()
    / ".mlflow"
    / "flood-risk"
)


MLFLOW_DB = (
    MLFLOW_HOME
    / "mlflow.db"
)


import os

import mlflow


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


MLFLOW_EXPERIMENT_NAME = (
    "flood-forecasting-portable"
)


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


print(
    "MLflow tracking URI:"
)

print(
    mlflow.get_tracking_uri()
)


experiment = (
    mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )
)


if experiment is None:

    raise RuntimeError(
        "Portable MLflow experiment "
        "was not found."
    )


print()

print(
    "Experiment:"
)

print(
    experiment.name
)


print()

print(
    "Experiment ID:"
)

print(
    experiment.experiment_id
)


print()

print(
    "Artifact location:"
)

print(
    experiment.artifact_location
)


if (
    str(
        experiment.artifact_location
    )
    .startswith(
        "file:///Users/"
    )
):

    raise RuntimeError(
        "Experiment still uses a "
        "machine-specific artifact path."
    )


print()

print(
    "Portable MLflow configuration PASSED."
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "predictions"
)

PREDICTION_LOG_PATH = (
    PREDICTION_DIR
    / "prediction_log.csv"
)

# ============================================================
# 4. CONFIGURE MLFLOW
# ============================================================

def configure_mlflow() -> MlflowClient:
    """
    Configure MLflow for either:

    1. Local development using the SQLite backend.
    2. Docker/remote serving using MLFLOW_TRACKING_URI.
    """

    uses_local_sqlite = (
        MLFLOW_TRACKING_URI.startswith(
            "sqlite:///"
        )
    )

    if (
        uses_local_sqlite
        and not MLFLOW_DB.exists()
    ):

        raise FileNotFoundError(
            "MLflow database was not found.\n\n"
            f"Expected database:\n"
            f"{MLFLOW_DB}\n\n"
            "Check the MLFLOW_TRACKING_URI used "
            "in the forecasting notebook."
        )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    print(
        "MLflow tracking URI:"
    )

    print(
        mlflow.get_tracking_uri()
    )

    print()

    if uses_local_sqlite:

        print(
            "MLflow DB exists:",
            MLFLOW_DB.exists(),
        )

        if MLFLOW_DB.exists():

            print(
                "MLflow DB path   :",
                MLFLOW_DB.resolve(),
            )

            print(
                "MLflow DB size   :",
                MLFLOW_DB.stat().st_size,
                "bytes",
            )

    else:

        print(
            "Using remote MLflow "
            "Tracking Server."
        )

    return client


# ============================================================
# 5. VALIDATE LIVE NVE DATA
# ============================================================

def validate_live_dataframe(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and clean the dataframe returned
    by the NVE ingestion module.
    """

    required_columns = [
        DATE_COL,
        TARGET_COL,
        RESERVOIR_COL,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "NVE dataframe is missing required columns:\n"
            f"{missing_columns}\n\n"
            f"Available columns:\n"
            f"{data.columns.tolist()}"
        )

    data = data.copy()

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    data[DATE_COL] = pd.to_datetime(
        data[DATE_COL],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    data[TARGET_COL] = pd.to_numeric(
        data[TARGET_COL],
        errors="coerce",
    )

    data[RESERVOIR_COL] = pd.to_numeric(
        data[RESERVOIR_COL],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            DATE_COL,
        ]
    )

    # --------------------------------------------------------
    # Sort and remove duplicated timestamps
    # --------------------------------------------------------

    data = (
        data
        .sort_values(DATE_COL)
        .drop_duplicates(
            subset=[DATE_COL],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Match training missing-value treatment
    # --------------------------------------------------------

    numeric_columns = [
        TARGET_COL,
        RESERVOIR_COL,
    ]

    data[numeric_columns] = (
        data[numeric_columns]
        .ffill()
        .bfill()
    )

    data = data.dropna(
        subset=required_columns
    )

    if data.empty:

        raise ValueError(
            "Live NVE dataframe is empty "
            "after validation."
        )

    return data


# ============================================================
# 6. DETECT TIME STEP
# ============================================================

def detect_time_step_hours(
    data: pd.DataFrame,
) -> float:
    """
    Detect median sampling interval in hours.
    """

    time_differences = (
        data[DATE_COL]
        .sort_values()
        .diff()
        .dropna()
    )

    if time_differences.empty:

        raise ValueError(
            "At least two observations are required "
            "to detect the sampling interval."
        )

    time_step_hours = (
        time_differences
        .median()
        .total_seconds()
        / 3600
    )

    if (
        not np.isfinite(time_step_hours)
        or time_step_hours <= 0
    ):

        raise ValueError(
            "Could not detect a valid time step."
        )

    return float(
        time_step_hours
    )


# ============================================================
# 7. HOURS TO ROW PERIODS
# ============================================================

def hours_to_periods(
    hours: int,
    time_step_hours: float,
) -> int:
    """
    Convert real hours to dataframe row periods.

    Example:
        6 hours / 1-hour sampling = 6 rows
    """

    return max(
        int(
            round(
                hours
                / time_step_hours
            )
        ),
        1,
    )


# ============================================================
# 8. TIME FEATURES
# ============================================================

def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add temporal and cyclic features.
    """

    data = data.copy()

    date_time = pd.to_datetime(
        data[DATE_COL],
        utc=True,
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["hour"] = (
        date_time.dt.hour
    )

    data["dayofyear"] = (
        date_time.dt.dayofyear
    )

    data["month"] = (
        date_time.dt.month
    )

    data["dayofweek"] = (
        date_time.dt.dayofweek
    )

    data["is_weekend"] = (
        data["dayofweek"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Hour cyclic encoding
    # --------------------------------------------------------

    data["sin_hour"] = np.sin(
        2
        * np.pi
        * data["hour"]
        / 24
    )

    data["cos_hour"] = np.cos(
        2
        * np.pi
        * data["hour"]
        / 24
    )

    # --------------------------------------------------------
    # Day-of-year cyclic encoding
    # --------------------------------------------------------

    data["sin_dayofyear"] = np.sin(
        2
        * np.pi
        * data["dayofyear"]
        / 365
    )

    data["cos_dayofyear"] = np.cos(
        2
        * np.pi
        * data["dayofyear"]
        / 365
    )

    # --------------------------------------------------------
    # Month cyclic encoding
    # --------------------------------------------------------

    data["sin_month"] = np.sin(
        2
        * np.pi
        * data["month"]
        / 12
    )

    data["cos_month"] = np.cos(
        2
        * np.pi
        * data["month"]
        / 12
    )

    return data


# ============================================================
# 9. LAG / ROLLING / CHANGE / SLOPE FEATURES
# ============================================================

def add_lag_rolling_change_features(
    data: pd.DataFrame,
    base_columns: list[str],
    time_step_hours: float,
) -> pd.DataFrame:
    """
    Create temporal features for each base variable.

    Features:
    - lag
    - rolling mean
    - rolling standard deviation
    - rolling minimum
    - rolling maximum
    - change
    - slope
    """

    data = data.copy()

    for base_column in base_columns:

        if base_column not in data.columns:

            print(
                "Skipping missing base column:",
                base_column,
            )

            continue

        print(
            "Engineering features for:",
            base_column,
        )

        # ====================================================
        # Lag features
        # ====================================================

        for lag_hours in LAGS:

            periods = hours_to_periods(
                hours=lag_hours,
                time_step_hours=time_step_hours,
            )

            lag_column = (
                f"{base_column}"
                f"_lag_{lag_hours}h"
            )

            data[lag_column] = (
                data[base_column]
                .shift(periods)
            )

        # ====================================================
        # Rolling features
        # ====================================================

        for window_hours in ROLLING_WINDOWS:

            periods = hours_to_periods(
                hours=window_hours,
                time_step_hours=time_step_hours,
            )

            rolling_values = (
                data[base_column]
                .rolling(periods)
            )

            data[
                f"{base_column}"
                f"_roll_mean_{window_hours}h"
            ] = (
                rolling_values.mean()
            )

            data[
                f"{base_column}"
                f"_roll_std_{window_hours}h"
            ] = (
                rolling_values.std()
            )

            data[
                f"{base_column}"
                f"_roll_min_{window_hours}h"
            ] = (
                rolling_values.min()
            )

            data[
                f"{base_column}"
                f"_roll_max_{window_hours}h"
            ] = (
                rolling_values.max()
            )

        # ====================================================
        # Change and slope features
        # ====================================================

        for lag_hours in CHANGE_SLOPE_LAGS:

            periods = hours_to_periods(
                hours=lag_hours,
                time_step_hours=time_step_hours,
            )

            lag_column = (
                f"{base_column}"
                f"_lag_{lag_hours}h"
            )

            change_column = (
                f"{base_column}"
                f"_change_{lag_hours}h"
            )

            slope_column = (
                f"{base_column}"
                f"_slope_{lag_hours}h"
            )

            if lag_column not in data.columns:

                data[lag_column] = (
                    data[base_column]
                    .shift(periods)
                )

            data[change_column] = (
                data[base_column]
                - data[lag_column]
            )

            data[slope_column] = (
                data[change_column]
                / lag_hours
            )

    return data


# ============================================================
# 10. BUILD LIVE INFERENCE FEATURES
# ============================================================

def build_inference_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Build production inference features.

    Future targets are deliberately NOT generated.
    """

    data = validate_live_dataframe(
        data
    )

    time_step_hours = (
        detect_time_step_hours(
            data
        )
    )

    base_columns = [
        TARGET_COL,
        RESERVOIR_COL,
    ]

    print(
        "Base numeric columns:",
        base_columns,
    )

    feature_data = add_time_features(
        data
    )

    feature_data = (
        add_lag_rolling_change_features(
            data=feature_data,
            base_columns=base_columns,
            time_step_hours=time_step_hours,
        )
    )

    return (
        feature_data,
        time_step_hours,
    )


# ============================================================
# 11. READ MODEL SIGNATURE
# ============================================================

def get_required_model_columns(
    model: Any,
) -> list[str]:
    """
    Get the exact expected model input columns
    from the MLflow model signature.
    """

    signature = (
        model.metadata.signature
    )

    if signature is None:

        raise ValueError(
            "The MLflow model does not contain "
            "an input signature."
        )

    if signature.inputs is None:

        raise ValueError(
            "The MLflow model signature "
            "contains no input schema."
        )

    required_columns = [
        input_item.name
        for input_item
        in signature.inputs.inputs
        if input_item.name is not None
    ]

    if not required_columns:

        raise ValueError(
            "No named feature columns were found "
            "in the MLflow model signature."
        )

    return required_columns


# ============================================================
# 12. LOAD CHAMPION MODEL
# ============================================================

def load_champion_model(
    client: MlflowClient,
    horizon: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Load the MLflow champion model for one horizon.
    """

    if horizon not in MODEL_NAMES:

        raise ValueError(
            f"Unsupported forecast horizon: "
            f"{horizon}"
        )

    registered_model_name = (
        MODEL_NAMES[horizon]
    )

    model_uri = (
        f"models:/"
        f"{registered_model_name}"
        f"@champion"
    )

    print()

    print(
        f"Loading {horizon}h champion:"
    )

    print(
        model_uri
    )

    # --------------------------------------------------------
    # Resolve champion alias
    # --------------------------------------------------------

    model_version = (
        client.get_model_version_by_alias(
            name=registered_model_name,
            alias="champion",
        )
    )

    # --------------------------------------------------------
    # Load registered model
    # --------------------------------------------------------

    model = (
        mlflow.pyfunc.load_model(
            model_uri
        )
    )

    tags = (
        model_version.tags
        or {}
    )

    source_model = tags.get(
        "source_model",
        "",
    )

    # --------------------------------------------------------
    # Detect delta model
    # --------------------------------------------------------

    delta_tag = tags.get(
        "is_delta_model"
    )

    if delta_tag is not None:

        is_delta_model = (
            str(delta_tag)
            .strip()
            .lower()
            == "true"
        )

    else:

        is_delta_model = (
            "delta"
            in str(
                source_model
            ).lower()
        )

        print(
            "WARNING: is_delta_model tag missing."
        )

        print(
            "Inferring model type from source_model."
        )

    metadata = {
        "horizon_hours": horizon,
        "registered_model": (
            registered_model_name
        ),
        "version": str(
            model_version.version
        ),
        "run_id": (
            model_version.run_id
        ),
        "source_model": source_model,
        "is_delta_model": (
            is_delta_model
        ),
        "model_uri": model_uri,
        "tags": tags,
    }

    print(
        "Version       :",
        metadata["version"],
    )

    print(
        "Source model  :",
        metadata["source_model"],
    )

    print(
        "Delta model   :",
        metadata["is_delta_model"],
    )

    return (
        model,
        metadata,
    )


# ============================================================
# 13. PREDICT ONE HORIZON
# ============================================================

def predict_horizon(
    model: Any,
    metadata: dict[str, Any],
    feature_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Predict one forecast horizon.

    Correctly handles absolute-target
    and delta-target estimators.
    """

    horizon = metadata[
        "horizon_hours"
    ]

    # --------------------------------------------------------
    # Get exact training features
    # --------------------------------------------------------

    required_columns = (
        get_required_model_columns(
            model
        )
    )

    # --------------------------------------------------------
    # Validate feature availability
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column
        not in feature_data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{horizon}h model is missing "
            "required inference features:\n\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Use latest observation only
    # --------------------------------------------------------

    latest_row = (
        feature_data
        .iloc[[-1]]
        .copy()
    )

    model_input = (
        latest_row[
            required_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Detect NaN values
    # --------------------------------------------------------

    nan_columns = (
        model_input
        .columns[
            model_input
            .isna()
            .any()
        ]
        .tolist()
    )

    if nan_columns:

        raise ValueError(
            f"{horizon}h latest inference row "
            "contains NaN values in:\n\n"
            f"{nan_columns}\n\n"
            "The NVE history may not contain "
            "enough observations for the largest "
            "lag or rolling window."
        )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction_output = (
        model.predict(
            model_input
        )
    )

    raw_prediction = float(
        np.asarray(
            prediction_output
        )
        .reshape(-1)[0]
    )

    current_level = float(
        latest_row[
            TARGET_COL
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Reconstruct future level
    # --------------------------------------------------------

    if metadata[
        "is_delta_model"
    ]:

        predicted_level = (
            current_level
            + raw_prediction
        )

    else:

        predicted_level = (
            raw_prediction
        )

    predicted_change = (
        predicted_level
        - current_level
    )

    return {
        "horizon_hours": horizon,
        "current_level": (
            current_level
        ),
        "raw_model_prediction": (
            raw_prediction
        ),
        "predicted_change": float(
            predicted_change
        ),
        "predicted_level": float(
            predicted_level
        ),
        "registered_model": metadata[
            "registered_model"
        ],
        "version": metadata[
            "version"
        ],
        "source_model": metadata[
            "source_model"
        ],
        "is_delta_model": metadata[
            "is_delta_model"
        ],
        "model_uri": metadata[
            "model_uri"
        ],
    }


# ============================================================
# 14. TEMPORARY SIMPLE RISK CLASSIFICATION
# ============================================================

def assign_simple_risk(
    current_level: float,
    predicted_level: float,
) -> str:
    """
    Temporary risk classification.

    This should later be replaced by the same
    risk/flood-threshold policy used during evaluation.
    """

    flood_threshold = 79.13

    predicted_change = (
        predicted_level
        - current_level
    )

    if predicted_level >= flood_threshold:

        return "HIGH"

    if predicted_change >= 0.10:

        return "HIGH"

    if predicted_change > 0:

        return "MEDIUM"

    return "LOW"


# ============================================================
# 15. PRINT FORECAST RESULTS
# ============================================================

def print_forecasts(
    forecasts: list[dict[str, Any]],
    observation_timestamp: pd.Timestamp,
    current_level: float,
) -> None:
    """
    Print human-readable forecast results.
    """

    print()

    print("=" * 80)

    print(
        "LIVE FLOOD FORECAST"
    )

    print("=" * 80)

    print(
        f"Station ID       : "
        f"{STATION_ID}"
    )

    print(
        f"Observation time : "
        f"{observation_timestamp}"
    )

    print(
        f"Current level    : "
        f"{current_level:.5f}"
    )

    print("-" * 80)

    for forecast in forecasts:

        horizon = forecast[
            "horizon_hours"
        ]

        print()

        print(
            f"{horizon} HOUR FORECAST"
        )

        print(
            f"Predicted level  : "
            f"{forecast['predicted_level']:.5f}"
        )

        print(
            f"Predicted change : "
            f"{forecast['predicted_change']:+.5f}"
        )

        print(
            f"Raw model output : "
            f"{forecast['raw_model_prediction']:+.5f}"
        )

        print(
            f"Risk             : "
            f"{forecast['risk_level']}"
        )

        print(
            f"Source model     : "
            f"{forecast['source_model']}"
        )

        print(
            f"Registry version : "
            f"v{forecast['version']}"
        )

        print(
            f"Delta model      : "
            f"{forecast['is_delta_model']}"
        )

        print(
            f"Model URI        : "
            f"{forecast['model_uri']}"
        )

    print()

    print("=" * 80)


# ============================================================
# 16. CREATE FORECAST DATAFRAME
# ============================================================

def create_forecast_dataframe(
    forecasts: list[dict[str, Any]],
    observation_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Convert predictions to a dataframe ready
    for future monitoring and prediction logging.
    """

    forecast_data = pd.DataFrame(
        forecasts
    )

    forecast_data.insert(
        0,
        "prediction_timestamp",
        pd.Timestamp.now(
            tz="UTC"
        ),
    )

    forecast_data.insert(
        1,
        "observation_timestamp",
        observation_timestamp,
    )

    forecast_data.insert(
        2,
        "station_id",
        STATION_ID,
    )

    forecast_data[
        "target_timestamp"
    ] = (
        forecast_data[
            "observation_timestamp"
        ]
        + pd.to_timedelta(
            forecast_data[
                "horizon_hours"
            ],
            unit="h",
        )
    )

    return forecast_data

# ============================================================
# SAVE PREDICTIONS FOR PRODUCTION MONITORING
# ============================================================

def save_prediction_log(
    forecast_data: pd.DataFrame,
) -> Path:
    """
    Append live predictions to the production
    prediction log.

    The log will later be joined with actual NVE
    observations using target_timestamp.
    """

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_data = (
        forecast_data.copy()
    )

    # --------------------------------------------------------
    # Columns needed for delayed performance monitoring
    # --------------------------------------------------------

    log_columns = [
        "prediction_timestamp",
        "observation_timestamp",
        "target_timestamp",
        "station_id",
        "horizon_hours",
        "current_level",
        "predicted_level",
        "predicted_change",
        "raw_model_prediction",
        "risk_level",
        "registered_model",
        "version",
        "source_model",
        "is_delta_model",
        "model_uri",
    ]

    missing_columns = [
        column
        for column in log_columns
        if column not in log_data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Forecast dataframe is missing "
            "prediction-log columns:\n"
            f"{missing_columns}"
        )

    log_data = log_data[
        log_columns
    ].copy()

    # --------------------------------------------------------
    # Prevent exact duplicate forecast records
    # --------------------------------------------------------

    duplicate_key = [
        "observation_timestamp",
        "station_id",
        "horizon_hours",
        "registered_model",
        "version",
    ]

    if PREDICTION_LOG_PATH.exists():

        existing_data = pd.read_csv(
            PREDICTION_LOG_PATH
        )

        combined_data = pd.concat(
            [
                existing_data,
                log_data,
            ],
            ignore_index=True,
        )

        combined_data = (
            combined_data
            .drop_duplicates(
                subset=duplicate_key,
                keep="last",
            )
            .sort_values(
                [
                    "observation_timestamp",
                    "horizon_hours",
                ]
            )
            .reset_index(drop=True)
        )

    else:

        combined_data = (
            log_data
            .sort_values(
                [
                    "observation_timestamp",
                    "horizon_hours",
                ]
            )
            .reset_index(drop=True)
        )

    combined_data.to_csv(
        PREDICTION_LOG_PATH,
        index=False,
    )

    print()

    print("=" * 80)

    print(
        "PREDICTIONS LOGGED"
    )

    print("=" * 80)

    print(
        "Prediction log:"
    )

    print(
        PREDICTION_LOG_PATH
    )

    print(
        "Total prediction records:",
        len(combined_data),
    )

    print(
        "New forecast rows processed:",
        len(log_data),
    )

    return PREDICTION_LOG_PATH
# ============================================================
# 17. MAIN LIVE INFERENCE
# ============================================================

def main() -> None:
    """
    Run end-to-end registry-driven live forecasting.
    """

    print("=" * 80)

    print(
        "FLOOD FORECAST — "
        "REGISTRY DRIVEN LIVE INFERENCE"
    )

    print("=" * 80)

    # ========================================================
    # Configure MLflow
    # ========================================================

    client = configure_mlflow()

    # ========================================================
    # Fetch live NVE observations
    # ========================================================

    print()

    print(
        "Fetching latest NVE observations..."
    )

    live_data = fetch_latest_nve_data(
        days=14,
        station_id=STATION_ID,
    )

    live_data = validate_live_dataframe(
        live_data
    )

    print()

    print(
        "Live rows:",
        len(live_data),
    )

    print(
        "Live columns:",
        live_data.columns.tolist(),
    )

    print(
        "Live data range:",
        live_data[
            DATE_COL
        ].min(),
        "to",
        live_data[
            DATE_COL
        ].max(),
    )

    # ========================================================
    # Build live features
    # ========================================================

    feature_data, time_step_hours = (
        build_inference_features(
            live_data
        )
    )

    latest_timestamp = (
        feature_data[
            DATE_COL
        ].iloc[-1]
    )

    current_level = float(
        feature_data[
            TARGET_COL
        ].iloc[-1]
    )

    print()

    print(
        "Detected time step:",
        time_step_hours,
        "hours",
    )

    print(
        "Feature dataframe shape:",
        feature_data.shape,
    )

    print(
        "Latest timestamp:",
        latest_timestamp,
    )

    print(
        "Current water level:",
        current_level,
    )

    # ========================================================
    # Load and execute registered champions
    # ========================================================

    forecasts: list[
        dict[str, Any]
    ] = []

    for horizon in HORIZONS:

        model, metadata = (
            load_champion_model(
                client=client,
                horizon=horizon,
            )
        )

        forecast = (
            predict_horizon(
                model=model,
                metadata=metadata,
                feature_data=feature_data,
            )
        )

        forecast[
            "risk_level"
        ] = assign_simple_risk(
            current_level=forecast[
                "current_level"
            ],
            predicted_level=forecast[
                "predicted_level"
            ],
        )

        forecasts.append(
            forecast
        )

    # ========================================================
    # Display forecasts
    # ========================================================

    print_forecasts(
        forecasts=forecasts,
        observation_timestamp=latest_timestamp,
        current_level=current_level,
    )

    # ========================================================
    # Prepare monitoring-ready dataframe
    # ========================================================

    forecast_dataframe = (
        create_forecast_dataframe(
            forecasts=forecasts,
            observation_timestamp=(
                latest_timestamp
            ),
        )
    )

    print()

    print(
        "Forecast dataframe:"
    )

    display_columns = [
        "station_id",
        "observation_timestamp",
        "target_timestamp",
        "horizon_hours",
        "current_level",
        "predicted_level",
        "predicted_change",
        "risk_level",
        "source_model",
        "version",
    ]

    print(
        forecast_dataframe[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save predictions for delayed production evaluation
    # ========================================================

    save_prediction_log(
        forecast_data=forecast_dataframe
    )
# ============================================================
# 18. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()

        print("=" * 80)

        print(
            "LIVE FORECAST FAILED"
        )

        print("=" * 80)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 80)

        raise