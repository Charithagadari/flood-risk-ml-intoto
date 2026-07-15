from __future__ import annotations

import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PREDICT_SCRIPT = (
    PROJECT_ROOT
    / "predict_live.py"
)


# IMPORTANT:
# Must match predict_live.py exactly.
PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "predictions"
    / "prediction_log.csv"
)


MLFLOW_DB_PATH = (
    Path.home()
    / ".mlflow"
    / "flood-risk"
    / "mlflow.db"
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH}",
)


PREDICTION_TIMEOUT_SECONDS = int(
    os.getenv(
        "PREDICTION_TIMEOUT_SECONDS",
        "300",
    )
)


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# 3. API RESPONSE SCHEMAS
# ============================================================

class ForecastItem(BaseModel):
    horizon: str
    predicted_water_level: float
    predicted_change: float
    risk_level: str
    model_name: str
    model_version: str
    model_alias: str


class PredictionResponse(BaseModel):
    status: str
    station_id: str
    observation_timestamp: str
    current_water_level: float
    forecasts: dict[str, ForecastItem]
    prediction_generated_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    mlflow_tracking_uri: str
    mlflow_database_exists: bool
    prediction_script_exists: bool
    prediction_log_exists: bool


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def get_prediction_log_row_count() -> int:
    """
    Return current number of prediction-log rows.
    """

    if not PREDICTION_LOG_PATH.exists():
        return 0

    try:

        prediction_log = pd.read_csv(
            PREDICTION_LOG_PATH
        )

        return len(
            prediction_log
        )

    except Exception:

        return 0


def run_live_prediction(
) -> subprocess.CompletedProcess[str]:
    """
    Execute the existing registry-driven
    live inference pipeline.
    """

    if not PREDICT_SCRIPT.exists():

        raise FileNotFoundError(
            "Prediction script was not found:\n"
            f"{PREDICT_SCRIPT}"
        )


    environment = (
        os.environ.copy()
    )


    environment[
        "MLFLOW_TRACKING_URI"
    ] = MLFLOW_TRACKING_URI


    process = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        capture_output=True,
        text=True,
        timeout=(
            PREDICTION_TIMEOUT_SECONDS
        ),
        env=environment,
        check=False,
    )


    return process


def read_prediction_log(
) -> pd.DataFrame:
    """
    Read the production prediction log.
    """

    if not PREDICTION_LOG_PATH.exists():

        raise FileNotFoundError(
            "Prediction log was not found:\n"
            f"{PREDICTION_LOG_PATH}"
        )


    prediction_log = pd.read_csv(
        PREDICTION_LOG_PATH
    )


    if prediction_log.empty:

        raise ValueError(
            "Prediction log is empty."
        )


    return prediction_log


def get_latest_forecast_rows(
    prediction_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the newest forecast group using
    observation_timestamp.

    predict_live.py creates one row for each:
        6h
        24h
        72h
    """

    required_columns = [
        "observation_timestamp",
        "station_id",
        "horizon_hours",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column
        not in prediction_log.columns
    ]


    if missing_columns:

        raise ValueError(
            "Prediction log is missing required columns:\n"
            f"{missing_columns}\n\n"
            "Available columns:\n"
            f"{prediction_log.columns.tolist()}"
        )


    observation_times = pd.to_datetime(
        prediction_log[
            "observation_timestamp"
        ],
        errors="coerce",
        utc=True,
    )


    if observation_times.isna().all():

        raise ValueError(
            "No valid observation timestamps "
            "were found in the prediction log."
        )


    latest_observation_time = (
        observation_times.max()
    )


    latest_rows = (
        prediction_log.loc[
            observation_times
            == latest_observation_time
        ]
        .copy()
    )


    latest_rows = (
        latest_rows
        .sort_values(
            "horizon_hours"
        )
        .reset_index(
            drop=True
        )
    )


    return latest_rows


def validate_forecast_rows(
    forecast_rows: pd.DataFrame,
) -> None:
    """
    Validate the exact prediction-log schema
    generated by predict_live.py.
    """

    required_columns = [
        "prediction_timestamp",
        "observation_timestamp",
        "target_timestamp",
        "station_id",
        "horizon_hours",
        "current_level",
        "predicted_level",
        "predicted_change",
        "risk_level",
        "registered_model",
        "version",
        "source_model",
        "is_delta_model",
        "model_uri",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column
        not in forecast_rows.columns
    ]


    if missing_columns:

        raise ValueError(
            "Prediction log schema mismatch.\n\n"
            "Missing columns:\n"
            f"{missing_columns}\n\n"
            "Available columns:\n"
            f"{forecast_rows.columns.tolist()}"
        )


def normalize_horizon(
    horizon_hours: Any,
) -> str:
    """
    Convert:
        6   -> 6h
        24  -> 24h
        72  -> 72h
    """

    horizon = int(
        float(
            horizon_hours
        )
    )


    return f"{horizon}h"


def build_prediction_response(
    forecast_rows: pd.DataFrame,
) -> PredictionResponse:
    """
    Convert prediction_log rows into
    the public API response.
    """

    validate_forecast_rows(
        forecast_rows
    )


    first_row = (
        forecast_rows.iloc[0]
    )


    station_id = str(
        first_row[
            "station_id"
        ]
    )


    observation_timestamp = str(
        first_row[
            "observation_timestamp"
        ]
    )


    current_water_level = float(
        first_row[
            "current_level"
        ]
    )


    forecasts: dict[
        str,
        ForecastItem,
    ] = {}


    for _, row in (
        forecast_rows.iterrows()
    ):

        horizon = normalize_horizon(
            row[
                "horizon_hours"
            ]
        )


        forecasts[
            horizon
        ] = ForecastItem(
            horizon=horizon,
            predicted_water_level=float(
                row[
                    "predicted_level"
                ]
            ),
            predicted_change=float(
                row[
                    "predicted_change"
                ]
            ),
            risk_level=str(
                row[
                    "risk_level"
                ]
            ),
            model_name=str(
                row[
                    "registered_model"
                ]
            ),
            model_version=str(
                row[
                    "version"
                ]
            ),
            model_alias="champion",
        )


    required_horizons = {
        "6h",
        "24h",
        "72h",
    }


    missing_horizons = (
        required_horizons
        - set(
            forecasts.keys()
        )
    )


    if missing_horizons:

        raise ValueError(
            "Latest forecast group is missing "
            "required horizons:\n"
            f"{sorted(missing_horizons)}"
        )


    return PredictionResponse(
        status="success",
        station_id=station_id,
        observation_timestamp=(
            observation_timestamp
        ),
        current_water_level=(
            current_water_level
        ),
        forecasts=forecasts,
        prediction_generated_at=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


# ============================================================
# 5. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "=" * 80
    )

    print(
        "FLOOD RISK FORECASTING API"
    )

    print(
        "=" * 80
    )


    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )


    print()

    print(
        "MLflow tracking URI:"
    )

    print(
        MLFLOW_TRACKING_URI
    )


    print()

    if MLFLOW_TRACKING_URI.startswith(
        ("http://", "https://")
    ):

        print(
            "MLflow mode:"
        )

        print(
            "Remote tracking server"
        )

    else:

        print(
            "MLflow DB exists:",
            MLFLOW_DB_PATH.exists(),
        )


    print()

    print(
        "Prediction script:"
    )

    print(
        PREDICT_SCRIPT
    )


    print()

    print(
        "Prediction log:"
    )

    print(
        PREDICTION_LOG_PATH
    )


    print()

    print(
        "Prediction script exists:",
        PREDICT_SCRIPT.exists(),
    )


    print(
        "Prediction log exists:",
        PREDICTION_LOG_PATH.exists(),
    )


    print()

    print(
        "API startup complete."
    )

    print(
        "=" * 80
    )


    yield


    print(
        "Flood Risk Forecasting API "
        "shutting down."
    )


# ============================================================
# 6. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=(
        "Flood Risk Forecasting API"
    ),
    description=(
        "Registry-driven live water-level "
        "forecasting API using MLflow "
        "champion models."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 7. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=False,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# ============================================================
# 8. ROUTES
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {
        "service": (
            "Flood Risk Forecasting API"
        ),
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    prediction_script_exists = (
        PREDICT_SCRIPT.exists()
    )

    prediction_log_exists = (
        PREDICTION_LOG_PATH.exists()
    )

    uses_remote_mlflow = (
        MLFLOW_TRACKING_URI.startswith(
            "http://"
        )
        or MLFLOW_TRACKING_URI.startswith(
            "https://"
        )
    )

    if uses_remote_mlflow:

        mlflow_database_exists = True

    else:

        mlflow_database_exists = (
            MLFLOW_DB_PATH.exists()
        )

    if (
        mlflow_database_exists
        and prediction_script_exists
    ):

        status = "healthy"

    else:

        status = "degraded"

    return HealthResponse(
        status=status,
        service=(
            "Flood Risk Forecasting API"
        ),
        mlflow_tracking_uri=(
            MLFLOW_TRACKING_URI
        ),
        mlflow_database_exists=(
            mlflow_database_exists
        ),
        prediction_script_exists=(
            prediction_script_exists
        ),
        prediction_log_exists=(
            prediction_log_exists
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict() -> PredictionResponse:
    """
    Run live registry-driven prediction.
    """

    try:

        process = (
            run_live_prediction()
        )


    except subprocess.TimeoutExpired as error:

        raise HTTPException(
            status_code=504,
            detail={
                "error": (
                    "Live prediction pipeline timed out."
                ),
                "timeout_seconds": (
                    PREDICTION_TIMEOUT_SECONDS
                ),
            },
        ) from error


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    "Could not start live "
                    "prediction pipeline."
                ),
                "message": str(
                    error
                ),
            },
        ) from error


    if process.returncode != 0:

        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    "Live prediction pipeline failed."
                ),
                "return_code": (
                    process.returncode
                ),
                "stdout": (
                    process.stdout[
                        -5000:
                    ]
                ),
                "stderr": (
                    process.stderr[
                        -5000:
                    ]
                ),
            },
        )


    try:

        prediction_log = (
            read_prediction_log()
        )


        latest_rows = (
            get_latest_forecast_rows(
                prediction_log
            )
        )


        return (
            build_prediction_response(
                latest_rows
            )
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    "Prediction succeeded, but "
                    "the API could not parse "
                    "the prediction log."
                ),
                "message": str(
                    error
                ),
                "prediction_stdout": (
                    process.stdout[
                        -5000:
                    ]
                ),
            },
        ) from error


@app.get(
    "/model-info"
)
def model_info() -> dict[str, Any]:

    prediction_log_exists = (
        PREDICTION_LOG_PATH.exists()
    )


    model_information: dict[
        str,
        Any,
    ] = {
        "tracking_uri": (
            MLFLOW_TRACKING_URI
        ),
        "registry_strategy": (
            "champion alias"
        ),
        "inference_pipeline": (
            "predict_live.py"
        ),
        "forecast_horizons": [
            "6h",
            "24h",
            "72h",
        ],
        "prediction_log": str(
            PREDICTION_LOG_PATH
        ),
    }


    if not prediction_log_exists:

        model_information[
            "champions"
        ] = {}

        return model_information


    try:

        prediction_log = (
            read_prediction_log()
        )


        latest_rows = (
            get_latest_forecast_rows(
                prediction_log
            )
        )


        champions = {}


        for _, row in (
            latest_rows.iterrows()
        ):

            horizon = normalize_horizon(
                row[
                    "horizon_hours"
                ]
            )


            champions[
                horizon
            ] = {
                "registered_model": str(
                    row[
                        "registered_model"
                    ]
                ),
                "version": str(
                    row[
                        "version"
                    ]
                ),
                "alias": "champion",
                "source_model": str(
                    row[
                        "source_model"
                    ]
                ),
                "model_uri": str(
                    row[
                        "model_uri"
                    ]
                ),
                "is_delta_model": bool(
                    row[
                        "is_delta_model"
                    ]
                ),
            }


        model_information[
            "champions"
        ] = champions


    except Exception as error:

        model_information[
            "champions"
        ] = {}


        model_information[
            "metadata_error"
        ] = str(
            error
        )


    return model_information