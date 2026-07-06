from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.nve_client import (
    fetch_water_level_range,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


PREDICTION_DIRECTORY = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "predictions"
)


PREDICTION_LOG_PATH = (
    PREDICTION_DIRECTORY
    / "prediction_log.csv"
)


EVALUATION_LOG_PATH = (
    PREDICTION_DIRECTORY
    / "evaluation_log.csv"
)


PERFORMANCE_SUMMARY_PATH = (
    PREDICTION_DIRECTORY
    / "performance_summary.csv"
)


STATION_ID = "1.15.0"


# ============================================================
# 2. MATCHING CONFIGURATION
# ============================================================

# Your NVE series uses hourly observations.
#
# An exact target timestamp should normally exist.
# This tolerance handles a small timestamp mismatch or
# occasional delayed/irregular observation.

MATCH_TOLERANCE = pd.Timedelta(
    "90min"
)


# ============================================================
# 3. LOAD PREDICTION LOG
# ============================================================

def load_prediction_log() -> pd.DataFrame:
    """
    Load and validate production predictions.
    """

    if not PREDICTION_LOG_PATH.exists():

        raise FileNotFoundError(
            "Prediction log was not found.\n\n"
            f"Expected:\n"
            f"{PREDICTION_LOG_PATH}\n\n"
            "Run predict_live.py first."
        )

    predictions = pd.read_csv(
        PREDICTION_LOG_PATH
    )

    required_columns = [
        "prediction_timestamp",
        "observation_timestamp",
        "target_timestamp",
        "station_id",
        "horizon_hours",
        "current_level",
        "predicted_level",
        "registered_model",
        "version",
        "source_model",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:

        raise ValueError(
            "Prediction log is missing "
            "required columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    timestamp_columns = [
        "prediction_timestamp",
        "observation_timestamp",
        "target_timestamp",
    ]

    for column in timestamp_columns:

        predictions[column] = (
            pd.to_datetime(
                predictions[column],
                errors="coerce",
                utc=True,
            )
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "horizon_hours",
        "current_level",
        "predicted_level",
        "version",
    ]

    for column in numeric_columns:

        predictions[column] = (
            pd.to_numeric(
                predictions[column],
                errors="coerce",
            )
        )

    predictions = (
        predictions
        .dropna(
            subset=[
                "target_timestamp",
                "horizon_hours",
                "predicted_level",
            ]
        )
        .sort_values(
            [
                "target_timestamp",
                "horizon_hours",
            ]
        )
        .reset_index(drop=True)
    )

    return predictions


# ============================================================
# 4. LOAD EXISTING EVALUATIONS
# ============================================================

def load_existing_evaluations() -> pd.DataFrame:
    """
    Load previous delayed evaluations if available.
    """

    if not EVALUATION_LOG_PATH.exists():

        return pd.DataFrame()

    evaluations = pd.read_csv(
        EVALUATION_LOG_PATH
    )

    timestamp_columns = [
        "prediction_timestamp",
        "observation_timestamp",
        "target_timestamp",
        "actual_timestamp",
        "evaluation_timestamp",
    ]

    for column in timestamp_columns:

        if column in evaluations.columns:

            evaluations[column] = (
                pd.to_datetime(
                    evaluations[column],
                    errors="coerce",
                    utc=True,
                )
            )

    return evaluations


# ============================================================
# 5. CREATE UNIQUE PREDICTION KEY
# ============================================================

def create_prediction_key(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Create a stable identifier for one deployed forecast.

    This allows us to avoid evaluating the same
    prediction repeatedly.
    """

    observation_time = (
        data["observation_timestamp"]
        .astype(str)
    )

    horizon = (
        data["horizon_hours"]
        .astype(str)
    )

    station = (
        data["station_id"]
        .astype(str)
    )

    registered_model = (
        data["registered_model"]
        .astype(str)
    )

    version = (
        data["version"]
        .astype(str)
    )

    return (
        station
        + "|"
        + observation_time
        + "|"
        + horizon
        + "|"
        + registered_model
        + "|v"
        + version
    )


# ============================================================
# 6. FIND MATURE PREDICTIONS
# ============================================================

def find_predictions_ready_for_evaluation(
    predictions: pd.DataFrame,
    existing_evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find predictions whose target time has passed
    and that have not already been evaluated.
    """

    predictions = predictions.copy()

    predictions[
        "prediction_key"
    ] = create_prediction_key(
        predictions
    )

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    mature_predictions = (
        predictions[
            predictions[
                "target_timestamp"
            ]
            <= now_utc
        ]
        .copy()
    )

    if existing_evaluations.empty:

        return mature_predictions

    if (
        "prediction_key"
        not in existing_evaluations.columns
    ):

        return mature_predictions

    evaluated_keys = set(
        existing_evaluations[
            "prediction_key"
        ]
        .dropna()
        .astype(str)
    )

    mature_predictions = (
        mature_predictions[
            ~mature_predictions[
                "prediction_key"
            ]
            .isin(
                evaluated_keys
            )
        ]
        .copy()
    )

    return mature_predictions


# ============================================================
# 7. MATCH ACTUAL OBSERVATION
# ============================================================

def match_actual_observation(
    target_timestamp: pd.Timestamp,
    actual_data: pd.DataFrame,
) -> tuple[
    pd.Timestamp | None,
    float | None,
    float | None,
]:
    """
    Match a forecast target timestamp to the nearest
    actual water-level observation.

    Returns:
        actual_timestamp
        actual_water_level
        match_difference_minutes
    """

    if actual_data.empty:

        return (
            None,
            None,
            None,
        )

    candidate_data = (
        actual_data.copy()
    )

    candidate_data[
        "timestamp_difference"
    ] = (
        candidate_data[
            "timestamp"
        ]
        - target_timestamp
    ).abs()

    nearest_index = (
        candidate_data[
            "timestamp_difference"
        ]
        .idxmin()
    )

    nearest_row = (
        candidate_data
        .loc[
            nearest_index
        ]
    )

    difference = nearest_row[
        "timestamp_difference"
    ]

    if difference > MATCH_TOLERANCE:

        return (
            None,
            None,
            float(
                difference
                .total_seconds()
                / 60
            ),
        )

    actual_timestamp = (
        nearest_row["timestamp"]
    )

    actual_water_level = float(
        nearest_row[
            "actual_water_level"
        ]
    )

    difference_minutes = float(
        difference
        .total_seconds()
        / 60
    )

    return (
        actual_timestamp,
        actual_water_level,
        difference_minutes,
    )


# ============================================================
# 8. EVALUATE PREDICTIONS
# ============================================================

def evaluate_mature_predictions(
    mature_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetch NVE ground truth and calculate production
    errors for matured predictions.
    """

    if mature_predictions.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Determine one API time range for all pending forecasts
    # --------------------------------------------------------

    minimum_target_time = (
        mature_predictions[
            "target_timestamp"
        ]
        .min()
    )

    maximum_target_time = (
        mature_predictions[
            "target_timestamp"
        ]
        .max()
    )

    actual_data = (
        fetch_water_level_range(
            start_datetime=(
                minimum_target_time
                - MATCH_TOLERANCE
            ),
            end_datetime=(
                maximum_target_time
                + MATCH_TOLERANCE
            ),
            station_id=STATION_ID,
        )
    )

    evaluation_rows = []

    for _, prediction in (
        mature_predictions.iterrows()
    ):

        target_timestamp = (
            prediction[
                "target_timestamp"
            ]
        )

        (
            actual_timestamp,
            actual_water_level,
            match_difference_minutes,
        ) = match_actual_observation(
            target_timestamp=(
                target_timestamp
            ),
            actual_data=actual_data,
        )

        # ----------------------------------------------------
        # Ground truth not available yet
        # ----------------------------------------------------

        if actual_water_level is None:

            print()

            print(
                "Actual observation unavailable:"
            )

            print(
                "Target timestamp:",
                target_timestamp,
            )

            print(
                "Horizon:",
                prediction[
                    "horizon_hours"
                ],
            )

            continue

        predicted_level = float(
            prediction[
                "predicted_level"
            ]
        )

        error = (
            predicted_level
            - actual_water_level
        )

        absolute_error = abs(
            error
        )

        squared_error = (
            error ** 2
        )

        # ----------------------------------------------------
        # Avoid division near zero even though water levels
        # in this series are far above zero.
        # ----------------------------------------------------

        if abs(actual_water_level) > 1e-12:

            absolute_percentage_error = (
                absolute_error
                / abs(actual_water_level)
                * 100
            )

        else:

            absolute_percentage_error = (
                np.nan
            )

        evaluation_row = (
            prediction.to_dict()
        )

        evaluation_row.update(
            {
                "actual_timestamp": (
                    actual_timestamp
                ),
                "actual_water_level": (
                    actual_water_level
                ),
                "match_difference_minutes": (
                    match_difference_minutes
                ),
                "error": float(
                    error
                ),
                "absolute_error": float(
                    absolute_error
                ),
                "squared_error": float(
                    squared_error
                ),
                "absolute_percentage_error": float(
                    absolute_percentage_error
                ),
                "evaluation_timestamp": (
                    pd.Timestamp.now(
                        tz="UTC"
                    )
                ),
            }
        )

        evaluation_rows.append(
            evaluation_row
        )

        print()

        print("-" * 70)

        print(
            "PREDICTION EVALUATED"
        )

        print("-" * 70)

        print(
            "Target timestamp :",
            target_timestamp,
        )

        print(
            "Actual timestamp :",
            actual_timestamp,
        )

        print(
            "Horizon          :",
            f"{int(prediction['horizon_hours'])}h",
        )

        print(
            "Predicted level  :",
            f"{predicted_level:.6f}",
        )

        print(
            "Actual level     :",
            f"{actual_water_level:.6f}",
        )

        print(
            "Error            :",
            f"{error:+.6f}",
        )

        print(
            "Absolute error   :",
            f"{absolute_error:.6f}",
        )

        print(
            "Model            :",
            prediction[
                "source_model"
            ],
        )

        print(
            "Version          :",
            f"v{prediction['version']}",
        )

    return pd.DataFrame(
        evaluation_rows
    )


# ============================================================
# 9. SAVE EVALUATION LOG
# ============================================================

def save_evaluation_log(
    new_evaluations: pd.DataFrame,
    existing_evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append newly evaluated predictions to the
    persistent production evaluation log.
    """

    if new_evaluations.empty:

        return existing_evaluations

    if existing_evaluations.empty:

        combined_evaluations = (
            new_evaluations.copy()
        )

    else:

        combined_evaluations = pd.concat(
            [
                existing_evaluations,
                new_evaluations,
            ],
            ignore_index=True,
        )

    combined_evaluations = (
        combined_evaluations
        .drop_duplicates(
            subset=[
                "prediction_key",
            ],
            keep="last",
        )
        .sort_values(
            [
                "target_timestamp",
                "horizon_hours",
            ]
        )
        .reset_index(drop=True)
    )

    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_evaluations.to_csv(
        EVALUATION_LOG_PATH,
        index=False,
    )

    print()

    print("=" * 80)

    print(
        "EVALUATION LOG SAVED"
    )

    print("=" * 80)

    print(
        "Path:",
        EVALUATION_LOG_PATH,
    )

    print(
        "Total evaluated predictions:",
        len(combined_evaluations),
    )

    return combined_evaluations


# ============================================================
# 10. CALCULATE PERFORMANCE SUMMARY
# ============================================================

def calculate_performance_summary(
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate production performance by forecast
    horizon, registered model, and model version.
    """

    if evaluations.empty:

        return pd.DataFrame()

    summary_rows = []

    grouping_columns = [
        "horizon_hours",
        "registered_model",
        "version",
        "source_model",
    ]

    for group_values, group_data in (
        evaluations.groupby(
            grouping_columns,
            dropna=False,
        )
    ):

        (
            horizon_hours,
            registered_model,
            version,
            source_model,
        ) = group_values

        mae = float(
            group_data[
                "absolute_error"
            ].mean()
        )

        rmse = float(
            np.sqrt(
                group_data[
                    "squared_error"
                ].mean()
            )
        )

        mean_error = float(
            group_data[
                "error"
            ].mean()
        )

        mape = float(
            group_data[
                "absolute_percentage_error"
            ].mean()
        )

        summary_rows.append(
            {
                "horizon_hours": (
                    horizon_hours
                ),
                "registered_model": (
                    registered_model
                ),
                "version": version,
                "source_model": (
                    source_model
                ),
                "prediction_count": int(
                    len(group_data)
                ),
                "production_mae": mae,
                "production_rmse": rmse,
                "production_mean_error": (
                    mean_error
                ),
                "production_mape_percent": (
                    mape
                ),
                "first_target_timestamp": (
                    group_data[
                        "target_timestamp"
                    ].min()
                ),
                "latest_target_timestamp": (
                    group_data[
                        "target_timestamp"
                    ].max()
                ),
                "calculated_at": (
                    pd.Timestamp.now(
                        tz="UTC"
                    )
                ),
            }
        )

    performance_summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "horizon_hours",
                "version",
            ]
        )
        .reset_index(drop=True)
    )

    return performance_summary


# ============================================================
# 11. SAVE PERFORMANCE SUMMARY
# ============================================================

def save_performance_summary(
    performance_summary: pd.DataFrame,
) -> None:
    """
    Save production metrics.
    """

    if performance_summary.empty:

        return

    performance_summary.to_csv(
        PERFORMANCE_SUMMARY_PATH,
        index=False,
    )

    print()

    print("=" * 80)

    print(
        "PRODUCTION PERFORMANCE SUMMARY"
    )

    print("=" * 80)

    display_columns = [
        "horizon_hours",
        "source_model",
        "version",
        "prediction_count",
        "production_mae",
        "production_rmse",
        "production_mean_error",
    ]

    print(
        performance_summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Summary saved:"
    )

    print(
        PERFORMANCE_SUMMARY_PATH
    )


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:
    """
    Run delayed production performance evaluation.
    """

    print("=" * 80)

    print(
        "FLOOD FORECAST — "
        "DELAYED GROUND-TRUTH EVALUATION"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    predictions = (
        load_prediction_log()
    )

    print()

    print(
        "Prediction records:",
        len(predictions),
    )

    print(
        "Prediction target range:",
        predictions[
            "target_timestamp"
        ].min(),
        "to",
        predictions[
            "target_timestamp"
        ].max(),
    )

    # --------------------------------------------------------
    # Load previous evaluations
    # --------------------------------------------------------

    existing_evaluations = (
        load_existing_evaluations()
    )

    print(
        "Previously evaluated:",
        len(existing_evaluations),
    )

    # --------------------------------------------------------
    # Find targets whose time has passed
    # --------------------------------------------------------

    mature_predictions = (
        find_predictions_ready_for_evaluation(
            predictions=predictions,
            existing_evaluations=(
                existing_evaluations
            ),
        )
    )

    print(
        "Ready for evaluation:",
        len(mature_predictions),
    )

    # --------------------------------------------------------
    # Nothing ready yet
    # --------------------------------------------------------

    if mature_predictions.empty:

        future_targets = (
            predictions[
                predictions[
                    "target_timestamp"
                ]
                > pd.Timestamp.now(
                    tz="UTC"
                )
            ]
        )

        print()

        print("=" * 80)

        print(
            "NO NEW PREDICTIONS READY "
            "FOR EVALUATION"
        )

        print("=" * 80)

        if not future_targets.empty:

            next_target = (
                future_targets[
                    "target_timestamp"
                ].min()
            )

            print(
                "Next target timestamp:",
                next_target,
            )

        return

    # --------------------------------------------------------
    # Fetch actuals and evaluate
    # --------------------------------------------------------

    new_evaluations = (
        evaluate_mature_predictions(
            mature_predictions
        )
    )

    if new_evaluations.empty:

        print()

        print(
            "No matching actual NVE observations "
            "were available."
        )

        return

    # --------------------------------------------------------
    # Persist evaluation records
    # --------------------------------------------------------

    all_evaluations = (
        save_evaluation_log(
            new_evaluations=(
                new_evaluations
            ),
            existing_evaluations=(
                existing_evaluations
            ),
        )
    )

    # --------------------------------------------------------
    # Calculate current production metrics
    # --------------------------------------------------------

    performance_summary = (
        calculate_performance_summary(
            all_evaluations
        )
    )

    save_performance_summary(
        performance_summary
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()

        print("=" * 80)

        print(
            "PREDICTION EVALUATION FAILED"
        )

        print("=" * 80)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 80)

        raise