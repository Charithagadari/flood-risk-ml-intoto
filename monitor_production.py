from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient


# ============================================================
# 1. PROJECT CONFIGURATION
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


EVALUATION_LOG_PATH = (
    PREDICTION_DIRECTORY
    / "evaluation_log.csv"
)


PERFORMANCE_SUMMARY_PATH = (
    PREDICTION_DIRECTORY
    / "performance_summary.csv"
)


MONITORING_OUTPUT_PATH = (
    PREDICTION_DIRECTORY
    / "monitoring_summary.csv"
)


MONITORING_JSON_PATH = (
    PREDICTION_DIRECTORY
    / "monitoring_summary.json"
)


# ============================================================
# 2. MLFLOW CONFIGURATION
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


TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB}"
)


MONITORING_EXPERIMENT = (
    "flood_risk_monitoring"
)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_NAMES = {
    6: "flood_forecast_6h",
    24: "flood_forecast_24h",
    72: "flood_forecast_72h",
}


# ============================================================
# 4. MONITORING THRESHOLDS
# ============================================================

MINIMUM_EVALUATED_PREDICTIONS = {
    6: 24,
    24: 14,
    72: 7,
}


# Production MAE / test MAE.
#
# Example:
#
# production MAE = 0.040
# test MAE       = 0.020
#
# ratio = 2.0
#
# 1.5 means production MAE is 50% worse than test MAE.

WARNING_DEGRADATION_RATIO = 1.50

CRITICAL_DEGRADATION_RATIO = 2.00


# ============================================================
# 5. CONFIGURE MLFLOW
# ============================================================

def configure_mlflow() -> MlflowClient:
    """
    Configure the same MLflow backend used during
    training and registry-driven inference.
    """

    if not MLFLOW_DB.exists():

        raise FileNotFoundError(
            "MLflow SQLite database was not found.\n\n"
            f"Expected:\n{MLFLOW_DB}"
        )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        MONITORING_EXPERIMENT
    )

    client = MlflowClient()

    print(
        "MLflow tracking URI:"
    )

    print(
        mlflow.get_tracking_uri()
    )

    print(
        "Monitoring experiment:"
    )

    print(
        MONITORING_EXPERIMENT
    )

    return client


# ============================================================
# 6. LOAD EVALUATION LOG
# ============================================================

def load_evaluation_log() -> pd.DataFrame:
    """
    Load actual production forecast evaluations.
    """

    if not EVALUATION_LOG_PATH.exists():

        raise FileNotFoundError(
            "Production evaluation log was not found.\n\n"
            f"Expected:\n"
            f"{EVALUATION_LOG_PATH}\n\n"
            "Run evaluate_predictions.py first."
        )

    evaluations = pd.read_csv(
        EVALUATION_LOG_PATH
    )

    required_columns = [
        "horizon_hours",
        "registered_model",
        "version",
        "source_model",
        "predicted_level",
        "actual_water_level",
        "error",
        "absolute_error",
        "squared_error",
        "target_timestamp",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in evaluations.columns
    ]

    if missing_columns:

        raise ValueError(
            "Evaluation log is missing columns:\n"
            f"{missing_columns}"
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

    numeric_columns = [
        "horizon_hours",
        "version",
        "predicted_level",
        "actual_water_level",
        "error",
        "absolute_error",
        "squared_error",
    ]

    for column in numeric_columns:

        evaluations[column] = (
            pd.to_numeric(
                evaluations[column],
                errors="coerce",
            )
        )

    evaluations = (
        evaluations
        .dropna(
            subset=[
                "horizon_hours",
                "predicted_level",
                "actual_water_level",
                "absolute_error",
                "squared_error",
            ]
        )
        .sort_values(
            "target_timestamp"
        )
        .reset_index(drop=True)
    )

    if evaluations.empty:

        raise ValueError(
            "Evaluation log contains no valid "
            "production evaluation records."
        )

    return evaluations


# ============================================================
# 7. GET CHAMPION INFORMATION
# ============================================================

def get_champion_information(
    client: MlflowClient,
    horizon: int,
) -> dict[str, Any]:
    """
    Resolve current champion model and its source run.
    """

    if horizon not in MODEL_NAMES:

        raise ValueError(
            f"Unsupported horizon: {horizon}"
        )

    registered_model_name = (
        MODEL_NAMES[horizon]
    )

    champion_version = (
        client.get_model_version_by_alias(
            name=registered_model_name,
            alias="champion",
        )
    )

    source_run = client.get_run(
        champion_version.run_id
    )

    version_tags = (
        champion_version.tags
        or {}
    )

    run_metrics = (
        source_run.data.metrics
        or {}
    )

    return {
        "horizon_hours": horizon,
        "registered_model": (
            registered_model_name
        ),
        "champion_version": int(
            champion_version.version
        ),
        "champion_run_id": (
            champion_version.run_id
        ),
        "source_model": version_tags.get(
            "source_model",
            "",
        ),
        "selection_metric": version_tags.get(
            "selection_metric",
            "",
        ),
        "selection_metric_value": (
            version_tags.get(
                "selection_metric_value"
            )
        ),
        "run_metrics": run_metrics,
        "version_tags": version_tags,
    }


# ============================================================
# 8. FIND ORIGINAL TEST METRIC
# ============================================================

def find_test_metric(
    champion_information: dict[str, Any],
    metric_name: str,
) -> float | None:
    """
    Find a model test metric from the champion's
    original MLflow source run.

    Supports common metric naming conventions.
    """

    run_metrics = champion_information[
        "run_metrics"
    ]

    horizon = champion_information[
        "horizon_hours"
    ]

    metric_name_lower = (
        metric_name.lower()
    )

    candidates = [
        metric_name_lower,
        metric_name_lower.upper(),
        f"test_{metric_name_lower}",
        f"{metric_name_lower}_test",
        f"{metric_name_lower}_{horizon}h",
        f"test_{metric_name_lower}_{horizon}h",
    ]

    # --------------------------------------------------------
    # Exact candidate matching
    # --------------------------------------------------------

    for candidate in candidates:

        for (
            logged_metric_name,
            logged_metric_value,
        ) in run_metrics.items():

            if (
                logged_metric_name.lower()
                == candidate.lower()
            ):

                return float(
                    logged_metric_value
                )

    # --------------------------------------------------------
    # Fallback:
    # Search metric names containing MAE or RMSE.
    # --------------------------------------------------------

    partial_matches = []

    for (
        logged_metric_name,
        logged_metric_value,
    ) in run_metrics.items():

        if (
            metric_name_lower
            in logged_metric_name.lower()
        ):

            partial_matches.append(
                (
                    logged_metric_name,
                    logged_metric_value,
                )
            )

    if len(partial_matches) == 1:

        return float(
            partial_matches[0][1]
        )

    # --------------------------------------------------------
    # Registry fallback.
    #
    # Your registry uses MAE as the selection metric.
    # --------------------------------------------------------

    selection_metric = str(
        champion_information.get(
            "selection_metric",
            ""
        )
    ).lower()

    selection_value = (
        champion_information.get(
            "selection_metric_value"
        )
    )

    if (
        metric_name_lower == selection_metric
        and selection_value is not None
    ):

        try:

            return float(
                selection_value
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    return None


# ============================================================
# 9. CALCULATE PRODUCTION PERFORMANCE
# ============================================================

def calculate_champion_production_metrics(
    evaluations: pd.DataFrame,
    champion_information: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate performance only for evaluations produced
    by the CURRENT champion version.
    """

    horizon = champion_information[
        "horizon_hours"
    ]

    registered_model = champion_information[
        "registered_model"
    ]

    champion_version = champion_information[
        "champion_version"
    ]

    champion_evaluations = (
        evaluations[
            (
                evaluations[
                    "horizon_hours"
                ]
                == horizon
            )
            &
            (
                evaluations[
                    "registered_model"
                ]
                == registered_model
            )
            &
            (
                evaluations[
                    "version"
                ]
                == champion_version
            )
        ]
        .copy()
    )

    prediction_count = len(
        champion_evaluations
    )

    if prediction_count == 0:

        return {
            "prediction_count": 0,
            "production_mae": None,
            "production_rmse": None,
            "production_mean_error": None,
            "first_target_timestamp": None,
            "latest_target_timestamp": None,
        }

    production_mae = float(
        champion_evaluations[
            "absolute_error"
        ].mean()
    )

    production_rmse = float(
        np.sqrt(
            champion_evaluations[
                "squared_error"
            ].mean()
        )
    )

    production_mean_error = float(
        champion_evaluations[
            "error"
        ].mean()
    )

    return {
        "prediction_count": int(
            prediction_count
        ),
        "production_mae": (
            production_mae
        ),
        "production_rmse": (
            production_rmse
        ),
        "production_mean_error": (
            production_mean_error
        ),
        "first_target_timestamp": (
            champion_evaluations[
                "target_timestamp"
            ].min()
        ),
        "latest_target_timestamp": (
            champion_evaluations[
                "target_timestamp"
            ].max()
        ),
    }


# ============================================================
# 10. DETERMINE MONITORING STATUS
# ============================================================

def determine_monitoring_status(
    horizon: int,
    prediction_count: int,
    degradation_ratio: float | None,
) -> tuple[str, str]:
    """
    Decide production monitoring status.

    Returns:
        monitoring_status
        monitoring_reason
    """

    minimum_predictions = (
        MINIMUM_EVALUATED_PREDICTIONS[
            horizon
        ]
    )

    if prediction_count < minimum_predictions:

        return (
            "INSUFFICIENT_DATA",
            (
                f"{prediction_count} evaluated "
                f"predictions; minimum required "
                f"is {minimum_predictions}."
            ),
        )

    if degradation_ratio is None:

        return (
            "BASELINE_UNAVAILABLE",
            (
                "Original champion test MAE "
                "could not be resolved."
            ),
        )

    if (
        degradation_ratio
        >= CRITICAL_DEGRADATION_RATIO
    ):

        return (
            "CRITICAL",
            (
                "Production MAE is "
                f"{degradation_ratio:.2f}x "
                "the original test MAE."
            ),
        )

    if (
        degradation_ratio
        >= WARNING_DEGRADATION_RATIO
    ):

        return (
            "WARNING",
            (
                "Production MAE is "
                f"{degradation_ratio:.2f}x "
                "the original test MAE."
            ),
        )

    return (
        "HEALTHY",
        (
            "Production MAE remains within "
            "the configured degradation limit."
        ),
    )


# ============================================================
# 11. BUILD MONITORING SUMMARY
# ============================================================

def build_monitoring_summary(
    evaluations: pd.DataFrame,
    client: MlflowClient,
) -> pd.DataFrame:
    """
    Build monitoring status for every forecast horizon.
    """

    monitoring_rows = []

    for horizon in MODEL_NAMES:

        champion_information = (
            get_champion_information(
                client=client,
                horizon=horizon,
            )
        )

        production_metrics = (
            calculate_champion_production_metrics(
                evaluations=evaluations,
                champion_information=(
                    champion_information
                ),
            )
        )

        test_mae = find_test_metric(
            champion_information=(
                champion_information
            ),
            metric_name="mae",
        )

        test_rmse = find_test_metric(
            champion_information=(
                champion_information
            ),
            metric_name="rmse",
        )

        production_mae = (
            production_metrics[
                "production_mae"
            ]
        )

        # ----------------------------------------------------
        # Calculate degradation ratio
        # ----------------------------------------------------

        if (
            production_mae is not None
            and test_mae is not None
            and test_mae > 0
        ):

            degradation_ratio = float(
                production_mae
                / test_mae
            )

            degradation_percent = float(
                (
                    degradation_ratio
                    - 1
                )
                * 100
            )

        else:

            degradation_ratio = None

            degradation_percent = None

        # ----------------------------------------------------
        # Determine status
        # ----------------------------------------------------

        (
            monitoring_status,
            monitoring_reason,
        ) = determine_monitoring_status(
            horizon=horizon,
            prediction_count=(
                production_metrics[
                    "prediction_count"
                ]
            ),
            degradation_ratio=(
                degradation_ratio
            ),
        )

        monitoring_rows.append(
            {
                "monitoring_timestamp": (
                    pd.Timestamp.now(
                        tz="UTC"
                    )
                ),
                "horizon_hours": horizon,
                "registered_model": (
                    champion_information[
                        "registered_model"
                    ]
                ),
                "champion_version": (
                    champion_information[
                        "champion_version"
                    ]
                ),
                "champion_run_id": (
                    champion_information[
                        "champion_run_id"
                    ]
                ),
                "source_model": (
                    champion_information[
                        "source_model"
                    ]
                ),
                "prediction_count": (
                    production_metrics[
                        "prediction_count"
                    ]
                ),
                "minimum_predictions_required": (
                    MINIMUM_EVALUATED_PREDICTIONS[
                        horizon
                    ]
                ),
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "production_mae": (
                    production_mae
                ),
                "production_rmse": (
                    production_metrics[
                        "production_rmse"
                    ]
                ),
                "production_mean_error": (
                    production_metrics[
                        "production_mean_error"
                    ]
                ),
                "mae_degradation_ratio": (
                    degradation_ratio
                ),
                "mae_degradation_percent": (
                    degradation_percent
                ),
                "monitoring_status": (
                    monitoring_status
                ),
                "monitoring_reason": (
                    monitoring_reason
                ),
                "first_target_timestamp": (
                    production_metrics[
                        "first_target_timestamp"
                    ]
                ),
                "latest_target_timestamp": (
                    production_metrics[
                        "latest_target_timestamp"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        monitoring_rows
    )


# ============================================================
# 12. SAVE MONITORING SUMMARY
# ============================================================

def save_monitoring_summary(
    monitoring_summary: pd.DataFrame,
) -> None:
    """
    Save current monitoring state as CSV and JSON.
    """

    PREDICTION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    monitoring_summary.to_csv(
        MONITORING_OUTPUT_PATH,
        index=False,
    )

    json_records = (
        monitoring_summary
        .copy()
    )

    datetime_columns = [
        "monitoring_timestamp",
        "first_target_timestamp",
        "latest_target_timestamp",
    ]

    for column in datetime_columns:

        if column in json_records.columns:

            json_records[column] = (
                json_records[column]
                .astype(str)
            )

    with open(
        MONITORING_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            json_records.to_dict(
                orient="records"
            ),
            json_file,
            indent=2,
            default=str,
        )


# ============================================================
# 13. LOG MONITORING RUN TO MLFLOW
# ============================================================

def log_monitoring_to_mlflow(
    monitoring_summary: pd.DataFrame,
) -> str:
    """
    Log one production monitoring execution as an
    MLflow run.
    """

    monitoring_time = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    run_name = (
        "production_monitor_"
        + monitoring_time.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # ----------------------------------------------------
        # Monitoring configuration
        # ----------------------------------------------------

        mlflow.log_param(
            "monitoring_type",
            "delayed_ground_truth",
        )

        mlflow.log_param(
            "warning_degradation_ratio",
            WARNING_DEGRADATION_RATIO,
        )

        mlflow.log_param(
            "critical_degradation_ratio",
            CRITICAL_DEGRADATION_RATIO,
        )

        mlflow.log_param(
            "minimum_6h_predictions",
            MINIMUM_EVALUATED_PREDICTIONS[6],
        )

        mlflow.log_param(
            "minimum_24h_predictions",
            MINIMUM_EVALUATED_PREDICTIONS[24],
        )

        mlflow.log_param(
            "minimum_72h_predictions",
            MINIMUM_EVALUATED_PREDICTIONS[72],
        )

        # ----------------------------------------------------
        # Log horizon metrics
        # ----------------------------------------------------

        for _, row in (
            monitoring_summary.iterrows()
        ):

            horizon = int(
                row["horizon_hours"]
            )

            mlflow.log_metric(
                f"prediction_count_{horizon}h",
                float(
                    row["prediction_count"]
                ),
            )

            if pd.notna(
                row["test_mae"]
            ):

                mlflow.log_metric(
                    f"test_mae_{horizon}h",
                    float(
                        row["test_mae"]
                    ),
                )

            if pd.notna(
                row["test_rmse"]
            ):

                mlflow.log_metric(
                    f"test_rmse_{horizon}h",
                    float(
                        row["test_rmse"]
                    ),
                )

            if pd.notna(
                row["production_mae"]
            ):

                mlflow.log_metric(
                    f"production_mae_{horizon}h",
                    float(
                        row["production_mae"]
                    ),
                )

            if pd.notna(
                row["production_rmse"]
            ):

                mlflow.log_metric(
                    f"production_rmse_{horizon}h",
                    float(
                        row["production_rmse"]
                    ),
                )

            if pd.notna(
                row["production_mean_error"]
            ):

                mlflow.log_metric(
                    (
                        "production_mean_error_"
                        f"{horizon}h"
                    ),
                    float(
                        row[
                            "production_mean_error"
                        ]
                    ),
                )

            if pd.notna(
                row["mae_degradation_ratio"]
            ):

                mlflow.log_metric(
                    (
                        "mae_degradation_ratio_"
                        f"{horizon}h"
                    ),
                    float(
                        row[
                            "mae_degradation_ratio"
                        ]
                    ),
                )

            if pd.notna(
                row["mae_degradation_percent"]
            ):

                mlflow.log_metric(
                    (
                        "mae_degradation_percent_"
                        f"{horizon}h"
                    ),
                    float(
                        row[
                            "mae_degradation_percent"
                        ]
                    ),
                )

            # ------------------------------------------------
            # Metadata as tags
            # ------------------------------------------------

            mlflow.set_tag(
                f"status_{horizon}h",
                str(
                    row["monitoring_status"]
                ),
            )

            mlflow.set_tag(
                f"champion_model_{horizon}h",
                str(
                    row["registered_model"]
                ),
            )

            mlflow.set_tag(
                f"champion_version_{horizon}h",
                str(
                    row["champion_version"]
                ),
            )

            mlflow.set_tag(
                f"source_model_{horizon}h",
                str(
                    row["source_model"]
                ),
            )

        # ----------------------------------------------------
        # Overall status
        # ----------------------------------------------------

        statuses = set(
            monitoring_summary[
                "monitoring_status"
            ]
            .astype(str)
        )

        if "CRITICAL" in statuses:

            overall_status = "CRITICAL"

        elif "WARNING" in statuses:

            overall_status = "WARNING"

        elif "INSUFFICIENT_DATA" in statuses:

            overall_status = (
                "INSUFFICIENT_DATA"
            )

        elif "BASELINE_UNAVAILABLE" in statuses:

            overall_status = (
                "BASELINE_UNAVAILABLE"
            )

        else:

            overall_status = "HEALTHY"

        mlflow.set_tag(
            "overall_monitoring_status",
            overall_status,
        )

        # ----------------------------------------------------
        # Log monitoring artifacts
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(
                MONITORING_OUTPUT_PATH
            ),
            artifact_path="monitoring",
        )

        mlflow.log_artifact(
            str(
                MONITORING_JSON_PATH
            ),
            artifact_path="monitoring",
        )

        if EVALUATION_LOG_PATH.exists():

            mlflow.log_artifact(
                str(
                    EVALUATION_LOG_PATH
                ),
                artifact_path=(
                    "monitoring"
                ),
            )

        if PERFORMANCE_SUMMARY_PATH.exists():

            mlflow.log_artifact(
                str(
                    PERFORMANCE_SUMMARY_PATH
                ),
                artifact_path=(
                    "monitoring"
                ),
            )

        run_id = run.info.run_id

    return run_id


# ============================================================
# 14. PRINT MONITORING STATUS
# ============================================================

def print_monitoring_summary(
    monitoring_summary: pd.DataFrame,
) -> None:
    """
    Print current production monitoring status.
    """

    print()

    print("=" * 100)

    print(
        "PRODUCTION MODEL MONITORING"
    )

    print("=" * 100)

    for _, row in (
        monitoring_summary.iterrows()
    ):

        horizon = int(
            row["horizon_hours"]
        )

        print()

        print(
            f"{horizon} HOUR CHAMPION"
        )

        print("-" * 70)

        print(
            "Registered model :",
            row["registered_model"],
        )

        print(
            "Champion version :",
            f"v{row['champion_version']}",
        )

        print(
            "Source model     :",
            row["source_model"],
        )

        print(
            "Evaluated count  :",
            row["prediction_count"],
        )

        print(
            "Minimum required :",
            row[
                "minimum_predictions_required"
            ],
        )

        if pd.notna(
            row["test_mae"]
        ):

            print(
                "Original test MAE:",
                f"{row['test_mae']:.6f}",
            )

        else:

            print(
                "Original test MAE:",
                "UNAVAILABLE",
            )

        if pd.notna(
            row["production_mae"]
        ):

            print(
                "Production MAE  :",
                f"{row['production_mae']:.6f}",
            )

        else:

            print(
                "Production MAE  :",
                "UNAVAILABLE",
            )

        if pd.notna(
            row["mae_degradation_ratio"]
        ):

            print(
                "MAE ratio       :",
                (
                    f"{row['mae_degradation_ratio']:.2f}x"
                ),
            )

        else:

            print(
                "MAE ratio       :",
                "UNAVAILABLE",
            )

        print(
            "Status           :",
            row["monitoring_status"],
        )

        print(
            "Reason           :",
            row["monitoring_reason"],
        )

    print()

    print("=" * 100)


# ============================================================
# 15. MAIN
# ============================================================

def main() -> None:
    """
    Run production performance monitoring and log
    results to MLflow.
    """

    print("=" * 100)

    print(
        "FLOOD FORECAST — "
        "PRODUCTION MODEL MONITORING"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    client = configure_mlflow()

    # --------------------------------------------------------
    # Load delayed evaluations
    # --------------------------------------------------------

    evaluations = (
        load_evaluation_log()
    )

    print()

    print(
        "Evaluated production predictions:",
        len(evaluations),
    )

    print(
        "Evaluation time range:",
        evaluations[
            "target_timestamp"
        ].min(),
        "to",
        evaluations[
            "target_timestamp"
        ].max(),
    )

    # --------------------------------------------------------
    # Build monitoring state
    # --------------------------------------------------------

    monitoring_summary = (
        build_monitoring_summary(
            evaluations=evaluations,
            client=client,
        )
    )

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    save_monitoring_summary(
        monitoring_summary
    )

    # --------------------------------------------------------
    # Display monitoring state
    # --------------------------------------------------------

    print_monitoring_summary(
        monitoring_summary
    )

    # --------------------------------------------------------
    # Log monitoring execution into MLflow
    # --------------------------------------------------------

    run_id = (
        log_monitoring_to_mlflow(
            monitoring_summary
        )
    )

    print()

    print(
        "Monitoring summary:"
    )

    print(
        MONITORING_OUTPUT_PATH
    )

    print()

    print(
        "MLflow monitoring run ID:"
    )

    print(
        run_id
    )

    print()

    print(
        "MLflow experiment:"
    )

    print(
        MONITORING_EXPERIMENT
    )


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()

        print("=" * 100)

        print(
            "PRODUCTION MONITORING FAILED"
        )

        print("=" * 100)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 100)

        raise