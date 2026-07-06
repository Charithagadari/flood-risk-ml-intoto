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


RETRAINING_DIRECTORY = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "retraining"
)


RETRAINING_RESULT_PATH = (
    RETRAINING_DIRECTORY
    / "retraining_result.json"
)


PROMOTION_SUMMARY_PATH = (
    RETRAINING_DIRECTORY
    / "promotion_summary.csv"
)


PROMOTION_JSON_PATH = (
    RETRAINING_DIRECTORY
    / "promotion_summary.json"
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


PROMOTION_EXPERIMENT = (
    "flood_risk_model_promotion"
)


# ============================================================
# 3. REGISTERED MODELS
# ============================================================

MODEL_NAMES = {
    6: "flood_forecast_6h",
    24: "flood_forecast_24h",
    72: "flood_forecast_72h",
}


# ============================================================
# 4. QUALITY-GATE CONFIGURATION
# ============================================================

# Candidate must improve MAE by at least 5%.
#
# Example:
#
# champion MAE = 0.020
# candidate MAE = 0.018
#
# improvement:
#
# (0.020 - 0.018) / 0.020
# = 0.10
# = 10%
#
# 10% > 5% -> passes MAE gate.

MINIMUM_MAE_IMPROVEMENT_RATIO = 0.05


# Candidate RMSE may be slightly worse even when MAE improves.
#
# Allow at most a 5% RMSE regression.
#
# candidate_rmse / champion_rmse <= 1.05

MAXIMUM_RMSE_REGRESSION_RATIO = 1.05


# Require a valid MLflow model signature.

REQUIRE_MODEL_SIGNATURE = True


# ============================================================
# 5. CONFIGURE MLFLOW
# ============================================================

def configure_mlflow() -> MlflowClient:
    """
    Configure the same MLflow backend used by training,
    registry, monitoring, and live inference.
    """

    if not MLFLOW_DB.exists():

        raise FileNotFoundError(
            "MLflow SQLite database was not found.\n\n"
            f"Expected:\n"
            f"{MLFLOW_DB}"
        )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        PROMOTION_EXPERIMENT
    )

    client = MlflowClient()

    print(
        "MLflow tracking URI:"
    )

    print(
        mlflow.get_tracking_uri()
    )

    print()

    print(
        "Promotion experiment:"
    )

    print(
        PROMOTION_EXPERIMENT
    )

    return client


# ============================================================
# 6. LOAD RETRAINING RESULT
# ============================================================

def load_retraining_result() -> dict[str, Any]:
    """
    Load candidate versions created by retrain.py.
    """

    if not RETRAINING_RESULT_PATH.exists():

        raise FileNotFoundError(
            "Retraining result was not found.\n\n"
            f"Expected:\n"
            f"{RETRAINING_RESULT_PATH}\n\n"
            "Run retrain.py --force or wait for a "
            "monitoring-triggered retraining run."
        )

    with open(
        RETRAINING_RESULT_PATH,
        "r",
        encoding="utf-8",
    ) as result_file:

        retraining_result = json.load(
            result_file
        )

    required_keys = [
        "retraining_timestamp",
        "triggered_horizons",
        "old_champions",
        "new_candidates",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in retraining_result
    ]

    if missing_keys:

        raise ValueError(
            "Retraining result is missing keys:\n"
            f"{missing_keys}"
        )

    return retraining_result


# ============================================================
# 7. NORMALIZE JSON HORIZON KEYS
# ============================================================

def normalize_horizon_mapping(
    mapping: dict[str, Any],
) -> dict[int, Any]:
    """
    JSON converts integer dictionary keys into strings.

    Convert:
        "6" -> 6
        "24" -> 24
        "72" -> 72
    """

    normalized_mapping = {}

    for (
        horizon,
        value,
    ) in mapping.items():

        normalized_mapping[
            int(horizon)
        ] = value

    return normalized_mapping


# ============================================================
# 8. GET MODEL VERSION INFORMATION
# ============================================================

def get_model_version_information(
    client: MlflowClient,
    model_name: str,
    version: int,
) -> dict[str, Any]:
    """
    Read one registered model version and its source run.
    """

    model_version = (
        client.get_model_version(
            name=model_name,
            version=str(version),
        )
    )

    if model_version.run_id is None:

        raise ValueError(
            f"{model_name} v{version} "
            "does not have a source run ID."
        )

    source_run = client.get_run(
        model_version.run_id
    )

    tags = (
        model_version.tags
        or {}
    )

    metrics = (
        source_run.data.metrics
        or {}
    )

    params = (
        source_run.data.params
        or {}
    )

    return {
        "registered_model": model_name,
        "version": int(version),
        "run_id": model_version.run_id,
        "source_model": tags.get(
            "source_model",
            "",
        ),
        "model_version_tags": tags,
        "run_metrics": metrics,
        "run_params": params,
    }


# ============================================================
# 9. FIND A RUN METRIC
# ============================================================

def find_run_metric(
    model_information: dict[str, Any],
    metric_name: str,
) -> float | None:
    """
    Resolve MAE or RMSE from an MLflow source run.

    Handles metric naming differences such as:
        mae
        MAE
        test_mae
        mae_test
        mae_6h
    """

    run_metrics = model_information[
        "run_metrics"
    ]

    metric_lower = (
        metric_name.lower()
    )

    metric_candidates = [
        metric_lower,
        metric_lower.upper(),
        f"test_{metric_lower}",
        f"{metric_lower}_test",
    ]

    # --------------------------------------------------------
    # Exact matches
    # --------------------------------------------------------

    for candidate in metric_candidates:

        for (
            logged_name,
            logged_value,
        ) in run_metrics.items():

            if (
                logged_name.lower()
                == candidate.lower()
            ):

                return float(
                    logged_value
                )

    # --------------------------------------------------------
    # Partial matches
    # --------------------------------------------------------

    partial_matches = []

    for (
        logged_name,
        logged_value,
    ) in run_metrics.items():

        if (
            metric_lower
            in logged_name.lower()
        ):

            partial_matches.append(
                (
                    logged_name,
                    logged_value,
                )
            )

    if len(partial_matches) == 1:

        return float(
            partial_matches[0][1]
        )

    # --------------------------------------------------------
    # Registry selection metric fallback
    # --------------------------------------------------------

    version_tags = model_information[
        "model_version_tags"
    ]

    selection_metric = str(
        version_tags.get(
            "selection_metric",
            "",
        )
    ).lower()

    selection_metric_value = (
        version_tags.get(
            "selection_metric_value"
        )
    )

    if (
        selection_metric == metric_lower
        and selection_metric_value is not None
    ):

        try:

            return float(
                selection_metric_value
            )

        except (
            TypeError,
            ValueError,
        ):

            pass

    return None


# ============================================================
# 10. MODEL LOAD AND SIGNATURE CHECK
# ============================================================

def validate_model_load_and_signature(
    model_name: str,
    version: int,
) -> dict[str, Any]:
    """
    Load a concrete registered model version and verify
    that it has a usable model signature.
    """

    model_uri = (
        f"models:/"
        f"{model_name}/"
        f"{version}"
    )

    validation_result = {
        "model_uri": model_uri,
        "model_load_passed": False,
        "signature_passed": False,
        "input_feature_count": None,
        "validation_error": None,
    }

    try:

        model = mlflow.pyfunc.load_model(
            model_uri
        )

        validation_result[
            "model_load_passed"
        ] = True

        signature = (
            model.metadata.signature
        )

        if signature is None:

            if REQUIRE_MODEL_SIGNATURE:

                validation_result[
                    "validation_error"
                ] = (
                    "Model signature is missing."
                )

                return validation_result

            validation_result[
                "signature_passed"
            ] = True

            return validation_result

        if signature.inputs is None:

            if REQUIRE_MODEL_SIGNATURE:

                validation_result[
                    "validation_error"
                ] = (
                    "Model input signature is missing."
                )

                return validation_result

            validation_result[
                "signature_passed"
            ] = True

            return validation_result

        input_names = [
            input_item.name
            for input_item
            in signature.inputs.inputs
            if input_item.name is not None
        ]

        validation_result[
            "input_feature_count"
        ] = len(
            input_names
        )

        if (
            REQUIRE_MODEL_SIGNATURE
            and not input_names
        ):

            validation_result[
                "validation_error"
            ] = (
                "Model signature contains "
                "no named input features."
            )

            return validation_result

        validation_result[
            "signature_passed"
        ] = True

        return validation_result

    except Exception as error:

        validation_result[
            "validation_error"
        ] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return validation_result


# ============================================================
# 11. SIGNATURE COMPATIBILITY CHECK
# ============================================================

def get_signature_columns(
    model_name: str,
    version: int,
) -> list[str]:
    """
    Load model and return named signature input columns.
    """

    model_uri = (
        f"models:/"
        f"{model_name}/"
        f"{version}"
    )

    model = mlflow.pyfunc.load_model(
        model_uri
    )

    signature = (
        model.metadata.signature
    )

    if (
        signature is None
        or signature.inputs is None
    ):

        return []

    return [
        input_item.name
        for input_item
        in signature.inputs.inputs
        if input_item.name is not None
    ]


def check_signature_compatibility(
    model_name: str,
    champion_version: int,
    candidate_version: int,
) -> tuple[
    bool,
    str,
]:
    """
    Verify candidate expects the same model input columns
    as the current champion.

    This prevents silently promoting a model requiring a
    different production feature schema.
    """

    champion_columns = (
        get_signature_columns(
            model_name=model_name,
            version=champion_version,
        )
    )

    candidate_columns = (
        get_signature_columns(
            model_name=model_name,
            version=candidate_version,
        )
    )

    if not champion_columns:

        return (
            False,
            "Champion has no named input signature.",
        )

    if not candidate_columns:

        return (
            False,
            "Candidate has no named input signature.",
        )

    if (
        champion_columns
        != candidate_columns
    ):

        champion_set = set(
            champion_columns
        )

        candidate_set = set(
            candidate_columns
        )

        missing_from_candidate = sorted(
            champion_set
            - candidate_set
        )

        new_candidate_features = sorted(
            candidate_set
            - champion_set
        )

        reason = (
            "Feature signatures differ. "
            f"Missing from candidate: "
            f"{missing_from_candidate}. "
            f"New candidate features: "
            f"{new_candidate_features}."
        )

        return (
            False,
            reason,
        )

    return (
        True,
        (
            "Candidate input signature matches "
            "the champion."
        ),
    )


# ============================================================
# 12. CALCULATE QUALITY METRICS
# ============================================================

def calculate_quality_metrics(
    champion_information: dict[str, Any],
    candidate_information: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare offline candidate and champion metrics.
    """

    champion_mae = find_run_metric(
        model_information=(
            champion_information
        ),
        metric_name="mae",
    )

    candidate_mae = find_run_metric(
        model_information=(
            candidate_information
        ),
        metric_name="mae",
    )

    champion_rmse = find_run_metric(
        model_information=(
            champion_information
        ),
        metric_name="rmse",
    )

    candidate_rmse = find_run_metric(
        model_information=(
            candidate_information
        ),
        metric_name="rmse",
    )

    # --------------------------------------------------------
    # MAE improvement
    # --------------------------------------------------------

    if (
        champion_mae is not None
        and candidate_mae is not None
        and champion_mae > 0
    ):

        mae_improvement_ratio = float(
            (
                champion_mae
                - candidate_mae
            )
            / champion_mae
        )

        mae_improvement_percent = float(
            mae_improvement_ratio
            * 100
        )

    else:

        mae_improvement_ratio = None

        mae_improvement_percent = None

    # --------------------------------------------------------
    # RMSE ratio
    # --------------------------------------------------------

    if (
        champion_rmse is not None
        and candidate_rmse is not None
        and champion_rmse > 0
    ):

        rmse_candidate_to_champion_ratio = float(
            candidate_rmse
            / champion_rmse
        )

    else:

        rmse_candidate_to_champion_ratio = None

    return {
        "champion_mae": champion_mae,
        "candidate_mae": candidate_mae,
        "mae_improvement_ratio": (
            mae_improvement_ratio
        ),
        "mae_improvement_percent": (
            mae_improvement_percent
        ),
        "champion_rmse": champion_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_candidate_to_champion_ratio": (
            rmse_candidate_to_champion_ratio
        ),
    }


# ============================================================
# 13. APPLY PROMOTION QUALITY GATES
# ============================================================

def apply_quality_gates(
    quality_metrics: dict[str, Any],
    candidate_validation: dict[str, Any],
    signature_compatible: bool,
    signature_reason: str,
) -> dict[str, Any]:
    """
    Apply promotion quality gates.

    All required gates must pass.
    """

    gate_results = {}

    # --------------------------------------------------------
    # Gate 1: candidate model can load
    # --------------------------------------------------------

    gate_results[
        "model_load_gate"
    ] = bool(
        candidate_validation[
            "model_load_passed"
        ]
    )

    # --------------------------------------------------------
    # Gate 2: model signature exists
    # --------------------------------------------------------

    gate_results[
        "signature_gate"
    ] = bool(
        candidate_validation[
            "signature_passed"
        ]
    )

    # --------------------------------------------------------
    # Gate 3: serving schema compatibility
    # --------------------------------------------------------

    gate_results[
        "signature_compatibility_gate"
    ] = bool(
        signature_compatible
    )

    # --------------------------------------------------------
    # Gate 4: MAE improvement
    # --------------------------------------------------------

    mae_improvement_ratio = (
        quality_metrics[
            "mae_improvement_ratio"
        ]
    )

    if mae_improvement_ratio is None:

        gate_results[
            "mae_improvement_gate"
        ] = False

    else:

        gate_results[
            "mae_improvement_gate"
        ] = (
            mae_improvement_ratio
            >= MINIMUM_MAE_IMPROVEMENT_RATIO
        )

    # --------------------------------------------------------
    # Gate 5: RMSE regression
    # --------------------------------------------------------

    rmse_ratio = quality_metrics[
        "rmse_candidate_to_champion_ratio"
    ]

    if rmse_ratio is None:

        gate_results[
            "rmse_gate"
        ] = False

    else:

        gate_results[
            "rmse_gate"
        ] = (
            rmse_ratio
            <= MAXIMUM_RMSE_REGRESSION_RATIO
        )

    all_gates_passed = all(
        gate_results.values()
    )

    failed_gates = [
        gate_name
        for (
            gate_name,
            gate_passed,
        ) in gate_results.items()
        if not gate_passed
    ]

    reasons = []

    if not gate_results[
        "model_load_gate"
    ]:

        reasons.append(
            "Candidate model could not be loaded."
        )

    if not gate_results[
        "signature_gate"
    ]:

        reasons.append(
            candidate_validation.get(
                "validation_error"
            )
            or
            "Candidate signature validation failed."
        )

    if not gate_results[
        "signature_compatibility_gate"
    ]:

        reasons.append(
            signature_reason
        )

    if not gate_results[
        "mae_improvement_gate"
    ]:

        if mae_improvement_ratio is None:

            reasons.append(
                "MAE comparison metric is unavailable."
            )

        else:

            reasons.append(
                "Candidate MAE improvement is "
                f"{mae_improvement_ratio * 100:.2f}%; "
                "minimum required is "
                f"{MINIMUM_MAE_IMPROVEMENT_RATIO * 100:.2f}%."
            )

    if not gate_results[
        "rmse_gate"
    ]:

        if rmse_ratio is None:

            reasons.append(
                "RMSE comparison metric is unavailable."
            )

        else:

            reasons.append(
                "Candidate/champion RMSE ratio is "
                f"{rmse_ratio:.4f}; maximum allowed "
                "is "
                f"{MAXIMUM_RMSE_REGRESSION_RATIO:.4f}."
            )

    if all_gates_passed:

        decision_reason = (
            "Candidate passed all model promotion "
            "quality gates."
        )

    else:

        decision_reason = " ".join(
            reasons
        )

    return {
        **gate_results,
        "all_gates_passed": (
            all_gates_passed
        ),
        "failed_gates": failed_gates,
        "decision_reason": (
            decision_reason
        ),
    }


# ============================================================
# 14. PROMOTE CANDIDATE
# ============================================================

def promote_candidate(
    client: MlflowClient,
    model_name: str,
    old_champion_version: int,
    candidate_version: int,
) -> None:
    """
    Move champion alias to the approved candidate.

    Preserve the prior champion through a previous_champion
    alias for simple rollback.
    """

    # --------------------------------------------------------
    # Save old champion for rollback
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=model_name,
        alias="previous_champion",
        version=str(
            old_champion_version
        ),
    )

    # --------------------------------------------------------
    # Move champion
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(
            candidate_version
        ),
    )

    # --------------------------------------------------------
    # Add lifecycle tags
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=model_name,
        version=str(
            old_champion_version
        ),
        key="lifecycle_role",
        value="previous_champion",
    )

    client.set_model_version_tag(
        name=model_name,
        version=str(
            candidate_version
        ),
        key="lifecycle_role",
        value="champion",
    )

    client.set_model_version_tag(
        name=model_name,
        version=str(
            candidate_version
        ),
        key="promotion_timestamp",
        value=str(
            pd.Timestamp.now(
                tz="UTC"
            )
        ),
    )


# ============================================================
# 15. TAG REJECTED CANDIDATE
# ============================================================

def tag_rejected_candidate(
    client: MlflowClient,
    model_name: str,
    candidate_version: int,
    rejection_reason: str,
) -> None:
    """
    Mark candidate as rejected without deleting it.
    """

    client.set_model_version_tag(
        name=model_name,
        version=str(
            candidate_version
        ),
        key="lifecycle_role",
        value="rejected_candidate",
    )

    client.set_model_version_tag(
        name=model_name,
        version=str(
            candidate_version
        ),
        key="rejection_timestamp",
        value=str(
            pd.Timestamp.now(
                tz="UTC"
            )
        ),
    )

    # Keep the tag reasonably short.

    client.set_model_version_tag(
        name=model_name,
        version=str(
            candidate_version
        ),
        key="rejection_reason",
        value=str(
            rejection_reason
        )[:5000],
    )


# ============================================================
# 16. EVALUATE ONE HORIZON
# ============================================================

def evaluate_candidate_for_horizon(
    client: MlflowClient,
    horizon: int,
    old_champion: dict[str, Any],
    candidate: dict[str, Any],
    apply_promotion: bool,
) -> dict[str, Any]:
    """
    Run the complete candidate quality gate for one horizon.
    """

    model_name = MODEL_NAMES[
        horizon
    ]

    old_champion_version = int(
        old_champion["version"]
    )

    candidate_version = int(
        candidate["version"]
    )

    print()

    print("=" * 80)

    print(
        f"{horizon} HOUR MODEL PROMOTION GATE"
    )

    print("=" * 80)

    print(
        "Registered model :",
        model_name,
    )

    print(
        "Champion version :",
        f"v{old_champion_version}",
    )

    print(
        "Candidate version:",
        f"v{candidate_version}",
    )

    # --------------------------------------------------------
    # Load metadata and metrics
    # --------------------------------------------------------

    champion_information = (
        get_model_version_information(
            client=client,
            model_name=model_name,
            version=old_champion_version,
        )
    )

    candidate_information = (
        get_model_version_information(
            client=client,
            model_name=model_name,
            version=candidate_version,
        )
    )

    quality_metrics = (
        calculate_quality_metrics(
            champion_information=(
                champion_information
            ),
            candidate_information=(
                candidate_information
            ),
        )
    )

    # --------------------------------------------------------
    # Candidate load / signature validation
    # --------------------------------------------------------

    candidate_validation = (
        validate_model_load_and_signature(
            model_name=model_name,
            version=candidate_version,
        )
    )

    # --------------------------------------------------------
    # Champion/candidate serving schema compatibility
    # --------------------------------------------------------

    (
        signature_compatible,
        signature_reason,
    ) = check_signature_compatibility(
        model_name=model_name,
        champion_version=(
            old_champion_version
        ),
        candidate_version=(
            candidate_version
        ),
    )

    # --------------------------------------------------------
    # Quality gates
    # --------------------------------------------------------

    gate_results = apply_quality_gates(
        quality_metrics=quality_metrics,
        candidate_validation=(
            candidate_validation
        ),
        signature_compatible=(
            signature_compatible
        ),
        signature_reason=(
            signature_reason
        ),
    )

    # --------------------------------------------------------
    # Print metric comparison
    # --------------------------------------------------------

    print()

    print(
        "Champion source :",
        champion_information[
            "source_model"
        ],
    )

    print(
        "Candidate source:",
        candidate_information[
            "source_model"
        ],
    )

    print()

    print(
        "Champion MAE    :",
        quality_metrics[
            "champion_mae"
        ],
    )

    print(
        "Candidate MAE   :",
        quality_metrics[
            "candidate_mae"
        ],
    )

    print(
        "MAE improvement :",
        (
            f"{quality_metrics['mae_improvement_percent']:.2f}%"
            if quality_metrics[
                "mae_improvement_percent"
            ]
            is not None
            else "UNAVAILABLE"
        ),
    )

    print()

    print(
        "Champion RMSE   :",
        quality_metrics[
            "champion_rmse"
        ],
    )

    print(
        "Candidate RMSE  :",
        quality_metrics[
            "candidate_rmse"
        ],
    )

    print(
        "RMSE ratio      :",
        (
            f"{quality_metrics['rmse_candidate_to_champion_ratio']:.4f}"
            if quality_metrics[
                "rmse_candidate_to_champion_ratio"
            ]
            is not None
            else "UNAVAILABLE"
        ),
    )

    print()

    print(
        "Model loads     :",
        gate_results[
            "model_load_gate"
        ],
    )

    print(
        "Signature       :",
        gate_results[
            "signature_gate"
        ],
    )

    print(
        "Schema match    :",
        gate_results[
            "signature_compatibility_gate"
        ],
    )

    print(
        "MAE gate        :",
        gate_results[
            "mae_improvement_gate"
        ],
    )

    print(
        "RMSE gate       :",
        gate_results[
            "rmse_gate"
        ],
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if gate_results[
        "all_gates_passed"
    ]:

        decision = "APPROVED"

        if apply_promotion:

            promote_candidate(
                client=client,
                model_name=model_name,
                old_champion_version=(
                    old_champion_version
                ),
                candidate_version=(
                    candidate_version
                ),
            )

            action = "PROMOTED"

        else:

            action = "DRY_RUN_APPROVED"

    else:

        decision = "REJECTED"

        action = "NOT_PROMOTED"

        if apply_promotion:

            tag_rejected_candidate(
                client=client,
                model_name=model_name,
                candidate_version=(
                    candidate_version
                ),
                rejection_reason=(
                    gate_results[
                        "decision_reason"
                    ]
                ),
            )

    print()

    print(
        "Decision         :",
        decision,
    )

    print(
        "Action           :",
        action,
    )

    print(
        "Reason           :",
        gate_results[
            "decision_reason"
        ],
    )

    return {
        "promotion_timestamp": (
            pd.Timestamp.now(
                tz="UTC"
            )
        ),
        "horizon_hours": horizon,
        "registered_model": model_name,
        "old_champion_version": (
            old_champion_version
        ),
        "candidate_version": (
            candidate_version
        ),
        "champion_run_id": (
            champion_information[
                "run_id"
            ]
        ),
        "candidate_run_id": (
            candidate_information[
                "run_id"
            ]
        ),
        "champion_source_model": (
            champion_information[
                "source_model"
            ]
        ),
        "candidate_source_model": (
            candidate_information[
                "source_model"
            ]
        ),
        **quality_metrics,
        "candidate_model_load_passed": (
            candidate_validation[
                "model_load_passed"
            ]
        ),
        "candidate_signature_passed": (
            candidate_validation[
                "signature_passed"
            ]
        ),
        "candidate_input_feature_count": (
            candidate_validation[
                "input_feature_count"
            ]
        ),
        "signature_compatible": (
            signature_compatible
        ),
        "signature_reason": (
            signature_reason
        ),
        **gate_results,
        "decision": decision,
        "action": action,
    }


# ============================================================
# 17. SAVE PROMOTION SUMMARY
# ============================================================

def save_promotion_summary(
    promotion_summary: pd.DataFrame,
) -> None:
    """
    Persist promotion decisions.
    """

    RETRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    promotion_summary.to_csv(
        PROMOTION_SUMMARY_PATH,
        index=False,
    )

    json_data = (
        promotion_summary.copy()
    )

    for column in [
        "promotion_timestamp",
    ]:

        if column in json_data.columns:

            json_data[column] = (
                json_data[column]
                .astype(str)
            )

    with open(
        PROMOTION_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            json_data.to_dict(
                orient="records"
            ),
            json_file,
            indent=2,
            default=str,
        )

    print()

    print(
        "Promotion summary saved:"
    )

    print(
        PROMOTION_SUMMARY_PATH
    )

    print(
        PROMOTION_JSON_PATH
    )


# ============================================================
# 18. LOG PROMOTION GATE TO MLFLOW
# ============================================================

def log_promotion_to_mlflow(
    promotion_summary: pd.DataFrame,
    apply_promotion: bool,
) -> str:
    """
    Log the promotion quality-gate execution.
    """

    timestamp = pd.Timestamp.now(
        tz="UTC"
    )

    run_name = (
        "promotion_gate_"
        + timestamp.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_param(
            "apply_promotion",
            apply_promotion,
        )

        mlflow.log_param(
            "minimum_mae_improvement_ratio",
            MINIMUM_MAE_IMPROVEMENT_RATIO,
        )

        mlflow.log_param(
            "maximum_rmse_regression_ratio",
            MAXIMUM_RMSE_REGRESSION_RATIO,
        )

        mlflow.log_param(
            "require_model_signature",
            REQUIRE_MODEL_SIGNATURE,
        )

        approved_count = int(
            (
                promotion_summary[
                    "decision"
                ]
                == "APPROVED"
            ).sum()
        )

        rejected_count = int(
            (
                promotion_summary[
                    "decision"
                ]
                == "REJECTED"
            ).sum()
        )

        promoted_count = int(
            (
                promotion_summary[
                    "action"
                ]
                == "PROMOTED"
            ).sum()
        )

        mlflow.log_metric(
            "approved_candidate_count",
            float(
                approved_count
            ),
        )

        mlflow.log_metric(
            "rejected_candidate_count",
            float(
                rejected_count
            ),
        )

        mlflow.log_metric(
            "promoted_candidate_count",
            float(
                promoted_count
            ),
        )

        for _, row in (
            promotion_summary.iterrows()
        ):

            horizon = int(
                row["horizon_hours"]
            )

            if pd.notna(
                row["champion_mae"]
            ):

                mlflow.log_metric(
                    f"champion_mae_{horizon}h",
                    float(
                        row["champion_mae"]
                    ),
                )

            if pd.notna(
                row["candidate_mae"]
            ):

                mlflow.log_metric(
                    f"candidate_mae_{horizon}h",
                    float(
                        row["candidate_mae"]
                    ),
                )

            if pd.notna(
                row["mae_improvement_ratio"]
            ):

                mlflow.log_metric(
                    f"mae_improvement_ratio_{horizon}h",
                    float(
                        row[
                            "mae_improvement_ratio"
                        ]
                    ),
                )

            if pd.notna(
                row[
                    "rmse_candidate_to_champion_ratio"
                ]
            ):

                mlflow.log_metric(
                    (
                        "rmse_candidate_to_champion_ratio_"
                        f"{horizon}h"
                    ),
                    float(
                        row[
                            "rmse_candidate_to_champion_ratio"
                        ]
                    ),
                )

            mlflow.set_tag(
                f"decision_{horizon}h",
                str(
                    row["decision"]
                ),
            )

            mlflow.set_tag(
                f"action_{horizon}h",
                str(
                    row["action"]
                ),
            )

            mlflow.set_tag(
                f"candidate_version_{horizon}h",
                str(
                    row["candidate_version"]
                ),
            )

            mlflow.set_tag(
                f"previous_champion_version_{horizon}h",
                str(
                    row[
                        "old_champion_version"
                    ]
                ),
            )

        mlflow.log_artifact(
            str(
                PROMOTION_SUMMARY_PATH
            ),
            artifact_path="promotion",
        )

        mlflow.log_artifact(
            str(
                PROMOTION_JSON_PATH
            ),
            artifact_path="promotion",
        )

        if RETRAINING_RESULT_PATH.exists():

            mlflow.log_artifact(
                str(
                    RETRAINING_RESULT_PATH
                ),
                artifact_path="promotion",
            )

        return run.info.run_id


# ============================================================
# 19. VERIFY FINAL CHAMPION ALIASES
# ============================================================

def verify_champion_aliases(
    client: MlflowClient,
) -> None:
    """
    Print the final champion state after the promotion gate.
    """

    print()

    print("=" * 80)

    print(
        "FINAL CHAMPION STATE"
    )

    print("=" * 80)

    for (
        horizon,
        model_name,
    ) in MODEL_NAMES.items():

        champion = (
            client.get_model_version_by_alias(
                name=model_name,
                alias="champion",
            )
        )

        print(
            f"{horizon}h -> "
            f"{model_name} "
            f"v{champion.version}"
        )


# ============================================================
# 20. MAIN
# ============================================================

def main(
    apply_promotion: bool = False,
) -> None:
    """
    Run candidate/champion model quality gates.

    Default:
        dry-run only

    Use:
        python promote_candidate.py --apply

    to actually move champion aliases.
    """

    print("=" * 80)

    print(
        "FLOOD FORECAST — "
        "CANDIDATE / CHAMPION PROMOTION GATE"
    )

    print("=" * 80)

    if apply_promotion:

        print()

        print(
            "MODE: APPLY PROMOTION"
        )

        print(
            "Approved candidates may become champions."
        )

    else:

        print()

        print(
            "MODE: DRY RUN"
        )

        print(
            "Champion aliases will NOT be modified."
        )

    client = configure_mlflow()

    retraining_result = (
        load_retraining_result()
    )

    old_champions = (
        normalize_horizon_mapping(
            retraining_result[
                "old_champions"
            ]
        )
    )

    new_candidates = (
        normalize_horizon_mapping(
            retraining_result[
                "new_candidates"
            ]
        )
    )

    triggered_horizons = [
        int(horizon)
        for horizon in retraining_result[
            "triggered_horizons"
        ]
    ]

    print()

    print(
        "Retraining timestamp:"
    )

    print(
        retraining_result[
            "retraining_timestamp"
        ]
    )

    print()

    print(
        "Triggered horizons:"
    )

    print(
        triggered_horizons
    )

    print()

    print(
        "Available candidates:"
    )

    print(
        sorted(
            new_candidates.keys()
        )
    )

    promotion_rows = []

    # --------------------------------------------------------
    # Only evaluate triggered horizons that produced candidates
    # --------------------------------------------------------

    for horizon in triggered_horizons:

        if horizon not in MODEL_NAMES:

            print()

            print(
                "Skipping unsupported horizon:",
                horizon,
            )

            continue

        if horizon not in old_champions:

            print()

            print(
                "Skipping horizon because old champion "
                "metadata is missing:",
                horizon,
            )

            continue

        if horizon not in new_candidates:

            print()

            print(
                "Skipping horizon because no new candidate "
                "was created:",
                horizon,
            )

            continue

        promotion_result = (
            evaluate_candidate_for_horizon(
                client=client,
                horizon=horizon,
                old_champion=(
                    old_champions[
                        horizon
                    ]
                ),
                candidate=(
                    new_candidates[
                        horizon
                    ]
                ),
                apply_promotion=(
                    apply_promotion
                ),
            )
        )

        promotion_rows.append(
            promotion_result
        )

    if not promotion_rows:

        print()

        print(
            "No candidate models were available "
            "for promotion evaluation."
        )

        return

    promotion_summary = pd.DataFrame(
        promotion_rows
    )

    save_promotion_summary(
        promotion_summary
    )

    promotion_run_id = (
        log_promotion_to_mlflow(
            promotion_summary=(
                promotion_summary
            ),
            apply_promotion=(
                apply_promotion
            ),
        )
    )

    verify_champion_aliases(
        client
    )

    print()

    print("=" * 80)

    print(
        "PROMOTION GATE COMPLETE"
    )

    print("=" * 80)

    display_columns = [
        "horizon_hours",
        "old_champion_version",
        "candidate_version",
        "champion_mae",
        "candidate_mae",
        "mae_improvement_percent",
        "champion_rmse",
        "candidate_rmse",
        "decision",
        "action",
    ]

    print(
        promotion_summary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "MLflow promotion run ID:"
    )

    print(
        promotion_run_id
    )

    if not apply_promotion:

        print()

        print(
            "This was a DRY RUN."
        )

        print(
            "To apply approved promotions:"
        )

        print()

        print(
            "python promote_candidate.py --apply"
        )


# ============================================================
# 21. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import sys

    apply_changes = (
        "--apply"
        in sys.argv
    )

    try:

        main(
            apply_promotion=apply_changes
        )

    except Exception as error:

        print()

        print("=" * 80)

        print(
            "MODEL PROMOTION FAILED"
        )

        print("=" * 80)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 80)

        raise