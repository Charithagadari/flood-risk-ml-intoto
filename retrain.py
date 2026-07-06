from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow
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


TRAINING_NOTEBOOK = (
    PROJECT_ROOT
    / "notebooks"
    / "forecasting_model_mlflow_registry.ipynb"
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


MONITORING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "predictions"
    / "monitoring_summary.csv"
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


RETRAINING_EXPERIMENT = (
    "flood_risk_retraining"
)


MODEL_NAMES = {
    6: "flood_forecast_6h",
    24: "flood_forecast_24h",
    72: "flood_forecast_72h",
}


# ============================================================
# 3. CONFIGURE MLFLOW
# ============================================================

def configure_mlflow() -> MlflowClient:
    """
    Configure MLflow.
    """

    if not MLFLOW_DB.exists():

        raise FileNotFoundError(
            "MLflow database was not found:\n"
            f"{MLFLOW_DB}"
        )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        RETRAINING_EXPERIMENT
    )

    return MlflowClient()


# ============================================================
# 4. LOAD MONITORING STATE
# ============================================================

def load_monitoring_summary() -> pd.DataFrame:
    """
    Load the most recent production monitoring state.
    """

    if not MONITORING_SUMMARY_PATH.exists():

        raise FileNotFoundError(
            "Monitoring summary was not found.\n\n"
            f"Expected:\n"
            f"{MONITORING_SUMMARY_PATH}\n\n"
            "Run monitor_production.py first."
        )

    monitoring = pd.read_csv(
        MONITORING_SUMMARY_PATH
    )

    required_columns = [
        "horizon_hours",
        "monitoring_status",
        "prediction_count",
        "mae_degradation_ratio",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in monitoring.columns
    ]

    if missing_columns:

        raise ValueError(
            "Monitoring summary is missing columns:\n"
            f"{missing_columns}"
        )

    return monitoring


# ============================================================
# 5. DETERMINE WHETHER RETRAINING IS REQUIRED
# ============================================================

def determine_retraining_horizons(
    monitoring: pd.DataFrame,
    force: bool = False,
) -> list[int]:
    """
    Return horizons requiring candidate retraining.
    """

    if force:

        return sorted(
            MODEL_NAMES.keys()
        )

    retraining_statuses = {
        "WARNING",
        "CRITICAL",
    }

    triggered_rows = (
        monitoring[
            monitoring[
                "monitoring_status"
            ]
            .astype(str)
            .str.upper()
            .isin(
                retraining_statuses
            )
        ]
    )

    horizons = (
        triggered_rows[
            "horizon_hours"
        ]
        .astype(int)
        .tolist()
    )

    return sorted(
        set(horizons)
    )


# ============================================================
# 6. CAPTURE CURRENT CHAMPIONS
# ============================================================

def get_current_champions(
    client: MlflowClient,
) -> dict[int, dict]:
    """
    Store champion versions before retraining.
    """

    champions = {}

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

        champions[horizon] = {
            "registered_model": model_name,
            "version": int(
                champion.version
            ),
            "run_id": champion.run_id,
        }

    return champions


# ============================================================
# 7. EXECUTE TRAINING NOTEBOOK
# ============================================================

def execute_training_notebook() -> None:
    """
    Execute the existing training notebook.
    """

    if not TRAINING_NOTEBOOK.exists():

        raise FileNotFoundError(
            "Training notebook was not found:\n"
            f"{TRAINING_NOTEBOOK}"
        )

    print()

    print("=" * 80)

    print(
        "EXECUTING FORECAST TRAINING PIPELINE"
    )

    print("=" * 80)

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(TRAINING_NOTEBOOK),
        "--output",
        str(TRAINING_NOTEBOOK.name),
        "--output-dir",
        str(TRAINING_NOTEBOOK.parent),
        "--ExecutePreprocessor.timeout=-1",
    ]

    print(
        "Command:"
    )

    print(
        " ".join(command)
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


# ============================================================
# 8. FIND NEW MODEL VERSIONS
# ============================================================

def find_new_versions(
    client: MlflowClient,
    old_champions: dict[int, dict],
) -> dict[int, dict]:
    """
    Find newest registry versions created by retraining.
    """

    new_versions = {}

    for (
        horizon,
        model_name,
    ) in MODEL_NAMES.items():

        model_versions = (
            client.search_model_versions(
                f"name='{model_name}'"
            )
        )

        if not model_versions:

            raise ValueError(
                "No model versions found for "
                f"{model_name}"
            )

        newest_version = max(
            model_versions,
            key=lambda item: int(
                item.version
            ),
        )

        old_version = int(
            old_champions[
                horizon
            ]["version"]
        )

        newest_version_number = int(
            newest_version.version
        )

        if (
            newest_version_number
            <= old_version
        ):

            print()

            print(
                f"{horizon}h: no new model version "
                "was created."
            )

            continue

        new_versions[horizon] = {
            "registered_model": model_name,
            "version": (
                newest_version_number
            ),
            "run_id": (
                newest_version.run_id
            ),
        }

    return new_versions


# ============================================================
# 9. SET CANDIDATE ALIASES
# ============================================================

def assign_candidate_aliases(
    client: MlflowClient,
    new_versions: dict[int, dict],
) -> None:
    """
    Point candidate aliases at newly trained versions.
    """

    for (
        horizon,
        version_data,
    ) in new_versions.items():

        model_name = version_data[
            "registered_model"
        ]

        version = version_data[
            "version"
        ]

        client.set_registered_model_alias(
            name=model_name,
            alias="candidate",
            version=version,
        )

        client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key="lifecycle_role",
            value="candidate",
        )

        print(
            f"{horizon}h candidate -> "
            f"{model_name} v{version}"
        )


# ============================================================
# 10. SAVE RETRAINING RESULT
# ============================================================

def save_retraining_result(
    triggered_horizons: list[int],
    old_champions: dict[int, dict],
    new_versions: dict[int, dict],
) -> None:
    """
    Persist retraining metadata for the promotion gate.
    """

    RETRAINING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "retraining_timestamp": str(
            pd.Timestamp.now(
                tz="UTC"
            )
        ),
        "triggered_horizons": (
            triggered_horizons
        ),
        "old_champions": (
            old_champions
        ),
        "new_candidates": (
            new_versions
        ),
    }

    with open(
        RETRAINING_RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as result_file:

        json.dump(
            result,
            result_file,
            indent=2,
        )

    print()

    print(
        "Retraining result saved:"
    )

    print(
        RETRAINING_RESULT_PATH
    )


# ============================================================
# 11. LOG RETRAINING CONTROL RUN
# ============================================================

def log_retraining_run(
    triggered_horizons: list[int],
    old_champions: dict[int, dict],
    new_versions: dict[int, dict],
) -> str:
    """
    Log one retraining-control run to MLflow.
    """

    timestamp = pd.Timestamp.now(
        tz="UTC"
    )

    run_name = (
        "retraining_"
        + timestamp.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_param(
            "trigger_type",
            "production_performance",
        )

        mlflow.log_param(
            "triggered_horizons",
            ",".join(
                str(value)
                for value
                in triggered_horizons
            ),
        )

        mlflow.log_metric(
            "triggered_horizon_count",
            float(
                len(triggered_horizons)
            ),
        )

        for horizon in MODEL_NAMES:

            old_version = (
                old_champions[
                    horizon
                ]["version"]
            )

            mlflow.log_param(
                f"old_champion_version_{horizon}h",
                old_version,
            )

            if horizon in new_versions:

                mlflow.log_param(
                    f"candidate_version_{horizon}h",
                    new_versions[
                        horizon
                    ]["version"],
                )

        mlflow.log_artifact(
            str(
                RETRAINING_RESULT_PATH
            ),
            artifact_path="retraining",
        )

        return run.info.run_id


# ============================================================
# 12. MAIN
# ============================================================

def main(
    force: bool = False,
) -> None:
    """
    Run retraining control flow.
    """

    print("=" * 80)

    print(
        "FLOOD FORECAST — RETRAINING"
    )

    print("=" * 80)

    client = configure_mlflow()

    monitoring = (
        load_monitoring_summary()
    )

    triggered_horizons = (
        determine_retraining_horizons(
            monitoring=monitoring,
            force=force,
        )
    )

    print()

    print(
        "Retraining horizons:",
        triggered_horizons,
    )

    if not triggered_horizons:

        print()

        print(
            "No WARNING or CRITICAL horizons."
        )

        print(
            "Retraining was not triggered."
        )

        return

    old_champions = (
        get_current_champions(
            client
        )
    )

    print()

    print(
        "Current champions:"
    )

    for (
        horizon,
        champion,
    ) in old_champions.items():

        print(
            f"{horizon}h -> "
            f"{champion['registered_model']} "
            f"v{champion['version']}"
        )

    execute_training_notebook()

    new_versions = (
        find_new_versions(
            client=client,
            old_champions=(
                old_champions
            ),
        )
    )

    if not new_versions:

        print()

        print(
            "Training completed, but no new "
            "registry versions were found."
        )

        return

    assign_candidate_aliases(
        client=client,
        new_versions=new_versions,
    )

    save_retraining_result(
        triggered_horizons=(
            triggered_horizons
        ),
        old_champions=old_champions,
        new_versions=new_versions,
    )

    run_id = log_retraining_run(
        triggered_horizons=(
            triggered_horizons
        ),
        old_champions=old_champions,
        new_versions=new_versions,
    )

    print()

    print("=" * 80)

    print(
        "RETRAINING COMPLETE"
    )

    print("=" * 80)

    print(
        "Candidate models created."
    )

    print(
        "Champion aliases have NOT been changed."
    )

    print(
        "Retraining MLflow run:"
    )

    print(
        run_id
    )

    print()

    print(
        "Next command:"
    )

    print(
        "python promote_candidate.py"
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    force_retraining = (
        "--force"
        in sys.argv
    )

    try:

        main(
            force=force_retraining
        )

    except Exception as error:

        print()

        print("=" * 80)

        print(
            "RETRAINING FAILED"
        )

        print("=" * 80)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 80)

        raise