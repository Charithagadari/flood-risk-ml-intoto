# Flood Risk Forecasting — Detailed Project Guide

## 1. Purpose of This Document

This document explains how to run, understand, maintain, debug, and demonstrate the flood-risk forecasting MLOps system.

It covers:

- How the full system works
- Project architecture
- End-to-end execution flow
- MLflow configuration
- Docker usage
- Live inference
- Delayed evaluation
- Production monitoring
- Retraining
- Candidate promotion
- Common errors
- Recovery procedures
- Recommended debugging order

---

# 2. System Objective

The system forecasts future water levels using live hydrological observations from NVE.

Forecast horizons:

```text
6 hours
24 hours
72 hours
```

The project is designed as a complete machine-learning production lifecycle rather than a notebook-only experiment.

The main design principle is:

```text
Training code creates models.
MLflow tracks and registers them.
Production code loads only approved champion models.
Predictions are logged.
Future actual values are collected later.
Production performance is monitored.
Bad performance can trigger retraining.
New candidates must pass quality gates before promotion.
```

---

# 3. End-to-End Project Flow

## Stage A — Data Ingestion

The NVE ingestion client requests recent water-level and reservoir-volume data.

Input:

```text
NVE Hydrology API
```

Output:

```text
timestamp
water_level
reservoir_volume
```

The live data is validated before feature generation.

Validation includes:

- Timestamp parsing
- Numeric conversion
- Sorting
- Duplicate timestamp removal
- Forward filling
- Backward filling
- Required-column validation

---

## Stage B — Feature Engineering

The same feature logic must be used during both training and inference.

Feature families include:

### Temporal features

```text
hour
dayofyear
month
dayofweek
is_weekend
```

### Cyclic features

```text
sin_hour
cos_hour
sin_dayofyear
cos_dayofyear
sin_month
cos_month
```

### Lag features

```text
1h
3h
6h
12h
24h
48h
72h
```

### Rolling features

For configured windows:

```text
rolling mean
rolling standard deviation
rolling minimum
rolling maximum
```

### Change features

```text
current - lagged value
```

### Slope features

```text
change / elapsed hours
```

The MLflow model signature is used to determine the exact feature columns required by each deployed model.

This protects production from accidental schema mismatch.

---

## Stage C — Model Training

The training notebook trains and compares candidate models for each horizon.

Typical model families may include:

- Baseline persistence models
- Ridge regression
- XGBoost
- Delta-target models

A delta-target model predicts:

```text
future_level - current_level
```

The final future level is reconstructed as:

```text
predicted_level = current_level + predicted_delta
```

This can improve forecasting when the current level already contains strong information about the future state.

---

## Stage D — MLflow Tracking

MLflow stores:

- Parameters
- Metrics
- Artifacts
- Run metadata
- Model artifacts
- Model signatures
- Registered model versions

The local SQLite backend used by this project is typically:

```text
sqlite:////Users/charithagadari/.mlflow/flood-risk/mlflow.db
```

The portable experiment uses MLflow artifact storage rather than a hard-coded machine-specific artifact path.

---

## Stage E — Model Registry

There is a registered model for every horizon:

```text
flood_forecast_6h
flood_forecast_24h
flood_forecast_72h
```

Important aliases:

```text
champion
candidate
previous_champion
```

Meaning:

### `champion`

The model currently used by production inference.

### `candidate`

A newly retrained model waiting for evaluation.

### `previous_champion`

The previously deployed champion, retained for rollback.

---

# 4. Live Inference Flow

Run:

```bash
python predict_live.py
```

The script performs:

```text
Configure MLflow
    |
    v
Fetch latest NVE data
    |
    v
Validate observations
    |
    v
Detect sampling interval
    |
    v
Build production features
    |
    v
Load champion aliases
    |
    v
Read model signatures
    |
    v
Select exact required features
    |
    v
Generate predictions
    |
    v
Reconstruct delta forecasts if needed
    |
    v
Assign risk level
    |
    v
Append to prediction_log.csv
```

A successful run should show:

- Station ID
- Observation timestamp
- Current water level
- Predicted 6h level
- Predicted 24h level
- Predicted 72h level
- Registry version
- Source model
- Risk level

---

# 5. Prediction Logging

Predictions are saved to:

```text
src/data/predictions/prediction_log.csv
```

Important fields include:

```text
prediction_timestamp
observation_timestamp
target_timestamp
station_id
horizon_hours
current_level
predicted_level
predicted_change
raw_model_prediction
risk_level
registered_model
version
source_model
is_delta_model
model_uri
```

Why `target_timestamp` matters:

A 24-hour forecast generated at:

```text
2026-07-15 09:00 UTC
```

has a target timestamp of:

```text
2026-07-16 09:00 UTC
```

The forecast cannot be evaluated until that time has passed and the actual observation becomes available.

---

# 6. Delayed Ground-Truth Evaluation

Run:

```bash
python evaluate_predictions.py
```

The script:

1. Loads all production predictions.
2. Identifies forecasts whose target timestamps are in the past.
3. Skips predictions already evaluated.
4. Fetches actual NVE observations.
5. Matches each target timestamp with the nearest actual observation.
6. Calculates errors.
7. Saves evaluation results.

Outputs:

```text
src/data/predictions/evaluation_log.csv
src/data/predictions/performance_summary.csv
```

Metrics:

### Error

```text
predicted - actual
```

### Absolute error

```text
abs(predicted - actual)
```

### Squared error

```text
(predicted - actual)^2
```

### MAE

```text
mean absolute error
```

### RMSE

```text
sqrt(mean squared error)
```

---

# 7. Production Monitoring

Run:

```bash
python monitor_production.py
```

The monitoring system evaluates only predictions produced by the current champion model version.

For every horizon it calculates:

```text
production MAE
production RMSE
production mean error
production/test MAE ratio
```

The main degradation metric is:

```text
mae_degradation_ratio = production_mae / original_test_mae
```

Example:

```text
original test MAE = 0.020
production MAE    = 0.040
ratio             = 2.0
```

This means production error is twice the original test error.

---

# 8. Monitoring Status Logic

Possible statuses:

## `INSUFFICIENT_DATA`

There are not yet enough evaluated production predictions.

Configured minimums:

```text
6h  -> 24 evaluated predictions
24h -> 14 evaluated predictions
72h -> 7 evaluated predictions
```

No retraining should occur yet.

---

## `BASELINE_UNAVAILABLE`

The original model test MAE could not be resolved from MLflow.

Possible causes:

- MAE was not logged during training.
- The registered model points to a run without metrics.
- Metric names changed.

---

## `HEALTHY`

Production MAE remains below the configured warning threshold.

---

## `WARNING`

Default condition:

```text
production MAE >= 1.5 × test MAE
```

This can trigger retraining.

---

## `CRITICAL`

Default condition:

```text
production MAE >= 2.0 × test MAE
```

This can trigger retraining.

---

# 9. Retraining Flow

Normal execution:

```bash
python retrain.py
```

The script reads:

```text
src/data/predictions/monitoring_summary.csv
```

It retrains only horizons with:

```text
WARNING
CRITICAL
```

Forced demo execution:

```bash
python retrain.py --force
```

The forced mode retrains all horizons.

Flow:

```text
Read monitoring state
        |
        v
Determine triggered horizons
        |
        v
Capture current champion versions
        |
        v
Execute training notebook
        |
        v
Find newly registered versions
        |
        v
Assign candidate aliases
        |
        v
Save retraining_result.json
        |
        v
Log retraining run to MLflow
```

Important:

Retraining does not automatically change the champion alias.

This separation is deliberate.

A candidate must first pass the promotion gate.

---

# 10. Candidate Promotion Flow

Dry run:

```bash
python promote_candidate.py
```

Apply approved changes:

```bash
python promote_candidate.py --apply
```

Quality gates:

## Gate 1 — Model load

The candidate model must load successfully.

## Gate 2 — Signature exists

The model must contain a valid MLflow input signature.

## Gate 3 — Schema compatibility

The candidate must expect the same ordered feature schema as the champion.

## Gate 4 — MAE improvement

Default requirement:

```text
candidate improves MAE by at least 5%
```

## Gate 5 — RMSE safety

Default rule:

```text
candidate RMSE / champion RMSE <= 1.05
```

This allows at most 5% RMSE regression.

---

# 11. Promotion Outcomes

## Approved in dry run

```text
Decision: APPROVED
Action: DRY_RUN_APPROVED
```

No alias changes occur.

## Approved with `--apply`

```text
old champion -> previous_champion
candidate    -> champion
```

## Rejected

The model remains in MLflow but receives tags such as:

```text
lifecycle_role = rejected_candidate
rejection_timestamp
rejection_reason
```

This preserves auditability.

---

# 12. Docker Architecture

The project uses Docker Compose for services such as:

```text
MLflow Tracking Server
FastAPI inference API
```

Typical ports:

```text
MLflow -> 5001 on the host
API    -> 8000 on the host
```

Start services:

```bash
docker compose up --build -d
```

Check:

```bash
docker compose ps
```

View MLflow logs:

```bash
docker compose logs mlflow --tail=200
```

View API logs:

```bash
docker compose logs api --tail=200
```

Stop:

```bash
docker compose down
```

---

# 13. Recommended Execution Order

For normal use:

```bash
source /Users/charithagadari/.venvs/flood-risk/bin/activate
```

Set MLflow:

```bash
export MLFLOW_TRACKING_URI="sqlite:////Users/charithagadari/.mlflow/flood-risk/mlflow.db"
```

Run live inference:

```bash
python predict_live.py
```

Evaluate mature predictions:

```bash
python evaluate_predictions.py
```

Monitor production:

```bash
python monitor_production.py
```

Retrain only when required:

```bash
python retrain.py
```

For demo:

```bash
python retrain.py --force
```

Evaluate candidate:

```bash
python promote_candidate.py
```

Apply promotion only after reviewing the dry-run result:

```bash
python promote_candidate.py --apply
```

---

# 14. Troubleshooting Guide

This section should be followed in order. Do not change application code before identifying which layer is failing.

---

## Error A — Docker API socket not found

Example:

```text
failed to connect to the docker API
... docker.sock: no such file or directory
```

Meaning:

Docker Desktop or the Docker daemon is not running.

Fix:

1. Start Docker Desktop.
2. Wait for Docker Engine to become ready.
3. Run:

```bash
docker info
```

4. Then:

```bash
docker compose up --build -d
```

---

## Error B — MLflow container restarts with exit code 2

Example:

```text
Error: No such option '--allowed-hosts'
```

Meaning:

The installed MLflow version does not support that CLI option.

Find the option:

```bash
grep -R --line-number --exclude-dir=.git --exclude-dir=venv \
  -- "--allowed-hosts" .
```

Remove the unsupported option from the active `docker-compose.yml` command.

Then rebuild:

```bash
docker compose down
docker compose up --build -d
```

Check:

```bash
docker compose logs mlflow --tail=200
```

---

## Error C — MLflow health check fails

Test:

```bash
curl http://127.0.0.1:5001/health
```

Expected:

```text
OK
```

If it fails:

```bash
docker compose ps -a
```

Then:

```bash
docker compose logs mlflow --tail=300
```

Do not debug the health check before confirming the MLflow process itself is running.

---

## Error D — Docker build fails with blob or input/output error

Examples:

```text
blob sha256 ... input/output error
```

or:

```text
failed to extract layer ... input/output error
```

Meaning:

Docker Desktop's internal containerd storage or cached image layers may be corrupted.

Recovery order:

1. Quit Docker Desktop completely.
2. Restart Docker Desktop.
3. Check:

```bash
docker info
```

4. Remove project images if possible:

```bash
docker image rm -f flood-risk-ml-intoto-mlflow
docker image rm -f flood-risk-ml-intoto-api
```

5. Rebuild without cache:

```bash
docker compose build --no-cache
```

6. If needed:

```bash
docker builder prune -a
```

7. If needed:

```bash
docker system prune -a
```

8. Check host disk space:

```bash
df -h
```

9. Check Docker disk usage:

```bash
docker system df
```

10. If the storage error continues, use Docker Desktop:

```text
Settings -> Troubleshoot -> Clean / Purge data
```

Use factory reset only as a last resort.

---

## Error E — `python predict_live.py` appears stuck

First determine where it is stuck.

Use unbuffered output:

```bash
python -u predict_live.py
```

Check whether Python itself works:

```bash
python -u -c "print('Python works')"
```

Check MLflow import:

```bash
python -u -c "import mlflow; print('MLflow imported')"
```

If `import mlflow` hangs, verify that the correct virtual environment is active.

Example:

```bash
source /Users/charithagadari/.venvs/flood-risk/bin/activate
```

Then test again.

---

## Error F — Wrong MLflow tracking URI

Check:

```bash
echo $MLFLOW_TRACKING_URI
```

For local SQLite:

```bash
export MLFLOW_TRACKING_URI="sqlite:////Users/charithagadari/.mlflow/flood-risk/mlflow.db"
```

For Docker-hosted MLflow:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
```

Host-to-container and container-to-container addresses are different.

Examples:

```text
Host -> MLflow container: http://127.0.0.1:5001
API container -> MLflow container: http://mlflow:5000
```

Do not use `localhost` inside one container to reach another container.

---

## Error G — `NameError: TRACKING_URI is not defined`

Meaning:

The script defines one variable name but uses another.

For example:

```python
MLFLOW_TRACKING_URI = ...
```

but later:

```python
TRACKING_URI
```

Fix by using one consistent variable name throughout the file.

Recommended:

```python
MLFLOW_TRACKING_URI
```

---

## Error H — Model dependency mismatch warning

Example:

```text
psutil current: uninstalled
rich current: uninstalled
```

Meaning:

The current Python environment differs from the environment recorded with the MLflow model.

If prediction still succeeds, this is a warning rather than an immediate failure.

To align the environment:

```bash
pip install psutil==7.2.2 rich==15.0.0
```

For strict reproducibility, retrieve the model dependencies from MLflow and install the recorded environment.

---

## Error I — Model signature missing

Possible error:

```text
The MLflow model does not contain an input signature.
```

Meaning:

The model was logged without a signature.

Fix:

During training, log the model with a signature inferred from training examples.

Then register a new version.

Do not disable signature checks in production unless absolutely necessary.

---

## Error J — Missing inference features

Example:

```text
model is missing required inference features
```

Meaning:

The live feature engineering pipeline does not produce all columns expected by the trained model.

Possible causes:

- Training feature code changed.
- Inference feature code changed.
- Different lag/window configuration.
- New feature added during training but not production.

Fix:

1. Compare model signature columns with live feature columns.
2. Make training and inference feature configuration identical.
3. Retrain and re-register if the intended schema changed.

---

## Error K — NaN values in latest inference row

Meaning:

There is not enough historical data to calculate the largest lag or rolling window.

Example:

A 72-hour lag requires enough historical rows before the latest row can contain a valid 72-hour lag value.

Fix:

Fetch a longer data window.

For example, increase:

```python
days=14
```

if required.

Also inspect gaps in the source time series.

---

## Error L — No predictions ready for evaluation

Expected message:

```text
NO NEW PREDICTIONS READY FOR EVALUATION
```

This is not an error.

It means the forecast target timestamps are still in the future.

Example:

A 72-hour forecast can only be evaluated after 72 hours have passed.

Action:

Run `evaluate_predictions.py` again later.

---

## Error M — Actual NVE observation unavailable

Meaning:

The target timestamp has passed, but no sufficiently close NVE observation was found within the matching tolerance.

Possible causes:

- API delay
- Missing observations
- Irregular timestamps
- Temporary API issue

Action:

Run the evaluation again later.

Do not mark the prediction as evaluated until actual ground truth is available.

---

## Error N — Monitoring status is `INSUFFICIENT_DATA`

This is expected early in production.

Current minimum sample requirements:

```text
6h  -> 24
24h -> 14
72h -> 7
```

The system intentionally avoids retraining based on too few observations.

Action:

Continue generating and evaluating predictions.

---

## Error O — Monitoring says baseline unavailable

Meaning:

The original test MAE could not be found.

Check the source MLflow run:

- Was MAE logged?
- What was the metric name?
- Does the model version point to the expected run?
- Is `selection_metric_value` stored in model tags?

Fix the training/registration metadata and register a corrected version.

---

## Error P — Retraining does not trigger

Normal behavior if all horizons are:

```text
HEALTHY
INSUFFICIENT_DATA
BASELINE_UNAVAILABLE
```

Retraining triggers only for:

```text
WARNING
CRITICAL
```

For demonstration:

```bash
python retrain.py --force
```

---

## Error Q — Training completes but no new model version is found

Possible causes:

- Training notebook did not register a model.
- Registration failed.
- The same old registry version remains the latest.
- Notebook execution failed before registration.

Check:

1. Notebook execution output.
2. MLflow experiment runs.
3. Model Registry versions.
4. Whether the training code registers models for all horizons.

---

## Error R — Promotion rejected

This can be correct behavior.

A candidate may be rejected because:

- MAE improvement is less than 5%.
- RMSE regressed too much.
- Signature is missing.
- Feature schema differs.
- Candidate cannot be loaded.

Inspect:

```text
src/data/retraining/promotion_summary.csv
src/data/retraining/promotion_summary.json
```

Do not force promotion merely because retraining succeeded.

---

## Error S — Candidate schema differs from champion

Meaning:

The candidate expects different features.

This can indicate:

- Feature engineering changed.
- Training notebook changed.
- Production inference was not updated.

Safe options:

1. Keep the old champion.
2. Update and test the production feature pipeline.
3. Perform a controlled deployment of the new schema.
4. Only then promote the candidate.

---

## Error T — MLflow database repeatedly prints table initialization logs

Example:

```text
Creating initial MLflow database tables
Updating database tables
```

This may appear when new MLflow clients or stores are initialized.

If operations succeed, it is usually informational.

Investigate only if accompanied by database locking, migration, or connection errors.

---

## Error U — SQLite database lock

Possible error:

```text
database is locked
```

Possible cause:

Too many concurrent writers using SQLite.

For a local development project, SQLite is acceptable.

For multi-user or more realistic production deployment, use a dedicated database such as PostgreSQL.

---

## Error V — API cannot reach MLflow inside Docker

Do not use:

```text
http://127.0.0.1:5001
```

from one Docker container to another.

Use the Compose service name:

```text
http://mlflow:5000
```

Reason:

Inside the API container, `127.0.0.1` refers to the API container itself.

---

# 15. Debugging Order

Use this order whenever the project fails:

```text
1. Is the correct virtual environment active?
2. Is Python working?
3. Is MLflow import working?
4. Is MLFLOW_TRACKING_URI correct?
5. Is the MLflow backend reachable?
6. Is Docker running?
7. Are containers running?
8. Are container logs clean?
9. Can live NVE data be fetched?
10. Can champion aliases be resolved?
11. Can models load?
12. Does the model signature match live features?
13. Can prediction be generated?
14. Is the prediction log written?
15. Are target timestamps mature?
16. Can actual ground truth be fetched?
17. Can production metrics be calculated?
18. Is monitoring status valid?
19. Is retraining triggered only when appropriate?
20. Does the candidate pass promotion gates?
```

This prevents random code changes and makes debugging systematic.

---

# 16. Demo Flow for Interviews or Portfolio Presentation

A useful demonstration is:

### 1. Show MLflow Registry

Explain:

```text
Each horizon has a registered model.
Production uses the champion alias.
```

### 2. Run live inference

```bash
python predict_live.py
```

Explain:

```text
The script fetches live NVE data and dynamically loads the approved champion models.
```

### 3. Show prediction log

Explain:

```text
Every production prediction is stored with model version and target timestamp.
```

### 4. Run delayed evaluation

```bash
python evaluate_predictions.py
```

Explain:

```text
Only forecasts with available future ground truth are evaluated.
```

### 5. Run monitoring

```bash
python monitor_production.py
```

Explain:

```text
Production error is compared with the original test baseline.
```

### 6. Force retraining for demo

```bash
python retrain.py --force
```

Explain:

```text
Real production retraining would be triggered only by monitoring degradation.
```

### 7. Run promotion dry run

```bash
python promote_candidate.py
```

Explain:

```text
A newly trained model does not automatically become production champion.
```

### 8. Apply promotion only if approved

```bash
python promote_candidate.py --apply
```

Explain:

```text
The system preserves the old champion for rollback.
```

---

# 17. Recommended Improvements

Future improvements could include:

- PostgreSQL instead of SQLite for multi-service production use
- Scheduled orchestration using Airflow, Prefect, or Kubeflow
- Data drift monitoring in addition to performance monitoring
- Prometheus and Grafana dashboards
- Alerting through email or Slack
- Station-specific model configuration
- Multi-station training
- Automated integration tests
- CI/CD with GitHub Actions
- Object storage for MLflow artifacts
- Cloud deployment
- Model serving through a dedicated inference service
- Backtesting and rolling-window evaluation
- More robust flood-risk thresholds based on official hydrological warning levels

---

# 18. Git Commands to Save the Current Project

Check current branch:

```bash
git branch --show-current
```

Check modified files:

```bash
git status
```

Review changes:

```bash
git diff
```

Stage all changes:

```bash
git add .
```

Check staged changes:

```bash
git status
```

Commit:

```bash
git commit -m "Complete end-to-end flood forecasting MLOps pipeline"
```

Push current branch:

```bash
git push
```

If the branch has never been pushed:

```bash
git push -u origin $(git branch --show-current)
```

For `main` explicitly:

```bash
git push origin main
```

---

# 19. Recommended Commit Sequence

A clean history could use separate commits:

```bash
git add predict_live.py evaluate_predictions.py monitor_production.py

git commit -m "Add live inference and production monitoring pipeline"
```

```bash
git add retrain.py promote_candidate.py

git commit -m "Add retraining and model promotion workflow"
```

```bash
git add docker-compose.yml Dockerfile requirement.txt

git commit -m "Update Docker and MLflow deployment configuration"
```

```bash
git add README.md PROJECT_GUIDE.md

git commit -m "Add project architecture and operations documentation"
```

Then push:

```bash
git push origin main
```

---

# 20. Final System Summary

The completed system demonstrates the following production ML lifecycle:

```text
LIVE DATA
   |
   v
FEATURE ENGINEERING
   |
   v
MODEL TRAINING
   |
   v
MLFLOW TRACKING
   |
   v
MODEL REGISTRY
   |
   v
CHAMPION DEPLOYMENT
   |
   v
LIVE INFERENCE
   |
   v
PREDICTION LOGGING
   |
   v
DELAYED GROUND TRUTH
   |
   v
PRODUCTION EVALUATION
   |
   v
MONITORING
   |
   +------ HEALTHY ------> CONTINUE SERVING
   |
   +------ WARNING / CRITICAL
                    |
                    v
                RETRAINING
                    |
                    v
                 CANDIDATE
                    |
                    v
              QUALITY GATES
                    |
             +------+------+
             |             |
             v             v
           REJECT        PROMOTE
                            |
                            v
                      NEW CHAMPION
```

This is the core architecture of an end-to-end MLOps system.

