# Flood Risk Forecasting — End-to-End MLOps System

An end-to-end machine learning system for multi-horizon water-level forecasting and flood-risk monitoring using live hydrological data from the Norwegian Water Resources and Energy Directorate (NVE).

The project covers the complete ML lifecycle:

- Live data ingestion from the NVE Hydrology API
- Time-series feature engineering
- Multi-horizon forecasting for 6h, 24h, and 72h
- ML experiment tracking with MLflow
- Model Registry with champion/candidate aliases
- Registry-driven live inference
- Prediction logging
- Delayed ground-truth evaluation
- Production model monitoring
- Drift/performance degradation detection
- Retraining triggers
- Candidate evaluation and controlled model promotion
- Docker-based serving for the API and MLflow

---

## 1. Project Overview

The goal of this project is to demonstrate a realistic production-oriented ML workflow rather than only training a model in a notebook.

The system continuously follows this lifecycle:

```text
NVE Hydrology API
        |
        v
Data ingestion
        |
        v
Validation + preprocessing
        |
        v
Time-series feature engineering
        |
        v
Model training + evaluation
        |
        v
MLflow experiment tracking
        |
        v
MLflow Model Registry
        |
        v
Champion model aliases
        |
        v
Live inference
        |
        v
Prediction logging
        |
        v
Wait for future ground truth
        |
        v
Delayed evaluation
        |
        v
Production monitoring
        |
        +----------------------------+
        |                            |
        | HEALTHY                    | WARNING / CRITICAL
        |                            |
        v                            v
Continue serving              Trigger retraining
                                      |
                                      v
                               Candidate models
                                      |
                                      v
                              Promotion quality gates
                                      |
                           +----------+----------+
                           |                     |
                           v                     v
                        Reject                Promote
                                                 |
                                                 v
                                          New champion
```

---

## 2. Forecasting Task

The system forecasts water level for three future horizons:

- 6 hours
- 24 hours
- 72 hours

The current deployed champion models are loaded directly from the MLflow Model Registry using aliases such as:

```text
models:/flood_forecast_6h@champion
models:/flood_forecast_24h@champion
models:/flood_forecast_72h@champion
```

This allows the inference code to remain unchanged when a new model version is promoted.

---

## 3. Data Source

Data is fetched from the NVE Hydrology API.

Current station:

```text
Station ID: 1.15.0
```

Main variables:

```text
water_level
reservoir_volume
```

The live ingestion pipeline retrieves recent hourly observations and prepares them for inference.

---

## 4. Feature Engineering

The production feature pipeline includes:

### Calendar features

- Hour
- Day of year
- Month
- Day of week
- Weekend indicator

### Cyclic encodings

- Sine/cosine encoding for hour
- Sine/cosine encoding for day of year
- Sine/cosine encoding for month

### Lag features

Examples:

```text
1h
3h
6h
12h
24h
48h
72h
```

### Rolling statistics

For multiple windows:

- Mean
- Standard deviation
- Minimum
- Maximum

### Change and slope features

For each configured lag:

```text
change = current_value - lagged_value
slope  = change / elapsed_hours
```

The live inference feature schema is validated against the MLflow model signature before prediction.

---

## 5. Main Components

### `predict_live.py`

Performs registry-driven live inference.

Main responsibilities:

1. Configure MLflow.
2. Fetch latest NVE observations.
3. Validate the data.
4. Build inference features.
5. Load champion models from MLflow Registry.
6. Generate 6h, 24h, and 72h forecasts.
7. Assign a simple flood-risk category.
8. Save predictions to `prediction_log.csv`.

---

### `evaluate_predictions.py`

Performs delayed ground-truth evaluation.

A forecast cannot be evaluated immediately because the actual future water level is not available yet.

The script:

1. Loads logged predictions.
2. Finds predictions whose target timestamp has passed.
3. Fetches actual NVE observations.
4. Matches predictions with actual measurements.
5. Calculates prediction errors.
6. Updates `evaluation_log.csv`.
7. Produces `performance_summary.csv`.

Metrics include:

- MAE
- RMSE
- Mean error
- MAPE

---

### `monitor_production.py`

Monitors the performance of the current champion models.

It compares production MAE with the original model test MAE.

Possible statuses:

```text
INSUFFICIENT_DATA
BASELINE_UNAVAILABLE
HEALTHY
WARNING
CRITICAL
```

Current monitoring thresholds:

```text
WARNING  : production MAE >= 1.50 × test MAE
CRITICAL : production MAE >= 2.00 × test MAE
```

Minimum evaluated predictions are required before a health decision is made.

---

### `retrain.py`

Controls model retraining.

Normal mode:

```bash
python retrain.py
```

Retraining occurs only when monitoring reports:

```text
WARNING
CRITICAL
```

Demo or manual mode:

```bash
python retrain.py --force
```

This forces retraining for all supported horizons.

The script:

1. Reads the monitoring summary.
2. Identifies triggered horizons.
3. Records the current champion versions.
4. Executes the training notebook.
5. Finds newly registered model versions.
6. Assigns the `candidate` alias.
7. Saves retraining metadata.
8. Logs the retraining control run to MLflow.

---

### `promote_candidate.py`

Implements controlled model promotion.

Default dry run:

```bash
python promote_candidate.py
```

Apply approved promotions:

```bash
python promote_candidate.py --apply
```

Quality gates include:

- Candidate model can be loaded.
- Model signature exists.
- Candidate feature schema matches the current champion.
- Candidate MAE improves by the configured minimum.
- Candidate RMSE does not regress beyond the allowed threshold.

When approved:

```text
current champion -> previous_champion
candidate        -> champion
```

Rejected models remain in the registry with rejection metadata.

---

## 6. Project Architecture

```text
                         +----------------------+
                         |   NVE Hydrology API  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |   Data Ingestion     |
                         | src/ingestion/       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Validation & Feature |
                         | Engineering          |
                         +----------+-----------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
       +----------------------+            +----------------------+
       | Training Notebook    |            | Live Inference       |
       | Model Comparison     |            | predict_live.py      |
       +----------+-----------+            +----------+-----------+
                  |                                   |
                  v                                   v
       +----------------------+            +----------------------+
       | MLflow Experiments   |            | prediction_log.csv   |
       +----------+-----------+            +----------+-----------+
                  |                                   |
                  v                                   v
       +----------------------+            +----------------------+
       | MLflow Model Registry|            | Delayed Evaluation   |
       | champion/candidate   |            | evaluate_predictions |
       +----------+-----------+            +----------+-----------+
                  |                                   |
                  +-------------------+---------------+
                                      |
                                      v
                           +----------------------+
                           | Production Monitoring|
                           | monitor_production.py|
                           +----------+-----------+
                                      |
                         HEALTHY -----+----- WARNING/CRITICAL
                                                |
                                                v
                                     +----------------------+
                                     | Retraining Controller|
                                     | retrain.py           |
                                     +----------+-----------+
                                                |
                                                v
                                     +----------------------+
                                     | Candidate Model      |
                                     +----------+-----------+
                                                |
                                                v
                                     +----------------------+
                                     | Promotion Gate       |
                                     | promote_candidate.py |
                                     +----------+-----------+
                                                |
                                      Reject ---+--- Promote
                                                        |
                                                        v
                                                New Champion
```

---

## 7. Directory Structure

A simplified project structure is shown below:

```text
flood-risk-ml-intoto/
|
|-- app/
|   `-- main.py
|
|-- notebooks/
|   |-- 01_eda_reservoir_updated.ipynb
|   |-- forecasting_model.ipynb
|   |-- mlflow_configuration_v2.ipynb
|   `-- forecasting_model_mlflow_registry.ipynb
|
|-- src/
|   |-- ingestion/
|   |   `-- nve_client.py
|   |
|   `-- data/
|       |-- predictions/
|       |   |-- prediction_log.csv
|       |   |-- evaluation_log.csv
|       |   |-- performance_summary.csv
|       |   |-- monitoring_summary.csv
|       |   `-- monitoring_summary.json
|       |
|       `-- retraining/
|           |-- retraining_result.json
|           |-- promotion_summary.csv
|           `-- promotion_summary.json
|
|-- predict_live.py
|-- evaluate_predictions.py
|-- monitor_production.py
|-- retrain.py
|-- promote_candidate.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirement.txt
|-- README.md
`-- PROJECT_GUIDE.md
```

---

## 8. Environment Setup

Create and activate a virtual environment.

Example:

```bash
python3.12 -m venv ~/.venvs/flood-risk
source ~/.venvs/flood-risk/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirement.txt
```

---

## 9. MLflow Configuration

For local development using SQLite:

```bash
export MLFLOW_TRACKING_URI="sqlite:////Users/<your-user>/.mlflow/flood-risk/mlflow.db"
```

Example:

```bash
export MLFLOW_TRACKING_URI="sqlite:////Users/charithagadari/.mlflow/flood-risk/mlflow.db"
```

For Docker-hosted MLflow exposed on port 5001:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
```

Check the active URI:

```bash
echo $MLFLOW_TRACKING_URI
```

---

## 10. Running the Full Pipeline

### Step 1 — Start infrastructure

```bash
docker compose up --build -d
```

Check services:

```bash
docker compose ps
```

Check MLflow health:

```bash
curl http://127.0.0.1:5001/health
```

Expected result:

```text
OK
```

---

### Step 2 — Run live inference

```bash
python predict_live.py
```

Expected output includes:

- Live NVE observations
- Current water level
- Champion model versions
- 6h, 24h, and 72h forecasts
- Risk levels
- Prediction log path

---

### Step 3 — Evaluate mature predictions

```bash
python evaluate_predictions.py
```

Only forecasts whose target timestamps have already passed can be evaluated.

---

### Step 4 — Monitor production performance

```bash
python monitor_production.py
```

The monitoring result is saved to:

```text
src/data/predictions/monitoring_summary.csv
```

---

### Step 5 — Trigger retraining

Normal mode:

```bash
python retrain.py
```

Force retraining for demonstration:

```bash
python retrain.py --force
```

---

### Step 6 — Evaluate candidates

Dry run:

```bash
python promote_candidate.py
```

Apply approved promotion:

```bash
python promote_candidate.py --apply
```

---

## 11. Docker Services

Typical services:

```text
mlflow -> host port 5001
api    -> host port 8000
```

Useful commands:

```bash
docker compose ps
```

```bash
docker compose logs mlflow --tail=200
```

```bash
docker compose logs api --tail=200
```

```bash
docker compose down
```

```bash
docker compose up --build -d
```

---

## 12. Current Production Monitoring Example

A successful monitoring run produced the following behavior:

```text
6h  -> production MAE lower than test MAE, but insufficient samples
24h -> production MAE lower than test MAE, but insufficient samples
72h -> production MAE moderately higher than test MAE, but insufficient samples
```

This is expected. The monitoring system deliberately waits for enough production observations before triggering retraining.

---

## 13. Key MLOps Concepts Demonstrated

This project demonstrates:

- Experiment tracking
- Reproducible model training
- Model versioning
- Model Registry
- Champion/candidate deployment strategy
- Model signatures
- Feature-schema validation
- Registry-driven inference
- Delayed ground-truth evaluation
- Production model monitoring
- Performance degradation detection
- Automated retraining triggers
- Controlled promotion gates
- Rollback support
- Containerization
- API deployment

---

## 14. Recommended Git Workflow

Check changes:

```bash
git status
```

Review changes:

```bash
git diff
```

Stage the files:

```bash
git add .
```

Commit:

```bash
git commit -m "Complete end-to-end flood forecasting MLOps pipeline"
```

Push:

```bash
git push origin main
```

If working on another branch:

```bash
git branch --show-current
git push -u origin <branch-name>
```

---

## 15. Documentation

For detailed operational instructions, architecture explanation, failure recovery procedures, and troubleshooting, see:

```text
PROJECT_GUIDE.md
```

