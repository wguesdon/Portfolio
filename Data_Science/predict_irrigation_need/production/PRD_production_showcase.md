# PRD: Irrigation Prediction Production Showcase

## Goal

Build an end-to-end production ML system for irrigation need prediction. Demonstrate the full lifecycle from model training to serving predictions via a web app with drift monitoring.

This is a showcase project. The irrigation dataset is synthetic. The goal is to demonstrate production ML practices, not to build a real irrigation system.

## Architecture

```
User (browser)
    |
    v
Streamlit Frontend
    |
    v
FastAPI Backend
    |-- /predict      (single prediction)
    |-- /batch         (CSV upload)
    |-- /health        (service health)
    |-- /drift         (drift metrics)
    |
    v
Feature Pipeline --> CatBoost Model --> Calibrated Probabilities
    |
    v
SQLite Prediction Log --> Drift Monitor (daily batch)
```

## Components

### 1. Model Training

Train a single CatBoost model optimized for generalization, not competition score.

**Training approach:**
- 80/10/10 stratified split: train / validation / calibration
- No OOF. No stacking. No threshold hacking.
- Use v5 feature engineering (domain, magic formula, polynomial, log transforms)
- Skip 2-way interaction TE (requires training data at inference, adds serving complexity)
- Optuna tuning on validation set (30 trials)
- Early stopping on validation balanced accuracy

**Probability calibration:**
- Fit isotonic regression on the calibration set (10% held out)
- Output calibrated P(Low), P(Medium), P(High) that reflect true class frequencies
- Store calibrator as a pickle alongside the model

**Artifacts:**
- `model.cbm`: CatBoost model file (~10MB)
- `calibrator.pkl`: Isotonic regression calibrator
- `training_metadata.json`: feature list, training date, validation score, data hash
- `reference_distributions.json`: per-feature mean, std, quantiles from training data

### 2. Feature Pipeline

Standalone Python module that transforms raw input into model features.

**Input validation:**
- Check all required columns are present
- Validate value ranges (e.g. Soil_Moisture 0-100, Temperature_C -10 to 60)
- Flag missing values (impute with training median or reject)
- Return validation errors as structured JSON

**Feature computation:**
- Domain interactions (water balance, heat stress, etc.)
- Magic formula features (threshold flags + composite scores)
- Polynomial and log transforms
- No target encoding (would require training data at inference)
- No digit extraction (synthetic data artifact, not useful on real data)

**Output:**
- Feature DataFrame ready for model.predict_proba()
- List of computed feature names for debugging

### 3. Drift Monitoring

Detect when incoming data diverges from training distribution.

**Reference distributions:**
- Computed once from training data at model training time
- Per-feature: mean, std, min, max, 5th/25th/50th/75th/95th percentiles
- Per-class: expected prediction distribution from validation set

**Drift metrics (computed daily on prediction log):**
- PSI (Population Stability Index) per feature. Alert if PSI > 0.2.
- Prediction distribution shift: compare daily class proportions vs expected
- Feature out-of-range rate: % of predictions where any feature exceeds training range
- Volume anomaly: unusual spike or drop in prediction volume

**Storage:**
- SQLite table `predictions`: timestamp, input features, predicted class, probabilities
- SQLite table `drift_metrics`: date, feature, psi, oor_rate, pred_distribution

**Alerting:**
- Log warnings when drift exceeds thresholds
- Surface on /drift dashboard endpoint
- In a real system this would send Slack/email alerts

### 4. Web Application

**FastAPI backend:**

```
POST /predict
  Input: JSON with field conditions (19 features)
  Output: {
    "predicted_class": "Medium",
    "probabilities": {"Low": 0.12, "Medium": 0.83, "High": 0.05},
    "confidence": "high",
    "model_version": "v1_2026-04-04"
  }

POST /batch
  Input: CSV file upload
  Output: CSV with predictions appended

GET /health
  Output: {"status": "healthy", "model_version": "...", "last_drift_check": "..."}

GET /drift
  Output: drift metrics summary + per-feature PSI values
```

**Streamlit frontend:**
- Input form: sliders/dropdowns for all 19 features (soil, weather, crop, field)
- Prediction display: class label, probability bar chart, confidence indicator
- Batch upload: drag-and-drop CSV, download results
- Drift dashboard: PSI chart over time, feature distribution comparison plots

### 5. Deployment

**Docker container:**
- Single container with FastAPI + Streamlit + model + feature pipeline
- SQLite file for prediction log (mounted volume for persistence)
- Health check on /health endpoint

**Infrastructure options (pick one):**
- Local: `podman run -p 8000:8000 irrigation-predictor`
- AWS ECS/Fargate: single task, ALB in front
- AWS SageMaker endpoint: for the model only, separate app for frontend

**CI/CD:**
- GitHub Actions: on push to main, build Docker image, run tests, push to ECR
- Model retraining: manual trigger or scheduled (when new data available)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | CatBoost |
| Calibration | scikit-learn IsotonicRegression |
| Feature pipeline | pandas + numpy |
| API | FastAPI |
| Frontend | Streamlit |
| Prediction log | SQLite |
| Drift metrics | PSI (custom), scipy.stats.ks_2samp |
| Container | Podman/Docker |
| Deployment | AWS ECS or local |

## File Structure

```
production/
  PRD_production_showcase.md    # This document
  app/
    main.py                     # FastAPI application
    features.py                 # Feature pipeline (standalone, no training deps)
    model.py                    # Model loading, prediction, calibration
    drift.py                    # Drift monitoring logic
    validation.py               # Input validation
  frontend/
    streamlit_app.py            # Streamlit UI
  training/
    train_production.py         # Single model training script
    calibrate.py                # Probability calibration
    compute_reference.py        # Reference distribution computation
  artifacts/
    model.cbm                   # Trained model
    calibrator.pkl              # Probability calibrator
    training_metadata.json      # Training metadata
    reference_distributions.json # Drift reference
  tests/
    test_features.py
    test_predict.py
    test_drift.py
  Dockerfile
  docker-compose.yml
```

## Milestones

1. **Model training + calibration** (train_production.py, calibrate.py)
2. **Feature pipeline + validation** (features.py, validation.py)
3. **FastAPI backend** (main.py, model.py, /predict, /batch, /health)
4. **Drift monitoring** (drift.py, /drift, reference distributions)
5. **Streamlit frontend** (input form, prediction display, drift dashboard)
6. **Docker packaging + deployment** (Dockerfile, docker-compose.yml)

## Out of Scope

- Real-time streaming predictions (batch/request-response only)
- A/B testing infrastructure
- Multi-model serving
- GPU inference (CatBoost CPU inference is fast enough)
- Authentication/authorization (showcase only)
