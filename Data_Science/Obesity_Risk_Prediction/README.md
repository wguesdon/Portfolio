# Multi-Class Prediction of Obesity Risk

**Kaggle Playground Series Season 4, Episode 2**
**Final Rank: 30 / 3,587 (Top 1%)**

---

## Competition Overview

Multi-class classification predicting obesity risk level from lifestyle and anthropometric features.

- **Task**: Classify individuals into one of 7 obesity categories
- **Metric**: Accuracy
- **Dataset**: ~20,000 synthetic training samples, 16 features

**Target Classes**:
| Class | Description |
|-------|-------------|
| `Insufficient_Weight` | BMI below normal range |
| `Normal_Weight` | Healthy BMI |
| `Overweight_Level_I` | Mild overweight |
| `Overweight_Level_II` | Moderate overweight |
| `Obesity_Type_I` | Class I obesity |
| `Obesity_Type_II` | Class II obesity |
| `Obesity_Type_III` | Class III obesity (severe) |

---

## Approach

### Key Insight: BMI is Near-Deterministic

For synthetic Kaggle PS data derived from a real dataset, BMI (Weight / Height²) correlates extremely strongly with the target class. The feature engineering focuses on deriving BMI and related interaction features first, then adding lifestyle context.

### Pipeline

```
Raw Data (16 features)
       │
       ▼
Feature Engineering (aws/features.py)
  ├── BMI = Weight / Height²          ← primary discriminator
  ├── Age groups (0-20, 20-30, 30-45, 45+)
  ├── Activity score = FAF - TUE
  ├── Water intake categories
  ├── Meal frequency features
  ├── Binary encodings (Gender, SMOKE, SCC, FAVC)
  ├── Ordinal encodings (CAEC, CALC, MTRANS)
  └── Interaction features (BMI*FAF, Age*BMI)
       │
       ▼
Optuna Hyperparameter Tuning
  └── 100 trials, single 80/20 stratified split (fast)
       │
       ▼
5-Fold Stratified Cross-Validation
  ├── LightGBM  (ml.m5.4xlarge)
  ├── XGBoost   (ml.g4dn.xlarge, GPU)
  └── CatBoost  (ml.g5.xlarge, GPU)
       │
       ▼
Ensemble Optimization
  ├── Simple average
  ├── Optuna weighted average
  └── Stacking (Logistic Regression meta-learner)
       │
       ▼
Final Submission (argmax of probability matrix)
```

### AWS SageMaker Architecture

```
Local Machine
     │
     │  uv run python scripts/upload_data.py
     ▼
S3 Bucket (kaggle-ps4e2)
  ├── data/train.csv
  ├── data/test.csv
  ├── data/config.yaml
  └── data/features.py
     │
     │  uv run python scripts/launch_training.py
     │  (boto3 low-level API — no sagemaker package)
     ▼
SageMaker Training Jobs (Spot Instances)
  ├── playground-series-s4e2-lgb-{timestamp}
  ├── playground-series-s4e2-xgb-{timestamp}
  └── playground-series-s4e2-cat-{timestamp}
     │
     ▼
S3 Output
  ├── output/lgb/{job}/model.tar.gz  → lgb_oof.npy, lgb_test.npy
  ├── output/xgb/{job}/model.tar.gz  → xgb_oof.npy, xgb_test.npy
  └── output/cat/{job}/model.tar.gz  → cat_oof.npy, cat_test.npy
     │
     │  uv run python scripts/download_artifacts.py
     │  uv run python scripts/train_ensemble.py
     ▼
ensemble/
  ├── ensemble_oof.npy    (n_samples, 7)
  ├── ensemble_test.npy   (n_test, 7)
  └── ensemble_results.json
     │
     ▼
submissions/submission_aws_latest.csv
```

---

## Key Results

| Model | CV Accuracy |
|-------|------------|
| LightGBM | ~0.909 |
| XGBoost | ~0.908 |
| CatBoost | ~0.910 |
| **Ensemble** | **~0.913** |

- **Public Leaderboard**: Top 1%
- **Private Leaderboard**: 30 / 3,587 (Top 1%)

---

## AWS CLI Method — Key Design Decision

The `scripts/launch_training.py` uses `boto3.client('sagemaker').create_training_job()` directly — the low-level AWS API equivalent to:

```bash
aws sagemaker create-training-job --cli-input-json file://job.json
```

This removes the `sagemaker` Python package dependency. The `--save-json` flag writes the full job JSON to `/tmp/` so you can inspect or replay the exact API call.

---

## Repository Structure

```
Obesity_Risk_Prediction/
├── README.md
├── aws/
│   ├── config.yaml          Competition + AWS configuration
│   ├── features.py          Feature engineering module
│   └── run.sh               Pipeline wrapper (upload/train/download/ensemble/submit)
├── containers/
│   ├── lgb/
│   │   ├── Dockerfile       LightGBM container (CPU)
│   │   └── train.py         -> mirrors src/train_lgb.py
│   ├── xgb/
│   │   ├── Dockerfile       XGBoost container (GPU)
│   │   └── train.py         -> mirrors src/train_xgb.py
│   └── cat/
│       ├── Dockerfile       CatBoost container (GPU)
│       └── train.py         -> mirrors src/train_cat.py
├── scripts/
│   ├── launch_training.py   AWS CLI method (boto3 low-level, no sagemaker pkg)
│   ├── upload_data.py       Upload data + config to S3
│   ├── download_artifacts.py Download model.tar.gz from S3
│   └── train_ensemble.py    Blend OOF predictions, find best weights
└── src/
    ├── base_trainer.py      Abstract base: Optuna tuning + K-fold CV
    ├── ensemble.py          Ensemble methods (avg, weighted, stacking)
    ├── train_lgb.py         LightGBM trainer
    ├── train_xgb.py         XGBoost trainer (GPU)
    └── train_cat.py         CatBoost trainer (GPU)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | LightGBM 4.x, XGBoost 2.x, CatBoost 1.x |
| HPO | Optuna 3.x (TPE sampler) |
| CV | scikit-learn StratifiedKFold |
| Cloud | AWS SageMaker (Spot Instances) |
| Containers | Docker / ECR |
| Experiment Tracking | MLflow on SageMaker |
| AWS SDK | boto3 (low-level, no sagemaker package) |
| Runtime | Python 3.11, uv |

---

## Quickstart

```bash
# 1. Build and push containers to ECR (run from competition root)
docker build -f containers/lgb/Dockerfile -t {account}.dkr.ecr.us-east-1.amazonaws.com/kaggle-training-lgb:latest .
docker push {account}.dkr.ecr.us-east-1.amazonaws.com/kaggle-training-lgb:latest
# repeat for xgb, cat

# 2. Run full pipeline
./aws/run.sh all        # upload data + launch training
./aws/run.sh status     # check job progress
./aws/run.sh download   # after training completes
./aws/run.sh ensemble   # blend predictions
./aws/run.sh submit     # create submission CSV

# 3. Or step by step with dry-run
uv run python scripts/launch_training.py --config aws/config.yaml --dry-run --save-json
```
