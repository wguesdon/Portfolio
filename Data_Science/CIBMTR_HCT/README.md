# CIBMTR: Equity in post-HCT Survival Predictions

Survival prediction for ~28,800 allogeneic HCT patients using gradient boosting ensemble.
Kaggle competition focused on equitable performance across racial groups.

---

## Results

| Model | CV Stratified CI | OOF Stratified CI |
|---|---|---|
| XGBoost | 0.6523 ± 0.0066 | 0.6556 |
| LightGBM | 0.6503 ± 0.0067 | 0.6536 |
| **CatBoost** | **0.6532 ± 0.0072** | **0.6569** |
| **Ensemble** | — | **0.6928** (private LB) |

**Metric**: Stratified Concordance Index = mean(C-index per race group) − std(C-index per race group)

---

## Project Structure

```
CIBMTR_HCT/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_dictionary.csv
├── output/
│   ├── eda/          # 8 EDA plots
│   └── model/        # 6 model plots + cv_results.csv + submission.csv
├── src/
│   ├── config.py       # paths, constants, colour palettes
│   ├── features.py     # data loading & feature engineering
│   ├── metrics.py      # stratified concordance index
│   ├── eda.py          # EDA script → output/eda/
│   ├── train.py        # model training & CV → output/model/
│   ├── ensemble.py     # rank ensemble + submission.csv
│   └── run_pipeline.py # orchestrator
├── report.qmd          # Quarto HTML report
├── pyproject.toml
└── uv.lock
```

---

## Usage

```bash
# EDA only
uv run python src/eda.py

# Model training
uv run python src/train.py

# Ensemble + submission
uv run python src/ensemble.py

# Full pipeline (EDA + train + ensemble)
uv run python src/run_pipeline.py

# Full pipeline + Quarto report
uv run python src/run_pipeline.py --report
```

---

## Feature Engineering

- **Label encoding** for 34 categorical columns (combined train+test vocabulary)
- **`hla_total_match`**: sum of all HLA matching scores
- **`hla_missing_count`**: number of missing HLA fields
- **`age_gap`**: donor age − recipient age at HCT
- **Target**: `log(efs_time)` — log-transformed survival time (months)

## Models

- **XGBoost** — regression on `log(efs_time)`, 600 trees, depth 6, lr 0.04
- **LightGBM** — regression on `log(efs_time)`, 600 trees, 63 leaves, lr 0.04
- **CatBoost** — regression on `log(efs_time)`, 600 iterations, depth 6, lr 0.04

All evaluated with 5-fold stratified CV (stratified on EFS indicator).

## Ensemble

Rank-based weighted ensemble (XGBoost 38% + LightGBM 38% + CatBoost 24%).
Each model's predictions are negated and rank-normalised to `[0, 1]` before combining.
