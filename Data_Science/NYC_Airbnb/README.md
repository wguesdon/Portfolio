# NYC Airbnb 2019: EDA & Price Prediction

Exploratory analysis and price prediction for ~49,000 NYC Airbnb listings using the
Inside Airbnb 2019 dataset.

---

## Results

| Model | CV R² | CV RMSE |
|---|---|---|
| Ridge Regression | 0.560 | 0.459 |
| Random Forest | 0.613 | 0.431 |
| XGBoost | 0.620 | 0.426 |
| **LightGBM** | **0.622** | **0.425** |

**Improvement over original notebook**: +7 pp R² (0.55 → 0.622) without manual price-range splitting, using richer features and gradient boosting.

**Best predictor**: Location. Latitude, longitude, distance to Manhattan, and target-encoded neighbourhood consistently rank as the top features.

---

## Project Structure

```
NYC_Airbnb/
├── data/
│   └── AB_NYC_2019.csv
├── output/
│   ├── eda/          # 8 EDA plots
│   └── model/        # 5 model plots + cv_results.csv
├── src/
│   ├── config.py       # paths, constants, colour palettes
│   ├── features.py     # data loading & feature engineering
│   ├── eda.py          # EDA script → output/eda/
│   ├── train.py        # model training & evaluation → output/model/
│   └── run_pipeline.py # runs everything
├── report.qmd          # Quarto HTML report
├── pyproject.toml
└── uv.lock
```

---

## Usage

```bash
# EDA only
uv run python src/eda.py

# Model training only
uv run python src/train.py

# Full pipeline (EDA + training)
uv run python src/run_pipeline.py

# Full pipeline + Quarto report
uv run python src/run_pipeline.py --report

# Render report from pre-generated outputs
uv run python src/run_pipeline.py --report-only
# or directly:
quarto render report.qmd
```

---

## Feature Engineering

- **Log transforms** on all skewed count columns
- **Distance to Manhattan** (Haversine from Times Square)
- **Target encoding** for the 221 neighbourhoods (mean log-price per neighbourhood)
- **Interaction term**: borough × room type
- **Derived**: reviews per listing, availability rate

## Models

- **Ridge Regression** — linear baseline with standardisation
- **Random Forest** — 300 trees, max depth 14
- **XGBoost** — 500 rounds, lr 0.05, subsampling 0.8
- **LightGBM** — 500 rounds, 63 leaves, lr 0.05 *(best)*

All evaluated with 5-fold cross-validation on log(price + 1).
