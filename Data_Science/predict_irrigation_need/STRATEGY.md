# PS6E4 Strategy: Predicting Irrigation Need

Inspired by the 1st place PS6E3 solution (Chris Deotte, KGMON Playbook).

## Competition Facts

- **Task:** 3-class classification (Low, Medium, High)
- **Metric:** Balanced accuracy
- **Train:** 630,000 rows, 20 features (12 numeric, 8 categorical)
- **Test:** 270,000 rows
- **Original dataset:** 10,000 rows (irrigation_prediction.csv)
- **Target distribution:** Low 58.7%, Medium 38.0%, High 3.3% (imbalanced)
- **Deadline:** April 30, 2026

## Key Differences from PS6E3

| Aspect | PS6E3 (Churn) | PS6E4 (Irrigation) |
|---|---|---|
| Classes | 2 (binary) | 3 (multiclass) |
| Metric | ROC-AUC | Balanced accuracy |
| Imbalance | Moderate | Severe (High class is 3.3%) |
| Numeric features | 3 (charges, tenure) | 12 (soil, weather, field) |
| Categorical features | 16 (mostly services) | 8 (soil, crop, region) |
| Original data size | 7,032 rows | 10,000 rows |

Balanced accuracy means the minority class (High) matters as much as the majority class (Low). Misclassifying High samples is 17x more costly per-sample than misclassifying Low samples.

---

## Phase 1: EDA (Day 1)

### 1.1 Basic Analysis
- Distribution of all 12 numeric features by target class
- Distribution of all 8 categorical features by target class
- Correlation matrix of numeric features
- Identify which features separate the 3 classes best

### 1.2 Original vs Synthetic Comparison
- Compare distributions of each feature between original (10K) and synthetic (630K)
- Identify rounding artifacts in numeric columns
- Check if the synthetic generator preserved the feature correlations
- Measure how many unique values exist per numeric column in original vs synthetic

### 1.3 Class Imbalance Analysis
- Per-class feature distributions (especially for the rare High class)
- Identify decision boundaries between classes
- Check if certain categorical combinations are strongly predictive of High

---

## Phase 2: Feature Store (Days 2 to 5)

Build reusable feature transformers. Each feature set gets tested independently before inclusion.

### 2.1 Snap Features (from PS6E3 Solution 1, Section 2.1)

Map each synthetic numeric value to its nearest value in the original 10K dataset.

```python
# For each numeric column:
snap_value = nearest value in original data
snap_diff  = synthetic_value - snap_value  # generator noise
```

Apply to all 12 numeric columns. The snap values recover the "true" original feature. The diff captures the generator's perturbation.

### 2.2 Digit and Decimal Extraction (from PS6E3 Solution 1, Section 2.2)

Extract decimal structure from numeric columns:

```python
frac    = x - floor(x)
d1      = floor(frac * 10)          # 1st decimal digit
d2      = floor(frac * 100) % 10    # 2nd decimal digit
frac100 = round(frac * 100)
mod10   = floor(x) % 10
```

Apply to: Soil_Moisture, Soil_pH, Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh, Field_Area_hectare, Previous_Irrigation_mm, Organic_Carbon, Electrical_Conductivity.

### 2.3 Target Encoding (Nested, Leak-Free) (from PS6E3 Solution 1, Section 2.3)

For multiclass: compute per-class TE (probability of each class given the category value).

- Single columns: all 8 categorical features
- Bigrams: top pairs (Soil_Type x Crop_Type, Crop_Type x Season, Irrigation_Type x Water_Source, Region x Season, Crop_Growth_Stage x Crop_Type)
- Trigrams: top triples (Soil_Type x Crop_Type x Season)
- Stats: mean, std, median per class
- Original data priors: class probability per category value from the 10K original dataset (zero leakage)

All TE must use nested inner 5-fold within each outer fold.

### 2.4 Domain-Specific Arithmetic Interactions

Agriculture-driven features:

```python
# Water balance
water_balance       = Rainfall_mm - Previous_Irrigation_mm
water_per_hectare   = (Rainfall_mm + Previous_Irrigation_mm) / Field_Area_hectare

# Weather stress
heat_stress         = Temperature_C * (1 - Humidity / 100)
evapotranspiration  = Temperature_C * Sunlight_Hours / (Humidity + 1)
wind_chill          = Wind_Speed_kmh * (100 - Humidity) / 100
drying_index        = Sunlight_Hours * Wind_Speed_kmh / (Humidity + 1)

# Soil quality
soil_conductivity_ratio = Electrical_Conductivity / (Soil_pH + 1)
soil_moisture_deficit   = 50 - Soil_Moisture  # deviation from "ideal" midpoint
organic_efficiency      = Organic_Carbon / (Electrical_Conductivity + 0.01)

# Field efficiency
irrigation_density  = Previous_Irrigation_mm / (Field_Area_hectare + 0.01)
rainfall_coverage   = Rainfall_mm / (Field_Area_hectare + 0.01)
```

### 2.5 Frequency and Count Encoding (from PS6E3 Solution 1, Section 2.7)

```python
freq = all_data[col].value_counts(normalize=True)
all_data[f"freq_{col}"] = all_data[col].map(freq)
```

Apply to all categorical columns and snapped numeric columns.

### 2.6 Categorical Cross-Features (from PS6E3 Solution 1, Section 2.6)

```python
df["bi_Soil_Crop"]    = df["Soil_Type"] + "__" + df["Crop_Type"]       # 24 combos
df["bi_Crop_Season"]  = df["Crop_Type"] + "__" + df["Season"]          # 18 combos
df["bi_Crop_Stage"]   = df["Crop_Type"] + "__" + df["Crop_Growth_Stage"]  # 24 combos
df["bi_Region_Season"] = df["Region"] + "__" + df["Season"]            # 15 combos
df["bi_Irr_Water"]    = df["Irrigation_Type"] + "__" + df["Water_Source"]  # 16 combos
```

Then target-encode all bigrams with nested CV.

### 2.7 Multi-Scale Binning (from PS6E3 Solution 1, Section 2.5)

Quantile bins (50, 100, 500, 1000 bins) on numeric columns. Convert to categorical, then target-encode. Fine-grained bins on Rainfall_mm and Soil_Moisture are likely most useful since they have the widest ranges.

### 2.8 Service/Boolean Aggregations (adapted from PS6E3 Solution 1, Section 2.8)

```python
# Count of "favorable" conditions
favorable_soil     = (Soil_Type == "Loamy").astype(int)
has_mulching       = (Mulching_Used == "Yes").astype(int)
rainfed            = (Irrigation_Type == "Rainfed").astype(int)
uses_groundwater   = (Water_Source == "Groundwater").astype(int)
```

### 2.9 Original Dataset Lookup (from PS6E3 Solution 1, Section 2.9)

Build a KDTree on standardized numeric columns of the 10K original rows. For each synthetic row, find K nearest original neighbors and attach:
- Their majority irrigation need class
- Mean distance to nearest neighbors
- Class distribution of K neighbors

### 2.10 Projection Features (from PS6E3 Solution 1, Section 2.12)

PCA (8 components) and Gaussian Random Projection (8 components) fitted on the original 10K dataset, projected onto synthetic rows. Gives geometric information about how each row relates to the original data manifold.

---

## Phase 3: Models (Days 5 to 20)

All models use 5-fold StratifiedKFold (SEED=42). Each model saves OOF predictions as `(n_samples, 3)` arrays (one probability per class) plus test predictions.

### 3.1 XGBoost (target: 10 to 15 models)

Best single model candidate. Key settings for multiclass balanced accuracy:
- `objective: "multi:softprob"`, `num_class: 3`
- `eval_metric: "mlogloss"`
- Low learning rates (0.005 to 0.02)
- `max_depth: 6 to 8`
- Per-class sample weights tuned via Optuna (see discussions/manual_class_weights.md)
  - v3 used auto balanced weights. v4 tunes w_low, w_med, w_high as hyperparameters.
  - The High class gets ~200 to 400x weight, far beyond the ~30x frequency ratio.
  - This directly optimizes balanced accuracy by pushing minority class recall higher.

Experiment variants:
- v100: Baseline with raw features + basic TE
- v200: Add snap features + digit extraction
- v300: Add domain arithmetic interactions
- v400: Full feature union (all feature sets combined)
- v500: Bigram/trigram TE focus
- v600: Anchor-based TE (snap + categorical)
- Multi-seed variants (3 seeds, rank-blend before averaging)

### 3.2 LightGBM (target: 8 to 12 models)

Leaf-wise growth for complementary predictions:
- `objective: "multiclass"`, `num_class: 3`
- v3: `is_unbalance: True`. v4: Optuna-tuned per-class `sample_weight`.
- Low learning rates (0.005 to 0.01)
- `num_leaves: 63 to 127`
- Try `boosting_type: "goss"` for diversity

### 3.3 CatBoost (target: 8 to 12 models)

Native categorical handling:
- `loss_function: "MultiClass"`
- v3: `auto_class_weights: "Balanced"`. v4: Optuna-tuned `class_weights` list.
- Pre-compute interaction features (needed for oblivious trees)
- Tune `random_strength` and `bagging_temperature`

### 3.4 RealMLP (target: 4 to 6 models)

Via pytabkit:
- PLR embeddings for numerics
- Internal 8-member ensemble
- Vary feature sets (raw, snap, digit, full)
- Strong diversity contributor

### 3.5 TabM (target: 3 to 5 models)

Via pytabkit:
- `tabm-normal` architecture
- Multiplicative interactions with k=32 basis components
- Round features boost this model most

### 3.6 Other Architectures (target: 5 to 10 models for diversity)

- FT-Transformer (2 models): self-attention across feature tokens
- Embedding MLP (2 models): categorical embeddings + numerics through MLP
- Random Forest via cuML or sklearn (1 to 2 models): bagging for diversity
- YDF with max_depth=2 (1 model): ultra-shallow for diversity

---

## Phase 4: Stacking and Ensemble (Days 20 to 28)

### Level 1: Feature Extraction
- KNN class probabilities from original data
- PCA/GRP projections
- Nested target encoding outputs
- These become extra columns for Level 2

### Level 2: Base Models
- All models from Phase 3
- Each saves `oof_{name}_v{ver}.npy` shape `(630000, 3)` and `pred_{name}_v{ver}.npy` shape `(270000, 3)`

### Level 3: Stacked Models
- Train new XGBoost/LightGBM models using raw features + Level 2 OOF predictions as extra columns
- 5-fold CV, same folds as Level 2
- Produces new OOF and test predictions

### Level 4: Meta-Learner
- L2 Logistic Regression on all Level 2 + Level 3 OOF predictions
- `multi_class: "multinomial"`, `solver: "lbfgs"`
- Hill climbing to select best model subset first
- Convert probabilities to class labels using argmax
- Optionally tune decision thresholds per class to maximize balanced accuracy

### Model Selection via Hill Climbing

Greedy forward selection optimizing balanced accuracy on OOF:
1. Start with the single best model
2. Add the model that improves ensemble balanced accuracy the most
3. Repeat until no model improves the score
4. This selects a diverse, complementary subset

### Threshold Optimization

Since metric is balanced accuracy (not log-loss), optimize class decision thresholds:
```python
# Instead of argmax, find thresholds that maximize balanced accuracy
# Grid search over threshold pairs for the 3-class boundary
```

This is a free post-processing boost that costs nothing in model complexity.

---

## Tools and Infrastructure

### tabml Usage
- **Version: 0.6.0** (commit `f957257`)
- Install: `pip install git+https://github.com/wguesdon/tabml.git@f957257`
- `tabml.models.XGBoostModel`, `LightGBMModel`, `CatBoostModel` for model wrappers (v3 trainers)
- `tabml.nn_models` for neural networks (RealMLP, TabM, FTTransformer, EmbeddingMLP)
- `tabml.ensemble.OOFEnsemble` for OOF generation and weight optimization
- `tabml.tracking.ModelTracker` for CV/LB score logging (SQLite)
- tabml stays general purpose. Competition-specific logic lives in the trainer scripts.
- Pin to this version when retraining old models. Newer tabml versions may change behavior.

### Training Infrastructure
- AWS SageMaker for all training (spot instances)
- Docker images built via EC2 spot jumpbox (`aws_training/scripts/ec2_build_push.sh`)
- S3 for data and artifact storage
- 5-fold StratifiedKFold CV throughout

### Versioning Requirement

**All trainer and ensemble versions must remain in the codebase.** When code is updated, the old version stays as a separate file. This is a hard requirement for reproducibility.

#### Trainers

```
aws/
  train_xgb.py       # Dispatcher: routes via SM_HP_TRAINER_VERSION
  train_xgb_v3.py    # Auto balanced weights, tabml wrapper
  train_xgb_v4.py    # Optuna-tuned per-class weights, raw xgboost
  train_xgb_v5.py    # + orig downweight, BA eval metric, poly/log features
```

| Version | Weighting | Features | Notes |
|---------|-----------|----------|-------|
| v3 | Auto balanced, tabml | features.py base | Original models, tabml 0.6.0 |
| v4 | Optuna-tuned class weights | features.py base | Raw xgb/lgb/catboost |
| v5 | Optuna-tuned + orig downweight | base + poly + log | BA eval metric, enable_categorical |

#### Ensemble scripts

```
scripts/
  ensemble_v1.py     # Hill climbing + Nelder-Mead thresholds
  ensemble_v2.py     # + differential evolution thresholds
  ensemble_v3.py     # + log-space bias tuning
```

#### Features

Features are additive. New features are added as new functions in `features.py`. Old functions are never modified. Each trainer version controls which feature functions it calls.

```python
# features.py grows over time, never shrinks:
create_features()          # v2+: domain, freq, boolean, digit, magic formula
add_original_te_priors()   # v2+: target mean from 10K original
create_2way_interactions() # v2+: all C(19,2) pairs
add_polynomial_features()  # v5+: squared terms for key numerics
add_log_transforms()       # v5+: log1p for right-skewed columns
```

#### Rules

1. Create `train_{model}_v{N}.py` with the new logic
2. Update the dispatcher to add the new version
3. Update Dockerfiles to COPY the new file
4. Never delete or modify old version files or old feature functions
5. Document which feature functions each trainer version calls

---

## Priorities

The 1st place PS6E3 solution identified feature engineering as the dominant driver. For PS6E4:

1. **Highest impact:** Magic formula features (BA=1.0 on original data, see discussions/original_data_formula.md)
2. **Highest impact:** 2-way interaction target encoding (+0.009 CV from v1 to v2)
3. **High impact:** Original dataset TE priors (zero leakage signal)
4. **High impact:** Domain arithmetic interactions (water balance, heat stress, soil quality)
5. **High impact:** Optuna-tuned per-class weights (discussions/manual_class_weights.md)
6. **Medium impact:** Digit extraction, frequency encoding, feature-engine binning
7. **Medium impact:** Model diversity (GBDT + NN architectures + different seeds)
8. **Medium impact:** Threshold optimization via differential evolution
9. **Lower impact:** PCA/projection features, snap features, radix interactions

The class imbalance (High = 3.3%) combined with balanced accuracy metric means getting the High class right is critical. Feature engineering and class weight tuning should focus on separating High from Medium.

## Current Results

| Model | CV | LB | Version |
|-------|------|------|---------|
| 14-model blend + diff evolution thresholds | 0.97930 | 0.97876 | v3 ensemble |
| LR stacker L3 + diff evolution thresholds | 0.97911 | 0.97828 | v3 ensemble |
| Best single CatBoost | 0.97834 | - | v3 |
| Target: Chris Deotte #1 | - | 0.97970 | - |

Gap to #1: 0.00094 on public LB.
