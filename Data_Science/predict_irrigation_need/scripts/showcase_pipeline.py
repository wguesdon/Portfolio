# %% [markdown]
# # PS6E4 Irrigation Prediction - Full Pipeline Showcase
#
# End-to-end pipeline: EDA, feature engineering, 3 GBDT models with OOF,
# hill climbing ensemble, threshold optimization, and submission.
#
# Competition: Playground Series S6E4 - Predicting Irrigation Need
# Metric: Balanced Accuracy (3-class: Low, Medium, High)

# %%
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

# Force unbuffered output so logs appear immediately on EC2
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

_START = time.time()


def log(msg):
    """Print a timestamped log message."""
    elapsed = time.time() - _START
    mins, secs = divmod(int(elapsed), 60)
    print(f"[{mins:3d}m{secs:02d}s] {msg}", flush=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    DecisionTreeDiscretiser,
)
from feature_engine.encoding import (
    CountFrequencyEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# %%
# CLI arguments
import argparse

_parser = argparse.ArgumentParser(description="PS6E4 Showcase Pipeline")
_parser.add_argument(
    "--test", action="store_true",
    help="Test mode: 10%% stratified subsample, 3 folds, 200 estimators. "
         "Runs in ~2 min to validate the full pipeline before a real run.",
)
_parser.add_argument(
    "--sample-frac", type=float, default=0.1,
    help="Fraction of data to use in test mode (default: 0.1)",
)
# Parse only known args so notebook %% cells don't break
ARGS, _ = _parser.parse_known_args()
TEST_MODE = ARGS.test

if TEST_MODE:
    print("=" * 50)
    print("  TEST MODE: subsampled data, fast training")
    print("=" * 50)

# %%
# Configuration
# Supports running from scripts/ dir or from the competition root
_script_path = Path(__file__).resolve() if "__file__" in dir() else Path.cwd()
BASE_DIR = _script_path.parent.parent if _script_path.name.endswith(".py") else _script_path
if not (BASE_DIR / "data").exists() and (Path.cwd() / "data").exists():
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "raw"
PRED_DIR = BASE_DIR / "predictions"
SUB_DIR = BASE_DIR / "submissions"
for d in [PRED_DIR, SUB_DIR, BASE_DIR / "output" / "eda"]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = "Irrigation_Need"
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET_INV = {0: "Low", 1: "Medium", 2: "High"}
N_SPLITS = 3 if TEST_MODE else 5
N_ESTIMATORS = 200 if TEST_MODE else 15000
EARLY_STOP = 50 if TEST_MODE else 500
SEED = 42

NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

# %% [markdown]
# ## 1. Load Data

# %%
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
orig = pd.read_csv(DATA_DIR / "irrigation_prediction.csv")

train[TARGET] = train[TARGET].map(TARGET_MAP)
orig[TARGET] = orig[TARGET].map(TARGET_MAP)

# Subsample in test mode for fast pipeline validation
if TEST_MODE:
    from sklearn.model_selection import train_test_split as _tts
    frac = ARGS.sample_frac
    train, _ = _tts(train, train_size=frac, stratify=train[TARGET], random_state=SEED)
    test = test.sample(frac=frac, random_state=SEED)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

log(f"Data loaded: Train={train.shape}, Test={test.shape}, Orig={orig.shape}")
if TEST_MODE:
    print(f"[TEST] {N_SPLITS} folds, {N_ESTIMATORS} estimators, early_stop={EARLY_STOP}")

# %% [markdown]
# ## 2. EDA Summary
#
# Key findings from our EDA analysis:
# - Class imbalance: Low 58.7%, Medium 37.9%, High 3.3%
# - Top features: Soil_Moisture (F=82K), Wind_Speed_kmh (F=22K), Temperature_C (F=22K)
# - Original dataset has only 10K rows. Synthetic generator distorted the rare High class.
# - 6 numeric features have identical unique counts in original vs synthetic (snap features effective).

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Target distribution
train[TARGET].map(TARGET_INV).value_counts().plot.bar(ax=axes[0], color=["#4C72B0", "#DD8452", "#C44E52"])
axes[0].set_title("Target Distribution")
axes[0].set_ylabel("Count")

# Top numeric features by variance
feature_importance = train[NUMS].std().sort_values(ascending=False)
feature_importance.plot.barh(ax=axes[1], color="#4C72B0")
axes[1].set_title("Numeric Feature Std Dev")

# Class balance comparison: original vs synthetic
orig_dist = orig[TARGET].value_counts(normalize=True).sort_index()
train_dist = train[TARGET].value_counts(normalize=True).sort_index()
pd.DataFrame({"Original": orig_dist, "Synthetic": train_dist}).plot.bar(ax=axes[2])
axes[2].set_title("Original vs Synthetic Distribution")
axes[2].set_xticklabels(["Low", "Medium", "High"], rotation=0)

plt.tight_layout()
plt.savefig(BASE_DIR / "output" / "eda" / "showcase_eda.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Feature Engineering
#
# Four layers of features:
# 1. **Domain interactions** - physics-inspired feature crosses
# 2. **feature-engine transformers** - discretisation, count-frequency, decision tree encoding
# 3. **Original dataset TE priors** - target mean per feature from the 10K original rows
# 4. **2-way interaction columns** - all C(19,2) pairs, target-encoded inside CV loop

# %%
def create_domain_features(df):
    """Create domain-specific interaction features."""
    out = df.copy()
    out["water_balance"] = out["Rainfall_mm"] - out["Previous_Irrigation_mm"]
    out["water_per_hectare"] = (
        (out["Rainfall_mm"] + out["Previous_Irrigation_mm"])
        / (out["Field_Area_hectare"] + 0.01)
    )
    out["total_water"] = out["Rainfall_mm"] + out["Previous_Irrigation_mm"]
    out["heat_stress"] = out["Temperature_C"] * (1 - out["Humidity"] / 100)
    out["evapotranspiration"] = (
        out["Temperature_C"] * out["Sunlight_Hours"] / (out["Humidity"] + 1)
    )
    out["drying_index"] = (
        out["Sunlight_Hours"] * out["Wind_Speed_kmh"] / (out["Humidity"] + 1)
    )
    out["temp_wind"] = out["Temperature_C"] * out["Wind_Speed_kmh"]
    out["moisture_deficit"] = 50 - out["Soil_Moisture"]
    out["moisture_temp_ratio"] = out["Soil_Moisture"] / (out["Temperature_C"] + 1)
    out["irrigation_density"] = (
        out["Previous_Irrigation_mm"] / (out["Field_Area_hectare"] + 0.01)
    )

    # Frequency encoding for categoricals
    for col in CATS:
        freq = out[col].value_counts(normalize=True)
        out[f"freq_{col}"] = out[col].map(freq)

    # Boolean flags
    out["is_rainfed"] = (out["Irrigation_Type"] == "Rainfed").astype(np.int8)
    out["low_moisture"] = (out["Soil_Moisture"] < 25).astype(np.int8)
    out["high_temp"] = (out["Temperature_C"] > 35).astype(np.int8)

    # Digit extraction
    for col in NUMS:
        vals = out[col].values
        frac = vals - np.floor(vals)
        out[f"{col}_d1"] = np.floor(frac * 10).astype(np.int8)

    return out


def create_feature_engine_features(train_df, test_df, y_train):
    """Create features using feature-engine transformers.

    Applied to train+test jointly (discretisers) or fitted on train only
    (encoders that use the target). All transformers produce numeric output
    safe for any downstream model.

    Args:
        train_df: Training DataFrame with numeric and categorical columns.
        test_df: Test DataFrame.
        y_train: Training target (numeric, same length as train_df).

    Returns:
        Tuple of (train_df, test_df, list of new column names).
    """
    new_cols = []
    # Use column subsets so feature-engine doesn't complain about column mismatch
    nums_present = [c for c in NUMS if c in train_df.columns]
    cat_cols_present = [c for c in CATS if c in train_df.columns]

    # --- Discretisation: bin numerics into quantile and equal-width buckets ---
    # EqualFrequencyDiscretiser: each bin has roughly the same number of samples.
    efd = EqualFrequencyDiscretiser(q=10, variables=nums_present, return_boundaries=False)
    efd.fit(train_df[nums_present])
    efd_train = efd.transform(train_df[nums_present])
    efd_test = efd.transform(test_df[nums_present])
    for col in nums_present:
        name = f"efd10_{col}"
        train_df[name] = efd_train[col].values
        test_df[name] = efd_test[col].values
        new_cols.append(name)

    # EqualWidthDiscretiser: fixed-width bins across the value range.
    ewd = EqualWidthDiscretiser(bins=15, variables=nums_present, return_boundaries=False)
    ewd.fit(train_df[nums_present])
    ewd_train = ewd.transform(train_df[nums_present])
    ewd_test = ewd.transform(test_df[nums_present])
    for col in nums_present:
        name = f"ewd15_{col}"
        train_df[name] = ewd_train[col].values
        test_df[name] = ewd_test[col].values
        new_cols.append(name)

    # --- Categorical encoding: count-frequency ---
    # CountFrequencyEncoder: replaces categories with their frequency.
    cfe = CountFrequencyEncoder(encoding_method="frequency", variables=cat_cols_present)
    cfe.fit(train_df[cat_cols_present])
    cfe_train = cfe.transform(train_df[cat_cols_present])
    cfe_test = cfe.transform(test_df[cat_cols_present])
    for col in cat_cols_present:
        name = f"cfe_{col}"
        train_df[name] = cfe_train[col].values
        test_df[name] = cfe_test[col].values
        new_cols.append(name)

    # --- DecisionTreeDiscretiser: supervised binning ---
    # Fits a shallow decision tree per feature to find optimal split points.
    dtd = DecisionTreeDiscretiser(
        variables=nums_present,
        regression=False,
        scoring="balanced_accuracy",
        cv=3,
        param_grid={"max_depth": [2, 3]},
    )
    dtd.fit(train_df[nums_present], y_train)
    dtd_train = dtd.transform(train_df[nums_present])
    dtd_test = dtd.transform(test_df[nums_present])
    for col in nums_present:
        name = f"dtd_{col}"
        train_df[name] = dtd_train[col].values
        test_df[name] = dtd_test[col].values
        new_cols.append(name)

    print(f"  feature-engine features: {len(new_cols)} columns")
    return train_df, test_df, new_cols


def add_original_te_priors(train_df, test_df, orig_df):
    """Add target mean from original dataset as features."""
    orig_enc = orig_df.copy()
    if not pd.api.types.is_numeric_dtype(orig_enc[TARGET]):
        orig_enc[TARGET] = orig_enc[TARGET].map(TARGET_MAP)
    orig_enc[TARGET] = pd.to_numeric(orig_enc[TARGET], errors="coerce")
    orig_enc = orig_enc.dropna(subset=[TARGET])

    te_cols = []
    for col in NUMS + CATS:
        if col not in orig_enc.columns:
            continue
        te_name = f"TE_ORIG_{col}"
        means = orig_enc.groupby(col)[TARGET].mean().astype("float32")
        means.name = te_name
        train_df = train_df.merge(means, on=col, how="left")
        train_df[te_name] = train_df[te_name].fillna(0.5)
        test_df = test_df.merge(means, on=col, how="left")
        test_df[te_name] = test_df[te_name].fillna(0.5)
        te_cols.append(te_name)
    return train_df, test_df, te_cols


def create_2way_interactions(train_df, test_df):
    """Create all 2-way feature interactions as factorized integers."""
    all_cols = NUMS + CATS
    interaction_cols = []
    for col_a, col_b in combinations(all_cols, 2):
        name = f"{col_a}-{col_b}"
        combined = pd.concat(
            [
                train_df[col_a].astype(str) + "_" + train_df[col_b].astype(str),
                test_df[col_a].astype(str) + "_" + test_df[col_b].astype(str),
            ],
            ignore_index=True,
        )
        encoded, _ = combined.factorize()
        if pd.Series(encoded).nunique() > len(combined) // 2:
            continue
        train_df[name] = encoded[: len(train_df)]
        test_df[name] = encoded[len(train_df) :]
        interaction_cols.append(name)
    return train_df, test_df, interaction_cols


# %%
# Apply feature engineering
train = create_domain_features(train)
test = create_domain_features(test)

log("Starting feature-engine transformers...")
train, test, fe_cols = create_feature_engine_features(train, test, y_train=train[TARGET])

train, test, te_prior_cols = add_original_te_priors(train, test, orig)
print(f"Original TE priors: {len(te_prior_cols)} columns")

train, test, interaction_cols = create_2way_interactions(train, test)
print(f"2-way interactions: {len(interaction_cols)} columns")

# Build feature list (exclude raw categoricals, id, target)
exclude = {"id", TARGET} | set(CATS)
feature_cols = [c for c in train.columns if c not in exclude]
log(f"Feature engineering complete: {len(feature_cols)} total features")

X = train[feature_cols]
y = train[TARGET].values
X_test = test[feature_cols]
test_ids = test["id"]

# Prepare original data for injection into training folds
orig_fe = create_domain_features(orig)
for col in feature_cols:
    if col not in orig_fe.columns:
        orig_fe[col] = 0
X_orig = orig_fe[feature_cols]
y_orig = orig[TARGET].values
print(f"Original data for injection: {len(X_orig)} rows")

# %% [markdown]
# ## 4. Model Training with OOF Predictions
#
# Train XGBoost, LightGBM, and CatBoost with 5-fold stratified CV.
# Each fold: append original data to training, apply target encoding
# on interaction columns, compute balanced sample weights.

# %%
def apply_fold_te(X_tr, X_va, X_te, y_tr, te_cols, seed=42):
    """Target-encode interaction columns inside a CV fold."""
    te_present = [c for c in te_cols if c in X_tr.columns]
    if not te_present:
        return X_tr, X_va, X_te

    encoder = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)

    te_tr = pd.DataFrame(encoder.fit_transform(X_tr[te_present], y_tr), index=X_tr.index)
    te_tr.columns = encoder.get_feature_names_out(te_present)
    te_va = pd.DataFrame(encoder.transform(X_va[te_present]), index=X_va.index)
    te_va.columns = encoder.get_feature_names_out(te_present)
    te_te = pd.DataFrame(encoder.transform(X_te[te_present]), index=X_te.index)
    te_te.columns = encoder.get_feature_names_out(te_present)

    X_tr = pd.concat([X_tr.drop(columns=te_present).reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    X_va = pd.concat([X_va.drop(columns=te_present).reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_te.drop(columns=te_present).reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
    return X_tr, X_va, X_te


def train_model_oof(model_name, create_model_fn, X, y, X_test, X_orig, y_orig,
                    interaction_cols, n_splits=N_SPLITS, seed=42):
    """Train a model with OOF predictions, original data injection, and in-fold TE.

    Args:
        model_name: Name for logging and file saving.
        create_model_fn: Callable that returns (model, fit_kwargs_fn).
            fit_kwargs_fn(X_tr, y_tr, X_va, y_va) returns kwargs for model.fit().
        X: Training features DataFrame.
        y: Training target array.
        X_test: Test features DataFrame.
        X_orig: Original dataset features DataFrame.
        y_orig: Original dataset target array.
        interaction_cols: Columns to target-encode inside each fold.
        n_splits: Number of CV folds.
        seed: Random state.

    Returns:
        Tuple of (oof_preds, test_preds, cv_score, fold_scores).
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    fold_scores = []

    log(f"  {model_name}: starting {n_splits}-fold CV")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx].copy()
        X_va = X.iloc[val_idx].copy()
        X_te = X_test.copy()

        # Inject original data into training fold
        X_tr = pd.concat([X_tr, X_orig], axis=0).reset_index(drop=True)
        y_tr = np.concatenate([y_tr, y_orig])

        # In-fold target encoding
        X_tr, X_va, X_te = apply_fold_te(X_tr, X_va, X_te, pd.Series(y_tr), interaction_cols, seed)

        # Train
        model, fit_kwargs_fn = create_model_fn()
        fit_kwargs = fit_kwargs_fn(X_tr, y_tr, X_va, y[val_idx])
        model.fit(X_tr, y_tr, **fit_kwargs)

        oof_preds[val_idx] = model.predict_proba(X_va)
        test_preds += model.predict_proba(X_te) / n_splits

        score = balanced_accuracy_score(y[val_idx], oof_preds[val_idx].argmax(axis=1))
        fold_scores.append(score)
        log(f"  {model_name} fold {fold + 1}/{n_splits}: {score:.5f}")

        del model, X_tr, X_va, X_te
        gc.collect()

    cv_score = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
    log(f"  {model_name} CV: {cv_score:.5f} (+/- {np.std(fold_scores):.5f})")

    np.save(PRED_DIR / f"oof_{model_name}.npy", oof_preds)
    np.save(PRED_DIR / f"pred_{model_name}.npy", test_preds)
    return oof_preds, test_preds, cv_score, fold_scores


# %% [markdown]
# ### 4a. XGBoost

# %%
print("=" * 50)
print("Training XGBoost")
print("=" * 50)


def create_xgb():
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.01,
        n_estimators=N_ESTIMATORS,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=1024,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if __import__("shutil").which("nvidia-smi") else "cpu",
        verbosity=0,
    )

    def fit_kwargs(X_tr, y_tr, X_va, y_va):
        weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        return {
            "eval_set": [(X_va, y_va)],
            "sample_weight": weights,
            "verbose": False,
        }

    return model, fit_kwargs


xgb_oof, xgb_test, xgb_cv, xgb_folds = train_model_oof(
    "xgb_showcase", create_xgb, X, y, X_test, X_orig, y_orig, interaction_cols,
)

# %% [markdown]
# ### 4b. LightGBM

# %%
print("=" * 50)
print("Training LightGBM")
print("=" * 50)


def create_lgb():
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        boosting_type="gbdt",
        num_leaves=77,
        learning_rate=0.01,
        n_estimators=N_ESTIMATORS,
        min_child_samples=56,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=7.0,
        is_unbalance=True,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    def fit_kwargs(X_tr, y_tr, X_va, y_va):
        return {
            "eval_set": [(X_va, y_va)],
            "callbacks": [
                __import__("lightgbm").early_stopping(EARLY_STOP, verbose=False),
                __import__("lightgbm").log_evaluation(0),
            ],
        }

    return model, fit_kwargs


lgb_oof, lgb_test, lgb_cv, lgb_folds = train_model_oof(
    "lgb_showcase", create_lgb, X, y, X_test, X_orig, y_orig, interaction_cols,
)

# %% [markdown]
# ### 4c. CatBoost

# %%
print("=" * 50)
print("Training CatBoost")
print("=" * 50)


def create_cat():
    model = CatBoostClassifier(
        loss_function="MultiClass",
        classes_count=3,
        auto_class_weights="Balanced",
        depth=6,
        learning_rate=0.04,
        iterations=N_ESTIMATORS,
        l2_leaf_reg=0.5,
        random_strength=8.0,
        bagging_temperature=0.4,
        border_count=239,
        random_seed=SEED,
        verbose=0,
    )

    def fit_kwargs(X_tr, y_tr, X_va, y_va):
        return {
            "eval_set": (X_va, y_va),
            "early_stopping_rounds": EARLY_STOP,
        }

    return model, fit_kwargs


cat_oof, cat_test, cat_cv, cat_folds = train_model_oof(
    "cat_showcase", create_cat, X, y, X_test, X_orig, y_orig, interaction_cols,
)

# %% [markdown]
# ## 5. Model Comparison

# %%
results = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "CatBoost"],
    "CV Score": [xgb_cv, lgb_cv, cat_cv],
    "Std": [np.std(xgb_folds), np.std(lgb_folds), np.std(cat_folds)],
})
print(results.to_string(index=False))

# OOF correlation between models
corr_data = {
    "XGB_Low": xgb_oof[:, 0], "LGB_Low": lgb_oof[:, 0], "CAT_Low": cat_oof[:, 0],
    "XGB_Med": xgb_oof[:, 1], "LGB_Med": lgb_oof[:, 1], "CAT_Med": cat_oof[:, 1],
    "XGB_High": xgb_oof[:, 2], "LGB_High": lgb_oof[:, 2], "CAT_High": cat_oof[:, 2],
}
corr_df = pd.DataFrame(corr_data)
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), mask=mask, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=0.8, vmax=1.0, ax=ax)
ax.set_title("Base Model Prediction Correlation")
plt.tight_layout()
plt.savefig(BASE_DIR / "output" / "eda" / "showcase_correlation.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Hill Climbing Ensemble
#
# Greedy forward selection: start with the best single model, iteratively add
# the model that improves the ensemble balanced accuracy the most.

# %%
def hill_climbing_ensemble(oof_list, y_true, n_iterations=2000, seed=42):
    """Find optimal model weights via hill climbing.

    Args:
        oof_list: List of (name, oof_preds) tuples. Each oof_preds is (N, 3).
        y_true: Ground truth labels.
        n_iterations: Number of search iterations.
        seed: Random state.

    Returns:
        Tuple of (best_weights dict, best_score).
    """
    rng = np.random.RandomState(seed)
    names = [name for name, _ in oof_list]
    oofs = [oof for _, oof in oof_list]
    n_models = len(oofs)

    # Start with equal weights
    weights = np.ones(n_models) / n_models
    blend = sum(w * o for w, o in zip(weights, oofs))
    best_score = balanced_accuracy_score(y_true, blend.argmax(axis=1))
    best_weights = weights.copy()

    for i in range(n_iterations):
        # Perturb weights
        lr = max(0.01, 0.1 * (1 - i / n_iterations))
        new_weights = best_weights + rng.randn(n_models) * lr
        new_weights = np.clip(new_weights, 0, None)
        new_weights /= new_weights.sum()

        blend = sum(w * o for w, o in zip(new_weights, oofs))
        score = balanced_accuracy_score(y_true, blend.argmax(axis=1))

        if score > best_score:
            best_score = score
            best_weights = new_weights.copy()

    result = {name: float(w) for name, w in zip(names, best_weights)}
    return result, best_score


# %%
oof_models = [
    ("XGBoost", xgb_oof),
    ("LightGBM", lgb_oof),
    ("CatBoost", cat_oof),
]

weights, ensemble_cv = hill_climbing_ensemble(oof_models, y)
log(f"Hill Climbing Ensemble CV: {ensemble_cv:.5f}")
print(f"Weights: {weights}")

# Apply weights to test predictions
test_blend = sum(
    weights[name] * pred
    for name, pred in [("XGBoost", xgb_test), ("LightGBM", lgb_test), ("CatBoost", cat_test)]
)

# %% [markdown]
# ## 7. Threshold Optimization
#
# Optimize class-specific multipliers on the blended probabilities to maximize
# balanced accuracy. Uses Nelder-Mead optimization.

# %%
oof_blend = sum(
    weights[name] * oof
    for name, oof in oof_models
)

baseline_ensemble = balanced_accuracy_score(y, oof_blend.argmax(axis=1))
print(f"Ensemble before threshold optimization: {baseline_ensemble:.5f}")


def optimize_thresholds(class_weights):
    weighted = oof_blend * class_weights
    preds = weighted.argmax(axis=1)
    return -balanced_accuracy_score(y, preds)


result = minimize(optimize_thresholds, [1.0, 1.0, 1.0], method="Nelder-Mead")
best_thresholds = result.x
optimized_preds = (oof_blend * best_thresholds).argmax(axis=1)
optimized_cv = balanced_accuracy_score(y, optimized_preds)

print(f"Optimized thresholds: {best_thresholds}")
log(f"Ensemble after threshold optimization: {optimized_cv:.5f}")
log(f"Gain from threshold optimization: +{optimized_cv - baseline_ensemble:.5f}")

# %% [markdown]
# ## 8. Final Submission

# %%
# Apply thresholds to test blend
final_test_probs = test_blend * best_thresholds
final_test_classes = final_test_probs.argmax(axis=1)

le = LabelEncoder()
le.fit(["Low", "Medium", "High"])
labels = le.inverse_transform(final_test_classes)

submission = pd.DataFrame({"id": test_ids, TARGET: labels})
sub_path = SUB_DIR / "sub_showcase_ensemble.csv"
submission.to_csv(sub_path, index=False)
print(f"Submission saved: {sub_path} ({len(submission)} rows)")
print(submission[TARGET].value_counts())

# %% [markdown]
# ## 9. OOF Diagnostics

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(y, optimized_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
axes[0].set_title(f"Ensemble Confusion Matrix (CV={optimized_cv:.5f})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Per-class accuracy
for cls in range(3):
    mask = y == cls
    cls_acc = (optimized_preds[mask] == cls).mean()
    print(f"  Class {TARGET_INV[cls]}: accuracy={cls_acc:.4f} (n={mask.sum()})")

# Weight distribution
model_names = list(weights.keys())
model_weights = [weights[n] for n in model_names]
axes[1].barh(model_names, model_weights, color=["#4C72B0", "#DD8452", "#C44E52"])
axes[1].set_title("Ensemble Weights (Hill Climbing)")
axes[1].set_xlabel("Weight")

plt.tight_layout()
plt.savefig(BASE_DIR / "output" / "eda" / "showcase_ensemble.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Component | Details |
# |-----------|---------|
# | Features | Domain interactions + feature-engine (discretisers, encoders) + original TE priors + 2-way interaction TE |
# | Models | XGBoost, LightGBM, CatBoost (5-fold CV, original data injection) |
# | Ensemble | Hill climbing weight optimization |
# | Post-processing | Nelder-Mead threshold optimization for balanced accuracy |

# %%
print("\n" + "=" * 50)
print("PIPELINE SUMMARY")
print("=" * 50)
print(f"Features:       {len(feature_cols)}")
print(f"feature-engine: {len(fe_cols)} columns (discretisers + encoders)")
print(f"Interaction TE: {len(interaction_cols)} columns")
print(f"Orig TE priors: {len(te_prior_cols)} columns")
print(f"")
print(f"XGBoost CV:     {xgb_cv:.5f}")
print(f"LightGBM CV:    {lgb_cv:.5f}")
print(f"CatBoost CV:    {cat_cv:.5f}")
print(f"")
print(f"Ensemble CV:    {baseline_ensemble:.5f}")
print(f"+ Thresholds:   {optimized_cv:.5f}")
print(f"Weights:        {weights}")
print(f"Thresholds:     {list(best_thresholds)}")
print(f"")
print(f"Submission:     {sub_path}")
