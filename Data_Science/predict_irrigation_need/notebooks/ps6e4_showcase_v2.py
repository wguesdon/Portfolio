# %% [markdown]
# # PS6E4: Predicting Irrigation Need
# ## Multi-Model Ensemble with Feature Engineering
#
# **Competition:** [Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4)
# **Metric:** Balanced Accuracy (3-class: Low, Medium, High)
# **Approach:** Feature-engine pipeline + 4 models (XGBoost, LightGBM, CatBoost, RealMLP) + Hill Climbing Ensemble
#
# This notebook shows the full pipeline. Models were trained on AWS SageMaker (A10G GPU).
# Only the EDA and ensemble sections run on Kaggle. Training code is included for reference.
#
# **Training script:** `scripts/showcase_v2_train.py`
#
# Generated with [Claude Code](https://claude.ai/claude-code) (Opus 4.6).

# %% [markdown]
# ## 1. Setup

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

# Seaborn theme for clean, modern plots
sns.set_theme(
    style="whitegrid",
    palette="muted",
    font_scale=1.1,
    rc={
        "figure.figsize": (12, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    },
)

# Consistent color palette for irrigation classes
CLASS_COLORS = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}
CLASS_ORDER = ["Low", "Medium", "High"]
CLASS_PALETTE = [CLASS_COLORS[c] for c in CLASS_ORDER]

# %% [markdown]
# ## 2. Load Data

# %%
# Detect Kaggle vs local environment
KAGGLE_DIR = Path("/kaggle/input/playground-series-s6e4")
DATASET_DIR = Path("/kaggle/input/ps6e4-showcase-v2-predictions")

if KAGGLE_DIR.exists():
    DATA_DIR = KAGGLE_DIR
    PRED_DIR = DATASET_DIR
    ORIG_PATH = DATASET_DIR / "irrigation_prediction.csv"
    ON_KAGGLE = True
else:
    BASE_DIR = Path(".")
    if not (BASE_DIR / "data").exists():
        BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    PRED_DIR = BASE_DIR / "predictions"
    ORIG_PATH = DATA_DIR / "irrigation_prediction.csv"
    ON_KAGGLE = False

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
orig = pd.read_csv(ORIG_PATH)

NUMERIC_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

le = LabelEncoder()
le.fit(["High", "Low", "Medium"])
y = le.transform(train["Irrigation_Need"])

print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Original: {orig.shape}")

# %% [markdown]
# ## 3. Exploratory Data Analysis
#
# The dataset has 630K synthetic training rows generated from a 10K original dataset.
# The target has 3 classes with significant imbalance.

# %% [markdown]
# ### 3.1 Target Distribution

# %%
fig, ax = plt.subplots(figsize=(8, 4))
counts = train["Irrigation_Need"].value_counts()
pcts = counts / len(train) * 100

bars = sns.barplot(
    x=pcts[CLASS_ORDER].values, y=CLASS_ORDER,
    palette=CLASS_PALETTE, ax=ax, edgecolor="white", linewidth=1.5,
)
for i, cls in enumerate(CLASS_ORDER):
    ax.text(
        pcts[cls] + 0.8, i, f"{counts[cls]:,}  ({pcts[cls]:.1f}%)",
        va="center", fontweight="bold", fontsize=11,
    )

ax.set_xlabel("Percentage of Training Samples")
ax.set_title("Target Distribution: Irrigation Need", fontweight="bold")
ax.set_xlim(0, 70)
sns.despine(left=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# The High class represents only 3.3% of samples. Balanced accuracy gives equal
# weight to each class, so correctly predicting the rare High class is critical.

# %% [markdown]
# ### 3.2 Numeric Feature Distributions by Class

# %%
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    ax = axes[i]
    for cls in CLASS_ORDER:
        subset = train.loc[train["Irrigation_Need"] == cls, col]
        sns.kdeplot(
            subset, ax=ax, label=cls, color=CLASS_COLORS[cls],
            fill=True, alpha=0.2, linewidth=1.5,
        )
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    if i == 0:
        ax.legend(title="Class", framealpha=0.9)
    else:
        ax.legend().set_visible(False)

# Hide unused subplot
axes[-1].set_visible(False)

fig.suptitle(
    "Feature Distributions by Irrigation Class",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key observations:**
# - **Soil_Moisture** is the strongest separator. Low moisture strongly predicts High irrigation need.
# - **Wind_Speed_kmh** and **Temperature_C** also show clear class separation.
# - **Soil_pH** and **Organic_Carbon** have weak discriminative power.

# %% [markdown]
# ### 3.3 Correlation Heatmap

# %%
fig, ax = plt.subplots(figsize=(10, 8))
corr = train[NUMERIC_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
)
ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# Most features are weakly correlated. This is good for building diverse models.

# %% [markdown]
# ### 3.4 Top Features: Violin Plots by Class

# %%
top_features = [
    "Soil_Moisture", "Wind_Speed_kmh", "Temperature_C",
    "Humidity", "Rainfall_mm", "Previous_Irrigation_mm",
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    ax = axes[i]
    sns.violinplot(
        data=train, x="Irrigation_Need", y=col, order=CLASS_ORDER,
        palette=CLASS_COLORS, ax=ax, inner="quartile", linewidth=1,
        saturation=0.8, cut=0,
    )
    ax.set_title(col, fontweight="bold", fontsize=12)
    ax.set_xlabel("")

fig.suptitle(
    "Top Feature Distributions by Class",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# The violin plots reveal that **High** irrigation samples cluster tightly at low
# soil moisture and high wind speed. These are the features the magic formula uses.

# %% [markdown]
# ### 3.5 Categorical Feature Composition

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(CAT_COLS):
    ax = axes[i]
    ct = pd.crosstab(train[col], train["Irrigation_Need"], normalize="index")
    ct = ct[CLASS_ORDER]
    ct.plot(
        kind="barh", stacked=True, ax=ax,
        color=CLASS_PALETTE, edgecolor="white", linewidth=0.5,
    )
    ax.set_title(col, fontweight="bold", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    if i != 0:
        ax.legend().set_visible(False)
    else:
        ax.legend(title="Class", bbox_to_anchor=(1.0, 1.0), fontsize=9)

fig.suptitle(
    "Class Composition per Category Level",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key observations:**
# - **Crop_Growth_Stage**: Harvest and Sowing stages are almost entirely Low irrigation need.
# - **Mulching_Used**: "Yes" strongly associates with Low irrigation need.
# - These categorical patterns form the basis of the "magic formula" discovered by Chris Deotte.

# %% [markdown]
# ### 3.6 Original vs Synthetic Data

# %%
compare_cols = [
    "Soil_Moisture", "Temperature_C", "Rainfall_mm",
    "Wind_Speed_kmh", "Humidity", "Previous_Irrigation_mm",
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(compare_cols):
    ax = axes[i]
    sns.kdeplot(
        train[col], ax=ax, label=f"Synthetic ({len(train):,})",
        color="#3498db", linewidth=2,
    )
    sns.kdeplot(
        orig[col], ax=ax, label=f"Original ({len(orig):,})",
        color="#e74c3c", linewidth=2, linestyle="--",
    )
    ax.set_title(col, fontweight="bold", fontsize=11)
    ax.set_ylabel("")
    if i == 0:
        ax.legend(fontsize=10)
    else:
        ax.legend().set_visible(False)

fig.suptitle(
    "Distribution Comparison: Synthetic vs Original Data",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# The synthetic generator preserved the overall distributions well.
# Subtle differences exist in the tails, especially for rare combinations.

# %% [markdown]
# ### 3.7 Magic Formula Discovery
#
# Chris Deotte discovered that the original 10K dataset was generated by a deterministic formula:
#
# ```
# high_score = (Soil_Moisture < 25)*2 + (Rainfall_mm < 300)*2 + (Temperature_C > 30) + (Wind_Speed_kmh > 10)
# low_score  = (Crop_Growth_Stage == "Harvest")*2 + (Crop_Growth_Stage == "Sowing")*2 + (Mulching_Used == "Yes")
# magic_score = high_score - low_score
#
# Prediction: score <= 0 -> Low, score >= 4 -> High, else -> Medium
# ```
#
# This achieves BA = 1.0 on the original data. The synthetic generator adds noise and label flips.

# %%
# Compute magic score for the synthetic data
magic_score = (
    (train["Soil_Moisture"] < 25).astype(int) * 2
    + (train["Rainfall_mm"] < 300).astype(int) * 2
    + (train["Temperature_C"] > 30).astype(int)
    + (train["Wind_Speed_kmh"] > 10).astype(int)
    - (train["Crop_Growth_Stage"] == "Harvest").astype(int) * 2
    - (train["Crop_Growth_Stage"] == "Sowing").astype(int) * 2
    - (train["Mulching_Used"] == "Yes").astype(int)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: magic score distribution by class
for cls in CLASS_ORDER:
    mask = train["Irrigation_Need"] == cls
    sns.kdeplot(
        magic_score[mask], ax=axes[0], label=cls, color=CLASS_COLORS[cls],
        fill=True, alpha=0.2, linewidth=1.5,
    )
axes[0].set_title("Magic Score Distribution (Synthetic Data)", fontweight="bold")
axes[0].set_xlabel("Magic Score")
axes[0].legend(title="Class")

# Right: class composition per magic score value
score_class = pd.crosstab(magic_score, train["Irrigation_Need"], normalize="index")
score_class = score_class[CLASS_ORDER]
score_class.plot(
    kind="bar", stacked=True, color=CLASS_PALETTE, ax=axes[1],
    edgecolor="white", linewidth=0.5,
)
axes[1].set_title("Class Proportions per Magic Score", fontweight="bold")
axes[1].set_xlabel("Magic Score")
axes[1].set_ylabel("Proportion")
axes[1].legend(title="Class", fontsize=9)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# %%
# Magic formula accuracy on synthetic data
magic_pred = np.where(magic_score <= 0, "Low", np.where(magic_score >= 4, "High", "Medium"))
magic_ba = balanced_accuracy_score(train["Irrigation_Need"], magic_pred)
print(f"Magic formula balanced accuracy on synthetic data: {magic_ba:.4f}")
print("The formula is not perfect on synthetic data due to noise, but provides strong signal.")

# %% [markdown]
# ## 4. Feature Engineering (Reference Code)
#
# The following code ran during training on AWS SageMaker. It uses
# [feature-engine](https://feature-engine.readthedocs.io/) for reproducible,
# sklearn-compatible transformations.
#
# **Full training script:** `scripts/showcase_v2_train.py`
#
# ### Feature-engine transformers used:
#
# | Transformer | Purpose | Target-aware |
# |-------------|---------|:---:|
# | `CountFrequencyEncoder` | Encode categories by frequency | No |
# | `EqualFrequencyDiscretiser` | Quantile binning (10 bins) | No |
# | `EqualWidthDiscretiser` | Fixed-width binning (15 bins) | No |
# | `RelativeFeatures` | Pairwise differences and products | No |
# | `MathFeatures` | Aggregate stats (mean, std) of feature groups | No |
# | `DecisionTreeDiscretiser` | Supervised binning via decision tree | Yes (per-fold) |
# | `MeanEncoder` | Target encoding for interaction columns | Yes (per-fold) |
#
# ### Other features:
# - **Magic formula**: 15 features from threshold flags, composite scores, boundary distances
# - **Domain interactions**: water balance, heat stress, ET proxy, drying index
# - **Original TE priors**: target mean per category from the 10K original dataset
# - **2-way interactions**: all C(19,2) pairs, factorized as integers
# - **Label encoding**: ordinal encoding for tree models

# %%
# Feature engineering code (reference, not executed on Kaggle)
TRAIN_MODE = False

if TRAIN_MODE:
    from feature_engine.encoding import CountFrequencyEncoder, MeanEncoder
    from feature_engine.creation import RelativeFeatures, MathFeatures
    from feature_engine.discretisation import (
        EqualFrequencyDiscretiser, EqualWidthDiscretiser, DecisionTreeDiscretiser,
    )

    # --- Static features (no target dependency) ---

    # 1. Frequency encoding for categoricals
    cfe = CountFrequencyEncoder(encoding_method="frequency", variables=CAT_COLS)
    cfe.fit(train_df)
    train_df = cfe.transform(train_df)
    test_df = cfe.transform(test_df)

    # 2. Quantile binning (10 equal-frequency bins)
    efd = EqualFrequencyDiscretiser(q=10, variables=BIN_COLS, return_boundaries=False)
    efd.fit(train_df)
    # Transform copies and add as new columns to keep originals
    efd_train = efd.transform(train_df[BIN_COLS].copy())
    for col in BIN_COLS:
        train_df[f"efd10_{col}"] = efd_train[col]

    # 3. Fixed-width binning (15 bins)
    ewd = EqualWidthDiscretiser(bins=15, variables=BIN_COLS, return_boundaries=False)
    ewd.fit(train_df)

    # 4. Relative features: pairwise interactions
    rel_tf = RelativeFeatures(
        variables=["Soil_Moisture", "Temperature_C", "Humidity"],
        reference=["Wind_Speed_kmh", "Rainfall_mm"],
        func=["sub", "mul"],  # Creates 3x2x2 = 12 features
    )
    rel_tf.fit(train_df)
    train_df = rel_tf.transform(train_df)

    # 5. Aggregate math features
    math_tf = MathFeatures(
        variables=["Soil_Moisture", "Temperature_C", "Humidity", "Wind_Speed_kmh"],
        func=["mean", "std"],  # 2 aggregate features
    )
    math_tf.fit(train_df)
    train_df = math_tf.transform(train_df)

    # --- Per-fold features (target-dependent, inside CV loop) ---

    # 6. Supervised binning: decision tree finds optimal split points
    dtd = DecisionTreeDiscretiser(
        variables=BIN_COLS,
        regression=False,
        scoring="balanced_accuracy",
        cv=3,
        param_grid={"max_depth": [2, 3]},
    )
    dtd.fit(X_train_fold, y_train_fold)  # Fit on training fold only

    # 7. Mean encoding: target mean per interaction category
    # Creates bigram cross-features, then encodes by mean target value
    for c1, c2 in INTERACTION_PAIRS:
        X_train_fold[f"bigram_{c1}_{c2}"] = (
            X_train_fold[c1].astype(str) + "_" + X_train_fold[c2].astype(str)
        )
    # Global mean used as fallback for unseen categories in validation/test

    print("Feature engineering complete")

# %% [markdown]
# ## 5. Model Training (Reference Code)
#
# Four models trained with fixed hyperparameters on 5-fold StratifiedKFold CV.
# No Optuna or hyperparameter search. Each fold injects the original 10K rows
# into the training set and applies in-fold target encoding.
#
# ### Model configurations:
#
# | Model | Key Parameters | Notes |
# |-------|---------------|-------|
# | XGBoost | depth=6, lr=0.01, 15K rounds, balanced weights | GPU hist |
# | LightGBM | leaves=77, lr=0.01, is_unbalance=True | CPU, GOSS |
# | CatBoost | depth=6, lr=0.04, auto balanced weights | GPU |
# | RealMLP | n_ens=5, pytabkit defaults | Neural net, GPU |
#
# Training was run on AWS SageMaker with an A10G GPU instance:
# ```bash
# uv run python scripts/showcase_v2_train.py
# ```

# %%
# Training code (reference, not executed on Kaggle)
if TRAIN_MODE:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
    from sklearn.utils.class_weight import compute_sample_weight

    # XGBoost: GPU histogram with balanced sample weights
    xgb_model = XGBClassifier(
        objective="multi:softprob", num_class=3, max_depth=6,
        learning_rate=0.01, n_estimators=15000, subsample=0.8,
        colsample_bytree=0.8, max_bin=1024, tree_method="hist",
        device="cuda", eval_metric="mlogloss",
    )
    weights = compute_sample_weight("balanced", y_train_fold)
    xgb_model.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, y_val)], verbose=False)

    # LightGBM: GBDT with automatic unbalanced handling
    lgb_model = LGBMClassifier(
        objective="multiclass", num_leaves=77, learning_rate=0.01,
        n_estimators=15000, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=7.0, is_unbalance=True, verbosity=-1,
    )

    # CatBoost: native GPU training with auto balanced weights
    cat_model = CatBoostClassifier(
        loss_function="MultiClass", auto_class_weights="Balanced",
        depth=6, learning_rate=0.04, iterations=15000,
        l2_leaf_reg=0.5, task_type="GPU", verbose=0,
    )

    # RealMLP: neural network with internal ensembling
    realmlp_model = RealMLP_TD_Classifier(
        n_cv=1, n_refit=0, n_ens=5, device="cuda", verbosity=0,
    )

    print("Models trained successfully")

# %% [markdown]
# ## 6. Ensemble: Hill Climbing + Threshold Optimization
#
# Load pre-computed OOF and test predictions from the companion dataset.
# Use hill climbing to find optimal blend weights, then differential
# evolution for class-specific threshold multipliers.

# %%
# Load predictions
MODEL_NAMES = ["xgb_sc2", "lgb_sc2", "cat_sc2", "realmlp_sc2"]
DISPLAY_NAMES = ["XGBoost", "LightGBM", "CatBoost", "RealMLP"]

oof_dict = {}
pred_dict = {}
available_models = []

for name, display in zip(MODEL_NAMES, DISPLAY_NAMES):
    oof_path = PRED_DIR / f"oof_{name}.npy"
    pred_path = PRED_DIR / f"pred_{name}.npy"

    if oof_path.exists() and pred_path.exists():
        oof = np.load(oof_path)
        if oof.shape[0] == len(y):
            oof_dict[display] = oof
            pred_dict[display] = np.load(pred_path)
            available_models.append(display)
            cv = balanced_accuracy_score(y, oof.argmax(axis=1))
            print(f"  {display:12s}  CV = {cv:.5f}  shape = {oof.shape}")
        else:
            print(f"  {display:12s}  SKIPPED (shape mismatch: {oof.shape[0]} vs {len(y)})")
    else:
        print(f"  {display:12s}  NOT FOUND")

print(f"\nLoaded {len(available_models)} models: {available_models}")

# %% [markdown]
# ### 6.1 Individual Model Scores

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of CV scores
scores = {name: balanced_accuracy_score(y, oof_dict[name].argmax(axis=1))
          for name in available_models}
colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][:len(available_models)]

ax = axes[0]
bars = ax.barh(
    list(scores.keys()), list(scores.values()),
    color=colors, edgecolor="white", linewidth=1.5,
)
for bar, score in zip(bars, scores.values()):
    ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
            f"{score:.5f}", va="center", fontweight="bold")
ax.set_xlabel("Balanced Accuracy (CV)")
ax.set_title("Individual Model Performance", fontweight="bold")
ax.set_xlim(min(scores.values()) - 0.005, max(scores.values()) + 0.003)

# Correlation heatmap of OOF predictions
corr_data = {}
for name in available_models:
    for ci, cls in enumerate(CLASS_ORDER):
        corr_data[f"{name[:3]}_{cls}"] = oof_dict[name][:, ci]
corr_df = pd.DataFrame(corr_data)
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

ax = axes[1]
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    vmin=0.7, vmax=1.0, ax=ax, linewidths=0.5, annot_kws={"size": 8},
)
ax.set_title("OOF Prediction Correlation", fontweight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2 Hill Climbing Ensemble

# %%
def hill_climbing_ensemble(oof_dict, y_true, n_iterations=5000, seed=42):
    """Find optimal blend weights via stochastic hill climbing.

    Args:
        oof_dict: Dict mapping model name to OOF predictions (N, 3).
        y_true: Ground truth labels.
        n_iterations: Number of search iterations.
        seed: Random state.

    Returns:
        Tuple of (weights dict, best score).
    """
    rng = np.random.RandomState(seed)
    names = list(oof_dict.keys())
    oofs = [oof_dict[n] for n in names]
    n_models = len(oofs)

    weights = np.ones(n_models) / n_models
    blend = sum(w * o for w, o in zip(weights, oofs))
    best_score = balanced_accuracy_score(y_true, blend.argmax(axis=1))
    best_weights = weights.copy()

    for i in range(n_iterations):
        lr = max(0.003, 0.1 * (1 - i / n_iterations))
        new_weights = best_weights + rng.randn(n_models) * lr
        new_weights = np.clip(new_weights, 0, None)
        new_weights /= new_weights.sum()

        blend = sum(w * o for w, o in zip(new_weights, oofs))
        score = balanced_accuracy_score(y_true, blend.argmax(axis=1))

        if score > best_score:
            best_score = score
            best_weights = new_weights.copy()

    return {n: float(w) for n, w in zip(names, best_weights)}, best_score

weights, ensemble_cv = hill_climbing_ensemble(oof_dict, y)
print(f"Hill Climbing Ensemble CV: {ensemble_cv:.5f}")
print(f"Weights: {weights}")

# %% [markdown]
# ### 6.3 Differential Evolution Threshold Optimization

# %%
from scipy.optimize import differential_evolution

# Blended OOF predictions
oof_blend = sum(weights[n] * oof_dict[n] for n in available_models)
test_blend = sum(weights[n] * pred_dict[n] for n in available_models)

baseline_cv = balanced_accuracy_score(y, oof_blend.argmax(axis=1))
print(f"Ensemble before thresholds: {baseline_cv:.5f}")


def neg_ba(thresholds):
    """Negative balanced accuracy for optimization."""
    return -balanced_accuracy_score(y, (oof_blend * thresholds).argmax(axis=1))


result = differential_evolution(
    neg_ba,
    bounds=[(0.3, 4.0), (0.3, 4.0), (0.3, 4.0)],
    seed=42, maxiter=2000, popsize=30, tol=1e-12,
)
best_thresholds = result.x
optimized_cv = -result.fun

print(f"Optimized thresholds: {np.round(best_thresholds, 4)}")
print(f"Ensemble after thresholds: {optimized_cv:.5f}")
print(f"Gain from thresholds:  +{optimized_cv - baseline_cv:.5f}")

# %% [markdown]
# ### 6.4 Ensemble Diagnostics

# %%
from sklearn.metrics import confusion_matrix

final_oof_preds = (oof_blend * best_thresholds).argmax(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
cm = confusion_matrix(y, final_oof_preds)
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
    xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
    linewidths=0.5, cbar_kws={"shrink": 0.8},
)
axes[0].set_title(f"Confusion Matrix (CV={optimized_cv:.5f})", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Per-class recall
recalls = []
for ci, cls in enumerate(CLASS_ORDER):
    mask = y == ci
    recall = (final_oof_preds[mask] == ci).mean()
    recalls.append(recall)
    print(f"  {cls:8s} recall: {recall:.4f}  (n={mask.sum():,})")

ax = axes[1]
bars = ax.bar(CLASS_ORDER, recalls, color=CLASS_PALETTE, edgecolor="white", linewidth=1.5)
for bar, r in zip(bars, recalls):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 0.005, f"{r:.4f}",
            ha="center", fontweight="bold")
ax.set_ylabel("Recall")
ax.set_title("Per-Class Recall", fontweight="bold")
ax.set_ylim(0.9, 1.0)
ax.axhline(y=optimized_cv, color="gray", linestyle="--", alpha=0.5, label="Mean")
ax.legend()

# Weight distribution
ax = axes[2]
model_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"][:len(available_models)]
w_names = list(weights.keys())
w_values = [weights[n] for n in w_names]
bars = ax.barh(w_names, w_values, color=model_colors, edgecolor="white", linewidth=1.5)
for bar, w in zip(bars, w_values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontweight="bold")
ax.set_title("Ensemble Weights", fontweight="bold")
ax.set_xlabel("Weight")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Submission

# %%
# Apply thresholds to test blend
final_test_preds = (test_blend * best_thresholds).argmax(axis=1)
labels = le.inverse_transform(final_test_preds)

submission = pd.DataFrame({"id": test["id"], "Irrigation_Need": labels})
submission.to_csv("submission.csv", index=False)

print(f"Submission saved: submission.csv ({len(submission)} rows)")
print(f"Class distribution:")
print(submission["Irrigation_Need"].value_counts().to_string())
print(f"\nFinal CV: {optimized_cv:.5f}")

# %% [markdown]
# ## Summary
#
# | Component | Details |
# |-----------|---------|
# | **Features** | Magic formula + domain + feature-engine (7 transformers) + original TE priors + 2-way interactions |
# | **Models** | XGBoost, LightGBM, CatBoost, RealMLP (5-fold CV, original data injection) |
# | **Ensemble** | Hill climbing weight optimization |
# | **Post-processing** | Differential evolution threshold optimization |
# | **Infrastructure** | AWS SageMaker A10G GPU, feature-engine, pytabkit |
