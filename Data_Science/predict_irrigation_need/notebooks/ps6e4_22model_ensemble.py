# %% [markdown]
# # PS6E4 22-Model Ensemble: GBDT + RealMLP + TabM + GNN
#
# **Competition:** [Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4)
#
# **Metric:** Balanced Accuracy (3-class: Low, Medium, High)
#
# **Approach:**
# - Load pre-computed OOF and test predictions from 22 models
# - 7 CatBoost, 5 LightGBM, 6 XGBoost, 2 RealMLP, 1 TabM, 1 GNN
# - Greedy forward selection, LightGBM stacking, rank averaging
# - Log-space bias tuning and differential evolution threshold optimization
#
# All models were trained on AWS SageMaker. This notebook only does ensembling.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

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

CLASS_COLORS = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}
CLASS_ORDER = ["Low", "Medium", "High"]

print("Setup complete.")

# %% [markdown]
# ## 2. Load Data and Predictions

# %%
COMP_DIR = Path("/kaggle/input/competitions/playground-series-s6e4")
PRED_DIR = Path("/kaggle/input/ps6e4-22-model-ensemble-predictions")

train = pd.read_csv(COMP_DIR / "train.csv")
test = pd.read_csv(COMP_DIR / "test.csv")

le = LabelEncoder()
le.fit(["High", "Low", "Medium"])  # High=0, Low=1, Medium=2
y_train = le.transform(train["Irrigation_Need"])

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# %%
MODEL_NAMES = [
    "cat_s44", "cat_digit", "cat-2026-04-03-13-29-13-994",
    "cat-2026-04-03-12-44-01-719", "cat_s43",
    "cat-2026-04-03-12-44-05-363", "cat_v2",
    "realmlp_mahog",
    "lgb_v5", "lgb_digit", "xgb_v5",
    "xgb-2026-04-03-11-12-41-698", "xgb-2026-04-03-11-12-38-123",
    "xgb_v2", "xgb_s44", "xgb_s43",
    "lgb_s44", "lgb_v2", "lgb_s43",
    "realmlp_v3fix", "tabm_v3fix", "gnn",
]

oof_dict = {}
pred_dict = {}

print(f"{'Model':<42} {'CV (bal_acc)':>12}")
print("-" * 56)

for name in MODEL_NAMES:
    oof = np.load(PRED_DIR / f"oof_{name}.npy")
    pred = np.load(PRED_DIR / f"pred_{name}.npy")
    oof_dict[name] = oof
    pred_dict[name] = pred
    score = balanced_accuracy_score(y_train, oof.argmax(axis=1))
    print(f"{name:<42} {score:.6f}")

print(f"\nLoaded {len(MODEL_NAMES)} models.")

# %% [markdown]
# ## 3. Model Diversity Analysis
#
# Correlation heatmap of OOF argmax predictions between all 22 models.
# Lower correlation between models means more room for ensemble gains.

# %%
oof_preds_argmax = np.column_stack([oof_dict[m].argmax(axis=1) for m in MODEL_NAMES])
corr = np.corrcoef(oof_preds_argmax.T)

short_names = []
for n in MODEL_NAMES:
    if len(n) > 15:
        short_names.append(n[:7] + ".." + n[-6:])
    else:
        short_names.append(n)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    corr,
    xticklabels=short_names,
    yticklabels=short_names,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0.8,
    vmax=1.0,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("OOF Prediction Correlation (argmax) -- 22 Models")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Greedy Forward Selection
#
# Iteratively add the model that improves ensemble balanced accuracy the most.
# Start with the best single model, keep adding until no improvement.

# %%
def ensemble_proba(model_names, oof_dict):
    """Simple probability averaging for a list of models."""
    return np.mean([oof_dict[m] for m in model_names], axis=0)


# Find best single model
best_single_score = 0
best_single_name = None
for name in MODEL_NAMES:
    score = balanced_accuracy_score(y_train, oof_dict[name].argmax(axis=1))
    if score > best_single_score:
        best_single_score = score
        best_single_name = name

print(f"Best single model: {best_single_name} = {best_single_score:.6f}")

# Greedy forward selection
selected = [best_single_name]
remaining = [m for m in MODEL_NAMES if m != best_single_name]
current_score = best_single_score

print(f"\n{'Step':<6} {'Added Model':<42} {'Ensemble CV':>12} {'Delta':>10}")
print("-" * 72)
print(f"{1:<6} {best_single_name:<42} {best_single_score:.6f} {'--':>10}")

step = 2
while remaining:
    best_add_score = current_score
    best_add_name = None

    for name in remaining:
        trial = selected + [name]
        proba = ensemble_proba(trial, oof_dict)
        score = balanced_accuracy_score(y_train, proba.argmax(axis=1))
        if score > best_add_score:
            best_add_score = score
            best_add_name = name

    if best_add_name is None:
        print(f"\nNo further improvement. Stopping with {len(selected)} models.")
        break

    delta = best_add_score - current_score
    selected.append(best_add_name)
    remaining.remove(best_add_name)
    current_score = best_add_score
    print(f"{step:<6} {best_add_name:<42} {current_score:.6f} {delta:>+10.6f}")
    step += 1

greedy_score = current_score
greedy_models = selected.copy()
print(f"\nGreedy ensemble: {len(greedy_models)} models, CV = {greedy_score:.6f}")

# %%
# Build greedy ensemble predictions
greedy_oof = ensemble_proba(greedy_models, oof_dict)
greedy_test = np.mean([pred_dict[m] for m in greedy_models], axis=0)

# %% [markdown]
# ## 5. LightGBM Stacker
#
# 5-fold stacker using all 22 models' OOF probabilities as features (66 features).

# %%
# Build stacking features: 22 models x 3 classes = 66 features
oof_stack_features = np.hstack([oof_dict[m] for m in MODEL_NAMES])
test_stack_features = np.hstack([pred_dict[m] for m in MODEL_NAMES])

feature_names = []
for m in MODEL_NAMES:
    for c in range(3):
        feature_names.append(f"{m}_c{c}")

print(f"Stacking features shape: {oof_stack_features.shape}")

# %%
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

stacker_oof = np.zeros((len(y_train), 3))
stacker_test_preds = np.zeros((len(test), 3))

lgb_params = dict(
    num_leaves=31,
    n_estimators=300,
    learning_rate=0.05,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(oof_stack_features, y_train)):
    X_tr = oof_stack_features[train_idx]
    y_tr = y_train[train_idx]
    X_val = oof_stack_features[val_idx]
    y_val = y_train[val_idx]

    model = LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)

    stacker_oof[val_idx] = model.predict_proba(X_val)
    stacker_test_preds += model.predict_proba(test_stack_features) / N_FOLDS

    fold_score = balanced_accuracy_score(y_val, stacker_oof[val_idx].argmax(axis=1))
    fold_scores.append(fold_score)
    print(f"  Fold {fold_idx + 1}: {fold_score:.6f}")

stacker_score = balanced_accuracy_score(y_train, stacker_oof.argmax(axis=1))
print(f"\nLGBM Stacker CV: {stacker_score:.6f} (mean fold: {np.mean(fold_scores):.6f})")

# %% [markdown]
# ## 6. Rank Averaging
#
# Convert each model's probabilities to ranks, then average across models.

# %%
def rank_average(prob_dict, model_names):
    """Rank-average probabilities across models."""
    n = prob_dict[model_names[0]].shape[0]
    ranked = np.zeros((n, 3))
    for name in model_names:
        proba = prob_dict[name]
        for c in range(3):
            order = proba[:, c].argsort().argsort()
            ranked[:, c] += order
    ranked /= len(model_names)
    return ranked


rank_oof = rank_average(oof_dict, MODEL_NAMES)
rank_test = rank_average(pred_dict, MODEL_NAMES)
rank_score = balanced_accuracy_score(y_train, rank_oof.argmax(axis=1))
print(f"Rank averaging CV (all 22 models): {rank_score:.6f}")

# Rank average with greedy-selected models only
rank_greedy_oof = rank_average(oof_dict, greedy_models)
rank_greedy_test = rank_average(pred_dict, greedy_models)
rank_greedy_score = balanced_accuracy_score(y_train, rank_greedy_oof.argmax(axis=1))
print(f"Rank averaging CV (greedy subset): {rank_greedy_score:.6f}")

# %% [markdown]
# ## 7. Log-Space Bias Tuning
#
# For each ensemble method, tune per-class multiplicative biases using a
# multi-resolution grid search in log space. This adjusts decision boundaries
# without retraining.

# %%
def tune_bias_log(proba, y_true):
    """Tune per-class multiplicative biases via multi-resolution log-space search."""
    best_score = balanced_accuracy_score(y_true, proba.argmax(axis=1))
    best_bias = np.ones(3)
    for resolution in [0.1, 0.01, 0.001]:
        improved = True
        while improved:
            improved = False
            for c in range(3):
                for delta in [-resolution, resolution]:
                    trial = best_bias.copy()
                    trial[c] = np.exp(np.log(trial[c]) + delta)
                    score = balanced_accuracy_score(y_true, (proba * trial).argmax(axis=1))
                    if score > best_score:
                        best_score = score
                        best_bias = trial
                        improved = True
    return best_bias, best_score


# Apply to each ensemble method
methods = {
    "Greedy Avg": (greedy_oof, greedy_test),
    "Stacker": (stacker_oof, stacker_test_preds),
    "Rank All": (rank_oof, rank_test),
    "Rank Greedy": (rank_greedy_oof, rank_greedy_test),
}

bias_results = {}
print(f"{'Method':<20} {'Before':>10} {'After':>10} {'Bias':>30}")
print("-" * 72)

for method_name, (oof_proba, test_proba) in methods.items():
    before = balanced_accuracy_score(y_train, oof_proba.argmax(axis=1))
    bias, after = tune_bias_log(oof_proba, y_train)
    bias_results[method_name] = {
        "bias": bias,
        "score": after,
        "oof": oof_proba,
        "test": test_proba,
    }
    bias_str = f"[{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}]"
    print(f"{method_name:<20} {before:>10.6f} {after:>10.6f} {bias_str:>30}")

# %% [markdown]
# ## 8. Differential Evolution Thresholds
#
# For each ensemble method, optimize per-class thresholds via scipy
# differential_evolution. Instead of simple argmax, we subtract per-class
# thresholds before taking the argmax.

# %%
def optimize_thresholds_de(proba, y_true, seed=42):
    """Optimize per-class thresholds using differential evolution."""
    def objective(thresholds):
        adjusted = proba - thresholds.reshape(1, 3)
        preds = adjusted.argmax(axis=1)
        return -balanced_accuracy_score(y_true, preds)

    bounds = [(-0.5, 0.5)] * 3
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=200,
        tol=1e-8,
        polish=True,
    )
    best_thresholds = result.x
    best_score = -result.fun
    return best_thresholds, best_score


de_results = {}
print(f"{'Method':<20} {'Before':>10} {'After DE':>10} {'Thresholds':>36}")
print("-" * 78)

for method_name, (oof_proba, test_proba) in methods.items():
    before = balanced_accuracy_score(y_train, oof_proba.argmax(axis=1))
    thresholds, after = optimize_thresholds_de(oof_proba, y_train)
    de_results[method_name] = {
        "thresholds": thresholds,
        "score": after,
        "oof": oof_proba,
        "test": test_proba,
    }
    thresh_str = f"[{thresholds[0]:+.4f}, {thresholds[1]:+.4f}, {thresholds[2]:+.4f}]"
    print(f"{method_name:<20} {before:>10.6f} {after:>10.6f} {thresh_str:>36}")

# %% [markdown]
# ## 9. Results Summary

# %%
# Collect all results
all_results = {}

# Raw ensemble scores
for method_name, (oof_proba, _) in methods.items():
    score = balanced_accuracy_score(y_train, oof_proba.argmax(axis=1))
    all_results[f"{method_name} (raw)"] = score

# Bias-tuned scores
for method_name, info in bias_results.items():
    all_results[f"{method_name} + Bias"] = info["score"]

# DE threshold scores
for method_name, info in de_results.items():
    all_results[f"{method_name} + DE"] = info["score"]

# Sort and display
results_df = pd.DataFrame(
    [(name, score) for name, score in all_results.items()],
    columns=["Method", "CV (bal_acc)"],
).sort_values("CV (bal_acc)", ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))

best_method = results_df.iloc[0]["Method"]
best_cv = results_df.iloc[0]["CV (bal_acc)"]
print(f"\nBest method: {best_method} with CV = {best_cv:.6f}")

# %% [markdown]
# ## 10. Submission

# %%
# Determine which ensemble and post-processing to use
# Parse best method name to get the base method and post-processing type
if "+" in best_method:
    base_name = best_method.split(" + ")[0].strip()
    pp_type = best_method.split(" + ")[1].strip()
else:
    base_name = best_method.replace(" (raw)", "").strip()
    pp_type = "raw"

base_test_proba = methods[base_name][1]

if pp_type == "Bias":
    bias = bias_results[base_name]["bias"]
    final_proba = base_test_proba * bias
    final_preds = final_proba.argmax(axis=1)
elif pp_type == "DE":
    thresholds = de_results[base_name]["thresholds"]
    final_preds = (base_test_proba - thresholds.reshape(1, 3)).argmax(axis=1)
else:
    final_preds = base_test_proba.argmax(axis=1)

submission = pd.DataFrame({
    "id": test["id"],
    "Irrigation_Need": le.inverse_transform(final_preds),
})

print(f"Submission shape: {submission.shape}")
print(f"\nClass distribution:")
print(submission["Irrigation_Need"].value_counts().sort_index())
print(f"\nFirst rows:")
print(submission.head(10))

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")

# %% [markdown]
# ## Summary
#
# **22 models ensembled:**
# - 7 CatBoost (cat_s44, cat_digit, cat-2026-04-03-13-29-13-994,
#   cat-2026-04-03-12-44-01-719, cat_s43, cat-2026-04-03-12-44-05-363, cat_v2)
# - 5 LightGBM (lgb_v5, lgb_digit, lgb_s44, lgb_v2, lgb_s43)
# - 6 XGBoost (xgb_v5, xgb-2026-04-03-11-12-41-698,
#   xgb-2026-04-03-11-12-38-123, xgb_v2, xgb_s44, xgb_s43)
# - 2 RealMLP (realmlp_mahog, realmlp_v3fix)
# - 1 TabM (tabm_v3fix)
# - 1 GNN (gnn)
#
# All models were trained on AWS SageMaker with diverse hyperparameters and
# feature sets. This notebook loads pre-computed OOF and test predictions,
# then explores greedy selection, LightGBM stacking, rank averaging, bias
# tuning, and differential evolution threshold optimization.
#
# The best CV score and winning ensemble method are printed above.
