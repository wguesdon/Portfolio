"""
CIBMTR — Rank-Based Ensemble & Submission Generator

Trains all three models on full training data, generates rank-ensemble
predictions for the test set, and writes submission.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

from config import (
    MODEL_DIR, DPI, PALETTE, MODEL_COLORS,
    EFS_COL, TIME_COL, GROUP_COL, ID_COL, SUBMIT_FILE,
    RANDOM_STATE,
)
from features import load_and_prepare, get_Xy

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
})


def save(fig, name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path.name}")


def rank_transform(arr: np.ndarray) -> np.ndarray:
    """Normalise predictions to [0, 1] percentile ranks."""
    return pd.Series(arr).rank(pct=True).values


def rank_ensemble(pred_dict: dict, weights: dict) -> np.ndarray:
    """
    Weighted rank ensemble.
    Models predict log(efs_time): higher = longer survival = LOWER risk.
    Negate before ranking so the ensemble score = risk (higher = more risk).
    """
    ensemble = np.zeros(len(next(iter(pred_dict.values()))))
    total_weight = sum(weights.values())
    for mname, preds in pred_dict.items():
        w = weights.get(mname, 1.0) / total_weight
        ensemble += w * rank_transform(-preds)
    return ensemble


def plot_prediction_correlations(pred_dict: dict) -> None:
    """Correlation heatmap of model predictions."""
    df_preds = pd.DataFrame(pred_dict)
    corr = df_preds.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=0, vmax=1, square=True,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 11})
    ax.set_title("Model Prediction Correlations (Test Set)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "05_prediction_correlations")


def plot_ensemble_distribution(ensemble_preds: np.ndarray, test_ids) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ensemble_preds, bins=40, color=MODEL_COLORS["Ensemble"],
            alpha=0.85, edgecolor="white")
    ax.set_xlabel("Ensemble Risk Score (rank-normalised)")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble Prediction Distribution — Test Set",
                 fontweight="bold", fontsize=13)
    ax.axvline(np.mean(ensemble_preds), color="#333", ls="--", lw=1.5,
               label=f"Mean: {np.mean(ensemble_preds):.3f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "06_ensemble_distribution")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data (train + test)...")
    df_train, df_test, feature_cols = load_and_prepare(include_test=True)
    X_train, y_time, y_efs = get_Xy(df_train, feature_cols)
    X_test = df_test[feature_cols]
    test_ids = df_test[ID_COL].values
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    cat_idx = [i for i, c in enumerate(feature_cols)
               if c in df_train.select_dtypes(include="object").columns]

    # ── Train full models ──────────────────────────────────────────────────────
    models = {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
            enable_categorical=True,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
        ),
        "CatBoost": cb.CatBoostRegressor(
            iterations=600, depth=6, learning_rate=0.04,
            l2_leaf_reg=1.0, subsample=0.8, colsample_bylevel=0.7,
            random_seed=RANDOM_STATE, verbose=0,
        ),
    }

    # Weights based on CV performance (XGBoost ≈ LightGBM > CatBoost for this task)
    weights = {"XGBoost": 0.38, "LightGBM": 0.38, "CatBoost": 0.24}

    test_preds = {}
    print("\nTraining final models on full training set:")
    for mname, model in models.items():
        print(f"  {mname}...")
        if isinstance(model, cb.CatBoostRegressor) and cat_idx:
            model.fit(X_train, y_time, cat_features=cat_idx, verbose=0)
        else:
            model.fit(X_train, y_time)
        test_preds[mname] = model.predict(X_test)
        print(f"    Predictions range: [{test_preds[mname].min():.3f}, {test_preds[mname].max():.3f}]")

    # ── Rank ensemble ─────────────────────────────────────────────────────────
    ensemble_preds = rank_ensemble(test_preds, weights)
    print(f"\nEnsemble predictions range: [{ensemble_preds.min():.3f}, {ensemble_preds.max():.3f}]")

    # ── Save submission ────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COL: test_ids, "prediction": ensemble_preds})
    submission.to_csv(SUBMIT_FILE, index=False)
    print(f"\n  Submission saved → {SUBMIT_FILE}")
    print(submission.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating ensemble plots:")
    plot_prediction_correlations(test_preds)
    plot_ensemble_distribution(ensemble_preds, test_ids)

    print("\nEnsemble complete.")


if __name__ == "__main__":
    main()
