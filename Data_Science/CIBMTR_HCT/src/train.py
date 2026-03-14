"""
CIBMTR — Model Training & Evaluation

Trains XGBoost, LightGBM, and CatBoost with 5-fold stratified CV.
Produces CV results, feature importance, and per-group C-index plots.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import (
    MODEL_DIR, DPI, RANDOM_STATE, N_FOLDS, PALETTE, MODEL_COLORS,
    EFS_COL, TIME_COL, GROUP_COL, ID_COL, RACE_PALETTE,
)
from features import load_and_prepare, get_Xy
from metrics import stratified_concordance_index

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


# ── Model definitions ─────────────────────────────────────────────────────────

def get_xgb_model():
    return xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        enable_categorical=True,
    )


def get_lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=600,
        num_leaves=63,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def get_catboost_model():
    return cb.CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=1.0,
        subsample=0.8,
        colsample_bylevel=0.7,
        random_seed=RANDOM_STATE,
        verbose=0,
    )


# ── Cross-Validation ──────────────────────────────────────────────────────────

def run_cv(model_fn, X, y_time, y_efs, df_train, feature_cols, cat_cols=None):
    """
    5-fold stratified CV; returns OOF predictions and per-fold scores.
    model_fn() returns a fresh model instance.
    """
    # Stratify on efs (event indicator)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_efs), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y_time.iloc[tr_idx]

        model = model_fn()

        if isinstance(model, cb.CatBoostRegressor) and cat_cols:
            model.fit(X_tr, y_tr, cat_features=cat_cols, verbose=0)
        else:
            model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        df_val = df_train.iloc[val_idx].copy()
        df_val["prediction"] = preds
        scores = stratified_concordance_index(
            df=df_val,
            group_col=GROUP_COL,
            time_col=TIME_COL,
            event_col=EFS_COL,
            pred_col="prediction",
        )
        fold_scores.append(scores)
        print(f"    Fold {fold}: stratified_CI = {scores['stratified_ci']:.4f}  "
              f"(macro={scores['macro_ci']:.4f}, std={scores['std_ci']:.4f})")

    # OOF aggregate score
    df_oof = df_train.copy()
    df_oof["prediction"] = oof_preds
    oof_scores = stratified_concordance_index(
        df=df_oof, group_col=GROUP_COL, time_col=TIME_COL,
        event_col=EFS_COL, pred_col="prediction",
    )

    mean_strat = np.mean([s["stratified_ci"] for s in fold_scores])
    std_strat  = np.std([s["stratified_ci"]  for s in fold_scores])

    return oof_preds, fold_scores, oof_scores, mean_strat, std_strat


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cv_scores(all_results: dict) -> None:
    model_names = list(all_results.keys())
    means = [all_results[m]["mean_strat"] for m in model_names]
    stds  = [all_results[m]["std_strat"]  for m in model_names]
    colors = [MODEL_COLORS.get(m, PALETTE[i]) for i, m in enumerate(model_names)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(model_names, means, yerr=stds, capsize=8,
                  color=colors, alpha=0.88,
                  error_kw={"elinewidth": 2, "capthick": 2})
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                f"{val:.4f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Stratified C-Index")
    ax.set_title(f"Model Comparison — {N_FOLDS}-Fold CV\n"
                 "Stratified Concordance Index (higher = better)",
                 fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(means) * 1.15)
    fig.tight_layout()
    save(fig, "01_model_cv_comparison")


def plot_per_group_ci(all_results: dict) -> None:
    """Bar chart of per-race C-index for each model + ensemble."""
    race_short = {
        "White":                                    "White",
        "Black or African-American":                "Black / AA",
        "Asian":                                    "Asian",
        "More than one race":                       "Multiracial",
        "Native Hawaiian or other Pacific Islander":"NHPI",
        "American Indian or Alaska Native":         "AI/AN",
    }

    # Collect per-group OOF scores
    models_data = {}
    for mname, res in all_results.items():
        oof_scores = res["oof_scores"]
        group_cis = {k: v for k, v in oof_scores.items() if k.startswith("ci_")}
        models_data[mname] = group_cis

    # Build DataFrame
    df_plot = pd.DataFrame(models_data).T

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(df_plot.shape[1])
    width = 0.22
    model_names = list(models_data.keys())

    for i, mname in enumerate(model_names):
        vals = df_plot.loc[mname].values
        color = MODEL_COLORS.get(mname, PALETTE[i])
        bars = ax.bar(x + i * width, vals, width, label=mname,
                      color=color, alpha=0.85)

    ax.axhline(0.5, color="#333", ls="--", lw=1.2, alpha=0.6, label="Random (0.5)")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    col_labels = [race_short.get(c.replace("ci_", "").replace("_", " ").title(), c)
                  for c in df_plot.columns]
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_ylabel("C-Index")
    ax.set_title("Per-Race Group C-Index (OOF)\nEquity check across models",
                 fontweight="bold", fontsize=13)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    save(fig, "02_per_group_ci")


def plot_feature_importance(model, feature_cols: list, model_name: str,
                             top_n: int = 20) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()
    else:
        return

    fi = (
        pd.Series(importances, index=feature_cols[:len(importances)])
        .sort_values(ascending=False)
        .head(top_n)
    )

    cmap_colors = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(fi)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi.index[::-1], fi.values[::-1], color=cmap_colors, alpha=0.9)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    save(fig, f"03_feature_importance_{model_name.lower().replace(' ', '_')}")


def plot_oof_survival_curves(df_train: pd.DataFrame, oof_dict: dict) -> None:
    """Scatter: predicted risk vs actual survival time by event status."""
    from lifelines import KaplanMeierFitter

    n = len(oof_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (mname, preds) in zip(axes, oof_dict.items()):
        color = MODEL_COLORS.get(mname, "#2196F3")
        censored = df_train[EFS_COL] == 0
        event    = df_train[EFS_COL] == 1

        ax.scatter(preds[censored], df_train.loc[censored, TIME_COL],
                   s=2, alpha=0.2, color="#2196F3", label="Censored", rasterized=True)
        ax.scatter(preds[event],    df_train.loc[event, TIME_COL],
                   s=2, alpha=0.2, color="#E63946", label="Event", rasterized=True)
        ax.set_xlabel("Predicted log(EFS Time)")
        ax.set_ylabel("Actual EFS Time (months)")
        ax.set_title(f"{mname}\nPredicted vs Actual", fontweight="bold")
        ax.legend(frameon=False, markerscale=4)

    fig.suptitle("Predicted Risk Score vs Actual Survival Time (OOF)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "04_predicted_vs_actual")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading and preparing features...")
    df_train, _, feature_cols = load_and_prepare(include_test=False)
    X, y_time, y_efs = get_Xy(df_train, feature_cols)
    print(f"  {len(X):,} patients | {X.shape[1]} features")

    cat_idx = [i for i, c in enumerate(feature_cols)
               if c in df_train.select_dtypes(include="object").columns]

    model_configs = [
        ("XGBoost",  get_xgb_model,      None),
        ("LightGBM", get_lgb_model,       None),
        ("CatBoost", get_catboost_model,  cat_idx),
    ]

    all_results = {}
    oof_dict = {}

    print(f"\nRunning {N_FOLDS}-fold CV:")
    for mname, model_fn, cat_cols in model_configs:
        print(f"\n  [{mname}]")
        oof_preds, fold_scores, oof_scores, mean_strat, std_strat = run_cv(
            model_fn, X, y_time, y_efs, df_train, feature_cols, cat_cols
        )
        print(f"  OOF Stratified CI = {oof_scores['stratified_ci']:.4f}  "
              f"CV mean = {mean_strat:.4f} ± {std_strat:.4f}")
        all_results[mname] = {
            "oof_preds": oof_preds,
            "fold_scores": fold_scores,
            "oof_scores": oof_scores,
            "mean_strat": mean_strat,
            "std_strat": std_strat,
        }
        oof_dict[mname] = oof_preds

    # Save results summary
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        mname: {
            "CV Stratified CI (mean)": res["mean_strat"],
            "CV Stratified CI (std)":  res["std_strat"],
            "OOF Stratified CI":       res["oof_scores"]["stratified_ci"],
            "OOF Macro CI":            res["oof_scores"]["macro_ci"],
            "OOF Micro CI":            res["oof_scores"]["micro_ci"],
        }
        for mname, res in all_results.items()
    }).T.round(4)
    summary.to_csv(MODEL_DIR / "cv_results.csv")
    print(f"\n  Results saved → {MODEL_DIR / 'cv_results.csv'}")

    print("\nGenerating model plots:")
    plot_cv_scores(all_results)
    plot_per_group_ci(all_results)
    plot_oof_survival_curves(df_train, oof_dict)

    # Train final models on full data for feature importance
    print("\n  Training final models for feature importance...")
    for mname, model_fn, cat_cols in model_configs:
        model = model_fn()
        if isinstance(model, cb.CatBoostRegressor) and cat_cols:
            model.fit(X, y_time, cat_features=cat_cols, verbose=0)
        else:
            model.fit(X, y_time)
        plot_feature_importance(model, feature_cols, mname)

    print(f"\nAll model outputs saved to  output/model/")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
