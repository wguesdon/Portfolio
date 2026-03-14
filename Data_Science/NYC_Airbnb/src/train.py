"""
NYC Airbnb 2019 — Model Training & Evaluation

Trains four models (Ridge, Random Forest, XGBoost, LightGBM) with 5-fold CV,
then produces comparison charts, feature importance, actual-vs-predicted, and
residual analysis for the best model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, learning_curve
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from config import MODEL_DIR, DPI, RANDOM_STATE, N_FOLDS, PALETTE
from features import load_and_clean, build_feature_matrix

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
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


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ── Models ────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=10.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=14,
            min_samples_leaf=4, n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500, max_depth=6,
            learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63,
            learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=20,
            n_jobs=-1, random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


# ── Cross-Validation ──────────────────────────────────────────────────────────

def evaluate_cv(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Return per-fold metrics and out-of-fold predictions."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2_scores, rmse_scores, mae_scores = [], [], []
    oof_preds = np.zeros(len(y))

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        r2_scores.append(r2_score(y_val, preds))
        rmse_scores.append(rmse(y_val, preds))
        mae_scores.append(mean_absolute_error(y_val, preds))

    return {
        "r2_mean":   np.mean(r2_scores),
        "r2_std":    np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std":  np.std(rmse_scores),
        "mae_mean":  np.mean(mae_scores),
        "mae_std":   np.std(mae_scores),
        "oof_preds": oof_preds,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    models = list(results.keys())
    colors = PALETTE[: len(models)]

    r2_means  = [results[m]["r2_mean"]   for m in models]
    r2_stds   = [results[m]["r2_std"]    for m in models]
    rmse_means = [results[m]["rmse_mean"] for m in models]
    rmse_stds  = [results[m]["rmse_std"]  for m in models]
    mae_means  = [results[m]["mae_mean"]  for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ekw = {"elinewidth": 2, "capthick": 2}

    ax = axes[0]
    bars = ax.bar(models, r2_means, yerr=r2_stds, capsize=7,
                  color=colors, alpha=0.85, error_kw=ekw)
    ax.set_title("CV R² Score\n(higher is better)", fontweight="bold")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=12)

    ax = axes[1]
    bars = ax.bar(models, rmse_means, yerr=rmse_stds, capsize=7,
                  color=colors, alpha=0.85, error_kw=ekw)
    ax.set_title("CV RMSE — log(price) scale\n(lower is better)", fontweight="bold")
    ax.set_ylabel("RMSE")
    for bar, val in zip(bars, rmse_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=12)

    ax = axes[2]
    bars = ax.bar(models, mae_means, color=colors, alpha=0.85)
    ax.set_title("CV MAE — log(price) scale\n(lower is better)", fontweight="bold")
    ax.set_ylabel("MAE")
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=12)

    fig.suptitle(f"NYC Airbnb — Model Comparison ({N_FOLDS}-Fold CV)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_model_comparison")


def plot_actual_vs_predicted(y_true: pd.Series, oof_dict: dict) -> None:
    n = len(oof_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, oof_dict.items()):
        r2 = r2_score(y_true, y_pred)
        ax.scatter(y_true, y_pred, s=2, alpha=0.25, color="#2196F3", rasterized=True)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Perfect prediction")
        ax.set_title(f"{name}\nR² = {r2:.3f}", fontweight="bold")
        ax.set_xlabel("Actual  log(Price + 1)")
        ax.set_ylabel("Predicted  log(Price + 1)")
        ax.legend(frameon=False)

    fig.suptitle("NYC Airbnb — Actual vs Predicted (OOF)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_actual_vs_predicted")


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_pred, residuals, s=2, alpha=0.25, color="#FF5722", rasterized=True)
    ax.axhline(0, color="#333", lw=1.8, ls="--")
    ax.set_xlabel("Predicted  log(Price + 1)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted", fontweight="bold")

    ax = axes[1]
    sns.histplot(residuals, bins=60, kde=True, color="#FF5722", ax=ax,
                 line_kws={"lw": 2})
    ax.axvline(0, color="#333", lw=1.8, ls="--")
    ax.set_xlabel("Residual")
    ax.set_title("Residual Distribution", fontweight="bold")

    fig.suptitle(f"NYC Airbnb — {model_name}: Residual Analysis",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_residuals")


def plot_feature_importance(model, feature_cols: list, model_name: str,
                             top_n: int = 20) -> None:
    # Unwrap sklearn Pipeline if needed
    m = model.named_steps.get("reg", model) if hasattr(model, "named_steps") else model

    if hasattr(m, "feature_importances_"):
        importances = m.feature_importances_
    elif hasattr(m, "coef_"):
        importances = np.abs(m.coef_)
    else:
        print(f"  Feature importance not available for {model_name}")
        return

    fi = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .head(top_n)
    )

    cmap_colors = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(fi)))[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi.index[::-1], fi.values[::-1], color=cmap_colors, alpha=0.9)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    save(fig, "04_feature_importance")


def plot_learning_curve(model, X: pd.DataFrame, y: pd.Series,
                         model_name: str) -> None:
    # Use a fast clone with fewer estimators for the learning curve sweep
    import copy
    lc_model = copy.deepcopy(model)
    if hasattr(lc_model, "n_estimators"):
        lc_model.set_params(n_estimators=100)
    elif hasattr(lc_model, "named_steps"):
        pass  # Pipeline — leave as-is

    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=3, scoring="r2", n_jobs=-1,
    )
    tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
    vl_mean, vl_std = val_scores.mean(1),   val_scores.std(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, tr_mean, "o-", color="#2196F3", label="Training")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    color="#2196F3", alpha=0.15)
    ax.plot(train_sizes, vl_mean, "o-", color="#FF5722", label="Validation (CV)")
    ax.fill_between(train_sizes, vl_mean - vl_std, vl_mean + vl_std,
                    color="#FF5722", alpha=0.15)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R² Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Learning Curve — {model_name}", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "05_learning_curve")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading and engineering features...")
    df = load_and_clean()
    X, y, feature_cols = build_feature_matrix(df)
    print(f"  {len(X):,} listings | {X.shape[1]} features")

    models = get_models()
    results: dict[str, dict] = {}

    print(f"\nRunning {N_FOLDS}-fold cross-validation:")
    for name, model in models.items():
        print(f"  {name:<20}", end="", flush=True)
        res = evaluate_cv(model, X, y)
        results[name] = res
        print(
            f"R² = {res['r2_mean']:.4f} ± {res['r2_std']:.4f}  "
            f"RMSE = {res['rmse_mean']:.4f}  "
            f"MAE = {res['mae_mean']:.4f}"
        )

    # Save results table
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        name: {
            "R2 Mean":   res["r2_mean"],
            "R2 Std":    res["r2_std"],
            "RMSE Mean": res["rmse_mean"],
            "RMSE Std":  res["rmse_std"],
            "MAE Mean":  res["mae_mean"],
        }
        for name, res in results.items()
    }).T.round(4)
    results_df.to_csv(MODEL_DIR / "cv_results.csv")
    print(f"\n  Results → {MODEL_DIR / 'cv_results.csv'}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["r2_mean"])
    print(f"\n  Best model: {best_name}  (R² = {results[best_name]['r2_mean']:.4f})")

    print("\nGenerating model plots:")
    plot_model_comparison(results)

    oof_dict = {
        name: results[name]["oof_preds"]
        for name in ["XGBoost", "LightGBM"]
        if name in results
    }
    plot_actual_vs_predicted(y, oof_dict)

    best_model = models[best_name]
    best_model.fit(X, y)
    plot_residuals(y, results[best_name]["oof_preds"], best_name)
    plot_feature_importance(best_model, feature_cols, best_name)

    print(f"\n  Computing learning curve for {best_name} (this may take a moment)...")
    plot_learning_curve(best_model, X, y, best_name)

    print(f"\nAll model plots saved to  {MODEL_DIR.relative_to(MODEL_DIR.parent.parent)}/")
    print("\n" + results_df.to_string())


if __name__ == "__main__":
    main()
