#!/usr/bin/env python
"""Ensemble v4: greedy forward selection + stacking + multiple blend strategies.

Improvements over v3:
  - Greedy forward model selection (only adds models that improve CV)
  - LightGBM stacker on OOF probabilities (5-fold CV)
  - Logistic regression stacker
  - Rank averaging
  - All methods combined with log-space bias and diffevol thresholds
  - Picks the overall best

Usage:
    uv run python scripts/ensemble_v4.py --models "model1,model2,..."
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
from scipy.stats import rankdata


def load_predictions(pred_dir, model_names):
    """Load OOF and test predictions for all models.

    Args:
        pred_dir: Path to predictions directory.
        model_names: List of model name strings.

    Returns:
        Tuple of (oof_dict, pred_dict).
    """
    oof_dict = {}
    pred_dict = {}
    for name in model_names:
        oof_dict[name] = np.load(pred_dir / f"oof_{name}.npy")
        pred_dict[name] = np.load(pred_dir / f"pred_{name}.npy")
    return oof_dict, pred_dict


def tune_bias_log(proba, y_true):
    """Log-space bias tuning via multi-step grid climb.

    Args:
        proba: Class probability array (N, 3).
        y_true: Ground truth labels.

    Returns:
        Tuple of (best_bias array, best score).
    """
    def preds_from_bias(bias):
        return np.argmax(np.log(np.clip(proba, 1e-15, 1.0)) + bias, axis=1)

    best_bias = np.zeros(proba.shape[1], dtype=np.float64)
    best_score = balanced_accuracy_score(y_true, preds_from_bias(best_bias))

    for step in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
        improved = True
        while improved:
            improved = False
            for ci in range(proba.shape[1]):
                for d in (-1.0, 1.0):
                    candidate = best_bias.copy()
                    candidate[ci] += d * step
                    s = balanced_accuracy_score(y_true, preds_from_bias(candidate))
                    if s > best_score + 1e-8:
                        best_bias = candidate
                        best_score = s
                        improved = True

    return best_bias, best_score


def diffevol_thresholds(proba, y_true):
    """Differential evolution for multiplicative class thresholds.

    Args:
        proba: Class probability array (N, 3).
        y_true: Ground truth labels.

    Returns:
        Tuple of (best_thresholds array, best score).
    """
    def neg_ba(thresholds):
        return -balanced_accuracy_score(y_true, (proba * thresholds).argmax(axis=1))

    result = differential_evolution(
        neg_ba,
        bounds=[(0.3, 4.0), (0.3, 4.0), (0.3, 4.0)],
        seed=42,
        maxiter=2000,
        popsize=30,
        tol=1e-12,
    )
    return result.x, -result.fun


def apply_best_threshold(oof_proba, test_proba, y_true):
    """Apply the best threshold method (log bias or diffevol).

    Args:
        oof_proba: OOF probability array.
        test_proba: Test probability array.
        y_true: Ground truth labels.

    Returns:
        Tuple of (test_preds, cv_score, method_name).
    """
    bias, score_bias = tune_bias_log(oof_proba, y_true)
    thresh, score_de = diffevol_thresholds(oof_proba, y_true)

    if score_bias >= score_de:
        test_preds = np.argmax(
            np.log(np.clip(test_proba, 1e-15, 1.0)) + bias, axis=1
        )
        return test_preds, score_bias, "log_bias"
    else:
        test_preds = (test_proba * thresh).argmax(axis=1)
        return test_preds, score_de, "diffevol"


def greedy_forward_selection(oof_dict, y_true):
    """Greedy forward model selection.

    Starts empty, adds the model that improves ensemble CV the most
    at each step. Stops when no model improves the score.

    Args:
        oof_dict: Dict mapping model name to OOF predictions (N, 3).
        y_true: Ground truth labels.

    Returns:
        Tuple of (selected model names, weights dict, best score).
    """
    names = list(oof_dict.keys())
    selected = []
    remaining = set(names)
    best_score = 0.0

    while remaining:
        best_add = None
        best_add_score = best_score

        for candidate in sorted(remaining):
            trial = selected + [candidate]
            # Equal weight blend
            blend = np.mean([oof_dict[n] for n in trial], axis=0)
            # Apply log bias tuning (fast, no diffevol)
            _, score = tune_bias_log(blend, y_true)

            if score > best_add_score + 1e-6:
                best_add_score = score
                best_add = candidate

        if best_add is None:
            break

        selected.append(best_add)
        remaining.remove(best_add)
        best_score = best_add_score
        print(f"  +{best_add:30s}  CV: {best_score:.5f}  (n={len(selected)})")

    # Compute equal weights for selected
    n = len(selected)
    weights = {name: 1.0 / n for name in selected}
    return selected, weights, best_score


def rank_average(oof_dict, pred_dict, model_names):
    """Rank averaging: convert probabilities to ranks then average.

    Args:
        oof_dict: Dict mapping model name to OOF predictions.
        pred_dict: Dict mapping model name to test predictions.
        model_names: List of model names to include.

    Returns:
        Tuple of (oof_blend, test_blend).
    """
    oof_ranks = []
    test_ranks = []
    for name in model_names:
        oof = oof_dict[name]
        test = pred_dict[name]
        # Rank each class column independently
        oof_r = np.column_stack([rankdata(oof[:, c]) for c in range(oof.shape[1])])
        test_r = np.column_stack([rankdata(test[:, c]) for c in range(test.shape[1])])
        oof_ranks.append(oof_r)
        test_ranks.append(test_r)

    oof_blend = np.mean(oof_ranks, axis=0)
    test_blend = np.mean(test_ranks, axis=0)
    return oof_blend, test_blend


def lgb_stacker(oof_dict, pred_dict, y_true, model_names, n_splits=5, seed=42):
    """LightGBM stacker on OOF probabilities.

    Uses all model OOF probabilities as features and trains a LightGBM
    classifier with 5-fold CV.

    Args:
        oof_dict: Dict mapping model name to OOF predictions.
        pred_dict: Dict mapping model name to test predictions.
        y_true: Ground truth labels.
        model_names: List of model names to use as features.
        n_splits: Number of CV folds.
        seed: Random state.

    Returns:
        Tuple of (oof_proba, test_proba, cv_score).
    """
    import lightgbm as lgb

    # Build feature matrix: concatenate all model probabilities
    X_oof = np.hstack([oof_dict[n] for n in model_names])
    X_test = np.hstack([pred_dict[n] for n in model_names])

    n_classes = 3
    oof_proba = np.zeros((len(y_true), n_classes))
    test_probas = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y_true)):
        X_tr, X_va = X_oof[tr_idx], X_oof[va_idx]
        y_tr, y_va = y_true[tr_idx], y_true[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": seed,
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        oof_proba[va_idx] = model.predict(X_va)
        test_probas.append(model.predict(X_test))

    test_proba = np.mean(test_probas, axis=0)
    cv_score = balanced_accuracy_score(y_true, oof_proba.argmax(axis=1))
    return oof_proba, test_proba, cv_score


def lr_stacker(oof_dict, pred_dict, y_true, model_names, n_splits=5, seed=42):
    """Logistic regression stacker on OOF probabilities.

    Args:
        oof_dict: Dict mapping model name to OOF predictions.
        pred_dict: Dict mapping model name to test predictions.
        y_true: Ground truth labels.
        model_names: List of model names to use as features.
        n_splits: Number of CV folds.
        seed: Random state.

    Returns:
        Tuple of (oof_proba, test_proba, cv_score).
    """
    X_oof = np.hstack([oof_dict[n] for n in model_names])
    X_test = np.hstack([pred_dict[n] for n in model_names])

    n_classes = 3
    oof_proba = np.zeros((len(y_true), n_classes))
    test_probas = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y_true)):
        X_tr, X_va = X_oof[tr_idx], X_oof[va_idx]
        y_tr, y_va = y_true[tr_idx], y_true[va_idx]

        model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial",
            random_state=seed,
        )
        model.fit(X_tr, y_tr)

        oof_proba[va_idx] = model.predict_proba(X_va)
        test_probas.append(model.predict_proba(X_test))

    test_proba = np.mean(test_probas, axis=0)
    cv_score = balanced_accuracy_score(y_true, oof_proba.argmax(axis=1))
    return oof_proba, test_proba, cv_score


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble v4: greedy selection + stacking"
    )
    parser.add_argument("--pred-dir", default="predictions")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument(
        "--models", default="",
        help="Comma-separated model names. Empty = auto-discover.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    train = pd.read_csv(data_dir / "train.csv")
    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])
    y = le.transform(train["Irrigation_Need"])
    test_ids = pd.read_csv(data_dir / ".." / ".." / "data" / "raw" / "test.csv")["id"]

    # Discover or use specified models
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        oof_files = sorted(pred_dir.glob("oof_*.npy"))
        model_names = [
            f.stem.replace("oof_", "") for f in oof_files
            if (pred_dir / f"pred_{f.stem.replace('oof_', '')}.npy").exists()
            and np.load(f).shape[0] == len(y)
        ]
    print(f"Candidate models ({len(model_names)}): {model_names}")

    oof_dict, pred_dict = load_predictions(pred_dir, model_names)

    # Individual scores
    print("\n=== Individual Model CV ===")
    scores = {}
    for name in model_names:
        score = balanced_accuracy_score(y, oof_dict[name].argmax(axis=1))
        scores[name] = score
        print(f"  {name:40s}  {score:.5f}")

    results = {}

    # --- Method 1: Greedy forward selection + equal weight ---
    print("\n=== Greedy Forward Selection ===")
    selected, gfs_weights, gfs_score = greedy_forward_selection(oof_dict, y)
    print(f"Selected {len(selected)} models, CV: {gfs_score:.5f}")

    oof_gfs = np.mean([oof_dict[n] for n in selected], axis=0)
    test_gfs = np.mean([pred_dict[n] for n in selected], axis=0)
    test_preds, cv, method = apply_best_threshold(oof_gfs, test_gfs, y)
    results[f"gfs_{len(selected)}m_{method}"] = (test_preds, cv, oof_gfs, test_gfs)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    # --- Method 2: Equal weight all models + thresholds ---
    print("\n=== Equal Weight All Models ===")
    oof_eq = np.mean([oof_dict[n] for n in model_names], axis=0)
    test_eq = np.mean([pred_dict[n] for n in model_names], axis=0)
    test_preds, cv, method = apply_best_threshold(oof_eq, test_eq, y)
    results[f"equal_{len(model_names)}m_{method}"] = (test_preds, cv, oof_eq, test_eq)
    print(f"  CV: {cv:.5f} ({method})")

    # --- Method 3: Rank averaging ---
    print("\n=== Rank Averaging ===")
    oof_rank, test_rank = rank_average(oof_dict, pred_dict, model_names)
    test_preds, cv, method = apply_best_threshold(oof_rank, test_rank, y)
    results[f"rank_{len(model_names)}m_{method}"] = (
        test_preds, cv, oof_rank, test_rank,
    )
    print(f"  CV: {cv:.5f} ({method})")

    # Rank avg on greedy-selected subset
    oof_rank_gfs, test_rank_gfs = rank_average(oof_dict, pred_dict, selected)
    test_preds, cv, method = apply_best_threshold(oof_rank_gfs, test_rank_gfs, y)
    results[f"rank_gfs_{len(selected)}m_{method}"] = (
        test_preds, cv, oof_rank_gfs, test_rank_gfs,
    )
    print(f"  Rank avg (greedy subset): {cv:.5f} ({method})")

    # --- Method 4: LightGBM stacker ---
    print("\n=== LightGBM Stacker ===")
    oof_lgb, test_lgb, cv_lgb = lgb_stacker(
        oof_dict, pred_dict, y, model_names,
    )
    print(f"  Raw LGB CV: {cv_lgb:.5f}")
    test_preds, cv, method = apply_best_threshold(oof_lgb, test_lgb, y)
    results[f"lgb_stack_{method}"] = (test_preds, cv, oof_lgb, test_lgb)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    # LGB stacker on greedy subset
    oof_lgb_gfs, test_lgb_gfs, cv_lgb_gfs = lgb_stacker(
        oof_dict, pred_dict, y, selected,
    )
    print(f"  LGB on greedy subset raw: {cv_lgb_gfs:.5f}")
    test_preds, cv, method = apply_best_threshold(oof_lgb_gfs, test_lgb_gfs, y)
    results[f"lgb_gfs_{method}"] = (test_preds, cv, oof_lgb_gfs, test_lgb_gfs)
    print(f"  After threshold: {cv:.5f} ({method})")

    # --- Method 5: Logistic Regression stacker ---
    print("\n=== Logistic Regression Stacker ===")
    oof_lr, test_lr, cv_lr = lr_stacker(oof_dict, pred_dict, y, model_names)
    print(f"  Raw LR CV: {cv_lr:.5f}")
    test_preds, cv, method = apply_best_threshold(oof_lr, test_lr, y)
    results[f"lr_stack_{method}"] = (test_preds, cv, oof_lr, test_lr)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("=== RESULTS SUMMARY ===")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    for name, (preds, cv, _, _) in sorted_results:
        print(f"  {name:40s}  CV: {cv:.5f}")

    # Save best submission
    best_name, (best_preds, best_cv, _, _) = sorted_results[0]
    labels = le.inverse_transform(best_preds)
    sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
    sub_path = output_dir / f"sub_ensemble_v4_{best_name}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nBest: {best_name} (CV {best_cv:.5f})")
    print(f"Submission saved: {sub_path}")
    print(sub["Irrigation_Need"].value_counts())

    # Also save second best if different
    if len(sorted_results) > 1:
        name2, (preds2, cv2, _, _) = sorted_results[1]
        labels2 = le.inverse_transform(preds2)
        sub2 = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels2})
        sub2_path = output_dir / f"sub_ensemble_v4_{name2}.csv"
        sub2.to_csv(sub2_path, index=False)
        print(f"\n2nd best: {name2} (CV {cv2:.5f}) -> {sub2_path}")


if __name__ == "__main__":
    main()
