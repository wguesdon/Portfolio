#!/usr/bin/env python
"""Ensemble v3: log-space bias tuning + differential evolution.

v3 changes over v2:
  - Log-space bias tuning (additive shift in log(p) space)
  - Grid climb with decreasing step sizes (1.0 -> 0.005)
  - Combines with differential evolution and picks the best
  Source: Mahog's top-5 notebook

Usage:
    uv run python scripts/ensemble_v3.py
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import differential_evolution


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


def hill_climbing_ensemble(oof_dict, y_true, n_iterations=10000, seed=42):
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


def tune_bias_log(proba, y_true):
    """Log-space bias tuning via multi-step grid climb.

    Adds an additive bias to log(p) before argmax. This is equivalent
    to adjusting class priors. The grid climb explores decreasing step
    sizes from 1.0 down to 0.005.

    Source: Mahog's top-5 notebook.

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


def main():
    parser = argparse.ArgumentParser(description="Ensemble v3 with log-space bias tuning")
    parser.add_argument("--pred-dir", default="predictions",
                        help="Directory with OOF/test .npy files")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Directory with train.csv")
    parser.add_argument("--output-dir", default="submissions",
                        help="Directory for submission CSV")
    parser.add_argument("--models", default="",
                        help="Comma-separated list of model names to use. "
                             "If empty, auto-discovers all valid OOF/pred pairs.")
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
        print(f"Using {len(model_names)} specified models: {model_names}")
    else:
        oof_files = sorted(pred_dir.glob("oof_*.npy"))
        model_names = [f.stem.replace("oof_", "") for f in oof_files
                       if (pred_dir / f"pred_{f.stem.replace('oof_', '')}.npy").exists()
                       and np.load(f).shape[0] == len(y)]
        print(f"Auto-discovered {len(model_names)} models: {model_names}")

    oof_dict, pred_dict = load_predictions(pred_dir, model_names)

    # Individual scores
    print("\nIndividual model CV:")
    for name in model_names:
        score = balanced_accuracy_score(y, oof_dict[name].argmax(axis=1))
        print(f"  {name:25s}  {score:.5f}")

    # Hill climbing weights
    print("\nHill climbing (10000 iterations)...")
    weights, hc_score = hill_climbing_ensemble(oof_dict, y)
    print(f"  HC score: {hc_score:.5f}")

    # Blend
    oof_blend = sum(weights[n] * oof_dict[n] for n in model_names)
    test_blend = sum(weights[n] * pred_dict[n] for n in model_names)

    # Method A: Log-space bias tuning
    print("\nMethod A: Log-space bias tuning...")
    bias_A, score_A = tune_bias_log(oof_blend, y)
    print(f"  Bias: {np.round(bias_A, 4)}")
    print(f"  Score: {score_A:.5f}")

    # Method B: Differential evolution
    print("\nMethod B: Differential evolution thresholds...")
    thresh_B, score_B = diffevol_thresholds(oof_blend, y)
    print(f"  Thresholds: {np.round(thresh_B, 4)}")
    print(f"  Score: {score_B:.5f}")

    # Pick best
    if score_A >= score_B:
        print(f"\nBest: log-space bias (CV {score_A:.5f})")
        final_preds = np.argmax(np.log(np.clip(test_blend, 1e-15, 1.0)) + bias_A, axis=1)
        best_score = score_A
        method = "log_bias"
    else:
        print(f"\nBest: diff evolution (CV {score_B:.5f})")
        final_preds = (test_blend * thresh_B).argmax(axis=1)
        best_score = score_B
        method = "diffevol"

    # Save submission
    labels = le.inverse_transform(final_preds)
    sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
    n = len(model_names)
    sub_path = output_dir / f"sub_ensemble_v3_{n}models_{method}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSubmission saved: {sub_path}")
    print(f"CV: {best_score:.5f}")
    print(sub["Irrigation_Need"].value_counts())


if __name__ == "__main__":
    main()
