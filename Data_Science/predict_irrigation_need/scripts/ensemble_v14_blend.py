#!/usr/bin/env python
"""Ensemble v14: blend multiple LGB stackers across different pools and seeds.

Runs the LGB stacker locally on v11, v12, v13 model pools and also with
multiple seeds, then blends the resulting OOF/test probabilities.

Usage:
    uv run python Playground_Series/PS6E4/scripts/ensemble_v14_blend.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from ensemble_v4 import (
    load_predictions,
    lgb_stacker,
    apply_best_threshold,
)

BASE = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4")
PRED_DIR = BASE / "predictions"
DATA_DIR = BASE / "data" / "raw"
OUT_DIR = BASE / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# v11 pool (30 models, pre lgb_ote_shallow/deep and pre cuML)
V11_MODELS = [
    "xgb_v2", "xgb_s43", "xgb_s44",
    "xgb-2026-04-03-11-12-38-123", "xgb-2026-04-03-11-12-41-698",
    "lgb_v2", "lgb_s43", "lgb_s44",
    "cat_v2", "cat_s43", "cat_s44",
    "cat-2026-04-03-12-44-01-719", "cat-2026-04-03-12-44-05-363",
    "cat-2026-04-03-13-29-13-994",
    "realmlp_mahog", "lgb_v5", "xgb_v5",
    "realmlp_v3fix", "tabm_v3fix",
    "lgb_digit", "cat_digit",
    "xgb_ote", "cat_ote", "lgb_ote",
    "xgb_ote_s43", "xgb_ote_s44", "xgb_ote_magic",
    "lr_ote", "knn_ote", "et_ote",
]

V12_MODELS = V11_MODELS + ["lgb_ote_shallow", "lgb_ote_deep"]

V13_MODELS = V12_MODELS + [
    "rf_ote", "svm_ote", "gnb_ote", "knn15_ote",
    "lr_cuml", "lr_l1_cuml", "knn5_lite",
]

# Load y_true
train = pd.read_csv(DATA_DIR / "train.csv")
le = LabelEncoder()
le.fit(["High", "Low", "Medium"])
y = le.transform(train["Irrigation_Need"])
test_ids = pd.read_csv(DATA_DIR / "test.csv")["id"]


def run_stacker(models, pool_name, seeds=(42,)):
    """Run LGB stacker on a pool with multiple seeds, average the probabilities."""
    oof_dict, pred_dict = load_predictions(PRED_DIR, models)
    oof_list, test_list = [], []
    for seed in seeds:
        print(f"  [{pool_name}] LGB stacker seed {seed}...")
        oof, test, cv = lgb_stacker(oof_dict, pred_dict, y, models, seed=seed)
        print(f"    raw CV: {cv:.5f}")
        oof_list.append(oof)
        test_list.append(test)
    oof_mean = np.mean(oof_list, axis=0)
    test_mean = np.mean(test_list, axis=0)
    raw_cv = balanced_accuracy_score(y, oof_mean.argmax(axis=1))
    print(f"  [{pool_name}] Averaged over {len(seeds)} seeds, raw CV: {raw_cv:.5f}")
    return oof_mean, test_mean


def main():
    results = {}

    # v12 stacker with 3 seeds (seed averaging)
    print("\n=== v12 pool (32 models) with seed averaging ===")
    oof_v12, test_v12 = run_stacker(V12_MODELS, "v12", seeds=(42, 43, 44))
    preds, cv, method = apply_best_threshold(oof_v12, test_v12, y)
    results[f"v12_seed3_{method}"] = (preds, cv, oof_v12, test_v12)
    print(f"  After thresholds: {cv:.5f} ({method})")

    # v11 stacker with 3 seeds
    print("\n=== v11 pool (30 models) with seed averaging ===")
    oof_v11, test_v11 = run_stacker(V11_MODELS, "v11", seeds=(42, 43, 44))
    preds, cv, method = apply_best_threshold(oof_v11, test_v11, y)
    results[f"v11_seed3_{method}"] = (preds, cv, oof_v11, test_v11)
    print(f"  After thresholds: {cv:.5f} ({method})")

    # Blend v11 + v12 (equal weight)
    print("\n=== Blend: v11 + v12 stackers (equal weight) ===")
    oof_blend = (oof_v11 + oof_v12) / 2
    test_blend = (test_v11 + test_v12) / 2
    preds, cv, method = apply_best_threshold(oof_blend, test_blend, y)
    results[f"blend_v11v12_{method}"] = (preds, cv, oof_blend, test_blend)
    print(f"  After thresholds: {cv:.5f} ({method})")

    # Summary
    print("\n" + "=" * 60)
    print("=== RESULTS SUMMARY ===")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    for name, (preds, cv, _, _) in sorted_results:
        print(f"  {name:40s}  CV: {cv:.5f}")

    # Save submissions
    for name, (preds, cv, oof, test) in sorted_results:
        labels = le.inverse_transform(preds)
        sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
        sub_path = OUT_DIR / f"sub_v14_{name}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path.name} (CV {cv:.5f})")
        # Save raw probabilities too for later reuse
        np.save(PRED_DIR / f"oof_stack_{name}.npy", oof.astype(np.float32))
        np.save(PRED_DIR / f"pred_stack_{name}.npy", test.astype(np.float32))


if __name__ == "__main__":
    main()
