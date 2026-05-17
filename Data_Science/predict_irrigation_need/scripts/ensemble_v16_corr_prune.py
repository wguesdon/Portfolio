#!/usr/bin/env python
"""Ensemble v16: correlation-pruned LGB stacker.

Many models in the 41-candidate pool are near-duplicates (xgb_ote_*,
lr variants, etc). Stacker coefficients become unstable. This script:

1. Compute pairwise correlation of OOF probabilities across all models.
2. Greedily keep a model iff its max correlation with already-kept
   models is below a threshold (0.99 by default).
3. Run LGB stacker on the pruned set + log_bias threshold tuning.
4. Compare vs full-pool stacker.

Usage:
    uv run python Playground_Series/PS6E4/scripts/ensemble_v16_corr_prune.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from ensemble_v4 import load_predictions, lgb_stacker, apply_best_threshold

BASE = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4")
PRED_DIR = BASE / "predictions"
DATA_DIR = BASE / "data" / "raw"
OUT_DIR = BASE / "submissions"
OUT_DIR.mkdir(exist_ok=True)

# Full v15 pool (41 models)
ALL_MODELS = [
    "xgb_v2", "xgb_s43", "xgb_s44",
    "xgb-2026-04-03-11-12-38-123", "xgb-2026-04-03-11-12-41-698",
    "lgb_v2", "lgb_s43", "lgb_s44",
    "cat_v2", "cat_s43", "cat_s44",
    "cat-2026-04-03-12-44-01-719", "cat-2026-04-03-12-44-05-363",
    "cat-2026-04-03-13-29-13-994",
    "realmlp_mahog", "lgb_v5", "xgb_v5", "realmlp_v3fix", "tabm_v3fix",
    "lgb_digit", "cat_digit",
    "xgb_ote", "cat_ote", "lgb_ote",
    "xgb_ote_s43", "xgb_ote_s44", "xgb_ote_magic",
    "lr_ote", "knn_ote", "et_ote",
    "lgb_ote_shallow", "lgb_ote_deep",
    "rf_ote", "svm_ote", "gnb_ote", "knn15_ote",
    "lr_cuml", "lr_l1_cuml", "knn5_lite",
    "xgb_ote_shallow_gpu", "lr_elastic",
]

CORR_THRESHOLD = 0.995


def compute_individual_cvs(oof_dict, y):
    return {n: balanced_accuracy_score(y, oof_dict[n].argmax(axis=1)) for n in oof_dict}


def compute_pairwise_corr(oof_dict, models):
    """Pearson correlation of flattened OOF probabilities."""
    n = len(models)
    corr = np.zeros((n, n))
    vecs = {m: oof_dict[m].flatten() for m in models}
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                corr[i, j] = 1.0
            elif i < j:
                c = np.corrcoef(vecs[a], vecs[b])[0, 1]
                corr[i, j] = c
                corr[j, i] = c
    return corr


def prune_by_correlation(oof_dict, y, models, threshold=0.99):
    """Greedy: sort by individual CV desc; keep a model iff max corr with
    already-kept < threshold."""
    cvs = compute_individual_cvs(oof_dict, y)
    ordered = sorted(models, key=lambda m: -cvs[m])
    corr = compute_pairwise_corr(oof_dict, ordered)
    name_to_idx = {m: i for i, m in enumerate(ordered)}

    kept = []
    dropped = []
    for m in ordered:
        if not kept:
            kept.append(m)
            continue
        i = name_to_idx[m]
        max_corr = max(corr[i, name_to_idx[k]] for k in kept)
        if max_corr < threshold:
            kept.append(m)
        else:
            dropped.append((m, cvs[m], max_corr))
    return kept, dropped, cvs


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])
    y = le.transform(train["Irrigation_Need"])
    test_ids = pd.read_csv(DATA_DIR / "test.csv")["id"]

    oof_dict, pred_dict = load_predictions(PRED_DIR, ALL_MODELS)

    print(f"Loaded {len(ALL_MODELS)} models.\n")

    # Prune
    kept, dropped, cvs = prune_by_correlation(
        oof_dict, y, ALL_MODELS, threshold=CORR_THRESHOLD
    )
    print(f"=== Pruning at correlation threshold {CORR_THRESHOLD} ===")
    print(f"Kept ({len(kept)}):")
    for m in kept:
        print(f"  {m:40s}  CV: {cvs[m]:.5f}")
    print(f"\nDropped ({len(dropped)}):")
    for m, cv, mc in dropped:
        print(f"  {m:40s}  CV: {cv:.5f}  max_corr: {mc:.4f}")

    # Run LGB stacker on kept set
    print(f"\n=== LGB stacker on {len(kept)} kept models ===")
    oof_s, test_s, cv_s = lgb_stacker(oof_dict, pred_dict, y, kept)
    print(f"  Raw CV: {cv_s:.5f}")
    preds, cv, method = apply_best_threshold(oof_s, test_s, y)
    print(f"  After thresholds: {cv:.5f} ({method})")

    # Save submission
    labels = le.inverse_transform(preds)
    sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
    name = f"sub_v16_corrprune{int(CORR_THRESHOLD*100)}_{len(kept)}m_{method}.csv"
    sub_path = OUT_DIR / name
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")
    print(f"Final CV: {cv:.5f}  ({len(kept)} models, threshold={CORR_THRESHOLD})")

    # Also save OOF + test probs for future blending
    np.save(PRED_DIR / f"oof_stack_v16_corrprune.npy", oof_s.astype(np.float32))
    np.save(PRED_DIR / f"pred_stack_v16_corrprune.npy", test_s.astype(np.float32))


if __name__ == "__main__":
    main()
