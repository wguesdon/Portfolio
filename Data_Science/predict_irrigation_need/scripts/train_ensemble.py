#!/usr/bin/env python
"""SageMaker entry point for ensemble_v4.

Reads data from /opt/ml/input/data/training/ and writes results
to /opt/ml/model/.
"""
import os
import sys

# SageMaker paths
INPUT_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# Patch ensemble_v4 paths
sys.argv = [
    "ensemble_v4.py",
    "--pred-dir", os.path.join(INPUT_DIR, "predictions"),
    "--data-dir", INPUT_DIR,
    "--output-dir", MODEL_DIR,
    "--models", "xgb_v2,xgb_s43,xgb_s44,xgb-2026-04-03-11-12-38-123,"
    "xgb-2026-04-03-11-12-41-698,lgb_v2,lgb_s43,lgb_s44,"
    "cat_v2,cat_s43,cat_s44,cat-2026-04-03-12-44-01-719,"
    "cat-2026-04-03-12-44-05-363,cat-2026-04-03-13-29-13-994,"
    "realmlp_mahog,lgb_v5,xgb_v5,realmlp_v3fix,tabm_v3fix,lgb_digit,cat_digit,"
    "xgb_ote,cat_ote,lgb_ote,xgb_ote_s43,xgb_ote_s44,xgb_ote_magic,"
    "lr_ote,knn_ote,et_ote,"
    "lgb_ote_shallow,lgb_ote_deep,"
    "rf_ote,svm_ote,gnb_ote,knn15_ote,"
    "lr_cuml,lr_l1_cuml,knn5_lite,"
    "xgb_ote_shallow_gpu,lr_elastic,"
    "xgb_ote_shallow,cat_ote_deep",
]

# Fix test.csv path in ensemble_v4 - monkey-patch before import
import ensemble_v4
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

_orig_main = ensemble_v4.main


def patched_main():
    """Run ensemble with SageMaker-compatible paths."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="predictions")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--models", default="")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])
    y = le.transform(train["Irrigation_Need"])
    test_ids = pd.read_csv(data_dir / "test.csv")["id"]

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Candidate models ({len(model_names)}): {model_names}")

    oof_dict, pred_dict = ensemble_v4.load_predictions(pred_dir, model_names)

    from sklearn.metrics import balanced_accuracy_score

    print("\n=== Individual Model CV ===")
    for name in model_names:
        score = balanced_accuracy_score(y, oof_dict[name].argmax(axis=1))
        print(f"  {name:40s}  {score:.5f}")

    results = {}

    # Greedy forward selection
    print("\n=== Greedy Forward Selection ===")
    selected, gfs_weights, gfs_score = ensemble_v4.greedy_forward_selection(
        oof_dict, y,
    )
    print(f"Selected {len(selected)} models, CV: {gfs_score:.5f}")

    oof_gfs = np.mean([oof_dict[n] for n in selected], axis=0)
    test_gfs = np.mean([pred_dict[n] for n in selected], axis=0)
    test_preds, cv, method = ensemble_v4.apply_best_threshold(oof_gfs, test_gfs, y)
    results[f"gfs_{len(selected)}m_{method}"] = (test_preds, cv, oof_gfs, test_gfs)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    # Equal weight all
    print("\n=== Equal Weight All Models ===")
    oof_eq = np.mean([oof_dict[n] for n in model_names], axis=0)
    test_eq = np.mean([pred_dict[n] for n in model_names], axis=0)
    test_preds, cv, method = ensemble_v4.apply_best_threshold(oof_eq, test_eq, y)
    results[f"equal_{len(model_names)}m_{method}"] = (test_preds, cv, oof_eq, test_eq)
    print(f"  CV: {cv:.5f} ({method})")

    # Rank averaging
    print("\n=== Rank Averaging ===")
    oof_rank, test_rank = ensemble_v4.rank_average(
        oof_dict, pred_dict, model_names,
    )
    test_preds, cv, method = ensemble_v4.apply_best_threshold(
        oof_rank, test_rank, y,
    )
    results[f"rank_{len(model_names)}m_{method}"] = (
        test_preds, cv, oof_rank, test_rank,
    )
    print(f"  CV: {cv:.5f} ({method})")

    oof_rank_gfs, test_rank_gfs = ensemble_v4.rank_average(
        oof_dict, pred_dict, selected,
    )
    test_preds, cv, method = ensemble_v4.apply_best_threshold(
        oof_rank_gfs, test_rank_gfs, y,
    )
    results[f"rank_gfs_{len(selected)}m_{method}"] = (
        test_preds, cv, oof_rank_gfs, test_rank_gfs,
    )
    print(f"  Rank avg (greedy subset): {cv:.5f} ({method})")

    # LGB stacker
    print("\n=== LightGBM Stacker ===")
    oof_lgb, test_lgb, cv_lgb = ensemble_v4.lgb_stacker(
        oof_dict, pred_dict, y, model_names,
    )
    print(f"  Raw LGB CV: {cv_lgb:.5f}")
    test_preds, cv, method = ensemble_v4.apply_best_threshold(oof_lgb, test_lgb, y)
    results[f"lgb_stack_{method}"] = (test_preds, cv, oof_lgb, test_lgb)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    oof_lgb_gfs, test_lgb_gfs, cv_lgb_gfs = ensemble_v4.lgb_stacker(
        oof_dict, pred_dict, y, selected,
    )
    print(f"  LGB on greedy subset raw: {cv_lgb_gfs:.5f}")
    test_preds, cv, method = ensemble_v4.apply_best_threshold(
        oof_lgb_gfs, test_lgb_gfs, y,
    )
    results[f"lgb_gfs_{method}"] = (test_preds, cv, oof_lgb_gfs, test_lgb_gfs)
    print(f"  After threshold: {cv:.5f} ({method})")

    # LR stacker
    print("\n=== Logistic Regression Stacker ===")
    oof_lr, test_lr, cv_lr = ensemble_v4.lr_stacker(
        oof_dict, pred_dict, y, model_names,
    )
    print(f"  Raw LR CV: {cv_lr:.5f}")
    test_preds, cv, method = ensemble_v4.apply_best_threshold(oof_lr, test_lr, y)
    results[f"lr_stack_{method}"] = (test_preds, cv, oof_lr, test_lr)
    print(f"  After threshold tuning: {cv:.5f} ({method})")

    # Summary
    print("\n" + "=" * 60)
    print("=== RESULTS SUMMARY ===")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: -x[1][1])
    for name, (preds, cv, _, _) in sorted_results:
        print(f"  {name:40s}  CV: {cv:.5f}")

    # Save all submissions
    for name, (preds, cv, _, _) in sorted_results:
        labels = le.inverse_transform(preds)
        sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
        sub_path = output_dir / f"sub_v4_{name}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path}")

    best_name, (_, best_cv, _, _) = sorted_results[0]
    print(f"\nBest: {best_name} (CV {best_cv:.5f})")


if __name__ == "__main__":
    patched_main()
