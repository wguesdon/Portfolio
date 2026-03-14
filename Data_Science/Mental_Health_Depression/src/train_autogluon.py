#!/usr/bin/env python
"""
AutoGluon trainer for mental health depression prediction (PS4E11).

Does NOT use base_trainer — AutoGluon manages its own CV, preprocessing, and ensembling.
Uses num_bag_folds=N so predict_proba(train_data) returns true OOF predictions
(each sample predicted only by models that never saw it during training).

OOF and test arrays saved as (n_samples, 2) to match lgb/xgb/cat artifacts.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from autogluon.tabular import TabularPredictor


def main():
    # SageMaker standard paths
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)

    # Load config
    with open(os.path.join(input_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)

    # SageMaker hyperparameters override config values
    sm_hps = {}
    hp_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hp_path):
        import json as _json
        with open(hp_path) as f:
            sm_hps = _json.load(f)

    target = config["competition"]["target_column"]
    n_folds = int(sm_hps.get("n_folds", config.get("training", {}).get("n_folds", 10)))
    time_limit = int(sm_hps.get(
        "time_limit_seconds",
        config.get("models", {}).get("autogluon", {}).get("time_limit_seconds", 7200)
    ))

    # Load raw data — AutoGluon handles its own preprocessing
    train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test = pd.read_csv(os.path.join(input_dir, "test.csv"))

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Drop id from train; keep all other columns (Name, Sleep Duration, etc.)
    train_features = train.drop(columns=["id"], errors="ignore")
    test_features = test.drop(columns=["id"], errors="ignore")

    print(f"Target distribution:\n{train_features[target].value_counts()}")

    # Train AutoGluon with bagging.
    # num_bag_folds=N → AutoGluon trains N models, each holding out 1/N of data.
    # predict_proba(train_data) then returns OOF predictions (no leakage).
    # num_stack_levels=0 keeps OOF strictly clean (no meta-learner on top).
    temp_model_path = tempfile.mkdtemp()

    print(f"\n{'='*50}")
    print(f"Training AutoGluon (time_limit={time_limit}s, num_bag_folds={n_folds})")
    print(f"{'='*50}")

    predictor = TabularPredictor(
        label=target,
        eval_metric="accuracy",
        path=temp_model_path,
    ).fit(
        train_features,
        presets="best_quality",
        time_limit=time_limit,
        num_bag_folds=n_folds,
        num_stack_levels=0,   # No stacking — keeps OOF predictions clean
        verbosity=2,
    )

    print("\nLeaderboard:")
    print(predictor.leaderboard(silent=True).to_string())

    # OOF predictions — AutoGluon returns OOF when num_bag_folds > 0
    print("\nGenerating OOF predictions...")
    oof_df = predictor.predict_proba(train_features, as_multiclass=True)

    # Ensure consistent column ordering [0, 1]
    cols = sorted(oof_df.columns)
    oof_array = oof_df[cols].values.astype(np.float32)
    print(f"OOF shape: {oof_array.shape}")

    # Test predictions
    print("Generating test predictions...")
    test_df = predictor.predict_proba(test_features, as_multiclass=True)
    test_array = test_df[cols].values.astype(np.float32)
    print(f"Test shape: {test_array.shape}")

    # CV accuracy from OOF
    y_true = train_features[target].values
    oof_acc = float((np.argmax(oof_array, axis=1) == y_true).mean())
    print(f"\nOOF Accuracy: {oof_acc:.5f}")

    # Save artifacts
    np.save(os.path.join(model_dir, "autogluon_oof.npy"), oof_array)
    np.save(os.path.join(model_dir, "autogluon_test.npy"), test_array)

    results = {
        "model": "autogluon",
        "cv_accuracy_mean": oof_acc,
        "cv_accuracy_std": 0.0,
        "fold_scores": [],
        "best_params": {"presets": "best_quality", "num_bag_folds": n_folds, "num_stack_levels": 0},
        "n_trials": 0,
        "n_folds": n_folds,
        "n_features": train_features.shape[1] - 1,
        "oof_shape": list(oof_array.shape),
        "test_shape": list(test_array.shape),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(model_dir, "autogluon_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to {model_dir}")
    print(f"  OOF shape:  {oof_array.shape}")
    print(f"  Test shape: {test_array.shape}")
    print(f"\n{'='*50}")
    print(f"AUTOGLUON COMPLETE — OOF Accuracy: {oof_acc:.5f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
