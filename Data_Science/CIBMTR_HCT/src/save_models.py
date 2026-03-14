"""
CIBMTR — Train and save all models for Kaggle dataset upload.

Saves to: kaggle_submission/dataset/
  - xgboost_model.json
  - lightgbm_model.txt
  - catboost_model.cbm
  - encoders.json   (label mapping for each categorical column)
  - feature_cols.json
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import RANDOM_STATE, CATEGORICAL_COLS
from features import load_and_prepare, get_Xy

SAVE_DIR = Path(__file__).parent.parent / "kaggle_submission" / "dataset"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Loading and preparing features...")
    df_train, df_test, feature_cols = load_and_prepare(include_test=True)
    X_train, y_time, y_efs = get_Xy(df_train, feature_cols)
    print(f"  Train: {len(X_train):,}  |  Features: {len(feature_cols)}")

    # Save feature column list
    with open(SAVE_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved  feature_cols.json  ({len(feature_cols)} features)")

    # Build and save encoder mappings (string → int for each categorical col)
    # Reconstruct from the raw data so the notebook can replicate preprocessing
    df_raw = pd.read_csv(Path(__file__).parent.parent / "data" / "train.csv")
    df_raw_test = pd.read_csv(Path(__file__).parent.parent / "data" / "test.csv")

    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df_raw.columns:
            continue
        combined = pd.concat([df_raw[col], df_raw_test[col]]).fillna("__missing__").astype(str)
        categories = sorted(combined.unique())
        encoders[col] = {cat: i for i, cat in enumerate(categories)}

    with open(SAVE_DIR / "encoders.json", "w") as f:
        json.dump(encoders, f, indent=2)
    print(f"  Saved  encoders.json  ({len(encoders)} columns)")

    cat_idx = [i for i, c in enumerate(feature_cols)
               if c in df_train.select_dtypes(include="object").columns]

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        enable_categorical=True,
    )
    xgb_model.fit(X_train, y_time)
    xgb_model.save_model(SAVE_DIR / "xgboost_model.json")
    print(f"  Saved  xgboost_model.json")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=600, num_leaves=63, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    lgb_model.fit(X_train, y_time)
    lgb_model.booster_.save_model(str(SAVE_DIR / "lightgbm_model.txt"))
    print(f"  Saved  lightgbm_model.txt")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    print("Training CatBoost...")
    cb_model = cb.CatBoostRegressor(
        iterations=600, depth=6, learning_rate=0.04,
        l2_leaf_reg=1.0, subsample=0.8, colsample_bylevel=0.7,
        random_seed=RANDOM_STATE, verbose=0,
    )
    cb_model.fit(X_train, y_time, cat_features=cat_idx if cat_idx else None, verbose=0)
    cb_model.save_model(str(SAVE_DIR / "catboost_model.cbm"))
    print(f"  Saved  catboost_model.cbm")

    print(f"\nAll model files saved to  {SAVE_DIR.relative_to(SAVE_DIR.parent.parent)}/")


if __name__ == "__main__":
    main()
