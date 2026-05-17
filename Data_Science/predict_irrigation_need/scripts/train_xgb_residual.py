#!/usr/bin/env python
"""SageMaker entry point: XGBoost residual model for PS6E4.

The exact formula (Chris Deotte) perfectly classifies the original 10K data.
The synthetic 630K data has noise from the generation process.
This script trains a model on the RESIDUALS between the formula's predictions
and actual labels, then combines formula + corrections at inference.

Pipeline:
  1. Compute exact formula predictions for all samples
  2. Build residual targets: actual_one_hot - formula_one_hot
  3. Train 3 XGBoost regressors (one per class residual) with OTE features
  4. Final prediction = formula_proba + predicted_residual, then argmax

Saves oof_xgb_residual.npy and pred_xgb_residual.npy with our label encoding.
"""
import os
import gc
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

INPUT_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 3
TARGET_COL = "Irrigation_Need"


# ============================================================
# EXACT FORMULA
# ============================================================

def exact_formula_pred(df):
    """Compute the exact formula prediction for each row.

    Returns array of shape (n,) with predicted class indices
    using the internal label map (Low=0, Medium=1, High=2).
    """
    soil_dry = (df["Soil_Moisture"] < 25).astype(int)
    rain_low = (df["Rainfall_mm"] < 300).astype(int)
    temp_hot = (df["Temperature_C"] > 30).astype(int)
    wind_high = (df["Wind_Speed_kmh"] > 10).astype(int)
    harvest = (df["Crop_Growth_Stage"] == "Harvest").astype(int)
    sowing = (df["Crop_Growth_Stage"] == "Sowing").astype(int)
    mulch = (df["Mulching_Used"] == "Yes").astype(int)

    high_score = soil_dry * 2 + rain_low * 2 + temp_hot + wind_high
    low_score = harvest * 2 + sowing * 2 + mulch
    magic_score = high_score - low_score

    # score <= 0 -> Low(0), score >= 4 -> High(2), else Medium(1)
    pred = np.where(magic_score <= 0, 0, np.where(magic_score >= 4, 2, 1))
    return pred, magic_score


def formula_to_onehot(pred, n_classes=3):
    """Convert class predictions to one-hot probability matrix."""
    onehot = np.zeros((len(pred), n_classes), dtype=np.float32)
    for i, c in enumerate(pred):
        onehot[i, c] = 1.0
    return onehot


# ============================================================
# ORDERED TARGET ENCODER
# ============================================================

class OrderedTE:
    """Ordered (leakage-free) multi-class Target Encoder with Bayesian smoothing."""

    def __init__(self, a=1.0):
        self.a = a

    def fit(self, train, category_cols=None, target_col="target"):
        self.train = train
        self.target_col = target_col
        self.category_cols = category_cols or []
        self.classes_ = sorted(train[target_col].unique())
        self.num_classes_ = len(self.classes_)
        self.global_prior_ = (
            train[target_col].value_counts(normalize=True).sort_index().values
        )

        for c in self.category_cols:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y_binary = (train[target_col] == cls).astype(int)
                df = train[[c]].copy()
                df["y"] = y_binary.values
                df["cnt"] = 1
                df["cum_cnt"] = df.groupby(c)["cnt"].cumsum() - df["cnt"]
                df["cum_sum"] = df.groupby(c)["y"].cumsum() - df["y"]

                smooth_prior = self.a * self.global_prior_[k]
                te_col = f"{c}_TE_cls{cls}"
                df[te_col] = (df["cum_sum"] + smooth_prior) / (df["cum_cnt"] + self.a)
                df.loc[df["cum_cnt"] == -1, te_col] = self.global_prior_[k]
                self.train[te_col] = df[te_col].values

                stats_df = df.groupby(c)["y"].agg(["count", "sum"]).reset_index()
                stats_df.columns = [c, f"{c}_count_cls{cls}", f"{c}_sum_cls{cls}"]
                stats_df[f"{c}_prior_cls{cls}"] = self.global_prior_[k]
                stats_list.append(stats_df)

            combined_stats = stats_list[0]
            for i in range(1, len(stats_list)):
                combined_stats = combined_stats.merge(stats_list[i], on=c, how="outer")
            setattr(self, f"{c}_stats", combined_stats)

        return self.train

    def transform(self, test):
        for c in self.category_cols:
            stats_df = getattr(self, f"{c}_stats")
            test = test.merge(stats_df, on=c, how="left")

            for k, cls in enumerate(self.classes_):
                te_col = f"{c}_TE_cls{cls}"
                count_col = f"{c}_count_cls{cls}"
                sum_col = f"{c}_sum_cls{cls}"
                prior_col = f"{c}_prior_cls{cls}"

                if count_col in test.columns:
                    test[te_col] = (
                        (test[sum_col] + self.a * test[prior_col])
                        / (test[count_col] + self.a)
                    )
                    test[te_col] = test[te_col].fillna(test[prior_col])
                    test.drop([count_col, sum_col, prior_col], axis=1, inplace=True)
                else:
                    test[te_col] = self.global_prior_[k]

            gc.collect()
        return test


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(train_df, test_df, nums, cats):
    M = train_df[nums].max()

    def fe(df):
        for c in nums:
            for k in range(-4, 4):
                df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype("int8")
            if M[c] < 10:
                df[c] = df[c].round(3)
            elif M[c] < 100:
                df[c] = df[c].round(2)
            else:
                df[c] = df[c].round(1)
        return df

    train_df = fe(train_df)
    test_df = fe(test_df)

    # Add magic formula features as additional context for the residual model
    for df in [train_df, test_df]:
        _, magic_score = exact_formula_pred(df)
        df["magic_score"] = magic_score.astype(np.float32)
        df["magic_soil_margin"] = (25 - df["Soil_Moisture"]).astype(np.float32)
        df["magic_rain_margin"] = (300 - df["Rainfall_mm"]).astype(np.float32)
        df["magic_temp_margin"] = (df["Temperature_C"] - 30).astype(np.float32)
        df["magic_wind_margin"] = (df["Wind_Speed_kmh"] - 10).astype(np.float32)
        df["magic_boundary_dist"] = df[
            ["magic_soil_margin", "magic_rain_margin",
             "magic_temp_margin", "magic_wind_margin"]
        ].abs().min(axis=1).astype(np.float32)

    magic_nums = ["magic_score", "magic_soil_margin", "magic_rain_margin",
                  "magic_temp_margin", "magic_wind_margin", "magic_boundary_dist"]

    drop = [c for c in test_df.columns if test_df[c].nunique() == 1]
    if drop:
        print(f"Dropping {len(drop)} constant columns")
        train_df.drop(drop, axis=1, inplace=True)
        test_df.drop(drop, axis=1, inplace=True)

    category_cols = cats + [c for c in test_df.columns if "digit" in c]
    for c in category_cols:
        freq = train_df[c].value_counts()
        mapping = {val: idx for idx, (val, _) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train_df[c] = train_df[c].map(lambda x, m=mapping, d=mapping_default: m.get(x, d))
        test_df[c] = test_df[c].map(lambda x, m=mapping, d=mapping_default: m.get(x, d))

    features = category_cols + nums + magic_nums
    return train_df, test_df, features


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Residual Model (Formula + OTE corrections)")
    print("=" * 60)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    print(f"Train: {train.shape}, Test: {test.shape}")

    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])

    # Internal label encoding
    target2idx = {v: i for i, v in enumerate(train[TARGET_COL].unique())}
    idx2target = {v: k for k, v in target2idx.items()}
    print(f"Internal label map: {target2idx}")

    # Compute formula predictions BEFORE encoding target
    formula_pred_train, _ = exact_formula_pred(train)
    formula_pred_test, _ = exact_formula_pred(test)

    # Remap formula preds to internal encoding
    # Formula uses: Low=0, Medium=1, High=2
    # Internal uses: whatever train[TARGET_COL].unique() gives
    formula_label_map = {"Low": 0, "Medium": 1, "High": 2}
    internal_remap = {}
    for label, formula_idx in formula_label_map.items():
        internal_remap[formula_idx] = target2idx[label]
    formula_pred_train = np.array([internal_remap[p] for p in formula_pred_train])
    formula_pred_test = np.array([internal_remap[p] for p in formula_pred_test])

    train[TARGET_COL] = train[TARGET_COL].map(target2idx)

    # Compute formula accuracy
    formula_ba = balanced_accuracy_score(train[TARGET_COL], formula_pred_train)
    formula_agree = (train[TARGET_COL].values == formula_pred_train).mean()
    print(f"Formula BA on train: {formula_ba:.5f}")
    print(f"Formula agreement rate: {formula_agree:.4f}")

    # Compute residual targets
    # actual_one_hot - formula_one_hot gives correction vectors in [-1, 1]
    actual_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[train[TARGET_COL].values]
    formula_onehot_train = formula_to_onehot(formula_pred_train, NUM_CLASSES)
    residuals = actual_onehot - formula_onehot_train

    print(f"Residual stats per class:")
    for c in range(NUM_CLASSES):
        print(f"  Class {c}: mean={residuals[:, c].mean():.5f}, "
              f"std={residuals[:, c].std():.5f}, "
              f"nonzero={np.count_nonzero(residuals[:, c])}")

    cats = [c for c in test.columns if train[c].dtype == object and c != "id"]
    nums = [c for c in test.columns if c not in cats and c != "id"]

    train.drop("id", axis=1, inplace=True)
    test_ids = test["id"].copy()
    test.drop("id", axis=1, inplace=True)

    train, test, features = feature_engineering(train, test, nums, cats)
    print(f"Features after FE: {len(features)}, Train cols: {train.shape[1]}")

    X = train.drop([TARGET_COL], axis=1)
    y = train[TARGET_COL]
    test_X = test.copy()

    # OOF residual predictions (3 regressors, one per class)
    oof_residuals = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    test_residuals = np.zeros((len(test_X), NUM_CLASSES), dtype=np.float32)

    xgb_params = {
        "max_depth": 4,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "n_estimators": 512,
        "learning_rate": 0.1,
        "early_stopping_rounds": 50,
        "random_state": 2026,
        "n_jobs": -1,
        "tree_method": "hist",
        "max_bin": 10000,
        "device": "cpu",
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {N_FOLDS}")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        res_train = residuals[train_idx]
        res_val = residuals[val_idx]

        # OTE encoding
        te = OrderedTE()
        full_df = pd.concat((X_train, y_train), axis=1)

        te_train = pd.concat([
            te.fit(
                full_df.sample(frac=1, random_state=42 + i),
                category_cols=features,
                target_col=TARGET_COL,
            )
            for i in range(4)
        ])

        # For residual training, we need the original (non-4x) indices
        # Use transform instead to get proper non-leaky encoding
        X_train_te = te.transform(X_train.copy())
        X_val_te = te.transform(X_val)
        X_test_te = te.transform(test_X.copy())

        del full_df, te_train
        gc.collect()

        X_train_te.drop(cats, axis=1, inplace=True)
        X_val_te.drop(cats, axis=1, inplace=True)
        X_test_te.drop(cats, axis=1, inplace=True)

        # Train one regressor per class residual
        fold_test_res = np.zeros((len(test_X), NUM_CLASSES), dtype=np.float32)

        for c in range(NUM_CLASSES):
            model = XGBRegressor(**xgb_params)
            model.fit(
                X_train_te, res_train[:, c],
                eval_set=[(X_val_te, res_val[:, c])],
                verbose=0,
            )
            oof_residuals[val_idx, c] = model.predict(X_val_te)
            fold_test_res[:, c] = model.predict(X_test_te)
            del model

        test_residuals += fold_test_res / N_FOLDS

        # Combine formula + residual for this fold's validation
        formula_val = formula_onehot_train[val_idx]
        corrected_val = formula_val + oof_residuals[val_idx]
        fold_score = balanced_accuracy_score(y_val, corrected_val.argmax(axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold} BA (formula only): "
              f"{balanced_accuracy_score(y_val, formula_pred_train[val_idx]):.5f}")
        print(f"  Fold {fold} BA (formula + residual): {fold_score:.5f}")

        del X_train_te, X_val_te, X_test_te
        gc.collect()

    # Final OOF evaluation
    corrected_oof = formula_onehot_train + oof_residuals
    oof_cv = balanced_accuracy_score(y.values, corrected_oof.argmax(axis=1))
    print(f"\n{'='*60}")
    print(f"  Formula-only CV: {formula_ba:.5f}")
    print(f"  Residual model CV: {oof_cv:.5f}")
    print(f"  Improvement: +{oof_cv - formula_ba:.5f}")
    print(f"  Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"{'='*60}")

    # Final predictions: formula + residual corrections
    formula_onehot_test = formula_to_onehot(formula_pred_test, NUM_CLASSES)
    corrected_test = formula_onehot_test + test_residuals

    # Remap to our label encoding: High=0, Low=1, Medium=2
    reorder = []
    our_labels = le.classes_
    for our_idx, label in enumerate(our_labels):
        internal_idx = target2idx[label]
        reorder.append(internal_idx)

    print(f"Internal label map: {target2idx}")
    print(f"Our label order: {list(our_labels)}")
    print(f"Reorder columns: {reorder}")

    oof_reordered = corrected_oof[:, reorder]
    pred_reordered = corrected_test[:, reorder]

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "oof_xgb_residual.npy"), oof_reordered.astype(np.float32))
    np.save(os.path.join(MODEL_DIR, "pred_xgb_residual.npy"), pred_reordered.astype(np.float32))

    print(f"\nSaved oof_xgb_residual.npy {oof_reordered.shape}")
    print(f"Saved pred_xgb_residual.npy {pred_reordered.shape}")
    print("Label order: High=0, Low=1, Medium=2")
    print("\nDone.")
