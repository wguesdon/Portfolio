#!/usr/bin/env python
"""SageMaker entry point: RealMLP + Ordered Target Encoding for PS6E4.

Same OTE pipeline as train_xgb_ote.py but with RealMLP (pytabkit) classifier.
Uses StratifiedKFold(seed=42) matching our ensemble fold strategy.
Saves oof_realmlp_ote.npy and pred_realmlp_ote.npy with our label encoding (High=0, Low=1, Medium=2).
"""
import subprocess
import sys
import os
import gc
import warnings

warnings.filterwarnings("ignore")

# Install pytabkit (not in DLC by default)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit"])

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

INPUT_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 3
TARGET_COL = "Irrigation_Need"


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

    features = category_cols + nums
    return train_df, test_df, features


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import torch
    print("=" * 60)
    print("RealMLP + Ordered Target Encoding")
    print("=" * 60)
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    print(f"Train: {train.shape}, Test: {test.shape}")

    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])

    target2idx = {v: i for i, v in enumerate(train[TARGET_COL].unique())}
    idx2target = {v: k for k, v in target2idx.items()}
    print(f"Internal label map: {target2idx}")

    train[TARGET_COL] = train[TARGET_COL].map(target2idx)

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

    oof_preds = np.zeros((len(y), NUM_CLASSES))
    test_preds = np.zeros((len(test_X), NUM_CLASSES))

    # OTE does NOT use 4x shuffle for RealMLP because:
    # - RealMLP trains on raw data, not 4x expanded data
    # - We apply OTE transform (full-fold stats) to both train and val
    # - This avoids the 4x memory overhead for the neural net

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {N_FOLDS}")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        # OTE: fit on shuffled train, transform val and test
        te = OrderedTE()
        full_df = pd.concat((X_train, y_train), axis=1)

        # Fit OTE (single pass, no 4x expansion for NN)
        te.fit(
            full_df.sample(frac=1, random_state=42),
            category_cols=features,
            target_col=TARGET_COL,
        )

        # For training, use the transform (full-fold stats, not cumulative)
        # to avoid information leakage artifacts from the cumulative encoding
        X_train_te = te.transform(X_train.copy())
        X_val_te = te.transform(X_val)
        X_test_te = te.transform(test_X.copy())

        # Drop original categoricals
        X_train_te.drop(cats, axis=1, inplace=True)
        X_val_te.drop(cats, axis=1, inplace=True)
        X_test_te.drop(cats, axis=1, inplace=True)

        # Fill NaNs (OTE can produce NaNs for unseen categories) and convert
        X_train_te = X_train_te.fillna(0)
        X_val_te = X_val_te.fillna(0)
        X_test_te = X_test_te.fillna(0)

        # Convert to float32 numpy arrays (RealMLP expects numeric, no NaNs)
        X_tr_np = X_train_te.values.astype(np.float32)
        X_va_np = X_val_te.values.astype(np.float32)
        X_te_np = X_test_te.values.astype(np.float32)
        y_tr_np = y_train.values

        print(f"  Features: {X_tr_np.shape[1]}")
        print(f"  Train: {X_tr_np.shape[0]}, Val: {X_va_np.shape[0]}")

        model = RealMLP_TD_Classifier(
            n_cv=1,
            n_refit=0,
            n_ens=5,
            device=device,
            verbosity=2,
        )
        model.fit(X_tr_np, y_tr_np)

        y_pred = model.predict_proba(X_va_np)
        oof_preds[val_idx] = y_pred

        test_preds += model.predict_proba(X_te_np) / N_FOLDS

        fold_score = balanced_accuracy_score(y_val, np.argmax(y_pred, axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold} balanced accuracy: {fold_score:.5f}")

        del X_train_te, X_val_te, X_test_te, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_cv = balanced_accuracy_score(y.values, np.argmax(oof_preds, axis=1))
    print(f"\n{'='*60}")
    print(f"  Overall OOF CV: {oof_cv:.5f}")
    print(f"  Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"{'='*60}")

    reorder = []
    our_labels = le.classes_
    for our_idx, label in enumerate(our_labels):
        internal_idx = target2idx[label]
        reorder.append(internal_idx)

    print(f"Internal label map: {target2idx}")
    print(f"Our label order: {list(our_labels)}")
    print(f"Reorder columns: {reorder}")

    oof_reordered = oof_preds[:, reorder]
    pred_reordered = test_preds[:, reorder]

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "oof_realmlp_ote.npy"), oof_reordered.astype(np.float32))
    np.save(os.path.join(MODEL_DIR, "pred_realmlp_ote.npy"), pred_reordered.astype(np.float32))

    print(f"\nSaved oof_realmlp_ote.npy {oof_reordered.shape}")
    print(f"Saved pred_realmlp_ote.npy {pred_reordered.shape}")
    print("Label order: High=0, Low=1, Medium=2")
    print("\nDone.")
