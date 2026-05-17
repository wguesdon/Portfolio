#!/usr/bin/env python
"""SageMaker entry point: XGBoost + Ordered Target Encoding for PS6E4.

Based on emanuellcs/predicting-irrigation-need-xgboost-ote (LB 0.98011).
Adapted to use StratifiedKFold(seed=42) matching our ensemble fold strategy.
Saves oof_xgb_ote.npy and pred_xgb_ote.npy with our label encoding (High=0, Low=1, Medium=2).
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
from xgboost import XGBClassifier

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
        """Fit on shuffled training data using cumulative leave-one-out encoding."""
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
        """Apply fitted encoding to validation/test data."""
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
    """Digit extraction + adaptive rounding + frequency encoding."""
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

    # Drop constant columns
    drop = [c for c in test_df.columns if test_df[c].nunique() == 1]
    if drop:
        print(f"Dropping {len(drop)} constant columns")
        train_df.drop(drop, axis=1, inplace=True)
        test_df.drop(drop, axis=1, inplace=True)

    # Frequency-based ordinal encoding for categoricals and digit features
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
    print("=" * 60)
    print("XGBoost + Ordered Target Encoding")
    print("=" * 60)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    print(f"Train: {train.shape}, Test: {test.shape}")

    # Our label encoding: High=0, Low=1, Medium=2 (alphabetical via LabelEncoder)
    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])

    # For internal training, use integer encoding matching the notebook
    # (Low=0, Medium=1, High=2 based on unique() order)
    # We'll remap at the end
    target2idx = {v: i for i, v in enumerate(train[TARGET_COL].unique())}
    idx2target = {v: k for k, v in target2idx.items()}
    print(f"Internal label map: {target2idx}")

    train[TARGET_COL] = train[TARGET_COL].map(target2idx)

    cats = [c for c in test.columns if train[c].dtype == object and c != "id"]
    nums = [c for c in test.columns if c not in cats and c != "id"]

    train.drop("id", axis=1, inplace=True)
    test_ids = test["id"].copy()
    test.drop("id", axis=1, inplace=True)

    print(f"Cats: {len(cats)}, Nums: {len(nums)}")

    # Feature engineering
    train, test, features = feature_engineering(train, test, nums, cats)
    print(f"Features after FE: {len(features)}, Train cols: {train.shape[1]}")

    # Compute sample weights
    unique_classes, class_sample_counts = np.unique(train[TARGET_COL].values, return_counts=True)
    count_dict = dict(zip(unique_classes, class_sample_counts))
    avg_count = len(train) / len(unique_classes)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[TARGET_COL]])

    # Prepare data
    X = train.drop([TARGET_COL], axis=1)
    y = train[TARGET_COL]
    test_X = test.copy()

    oof_preds = np.zeros((len(y), NUM_CLASSES))
    test_preds = np.zeros((len(test_X), NUM_CLASSES))

    # XGBoost params (from the notebook, adapted for CPU)
    xgb_params = {
        "max_depth": 4,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "n_estimators": 512,
        "learning_rate": 0.1,
        "early_stopping_rounds": 1024,
        "random_state": 2026,
        "n_jobs": -1,
        "enable_categorical": True,
        "alpha": 5,
        "reg_lambda": 5,
        "max_leaves": 30,
        "min_child_weight": 2,
        "tree_method": "hist",
        "max_bin": 10000,
        "device": "cpu",
    }

    # StratifiedKFold to match our ensemble
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold} / {N_FOLDS}")
        print(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
        train_w = sample_weights[train_idx]

        # Ordered Target Encoding: 4x shuffle
        te = OrderedTE()
        full_df = pd.concat((X_train, y_train), axis=1)
        full_df["weight"] = train_w

        te_train = pd.concat([
            te.fit(
                full_df.sample(frac=1, random_state=42 + i),
                category_cols=features,
                target_col=TARGET_COL,
            )
            for i in range(4)
        ])

        X_train = te_train.drop([TARGET_COL, "weight"], axis=1)
        y_train = te_train[TARGET_COL]
        train_w = te_train["weight"]

        del full_df, te_train
        gc.collect()

        X_val_te = te.transform(X_val)
        X_test_te = te.transform(test_X.copy())

        # Drop original categoricals (replaced by TE features)
        X_train.drop(cats, axis=1, inplace=True)
        X_val_te.drop(cats, axis=1, inplace=True)
        X_test_te.drop(cats, axis=1, inplace=True)

        # Train XGBoost
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val_te, y_val)],
            sample_weight=train_w,
            verbose=100,
        )

        y_pred = model.predict_proba(X_val_te)
        oof_preds[val_idx] = y_pred

        test_preds += model.predict_proba(X_test_te) / N_FOLDS

        fold_score = balanced_accuracy_score(y_val, np.argmax(y_pred, axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold} balanced accuracy: {fold_score:.5f}")

        del X_train, X_val_te, X_test_te, model
        gc.collect()

    oof_cv = balanced_accuracy_score(y.values, np.argmax(oof_preds, axis=1))
    print(f"\n{'='*60}")
    print(f"  Overall OOF CV: {oof_cv:.5f}")
    print(f"  Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"{'='*60}")

    # Remap to our label encoding: High=0, Low=1, Medium=2
    # Internal encoding depends on train[TARGET_COL].unique() order
    # Build mapping: internal_idx -> label_string -> our_idx
    # internal: target2idx = {label: idx}
    # ours: le.transform([label]) -> our_idx
    # OOF/pred columns are [internal_0_prob, internal_1_prob, internal_2_prob]
    # Need to reorder columns so column i = our label i's probability

    internal_to_label = idx2target  # {0: "Low", 1: "Medium", 2: "High"} (or similar)
    reorder = []
    our_labels = le.classes_  # ["High", "Low", "Medium"]
    for our_idx, label in enumerate(our_labels):
        internal_idx = target2idx[label]
        reorder.append(internal_idx)

    print(f"Internal label map: {target2idx}")
    print(f"Our label order: {list(our_labels)}")
    print(f"Reorder columns: {reorder}")

    oof_reordered = oof_preds[:, reorder]
    pred_reordered = test_preds[:, reorder]

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.save(os.path.join(MODEL_DIR, "oof_xgb_ote.npy"), oof_reordered.astype(np.float32))
    np.save(os.path.join(MODEL_DIR, "pred_xgb_ote.npy"), pred_reordered.astype(np.float32))

    print(f"\nSaved oof_xgb_ote.npy {oof_reordered.shape}")
    print(f"Saved pred_xgb_ote.npy {pred_reordered.shape}")
    print("Label order: High=0, Low=1, Medium=2")
    print("\nDone.")
