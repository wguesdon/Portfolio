"""TabPFN + OTE for PS6E4 on Kaggle GPU.

TabPFN v2 is a pre-trained transformer for tabular data. Context size is limited
(~10K rows for v2). We apply it by subsampling per fold: draw several stratified
subsamples of the fold's training data, fit TabPFN on each, average the val/test
predictions.

Outputs: oof_tabpfn.npy and pred_tabpfn.npy with label order High=0, Low=1, Medium=2.
"""
import os
import gc
import subprocess
import sys

# Install TabPFN (not pre-installed on Kaggle)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "tabpfn"],
    stdout=subprocess.DEVNULL,
)

# TabPFN license token via Kaggle Secrets.
# Setup (one-time, in Kaggle notebook UI):
#   Add-ons menu (or the key icon in the left sidebar) -> Secrets -> Add new secret
#   Label: TABPFN_TOKEN
#   Value: <your token from https://ux.priorlabs.ai/account>
#   Toggle "Attach to this notebook"
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier

INPUT = "/kaggle/input/competitions/playground-series-s6e4"
OUT = "/kaggle/working"

SEED = 42
N_FOLDS = 5
NUM_CLASSES = 3
TARGET_COL = "Irrigation_Need"

# TabPFN v2 context capacity: use ~10K per subsample, average over N_SUBSAMPLES
SUBSAMPLE_SIZE = 10000
N_SUBSAMPLES = 5


class OrderedTE:
    def __init__(self, a=1.0):
        self.a = a

    def fit(self, train, category_cols=None, target_col="target"):
        self.train = train
        self.target_col = target_col
        self.category_cols = category_cols or []
        self.classes_ = sorted(train[target_col].unique())
        self.num_classes_ = len(self.classes_)
        self.global_prior_ = train[target_col].value_counts(normalize=True).sort_index().values
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
                    test[te_col] = (test[sum_col] + self.a * test[prior_col]) / (test[count_col] + self.a)
                    test[te_col] = test[te_col].fillna(test[prior_col])
                    test.drop([count_col, sum_col, prior_col], axis=1, inplace=True)
                else:
                    test[te_col] = self.global_prior_[k]
            gc.collect()
        return test


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
        train_df.drop(drop, axis=1, inplace=True)
        test_df.drop(drop, axis=1, inplace=True)
    category_cols = cats + [c for c in test_df.columns if "digit" in c]
    for c in category_cols:
        freq = train_df[c].value_counts()
        mapping = {val: idx for idx, (val, _) in enumerate(freq[freq >= 5].items())}
        default = len(mapping)
        train_df[c] = train_df[c].map(lambda x, m=mapping, d=default: m.get(x, d))
        test_df[c] = test_df[c].map(lambda x, m=mapping, d=default: m.get(x, d))
    features = category_cols + nums
    return train_df, test_df, features


print("TabPFN + OTE (Kaggle GPU)")

train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

le = LabelEncoder()
le.fit(["High", "Low", "Medium"])

target2idx = {v: i for i, v in enumerate(train[TARGET_COL].unique())}
train[TARGET_COL] = train[TARGET_COL].map(target2idx)

nums = [c for c in test.columns if pd.api.types.is_numeric_dtype(train[c]) and c != "id"]
cats = [c for c in test.columns if c not in nums and c != "id"]

train.drop("id", axis=1, inplace=True)
test_ids = test["id"].copy()
test.drop("id", axis=1, inplace=True)

train, test, features = feature_engineering(train, test, nums, cats)
print(f"Features: {len(features)}")

X = train.drop([TARGET_COL], axis=1)
y = train[TARGET_COL]
test_X = test.copy()

oof_preds = np.zeros((len(y), NUM_CLASSES))
test_preds = np.zeros((len(test_X), NUM_CLASSES))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFOLD {fold}/{N_FOLDS}")
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # OTE with 4x shuffle (use only last fit; no 4x concat - TabPFN can't use all data anyway)
    te = OrderedTE()
    full_df = pd.concat((X_train, y_train), axis=1)
    for i in range(4):
        te.fit(
            full_df.sample(frac=1, random_state=42 + i),
            category_cols=features,
            target_col=TARGET_COL,
        )

    X_train_te = te.transform(X_train.copy())
    X_val_te = te.transform(X_val)
    X_test_te = te.transform(test_X.copy())

    del full_df
    gc.collect()

    X_train_te.drop(cats, axis=1, inplace=True)
    X_val_te.drop(cats, axis=1, inplace=True)
    X_test_te.drop(cats, axis=1, inplace=True)

    X_train_te = X_train_te.fillna(0)
    X_val_te = X_val_te.fillna(0)
    X_test_te = X_test_te.fillna(0)

    # Scale features (helps TabPFN generalize)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_te).astype(np.float32)
    X_val_np = scaler.transform(X_val_te).astype(np.float32)
    X_test_np = scaler.transform(X_test_te).astype(np.float32)
    y_train_np = y_train.values.astype(np.int32)

    # TabPFN: subsample + ensemble
    val_probs = np.zeros((len(X_val_np), NUM_CLASSES))
    test_probs = np.zeros((len(X_test_np), NUM_CLASSES))

    rng = np.random.default_rng(SEED + fold)
    n_train = len(X_train_np)

    for s in range(N_SUBSAMPLES):
        # Stratified sample of SUBSAMPLE_SIZE
        sub_idx = []
        for cls in range(NUM_CLASSES):
            cls_idx = np.where(y_train_np == cls)[0]
            n_cls = max(1, SUBSAMPLE_SIZE * len(cls_idx) // n_train)
            picked = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            sub_idx.extend(picked)
        sub_idx = np.array(sub_idx)

        Xs = X_train_np[sub_idx]
        ys = y_train_np[sub_idx]

        model = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True)
        model.fit(Xs, ys)

        # Batch predict_proba to fit in T4 16GB VRAM (~10K rows per batch safe)
        BATCH = 5000
        def batched_predict_proba(X):
            out = np.zeros((len(X), NUM_CLASSES), dtype=np.float32)
            for start in range(0, len(X), BATCH):
                end = min(start + BATCH, len(X))
                out[start:end] = model.predict_proba(X[start:end])
            return out

        val_probs += batched_predict_proba(X_val_np) / N_SUBSAMPLES
        test_probs += batched_predict_proba(X_test_np) / N_SUBSAMPLES

        del model
        gc.collect()
        print(f"  subsample {s+1}/{N_SUBSAMPLES} done")

    oof_preds[val_idx] = val_probs
    test_preds += test_probs / N_FOLDS

    fold_score = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
    fold_scores.append(fold_score)
    print(f"  Fold {fold} balanced accuracy: {fold_score:.5f}")

    del X_train_te, X_val_te, X_test_te
    gc.collect()

oof_cv = balanced_accuracy_score(y.values, np.argmax(oof_preds, axis=1))
print(f"\nOverall OOF CV: {oof_cv:.5f}")
print(f"Fold scores: {fold_scores}")

reorder = [target2idx[label] for label in le.classes_]
oof_reordered = oof_preds[:, reorder]
pred_reordered = test_preds[:, reorder]

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "oof_tabpfn.npy"), oof_reordered.astype(np.float32))
np.save(os.path.join(OUT, "pred_tabpfn.npy"), pred_reordered.astype(np.float32))
submission = pd.DataFrame({"id": test_ids, TARGET_COL: le.inverse_transform(pred_reordered.argmax(axis=1))})
submission.to_csv(os.path.join(OUT, "submission_tabpfn.csv"), index=False)
print(f"Saved to {OUT}. Label order: High=0, Low=1, Medium=2")
