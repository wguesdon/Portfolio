"""
CIBMTR — Feature Engineering

Loads and preprocesses train/test data for model training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    TRAIN_FILE, TEST_FILE, CATEGORICAL_COLS, NUMERICAL_COLS,
    EFS_COL, TIME_COL, GROUP_COL, ID_COL,
)


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_categoricals(df_train: pd.DataFrame, df_test: pd.DataFrame | None = None):
    """
    Label-encode categorical columns in place.
    Combines train+test to handle all categories.
    Returns (df_train, df_test, encoders).
    """
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df_train.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([
            df_train[col],
            df_test[col] if df_test is not None else pd.Series(dtype=object),
        ]).fillna("__missing__").astype(str)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col].fillna("__missing__").astype(str))
        if df_test is not None and col in df_test.columns:
            df_test[col] = le.transform(df_test[col].fillna("__missing__").astype(str))
        encoders[col] = le
    return df_train, df_test, encoders


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""
    # HLA composite score
    hla_cols = [c for c in NUMERICAL_COLS if c.startswith("hla_")]
    df["hla_total_match"] = df[hla_cols].sum(axis=1)
    df["hla_missing_count"] = df[hla_cols].isna().sum(axis=1)

    # Donor/recipient age gap
    if "donor_age" in df.columns and "age_at_hct" in df.columns:
        df["age_gap"] = df["donor_age"] - df["age_at_hct"]

    return df


def load_and_prepare(include_test=False):
    """
    Load and preprocess training (and optionally test) data.

    Returns
    -------
    df_train : pd.DataFrame  — full training set with encoded features
    df_test  : pd.DataFrame or None
    feature_cols : list[str]
    """
    df_train = load_raw(TRAIN_FILE)
    df_test  = load_raw(TEST_FILE) if include_test else None

    # Binary encode target
    df_train[EFS_COL] = (df_train[EFS_COL] == 1).astype(int)

    # Add derived features
    df_train = add_derived_features(df_train)
    if df_test is not None:
        df_test = add_derived_features(df_test)

    # Encode categoricals
    df_train, df_test, _ = encode_categoricals(df_train, df_test)

    # Feature columns = all columns except targets/id/group
    exclude = {ID_COL, EFS_COL, TIME_COL}
    feature_cols = [c for c in df_train.columns if c not in exclude]

    return df_train, df_test, feature_cols


def get_Xy(df_train: pd.DataFrame, feature_cols: list):
    X = df_train[feature_cols]
    y_time = np.log(df_train[TIME_COL].clip(lower=0.01))
    y_efs  = df_train[EFS_COL].astype(int)
    return X, y_time, y_efs
