"""
Feature engineering for Obesity Risk Prediction (PS4E2).

Matching the winning notebook (rank 30/3587): each model handles its own
preprocessing internally via sklearn pipelines. This module provides the
canonical feature column list and a no-op transform so base_trainer stays
generic across projects.
"""
import pandas as pd

# Raw feature columns after dropping 'id' and target
RAW_FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history_with_overweight",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df unchanged — preprocessing is done per-model in each pipeline."""
    return df


def get_feature_columns() -> list[str]:
    """Return raw feature columns (no id, no target)."""
    return RAW_FEATURE_COLUMNS
