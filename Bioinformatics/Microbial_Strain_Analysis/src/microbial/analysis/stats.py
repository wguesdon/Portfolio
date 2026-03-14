"""Summary statistics and EDA computations."""

import pandas as pd
import numpy as np
from scipy import stats


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query("missing_count > 0")


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").describe().T.round(3)


def category_counts(df: pd.DataFrame, col: str, top_n: int | None = None) -> pd.DataFrame:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    if top_n:
        counts = counts.head(top_n)
    return counts


def phylum_summary(strains: pd.DataFrame) -> pd.DataFrame:
    return category_counts(strains, "phylum")


def class_summary(strains: pd.DataFrame) -> pd.DataFrame:
    return category_counts(strains, "class")


def o2_summary(strains: pd.DataFrame) -> pd.DataFrame:
    return category_counts(strains, "o2_tol")


def temp_range_summary(strains: pd.DataFrame) -> pd.DataFrame:
    strains = strains.copy()
    strains["temp_range"] = strains["temp_max"] - strains["temp_min"]
    return strains.groupby("o2_tol")[["temp_opt", "temp_range"]].describe().round(2)


def ph_range_summary(strains: pd.DataFrame) -> pd.DataFrame:
    strains = strains.copy()
    strains["ph_range"] = strains["ph_max"] - strains["ph_min"]
    return strains.groupby("o2_tol")[["ph_opt", "ph_range"]].describe().round(2)


def correlation_matrix(strains: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["temp_min", "temp_max", "temp_opt", "temp_range",
                "ph_min", "ph_max", "ph_opt", "ph_range"]
    df = strains.copy()
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["ph_range"] = df["ph_max"] - df["ph_min"]
    available = [c for c in num_cols if c in df.columns]
    return df[available].corr().round(3)


def top_compounds(compound_freq: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return compound_freq.head(n)


def strains_per_phylum_o2(strains_with_media: pd.DataFrame) -> pd.DataFrame:
    return (
        strains_with_media.drop_duplicates("strain_id")
        .groupby(["phylum", "o2_tol"])["strain_id"]
        .count()
        .reset_index(name="count")
    )


def media_per_strain(strain_media: pd.DataFrame) -> pd.DataFrame:
    return (
        strain_media.groupby("strain_id")["medium_id"]
        .nunique()
        .reset_index(name="media_count")
    )
