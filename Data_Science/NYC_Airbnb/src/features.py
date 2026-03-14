"""
Data loading, cleaning, and feature engineering for NYC Airbnb price prediction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATA_FILE, MANHATTAN_CENTER_LAT, MANHATTAN_CENTER_LON


# ── Loading & Cleaning ────────────────────────────────────────────────────────

def load_and_clean(path=DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["id", "host_name", "last_review", "name"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = df[df["price"] > 0].reset_index(drop=True)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2: float, lon2: float) -> pd.Series:
    """Vectorised Haversine distance in kilometres."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return pd.Series(R * 2 * np.arcsin(np.sqrt(a)), index=lat1.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform right-skewed counts
    skewed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    for col in skewed:
        df[f"log_{col}"] = np.log1p(df[col])

    # Geographic: distance from Times Square
    df["dist_manhattan_km"] = _haversine_km(
        df["latitude"], df["longitude"],
        MANHATTAN_CENTER_LAT, MANHATTAN_CENTER_LON,
    )
    df["log_dist_manhattan"] = np.log1p(df["dist_manhattan_km"])

    # Ordinal encodings
    room_map = {"Entire home/apt": 2, "Private room": 1, "Shared room": 0}
    df["room_type_enc"] = df["room_type"].map(room_map)

    borough_map = {
        "Manhattan": 4, "Brooklyn": 3, "Queens": 2,
        "Staten Island": 1, "Bronx": 0,
    }
    df["borough_enc"] = df["neighbourhood_group"].map(borough_map)

    # Interaction: borough × room type
    df["borough_x_room"] = df["borough_enc"] * df["room_type_enc"]

    # Popularity proxy: reviews relative to host's total listings
    df["reviews_per_listing"] = (
        df["number_of_reviews"] / (df["calculated_host_listings_count"] + 1)
    )
    df["log_reviews_per_listing"] = np.log1p(df["reviews_per_listing"])

    # Availability rate (0–1)
    df["availability_rate"] = df["availability_365"] / 365.0

    # Mean target encoding for high-cardinality neighbourhood column.
    # Encoding is computed on the full dataset — acceptable for EDA-style
    # portfolio work; production code would embed this inside each CV fold.
    log_price = np.log1p(df["price"])
    global_mean = log_price.mean()
    for col in ["neighbourhood", "neighbourhood_group", "room_type"]:
        means = df.groupby(col)["price"].transform(lambda x: np.log1p(x).mean())
        df[f"te_{col}"] = means.fillna(global_mean)

    # Log-price target
    df["log_price"] = np.log1p(df["price"])

    return df


FEATURE_COLS = [
    # Geographic
    "latitude", "longitude", "log_dist_manhattan",
    # Ordinal-encoded categoricals
    "room_type_enc", "borough_enc", "borough_x_room",
    # Target-encoded high-cardinality columns
    "te_neighbourhood", "te_neighbourhood_group", "te_room_type",
    # Log-transformed numerics
    "log_minimum_nights", "log_number_of_reviews", "log_reviews_per_month",
    "log_calculated_host_listings_count", "log_availability_365",
    # Derived
    "log_reviews_per_listing", "availability_rate",
]


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y, feature_names) ready for modelling."""
    df = add_features(df)
    X = df[FEATURE_COLS].copy()
    y = df["log_price"].copy()
    return X, y, FEATURE_COLS
