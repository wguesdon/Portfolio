"""Feature engineering for PS6E4: Predicting Irrigation Need.

Builds a feature store inspired by the 1st place PS6E3 solution.
Uses feature-engine transformers where possible, custom transformers
for competition-specific techniques (snap, digit extraction).

All engineered DataFrames are saved to data/features/.

Usage:
    uv run python Playground_Series/PS6E4/scripts/02_feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

# feature-engine imports
from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
)
from feature_engine.creation import (
    MathFeatures,
    RelativeFeatures,
    CyclicalFeatures,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    DecisionTreeDiscretiser,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
FEAT_DIR = BASE_DIR / "data" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
SEED = 42

NUMERIC_COLS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

CAT_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

# Columns where original and synthetic have identical/near-identical unique counts
SNAP_PRIORITY_COLS = [
    "Soil_pH",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
]

# Top bigram pairs based on EDA
BIGRAM_PAIRS = [
    ("Crop_Growth_Stage", "Crop_Type"),
    ("Crop_Growth_Stage", "Irrigation_Type"),
    ("Irrigation_Type", "Season"),
    ("Crop_Type", "Season"),
    ("Mulching_Used", "Irrigation_Type"),
    ("Soil_Type", "Crop_Type"),
    ("Region", "Season"),
    ("Irrigation_Type", "Water_Source"),
]

TRIGRAM_TRIPLES = [
    ("Crop_Growth_Stage", "Irrigation_Type", "Season"),
    ("Soil_Type", "Crop_Type", "Season"),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "irrigation_prediction.csv")

y_train = train[TARGET].copy()
train_ids = train["id"].copy()
test_ids = test["id"].copy()
n_train = len(train)
n_test = len(test)

# Combine train+test for consistent feature engineering
all_data = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
print(f"Combined: {all_data.shape} (train={n_train}, test={n_test})")


# ===================================================================
# Custom transformers (not available in feature-engine)
# ===================================================================
class SnapTransformer(BaseEstimator, TransformerMixin):
    """Map synthetic values to nearest original dataset value.

    For each numeric column, finds the closest value in the original
    dataset. Returns snap_value and snap_diff (synthetic - snapped).
    Inspired by PS6E3 1st place Solution Section 2.1.
    """

    def __init__(self, original_df: pd.DataFrame, columns: list[str]):
        self.original_df = original_df
        self.columns = columns
        self.sorted_originals_: dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            if col in self.original_df.columns:
                self.sorted_originals_[col] = np.sort(
                    self.original_df[col].dropna().unique()
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for col, orig_vals in self.sorted_originals_.items():
            values = X[col].values
            idx = np.searchsorted(orig_vals, values, side="left")
            idx = np.clip(idx, 1, len(orig_vals) - 1)
            left = orig_vals[idx - 1]
            right = orig_vals[idx]
            snap = np.where(np.abs(values - left) <= np.abs(values - right), left, right)
            result[f"{col}_snap"] = snap
            result[f"{col}_snap_diff"] = values - snap
        return result


class DigitExtractor(BaseEstimator, TransformerMixin):
    """Extract decimal digit features from numeric columns.

    For each column, extracts:
    - d1: 1st decimal digit
    - d2: 2nd decimal digit
    - frac100: 2-digit fractional part as integer
    - mod10: integer part mod 10

    Inspired by PS6E3 1st place Solution Section 2.2.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for col in self.columns:
            values = X[col].values
            frac = values - np.floor(values)
            result[f"{col}_d1"] = np.floor(frac * 10).astype(np.int8)
            result[f"{col}_d2"] = (np.floor(frac * 100) % 10).astype(np.int8)
            result[f"{col}_frac100"] = np.round(frac * 100).astype(np.int16)
            result[f"{col}_mod10"] = (np.floor(values) % 10).astype(np.int8)
        return result


class KDTreeLookup(BaseEstimator, TransformerMixin):
    """Nearest neighbor lookup against the original dataset.

    For each row, finds K nearest neighbors in the original data
    and returns: distance, majority class, and per-class vote fractions.

    Inspired by PS6E3 1st place Solution Section 2.9.
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        columns: list[str],
        target_col: str,
        k: int = 5,
    ):
        self.original_df = original_df
        self.columns = columns
        self.target_col = target_col
        self.k = k

    def fit(self, X: pd.DataFrame, y=None):
        self.scaler_ = StandardScaler()
        orig_values = self.original_df[self.columns].values
        self.scaled_orig_ = self.scaler_.fit_transform(orig_values)
        self.tree_ = cKDTree(self.scaled_orig_)

        self.le_ = LabelEncoder()
        self.orig_labels_ = self.le_.fit_transform(self.original_df[self.target_col])
        self.n_classes_ = len(self.le_.classes_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler_.transform(X[self.columns].values)
        distances, indices = self.tree_.query(scaled, k=self.k)

        result = pd.DataFrame(index=X.index)
        result["nn_mean_dist"] = distances.mean(axis=1)
        result["nn_min_dist"] = distances.min(axis=1)

        neighbor_labels = self.orig_labels_[indices]
        for i, cls_name in enumerate(self.le_.classes_):
            result[f"nn_vote_{cls_name}"] = (neighbor_labels == i).mean(axis=1)

        majority = np.array(
            [np.bincount(row, minlength=self.n_classes_).argmax() for row in neighbor_labels]
        )
        result["nn_majority_class"] = majority
        return result


# ===================================================================
# Feature Group 1: Snap Features
# ===================================================================
print("\n[1/8] Snap features...")
snap = SnapTransformer(original_df=original, columns=NUMERIC_COLS)
snap.fit(all_data)
snap_df = snap.transform(all_data)
print(f"  Shape: {snap_df.shape}")

# ===================================================================
# Feature Group 2: Digit Extraction
# ===================================================================
print("[2/8] Digit extraction...")
digit = DigitExtractor(columns=NUMERIC_COLS)
digit_df = digit.transform(all_data)
print(f"  Shape: {digit_df.shape}")

# ===================================================================
# Feature Group 3: Domain Arithmetic Interactions
# ===================================================================
print("[3/8] Domain arithmetic interactions...")
domain_df = pd.DataFrame(index=all_data.index)

# Water balance
domain_df["water_balance"] = all_data["Rainfall_mm"] - all_data["Previous_Irrigation_mm"]
domain_df["water_per_hectare"] = (
    (all_data["Rainfall_mm"] + all_data["Previous_Irrigation_mm"])
    / (all_data["Field_Area_hectare"] + 0.01)
)
domain_df["total_water"] = all_data["Rainfall_mm"] + all_data["Previous_Irrigation_mm"]

# Weather stress
domain_df["heat_stress"] = all_data["Temperature_C"] * (1 - all_data["Humidity"] / 100)
domain_df["evapotranspiration"] = (
    all_data["Temperature_C"]
    * all_data["Sunlight_Hours"]
    / (all_data["Humidity"] + 1)
)
domain_df["wind_chill"] = (
    all_data["Wind_Speed_kmh"] * (100 - all_data["Humidity"]) / 100
)
domain_df["drying_index"] = (
    all_data["Sunlight_Hours"]
    * all_data["Wind_Speed_kmh"]
    / (all_data["Humidity"] + 1)
)
domain_df["temp_wind_interaction"] = (
    all_data["Temperature_C"] * all_data["Wind_Speed_kmh"]
)

# Soil quality
domain_df["soil_conductivity_ratio"] = all_data["Electrical_Conductivity"] / (
    all_data["Soil_pH"] + 1
)
domain_df["soil_moisture_deficit"] = 50 - all_data["Soil_Moisture"]
domain_df["organic_efficiency"] = all_data["Organic_Carbon"] / (
    all_data["Electrical_Conductivity"] + 0.01
)

# Field efficiency
domain_df["irrigation_density"] = all_data["Previous_Irrigation_mm"] / (
    all_data["Field_Area_hectare"] + 0.01
)
domain_df["rainfall_coverage"] = all_data["Rainfall_mm"] / (
    all_data["Field_Area_hectare"] + 0.01
)

# Moisture interactions (top ANOVA feature)
domain_df["moisture_temp_ratio"] = all_data["Soil_Moisture"] / (
    all_data["Temperature_C"] + 1
)
domain_df["moisture_wind_ratio"] = all_data["Soil_Moisture"] / (
    all_data["Wind_Speed_kmh"] + 1
)
domain_df["moisture_rainfall_ratio"] = all_data["Soil_Moisture"] / (
    all_data["Rainfall_mm"] + 1
)

print(f"  Shape: {domain_df.shape}")

# ===================================================================
# Feature Group 4: Frequency / Count Encoding (feature-engine)
# ===================================================================
print("[4/8] Frequency encoding (feature-engine)...")

freq_encoder = CountFrequencyEncoder(
    encoding_method="frequency", variables=CAT_COLS
)
freq_df = freq_encoder.fit_transform(all_data[CAT_COLS])
freq_df.columns = [f"freq_{c}" for c in CAT_COLS]

# Also frequency-encode snapped numeric columns
snap_cat_cols = [f"{c}_snap" for c in SNAP_PRIORITY_COLS]
snap_for_freq = snap_df[snap_cat_cols].copy()
for col in snap_cat_cols:
    freqs = snap_for_freq[col].value_counts(normalize=True)
    snap_for_freq[f"freq_{col}"] = snap_for_freq[col].map(freqs)
snap_freq_df = snap_for_freq[[f"freq_{c}" for c in snap_cat_cols]]

freq_all = pd.concat([freq_df, snap_freq_df], axis=1)
print(f"  Shape: {freq_all.shape}")

# ===================================================================
# Feature Group 5: Categorical Cross-Features (Bigrams / Trigrams)
# ===================================================================
print("[5/8] Categorical cross-features...")
cross_df = pd.DataFrame(index=all_data.index)

for col_a, col_b in BIGRAM_PAIRS:
    name = f"bi_{col_a}__{col_b}"
    cross_df[name] = all_data[col_a].astype(str) + "__" + all_data[col_b].astype(str)

for cols in TRIGRAM_TRIPLES:
    name = "tri_" + "__".join(cols)
    cross_df[name] = (
        all_data[cols[0]].astype(str)
        + "__"
        + all_data[cols[1]].astype(str)
        + "__"
        + all_data[cols[2]].astype(str)
    )

# Frequency-encode the cross-features
cross_freq_df = pd.DataFrame(index=all_data.index)
for col in cross_df.columns:
    freqs = cross_df[col].value_counts(normalize=True)
    cross_freq_df[f"freq_{col}"] = cross_df[col].map(freqs)

print(f"  Cross features: {cross_df.shape}")
print(f"  Cross frequencies: {cross_freq_df.shape}")

# ===================================================================
# Feature Group 6: Multi-Scale Binning (feature-engine)
# ===================================================================
print("[6/8] Multi-scale binning (feature-engine)...")
bin_target_cols = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Humidity",
    "Previous_Irrigation_mm",
]

bin_frames = []

# Equal-frequency binning at multiple scales
for n_bins in [50, 200]:
    eq_freq = EqualFrequencyDiscretiser(
        q=n_bins, variables=bin_target_cols, return_boundaries=True
    )
    binned = eq_freq.fit_transform(all_data[bin_target_cols])
    binned.columns = [f"{c}_eqf{n_bins}" for c in bin_target_cols]
    bin_frames.append(binned)

# Equal-width binning
for n_bins in [50, 200]:
    eq_width = EqualWidthDiscretiser(
        bins=n_bins, variables=bin_target_cols, return_boundaries=True
    )
    binned = eq_width.fit_transform(all_data[bin_target_cols])
    binned.columns = [f"{c}_eqw{n_bins}" for c in bin_target_cols]
    bin_frames.append(binned)

binning_df = pd.concat(bin_frames, axis=1)
print(f"  Shape: {binning_df.shape}")

# ===================================================================
# Feature Group 7: Boolean / Condition Aggregations
# ===================================================================
print("[7/8] Boolean aggregations...")
bool_df = pd.DataFrame(index=all_data.index)

bool_df["is_rainfed"] = (all_data["Irrigation_Type"] == "Rainfed").astype(np.int8)
bool_df["is_flowering"] = (all_data["Crop_Growth_Stage"] == "Flowering").astype(np.int8)
bool_df["is_harvest"] = (all_data["Crop_Growth_Stage"] == "Harvest").astype(np.int8)
bool_df["has_mulching"] = (all_data["Mulching_Used"] == "Yes").astype(np.int8)
bool_df["is_loamy"] = (all_data["Soil_Type"] == "Loamy").astype(np.int8)
bool_df["is_sandy"] = (all_data["Soil_Type"] == "Sandy").astype(np.int8)
bool_df["uses_groundwater"] = (all_data["Water_Source"] == "Groundwater").astype(np.int8)

# Stress condition flags
bool_df["low_moisture"] = (all_data["Soil_Moisture"] < 25).astype(np.int8)
bool_df["high_temp"] = (all_data["Temperature_C"] > 35).astype(np.int8)
bool_df["high_wind"] = (all_data["Wind_Speed_kmh"] > 15).astype(np.int8)
bool_df["low_rainfall"] = (all_data["Rainfall_mm"] < 500).astype(np.int8)
bool_df["stress_count"] = (
    bool_df["low_moisture"]
    + bool_df["high_temp"]
    + bool_df["high_wind"]
    + bool_df["low_rainfall"]
)

print(f"  Shape: {bool_df.shape}")

# ===================================================================
# Feature Group 8: KDTree Original Data Lookup
# ===================================================================
print("[8/8] KDTree original data lookup...")
common_num = [c for c in NUMERIC_COLS if c in original.columns]
kdtree = KDTreeLookup(
    original_df=original,
    columns=common_num,
    target_col=TARGET,
    k=5,
)
kdtree.fit(all_data)
kdtree_df = kdtree.transform(all_data)
print(f"  Shape: {kdtree_df.shape}")

# ===================================================================
# Original data priors (zero-leakage target stats from original 10K)
# ===================================================================
print("\nComputing original data priors...")
prior_df = pd.DataFrame(index=all_data.index)
orig_cat_cols = [c for c in CAT_COLS if c in original.columns]

for col in orig_cat_cols:
    ct = pd.crosstab(original[col], original[TARGET], normalize="index")
    for cls in TARGET_ORDER:
        if cls in ct.columns:
            mapping = ct[cls].to_dict()
            prior_df[f"orig_prior_{col}_{cls}"] = all_data[col].map(mapping)

print(f"  Shape: {prior_df.shape}")

# ===================================================================
# Save all feature groups
# ===================================================================
print("\nSaving feature groups...")

feature_groups = {
    "snap": snap_df,
    "digits": digit_df,
    "domain": domain_df,
    "frequency": freq_all,
    "cross_features": cross_df,
    "cross_freq": cross_freq_df,
    "binning": binning_df,
    "boolean": bool_df,
    "kdtree": kdtree_df,
    "orig_priors": prior_df,
}

for name, df in feature_groups.items():
    path = FEAT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  {name}: {df.shape} -> {path.name}")

# Save raw numeric and categorical for convenience
all_data[NUMERIC_COLS].to_parquet(FEAT_DIR / "raw_numeric.parquet", index=False)

# Ordinal-encode categoricals for tree models
le_dict = {}
cat_encoded = pd.DataFrame(index=all_data.index)
for col in CAT_COLS:
    le = LabelEncoder()
    cat_encoded[col] = le.fit_transform(all_data[col])
    le_dict[col] = le
cat_encoded.to_parquet(FEAT_DIR / "raw_categorical_encoded.parquet", index=False)

# Save cross-features ordinal-encoded for tree models
cross_encoded = pd.DataFrame(index=all_data.index)
for col in cross_df.columns:
    le = LabelEncoder()
    cross_encoded[col] = le.fit_transform(cross_df[col])
cross_encoded.to_parquet(FEAT_DIR / "cross_features_encoded.parquet", index=False)

# Save target and IDs
pd.DataFrame({"id": train_ids, TARGET: y_train}).to_parquet(
    FEAT_DIR / "target.parquet", index=False
)
pd.DataFrame({"id": test_ids}).to_parquet(FEAT_DIR / "test_ids.parquet", index=False)

# Save metadata
meta = {
    "n_train": n_train,
    "n_test": n_test,
    "numeric_cols": NUMERIC_COLS,
    "cat_cols": CAT_COLS,
    "snap_priority_cols": SNAP_PRIORITY_COLS,
    "bigram_pairs": BIGRAM_PAIRS,
    "trigram_triples": TRIGRAM_TRIPLES,
    "target_order": TARGET_ORDER,
    "feature_groups": list(feature_groups.keys()),
}
import json

with open(FEAT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# ===================================================================
# Summary
# ===================================================================
total_features = sum(df.shape[1] for df in feature_groups.values())
print(f"\nDone. Total engineered features: {total_features}")
print(f"Feature store saved to: {FEAT_DIR}")
print("\nFeature group summary:")
for name, df in feature_groups.items():
    print(f"  {name:20s}: {df.shape[1]:4d} columns")
