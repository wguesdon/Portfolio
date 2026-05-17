"""Additional feature engineering: projections, radix, artifact detection.

Complements 02_feature_engineering.py with techniques from PS6E3 Solution 1
Sections 2.10, 2.11, 2.12 plus UMAP for nonlinear manifold features.

Usage:
    uv run python Playground_Series/PS6E4/scripts/02b_features_projections.py
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from umap import UMAP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
FEAT_DIR = BASE_DIR / "data" / "features"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
SEED = 42

with open(FEAT_DIR / "metadata.json") as f:
    meta = json.load(f)

NUMERIC_COLS = meta["numeric_cols"]
CAT_COLS = meta["cat_cols"]
N_TRAIN = meta["n_train"]
N_TEST = meta["n_test"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "irrigation_prediction.csv")

all_data = pd.concat(
    [train.drop(columns=[TARGET]), test], ignore_index=True
)
print(f"Combined: {all_data.shape}")

common_num = [c for c in NUMERIC_COLS if c in original.columns]

# ===================================================================
# Feature Group 9: PCA + Gaussian Random Projection (Solution 1 Section 2.12)
# Fit on original 10K data, project synthetic rows.
# ===================================================================
print("\n[1/5] PCA + Gaussian Random Projection...")

scaler = StandardScaler()
orig_scaled = scaler.fit_transform(original[common_num])
all_scaled = scaler.transform(all_data[common_num])

n_pca = min(8, len(common_num))
pca = PCA(n_components=n_pca, random_state=SEED)
pca.fit(orig_scaled)
pca_features = pca.transform(all_scaled)
pca_df = pd.DataFrame(
    pca_features,
    columns=[f"pca_{i}" for i in range(n_pca)],
)
print(f"  PCA: {pca_df.shape}, explained variance: {pca.explained_variance_ratio_.sum():.3f}")

n_grp = min(8, len(common_num))
grp = GaussianRandomProjection(n_components=n_grp, random_state=SEED)
grp.fit(orig_scaled)
grp_features = grp.transform(all_scaled)
grp_df = pd.DataFrame(
    grp_features,
    columns=[f"grp_{i}" for i in range(n_grp)],
)
print(f"  GRP: {grp_df.shape}")

projection_df = pd.concat([pca_df, grp_df], axis=1)

# ===================================================================
# Feature Group 10: UMAP (nonlinear manifold projection)
# Fit on original 10K data, project synthetic rows.
# ===================================================================
print("\n[2/5] UMAP projections...")

umap_components = []
for n_comp, n_neighbors, min_dist in [(3, 15, 0.1), (3, 50, 0.5)]:
    tag = f"umap_n{n_neighbors}_d{min_dist}"
    print(f"  Fitting {tag}...")
    mapper = UMAP(
        n_components=n_comp,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=SEED,
    )
    mapper.fit(orig_scaled)
    umap_features = mapper.transform(all_scaled)
    umap_part = pd.DataFrame(
        umap_features,
        columns=[f"{tag}_{i}" for i in range(n_comp)],
    )
    umap_components.append(umap_part)

umap_df = pd.concat(umap_components, axis=1)
print(f"  UMAP total: {umap_df.shape}")

# ===================================================================
# Feature Group 11: Radix Interaction Features (Solution 1 Section 2.10)
# Encode (snapped numeric, categorical) pair as a single integer.
# ===================================================================
print("\n[3/5] Radix interaction features...")

snap_df = pd.read_parquet(FEAT_DIR / "snap.parquet")

# Use the top numeric cols (by ANOVA) and top categorical cols (by EDA)
radix_num_cols = [
    "Soil_Moisture_snap",
    "Temperature_C_snap",
    "Wind_Speed_kmh_snap",
    "Rainfall_mm_snap",
]
radix_cat_cols = [
    "Crop_Growth_Stage",
    "Irrigation_Type",
    "Crop_Type",
    "Season",
]

le_cats = {}
for col in radix_cat_cols:
    le = LabelEncoder()
    le_cats[col] = le.fit_transform(all_data[col])

radix_df = pd.DataFrame(index=all_data.index)
for num_col in radix_num_cols:
    snap_int = (snap_df[num_col].values * 100).astype(np.int64)
    for cat_col in radix_cat_cols:
        cat_code = le_cats[cat_col]
        radix_name = f"radix_{num_col}__{cat_col}"
        radix_df[radix_name] = snap_int + cat_code * 10_000_000

print(f"  Shape: {radix_df.shape}")

# ===================================================================
# Feature Group 12: Synthetic Artifact Detection (Solution 1 Section 2.11)
# ===================================================================
print("\n[4/5] Synthetic artifact detection...")

artifact_df = pd.DataFrame(index=all_data.index)

# Fractional fingerprints per row
for col in NUMERIC_COLS:
    vals = all_data[col].values
    frac = vals - np.floor(vals)
    artifact_df[f"{col}_is_round"] = (np.abs(frac) < 0.005).astype(np.int8)
    artifact_df[f"{col}_is_half"] = (np.abs(frac - 0.5) < 0.005).astype(np.int8)
    artifact_df[f"{col}_is_quarter"] = (
        (np.abs(frac - 0.25) < 0.005) | (np.abs(frac - 0.75) < 0.005)
    ).astype(np.int8)

# Row-level counts
round_cols = [c for c in artifact_df.columns if c.endswith("_is_round")]
half_cols = [c for c in artifact_df.columns if c.endswith("_is_half")]
quarter_cols = [c for c in artifact_df.columns if c.endswith("_is_quarter")]
artifact_df["round_count"] = artifact_df[round_cols].sum(axis=1).astype(np.int8)
artifact_df["half_count"] = artifact_df[half_cols].sum(axis=1).astype(np.int8)
artifact_df["quarter_count"] = artifact_df[quarter_cols].sum(axis=1).astype(np.int8)

# Drift ratios: log1p(synthetic_freq / original_freq) per numeric column
# Measures how much the generator over/under-sampled each value
drift_cols_to_use = [
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Wind_Speed_kmh",
    "Humidity",
    "Previous_Irrigation_mm",
]
for col in drift_cols_to_use:
    if col not in original.columns:
        continue
    # Bin to 100 bins for stable frequency estimates
    bins = np.linspace(
        min(original[col].min(), all_data[col].min()),
        max(original[col].max(), all_data[col].max()),
        101,
    )
    orig_binned = pd.cut(original[col], bins=bins, labels=False, include_lowest=True)
    syn_binned = pd.cut(all_data[col], bins=bins, labels=False, include_lowest=True)

    orig_freq = orig_binned.value_counts(normalize=True)
    syn_freq = syn_binned.value_counts(normalize=True)

    ratio = syn_freq / (orig_freq + 1e-8)
    drift_map = ratio.to_dict()
    artifact_df[f"drift_{col}"] = syn_binned.map(drift_map).fillna(1.0).values
    artifact_df[f"drift_{col}"] = np.log1p(artifact_df[f"drift_{col}"])

print(f"  Shape: {artifact_df.shape}")

# ===================================================================
# Feature Group 13: KNN Target Encoding within CV (Solution 1 Level 1)
# Precompute KNN distances at multiple K values for model diversity
# ===================================================================
print("\n[5/5] Multi-K nearest neighbor features...")

from scipy.spatial import cKDTree

orig_scaler = StandardScaler()
orig_scaled_knn = orig_scaler.fit_transform(original[common_num])
all_scaled_knn = orig_scaler.transform(all_data[common_num])
tree = cKDTree(orig_scaled_knn)

le_target = LabelEncoder()
orig_labels = le_target.fit_transform(original[TARGET])
n_classes = len(le_target.classes_)

knn_df = pd.DataFrame(index=all_data.index)
for k in [1, 3, 10, 25]:
    print(f"  K={k}...")
    distances, indices = tree.query(all_scaled_knn, k=k)
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    knn_df[f"knn_{k}_mean_dist"] = distances.mean(axis=1)
    knn_df[f"knn_{k}_std_dist"] = distances.std(axis=1) if k > 1 else 0.0

    neighbor_labels = orig_labels[indices]
    for i, cls_name in enumerate(le_target.classes_):
        knn_df[f"knn_{k}_vote_{cls_name}"] = (neighbor_labels == i).mean(axis=1)

print(f"  Shape: {knn_df.shape}")

# ===================================================================
# Save
# ===================================================================
print("\nSaving...")

new_groups = {
    "projections": projection_df,
    "umap": umap_df,
    "radix": radix_df,
    "artifacts": artifact_df,
    "knn_multi": knn_df,
}

for name, df in new_groups.items():
    path = FEAT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  {name}: {df.shape} -> {path.name}")

# Update metadata
with open(FEAT_DIR / "metadata.json") as f:
    meta = json.load(f)
meta["feature_groups"].extend(list(new_groups.keys()))
with open(FEAT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

total_new = sum(df.shape[1] for df in new_groups.values())
print(f"\nDone. New features: {total_new}")
print(f"  projections: {projection_df.shape[1]} (PCA 8 + GRP 8)")
print(f"  umap:        {umap_df.shape[1]} (2 configs x 3 components)")
print(f"  radix:       {radix_df.shape[1]} (4 numeric x 4 categorical)")
print(f"  artifacts:   {artifact_df.shape[1]} (rounding flags + drift ratios)")
print(f"  knn_multi:   {knn_df.shape[1]} (K=1,3,10,25 x distances + class votes)")
