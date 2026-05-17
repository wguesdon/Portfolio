"""EDA for PS6E4: Predicting Irrigation Need.

Generates plots and summary statistics for the competition dataset.
All figures are saved to output/eda/.

Usage:
    uv run python Playground_Series/PS6E4/eda/01_eda.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
TARGET = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
TARGET_PALETTE = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
original = pd.read_csv(DATA_DIR / "irrigation_prediction.csv")

NUMERIC_COLS = train.select_dtypes(include="number").columns.drop("id").tolist()
CAT_COLS = [
    c
    for c in train.select_dtypes(include=["object", "string"]).columns
    if c != TARGET
]

print(f"Train: {train.shape}, Test: {test.shape}, Original: {original.shape}")
print(f"Numeric: {NUMERIC_COLS}")
print(f"Categorical: {CAT_COLS}")

# ===================================================================
# 1. Target distribution
# ===================================================================
print("\n[1/9] Target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = train[TARGET].value_counts().reindex(TARGET_ORDER)
pcts = counts / counts.sum() * 100

sns.barplot(x=counts.index, y=counts.values, palette=TARGET_PALETTE, ax=axes[0])
axes[0].set_title("Target Class Counts (Train)")
axes[0].set_ylabel("Count")
for i, (v, p) in enumerate(zip(counts.values, pcts.values)):
    axes[0].text(i, v + 1000, f"{v:,}\n({p:.1f}%)", ha="center", fontsize=10)

if TARGET in original.columns:
    orig_counts = original[TARGET].value_counts().reindex(TARGET_ORDER)
    orig_pcts = orig_counts / orig_counts.sum() * 100
    sns.barplot(
        x=orig_counts.index,
        y=orig_counts.values,
        palette=TARGET_PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Target Class Counts (Original 10K)")
    axes[1].set_ylabel("Count")
    for i, (v, p) in enumerate(zip(orig_counts.values, orig_pcts.values)):
        axes[1].text(i, v + 20, f"{v:,}\n({p:.1f}%)", ha="center", fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 2. Numeric feature distributions by target
# ===================================================================
print("[2/9] Numeric distributions by target...")
n_num = len(NUMERIC_COLS)
n_cols_grid = 3
n_rows_grid = (n_num + n_cols_grid - 1) // n_cols_grid

fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(6 * n_cols_grid, 4.5 * n_rows_grid))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_COLS):
    for label in TARGET_ORDER:
        subset = train.loc[train[TARGET] == label, col]
        axes[i].hist(
            subset,
            bins=80,
            alpha=0.5,
            label=label,
            color=TARGET_PALETTE[label],
            density=True,
        )
    axes[i].set_title(col, fontsize=12)
    axes[i].legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Numeric Feature Distributions by Target", fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "02_numeric_by_target.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 3. Categorical feature distributions by target
# ===================================================================
print("[3/9] Categorical distributions by target...")
n_cat = len(CAT_COLS)
n_cols_grid = 2
n_rows_grid = (n_cat + n_cols_grid - 1) // n_cols_grid

fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(8 * n_cols_grid, 5 * n_rows_grid))
axes = axes.flatten()

for i, col in enumerate(CAT_COLS):
    ct = pd.crosstab(train[col], train[TARGET], normalize="index")[TARGET_ORDER]
    ct.plot(kind="bar", stacked=True, color=[TARGET_PALETTE[c] for c in TARGET_ORDER], ax=axes[i])
    axes[i].set_title(f"{col} (target rate per category)", fontsize=12)
    axes[i].set_ylabel("Proportion")
    axes[i].legend(fontsize=8)
    axes[i].tick_params(axis="x", rotation=30)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Categorical Features: Target Proportion per Category", fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "03_categorical_by_target.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 4. Correlation matrix
# ===================================================================
print("[4/9] Correlation matrix...")
fig, ax = plt.subplots(figsize=(12, 10))
corr = train[NUMERIC_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    ax=ax,
    vmin=-1,
    vmax=1,
)
ax.set_title("Numeric Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "04_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 5. Original vs synthetic distribution comparison
# ===================================================================
print("[5/9] Original vs synthetic comparison...")
orig_num_cols = [c for c in NUMERIC_COLS if c in original.columns]
n_comp = len(orig_num_cols)
n_cols_grid = 3
n_rows_grid = (n_comp + n_cols_grid - 1) // n_cols_grid

fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(6 * n_cols_grid, 4.5 * n_rows_grid))
axes = axes.flatten()

for i, col in enumerate(orig_num_cols):
    axes[i].hist(
        train[col], bins=100, alpha=0.5, label="Synthetic (630K)", density=True, color="#2196F3"
    )
    axes[i].hist(
        original[col], bins=100, alpha=0.5, label="Original (10K)", density=True, color="#FF5722"
    )
    axes[i].set_title(col, fontsize=12)
    axes[i].legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Original (10K) vs Synthetic (630K) Distributions", fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "05_original_vs_synthetic.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 6. Unique values comparison (rounding artifacts)
# ===================================================================
print("[6/9] Unique values and rounding artifacts...")
rows = []
for col in orig_num_cols:
    rows.append(
        {
            "feature": col,
            "original_unique": original[col].nunique(),
            "synthetic_unique": train[col].nunique(),
            "test_unique": test[col].nunique() if col in test.columns else None,
            "original_min": original[col].min(),
            "synthetic_min": train[col].min(),
            "original_max": original[col].max(),
            "synthetic_max": train[col].max(),
            "original_mean": round(original[col].mean(), 4),
            "synthetic_mean": round(train[col].mean(), 4),
        }
    )
unique_df = pd.DataFrame(rows)
unique_df.to_csv(OUTPUT_DIR / "06_unique_values_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(orig_num_cols))
width = 0.35
ax.bar(x - width / 2, unique_df["original_unique"], width, label="Original (10K)", color="#FF5722")
ax.bar(x + width / 2, unique_df["synthetic_unique"], width, label="Synthetic (630K)", color="#2196F3")
ax.set_xticks(x)
ax.set_xticklabels(orig_num_cols, rotation=45, ha="right")
ax.set_ylabel("Number of Unique Values")
ax.set_title("Unique Values per Numeric Feature: Original vs Synthetic")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "06_unique_values.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 7. Decimal digit distribution (synthetic artifact detection)
# ===================================================================
print("[7/9] Decimal digit distributions...")
sample_cols = ["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Soil_pH"]
sample_cols = [c for c in sample_cols if c in train.columns]

fig, axes = plt.subplots(len(sample_cols), 2, figsize=(14, 4 * len(sample_cols)))
if len(sample_cols) == 1:
    axes = axes.reshape(1, -1)

for i, col in enumerate(sample_cols):
    frac_syn = train[col] - np.floor(train[col])
    d1_syn = np.floor(frac_syn * 10).astype(int)

    frac_orig = original[col] - np.floor(original[col])
    d1_orig = np.floor(frac_orig * 10).astype(int)

    axes[i, 0].hist(d1_syn, bins=np.arange(11) - 0.5, density=True, color="#2196F3", alpha=0.7)
    axes[i, 0].set_title(f"{col}: 1st decimal digit (Synthetic)")
    axes[i, 0].set_xticks(range(10))

    axes[i, 1].hist(d1_orig, bins=np.arange(11) - 0.5, density=True, color="#FF5722", alpha=0.7)
    axes[i, 1].set_title(f"{col}: 1st decimal digit (Original)")
    axes[i, 1].set_xticks(range(10))

fig.suptitle("First Decimal Digit Distribution (Artifact Detection)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "07_decimal_digits.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 8. KDTree nearest neighbor analysis (original data lookup)
# ===================================================================
print("[8/9] Nearest neighbor analysis to original data...")
common_num = [c for c in orig_num_cols if c in original.columns and c in train.columns]

scaler = StandardScaler()
orig_scaled = scaler.fit_transform(original[common_num])
train_sample = train.sample(n=50_000, random_state=42)
train_scaled = scaler.transform(train_sample[common_num])

tree = cKDTree(orig_scaled)
distances, indices = tree.query(train_scaled, k=1)

train_sample = train_sample.copy()
train_sample["nn_distance"] = distances
train_sample["nn_target"] = original[TARGET].iloc[indices].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label in TARGET_ORDER:
    subset = train_sample.loc[train_sample[TARGET] == label, "nn_distance"]
    axes[0].hist(subset, bins=80, alpha=0.5, label=label, color=TARGET_PALETTE[label], density=True)
axes[0].set_title("Distance to Nearest Original Neighbor by Target")
axes[0].set_xlabel("Euclidean Distance (standardized)")
axes[0].legend()

match_rate = (train_sample[TARGET] == train_sample["nn_target"]).groupby(train_sample[TARGET]).mean()
match_rate = match_rate.reindex(TARGET_ORDER)
sns.barplot(x=match_rate.index, y=match_rate.values, palette=TARGET_PALETTE, ax=axes[1])
axes[1].set_title("Target Match Rate with Nearest Original Neighbor")
axes[1].set_ylabel("Match Rate")
for i, v in enumerate(match_rate.values):
    axes[1].text(i, v + 0.01, f"{v:.2%}", ha="center")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "08_nn_original_lookup.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 9. Class separability: top features by ANOVA F-statistic
# ===================================================================
print("[9/9] Feature importance (ANOVA F-stat)...")
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(train[TARGET])

f_scores, p_values = f_classif(train[NUMERIC_COLS], y_encoded)

importance_df = (
    pd.DataFrame({"feature": NUMERIC_COLS, "f_score": f_scores, "p_value": p_values})
    .sort_values("f_score", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(OUTPUT_DIR / "09_feature_importance_anova.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="f_score", y="feature", color="#2196F3", ax=ax)
ax.set_title("Numeric Feature Importance (ANOVA F-Statistic)")
ax.set_xlabel("F-Score (higher = better class separation)")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "09_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Summary report
# ===================================================================
print("\nWriting summary report...")
report_lines = [
    "# EDA Summary: PS6E4 Predicting Irrigation Need",
    "",
    "## Dataset",
    f"- Train: {train.shape[0]:,} rows x {train.shape[1]} columns",
    f"- Test: {test.shape[0]:,} rows x {test.shape[1]} columns",
    f"- Original: {original.shape[0]:,} rows x {original.shape[1]} columns",
    f"- Numeric features: {len(NUMERIC_COLS)}",
    f"- Categorical features: {len(CAT_COLS)}",
    f"- Missing values: {train.isnull().sum().sum()}",
    "",
    "## Target Distribution",
    f"- Low: {counts['Low']:,} ({pcts['Low']:.1f}%)",
    f"- Medium: {counts['Medium']:,} ({pcts['Medium']:.1f}%)",
    f"- High: {counts['High']:,} ({pcts['High']:.1f}%)",
    "",
    "## Unique Values (Original vs Synthetic)",
]
for _, row in unique_df.iterrows():
    report_lines.append(
        f"- {row['feature']}: {int(row['original_unique'])} original, "
        f"{int(row['synthetic_unique'])} synthetic"
    )

report_lines.extend(
    [
        "",
        "## Top Features by ANOVA F-Statistic",
    ]
)
for _, row in importance_df.iterrows():
    report_lines.append(f"- {row['feature']}: F={row['f_score']:.1f} (p={row['p_value']:.2e})")

report_lines.extend(
    [
        "",
        "## Nearest Neighbor Match Rate (Synthetic to Original)",
        f"- Overall: {(train_sample[TARGET] == train_sample['nn_target']).mean():.2%}",
    ]
)
for label in TARGET_ORDER:
    rate = match_rate[label]
    report_lines.append(f"- {label}: {rate:.2%}")

report_lines.extend(
    [
        "",
        "## Generated Files",
        "- 01_target_distribution.png",
        "- 02_numeric_by_target.png",
        "- 03_categorical_by_target.png",
        "- 04_correlation_matrix.png",
        "- 05_original_vs_synthetic.png",
        "- 06_unique_values.png / 06_unique_values_comparison.csv",
        "- 07_decimal_digits.png",
        "- 08_nn_original_lookup.png",
        "- 09_feature_importance.png / 09_feature_importance_anova.csv",
    ]
)

report_path = OUTPUT_DIR / "00_eda_summary.md"
report_path.write_text("\n".join(report_lines))

print(f"\nDone. All outputs saved to {OUTPUT_DIR}/")
