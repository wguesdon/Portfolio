"""
NYC Airbnb 2019 — Exploratory Data Analysis
Generates all EDA plots to output/eda/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

from config import (
    DATA_FILE, EDA_DIR, DPI,
    BOROUGH_PALETTE, ROOM_PALETTE, BOROUGH_ORDER, RANDOM_STATE,
)
from features import load_and_clean

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
    "font.family":       "DejaVu Sans",
})


def save(fig, name: str) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path.name}")


# ── 1. Price Distribution ─────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame) -> None:
    clip_p99 = df["price"].quantile(0.99)
    log_prices = np.log1p(df["price"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.histplot(
        df.loc[df["price"] <= clip_p99, "price"],
        bins=60, color="#2196F3", kde=True, ax=ax,
        line_kws={"lw": 2},
    )
    ax.axvline(df["price"].median(), color="#E63946", ls="--", lw=1.8,
               label=f"Median  ${df['price'].median():.0f}")
    ax.set_title("Price Distribution (clipped at 99th pct)", fontweight="bold")
    ax.set_xlabel("Price (USD / night)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(frameon=False)

    ax = axes[1]
    sns.histplot(log_prices, bins=60, color="#FF5722", kde=True, ax=ax,
                 line_kws={"lw": 2})
    ax.axvline(log_prices.median(), color="#333", ls="--", lw=1.8,
               label=f"Median  {log_prices.median():.2f}")
    ax.set_title("log(Price + 1) Distribution", fontweight="bold")
    ax.set_xlabel("log(Price + 1)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    fig.suptitle("NYC Airbnb 2019 — Price Distribution",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "01_price_distribution")


# ── 2. Price by Borough ───────────────────────────────────────────────────────
def plot_price_by_borough(df: pd.DataFrame) -> None:
    clip = df["price"].quantile(0.99)
    df_c = df[df["price"] <= clip]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    sns.violinplot(
        data=df_c, x="neighbourhood_group", y="price",
        hue="neighbourhood_group", order=BOROUGH_ORDER,
        palette=BOROUGH_PALETTE, legend=False,
        inner="quartile", cut=0, ax=ax,
    )
    ax.set_title("Price Distribution by Borough", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Price (USD / night)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    ax = axes[1]
    stats = (
        df.groupby("neighbourhood_group")["price"]
        .agg(["median", "mean"])
        .reindex(BOROUGH_ORDER)
        .reset_index()
    )
    x = np.arange(len(stats))
    w = 0.35
    colors = [BOROUGH_PALETTE[b] for b in stats["neighbourhood_group"]]
    b1 = ax.bar(x - w / 2, stats["median"], w, color=colors, alpha=0.85, label="Median")
    b2 = ax.bar(x + w / 2, stats["mean"],   w, color=colors, alpha=0.45, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["neighbourhood_group"])
    ax.set_title("Median vs Mean Price by Borough", fontweight="bold")
    ax.set_ylabel("Price (USD / night)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(frameon=False)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"${bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("NYC Airbnb 2019 — Price by Borough",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_price_by_borough")


# ── 3. Price by Room Type ─────────────────────────────────────────────────────
def plot_price_by_room_type(df: pd.DataFrame) -> None:
    clip = df["price"].quantile(0.99)
    room_order = ["Entire home/apt", "Private room", "Shared room"]
    df_c = df[df["price"] <= clip]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sns.violinplot(
        data=df_c, x="room_type", y="price",
        hue="room_type", order=room_order,
        palette=ROOM_PALETTE, legend=False,
        inner="quartile", cut=0, ax=ax,
    )
    ax.set_title("Price Distribution by Room Type", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Price (USD / night)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))

    ax = axes[1]
    counts = df["room_type"].value_counts().reindex(room_order)
    bars = ax.barh(
        counts.index, counts.values,
        color=[ROOM_PALETTE[r] for r in counts.index], alpha=0.85,
    )
    ax.set_title("Listing Count by Room Type", fontweight="bold")
    ax.set_xlabel("Number of Listings")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 150, bar.get_y() + bar.get_height() / 2,
                f"{w:,.0f}", va="center", fontsize=10)
    ax.invert_yaxis()

    fig.suptitle("NYC Airbnb 2019 — Room Types",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_room_type_analysis")


# ── 4. Geographic Maps ────────────────────────────────────────────────────────
def plot_geographic(df: pd.DataFrame) -> None:
    mask = (
        df["latitude"].between(40.48, 40.94) &
        df["longitude"].between(-74.3, -73.65)
    )
    geo = df[mask].copy()
    geo["log_price"] = np.log1p(geo["price"])

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Coloured by log-price
    ax = axes[0]
    sc = ax.scatter(
        geo["longitude"], geo["latitude"],
        c=geo["log_price"], cmap="YlOrRd",
        s=1.5, alpha=0.5, rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log(Price + 1)", fontsize=9)
    ax.set_title("Listing Prices", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    # Coloured by borough
    ax = axes[1]
    for borough in BOROUGH_ORDER:
        sub = geo[geo["neighbourhood_group"] == borough]
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=BOROUGH_PALETTE[borough], s=1.5, alpha=0.4,
                   label=borough, rasterized=True)
    ax.legend(markerscale=6, frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Listings by Borough", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    # Coloured by room type
    ax = axes[2]
    for rtype in ["Entire home/apt", "Private room", "Shared room"]:
        sub = geo[geo["room_type"] == rtype]
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=ROOM_PALETTE[rtype], s=1.5, alpha=0.4,
                   label=rtype, rasterized=True)
    ax.legend(markerscale=6, frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Listings by Room Type", fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    fig.suptitle("NYC Airbnb 2019 — Geographic Distribution",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "04_geographic_maps")


# ── 5. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame) -> None:
    num_cols = [
        "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365", "latitude", "longitude",
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    save(fig, "05_correlation_heatmap")


# ── 6. Top Neighbourhoods ─────────────────────────────────────────────────────
def plot_top_neighbourhoods(df: pd.DataFrame) -> None:
    # Map each neighbourhood to its borough for colouring
    nb_borough = (
        df[["neighbourhood", "neighbourhood_group"]]
        .drop_duplicates()
        .set_index("neighbourhood")["neighbourhood_group"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    ax = axes[0]
    top_count = df["neighbourhood"].value_counts().head(15)
    colors_c = [BOROUGH_PALETTE[nb_borough[n]] for n in top_count.index]
    bars = ax.barh(top_count.index[::-1], top_count.values[::-1],
                   color=colors_c[::-1], alpha=0.85)
    ax.set_title("Top 15 Neighbourhoods by Listing Count", fontweight="bold")
    ax.set_xlabel("Number of Listings")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 20, bar.get_y() + bar.get_height() / 2,
                f"{w:,.0f}", va="center", fontsize=9)

    ax = axes[1]
    median_price = (
        df[df["price"] <= df["price"].quantile(0.99)]
        .groupby("neighbourhood")["price"]
        .agg(["median", "count"])
        .query("count >= 30")
        .nlargest(15, "median")
    )
    colors_p = [BOROUGH_PALETTE[nb_borough[n]] for n in median_price.index]
    bars2 = ax.barh(median_price.index[::-1], median_price["median"][::-1],
                    color=colors_p[::-1], alpha=0.85)
    ax.set_title("Top 15 Neighbourhoods by Median Price\n(min. 30 listings)",
                 fontweight="bold")
    ax.set_xlabel("Median Price (USD / night)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
                f"${w:.0f}", va="center", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in BOROUGH_PALETTE.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("NYC Airbnb 2019 — Neighbourhood Analysis",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "06_neighbourhood_analysis")


# ── 7. Feature Distributions (Raw vs Log) ────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame) -> None:
    features = {
        "minimum_nights":                 "Minimum Nights",
        "number_of_reviews":              "Number of Reviews",
        "reviews_per_month":              "Reviews / Month",
        "calculated_host_listings_count": "Host Listings Count",
        "availability_365":               "Availability (days/yr)",
    }

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, (col, label) in enumerate(features.items()):
        clip = df[col].quantile(0.99)

        ax = axes[0, i]
        sns.histplot(df.loc[df[col] <= clip, col], bins=40, color="#2196F3",
                     kde=True, ax=ax, line_kws={"lw": 1.5})
        ax.set_title(f"{label}\n(raw)", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count" if i == 0 else "")

        ax = axes[1, i]
        sns.histplot(np.log1p(df[col]), bins=40, color="#FF5722",
                     kde=True, ax=ax, line_kws={"lw": 1.5})
        ax.set_title(f"{label}\n(log scale)", fontsize=10, fontweight="bold")
        ax.set_xlabel("log(x + 1)")
        ax.set_ylabel("Count" if i == 0 else "")

    fig.suptitle(
        "NYC Airbnb 2019 — Feature Distributions (Raw vs Log-Transformed)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "07_feature_distributions")


# ── 8. Price vs Key Features ──────────────────────────────────────────────────
def plot_price_vs_features(df: pd.DataFrame) -> None:
    sample = df.sample(min(10_000, len(df)), random_state=RANDOM_STATE)
    log_price = np.log1p(sample["price"])
    colors = [BOROUGH_PALETTE[b] for b in sample["neighbourhood_group"]]

    pairs = [
        ("number_of_reviews",              "Number of Reviews"),
        ("reviews_per_month",              "Reviews / Month"),
        ("calculated_host_listings_count", "Host Listings Count"),
        ("availability_365",               "Availability (days/yr)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (col, label) in zip(axes.flat, pairs):
        ax.scatter(np.log1p(sample[col]), log_price,
                   c=colors, s=5, alpha=0.3, rasterized=True)
        ax.set_xlabel(f"log({label} + 1)")
        ax.set_ylabel("log(Price + 1)")
        ax.set_title(f"Price vs {label}", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in BOROUGH_PALETTE.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle("NYC Airbnb 2019 — Price vs Key Features (coloured by borough)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_price_vs_features")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    df = load_and_clean()
    print(f"  {len(df):,} listings | {df['neighbourhood'].nunique()} neighbourhoods")
    print(f"  Price range: ${df['price'].min()} – ${df['price'].max():,}")

    print("\nGenerating EDA plots:")
    plot_price_distribution(df)
    plot_price_by_borough(df)
    plot_price_by_room_type(df)
    plot_geographic(df)
    plot_correlation(df)
    plot_top_neighbourhoods(df)
    plot_feature_distributions(df)
    plot_price_vs_features(df)

    print(f"\nAll EDA plots saved to  {EDA_DIR.relative_to(EDA_DIR.parent.parent)}/")


if __name__ == "__main__":
    main()
