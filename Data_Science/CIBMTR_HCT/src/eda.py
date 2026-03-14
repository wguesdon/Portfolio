"""
CIBMTR — Exploratory Data Analysis

Generates 8 plots to output/eda/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from config import (
    TRAIN_FILE, EDA_DIR, DPI, PALETTE, RACE_PALETTE,
    EFS_COL, TIME_COL, GROUP_COL,
    CATEGORICAL_COLS, NUMERICAL_COLS,
)
from features import load_raw

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
})

RACE_ORDER = [
    "White", "Black or African-American", "Asian",
    "More than one race", "Native Hawaiian or other Pacific Islander",
    "American Indian or Alaska Native",
]
RACE_SHORT = {
    "White": "White",
    "Black or African-American": "Black / AA",
    "Asian": "Asian",
    "More than one race": "Multiracial",
    "Native Hawaiian or other Pacific Islander": "NHPI",
    "American Indian or Alaska Native": "AI/AN",
}


def save(fig, name: str) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path.name}")


def load_data():
    df = load_raw(TRAIN_FILE)
    df[EFS_COL] = (df[EFS_COL] == 1).astype(int)
    df["race_short"] = df[GROUP_COL].map(RACE_SHORT).fillna(df[GROUP_COL])
    return df


# ── Plot 1: Race group distribution ──────────────────────────────────────────

def plot_race_distribution(df: pd.DataFrame) -> None:
    counts = df[GROUP_COL].value_counts()
    labels = [RACE_SHORT.get(r, r) for r in counts.index]
    colors = [RACE_PALETTE.get(r, "#8B8B8B") for r in counts.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], counts.values[::-1], color=colors[::-1], alpha=0.88)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({val/len(df)*100:.1f}%)",
                va="center", fontsize=10)
    ax.set_xlabel("Number of patients")
    ax.set_title("Patient Distribution by Race Group\n(n = 28,800)",
                 fontweight="bold", fontsize=13)
    ax.set_xlim(0, counts.max() * 1.22)
    fig.tight_layout()
    save(fig, "01_race_distribution")


# ── Plot 2: Event rate by race group ─────────────────────────────────────────

def plot_event_rates(df: pd.DataFrame) -> None:
    stats = (
        df.groupby(GROUP_COL)[EFS_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "event_rate", "count": "n"})
        .reset_index()
    )
    stats["race_short"] = stats[GROUP_COL].map(RACE_SHORT).fillna(stats[GROUP_COL])
    stats = stats.sort_values("event_rate", ascending=False)
    colors = [RACE_PALETTE.get(r, "#8B8B8B") for r in stats[GROUP_COL]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(stats["race_short"], stats["event_rate"],
                  color=colors, alpha=0.88)
    for bar, val in zip(bars, stats["event_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.axhline(df[EFS_COL].mean(), color="#333", ls="--", lw=1.5,
               label=f"Overall mean ({df[EFS_COL].mean():.1%})")
    ax.set_ylabel("Event Rate (EFS = 1)")
    ax.set_title("Event Rate by Race Group\n(EFS = 1 means relapse/death occurred)",
                 fontweight="bold", fontsize=13)
    ax.set_ylim(0, stats["event_rate"].max() * 1.18)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "02_event_rates_by_race")


# ── Plot 3: Survival time distribution ───────────────────────────────────────

def plot_survival_time_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(df[TIME_COL], bins=60, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.set_xlabel("EFS Time (months)")
    ax.set_ylabel("Count")
    ax.set_title("EFS Time Distribution (raw)", fontweight="bold")
    ax.axvline(df[TIME_COL].median(), color="#E63946", lw=2, ls="--",
               label=f"Median: {df[TIME_COL].median():.1f} mo")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.hist(np.log1p(df[TIME_COL]), bins=60, color=PALETTE[1], alpha=0.8, edgecolor="white")
    ax.set_xlabel("log(1 + EFS Time)")
    ax.set_ylabel("Count")
    ax.set_title("EFS Time Distribution (log-transformed)", fontweight="bold")

    fig.suptitle("Event-Free Survival Time Distribution", fontweight="bold", fontsize=14)
    fig.tight_layout()
    save(fig, "03_survival_time_distribution")


# ── Plot 4: Kaplan-Meier curves by race ──────────────────────────────────────

def plot_km_curves(df: pd.DataFrame) -> None:
    races = [r for r in RACE_ORDER if r in df[GROUP_COL].unique()]

    fig, ax = plt.subplots(figsize=(12, 7))
    kmf = KaplanMeierFitter()

    for race in races:
        mask = df[GROUP_COL] == race
        kmf.fit(
            df.loc[mask, TIME_COL],
            event_observed=df.loc[mask, EFS_COL],
            label=RACE_SHORT.get(race, race),
        )
        color = RACE_PALETTE.get(race, "#8B8B8B")
        kmf.plot_survival_function(ax=ax, color=color, linewidth=2,
                                   ci_show=True, ci_alpha=0.08)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Event-Free Survival Probability")
    ax.set_title("Kaplan-Meier Survival Curves by Race Group",
                 fontweight="bold", fontsize=13)
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, "04_kaplan_meier_by_race")


# ── Plot 5: Missing values heatmap ────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> None:
    cols_with_missing = (
        df.drop(columns=[EFS_COL, TIME_COL, "race_short"])
        .isnull()
        .mean()
        .sort_values(ascending=False)
    )
    cols_with_missing = cols_with_missing[cols_with_missing > 0]

    if len(cols_with_missing) == 0:
        print("  No missing values found — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(cols_with_missing) * 0.3)))
    colors = [PALETTE[0] if v < 0.1 else PALETTE[1] if v < 0.3 else PALETTE[2]
              for v in cols_with_missing.values]
    ax.barh(cols_with_missing.index[::-1], cols_with_missing.values[::-1],
            color=colors[::-1], alpha=0.85)
    ax.axvline(0.1, color="#999", ls="--", lw=1, label="10%")
    ax.axvline(0.3, color="#E63946", ls="--", lw=1, label="30%")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Missing Value Rate by Feature", fontweight="bold", fontsize=13)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "05_missing_values")


# ── Plot 6: Key numerical features by event status ───────────────────────────

def plot_numeric_by_event(df: pd.DataFrame) -> None:
    key_cols = ["age_at_hct", "donor_age", "karnofsky_score",
                "comorbidity_score", "hla_high_res_10"]
    key_cols = [c for c in key_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(key_cols), figsize=(4 * len(key_cols), 5))

    for ax, col in zip(axes, key_cols):
        for event_val, label, color in [(0, "Censored", PALETTE[0]), (1, "Event", PALETTE[1])]:
            vals = df.loc[df[EFS_COL] == event_val, col].dropna()
            ax.hist(vals, bins=30, alpha=0.6, color=color, label=label,
                    density=True, edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("Value")

    fig.suptitle("Key Feature Distributions by Event Status",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "06_features_by_event_status")


# ── Plot 7: Top categorical features ─────────────────────────────────────────

def plot_categorical_event_rates(df: pd.DataFrame) -> None:
    key_cats = ["prim_disease_hct", "conditioning_intensity",
                "graft_type", "donor_related"]
    key_cats = [c for c in key_cats if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, col in zip(axes, key_cats):
        stats = (
            df.groupby(col)[EFS_COL]
            .agg(["mean", "count"])
            .rename(columns={"mean": "event_rate", "count": "n"})
            .sort_values("event_rate", ascending=False)
            .head(12)
        )
        colors = PALETTE[:len(stats)]
        bars = ax.bar(range(len(stats)), stats["event_rate"],
                      color=colors * (len(stats) // len(colors) + 1),
                      alpha=0.85)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats.index, rotation=30, ha="right", fontsize=8)
        ax.axhline(df[EFS_COL].mean(), color="#333", ls="--", lw=1.2, alpha=0.7)
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_ylabel("Event Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Event Rate by Key Categorical Features",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    save(fig, "07_categorical_event_rates")


# ── Plot 8: Correlation matrix of numerical features ─────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    num_cols += [TIME_COL, EFS_COL]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4,
        annot_kws={"size": 7}, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix — Numerical Features",
                 fontweight="bold", fontsize=13, pad=12)
    fig.tight_layout()
    save(fig, "08_correlation_matrix")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} patients | {df.shape[1]} columns")
    print(f"  Event rate: {df[EFS_COL].mean():.1%}")
    print(f"  Race groups: {df[GROUP_COL].nunique()}")

    print("\nGenerating EDA plots:")
    plot_race_distribution(df)
    plot_event_rates(df)
    plot_survival_time_distribution(df)
    plot_km_curves(df)
    plot_missing_values(df)
    plot_numeric_by_event(df)
    plot_categorical_event_rates(df)
    plot_correlation_matrix(df)

    print(f"\nAll EDA plots saved to  output/eda/")


if __name__ == "__main__":
    main()
