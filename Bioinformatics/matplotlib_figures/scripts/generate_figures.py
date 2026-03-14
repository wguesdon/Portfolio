#!/usr/bin/env python3
"""Generate publication-ready figures using matplotlib, seaborn, and SciencePlots.

Datasets: palmerpenguins, gapminder, iris (seaborn built-in).
Color palette: NPG (Nature Publishing Group).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 — registers styles with matplotlib
import seaborn as sns
from gapminder import gapminder
from palmerpenguins import load_penguins
from scipy import stats
from statannotations.Annotator import Annotator

# ── Style & palette ──────────────────────────────────────────────────────────
plt.style.use(["science", "nature", "no-latex"])

NPG_3 = ["#E64B35", "#4DBBD5", "#00A087"]
NPG_2 = ["#E64B35", "#4DBBD5"]

OUTDIR = Path(__file__).resolve().parent.parent / "output"
OUTDIR.mkdir(exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
penguins = load_penguins().dropna(subset=["bill_length_mm", "body_mass_g"])
iris = sns.load_dataset("iris")

SPECIES_ORDER = ["Adelie", "Chinstrap", "Gentoo"]
SPECIES_PAIRS = [("Adelie", "Chinstrap"), ("Chinstrap", "Gentoo"), ("Adelie", "Gentoo")]


def _savefig(fig: plt.Figure, name: str) -> None:
    """Save figure at 300 DPI and close it."""
    fig.savefig(OUTDIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ── 1. Density plot: penguin body mass by species ────────────────────────────
def fig01_density() -> None:
    """Density plot of body mass by species with mean lines and rug."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for species, color in zip(SPECIES_ORDER, NPG_3):
        data = penguins.loc[penguins["species"] == species, "body_mass_g"]
        sns.kdeplot(data, ax=ax, color=color, fill=True, alpha=0.3, label=species)
        sns.rugplot(data, ax=ax, color=color, alpha=0.5, height=0.03)
        ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1)
    ax.set_xlabel("Body Mass (g)")
    ax.set_ylabel("Density")
    ax.set_title("Body Mass Distribution by Penguin Species")
    ax.legend(title="Species")
    _savefig(fig, "01_density_body_mass.png")


# ── 2. Histogram: bill length by species ─────────────────────────────────────
def fig02_histogram() -> None:
    """Histogram of bill length by species with median lines and rug."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for species, color in zip(SPECIES_ORDER, NPG_3):
        data = penguins.loc[penguins["species"] == species, "bill_length_mm"]
        ax.hist(data, bins=15, alpha=0.5, color=color, edgecolor="white", label=species)
        sns.rugplot(data, ax=ax, color=color, alpha=0.5, height=0.03)
        ax.axvline(data.median(), color=color, linestyle="--", linewidth=1)
    ax.set_xlabel("Bill Length (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Bill Length Distribution by Species")
    ax.legend(title="Species")
    _savefig(fig, "02_histogram_bill_length.png")


# ── 3. Box plot with significance brackets ───────────────────────────────────
def fig03_boxplot() -> None:
    """Box plot of flipper length with jitter and Mann-Whitney annotations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=penguins, x="species", y="flipper_length_mm",
        hue="species", order=SPECIES_ORDER, palette=NPG_3, width=0.5,
        boxprops=dict(edgecolor="black"), legend=False, ax=ax,
    )
    sns.stripplot(
        data=penguins, x="species", y="flipper_length_mm",
        hue="species", order=SPECIES_ORDER, palette=NPG_3,
        size=3, alpha=0.4, jitter=True, legend=False, ax=ax,
    )
    annotator = Annotator(
        ax, SPECIES_PAIRS, data=penguins,
        x="species", y="flipper_length_mm", order=SPECIES_ORDER,
    )
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    annotator.apply_and_annotate()

    # Global Kruskal-Wallis
    groups = [g["flipper_length_mm"].values for _, g in penguins.groupby("species")]
    kw_stat, kw_p = stats.kruskal(*groups)
    ax.text(
        0.5, 0.02, f"Kruskal-Wallis p = {kw_p:.2e}",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("Flipper Length (mm)")
    ax.set_title("Flipper Length Comparison Across Species")
    _savefig(fig, "03_boxplot_flipper_length.png")


# ── 4. Violin plot with nested box plot ──────────────────────────────────────
def fig04_violin() -> None:
    """Violin plot of bill depth with nested box plot and significance stars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=penguins, x="species", y="bill_depth_mm",
        hue="species", order=SPECIES_ORDER, palette=NPG_3,
        inner=None, linewidth=1, legend=False, ax=ax,
    )
    sns.boxplot(
        data=penguins, x="species", y="bill_depth_mm",
        order=SPECIES_ORDER, width=0.15,
        boxprops=dict(facecolor="white", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="black"),
        fliersize=0, ax=ax,
    )
    annotator = Annotator(
        ax, SPECIES_PAIRS, data=penguins,
        x="species", y="bill_depth_mm", order=SPECIES_ORDER,
    )
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
    annotator.apply_and_annotate()

    # Global Kruskal-Wallis
    groups = [g["bill_depth_mm"].values for _, g in penguins.groupby("species")]
    kw_stat, kw_p = stats.kruskal(*groups)
    ax.text(
        0.5, 0.02, f"Kruskal-Wallis p = {kw_p:.2e}",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("Bill Depth (mm)")
    ax.set_title("Bill Depth Distribution with Statistical Tests")
    _savefig(fig, "04_violin_bill_depth.png")


# ── 5. Scatter plot with regression per species ──────────────────────────────
def fig05_scatter() -> None:
    """Scatter plot of bill length vs body mass with per-species regression."""
    fig, ax = plt.subplots(figsize=(8, 6))
    label_y_positions = [0.97, 0.92, 0.87]
    for species, color, ypos in zip(SPECIES_ORDER, NPG_3, label_y_positions):
        sub = penguins[penguins["species"] == species]
        sns.regplot(
            data=sub, x="bill_length_mm", y="body_mass_g",
            color=color, scatter_kws=dict(s=20, alpha=0.6),
            line_kws=dict(linewidth=1.5), ci=95, ax=ax,
        )
        r, p = stats.pearsonr(sub["bill_length_mm"], sub["body_mass_g"])
        ax.text(
            0.98, ypos, f"{species}: R = {r:.2f}, p = {p:.1e}",
            transform=ax.transAxes, ha="right",
            fontsize=8, color=color,
        )
    ax.set_xlabel("Bill Length (mm)")
    ax.set_ylabel("Body Mass (g)")
    ax.set_title("Bill Length vs Body Mass with Linear Regression")
    # Manual legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=c, linestyle="", markersize=5, label=s)
               for s, c in zip(SPECIES_ORDER, NPG_3)]
    ax.legend(handles=handles, title="Species")
    _savefig(fig, "05_scatter_bill_vs_mass.png")


# ── 6. Ordered bar plot: Americas 2007 life expectancy ───────────────────────
def fig06_barplot() -> None:
    """Ordered bar plot of life expectancy in the Americas (2007)."""
    americas = (
        gapminder.query("continent == 'Americas' and year == 2007")
        .sort_values("lifeExp")
        .copy()
    )
    median_le = americas["lifeExp"].median()
    americas["group"] = np.where(americas["lifeExp"] < median_le, "Below Median", "Above Median")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [NPG_2[0] if g == "Below Median" else NPG_2[1] for g in americas["group"]]
    ax.bar(americas["country"], americas["lifeExp"], color=colors, edgecolor="white")
    ax.set_ylabel("Life Expectancy (years)")
    ax.set_title("Life Expectancy in the Americas (2007)")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=NPG_2[0], edgecolor="white", label="Below Median"),
        Patch(facecolor=NPG_2[1], edgecolor="white", label="Above Median"),
    ]
    ax.legend(handles=legend_handles, title="Group")
    _savefig(fig, "06_barplot_lifeexp_americas.png")


# ── 7. Deviation plot: Europe 2007 GDP z-scores ─────────────────────────────
def fig07_deviation() -> None:
    """Horizontal bar plot of GDP per capita z-scores in Europe (2007)."""
    europe = gapminder.query("continent == 'Europe' and year == 2007").copy()
    europe["gdp_z"] = (europe["gdpPercap"] - europe["gdpPercap"].mean()) / europe["gdpPercap"].std()
    europe["group"] = np.where(europe["gdp_z"] < 0, "Below Average", "Above Average")
    europe = europe.sort_values("gdp_z")

    fig, ax = plt.subplots(figsize=(8, 10))
    colors = [NPG_2[0] if g == "Below Average" else NPG_2[1] for g in europe["group"]]
    ax.barh(europe["country"], europe["gdp_z"], color=colors, edgecolor="white")
    ax.set_xlabel("GDP per Capita (z-score)")
    ax.set_title("GDP per Capita Deviation in Europe (2007)")
    ax.axvline(0, color="black", linewidth=0.5)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=NPG_2[0], edgecolor="white", label="Below Average"),
        Patch(facecolor=NPG_2[1], edgecolor="white", label="Above Average"),
    ]
    ax.legend(handles=legend_handles, title="GDP Group")
    _savefig(fig, "07_deviation_gdp_europe.png")


# ── 8. Lollipop chart: Europe + USA GDP 2007 ────────────────────────────────
def fig08_lollipop() -> None:
    """Lollipop chart of GDP per capita for Europe and the USA (2007)."""
    gdp = gapminder.query(
        "(continent == 'Europe' or country == 'United States') and year == 2007"
    ).copy()
    gdp["region"] = np.where(gdp["country"] == "United States", "USA", "Europe")
    gdp["gdpPercap_k"] = (gdp["gdpPercap"] / 1000).round(1)
    gdp = gdp.sort_values("gdpPercap_k")

    fig, ax = plt.subplots(figsize=(8, 10))
    colors = [NPG_2[1] if r == "USA" else NPG_2[0] for r in gdp["region"]]
    ax.hlines(
        y=gdp["country"], xmin=0, xmax=gdp["gdpPercap_k"],
        color=colors, linewidth=1,
    )
    ax.scatter(gdp["gdpPercap_k"], gdp["country"], color=colors, s=40, zorder=3)
    for _, row in gdp.iterrows():
        ax.text(
            row["gdpPercap_k"], row["country"], f" {row['gdpPercap_k']}",
            va="center", fontsize=6,
        )
    ax.set_xlabel("GDP per Capita (thousands USD)")
    ax.set_title("GDP per Capita: Europe and the USA (2007)")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=NPG_2[0], linestyle="", label="Europe"),
        Line2D([0], [0], marker="o", color=NPG_2[1], linestyle="", label="USA"),
    ]
    ax.legend(handles=handles, title="Region")
    _savefig(fig, "08_lollipop_gdp_europe_usa.png")


# ── 9. Cleveland dot plot: iris sepal length ─────────────────────────────────
def fig09_cleveland() -> None:
    """Cleveland dot plot of top 10 sepal lengths per iris species."""
    iris_df = iris.copy()
    iris_df["name"] = iris_df["species"].astype(str) + "_" + (iris_df.index + 1).astype(str)
    top10 = pd.concat(
        [g.nlargest(10, "sepal_length") for _, g in iris_df.groupby("species")]
    ).sort_values("sepal_length", ascending=True)
    species_color = {s: c for s, c in zip(["setosa", "versicolor", "virginica"], NPG_3)}

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [species_color[s] for s in top10["species"]]
    ax.hlines(y=range(len(top10)), xmin=0, xmax=top10["sepal_length"], color=colors, linewidth=0.8)
    ax.scatter(top10["sepal_length"], range(len(top10)), color=colors, s=30, zorder=3)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["name"])
    # Color y-tick labels by species
    for tick, species in zip(ax.get_yticklabels(), top10["species"]):
        tick.set_color(species_color[species])
    ax.set_xlabel("Sepal Length (cm)")
    ax.set_title("Top 10 Sepal Lengths per Iris Species")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=c, linestyle="", label=s.title())
               for s, c in species_color.items()]
    ax.legend(handles=handles, title="Species")
    _savefig(fig, "09_cleveland_iris_sepal.png")


# ── 10. Multi-panel: mean ± SE by island ─────────────────────────────────────
def fig10_multipanel() -> None:
    """Two-panel grouped bar chart of body mass and flipper length by island."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, yvar, ylabel in [
        (ax1, "body_mass_g", "Body Mass (g)"),
        (ax2, "flipper_length_mm", "Flipper Length (mm)"),
    ]:
        summary = (
            penguins.groupby(["island", "species"])[yvar]
            .agg(["mean", "sem"])
            .reset_index()
        )
        islands = penguins["island"].unique()
        species_present = penguins["species"].unique()
        n_species = len(species_present)
        width = 0.25
        x = np.arange(len(islands))
        for i, (species, color) in enumerate(zip(SPECIES_ORDER, NPG_3)):
            sub = summary[summary["species"] == species]
            # Match to island order
            means = [sub.loc[sub["island"] == isl, "mean"].values for isl in islands]
            sems = [sub.loc[sub["island"] == isl, "sem"].values for isl in islands]
            means = [m[0] if len(m) > 0 else 0 for m in means]
            sems = [s[0] if len(s) > 0 else 0 for s in sems]
            offset = (i - (n_species - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, yerr=sems, color=color,
                          edgecolor="white", capsize=3, label=species)
        ax.set_xticks(x)
        ax.set_xticklabels(islands)
        ax.set_xlabel("Island")
        ax.set_ylabel(ylabel)

    # Shared legend at bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, title="Species",
               bbox_to_anchor=(0.5, -0.02))

    # Panel labels
    for label, ax in zip(["A", "B"], [ax1, ax2]):
        ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle("Penguin Morphometrics by Island", fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.15)
    _savefig(fig, "10_multipanel_island.png")


# ── 11. Faceted box plot: body mass by sex across species ────────────────────
def fig11_faceted() -> None:
    """Faceted box plot of body mass by sex with per-facet Mann-Whitney tests."""
    pen = penguins.dropna(subset=["sex"]).copy()
    species_list = SPECIES_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    for ax, species in zip(axes, species_list):
        sub = pen[pen["species"] == species]
        sns.boxplot(
            data=sub, x="sex", y="body_mass_g",
            hue="sex", palette=NPG_2, width=0.5, legend=False, ax=ax,
        )
        annotator = Annotator(
            ax, [("female", "male")], data=sub,
            x="sex", y="body_mass_g",
        )
        annotator.configure(test="Mann-Whitney", text_format="star", loc="inside")
        annotator.apply_and_annotate()
        ax.set_title(species)
        ax.set_xlabel("Sex")
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Body Mass (g)")

    fig.suptitle("Body Mass by Sex Across Penguin Species", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "11_faceted_boxplot_sex.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating publication-ready figures...")
    fig01_density()
    fig02_histogram()
    fig03_boxplot()
    fig04_violin()
    fig05_scatter()
    fig06_barplot()
    fig07_deviation()
    fig08_lollipop()
    fig09_cleveland()
    fig10_multipanel()
    fig11_faceted()
    print(f"\nAll 11 figures saved to {OUTDIR}")
