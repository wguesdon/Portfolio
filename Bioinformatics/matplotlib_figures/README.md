# Publication-Ready Figures with matplotlib, seaborn, and SciencePlots

**11 journal-quality figures** generated in Python using NPG (Nature Publishing Group) styling, significance brackets via statannotations, and clean SciencePlots themes.

---

## Overview

Producing clean, journal-quality figures in Python typically requires extensive matplotlib customisation. **SciencePlots** provides journal-style themes (including an NPG-inspired palette), **seaborn** offers a high-level statistical plotting interface, and **statannotations** adds significance brackets — the Python equivalent of ggpubr's `stat_compare_means()`.

## Datasets

- [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) — bill/flipper/mass measurements
- [Gapminder](https://www.gapminder.org/) — life expectancy and GDP by country
- [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) — sepal/petal measurements

## Figures Produced

| File | Description |
|------|-------------|
| `01_density_body_mass.png` | KDE of penguin body mass by species |
| `02_histogram_bill_length.png` | Histogram of bill length |
| `03_boxplot_flipper_length.png` | Boxplot of flipper length with stat brackets |
| `04_violin_bill_depth.png` | Violin plot of bill depth by species |
| `05_scatter_bill_vs_mass.png` | Scatter plot with regression lines |
| `06_barplot_lifeexp_americas.png` | Life expectancy bar chart (Americas) |
| `07_deviation_gdp_europe.png` | GDP deviation plot (Europe) |
| `08_lollipop_gdp_europe_usa.png` | Lollipop chart of GDP (Europe + USA) |
| `09_cleveland_iris_sepal.png` | Cleveland dot plot of iris sepal length |
| `10_multipanel_island.png` | Multi-panel figure by island |
| `11_faceted_boxplot_sex.png` | Faceted boxplot by sex |

## Project Structure

```
.
├── Containerfile
├── pyproject.toml
├── reports/
│   └── publication_ready_figures.qmd    # Quarto report with all code
├── scripts/
│   └── generate_figures.py              # Standalone figure generation
└── output/                              # PNG figures
```

## Quick Start

```bash
podman build -t matplotlib-figures -f Containerfile .
podman run --rm -v "$(pwd)/output:/project/output:rw" matplotlib-figures
```
