# Publication-Ready Figures with ggpubr and ggprism

**11 journal-quality figures** generated in R using Nature Publishing Group colour palettes, statistical annotations, and consistent styling — ready for biomedical publications.

---

## Overview

Producing clean, journal-quality figures in R often requires extensive ggplot2 customisation. **ggpubr** provides a high-level interface that simplifies adding statistical annotations, combining panels, and applying consistent styling. **ggprism** complements it with GraphPad Prism-style themes, minor tick guides, and bracket-style axis labels commonly expected in life-science publications.

## Datasets

- [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) — bill/flipper/mass measurements
- [Gapminder](https://www.gapminder.org/) — life expectancy and GDP by country
- [Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) — sepal/petal measurements

## Figures Produced

| File | Description |
|------|-------------|
| `01_density_body_mass.png` | Density plot of penguin body mass by species |
| `02_histogram_bill_length.png` | Histogram of bill length |
| `03_boxplot_flipper_length.png` | Boxplot of flipper length with stat brackets |
| `04_violin_bill_depth.png` | Violin plot of bill depth by species |
| `05_scatter_bill_vs_mass.png` | Scatter plot with regression lines |
| `06_barplot_lifeexp_americas.png` | Life expectancy bar chart (Americas) |
| `07_deviation_gdp_europe.png` | GDP deviation plot (Europe) |
| `08_lollipop_asia_lifeexp.png` | Lollipop chart of life expectancy (Asia) |
| `09_cleveland_iris_sepal.png` | Cleveland dot plot of iris sepal length |
| `10_multipanel_island.png` | Multi-panel figure by island |
| `11_faceted_boxplot_sex.png` | Faceted boxplot by sex |

## Project Structure

```
.
├── Containerfile
├── reports/
│   └── publication_ready_figures.qmd    # Quarto report with all code
├── scripts/
│   └── generate_figures.R               # Standalone figure generation
└── output/                              # PNG figures
```

## Quick Start

```bash
podman build -t ggpubr-figures -f Containerfile .
podman run --rm -v "$(pwd)/output:/project/output:rw" ggpubr-figures
```
