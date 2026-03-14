# R Basics for Bioinformatics Scientists

**From variables to statistical analysis with the Tidyverse** — a comprehensive Quarto tutorial designed for researchers new to R and data-driven analysis, with 19 example figures and an HTML report.

---

## Overview

This tutorial introduces R fundamentals in the context of bioinformatics workflows: variables, data types, data manipulation with dplyr/tidyr, and statistical visualisation with ggplot2. It is designed as a self-contained learning resource that can be rendered into an interactive HTML report.

## Contents

- Variables and data types
- Vectors, lists, and data frames
- Data import and manipulation with the Tidyverse
- Statistical summaries and hypothesis testing
- Visualisation with ggplot2

## Project Structure

```
.
├── Dockerfile
├── run.sh
├── reports/
│   └── r_basics_tutorial.qmd           # Quarto source
└── output/
    ├── r_basics_tutorial.html           # Rendered HTML report
    └── figures/                         # 19 example figures (PNG)
```

## Quick Start

```bash
podman build -t r-basics .
podman run --rm \
    -v "$(pwd)/reports:/project/reports:ro" \
    -v "$(pwd)/output:/project/output:rw" \
    r-basics
```
