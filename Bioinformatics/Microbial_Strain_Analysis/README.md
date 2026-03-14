# Microbial Strain Analysis

**Interactive Streamlit dashboard and EDA** — Exploratory data analysis for a bacterial strain database linking taxonomic classification, growth parameters, and media composition. Surfaces patterns in temperature/pH niches, O2 tolerance, and media formulations across phylogenetically diverse organisms.

---

## Challenge Brief

The goal was to take a relational dataset of bacterial strains and produce
something interesting and insightful — presented in a way that is accessible
to colleagues who are not data scientists or software developers.

Key requirements:
- Analyse and visualise the data to surface meaningful patterns
- Build an interactive interface (notebook, dashboard, or app)
- Document assumptions, findings, and ideas for further exploration
- Code primarily in Python

---

## Dataset

Four relational CSV files:

| File | Description |
|------|-------------|
| `strains.csv` | Bacterial strains with taxonomic classification (phylum, class) and growth parameters (temperature range, pH range, O₂ tolerance) |
| `strain_media.csv` | Maps each strain to one or more known preferred growth media |
| `media_compounds.csv` | Maps each medium to its constituent chemical compounds |
| `compounds.csv` | Compound ID → compound name lookup |

### Schema

```
strains ──< strain_media >── media_compounds >── compounds
  strain_id                    medium_id            compound_id
  phylum                                            compound_name
  class
  temp_min / temp_max / temp_opt
  ph_min / ph_max / ph_opt
  o2_tol
```

---

## Project Structure

```
.
├── Data/
│   ├── strains.csv
│   ├── strain_media.csv
│   ├── media_compounds.csv
│   ├── compounds.csv
│   └── strains_media_combined.csv    # pre-joined convenience table
├── Notebooks/
│   ├── EDA.ipynb                     # Exploratory data analysis + Sweetviz report
│   ├── Dashboard.ipynb               # Interactive Plotly/Dash dashboard
│   └── Data_display_app.ipynb        # Standalone display application
├── Output/
│   └── Report.html                   # Sweetviz EDA report
└── README.md
```

---

## Notebooks

### `EDA.ipynb`
- Distribution of phyla and classes
- Growth parameter distributions (temperature, pH) by taxonomy
- O₂ tolerance breakdown
- Media and compound frequency analysis
- Automated profiling report (Sweetviz → `Output/Report.html`)

### `Dashboard.ipynb`
- Interactive filters by phylum, class, O₂ tolerance
- Scatter plots of growth parameter ranges
- Media composition explorer

### `Data_display_app.ipynb`
- Lightweight standalone app for browsing strain records

---

## Setup

Notebooks were originally developed in Google Colab. To run locally:

```bash
# Install dependencies
pip install pandas plotly dash sweetviz jupyter

# Launch
jupyter notebook Notebooks/EDA.ipynb
```

Update `project_folder` in each notebook to point to your local `Data/` path:

```python
from pathlib import Path
project_folder = Path("../Data")
```

---

## Key Findings

- **Proteobacteria** dominate the dataset, with alphaproteobacteria being the
  most represented class
- Clear temperature niche separation between mesophiles and thermophiles
- Aerobic strains show a broader pH tolerance range than anaerobes
- Several strains share identical media compositions, suggesting convergent
  cultivation strategies across phylogenetically distant organisms
- A small number of compounds (Peptone, Agar, Distilled water) appear in the
  majority of media formulations

---

## Ideas for Further Exploration

- Predict preferred media from growth parameters using a classifier
- Cluster strains by compound profile to identify cultivation archetypes
- Network graph of strain–compound co-occurrence
- Map O₂ tolerance onto a phylogenetic tree
