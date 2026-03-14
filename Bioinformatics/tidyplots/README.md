# tidyplots Tutorial — Airway RNA-seq Dataset

A self-contained tutorial demonstrating [tidyplots](https://jbengler.github.io/tidyplots/) for
publication-ready bioinformatics visualisations, using the canonical **airway** RNA-seq dataset
(Himes et al. 2014, GSE52778).

---

## Biological Context

Airway smooth muscle cells from 4 donors were treated with **dexamethasone** (a glucocorticoid
used in asthma therapy) or left untreated. RNA-seq captures the transcriptional response.
Differential expression is computed with **DESeq2** (cell-line blocking design).

| Attribute | Value |
|-----------|-------|
| Organism | *Homo sapiens* |
| Tissue | Airway smooth muscle cells |
| Donors (cell lines) | 4 paired (N61311, N052611, N080611, N061011) |
| Contrast | dexamethasone vs untreated |
| Bioconductor package | [`airway`](https://bioconductor.org/packages/airway/) |

---

## Figures Produced

| File | Description |
|------|-------------|
| `01_volcano.png` | Volcano plot — LFC vs −log10(p-value), coloured by direction |
| `02_ma_plot.png` | MA plot — mean expression vs LFC |
| `03_top_degs_bar.png` | Horizontal bar chart of the top 20 DEGs ranked by FDR |
| `04_sample_medians.png` | Median VST expression per sample, coloured by condition |
| `05_expression_distribution.png` | Violin + box plot of global expression by condition |
| `06_key_genes_strip.png` | Strip plot (mean ± SEM + individual points) for top up-regulated genes |
| `07_pvalue_histogram.png` | Raw p-value distribution (QC diagnostic) |
| `08_significant_degs.png` | Scatter of significant DEGs: LFC vs −log10(padj) |
| `09_heatmap_top50_degs.png` | Row-scaled heatmap of the top 50 DEGs across all 8 samples |

---

## Project Structure

```
tidyplots_tutorial/
├── Dockerfile                # Minimal rocker/r-ver image with all dependencies
├── README.md
├── R/
│   └── tidyplots_airway.R   # Main analysis + plotting script
└── output/                  # Figures written here at runtime (git-ignored)
```

---

## Requirements

- [Podman](https://podman.io/) ≥ 4.x (rootless)
- No R installation required on the host

---

## Quick Start

### 1 — Build the image

```bash
cd Bioinformatics/tidyplots_tutorial

podman build -t tidyplots-airway .
```

The image is based on `rocker/r-ver:4.4.2` (minimal R, no RStudio).
First build takes ~10–15 min while Bioconductor packages compile; subsequent builds use
the layer cache and are instantaneous.

Approximate final image size: **~2 GB** (DESeq2 + its Bioconductor graph pull in ~350 packages).

### 2 — Run the analysis

```bash
podman run --rm \
    -v "$(pwd)/R:/project/R:ro" \
    -v "$(pwd)/output:/project/output:rw" \
    tidyplots-airway
```

Figures are written to `output/` on your host machine.

**Flags explained:**

| Flag | Purpose |
|------|---------|
| `--rm` | Remove container after exit |
| `-v $(pwd)/R:/project/R:ro` | Mount script directory (read-only) |
| `-v $(pwd)/output:/project/output:rw` | Mount output directory (read-write) |

### 3 — View results

```bash
ls output/
```

Open any `.png` in your image viewer. All files are ~150 DPI, suitable for reports and READMEs.

---

## Rebuilding After Script Changes

Because the R script is mounted as a volume, you **do not need to rebuild** the image when
editing `R/tidyplots_airway.R`. Just re-run `podman run …` and the updated script is picked up
automatically.

Only rebuild the image if you change the `Dockerfile` (e.g. add a new R package).

---

## tidyplots API Highlights

The tutorial demonstrates the core tidyplots verbs:

| Function | What it does |
|----------|-------------|
| `tidyplot(x, y, color)` | Initialise a plot (like `ggplot()`) |
| `add_point()` | Scatter layer |
| `add_bar()` | Bar layer |
| `add_violin()` | Violin layer |
| `add_boxplot()` | Box layer |
| `add_mean_bar()` | Mean bar (summary) |
| `add_sem_errorbar()` | SEM error bars |
| `add_data_points_beeswarm()` | Beeswarm jitter layer |
| `add_histogram()` | Histogram layer |
| `add_reference_lines()` | Horizontal / vertical reference lines |
| `split_plot(by)` | Facet into small multiples |
| `flip_plot()` | Swap x/y axes |
| `adjust_colors()` | Override colour palette |
| `adjust_labels()` | Set title, axis, legend labels |
| `theme_tidyplot()` | Clean minimal theme |
| `remove_legend()` | Drop legend |
| `save_plot()` | Write to file (PNG/PDF/SVG) |

---

## References

- Himes BE et al. (2014) RNA-Seq transcriptomics and directionality of glucocorticoid
  response in airway epithelium. *Am J Resp Crit Care Med* 189(12):1538–1546.
  [PMID: 24673577](https://pubmed.ncbi.nlm.nih.gov/24673577/)
- Love MI, Huber W, Anders S (2014) Moderated estimation of fold change and dispersion
  for RNA-seq data with DESeq2. *Genome Biol* 15:550.
- Engler JB (2024) tidyplots: Tidy plots for scientific papers.
  [CRAN](https://CRAN.R-project.org/package=tidyplots)
