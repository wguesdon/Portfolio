# ── Base: versioned minimal R image (no RStudio, no system extras) ──────────
FROM rocker/r-ver:4.4.2

LABEL maintainer="Portfolio 2026"
LABEL description="tidyplots tutorial — Airway RNA-seq (GSE52778)"

# ── System libraries required by Bioconductor/R packages ────────────────────
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libtiff5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install pak (fast parallel installer) ───────────────────────────────────
RUN Rscript -e "install.packages('pak', repos = 'https://r-lib.github.io/p/pak/stable/')"

# ── Install CRAN packages ────────────────────────────────────────────────────
RUN Rscript -e "pak::pkg_install(c( \
        'tidyplots', \
        'dplyr', \
        'tibble', \
        'tidyr', \
        'forcats', \
        'ggplot2', \
        'ggbeeswarm' \
    ), ask = FALSE)"

# ── Install Bioconductor packages ────────────────────────────────────────────
RUN Rscript -e "pak::pkg_install(c( \
        'bioc::BiocManager', \
        'bioc::airway', \
        'bioc::DESeq2', \
        'bioc::SummarizedExperiment', \
        'bioc::S4Vectors', \
        'bioc::ComplexHeatmap', \
        'bioc::circlize' \
    ), ask = FALSE)"

# ── Clean R package cache ────────────────────────────────────────────────────
RUN Rscript -e "pak::cache_clean()"

WORKDIR /project

CMD ["Rscript", "R/tidyplots_airway.R"]
