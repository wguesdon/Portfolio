# ── Base: versioned minimal R image ──────────────────────────────────────────
FROM rocker/r-ver:4.4.2

LABEL maintainer="Portfolio 2026"
LABEL description="R Basics for Bioinformatics Scientists — Quarto tutorial"

# ── System libraries required by R packages and Quarto ───────────────────────
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
        wget \
        gdebi-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install Quarto CLI ──────────────────────────────────────────────────────
RUN wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb \
    && gdebi -n quarto-1.6.42-linux-amd64.deb \
    && rm quarto-1.6.42-linux-amd64.deb

# ── Install pak (fast parallel installer) ────────────────────────────────────
RUN Rscript -e "install.packages('pak', repos = 'https://r-lib.github.io/p/pak/stable/')"

# ── Install R packages ──────────────────────────────────────────────────────
RUN Rscript -e "pak::pkg_install(c( \
        'tidyverse', \
        'readxl', \
        'writexl', \
        'broom', \
        'rstatix', \
        'ggpubr', \
        'patchwork', \
        'scales', \
        'palmerpenguins', \
        'knitr', \
        'rmarkdown' \
    ), ask = FALSE)"

# ── Clean R package cache ────────────────────────────────────────────────────
RUN Rscript -e "pak::cache_clean()"

WORKDIR /project

CMD ["quarto", "render", "reports/r_basics_tutorial.qmd"]
