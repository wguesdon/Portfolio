# report/Dockerfile
# R + Quarto environment for rendering the airway RNA-seq report.
#
# Build:
#   podman build -t rnaseq-report report/
#
# Render:
#   podman run --rm -v $(pwd):/data:z -w /data rnaseq-report \
#     quarto render report/airway.qmd --to html

FROM docker.io/rocker/r-ver:4.4.0

ARG QUARTO_VERSION=1.6.43
ARG TARGETARCH=amd64

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev \
    libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev \
    libtiff5-dev libjpeg-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Quarto
RUN curl -fsSL \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${TARGETARCH}.tar.gz" \
    | tar -xz -C /usr/local/ --strip-components=1 \
    && quarto --version

# R packages
RUN install2.r --error --skipinstalled \
    readr dplyr tidyr stringr ggplot2 patchwork scales knitr rmarkdown \
    pheatmap RColorBrewer ggrepel tibble
