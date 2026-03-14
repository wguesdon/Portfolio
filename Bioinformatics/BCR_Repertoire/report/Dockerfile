# report/Dockerfile
# Extends immcantation/airrflow with Quarto so the report can be rendered
# inside a container (no host R packages required).
#
# Build:
#   podman build -t bcr-report report/
#
# Render:
#   podman run --rm -v $(pwd):/data:z -w /data bcr-report \
#     quarto render report/galson2020.qmd --to html

FROM docker.io/immcantation/airrflow:4.1.0

ARG QUARTO_VERSION=1.6.43
ARG TARGETARCH=amd64

RUN curl -fsSL \
        "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${TARGETARCH}.tar.gz" \
    | tar -xz -C /usr/local/ --strip-components=1 \
    && quarto --version
