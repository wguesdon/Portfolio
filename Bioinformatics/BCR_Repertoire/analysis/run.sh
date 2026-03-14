#!/bin/bash
# analysis/run.sh
# Runs the BCR repertoire analysis inside the immcantation/airrflow container.
#
# Usage (from repository root):
#   bash analysis/run.sh
#
# The repository root is mounted as /data inside the container so that
# relative paths in the R script resolve correctly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="docker.io/immcantation/airrflow:4.1.0"

echo "============================================================"
echo "BCR Repertoire Analysis — Galson et al. 2020"
echo "Image  : ${IMAGE}"
echo "Repo   : ${REPO_ROOT}"
echo "Output : analysis/plots/"
echo "============================================================"

podman run --rm \
    -v "${REPO_ROOT}:/data:z" \
    -w /data \
    "${IMAGE}" \
    Rscript analysis/R/plot_repertoire.R

echo ""
echo "Done. Plots written to analysis/plots/"
