#!/bin/bash
# analysis/run.sh
# Generates all figures using R inside the rnaseq-report container.
#
# Usage (from repository root):
#   bash analysis/run.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="localhost/rnaseq-report:latest"

if ! podman image exists "$IMAGE" 2>/dev/null; then
    echo "Building ${IMAGE} …"
    podman build -t rnaseq-report "${REPO_ROOT}/report/"
fi

echo "Generating figures …"
podman run --rm \
    -v "${REPO_ROOT}:/data:z" \
    -w /data \
    "$IMAGE" \
    Rscript analysis/R/plot_de.R

echo ""
echo "Figures saved to: analysis/plots/"
