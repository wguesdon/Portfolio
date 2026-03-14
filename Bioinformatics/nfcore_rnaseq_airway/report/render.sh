#!/bin/bash
# report/render.sh
# Renders the Quarto report inside the rnaseq-report container.
#
# Usage (from repository root):
#   bash report/render.sh [--build]
#
# Options:
#   --build   Rebuild the rnaseq-report image before rendering.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="localhost/rnaseq-report:latest"
BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $BUILD || ! podman image exists "$IMAGE" 2>/dev/null; then
    echo "Building ${IMAGE} …"
    podman build -t rnaseq-report "${REPO_ROOT}/report/"
fi

echo "Rendering report/airway.qmd …"
podman run --rm \
    -v "${REPO_ROOT}:/data:z" \
    -w /data \
    "$IMAGE" \
    quarto render report/airway.qmd --to html

echo ""
echo "Report written to: report/airway.html"
