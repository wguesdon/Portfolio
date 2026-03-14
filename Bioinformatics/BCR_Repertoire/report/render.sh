#!/bin/bash
# report/render.sh
# Renders the Quarto report inside the bcr-report container.
#
# Usage (from repository root):
#   bash report/render.sh [--build]
#
# Options:
#   --build   Rebuild the bcr-report image before rendering.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="localhost/bcr-report:latest"
BUILD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $BUILD || ! podman image exists "$IMAGE" 2>/dev/null; then
    echo "Building ${IMAGE} …"
    podman build -t bcr-report "${REPO_ROOT}/report/"
fi

echo "Rendering report/galson2020.qmd …"
podman run --rm \
    -v "${REPO_ROOT}:/data:z" \
    -w /data \
    "$IMAGE" \
    quarto render report/galson2020.qmd --to html

echo ""
echo "Report written to: report/galson2020.html"
