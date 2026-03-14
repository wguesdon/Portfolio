#!/bin/bash
set -euo pipefail

IMAGE_NAME="r-basics-tutorial"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman build -t "$IMAGE_NAME" "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/output"

podman run --rm \
    -v "$SCRIPT_DIR/reports:/project/reports:rw" \
    -v "$SCRIPT_DIR/output:/project/output:rw" \
    "$IMAGE_NAME" \
    quarto render reports/r_basics_tutorial.qmd --output-dir /project/output

echo "Rendered tutorial: $SCRIPT_DIR/output/"
