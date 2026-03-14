#!/bin/bash
# scripts/upload_to_s3.sh
# Uploads a dataset's FASTQs and samplesheet to S3 and rewrites the
# samplesheet to use S3 paths.
#
# Usage:
#   export S3_BUCKET=your-bucket
#   ./scripts/upload_to_s3.sh --dataset galson2020

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$DATASET"   ]] && { echo "ERROR: --dataset is required"; exit 1; }
[[ -z "${S3_BUCKET:-}" ]] && { echo "ERROR: S3_BUCKET must be set"; exit 1; }

DATA_DIR="${REPO_ROOT}/data/${DATASET}"
S3_PREFIX="s3://${S3_BUCKET}/data/${DATASET}"

echo "Uploading ${DATASET} → ${S3_PREFIX}/"
aws s3 sync "${DATA_DIR}/fastq/" "${S3_PREFIX}/fastq/" \
    --exclude "*" --include "*.fastq.gz" \
    --storage-class INTELLIGENT_TIERING

# Rewrite local samplesheet to use S3 paths
S3_SHEET="${DATA_DIR}/samplesheet_s3.tsv"
sed "s|${DATA_DIR}/fastq/|${S3_PREFIX}/fastq/|g" \
    "${DATA_DIR}/samplesheet.tsv" > "$S3_SHEET"

aws s3 cp "$S3_SHEET" "${S3_PREFIX}/samplesheet.tsv"

echo "Done. S3 samplesheet: ${S3_PREFIX}/samplesheet.tsv"
