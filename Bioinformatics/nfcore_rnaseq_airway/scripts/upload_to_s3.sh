#!/bin/bash
# scripts/upload_to_s3.sh
# Uploads dataset files to S3 for AWS Batch pipeline execution.
#
# Usage:
#   ./scripts/upload_to_s3.sh --dataset DATASET [--bucket BUCKET] [--dry-run]
#
# What it does:
#   1. Rewrites samplesheet.csv with S3 paths → samplesheet_s3.csv
#   2. Uploads FASTQs, reference files, and samplesheets to S3
#
# Example:
#   export S3_BUCKET=my-nextflow-bucket
#   ./scripts/upload_to_s3.sh --dataset airway
#   ./scripts/upload_to_s3.sh --dataset airway --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET=""
S3_BUCKET="${S3_BUCKET:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";   shift 2 ;;
        --bucket)   S3_BUCKET="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true;   shift   ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 --dataset DATASET [--bucket BUCKET] [--dry-run]"
            exit 1
            ;;
    esac
done

[[ -z "$DATASET"   ]] && { echo "ERROR: --dataset is required"; exit 1; }
[[ -z "$S3_BUCKET" ]] && { echo "ERROR: S3_BUCKET env var is required"; exit 1; }

DATA_DIR="${REPO_ROOT}/data/${DATASET}"
S3_DATA="s3://${S3_BUCKET}/data/${DATASET}"

[[ -d "$DATA_DIR" ]] || { echo "ERROR: Data directory not found: $DATA_DIR — run: make data-${DATASET}"; exit 1; }

echo "=================================================================="
echo "Upload Dataset to S3"
echo "=================================================================="
printf "Dataset   : %s\n" "$DATASET"
printf "Local dir : %s\n" "$DATA_DIR"
printf "S3 dest   : %s\n" "$S3_DATA"
[[ "$DRY_RUN" == true ]] && echo "Mode      : DRY RUN"
echo "=================================================================="
echo ""

# ──────────────────────────────────────────────────────────────
# Step 1: Generate S3 samplesheet
# ──────────────────────────────────────────────────────────────
echo "Step 1: Generating samplesheet_s3.csv..."

SAMPLESHEET_LOCAL="${DATA_DIR}/samplesheet.csv"
SAMPLESHEET_S3="${DATA_DIR}/samplesheet_s3.csv"

[[ -f "$SAMPLESHEET_LOCAL" ]] || { echo "ERROR: samplesheet.csv not found — run: make data-${DATASET}"; exit 1; }

# Replace local absolute paths with S3 paths
sed "s|${DATA_DIR}/|${S3_DATA}/|g" "$SAMPLESHEET_LOCAL" > "$SAMPLESHEET_S3"

echo "  Created: samplesheet_s3.csv"
echo "  Preview:"
head -3 "$SAMPLESHEET_S3" | sed 's/^/    /'
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Upload files
# ──────────────────────────────────────────────────────────────
echo "Step 2: Uploading to S3..."
echo ""

_upload() {
    local LOCAL="$1" DEST="$2"
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] aws s3 cp $LOCAL $DEST"
    else
        echo "  $(basename "$LOCAL")..."
        aws s3 cp "$LOCAL" "$DEST" --quiet
    fi
}

# Samplesheets and design files
_upload "$SAMPLESHEET_S3"                    "${S3_DATA}/samplesheet_s3.csv"
_upload "${DATA_DIR}/samplesheet_de.csv"     "${S3_DATA}/samplesheet_de.csv"
_upload "${DATA_DIR}/contrasts.csv"          "${S3_DATA}/contrasts.csv"

# Reference files
for REF in human_genome.fa.gz human_transcripts.fa.gz human_annotation.gtf.gz; do
    [[ -f "${DATA_DIR}/${REF}" ]] && _upload "${DATA_DIR}/${REF}" "${S3_DATA}/${REF}"
done

# FASTQ files
for FQ in "${DATA_DIR}"/*_R{1,2}.fastq.gz; do
    [[ -f "$FQ" ]] && _upload "$FQ" "${S3_DATA}/$(basename "$FQ")"
done

echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Verify
# ──────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
    echo "Step 3: Verifying upload..."
    echo ""
    aws s3 ls "${S3_DATA}/" --human-readable
    echo ""
fi

echo "=================================================================="
echo "Upload complete!"
echo "=================================================================="
echo ""
echo "Next step:"
echo "  S3_BUCKET=${S3_BUCKET} ./scripts/run.sh --pipeline rnaseq --dataset ${DATASET} --env aws"
echo ""
