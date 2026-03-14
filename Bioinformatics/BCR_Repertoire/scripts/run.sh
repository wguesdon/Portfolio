#!/bin/bash
# scripts/run.sh
# Runs nf-core/airrflow for a given dataset and environment.
#
# Usage:
#   ./scripts/run.sh --dataset DATASET [OPTIONS]
#
# Required:
#   --dataset    Dataset name: galson2020
#
# Options:
#   --env        Execution environment: local (default) | aws
#   --resume     Pass -resume to Nextflow
#   --dry-run    Print the command without executing
#
# Environment variables:
#   CONTAINER_PROFILE  docker (default) | singularity
#   S3_BUCKET          Required when --env aws
#   MAX_MEMORY         Override config (e.g. 64.GB)
#   MAX_CPUS           Override config (e.g. 16)
#
# Examples:
#   ./scripts/run.sh --dataset galson2020
#   ./scripts/run.sh --dataset galson2020 --env aws
#   ./scripts/run.sh --dataset galson2020 --env aws --resume

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET=""
ENV="local"
RESUME=false
DRY_RUN=false
CONTAINER_PROFILE="${CONTAINER_PROFILE:-docker}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2"; shift 2 ;;
        --env)      ENV="$2";     shift 2 ;;
        --resume)   RESUME=true;  shift   ;;
        --dry-run)  DRY_RUN=true; shift   ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 --dataset DATASET [--env ENV] [--resume] [--dry-run]"
            exit 1
            ;;
    esac
done

[[ -z "$DATASET" ]] && { echo "ERROR: --dataset is required"; exit 1; }

# ── Load pipeline version ──────────────────────────────────────
PIPELINE_ENV_FILE="${REPO_ROOT}/pipelines/airrflow/pipeline.env"
[[ -f "$PIPELINE_ENV_FILE" ]] || { echo "ERROR: Missing $PIPELINE_ENV_FILE"; exit 1; }
# shellcheck source=/dev/null
source "$PIPELINE_ENV_FILE"

CONFIG_FILE="${REPO_ROOT}/configs/${ENV}.config"
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: Config not found: configs/${ENV}.config"; exit 1; }

# ── Locate Nextflow ────────────────────────────────────────────
if command -v nextflow &>/dev/null; then
    NF_CMD="nextflow"
elif [[ -x "$HOME/bin/nextflow" ]]; then
    NF_CMD="$HOME/bin/nextflow"
else
    echo "ERROR: Nextflow not found. Run: make install-nextflow"
    exit 1
fi

# ── Resolve paths ──────────────────────────────────────────────
DATA_DIR="${REPO_ROOT}/data/${DATASET}"
PARAMS_FILE="${REPO_ROOT}/pipelines/airrflow/params/${DATASET}.json"

PARAMS_ARGS=()
[[ -f "$PARAMS_FILE" ]] && PARAMS_ARGS+=(-params-file "$PARAMS_FILE")

if [[ "$ENV" == "aws" ]]; then
    [[ -z "${S3_BUCKET:-}" ]] && { echo "ERROR: S3_BUCKET is required for --env aws"; exit 1; }
    INPUT_SAMPLESHEET="s3://${S3_BUCKET}/data/${DATASET}/samplesheet.tsv"
    OUTDIR="s3://${S3_BUCKET}/results/airrflow/${DATASET}"
else
    INPUT_SAMPLESHEET="${DATA_DIR}/samplesheet.tsv"
    if [[ "$DRY_RUN" == false && ! -f "$INPUT_SAMPLESHEET" ]]; then
        echo "ERROR: Samplesheet not found: ${INPUT_SAMPLESHEET}"
        echo "Run: make data-${DATASET}"
        exit 1
    fi
    OUTDIR="${REPO_ROOT}/results/airrflow/${DATASET}"
fi

NF_COMMAND=(
    "$NF_CMD" run "$NF_PIPELINE_NAME"
    -profile "$CONTAINER_PROFILE"
    -c       "$CONFIG_FILE"
    -r       "$PIPELINE_VERSION"
    "${PARAMS_ARGS[@]}"
    --input  "$INPUT_SAMPLESHEET"
    --outdir "$OUTDIR"
)
[[ "$RESUME" == true ]] && NF_COMMAND+=(-resume)

echo "=================================================================="
echo "nf-core/airrflow Runner"
echo "=================================================================="
printf "Pipeline : %s v%s\n" "$NF_PIPELINE_NAME" "$PIPELINE_VERSION"
printf "Dataset  : %s\n"     "$DATASET"
printf "Env      : %s\n"     "$ENV"
printf "Input    : %s\n"     "$INPUT_SAMPLESHEET"
printf "Outdir   : %s\n"     "$OUTDIR"
[[ "$RESUME"  == true ]] && echo "Resume   : yes"
[[ "$DRY_RUN" == true ]] && echo "Dry-run  : yes"
echo "=================================================================="
echo ""
echo "Command:"
printf "  %s\n" "${NF_COMMAND[*]}"
echo ""

[[ "$DRY_RUN" == true ]] && { echo "[DRY RUN] Exiting."; exit 0; }

exec "${NF_COMMAND[@]}"
