#!/bin/bash
# scripts/run.sh
# Runs an nf-core pipeline for a given dataset and environment.
#
# Usage:
#   ./scripts/run.sh --pipeline PIPELINE --dataset DATASET [OPTIONS]
#
# Required:
#   --pipeline   Pipeline name: rnaseq | differentialabundance
#   --dataset    Dataset name:  airway
#
# Options:
#   --env        Execution environment: local (default) | aws
#   --resume     Pass -resume to Nextflow (restart from last checkpoint)
#   --dry-run    Print the Nextflow command without executing
#
# Environment variables:
#   CONTAINER_PROFILE  docker (default) | singularity
#   S3_BUCKET          Required when --env aws
#   MAX_MEMORY         Override config default (e.g. 32.GB)
#   MAX_CPUS           Override config default (e.g. 16)
#
# Examples:
#   ./scripts/run.sh --pipeline rnaseq --dataset airway
#   ./scripts/run.sh --pipeline rnaseq --dataset airway --env aws
#   ./scripts/run.sh --pipeline differentialabundance --dataset airway --env aws --resume

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ──────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────
PIPELINE=""
DATASET=""
ENV="local"
RESUME=false
DRY_RUN=false
CONTAINER_PROFILE="${CONTAINER_PROFILE:-docker}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline)  PIPELINE="$2"; shift 2 ;;
        --dataset)   DATASET="$2";  shift 2 ;;
        --env)       ENV="$2";      shift 2 ;;
        --resume)    RESUME=true;   shift   ;;
        --dry-run)   DRY_RUN=true;  shift   ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Usage: $0 --pipeline PIPELINE --dataset DATASET [--env ENV] [--resume] [--dry-run]"
            exit 1
            ;;
    esac
done

[[ -z "$PIPELINE" ]] && { echo "ERROR: --pipeline is required"; exit 1; }
[[ -z "$DATASET"  ]] && { echo "ERROR: --dataset is required";  exit 1; }

# ──────────────────────────────────────────────────────────────
# Load pipeline version info
# ──────────────────────────────────────────────────────────────
PIPELINE_ENV_FILE="${REPO_ROOT}/pipelines/${PIPELINE}/pipeline.env"
if [[ ! -f "$PIPELINE_ENV_FILE" ]]; then
    echo "ERROR: Pipeline config not found: pipelines/${PIPELINE}/pipeline.env"
    exit 1
fi
# shellcheck source=/dev/null
source "$PIPELINE_ENV_FILE"
# NF_PIPELINE_NAME and PIPELINE_VERSION are now set

CONFIG_FILE="${REPO_ROOT}/configs/${ENV}.config"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config not found: configs/${ENV}.config"
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Locate Nextflow
# ──────────────────────────────────────────────────────────────
if command -v nextflow &>/dev/null; then
    NF_CMD="nextflow"
elif [[ -x "$HOME/bin/nextflow" ]]; then
    NF_CMD="$HOME/bin/nextflow"
else
    echo "ERROR: Nextflow not found. Run: make install-nextflow"
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Build pipeline-specific arguments
# ──────────────────────────────────────────────────────────────
DATA_DIR="${REPO_ROOT}/data/${DATASET}"
RESULTS_DIR="${REPO_ROOT}/results"

NF_PROFILE=""
PARAMS_ARGS=()
PATH_ARGS=()

case "$PIPELINE" in

    rnaseq)
        NF_PROFILE="${CONTAINER_PROFILE}"

        if [[ "$ENV" == "aws" ]]; then
            S3_DATA="s3://${S3_BUCKET}/data/${DATASET}"
            PATH_ARGS+=(
                "--input"            "${S3_DATA}/samplesheet_s3.csv"
                "--fasta"            "${S3_DATA}/human_genome.fa.gz"
                "--transcript_fasta" "${S3_DATA}/human_transcripts.fa.gz"
                "--gtf"              "${S3_DATA}/human_annotation.gtf.gz"
            )
        else
            if [[ "$DRY_RUN" == false ]]; then
                for f in "${DATA_DIR}/samplesheet.csv" \
                          "${DATA_DIR}/human_genome.fa.gz" \
                          "${DATA_DIR}/human_transcripts.fa.gz" \
                          "${DATA_DIR}/human_annotation.gtf.gz"; do
                    [[ -f "$f" ]] || { echo "ERROR: Missing: $f — run: make data-${DATASET}"; exit 1; }
                done
            fi
            PATH_ARGS+=(
                "--input"            "${DATA_DIR}/samplesheet.csv"
                "--fasta"            "${DATA_DIR}/human_genome.fa.gz"
                "--transcript_fasta" "${DATA_DIR}/human_transcripts.fa.gz"
                "--gtf"              "${DATA_DIR}/human_annotation.gtf.gz"
            )
        fi

        PARAMS_FILE="${REPO_ROOT}/pipelines/${PIPELINE}/params/${DATASET}.json"
        [[ -f "$PARAMS_FILE" ]] && PARAMS_ARGS+=(-params-file "$PARAMS_FILE")
        ;;

    differentialabundance)
        NF_PROFILE="rnaseq,${CONTAINER_PROFILE}"

        if [[ "$ENV" == "aws" ]]; then
            S3_DATA="s3://${S3_BUCKET}/data/${DATASET}"
            S3_RESULTS="s3://${S3_BUCKET}/results/rnaseq/${DATASET}"
            PATH_ARGS+=(
                "--input"     "${S3_DATA}/samplesheet_de.csv"
                "--contrasts" "${S3_DATA}/contrasts.csv"
                "--matrix"    "${S3_RESULTS}/star_salmon/salmon.merged.gene_counts.tsv"
                "--gtf"       "${S3_DATA}/human_annotation.gtf.gz"
            )
        else
            COUNT_MATRIX="${RESULTS_DIR}/rnaseq/${DATASET}/star_salmon/salmon.merged.gene_counts.tsv"
            if [[ ! -f "$COUNT_MATRIX" && "$DRY_RUN" == false ]]; then
                echo "ERROR: Count matrix not found: ${COUNT_MATRIX}"
                echo "Run rnaseq first: ./scripts/run.sh --pipeline rnaseq --dataset ${DATASET}"
                exit 1
            fi
            PATH_ARGS+=(
                "--input"     "${DATA_DIR}/samplesheet_de.csv"
                "--contrasts" "${DATA_DIR}/contrasts.csv"
                "--matrix"    "${COUNT_MATRIX}"
                "--gtf"       "${DATA_DIR}/human_annotation.gtf.gz"
            )
            LENGTH_MATRIX="${RESULTS_DIR}/rnaseq/${DATASET}/star_salmon/salmon.merged.gene_lengths.tsv"
            [[ -f "$LENGTH_MATRIX" ]] && PATH_ARGS+=("--transcript_length_matrix" "${LENGTH_MATRIX}")
        fi

        PARAMS_FILE="${REPO_ROOT}/pipelines/${PIPELINE}/params/${DATASET}.json"
        [[ -f "$PARAMS_FILE" ]] && PARAMS_ARGS+=(-params-file "$PARAMS_FILE")
        ;;

    *)
        echo "ERROR: Unknown pipeline: ${PIPELINE}"
        echo "Available: rnaseq, differentialabundance"
        exit 1
        ;;
esac

# ──────────────────────────────────────────────────────────────
# Output directory
# ──────────────────────────────────────────────────────────────
if [[ "$ENV" == "aws" ]]; then
    [[ -z "${S3_BUCKET:-}" ]] && { echo "ERROR: S3_BUCKET is required for --env aws"; exit 1; }
    OUTDIR="s3://${S3_BUCKET}/results/${PIPELINE}/${DATASET}"
else
    OUTDIR="${RESULTS_DIR}/${PIPELINE}/${DATASET}"
fi

# ──────────────────────────────────────────────────────────────
# Build and run Nextflow command
# ──────────────────────────────────────────────────────────────
NF_COMMAND=(
    "$NF_CMD" run "$NF_PIPELINE_NAME"
    -profile "$NF_PROFILE"
    -c       "$CONFIG_FILE"
    -r       "$PIPELINE_VERSION"
    "${PARAMS_ARGS[@]}"
    "${PATH_ARGS[@]}"
    --outdir "$OUTDIR"
)
[[ "$RESUME" == true ]] && NF_COMMAND+=(-resume)

echo "=================================================================="
echo "nf-core Pipeline Runner"
echo "=================================================================="
printf "Pipeline : %s v%s\n" "$NF_PIPELINE_NAME" "$PIPELINE_VERSION"
printf "Dataset  : %s\n"     "$DATASET"
printf "Env      : %s\n"     "$ENV"
printf "Profile  : %s\n"     "$NF_PROFILE"
printf "Outdir   : %s\n"     "$OUTDIR"
[[ "$RESUME"  == true ]] && echo "Resume   : yes"
[[ "$DRY_RUN" == true ]] && echo "Dry-run  : yes"
echo "=================================================================="
echo ""
echo "Command:"
printf "  %s\n" "${NF_COMMAND[*]}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Exiting."
    exit 0
fi

exec "${NF_COMMAND[@]}"
