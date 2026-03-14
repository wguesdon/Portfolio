#!/bin/bash
#
# CIBMTR HCT — Pairwise Ranking NN Training Pipeline
# Trains PRL-NN on AWS SageMaker, downloads weights for Kaggle submission notebook.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITION_DIR="$(dirname "$SCRIPT_DIR")"
PORTFOLIO_ROOT="$(dirname "$(dirname "$COMPETITION_DIR")")"
AWS_SAGEMAKER="$PORTFOLIO_ROOT/aws/sagemaker"
CONFIG="$SCRIPT_DIR/config.yaml"

run_py() {
    (cd "$COMPETITION_DIR" && uv run python "$@")
}

echo "=================================================="
echo "CIBMTR HCT — PRL-NN AWS Pipeline"
echo "=================================================="
echo "Competition dir: $COMPETITION_DIR"
echo "Config:          $CONFIG"
echo ""

ACTION="${1:-help}"

case $ACTION in
    upload)
        echo "Uploading competition data to S3..."
        run_py "$AWS_SAGEMAKER/scripts/upload_data.py" --config "$CONFIG"
        ;;

    build)
        echo "Building and pushing NN container to ECR..."
        CONTAINER_DIR="$COMPETITION_DIR/containers/nn"
        ACCOUNT_ID="017787554638"
        REGION="us-east-1"
        REPO="kaggle-training/cibmtr-nn"
        ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO"

        # Login to ECR
        aws ecr get-login-password --region $REGION | \
            podman login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

        # Create repo if needed
        aws ecr describe-repositories --repository-names "$REPO" --region $REGION 2>/dev/null || \
            aws ecr create-repository --repository-name "$REPO" --region $REGION

        # Build and push
        podman build -t "$ECR_URI:latest" "$CONTAINER_DIR"
        podman push "$ECR_URI:latest"
        echo "Pushed: $ECR_URI:latest"
        ;;

    train)
        echo "Launching SageMaker training job..."
        run_py "$AWS_SAGEMAKER/scripts/launch_training.py" --config "$CONFIG"
        echo ""
        echo "Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs"
        ;;

    train-dry)
        echo "Dry run..."
        run_py "$AWS_SAGEMAKER/scripts/launch_training.py" --config "$CONFIG" --dry-run --save-json
        ;;

    download)
        echo "Downloading NN artifacts from S3..."
        run_py "$AWS_SAGEMAKER/scripts/download_artifacts.py" --config "$CONFIG"
        ;;

    status)
        echo "Checking training job status..."
        run_py "$AWS_SAGEMAKER/scripts/status.py" --config "$CONFIG"
        ;;

    package-kaggle)
        echo "Packaging NN weights for Kaggle dataset upload..."
        ARTIFACTS="$COMPETITION_DIR/artifacts/nn"
        KAGGLE_DS="$COMPETITION_DIR/kaggle_submission/nn_weights"
        mkdir -p "$KAGGLE_DS"

        # Copy model weights, metadata, and predictions
        for f in nn_fold*.pt nn_metadata.json scaler.pkl nn_test_rank.npy nn_test_cls.npy nn_oof_rank.npy nn_oof_cls.npy; do
            if [ -f "$ARTIFACTS/$f" ]; then
                cp "$ARTIFACTS/$f" "$KAGGLE_DS/"
                echo "  Copied $f"
            fi
        done

        # Create dataset-metadata.json
        cat > "$KAGGLE_DS/dataset-metadata.json" << 'DSMETA'
{
  "title": "CIBMTR HCT PRL-NN Weights",
  "id": "wguesdon/cibmtr-hct-nn-weights",
  "licenses": [{"name": "CC0-1.0"}]
}
DSMETA

        echo ""
        echo "Dataset ready at: $KAGGLE_DS"
        echo "Upload with: kaggle datasets create -p $KAGGLE_DS"
        ;;

    all)
        echo "Full pipeline: upload -> build -> train"
        echo ""
        $0 upload
        echo ""
        $0 build
        echo ""
        $0 train
        echo ""
        echo "After training completes:"
        echo "  $0 download"
        echo "  $0 package-kaggle"
        ;;

    *)
        echo "Usage: $0 [ACTION]"
        echo ""
        echo "Actions:"
        echo "  upload          Upload competition data to S3"
        echo "  build           Build + push NN Docker container to ECR"
        echo "  train           Launch SageMaker training job"
        echo "  train-dry       Dry run (show config, no launch)"
        echo "  download        Download trained artifacts from S3"
        echo "  status          Check training job status"
        echo "  package-kaggle  Package NN weights for Kaggle dataset upload"
        echo "  all             upload + build + train"
        echo ""
        echo "Full workflow:"
        echo "  $0 all               # Upload data, build container, launch training"
        echo "  $0 status            # Check progress"
        echo "  $0 download          # Get artifacts from S3"
        echo "  $0 package-kaggle    # Package for Kaggle"
        echo "  kaggle datasets create -p kaggle_submission/nn_weights"
        exit 1
        ;;
esac

echo ""
echo "Done!"
