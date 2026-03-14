#!/bin/bash
#
# Obesity Risk Prediction (PS4E2) full training pipeline.
# Runs: upload data -> launch training -> download artifacts -> train ensemble -> submit
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITION_DIR="$(dirname "$SCRIPT_DIR")"
PORTFOLIO_ROOT="$(dirname "$(dirname "$COMPETITION_DIR")")"
AWS_SAGEMAKER="$PORTFOLIO_ROOT/aws/sagemaker"
CONFIG="$SCRIPT_DIR/config.yaml"

# Run Python scripts using uv (from competition root for path resolution)
run_py() {
    (cd "$COMPETITION_DIR" && uv run python "$@")
}

echo "=================================================="
echo "PS4E2 Obesity Risk Prediction — AWS Pipeline"
echo "=================================================="
echo "Competition dir: $COMPETITION_DIR"
echo "Config:          $CONFIG"
echo "Shared scripts:  $AWS_SAGEMAKER/scripts/"
echo ""

# Parse command line
ACTION="${1:-all}"

case $ACTION in
    upload)
        echo "Uploading data to S3..."
        run_py "$AWS_SAGEMAKER/scripts/upload_data.py" --config "$CONFIG"
        ;;

    train)
        echo "Launching SageMaker training jobs (AWS CLI method)..."
        run_py "$AWS_SAGEMAKER/scripts/launch_training.py" --config "$CONFIG"
        echo ""
        echo "Jobs launched! Monitor at:"
        echo "  https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs"
        echo ""
        echo "Or run: $0 status"
        ;;

    train-dry)
        echo "Dry run — showing job configs without launching..."
        run_py "$AWS_SAGEMAKER/scripts/launch_training.py" --config "$CONFIG" --dry-run --save-json
        ;;

    download)
        echo "Downloading artifacts from S3..."
        run_py "$AWS_SAGEMAKER/scripts/download_artifacts.py" --config "$CONFIG"
        ;;

    ensemble)
        echo "Training ensemble from OOF predictions..."
        run_py "$AWS_SAGEMAKER/scripts/train_ensemble.py" --config "$CONFIG"
        ;;

    submit)
        echo "Creating Kaggle submission file..."
        cd "$COMPETITION_DIR"
        uv run python << PYEOF
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

comp_dir = Path("$COMPETITION_DIR")
submissions_dir = comp_dir / "submissions"
submissions_dir.mkdir(exist_ok=True)

# Classes in label-encoded order (0-6)
CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# Find predictions — try ensemble first, then individual models
possible_paths = [
    comp_dir / "ensemble" / "ensemble_test.npy",
    comp_dir / "artifacts" / "ensemble" / "ensemble_test.npy",
]

test_probs = None
for path in possible_paths:
    if path.exists():
        test_probs = np.load(path)
        print(f"Loaded ensemble predictions from: {path}")
        break

if test_probs is None:
    # Fall back to individual model
    for model in ["cat", "lgb", "xgb"]:
        for loc in ["artifacts", "aws_output"]:
            path = comp_dir / loc / model / f"{model}_test.npy"
            if path.exists():
                test_probs = np.load(path)
                print(f"Loaded {model} predictions from: {path} (single model)")
                break
        if test_probs is not None:
            break

if test_probs is None:
    print("ERROR: No test predictions found. Run 'download' and 'ensemble' first.")
    exit(1)

print(f"Predictions shape: {test_probs.shape}")

# Decode probabilities to class labels
if test_probs.ndim == 2 and test_probs.shape[1] == 7:
    pred_classes = np.argmax(test_probs, axis=1)
    pred_labels = [CLASSES[i] for i in pred_classes]
else:
    print(f"WARNING: Unexpected prediction shape {test_probs.shape}")
    pred_labels = test_probs

# Load test IDs
test_df = pd.read_csv(comp_dir / "data" / "test.csv")
print(f"Test samples: {len(test_df)}")

# Create submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "NObeyesdad": pred_labels,
})

# Save with timestamp and as latest
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = submissions_dir / f"submission_aws_{timestamp}.csv"
submission.to_csv(output_path, index=False)
print(f"Saved: {output_path}")

latest_path = submissions_dir / "submission_aws_latest.csv"
submission.to_csv(latest_path, index=False)
print(f"Saved: {latest_path}")

print(f"\nClass distribution:")
print(submission["NObeyesdad"].value_counts())
print(f"\nSample:")
print(submission.head())
PYEOF
        ;;

    status)
        echo "Checking SageMaker job status..."
        run_py "$AWS_SAGEMAKER/scripts/status.py" --config "$CONFIG"
        ;;

    all)
        echo "Running full pipeline: upload -> train"
        echo ""
        echo "Step 1: Upload data to S3"
        run_py "$AWS_SAGEMAKER/scripts/upload_data.py" --config "$CONFIG"
        echo ""
        echo "Step 2: Launch training jobs"
        run_py "$AWS_SAGEMAKER/scripts/launch_training.py" --config "$CONFIG"
        echo ""
        echo "Training jobs launched. Estimated time: 1-3 hours."
        echo "Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs"
        echo ""
        echo "After training completes, run:"
        echo "  $0 download   # Fetch artifacts from S3"
        echo "  $0 ensemble   # Find best ensemble weights"
        echo "  $0 submit     # Create submission CSV"
        ;;

    *)
        echo "Usage: $0 [ACTION]"
        echo ""
        echo "Actions:"
        echo "  upload     Upload data + config to S3"
        echo "  train      Launch SageMaker training jobs (boto3 low-level)"
        echo "  train-dry  Dry run: print job configs + save JSON (no launch)"
        echo "  download   Download model artifacts from S3 after training"
        echo "  ensemble   Blend OOF predictions, find optimal weights"
        echo "  submit     Create Kaggle submission CSV"
        echo "  status     Check training job status"
        echo "  all        Upload + launch training (default)"
        echo ""
        echo "Full workflow:"
        echo "  $0 all             # Upload data, launch training jobs"
        echo "  $0 status          # Check progress (repeat until complete)"
        echo "  $0 download        # Get artifacts from S3"
        echo "  $0 ensemble        # Blend predictions"
        echo "  $0 submit          # Create submission.csv"
        exit 1
        ;;
esac

echo ""
echo "Done!"
