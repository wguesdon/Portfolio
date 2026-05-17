#!/usr/bin/env python
"""Launch ExtraTrees, KNN, and Logistic Regression OTE training jobs on SageMaker.

All three are CPU-only models running on c5.9xlarge (36 vCPUs, 72 GB RAM).
Uses the AutoGluon DLC image (has sklearn, pandas, numpy).

Usage:
    cd /mnt/data/Github/Kaggle
    export $(grep -v '^#' .env | xargs)
    uv run python Playground_Series/PS6E4/scripts/launch_cpu_models.py
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.2-gpu-py311"
INSTANCE_TYPE = "ml.c5.9xlarge"
S3_DATA = "s3://kaggle-ps6e4/ensemble-data/"
SCRIPTS_DIR = "Playground_Series/PS6E4/scripts"

JOBS = [
    {
        "entry_point": "train_extratrees_ote.py",
        "base_job_name": "ps6e4-et-ote",
        "output_path": "s3://kaggle-ps6e4/output/et-ote/",
        "max_run": 14400,  # 4h should be plenty for ExtraTrees
    },
    {
        "entry_point": "train_knn_ote.py",
        "base_job_name": "ps6e4-knn-ote",
        "output_path": "s3://kaggle-ps6e4/output/knn-ote/",
        "max_run": 28800,  # 8h for KNN (slow on 630K rows)
    },
    {
        "entry_point": "train_lr_ote.py",
        "base_job_name": "ps6e4-lr-ote",
        "output_path": "s3://kaggle-ps6e4/output/lr-ote/",
        "max_run": 7200,  # 2h, LR is fast
    },
]


def main():
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name="us-east-1")
    )

    for job in JOBS:
        print(f"\nLaunching {job['entry_point']}...")
        estimator = PyTorch(
            entry_point=job["entry_point"],
            source_dir=SCRIPTS_DIR,
            image_uri=IMAGE_URI,
            role=ROLE,
            instance_count=1,
            instance_type=INSTANCE_TYPE,
            output_path=job["output_path"],
            base_job_name=job["base_job_name"],
            use_spot_instances=True,
            max_run=job["max_run"],
            max_wait=job["max_run"] * 2,
            sagemaker_session=session,
        )
        estimator.fit({"training": S3_DATA}, wait=False)
        print(f"  Job: {estimator.latest_training_job.name}")

    print("\nAll 3 jobs launched. Monitor with:")
    print("  uv run python -c \"... (see SESSION_RESUME.md)\"")


if __name__ == "__main__":
    main()
