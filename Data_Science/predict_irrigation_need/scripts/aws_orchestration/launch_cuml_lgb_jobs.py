#!/usr/bin/env python
"""Launch 4 SageMaker training jobs for PS6E4 new diversity models.

Jobs:
  - cuML RandomForest + OTE on g4dn.xlarge (GPU)
  - cuML SVM + OTE on g4dn.xlarge (GPU)
  - LightGBM shallow (depth=2) + OTE on c5.9xlarge (CPU)
  - LightGBM deep (depth=12) + OTE on c5.9xlarge (CPU)

Usage:
    cd /mnt/data/Github/Kaggle
    export $(grep -v '^#' .env | xargs)
    uv run python Playground_Series/PS6E4/scripts/launch_cuml_lgb_jobs.py
"""
import os
import shutil

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__))
ROLE = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
DATA_URI = "s3://kaggle-ps6e4/ensemble-data/"
PYTORCH_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311"
AUTOGLUON_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.2-gpu-py311"


def main():
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name="us-east-1")
    )

    jobs = [
        {
            "entry_point": "train_rf_ote_cuml.py",
            "base_job_name": "ps6e4-rf-ote-cuml",
            "instance_type": "ml.g4dn.xlarge",
            "image_uri": PYTORCH_IMAGE,
        },
        {
            "entry_point": "train_svm_ote_cuml.py",
            "base_job_name": "ps6e4-svm-ote-cuml",
            "instance_type": "ml.g4dn.xlarge",
            "image_uri": PYTORCH_IMAGE,
        },
        {
            "entry_point": "train_lgb_ote_shallow.py",
            "base_job_name": "ps6e4-lgb-ote-shallow",
            "instance_type": "ml.c5.9xlarge",
            "image_uri": AUTOGLUON_IMAGE,
        },
        {
            "entry_point": "train_lgb_ote_deep.py",
            "base_job_name": "ps6e4-lgb-ote-deep",
            "instance_type": "ml.c5.9xlarge",
            "image_uri": AUTOGLUON_IMAGE,
        },
    ]

    for job in jobs:
        src_dir = f"/tmp/sm_{job['base_job_name']}"
        os.makedirs(src_dir, exist_ok=True)
        shutil.copy(
            os.path.join(SCRIPTS_DIR, job["entry_point"]),
            os.path.join(src_dir, job["entry_point"]),
        )

        estimator = PyTorch(
            entry_point=job["entry_point"],
            source_dir=src_dir,
            image_uri=job["image_uri"],
            role=ROLE,
            instance_count=1,
            instance_type=job["instance_type"],
            output_path=f"s3://kaggle-ps6e4/output/{job['base_job_name']}/",
            base_job_name=job["base_job_name"],
            use_spot_instances=False,
            max_run=43200,
            sagemaker_session=session,
        )
        estimator.fit({"training": DATA_URI}, wait=False)
        print(f"Launched: {estimator.latest_training_job.name}  on {job['instance_type']}")


if __name__ == "__main__":
    main()
