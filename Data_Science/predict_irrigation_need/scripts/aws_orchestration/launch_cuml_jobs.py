#!/usr/bin/env python
"""Launch cuML RF and SVM SageMaker training jobs for PS6E4.

Uses the custom RAPIDS Docker image from ECR (kaggle-ps6e4-cuml).
Training scripts are baked into the image, no source_dir needed.
"""
import boto3

import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

SCRIPTS = {
    "rf": "train_rf_ote_cuml.py",
    "svm": "train_svm_ote_cuml.py",
}

for name, script in SCRIPTS.items():
    estimator = Estimator(
        image_uri=cuml_image,
        role=role,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        output_path=f"s3://kaggle-ps6e4/output/{name}-ote-cuml/",
        base_job_name=f"ps6e4-{name}-ote-cuml",
        sagemaker_session=session,
        disable_profiler=True,
        environment={"SAGEMAKER_PROGRAM": script},
    )
    estimator.fit({"training": "s3://kaggle-ps6e4/ensemble-data/"}, wait=False)
    print(f"Launched: {estimator.latest_training_job.name}")
