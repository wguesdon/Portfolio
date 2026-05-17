#!/usr/bin/env python
"""Launch cuML LR (C=0.1 L2), LR (C=0.1 L1), and KNN (k=5 lite) SageMaker jobs."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

JOBS = [
    {"name": "lr-cuml", "script": "train_lr_ote_cuml.py", "instance": "ml.g4dn.xlarge"},
    {"name": "lr-l1-cuml", "script": "train_lr_l1_ote_cuml.py", "instance": "ml.g4dn.xlarge"},
    {"name": "knn5-lite", "script": "train_knn5_lite_cuml.py", "instance": "ml.g4dn.xlarge"},
]

for job in JOBS:
    estimator = Estimator(
        image_uri=cuml_image,
        role=role,
        instance_count=1,
        instance_type=job["instance"],
        output_path=f"s3://kaggle-ps6e4/output/{job['name']}-ote-cuml/",
        base_job_name=f"ps6e4-{job['name']}-ote-cuml",
        sagemaker_session=session,
        disable_profiler=True,
        environment={"SAGEMAKER_PROGRAM": job["script"]},
    )
    estimator.fit({"training": "s3://kaggle-ps6e4/ensemble-data/"}, wait=False)
    print(f"Launched: {estimator.latest_training_job.name} ({job['instance']})")
