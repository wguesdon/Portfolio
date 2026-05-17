#!/usr/bin/env python
"""Launch cuML jobs with custom RAPIDS ECR image.

RF on g4dn.2xlarge (32GB) with 2x OTE baked in.
SVM on g4dn.xlarge (16GB), no augmentation.
"""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

JOBS = [
    {"name": "rf", "script": "train_rf_ote_cuml.py", "instance": "ml.g4dn.2xlarge"},
    {"name": "svm", "script": "train_svm_ote_cuml.py", "instance": "ml.g4dn.xlarge"},
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
