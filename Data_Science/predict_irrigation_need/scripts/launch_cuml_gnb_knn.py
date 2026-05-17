#!/usr/bin/env python
"""Launch cuML GaussianNB and KNN SageMaker training jobs for PS6E4."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

JOBS = [
    {"name": "gnb", "script": "train_gnb_ote_cuml.py", "instance": "ml.g4dn.xlarge"},
    {"name": "knn15", "script": "train_knn_ote_cuml.py", "instance": "ml.g4dn.xlarge"},
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
