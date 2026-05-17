#!/usr/bin/env python
"""Quick test: verify cuML image can save to /opt/ml/model/ on SageMaker."""
import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

estimator = Estimator(
    image_uri=cuml_image,
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path="s3://kaggle-ps6e4/output/cuml-test/",
    base_job_name="ps6e4-cuml-test",
    sagemaker_session=session,
    disable_profiler=True,
    max_run=600,
    environment={"SAGEMAKER_PROGRAM": "test_save.py"},
)
estimator.fit({"training": "s3://kaggle-ps6e4/ensemble-data/"}, wait=False)
print(f"Launched: {estimator.latest_training_job.name}")
