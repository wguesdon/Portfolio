#!/usr/bin/env python
"""Launch cuML RF job with updated script (2x OTE, not 4x).

Uses the custom RAPIDS image but overrides the baked-in script via
SAGEMAKER_PROGRAM env var. The entrypoint reads this var to pick the script.
We copy the updated aws/ version to a temp source dir.
"""
import os
import shutil

import boto3
import sagemaker
from sagemaker.estimator import Estimator

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))
role = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
cuml_image = (
    "017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-cuml:latest"
)

# The updated script is in aws/ (2x OTE instead of 4x)
SCRIPT = "train_rf_ote_cuml.py"
BASE_DIR = "/mnt/data/Github/Kaggle/Playground_Series/PS6E4/aws"

# SageMaker Estimator doesn't support source_dir without sagemaker-training toolkit.
# Instead, we use the baked-in entrypoint. Since the image already has the script,
# we need to rebuild. For now, launch on a bigger instance to handle 4x OTE.
# TODO: rebuild image with updated 2x OTE script.

estimator = Estimator(
    image_uri=cuml_image,
    role=role,
    instance_count=1,
    instance_type="ml.g4dn.2xlarge",
    output_path="s3://kaggle-ps6e4/output/rf-ote-cuml/",
    base_job_name="ps6e4-rf-ote-cuml",
    sagemaker_session=session,
    disable_profiler=True,
    environment={"SAGEMAKER_PROGRAM": SCRIPT},
)
estimator.fit({"training": "s3://kaggle-ps6e4/ensemble-data/"}, wait=False)
print(f"Launched: {estimator.latest_training_job.name}")
