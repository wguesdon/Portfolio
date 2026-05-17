#!/usr/bin/env python
"""Launch ensemble v17 on SageMaker (45 models: v15's 43 + xgb_ote_shallow CPU + cat_ote_deep)."""
import os
import shutil

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

SCRIPTS_DIR = os.path.dirname(__file__)
ROLE = "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
DATA_URI = "s3://kaggle-ps6e4/ensemble-data/"
AUTOGLUON_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.2-gpu-py311"
)

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))

src_dir = "/tmp/sm_ensemble_v17"
if os.path.exists(src_dir):
    shutil.rmtree(src_dir)
os.makedirs(src_dir)
shutil.copy(os.path.join(SCRIPTS_DIR, "train_ensemble.py"), src_dir)
shutil.copy(os.path.join(SCRIPTS_DIR, "ensemble_v4.py"), src_dir)

estimator = PyTorch(
    entry_point="train_ensemble.py",
    source_dir=src_dir,
    image_uri=AUTOGLUON_IMAGE,
    role=ROLE,
    instance_count=1,
    instance_type="ml.c5.9xlarge",
    output_path="s3://kaggle-ps6e4/output/ensemble-v17/",
    base_job_name="ps6e4-ensemble-v17",
    sagemaker_session=session,
    disable_profiler=True,
)
estimator.fit({"training": DATA_URI}, wait=False)
print(f"Launched: {estimator.latest_training_job.name}")
