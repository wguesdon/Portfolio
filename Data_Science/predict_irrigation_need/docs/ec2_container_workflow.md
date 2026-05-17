# EC2 Jumpbox Container Workflow

How we build Docker images and push them to Amazon ECR for SageMaker training.

## Why an EC2 jumpbox

Pushing multi-GB CUDA images from a local machine to ECR takes 25+ minutes over home internet. An EC2 instance in the same AWS region pushes in under a minute. The jumpbox is a temporary spot instance that builds the images, pushes them, and self-terminates.

| Step | Local | EC2 jumpbox |
|------|-------|-------------|
| Build 3 images | ~5 min | ~5 min |
| Push to ECR | 25+ min | <1 min |
| Total | ~30 min | ~9 min |
| Cost | $0 | ~$0.01 (spot) |

## Architecture

```
Local machine                   AWS (us-east-1)
+------------------+            +------------------+
| 1. Package build |  S3 upload | S3 bucket        |
|    context       | ---------> | context.tar.gz   |
+------------------+            +--------+---------+
                                         |
                                         v
                                +------------------+
                                | 2. EC2 spot      |
                                |    c5.xlarge     |
                                |    - pulls from S3|
                                |    - docker build |
                                |    - docker push  |
                                +--------+---------+
                                         |
                                         v
                                +------------------+
                                | 3. ECR repos     |
                                |    kaggle-ps6e4- |
                                |      xgb/lgb/cat |
                                +--------+---------+
                                         |
                                         v
                                +------------------+
                                | 4. SageMaker     |
                                |    training jobs  |
                                +------------------+
```

## Prerequisites

- AWS CLI configured with credentials (in `.env` at repo root)
- The `ec2_build_push.sh` script at `aws_training/scripts/ec2_build_push.sh`
- Dockerfiles in the competition `aws/` directory

```bash
cd /mnt/data/Github/Kaggle
export $(grep -v '^#' .env | xargs)
```

## Quick start

### Build and push all images

```bash
./aws_training/scripts/ec2_build_push.sh Playground_Series/PS6E4/aws
```

This detects all `Dockerfile.*` files in the directory and builds them all.

### Build specific images only

```bash
./aws_training/scripts/ec2_build_push.sh Playground_Series/PS6E4/aws xgb lgb
```

## What the script does

### Step 1: Create IAM role (first run only)

Creates `EC2DockerBuildRole` and `EC2DockerBuildProfile` with these permissions:
- ECR: push, pull, create repositories
- S3: read objects (to fetch build context)
- EC2: create tags, terminate self
- SSM: session manager (for optional log viewing)

### Step 2: Package and upload build context

```bash
# What the script does internally:
tar -czf /tmp/docker-build-context.tar.gz -C Playground_Series/PS6E4/aws .
aws s3 cp /tmp/docker-build-context.tar.gz s3://kaggle-ps6e4/docker-build/context.tar.gz
```

The tarball includes all Dockerfiles, training scripts, config files, and any other files in the `aws/` directory.

### Step 3: Launch EC2 spot instance

- Instance type: c5.xlarge (4 vCPU, 8 GB RAM). Override with `EC2_BUILD_INSTANCE=c5.2xlarge`.
- Volume: 50 GB gp3 EBS. Override with `EC2_BUILD_VOLUME=80`.
- AMI: latest Amazon Linux 2023
- Market: spot (one-time, no bidding needed)

The instance receives a cloud-init user-data script that runs on boot:
1. Installs Docker
2. Downloads and extracts the build context from S3
3. Logs into ECR
4. Builds each `Dockerfile.<name>` and pushes to `<account>.dkr.ecr.<region>.amazonaws.com/kaggle-ps6e4-<name>:latest`
5. Tags the instance with `BuildStatus=complete` or `BuildStatus=failed:<images>`
6. Self-terminates

### Step 4: Poll for completion

The script polls the `BuildStatus` EC2 tag every 15 seconds and reports elapsed time. Output looks like:

```
  [3m45s] Status: building
  [8m12s] Status: complete
All images built and pushed in 8m12s!
```

### Step 5: Cleanup

The S3 staging tarball is removed. The EC2 instance terminates itself.

## Dockerfile conventions

Dockerfiles live in `Playground_Series/PS6E4/aws/` and follow the naming pattern `Dockerfile.<model>`.

### Current Dockerfiles

| File | Base image | Contents |
|------|-----------|----------|
| `Dockerfile.xgb` | nvidia/cuda:12.1.0-runtime-ubuntu22.04 | XGBoost, Optuna, tabml |
| `Dockerfile.lgb` | python:3.10-slim | LightGBM |
| `Dockerfile.cat` | nvidia/cuda:12.1.0-runtime-ubuntu22.04 | CatBoost |
| `Dockerfile.nn` | pytorch base | PyTorch neural networks |
| `Dockerfile.autogluon` | autogluon base | AutoGluon |
| `Dockerfile.showcase_v2` | kaggle GPU image | XGBoost, LGB, CatBoost |

### Writing a new Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev build-essential git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /opt/ml/code

# Bootstrap pip (ensurepip is disabled on Ubuntu)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install dependencies
RUN pip install --no-cache-dir \
    scikit-learn>=1.4.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    your-package-here

# Copy training scripts
COPY train_mymodel.py ./
COPY config.yaml ./

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "train_mymodel.py"]
```

Key points:
- Use `WORKDIR /opt/ml/code` for SageMaker compatibility
- Set `PYTHONUNBUFFERED=1` so logs stream in real time
- The ENTRYPOINT should run your training script directly
- COPY only the files you need (not the whole repo)

## Manual workflow (without the script)

If you prefer doing it step by step or need to debug.

### Build locally with Podman

```bash
ACCOUNT_ID="017787554638"
REGION="us-east-1"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
IMAGE="kaggle-ps6e4-xgb"

cd Playground_Series/PS6E4/aws

# Build
podman build -f Dockerfile.xgb -t "$IMAGE" .

# Create ECR repo (first time only)
aws ecr describe-repositories --repository-names "$IMAGE" --region "$REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$IMAGE" --region "$REGION"

# Login to ECR
aws ecr get-login-password --region "$REGION" | \
    podman login --username AWS --password-stdin "$ECR_URI"

# Tag and push
podman tag "$IMAGE" "$ECR_URI/$IMAGE:latest"
podman push "$ECR_URI/$IMAGE:latest"
```

### Create ECR repository with AWS CLI

```bash
aws ecr create-repository \
    --repository-name kaggle-ps6e4-mymodel \
    --region us-east-1 \
    --image-scanning-configuration scanOnPush=false
```

### List ECR repositories

```bash
aws ecr describe-repositories \
    --region us-east-1 \
    --query 'repositories[?contains(repositoryName, `kaggle-ps6e4`)].repositoryName' \
    --output table
```

### Check when an image was last pushed

```bash
aws ecr describe-images \
    --repository-name kaggle-ps6e4-xgb \
    --region us-east-1 \
    --query 'imageDetails[0].{Pushed:imagePushedAt,Size:imageSizeInBytes,Tags:imageTags}' \
    --output table
```

### Delete an ECR image

```bash
aws ecr batch-delete-image \
    --repository-name kaggle-ps6e4-xgb \
    --image-ids imageTag=latest \
    --region us-east-1
```

### Delete an ECR repository

```bash
aws ecr delete-repository \
    --repository-name kaggle-ps6e4-mymodel \
    --region us-east-1 \
    --force
```

## Monitoring a jumpbox build

### Check build status tag

```bash
INSTANCE_ID="i-0abc123def456"

aws ec2 describe-tags \
    --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=BuildStatus" \
    --region us-east-1 \
    --query 'Tags[0].Value' \
    --output text
```

### Connect via SSM to view live logs

```bash
aws ssm start-session --target "$INSTANCE_ID" --region us-east-1
# Once connected:
tail -f /var/log/docker-build.log
```

### Find running jumpbox instances

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=docker-build-jumpbox" "Name=instance-state-name,Values=running" \
    --region us-east-1 \
    --query 'Reservations[].Instances[].{ID:InstanceId,Status:State.Name,Launched:LaunchTime}' \
    --output table
```

### Manually terminate a stuck jumpbox

```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-east-1
```

## Using custom images with SageMaker

Once pushed to ECR, reference the image in your SageMaker training job.

### With Python SDK

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_mymodel.py",
    source_dir="/tmp/sm_source/",
    image_uri="017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-xgb:latest",
    role="arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path="s3://kaggle-ps6e4/output/mymodel/",
    sagemaker_session=session,
)
estimator.fit({"training": "s3://kaggle-ps6e4/ensemble-data/"})
```

### With AWS CLI

```bash
aws sagemaker create-training-job \
    --training-job-name "ps6e4-mymodel-$(date +%Y%m%d-%H%M%S)" \
    --algorithm-specification \
        TrainingImage=017787554638.dkr.ecr.us-east-1.amazonaws.com/kaggle-ps6e4-xgb:latest,TrainingInputMode=File \
    --role-arn "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole" \
    --resource-config '{"InstanceType":"ml.g4dn.xlarge","InstanceCount":1,"VolumeSizeInGB":30}' \
    --input-data-config '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"s3://kaggle-ps6e4/ensemble-data/","S3DataDistributionType":"FullyReplicated"}}}]' \
    --output-data-config '{"S3OutputPath":"s3://kaggle-ps6e4/output/mymodel/"}' \
    --stopping-condition '{"MaxRuntimeInSeconds":43200}' \
    --region us-east-1
```

## Vs. AWS managed DLC images

Not all models need a custom container. AWS provides Deep Learning Containers (DLCs) with common frameworks pre-installed.

| Approach | When to use | Example |
|----------|-------------|---------|
| AWS DLC | Standard frameworks, pip install at runtime | cuML models, LGB/XGB with OTE |
| Custom ECR image | Custom dependencies, reproducibility, large installs | tabml, Optuna pipelines |

Current DLCs in use:
- `pytorch-training:2.5.1-gpu-py311` for GPU models (cuML, GNN, RealMLP)
- `autogluon-training:1.2-gpu-py311` for GBDT models (has XGBoost, LGB, CatBoost pre-installed)

## Configuration files

### Global: `aws_training/config/aws_config.yaml`

```yaml
aws:
  region: "us-east-1"
  account_id: "017787554638"
sagemaker:
  role_arn: "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole"
ecr:
  repository_prefix: "kaggle-training"
```

### Competition: `Playground_Series/PS6E4/aws/config.yaml`

```yaml
competition:
  s3_bucket: "kaggle-ps6e4"
  ecr_prefix: "kaggle-ps6e4"
```

The `ec2_build_push.sh` script reads both configs to determine ECR repo names and S3 staging bucket.

## Overriding defaults

```bash
# Bigger instance for faster builds
EC2_BUILD_INSTANCE=c5.2xlarge ./aws_training/scripts/ec2_build_push.sh ...

# More disk for large images
EC2_BUILD_VOLUME=80 ./aws_training/scripts/ec2_build_push.sh ...
```

## Orchestrator: run.sh

The `Playground_Series/PS6E4/aws/run.sh` script ties everything together.

```bash
cd /mnt/data/Github/Kaggle/Playground_Series/PS6E4

# Full pipeline: build images, upload data, launch training
./aws/run.sh all

# Individual steps
./aws/run.sh build     # Build and push Docker images (uses Podman locally)
./aws/run.sh upload    # Upload train/test data to S3
./aws/run.sh train     # Launch SageMaker jobs
./aws/run.sh status    # Check job status
./aws/run.sh sync      # Download predictions from S3
```

Note: `run.sh build` uses Podman locally (slow push). For faster pushes, use the EC2 jumpbox script instead.

## Troubleshooting

### Build fails with "no space left on device"

Increase the EBS volume: `EC2_BUILD_VOLUME=80 ./ec2_build_push.sh ...`

### Instance terminates before build completes

Check if it was a spot interruption. Look at the instance state reason:

```bash
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region us-east-1 \
    --query 'Reservations[0].Instances[0].StateReason.Message' --output text
```

### ECR login fails on the jumpbox

The IAM role might not have propagated yet. The script waits 10 seconds, but IAM propagation can take up to 30 seconds in rare cases. Re-run the script.

### Cannot connect via SSM

The SSM agent needs time to register after boot. Wait 2 minutes after launch, then retry.
