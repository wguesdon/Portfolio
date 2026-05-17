# SageMaker Training Guide for PS6E4

## Prerequisites

```bash
cd /mnt/data/Github/Kaggle
export $(grep -v '^#' .env | xargs)
```

## Launch training jobs

### With Python (recommended)

```bash
uv run python Playground_Series/PS6E4/scripts/launch_cuml_lgb_jobs.py
```

### With AWS CLI

```bash
# 1. Upload training script to S3
aws s3 cp Playground_Series/PS6E4/scripts/train_rf_ote_cuml.py \
  s3://kaggle-ps6e4/source/train_rf_ote_cuml.py

# 2. Create a training job
aws sagemaker create-training-job \
  --training-job-name "ps6e4-rf-ote-cuml-$(date +%Y-%m-%d-%H-%M-%S)" \
  --algorithm-specification \
    TrainingImage=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311,TrainingInputMode=File \
  --role-arn "arn:aws:iam::017787554638:role/SageMakerKaggleExecutionRole" \
  --input-data-config '[{
    "ChannelName": "training",
    "DataSource": {
      "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "s3://kaggle-ps6e4/ensemble-data/",
        "S3DataDistributionType": "FullyReplicated"
      }
    }
  }]' \
  --output-data-config '{"S3OutputPath": "s3://kaggle-ps6e4/output/rf-ote-cuml/"}' \
  --resource-config '{"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30}' \
  --stopping-condition '{"MaxRuntimeInSeconds": 43200}' \
  --hyper-parameters '{"sagemaker_program": "train_rf_ote_cuml.py", "sagemaker_submit_directory": "s3://kaggle-ps6e4/source/train_rf_ote_cuml.tar.gz"}' \
  --region us-east-1
```

Note: the Python SDK handles tar.gz packaging of the source directory automatically. With the CLI you must package and upload manually.

### Packaging source for CLI

```bash
# Package the script into a tar.gz
cd /tmp && mkdir -p sm_source
cp /mnt/data/Github/Kaggle/Playground_Series/PS6E4/scripts/train_rf_ote_cuml.py sm_source/
cd sm_source && tar -czf ../train_rf_ote_cuml.tar.gz . && cd ..
aws s3 cp train_rf_ote_cuml.tar.gz s3://kaggle-ps6e4/source/
```

## Monitor jobs

### List recent jobs

```bash
aws sagemaker list-training-jobs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 10 \
  --region us-east-1 \
  --query 'TrainingJobSummaries[?contains(TrainingJobName, `ps6e4`)].{Name:TrainingJobName,Status:TrainingJobStatus,Created:CreationTime}' \
  --output table
```

### Get job status and details

```bash
JOB_NAME="ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588"

aws sagemaker describe-training-job \
  --training-job-name "$JOB_NAME" \
  --region us-east-1 \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Instance:ResourceConfig.InstanceType,Runtime:TrainingTimeInSeconds,Failure:FailureReason}' \
  --output table
```

### Stream CloudWatch logs (live)

```bash
JOB_NAME="ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588"

# Get log stream name
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "$JOB_NAME" \
  --region us-east-1 \
  --query 'logStreams[*].logStreamName' \
  --output text

# Tail logs (follow mode)
aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix "$JOB_NAME" \
  --follow \
  --region us-east-1
```

### Monitor with Python (all jobs at once)

```bash
uv run python -c "
import boto3
from datetime import datetime, timezone
sm = boto3.client('sagemaker', region_name='us-east-1')
now = datetime.now(timezone.utc)
jobs = sm.list_training_jobs(MaxResults=15, SortBy='CreationTime', SortOrder='Descending')
for j in jobs['TrainingJobSummaries']:
    name = j['TrainingJobName']
    if 'ps6e4' not in name: continue
    status = j['TrainingJobStatus']
    detail = sm.describe_training_job(TrainingJobName=name)
    inst = detail['ResourceConfig']['InstanceType']
    elapsed = (now - detail['CreationTime']).total_seconds() / 3600
    if status == 'Completed': print(f'  DONE  {name}  {inst}  ({elapsed:.1f}h)')
    elif status == 'Failed': print(f'  FAIL  {name}  {detail.get(\"FailureReason\", \"\")[:80]}')
    elif status in ('InProgress',): print(f'  RUN   {name}  {inst}  ({elapsed:.1f}h)')
    else: print(f'  {status:5s} {name}')
"
```

### Check GPU/instance metrics

SageMaker publishes instance metrics to CloudWatch under the `/aws/sagemaker/TrainingJobs` namespace.

```bash
JOB_NAME="ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588"

# GPU utilization (g4dn instances)
aws cloudwatch get-metric-statistics \
  --namespace "/aws/sagemaker/TrainingJobs" \
  --metric-name "GPUUtilization" \
  --dimensions Name=Host,Value="$JOB_NAME/algo-1" \
  --start-time "$(date -u -d '2 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --period 300 \
  --statistics Average \
  --region us-east-1

# GPU memory utilization
aws cloudwatch get-metric-statistics \
  --namespace "/aws/sagemaker/TrainingJobs" \
  --metric-name "GPUMemoryUtilization" \
  --dimensions Name=Host,Value="$JOB_NAME/algo-1" \
  --start-time "$(date -u -d '2 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --period 300 \
  --statistics Average \
  --region us-east-1

# CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace "/aws/sagemaker/TrainingJobs" \
  --metric-name "CPUUtilization" \
  --dimensions Name=Host,Value="$JOB_NAME/algo-1" \
  --start-time "$(date -u -d '2 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --period 300 \
  --statistics Average \
  --region us-east-1

# Memory utilization
aws cloudwatch get-metric-statistics \
  --namespace "/aws/sagemaker/TrainingJobs" \
  --metric-name "MemoryUtilization" \
  --dimensions Name=Host,Value="$JOB_NAME/algo-1" \
  --start-time "$(date -u -d '2 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  --period 300 \
  --statistics Average \
  --region us-east-1
```

## Download artifacts

### Get the model artifact S3 path

```bash
JOB_NAME="ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588"

aws sagemaker describe-training-job \
  --training-job-name "$JOB_NAME" \
  --region us-east-1 \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text
```

This returns something like:
`s3://kaggle-ps6e4/output/rf-ote-cuml/ps6e4-rf-ote-cuml-.../output/model.tar.gz`

### Download and extract predictions

```bash
JOB_NAME="ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588"
PRED_DIR="/mnt/data/Github/Kaggle/Playground_Series/PS6E4/predictions"

# Get artifact URL
ARTIFACT=$(aws sagemaker describe-training-job \
  --training-job-name "$JOB_NAME" \
  --region us-east-1 \
  --query 'ModelArtifacts.S3ModelArtifacts' \
  --output text)

# Download and extract
aws s3 cp "$ARTIFACT" /tmp/model.tar.gz
tar -xzf /tmp/model.tar.gz -C /tmp/model_output/
cp /tmp/model_output/oof_*.npy "$PRED_DIR/"
cp /tmp/model_output/pred_*.npy "$PRED_DIR/"
```

### Batch download all completed jobs

```bash
PRED_DIR="/mnt/data/Github/Kaggle/Playground_Series/PS6E4/predictions"

for JOB_NAME in \
  "ps6e4-rf-ote-cuml-2026-04-10-23-20-15-588" \
  "ps6e4-svm-ote-cuml-2026-04-10-23-20-18-807" \
  "ps6e4-lgb-ote-shallow-2026-04-10-23-20-20-898" \
  "ps6e4-lgb-ote-deep-2026-04-10-23-20-22-952"
do
  echo "=== $JOB_NAME ==="
  ARTIFACT=$(aws sagemaker describe-training-job \
    --training-job-name "$JOB_NAME" \
    --region us-east-1 \
    --query 'ModelArtifacts.S3ModelArtifacts' \
    --output text 2>/dev/null)

  if [ -z "$ARTIFACT" ] || [ "$ARTIFACT" = "None" ]; then
    echo "  Not completed yet, skipping"
    continue
  fi

  TMPDIR=$(mktemp -d)
  aws s3 cp "$ARTIFACT" "$TMPDIR/model.tar.gz" --quiet
  tar -xzf "$TMPDIR/model.tar.gz" -C "$TMPDIR/"
  cp "$TMPDIR"/oof_*.npy "$PRED_DIR/" 2>/dev/null && echo "  Copied OOF predictions"
  cp "$TMPDIR"/pred_*.npy "$PRED_DIR/" 2>/dev/null && echo "  Copied test predictions"
  rm -rf "$TMPDIR"
done
```

## Instance types reference

| Instance | vCPUs | RAM | GPU | Use case |
|----------|-------|-----|-----|----------|
| ml.c5.9xlarge | 36 | 72 GB | none | GBDT models (XGB, LGB, CatBoost) |
| ml.g4dn.xlarge | 4 | 16 GB | 1x T4 (16GB) | cuML models, PyTorch (GNN, RealMLP) |

## Container images used

| Image | Contents |
|-------|----------|
| `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311` | PyTorch 2.5.1, CUDA 12, Python 3.11 |
| `763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.2-gpu-py311` | AutoGluon 1.2, XGBoost, LightGBM, CatBoost, Python 3.11 |

## Stop a running job

```bash
aws sagemaker stop-training-job \
  --training-job-name "$JOB_NAME" \
  --region us-east-1
```

## Cost estimation

Pricing (us-east-1, on-demand):
- ml.c5.9xlarge: ~$1.53/hr
- ml.g4dn.xlarge: ~$0.526/hr
- Spot instances (50-70% savings) can be enabled via `use_spot_instances=True` in the Python SDK
