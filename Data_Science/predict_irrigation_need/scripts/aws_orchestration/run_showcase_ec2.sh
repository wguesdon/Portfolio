#!/bin/bash
#
# Run the showcase pipeline on an EC2 GPU spot instance.
# Uploads data + script to S3, launches instance, polls for completion,
# downloads predictions and submission.
#
# Usage: ./run_showcase_ec2.sh [--on-demand] [instance-type]
#   Default instance: g4dn.xlarge (T4 GPU)
#   --on-demand: Use on-demand pricing (default: spot)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP_DIR="$(dirname "$SCRIPT_DIR")"
REGION="us-east-1"
S3_BUCKET="kaggle-ps6e4"
S3_PREFIX="showcase"
PROFILE_NAME="EC2DockerBuildProfile"
USE_SPOT=true

# Parse args
INSTANCE_TYPE="g4dn.xlarge"
for arg in "$@"; do
    case "$arg" in
        --on-demand) USE_SPOT=false ;;
        *) INSTANCE_TYPE="$arg" ;;
    esac
done

PRICING="on-demand"
$USE_SPOT && PRICING="spot"

echo "=================================================="
echo "Showcase Pipeline - EC2 GPU Runner"
echo "=================================================="
echo "Instance: $INSTANCE_TYPE ($PRICING)"
echo ""

# Step 1: Upload data and script to S3
echo "Uploading to S3..."
for f in train.csv test.csv irrigation_prediction.csv sample_submission.csv; do
    aws s3 cp "$COMP_DIR/data/raw/$f" "s3://$S3_BUCKET/$S3_PREFIX/$f" --region "$REGION" --quiet
done
aws s3 cp "$SCRIPT_DIR/showcase_pipeline.py" "s3://$S3_BUCKET/$S3_PREFIX/showcase_pipeline.py" --region "$REGION" --quiet
echo "Done."

# Step 2: Get Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text 2>/dev/null)

if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
    AMI_ID=$(aws ssm get-parameters \
        --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
        --region "$REGION" --query 'Parameters[0].Value' --output text)
fi
echo "AMI: $AMI_ID"

# Step 3: Write user-data to temp file
USERDATA_FILE=$(mktemp)
cat > "$USERDATA_FILE" <<OUTEREOF
#!/bin/bash
set -ex
exec > /var/log/showcase.log 2>&1

TOKEN=\$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
INSTANCE_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
AZ=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/placement/availability-zone)
REGION_META=\${AZ%?}

tag_status() { aws ec2 create-tags --resources "\$INSTANCE_ID" --tags "Key=ShowcaseStatus,Value=\$1" --region "\$REGION_META"; }
tag_status "setting-up"

# Install Python env
if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-pip python3-venv
fi
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --quiet xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn scipy feature-engine

# Download data
WORK=/opt/showcase
mkdir -p \$WORK/data/raw \$WORK/predictions \$WORK/submissions \$WORK/output/eda
cd \$WORK
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/train.csv data/raw/ --region ${REGION} --quiet
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/test.csv data/raw/ --region ${REGION} --quiet
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/irrigation_prediction.csv data/raw/ --region ${REGION} --quiet
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/sample_submission.csv data/raw/ --region ${REGION} --quiet
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/showcase_pipeline.py . --region ${REGION} --quiet

tag_status "training"

# Background log uploader: push logs to S3 every 60s so we can monitor remotely
(while true; do
    sleep 60
    aws s3 cp \$WORK/output.log s3://${S3_BUCKET}/${S3_PREFIX}/output.log --region ${REGION} --quiet 2>/dev/null
done) &
LOG_UPLOADER_PID=\$!

# Run pipeline (unbuffered output via PYTHONUNBUFFERED)
PYTHONUNBUFFERED=1 python showcase_pipeline.py 2>&1 | tee output.log
EXIT_CODE=\${PIPESTATUS[0]}

# Stop log uploader
kill \$LOG_UPLOADER_PID 2>/dev/null

# Upload results
aws s3 sync predictions/ s3://${S3_BUCKET}/${S3_PREFIX}/predictions/ --region ${REGION} --quiet
aws s3 sync submissions/ s3://${S3_BUCKET}/${S3_PREFIX}/submissions/ --region ${REGION} --quiet
aws s3 sync output/ s3://${S3_BUCKET}/${S3_PREFIX}/output/ --region ${REGION} --quiet
aws s3 cp output.log s3://${S3_BUCKET}/${S3_PREFIX}/output.log --region ${REGION} --quiet

if [ \$EXIT_CODE -eq 0 ]; then tag_status "complete"; else tag_status "failed"; fi

aws ec2 terminate-instances --instance-ids "\$INSTANCE_ID" --region "\$REGION_META"
OUTEREOF

USER_DATA_B64=$(base64 -w 0 "$USERDATA_FILE")
rm "$USERDATA_FILE"

# Step 4: Launch spot instance
echo "Launching..."
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)

SPOT_ARGS=""
if $USE_SPOT; then
    SPOT_ARGS='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$PROFILE_NAME" \
    $SPOT_ARGS \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3"}}]' \
    --user-data "$USER_DATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=showcase-runner},{Key=ShowcaseStatus,Value=launching}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "Instance: $INSTANCE_ID"
echo ""
echo "Monitor live logs:"
echo "  aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/output.log - | tail -5"
echo "  aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo ""

# Step 5: Poll
START_TIME=$(date +%s)
while true; do
    STATUS=$(aws ec2 describe-tags \
        --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=ShowcaseStatus" \
        --region "$REGION" --query 'Tags[0].Value' --output text 2>/dev/null || echo "unknown")

    ELAPSED=$(( $(date +%s) - START_TIME ))
    MINS=$(( ELAPSED / 60 ))
    SECS=$(( ELAPSED % 60 ))

    case "$STATUS" in
        complete)
            printf "\n\nCompleted in %dm%02ds!\n" "$MINS" "$SECS"
            break ;;
        failed)
            printf "\n\nFailed after %dm%02ds. Check: aws s3 cp s3://%s/%s/output.log .\n" "$MINS" "$SECS" "$S3_BUCKET" "$S3_PREFIX"
            exit 1 ;;
        None|unknown)
            INST_STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
                --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "unknown")
            [ "$INST_STATE" = "terminated" ] && { echo ""; echo "Instance terminated."; break; } ;;
    esac
    # Fetch latest log line from S3
    LAST_LOG=$(aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/output.log" - --region "$REGION" 2>/dev/null | tail -1 | cut -c1-80)
    printf "\r  [%dm%02ds] %-12s %s\033[K" "$MINS" "$SECS" "$STATUS" "$LAST_LOG"
    sleep 30
done

# Step 6: Download results
echo "Downloading results..."
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/predictions/" "$COMP_DIR/predictions/" --region "$REGION" --quiet
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/submissions/" "$COMP_DIR/submissions/" --region "$REGION" --quiet
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/output.log" "$COMP_DIR/output/showcase_output.log" --region "$REGION" --quiet

echo ""
echo "=================================================="
echo "Done! Results in:"
echo "  predictions/  submissions/  output/showcase_output.log"
echo "=================================================="
