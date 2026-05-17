# vast.ai and RunPod Workflow for Kaggle Training

Cheaper alternatives to SageMaker for GPU-heavy Kaggle competitions. Both expose a CLI similar in spirit to our `launch_*.py` SageMaker scripts, but you manage the VM lifecycle yourself rather than submitting a "training job" abstraction.

## Why consider these

SageMaker `ml.g4dn.xlarge` is ~$0.53/hr on-demand. vast.ai and RunPod GPU prices for equivalent or better hardware:

| Provider | GPU | Price/hr | Notes |
|----------|-----|----------|-------|
| SageMaker | T4 (ml.g4dn.xlarge) | $0.526 | Managed, auto-artifact upload |
| vast.ai | T4 | ~$0.15–0.25 | Interruptible can be cheaper |
| vast.ai | RTX 3090 | ~$0.30–0.40 | 24 GB VRAM, faster than T4 |
| vast.ai | RTX 4090 | ~$0.50–0.70 | |
| RunPod | T4 | ~$0.30 | Secure Cloud; Community Cloud is cheaper |
| RunPod | RTX 3090 | ~$0.45 | |
| RunPod | A100 40 GB | ~$1.19 | |

5–10× cheaper than SageMaker for the same GPU. Trade-off: no "submit job" abstraction, manual lifecycle management.

## Trade-offs vs SageMaker

**You lose:**
- Auto model artifact tarball → S3
- CloudWatch logs
- IAM role-based credential passthrough
- Spot instance management (vast.ai has interruptible, but semantics differ)
- Automatic instance termination on failure

**You keep:**
- GPU access
- Full scriptability
- Custom Docker images

**You gain:**
- Big cost savings
- Faster boot time (no SageMaker provisioning delay)
- Direct SSH access for debugging

## vast.ai setup

### Install CLI

```bash
uv tool install vastai
vastai --help
```

### Authenticate

```bash
# Get an API key from https://cloud.vast.ai/account/
export VAST_API_KEY="your-key"
# Or: vastai set api-key your-key
```

### Search for offers

```bash
# Cheapest T4 with 16 GB RAM
vastai search offers 'gpu_name=T4 dph<0.25 ram>16' -o 'dph+'

# Interruptible RTX 3090
vastai search offers 'gpu_name=RTX_3090 rentable=true verified=true' -o 'dph+' --type ask

# Output shows offer IDs, GPU, RAM, disk, price/hr, reliability
```

### Launch an instance

```bash
# Rent an instance using an offer ID (on-demand)
OFFER_ID=1234567
vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel \
    --disk 50 \
    --onstart-cmd "apt-get update && apt-get install -y rsync"

# Get instance ID and SSH info
vastai show instances
```

### Run training

```bash
INSTANCE_ID=1234567
SSH_HOST=$(vastai ssh-url $INSTANCE_ID)  # ssh://root@host:port

# Parse host and port
SSH_USER="root"
SSH_HOST_ONLY="ssh6.vast.ai"
SSH_PORT=12345

# Sync code
rsync -avz -e "ssh -p $SSH_PORT" ./aws/ "$SSH_USER@$SSH_HOST_ONLY:/workspace/"

# Run
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST_ONLY" \
    "cd /workspace && pip install -r requirements.txt && python train.py"

# Pull results
rsync -avz -e "ssh -p $SSH_PORT" "$SSH_USER@$SSH_HOST_ONLY:/workspace/out/" ./predictions/
```

### Destroy instance

```bash
vastai destroy instance $INSTANCE_ID
```

## RunPod setup

### Install CLI

```bash
# Linux x86_64
wget -qO- cli.runpod.net | sudo bash
runpodctl --help
```

### Authenticate

```bash
# Get API key from https://www.runpod.io/console/user/settings
runpodctl config --apiKey your-key
```

### Launch a pod

```bash
# List available GPU types
runpodctl get cloud

# Launch pod with custom Docker image
runpodctl create pod \
    --name kaggle-training \
    --gpuType "NVIDIA GeForce RTX 3090" \
    --imageName "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel" \
    --containerDiskSize 50 \
    --volumeSize 0 \
    --ports "22/tcp"

# Get pod details
runpodctl get pods
```

### SSH and run

```bash
POD_ID=<pod-id>
# Get SSH command
runpodctl ssh $POD_ID

# Or sync and run like vast.ai above
```

### Destroy pod

```bash
runpodctl remove pod $POD_ID
```

## Wrapper pattern: launch_vastai.sh

A thin wrapper to mirror our `launch_cuml_jobs.py` pattern:

```bash
#!/bin/bash
# Usage: ./launch_vastai.sh <script.py> <offer_id>
set -euo pipefail

SCRIPT="$1"
OFFER_ID="$2"
CODE_DIR="./aws"
RESULT_DIR="./predictions"

# 1. Launch
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
    --image pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel \
    --disk 50 \
    --onstart-cmd "pip install lightgbm xgboost catboost" \
    --raw | jq -r '.new_contract')

echo "Instance: $INSTANCE_ID"

# 2. Wait for ready (poll status)
while true; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.actual_status')
    [ "$STATUS" = "running" ] && break
    sleep 10
done

# 3. Get SSH details
SSH_INFO=$(vastai ssh-url "$INSTANCE_ID")
# Parse ssh://user@host:port
SSH_USER=$(echo "$SSH_INFO" | sed 's#ssh://##; s#@.*##')
SSH_HOST=$(echo "$SSH_INFO" | sed 's#.*@##; s#:.*##')
SSH_PORT=$(echo "$SSH_INFO" | sed 's#.*:##')

# 4. Sync code
rsync -avz -e "ssh -p $SSH_PORT" "$CODE_DIR/" "$SSH_USER@$SSH_HOST:/workspace/"

# 5. Run
ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
    "cd /workspace && python $SCRIPT"

# 6. Pull artifacts
rsync -avz -e "ssh -p $SSH_PORT" \
    "$SSH_USER@$SSH_HOST:/workspace/out/" "$RESULT_DIR/"

# 7. Destroy
vastai destroy instance "$INSTANCE_ID"
```

## Data transfer strategy

For large training data (like the 630K PS6E4 dataset), options:

1. **rsync from local** — slow over home internet for multi-GB datasets
2. **Download from Kaggle on the instance** — put `KAGGLE_API_TOKEN` in the onstart-cmd and use `kaggle competitions download`
3. **Pull from public S3 bucket** — no AWS auth needed if bucket is public-read
4. **Use HuggingFace datasets or a shared GitHub repo** for smaller datasets

Recommended for Kaggle: use `kaggle` CLI from the instance to pull data fresh. Avoids home-internet bottleneck.

```bash
# On the instance, in onstart-cmd or first command:
pip install kaggle
mkdir -p /root/.kaggle
echo "$KAGGLE_API_TOKEN" > /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token
kaggle competitions download -c playground-series-s6e4 -p /data
unzip /data/playground-series-s6e4.zip -d /data
```

## Interruptible / spot pricing

**vast.ai** has "interruptible" instances — you bid a price, and if someone bids higher they get the GPU. Cheaper but can be evicted. Good for long training where you can checkpoint.

**RunPod** Community Cloud is essentially always-on but contributed by individuals; cheaper than Secure Cloud but with lower uptime guarantees.

For Kaggle training where you can rerun a failed job, interruptible is fine and saves 30-50%.

## When to use what

| Use case | Recommendation |
|----------|----------------|
| Quick test with small model | Kaggle notebook (free) |
| One-off GPU model | vast.ai interruptible |
| Many models, scripted workflow | vast.ai or RunPod with wrapper script |
| Long training needing reliability | RunPod Secure Cloud or SageMaker |
| Need managed infra (S3 output, CloudWatch) | SageMaker |
| Need Docker image in ECR | SageMaker (or push to vast.ai-compatible registry) |

## Caveats

- **SSH keys**: vast.ai and RunPod require you to add your public key via their UI first. The CLI alone won't bootstrap access.
- **Data egress**: vast.ai charges a small fee for outbound bandwidth. Compress results before rsync.
- **Persistence**: by default, data is destroyed when the instance is terminated. Use volumes if you need data to persist across instances.
- **Reliability**: interruptible instances can be evicted mid-training. Checkpoint model state every N folds.
- **Docker image size**: pulling a multi-GB RAPIDS image from nvcr.io can take 2-10 min on first boot. Use a smaller base image where possible.

## Reference links

- vast.ai CLI docs: https://vast.ai/docs/cli/
- RunPod CLI docs: https://docs.runpod.io/cli/reference/runpodctl
- vast.ai pricing: https://vast.ai/pricing
- RunPod pricing: https://www.runpod.io/pricing
