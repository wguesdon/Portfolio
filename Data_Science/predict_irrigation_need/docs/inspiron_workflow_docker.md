# Inspiron Home Server Workflow

Use the Dell Inspiron (64 GB RAM, Intel i7, Podman) as a zero-cost CPU training runner for long-running jobs. No time limit, ideal for overnight training of XGB/LGB/CatBoost variants, ensemble re-runs, and pseudo-labeling experiments.

## Pros and cons vs SageMaker

**Pros:**
- Free. No per-hour cost.
- No 12-hour/24-hour time limit.
- 64 GB RAM is plenty for 630K-row datasets.
- Full control: SSH in for live debugging.

**Cons:**
- i7 (likely 4-8 cores) is ~4-9× slower than `c5.9xlarge` (36 cores) for LGB/XGB training.
- Single job at a time in practice (share CPU otherwise).
- No GPU.
- No managed artifact upload — results must be rsync'd back.

**When to use:**
- Overnight jobs that would cost $5-15 on SageMaker (OTE rebuilds, seed variants, pseudo-labeling, ensemble stacker re-runs).
- Iterative local development where you can kick off a run before bed.

**When not to use:**
- GPU jobs (cuML, NN).
- Jobs that need to finish in a fixed window.
- Jobs that benefit from 36-core parallelism (LGB/XGB `num_threads=-1` runs).

## One-time setup

```bash
# Test SSH works
ssh Inspiron "echo ok"

# Check podman present
ssh Inspiron "podman --version"
# If missing: sudo apt install podman  (on Debian/Ubuntu)

# Create workspace
ssh Inspiron "mkdir -p ~/kaggle/PS6E4/{data,code,predictions,logs}"
```

## Sync data (one-time)

Dataset is ~90 MB so LAN rsync is fast.

```bash
cd /mnt/data/Github/Kaggle

rsync -avz Playground_Series/PS6E4/data/raw/ \
    Inspiron:~/kaggle/PS6E4/data/

# Verify
ssh Inspiron "ls -la ~/kaggle/PS6E4/data/"
```

Also sync the ensemble-data predictions directory if you want to run ensemble v14-style stacking locally:

```bash
rsync -avz Playground_Series/PS6E4/predictions/ \
    Inspiron:~/kaggle/PS6E4/predictions/
```

## Per-run sync + launch

### 1. Sync code

Copy only the script(s) you need. The `aws/` directory has the pip-installing SageMaker variants; use those for Docker builds. The `scripts/` directory has the local-execution variants.

```bash
# Option A: sync everything (simplest)
rsync -avz Playground_Series/PS6E4/aws/ Inspiron:~/kaggle/PS6E4/code/aws/
rsync -avz Playground_Series/PS6E4/scripts/ Inspiron:~/kaggle/PS6E4/code/scripts/

# Option B: sync specific files
rsync -avz Playground_Series/PS6E4/aws/train_lgb_pseudo.py \
    Playground_Series/PS6E4/aws/Dockerfile.lgb \
    Inspiron:~/kaggle/PS6E4/code/
```

### 2. Build image (one-time per Dockerfile)

The existing `Dockerfile.lgb` targets SageMaker channels (reads `SM_CHANNEL_TRAINING`). We can reuse it by setting env vars to map to the local mount.

```bash
ssh Inspiron "cd ~/kaggle/PS6E4/code/aws && podman build -f Dockerfile.lgb -t kaggle-ps6e4-lgb ."
```

### 3. Run training in container

```bash
ssh Inspiron "podman run --rm \
    -v ~/kaggle/PS6E4/data:/opt/ml/input/data/training:ro \
    -v ~/kaggle/PS6E4/predictions:/opt/ml/model:rw \
    -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
    -e SM_MODEL_DIR=/opt/ml/model \
    kaggle-ps6e4-lgb \
    python train_lgb_pseudo.py \
    2>&1 | tee ~/kaggle/PS6E4/logs/lgb_pseudo_\$(date +%Y%m%d_%H%M%S).log"
```

For long jobs, use `nohup` / `tmux` so disconnection doesn't kill it:

```bash
ssh Inspiron "cd ~/kaggle/PS6E4 && \
    nohup podman run --rm \
        -v \$(pwd)/data:/opt/ml/input/data/training:ro \
        -v \$(pwd)/predictions:/opt/ml/model:rw \
        -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
        -e SM_MODEL_DIR=/opt/ml/model \
        kaggle-ps6e4-lgb \
        python train_lgb_pseudo.py \
        > logs/lgb_pseudo.log 2>&1 &"
```

Or use `tmux`:

```bash
ssh Inspiron "tmux new -d -s lgb_pseudo 'cd ~/kaggle/PS6E4 && podman run ... python train_lgb_pseudo.py 2>&1 | tee logs/lgb_pseudo.log'"

# Later, check progress:
ssh Inspiron "tmux attach -t lgb_pseudo"  # Ctrl+B D to detach
# Or just:
ssh Inspiron "tail -f ~/kaggle/PS6E4/logs/lgb_pseudo.log"
```

### 4. Pull results

```bash
rsync -avz Inspiron:~/kaggle/PS6E4/predictions/ \
    Playground_Series/PS6E4/predictions/
```

## Expected runtimes

Rough estimates on an i7 (4-8 cores, single instance) vs c5.9xlarge (36 vCPU):

| Job | SageMaker c5.9xlarge | Inspiron (i7) |
|-----|---------------------|---------------|
| lgb_ote_deep (5-fold, depth 12, OTE 4x) | 2h 7m | 8-12h overnight |
| lgb_pseudo (5-fold, 900K rows) | ~1.5h | 6-10h |
| xgb_ote_magic | 2h 50m | 10-15h |
| Ensemble (v4 + stackers, 30-40 models) | 45-50m | 2-4h |
| One seed variant of lgb_v5 | ~25m | 1.5-2.5h |

Ensemble re-runs and small LGB variants are practical on the Inspiron. Long XGB training is borderline; maybe better on SageMaker.

## Wrapper script pattern

A thin launcher that mirrors our SageMaker pattern:

```bash
#!/bin/bash
# Usage: ./inspiron_run.sh <dockerfile-suffix> <script-name>
# Example: ./inspiron_run.sh lgb train_lgb_ote_deep.py
set -euo pipefail

HOST="Inspiron"
REMOTE_BASE="~/kaggle/PS6E4"
LOCAL_BASE="/mnt/data/Github/Kaggle/Playground_Series/PS6E4"

DOCKERFILE_SUFFIX="$1"
SCRIPT="$2"
IMAGE="kaggle-ps6e4-${DOCKERFILE_SUFFIX}"
LOG_NAME="${SCRIPT%.py}_$(date +%Y%m%d_%H%M%S).log"

echo "=== Syncing code ==="
rsync -avz --delete \
    "${LOCAL_BASE}/aws/" "${HOST}:${REMOTE_BASE}/code/aws/"
rsync -avz --delete \
    "${LOCAL_BASE}/scripts/" "${HOST}:${REMOTE_BASE}/code/scripts/"

echo "=== Building image (incremental) ==="
ssh "$HOST" "cd ${REMOTE_BASE}/code/aws && \
    podman build -f Dockerfile.${DOCKERFILE_SUFFIX} -t ${IMAGE} ."

echo "=== Launching training (detached tmux) ==="
SESSION="kaggle_${DOCKERFILE_SUFFIX}_$$"
ssh "$HOST" "tmux new -d -s ${SESSION} 'cd ${REMOTE_BASE} && \
    podman run --rm \
        -v \$(pwd)/data:/opt/ml/input/data/training:ro \
        -v \$(pwd)/predictions:/opt/ml/model:rw \
        -e SM_CHANNEL_TRAINING=/opt/ml/input/data/training \
        -e SM_MODEL_DIR=/opt/ml/model \
        ${IMAGE} \
        python ${SCRIPT} 2>&1 | tee logs/${LOG_NAME}'"

echo ""
echo "Running in tmux session: ${SESSION}"
echo "Log: ${REMOTE_BASE}/logs/${LOG_NAME}"
echo ""
echo "Monitor:  ssh ${HOST} 'tail -f ${REMOTE_BASE}/logs/${LOG_NAME}'"
echo "Attach:   ssh -t ${HOST} 'tmux attach -t ${SESSION}'"
echo "When done pull results with:"
echo "  rsync -avz ${HOST}:${REMOTE_BASE}/predictions/ ${LOCAL_BASE}/predictions/"
```

Save this as `scripts/inspiron_run.sh`, chmod +x, and run:

```bash
./Playground_Series/PS6E4/scripts/inspiron_run.sh lgb train_lgb_ote_deep.py
```

## Recurring maintenance

- Clean up old images: `ssh Inspiron "podman image prune -f"` weekly.
- Rotate logs: `ssh Inspiron "find ~/kaggle/PS6E4/logs -name '*.log' -mtime +7 -delete"`.
- Disk usage check: `ssh Inspiron "df -h ~ && du -sh ~/kaggle/PS6E4/*"`.

## Security notes

- The Inspiron should only be SSH-accessible from trusted network (home LAN or Tailscale VPN). Don't expose Podman socket publicly.
- Podman runs rootless by default, so a compromised container can't escalate. Still, keep the base image pinned to a specific tag.
- Your `.env` with AWS + Kaggle credentials stays on the laptop; don't copy to the Inspiron unless needed. If you need `KAGGLE_API_TOKEN` for data download, set it only in the container env var, not in a persistent file.

## Troubleshooting

### "container exit 137"

Out of memory. Check what's memory-hungry; 64 GB should cover PS6E4 training easily, but 4x-OTE augmented training (900K rows) may peak at 10-20 GB.

```bash
ssh Inspiron "free -h"
```

### Slow training

i7 is CPU-bound. Verify thread count inside container:

```bash
ssh Inspiron "podman run --rm kaggle-ps6e4-lgb python -c 'import multiprocessing; print(multiprocessing.cpu_count())'"
```

If the script hardcodes `num_threads=36`, change it to `num_threads=-1` or `os.cpu_count()` to use all available cores on the Inspiron.

### Container can't find SM_CHANNEL_TRAINING

Confirm the `-v` mount path inside container matches the env var:

```bash
ssh Inspiron "podman run --rm -v ~/kaggle/PS6E4/data:/opt/ml/input/data/training kaggle-ps6e4-lgb ls /opt/ml/input/data/training"
```

Should list train.csv, test.csv.
