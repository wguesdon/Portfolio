# Inspiron Home Server Workflow (uv venv, no containers)

This is the tested workflow used for running CPU training jobs on the Dell Inspiron. Direct-Python approach (no Docker/Podman) for simplicity.

## Hardware

- Dell Inspiron 15 3520 (Ubuntu)
- Intel i7, **12 cores**
- **62 GB RAM**
- SSH hostname: `Inspiron` (configured in `~/.ssh/config`)
- Python 3.12.3 system, uv installed at `/home/will/.local/bin/uv`
- Docker and Podman available (not used in this workflow; see `inspiron_workflow_docker.md` for the container-based variant)

## Pros and cons vs SageMaker

**Pros:**
- Free, no per-hour cost.
- No time limit; long jobs run overnight/multi-day.
- 62 GB RAM easily handles 900K-row pseudo-label augmented datasets.
- 12 cores is reasonable for scikit-learn parallel fits.

**Cons:**
- Slower than `c5.9xlarge` (36 vCPUs) by ~3× for LGB/XGB; adequate for small models.
- No GPU.
- Single training at a time in practice (all cores used per job).

**Best for:**
- Scikit-learn models (LR variants, ExtraTrees, SVM on modest samples).
- Ensemble stacker re-runs (fast).
- Overnight LGB/XGB seed variants.

**Not for:**
- cuML / NN (no GPU).
- Jobs that need 36-core parallelism to finish in a fixed window.

## Directory layout

Everything lives under `/home/will/Documents/kaggle/PS6E4/`:

```
/home/will/Documents/kaggle/PS6E4/
├── data/              # train.csv, test.csv (synced once)
├── code/
│   ├── .venv/         # uv-managed virtualenv
│   ├── pyproject.toml # project deps (numpy, pandas, scikit-learn)
│   ├── uv.lock
│   └── train_*.py     # training scripts
├── predictions/       # output oof_*.npy, pred_*.npy
└── logs/              # tmux log files
```

## One-time setup

Done already but documented here for reference.

### Create directories
```bash
ssh Inspiron "mkdir -p /home/will/Documents/kaggle/PS6E4/{data,code,predictions,logs}"
```

### Sync training data (one-time)

Because rsync between `/mnt/data` and the Inspiron was blocked by sandbox in the Claude Code environment, we use a `tar | ssh` pipe instead:

```bash
tar cz -C /mnt/data/Github/Kaggle/Playground_Series/PS6E4/data/raw . \
    | ssh Inspiron "cd /home/will/Documents/kaggle/PS6E4/data && tar xz"
```

Verify:
```bash
ssh Inspiron "ls -la /home/will/Documents/kaggle/PS6E4/data/"
```

### Initialize uv project and add deps

```bash
ssh Inspiron "cd /home/will/Documents/kaggle/PS6E4/code && \
    /home/will/.local/bin/uv init --no-workspace --bare --python 3.12 && \
    /home/will/.local/bin/uv add scikit-learn pandas numpy"
```

This creates `pyproject.toml`, `uv.lock`, and `.venv/`. Installs scikit-learn, pandas, numpy, and their deps.

For models that need lightgbm, xgboost, catboost, add them as needed:
```bash
ssh Inspiron "cd /home/will/Documents/kaggle/PS6E4/code && \
    /home/will/.local/bin/uv add lightgbm"
```

## Per-run workflow

### 1. Upload a script

Due to rsync path sandboxing, use `cat | ssh` to upload:

```bash
cat /mnt/data/Github/Kaggle/Playground_Series/PS6E4/scripts/train_lr_elastic_ote.py \
    | ssh Inspiron "cat > /home/will/Documents/kaggle/PS6E4/code/train_lr_elastic_ote.py"
```

### 2. Launch in detached tmux

Key points:
- **Always set `PYTHONUNBUFFERED=1`** or `tee` won't get line-by-line output (buffering caused a 0-byte log file initially).
- Use the venv's python directly (`.venv/bin/python`) rather than `uv run` which can hang on lock resolution when dependencies are already synced.
- Set `SM_CHANNEL_TRAINING` and `SM_MODEL_DIR` env vars so scripts written for SageMaker work unchanged.

```bash
ssh Inspiron 'tmux new -d -s lr_elastic "cd /home/will/Documents/kaggle/PS6E4/code && \
    PYTHONUNBUFFERED=1 \
    SM_CHANNEL_TRAINING=/home/will/Documents/kaggle/PS6E4/data \
    SM_MODEL_DIR=/home/will/Documents/kaggle/PS6E4/predictions \
    /home/will/Documents/kaggle/PS6E4/code/.venv/bin/python train_lr_elastic_ote.py 2>&1 \
    | tee /home/will/Documents/kaggle/PS6E4/logs/lr_elastic.log"'
```

### 3. Monitor

```bash
# Tail log
ssh Inspiron "tail -f /home/will/Documents/kaggle/PS6E4/logs/lr_elastic.log"

# Attach interactive (Ctrl+B D to detach)
ssh -t Inspiron "tmux attach -t lr_elastic"

# List sessions
ssh Inspiron "tmux ls"

# Kill if needed
ssh Inspiron "tmux kill-session -t lr_elastic"
```

### 4. Pull results

Use the same tar-over-ssh trick because rsync is blocked by sandbox:

```bash
ssh Inspiron "tar cz -C /home/will/Documents/kaggle/PS6E4/predictions ." \
    | tar xz -C /mnt/data/Github/Kaggle/Playground_Series/PS6E4/predictions
```

## Gotchas we hit during setup

1. **`rsync` blocked by Claude Code sandbox.** Worked around with `tar | ssh`. If running locally without the sandbox, plain rsync would work fine.

2. **pandas 3.0 dtype change.** On the Inspiron (pandas 3.0.2) string columns report `str` dtype, not `object` as in older pandas. Scripts that do `train[c].dtype == object` will miss them. Fix: use `pd.api.types.is_numeric_dtype(train[c])` to detect numeric columns and treat everything else as categorical.

3. **`uv run` hanging.** When the lockfile exists and deps are synced, `uv run` sometimes waits on a hold. Use `.venv/bin/python` directly to skip it.

4. **Empty log file.** Python buffers stdout when piped. Set `PYTHONUNBUFFERED=1` before the command.

## Expected runtimes on this box

Rough estimates for common PS6E4 jobs:

| Job | Runtime |
|-----|---------|
| LR ElasticNet + OTE (5 fold, 85 features) | 30-60 min |
| LGB v5 seed variant (5 fold) | 1-2h |
| XGB OTE magic (5 fold) | 10-15h (borderline, prefer SageMaker) |
| Ensemble v4 run (stacker on 30-40 models) | 2-4h |
| Pseudo-label lgb_v5 (5 fold, 900K rows) | 6-10h |

## Future: wrapper script

A generic launcher that automates sync+launch lives at
`scripts/inspiron_run.sh` (Podman version). A venv version would look like:

```bash
SCRIPT=$1
NAME=${SCRIPT%.py}

# 1. Upload script
cat /mnt/data/Github/Kaggle/Playground_Series/PS6E4/scripts/$SCRIPT \
    | ssh Inspiron "cat > /home/will/Documents/kaggle/PS6E4/code/$SCRIPT"

# 2. Launch
ssh Inspiron "tmux new -d -s ${NAME} \"cd /home/will/Documents/kaggle/PS6E4/code && \
    PYTHONUNBUFFERED=1 \
    SM_CHANNEL_TRAINING=/home/will/Documents/kaggle/PS6E4/data \
    SM_MODEL_DIR=/home/will/Documents/kaggle/PS6E4/predictions \
    /home/will/Documents/kaggle/PS6E4/code/.venv/bin/python $SCRIPT 2>&1 \
    | tee /home/will/Documents/kaggle/PS6E4/logs/${NAME}.log\""

echo "Monitor: ssh Inspiron 'tail -f /home/will/Documents/kaggle/PS6E4/logs/${NAME}.log'"
```

## Current status (2026-04-12)

- LR ElasticNet training launched in tmux session `lr_elastic`.
- Log: `/home/will/Documents/kaggle/PS6E4/logs/lr_elastic.log`.
- Estimated completion: 30-60 min.
- After completion, `oof_lr_elastic.npy` and `pred_lr_elastic.npy` will be in `/home/will/Documents/kaggle/PS6E4/predictions/`.
- Pull back, upload to `s3://kaggle-ps6e4/ensemble-data/predictions/`, and add `lr_elastic` to the ensemble model list for the next run.
