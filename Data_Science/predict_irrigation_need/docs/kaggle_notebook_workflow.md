# Kaggle Notebook Workflow (Python SDK / CLI)

How to author, push, monitor, and fetch results from Kaggle Notebooks programmatically. Zero-cost alternative to SageMaker for GPU training (30 hours free T4 per week).

## Why use Kaggle Notebooks

- **Free T4 or T4x2 GPU** (30 hrs/week) or **P100** (limited)
- CLI-driven: `kaggle kernels push` and `kaggle kernels output` mirror the SageMaker launch/fetch pattern
- Competition data auto-mounted, no manual upload
- Private notebooks for experimentation, public for write-ups
- No container management, just a Python script or notebook

## Directory layout

Each notebook lives in its own directory:

```
Playground_Series/PS6E4/notebooks/<kernel_slug>/
├── kernel-metadata.json    # describes kernel, GPU flag, sources
├── <name>.py               # source (edit this)
└── <name>.ipynb            # generated, what gets pushed
```

## kernel-metadata.json template

```json
{
  "id": "wguesdon/my-kernel-slug",
  "title": "My Kernel Title",
  "code_file": "my_kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": ["playground-series-s6e4"],
  "kernel_sources": []
}
```

Key fields:
- `id`: `<username>/<slug>` — must be unique per user
- `is_private`: `true` to keep it unshared
- `enable_gpu`: `true` for T4 (or `"gpu_t4_x2"` for dual T4, `"gpu_p100"` for P100)
- `enable_internet`: `false` is faster and safer; use `true` only if your code needs pip install
- `competition_sources`: list competition slugs to attach
- `dataset_sources`: `["username/dataset-slug"]` for your own datasets
- `kernel_sources`: `["username/other-kernel"]` to chain kernels (use outputs as inputs)

## Kaggle data paths

For this repo's notebooks, competition data is mounted at:

```
/kaggle/input/competitions/playground-series-s6e4/
```

Hard-code this path. Use `/kaggle/working/` for output — only files written there are preserved as kernel outputs.

Dataset sources (if you use them) are mounted at `/kaggle/input/<dataset-slug>/`.
Kernel sources (outputs of another kernel) are mounted at `/kaggle/input/<source-kernel-slug>/`.

## Authoring: convert .py → .ipynb

Kaggle CLI accepts either, but for notebooks it's simpler to push a single-cell .ipynb generated from your Python script:

```python
import json
from pathlib import Path

code = Path("my_script.py").read_text()

nb = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("my_script.ipynb").write_text(json.dumps(nb, indent=1))
```

Edit the .py, regenerate the .ipynb, push. Keeps git diffs readable.

## Push a kernel

```bash
cd /mnt/data/Github/Kaggle
export $(grep KAGGLE_API_TOKEN .env)
uv run kaggle kernels push -p Playground_Series/PS6E4/notebooks/my_kernel/
```

Each push creates a new version. Output:
```
Kernel version 2 successfully pushed.  Please check progress at https://www.kaggle.com/code/...
```

## Monitor status

```bash
uv run kaggle kernels status wguesdon/ps6e4-xgb-ote-shallow-gpu
```

Possible states:
- `queued` — waiting for GPU slot
- `running` — executing
- `complete` — done, outputs available
- `error` — check logs
- `cancelled`

Poll every minute or so; Kaggle GPU queue usually grants a slot within 1-2 min.

## Fetch outputs

```bash
uv run kaggle kernels output wguesdon/ps6e4-xgb-ote-shallow-gpu \
    -p /mnt/data/Github/Kaggle/Playground_Series/PS6E4/predictions/
```

Downloads everything in `/kaggle/working/` from the last successful run. Includes:
- `*.npy`, `*.csv`, `*.pkl` files saved by your code
- `*.log` execution logs

## Get the full log (for errors)

The CLI doesn't stream live logs, but you can download the execution log after a run:

```bash
uv run kaggle kernels output wguesdon/ps6e4-xgb-ote-shallow-gpu -p /tmp/ -o
cat /tmp/*.log
```

Or just open the kernel URL in a browser: https://www.kaggle.com/code/<id>/output gives you the rendered output with any tracebacks.

## Common errors and fixes

### `FileNotFoundError: /kaggle/input/.../train.csv`
Use `/kaggle/input/competitions/<comp-slug>/` (note the `competitions/` segment). For datasets or kernel sources the segment is omitted.

### `ModuleNotFoundError: No module named 'xgboost'`
Kaggle's default image has most ML libs pre-installed. If something's missing, set `"enable_internet": true` in metadata and `!pip install <pkg>` in your notebook. But internet-enabled kernels start slower; prefer libraries already installed.

### Kernel stays in `queued` for 10+ min
GPU availability. Try again later, switch to CPU temporarily, or drop to a smaller GPU (T4 usually faster queue than T4x2 or P100).

### `ValueError: cannot reshape array of size ...` or kernel crashes silently
Memory issue. Kaggle GPU kernels have ~13 GB system RAM + GPU VRAM. Large datasets + large models can OOM without a clear error. Check the rendered output on the kernel page.

### "You have reached your weekly GPU quota of 30 hours"
You're out for the week. Either wait (quota resets weekly) or run on CPU instead (remove `enable_gpu`).

### `xgb` complains `device='cuda'` unavailable but kernel has GPU enabled
Some images need `tree_method='hist'` to accompany `device='cuda'`. Always set both. On some Kaggle images, also ensure CUDA version matches what XGBoost was built with.

## Chaining kernels

To feed the outputs of one kernel into another, use `kernel_sources`:

```json
{
  "id": "wguesdon/ps6e4-ensemble-using-gpu-oof",
  "kernel_sources": ["wguesdon/ps6e4-xgb-ote-shallow-gpu"],
  ...
}
```

The output files from the source kernel appear at `/kaggle/input/ps6e4-xgb-ote-shallow-gpu/`. Useful for ensemble kernels that aggregate multiple base-model kernels.

## Pulling a published kernel (for reference)

Looking at someone else's notebook:

```bash
uv run kaggle kernels pull yunsuxiaozi/pss6e4-xgb-cv-0-979805 -p /tmp/nb
```

Downloads the .ipynb. View cells programmatically:
```python
import json
nb = json.load(open("/tmp/nb/pss6e4-xgb-cv-0-979805.ipynb"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if src.strip():
        print(f"=== Cell {i} ===")
        print(src[:2000])
```

## GPU types and when to use

| Type | VRAM | Notes |
|------|------|-------|
| Nothing (CPU) | 0 | For stacker ensembles, LR, LGB/XGB CPU hist |
| T4 (`enable_gpu: true`) | 16 GB | Default GPU; fine for most tabular |
| T4x2 (`"gpu_t4_x2"`) | 32 GB | Two T4s, useful for large models or parallel folds |
| P100 (`"gpu_p100"`) | 16 GB | Older but fast; limited availability |

For PS6E4-style tabular: single T4 is more than enough.

## Current notebooks in this repo

- `notebooks/xgb_ote_shallow_gpu/` — XGB shallow + OTE on GPU (A/B test vs Inspiron CPU)
- `notebooks/ps6e4_27model_ensemble_v2.ipynb` — older ensemble kernel (not via this workflow)
- `notebooks/ps6e4_30model_ensemble.ipynb` — older ensemble kernel

## References

- Kaggle API docs: https://github.com/Kaggle/kaggle-api
- CLI reference: https://www.kaggle.com/docs/api
- Available environments: https://github.com/Kaggle/docker-python
