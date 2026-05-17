# Pushing Kaggle Kernels via API

## kernel-metadata.json

Every Kaggle kernel push requires a `kernel-metadata.json` in the same directory as the code file.

### Example: GPU script (private)

```json
{
  "id": "wguesdon/ps6e4-gnn-graphsage-training",
  "title": "PS6E4 GNN GraphSAGE Training",
  "code_file": "ps6e4_gnn_training.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "accelerator": "nvidiaTeslaT4",
  "enable_internet": true,
  "competition_sources": ["playground-series-s6e4"],
  "dataset_sources": [],
  "kernel_sources": [],
  "category_ids": ["deep learning"]
}
```

### Example: CPU notebook (public)

```json
{
  "id": "wguesdon/ps6e4-14-model-gbdt-ensemble",
  "title": "PS6E4 14 Model GBDT Ensemble",
  "code_file": "ps6e4_showcase.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": false,
  "enable_internet": false,
  "competition_sources": ["playground-series-s6e4"],
  "dataset_sources": ["wguesdon/ps6e4-irrigation-14-model-predictions"]
}
```

## Key fields

| Field | Values | Notes |
|-------|--------|-------|
| `kernel_type` | `"script"` or `"notebook"` | `.py` = script, `.ipynb` = notebook |
| `enable_gpu` | `true` / `false` | Enables GPU accelerator |
| `accelerator` | `"nvidiaTeslaT4"` | Optional. May not be honored by API |
| `enable_internet` | `true` / `false` | Needed for pip installs at runtime |
| `competition_sources` | list of slugs | Mounts competition data at `/kaggle/input/competitions/` |
| `dataset_sources` | list of slugs | Format: `"username/dataset-slug"` |
| `kernel_sources` | list of slugs | Other kernels whose output to mount |

## Commands

```bash
# Push a kernel
export $(grep KAGGLE_API_TOKEN /mnt/data/Github/Kaggle/.env)
uv run kaggle kernels push -p /path/to/dir/

# Check status
uv run kaggle kernels status wguesdon/kernel-slug

# Download output
uv run kaggle kernels output wguesdon/kernel-slug -p /tmp/output/

# Pull source code
uv run kaggle kernels pull wguesdon/kernel-slug -p /tmp/source/
```

## Gotchas

- Kaggle P100 GPUs (compute capability 6.0) are incompatible with recent PyTorch builds (sm_70+ only). Use SageMaker with T4 instead.
- The `accelerator` field in metadata may not override Kaggle's GPU assignment.
- `enable_internet: true` is required if the script does `pip install` at runtime.
- Kernel output is only available after the run completes successfully (status `"complete"`).
