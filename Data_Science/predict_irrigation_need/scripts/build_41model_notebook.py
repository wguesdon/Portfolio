"""Build the 41 model Kaggle notebook by editing the existing 30 model notebook.

Pulls the live `wguesdon/ps6e4-30-model-ensemble-with-stacking` notebook from
`/tmp/nb_pull_30/`, makes surgical edits to the cells that reference the model
count, swaps in the 41 model inventory and `MODEL_NAMES` list, repoints the
dataset paths to the new 41 model datasets, and writes the result to
`Playground_Series/PS6E4/notebooks/ps6e4_41model_ensemble.ipynb` along with the
matching `kernel-metadata.json`.

Run:
    uv run python Playground_Series/PS6E4/scripts/build_41model_notebook.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_NB = Path("/tmp/nb_pull_30/ps6e4-30-model-ensemble-with-stacking.ipynb")
OUT_DIR = REPO / "notebooks" / "ps6e4_41model_ensemble"
OUT_NB = OUT_DIR / "ps6e4-30-model-ensemble-with-stacking.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

# v15 41 model inventory rows for the markdown table, sorted by CV descending
# Pulled from SESSION_RESUME.md and predictions/*.npy
INVENTORY_ROWS: list[tuple[str, str, str, str]] = [
    ("lgb_ote", "LightGBM", "OTE", "0.97942"),
    ("xgb_ote", "XGBoost", "OTE", "0.97938"),
    ("xgb_ote_s44", "XGBoost", "OTE (seed 2044)", "0.97930"),
    ("xgb_ote_shallow_gpu", "XGBoost", "OTE shallow (Kaggle T4)", "0.97927"),
    ("xgb_ote_s43", "XGBoost", "OTE (seed 2043)", "0.97920"),
    ("cat_ote", "CatBoost", "OTE", "0.97919"),
    ("xgb_ote_magic", "XGBoost", "OTE + magic", "0.97910"),
    ("cat_s44", "CatBoost", "v3", "0.97834"),
    ("cat_digit", "CatBoost", "v3", "0.97828"),
    ("cat-...-13-29", "CatBoost", "v3", "0.97812"),
    ("cat-...-12-44-01", "CatBoost", "v3", "0.97804"),
    ("cat_s43", "CatBoost", "v3", "0.97802"),
    ("realmlp_mahog", "RealMLP", "mahogany", "0.97802"),
    ("cat-...-12-44-05", "CatBoost", "v3", "0.97801"),
    ("cat_v2", "CatBoost", "v3", "0.97786"),
    ("lgb_v5", "LightGBM", "v3", "0.97701"),
    ("lgb_ote_deep", "LightGBM", "OTE deep", "0.97660"),
    ("lgb_digit", "LightGBM", "v3", "0.97562"),
    ("xgb_v5", "XGBoost", "v3", "0.97561"),
    ("xgb-...-11-12-41", "XGBoost", "v3", "0.97262"),
    ("xgb-...-11-12-38", "XGBoost", "v3", "0.97256"),
    ("xgb_v2", "XGBoost", "v3", "0.97208"),
    ("xgb_s44", "XGBoost", "v3", "0.97199"),
    ("xgb_s43", "XGBoost", "v3", "0.97185"),
    ("lgb_s44", "LightGBM", "v3", "0.97171"),
    ("lgb_v2", "LightGBM", "v3", "0.97140"),
    ("lgb_s43", "LightGBM", "v3", "0.97131"),
    ("realmlp_v3fix", "RealMLP", "v3", "0.97108"),
    ("tabm_v3fix", "TabM", "v3", "0.97053"),
    ("lgb_ote_shallow", "LightGBM", "OTE shallow", "0.97048"),
    ("et_ote", "ExtraTrees", "OTE", "0.96156"),
    ("svm_ote", "cuML SVM RBF", "OTE", "0.96154"),
    ("rf_ote", "cuML RandomForest", "OTE", "0.96005"),
    ("lr_elastic", "Logistic Regression", "OTE ElasticNet", "0.95568"),
    ("lr_l1_cuml", "cuML LR L1", "OTE", "0.95554"),
    ("lr_cuml", "cuML LR L2", "OTE", "0.95522"),
    ("lr_ote", "Logistic Regression", "OTE", "0.93743"),
    ("gnb_ote", "cuML GaussianNB", "OTE", "0.90860"),
    ("knn5_lite", "KNN k=5", "OTE lite features", "0.90700"),
    ("knn_ote", "KNN", "OTE", "0.77552"),
    ("knn15_ote", "cuML KNN k=15", "OTE", "0.71937"),
]


INTRO_MD = """# PS6E4 12th Place: 41-Model Ensemble with Stacking

**Competition:** [Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4)
**Final placement:** 12 of 457
**Private LB:** 0.98082

**Task:** 3-class classification (Low, Medium, High) of irrigation need
**Metric:** Balanced Accuracy

This notebook covers:
1. **EDA** of the competition data
2. **Workflow overview**: how 41 models were trained on AWS SageMaker, Kaggle GPU, and a local Inspiron
3. **Ensemble** from pre-computed OOF predictions using greedy selection, LightGBM stacking, rank averaging, and threshold optimization

All 41 models were trained off Kaggle. This notebook only runs the ensembling step, which is CPU-only. The stacker output reproduces the v15 submission that placed 12th (private LB 0.98082).

For the full write up see the discussion post.
"""


WORKFLOW_MD = """## 3. Workflow Overview

### Architecture

41 models were trained across **AWS SageMaker** (CPU and GPU), **Kaggle GPU notebooks** (T4), and a **local Dell Inspiron** (12 core CPU). Everything was driven from the command line. The training code is attached as a separate Kaggle dataset (`ps6e4-training-code-41`).

```
Local machine                    AWS SageMaker
=============                    =============

scripts/         ──upload──>     S3 bucket
  train_xgb_ote.py               s3://kaggle-ps6e4/
  train_cat_ote.py                  ├── ensemble-data/
  train_lgb_ote.py                  │   ├── train.csv
  ensemble_v4.py                    │   ├── test.csv
  ...                               │   └── predictions/
                                    │       ├── oof_*.npy
                                    │       └── pred_*.npy
Launch jobs via   ──API──>       Training Jobs
SageMaker SDK                      ml.c5.9xlarge (CPU, GBDTs)
                                   ml.g4dn.xlarge (GPU, cuML and NNs)

Download results  <──S3──        model.tar.gz
  oof_*.npy                        oof predictions (N_train, 3)
  pred_*.npy                       test predictions (N_test, 3)
```

Each model script:
1. Reads train/test CSV from the SageMaker input channel
2. Runs 5-fold StratifiedKFold (seed=42, consistent across all models)
3. Saves out-of-fold (OOF) predictions and test predictions as numpy arrays
4. Packages outputs into `model.tar.gz` uploaded to S3

### Two Feature Engineering Pipelines

**Pipeline 1: v3 features** — domain-driven physics ratios, digit extraction, frequency encoding, snapping, priors from the original 10K dataset, Optuna tuning. Drives 21 models in the pool (CatBoost, LightGBM, XGBoost variants and RealMLP/TabM neural nets).

**Pipeline 2: Ordered Target Encoding (OTE)** — adapted from [emanuellcs's notebook](https://www.kaggle.com/code/emanuellcs/predicting-irrigation-need-xgboost-ote) (LB 0.98011). Bayesian-smoothed cumulative leave-one-out target encoding with 4x shuffle. Digit extraction. Frequency-based ordinal encoding for categoricals. Drives the other 20 models, including six algorithm families (XGB, LGB, CatBoost, ExtraTrees, KNN, LR ElasticNet, cuML RF/SVM/GNB).

### Why This Works

The two pipelines produce fundamentally different representations of the same data. v3 is domain-driven. OTE is target-driven. The stacker learns when to trust each model on a per-sample basis. Adding deliberately weak diversity models (LR at CV 0.94, KNN at CV 0.78, GNB at CV 0.91) hurt my equal-weight blends but helped the stacker by providing genuinely different error structures.
"""


MODEL_NAMES_CODE = '''MODEL_NAMES = [
    # OTE pipeline GBDTs (top tier, 6)
    "lgb_ote", "xgb_ote", "xgb_ote_s44", "xgb_ote_s43", "cat_ote", "xgb_ote_magic",
    # OTE Kaggle T4 shallow XGB
    "xgb_ote_shallow_gpu",
    # v3 pipeline CatBoost (7)
    "cat_s44", "cat_digit", "cat-2026-04-03-13-29-13-994",
    "cat-2026-04-03-12-44-01-719", "cat_s43",
    "cat-2026-04-03-12-44-05-363", "cat_v2",
    # v3 pipeline LightGBM (5)
    "lgb_v5", "lgb_digit", "lgb_s44", "lgb_v2", "lgb_s43",
    # v3 pipeline XGBoost (6)
    "xgb_v5", "xgb-2026-04-03-11-12-41-698", "xgb-2026-04-03-11-12-38-123",
    "xgb_v2", "xgb_s44", "xgb_s43",
    # OTE LightGBM depth variants (2)
    "lgb_ote_shallow", "lgb_ote_deep",
    # Neural networks (3)
    "realmlp_mahog", "realmlp_v3fix", "tabm_v3fix",
    # OTE diversity models, individually weak but key for the stacker (12)
    "et_ote",
    "svm_ote", "rf_ote", "gnb_ote",
    "lr_elastic", "lr_l1_cuml", "lr_cuml", "lr_ote",
    "knn5_lite", "knn_ote", "knn15_ote",
]
assert len(MODEL_NAMES) == 41, f"expected 41 models, got {len(MODEL_NAMES)}"

oof_dict = {}
pred_dict = {}

print(f"{'#':<4} {'Model':<42} {'CV (bal_acc)':>12}")
print("-" * 60)

for i, name in enumerate(MODEL_NAMES, 1):
    oof = np.load(PRED_DIR / f"oof_{name}.npy")
    pred = np.load(PRED_DIR / f"pred_{name}.npy")
    oof_dict[name] = oof
    pred_dict[name] = pred
    score = balanced_accuracy_score(y_train, oof.argmax(axis=1))
    print(f"{i:<4} {name:<42} {score:.5f}")

print(f"\\nLoaded {len(MODEL_NAMES)} models.")
print(f"OOF shape: {oof.shape}, Test shape: {pred.shape}")
'''


def _build_inventory_md() -> str:
    """Render the 41 row markdown table for the model inventory cell."""
    lines = [
        "### 3.1 Model Inventory",
        "",
        "| # | Model | Algorithm | Pipeline | CV |",
        "|---|-------|-----------|----------|----|",
    ]
    for i, (name, algo, pipe, cv) in enumerate(INVENTORY_ROWS, 1):
        lines.append(f"| {i} | {name} | {algo} | {pipe} | {cv} |")
    return "\n".join(lines)


def _replace_in_cell(src: str, replacements: list[tuple[str, str]]) -> str:
    """Apply each (old, new) replacement in order to the cell source."""
    for old, new in replacements:
        src = src.replace(old, new)
    return src


def _join_source(cell: dict) -> str:
    """Return cell source as a single string."""
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _set_source(cell: dict, text: str) -> None:
    """Write the cell source as a list of lines, preserving trailing newlines."""
    cell["source"] = text.splitlines(keepends=True)


def main() -> None:
    """Edit the notebook, write the new ipynb and kernel metadata.

    Raises:
        FileNotFoundError: if the source notebook is not in /tmp/nb_pull_30/.
    """
    if not SRC_NB.exists():
        raise FileNotFoundError(
            f"{SRC_NB} not found. Run `kaggle kernels pull "
            "wguesdon/ps6e4-30-model-ensemble-with-stacking -p /tmp/nb_pull_30/ -m` first."
        )

    nb = json.loads(SRC_NB.read_text())

    # Cell 0: title and intro
    _set_source(nb["cells"][0], INTRO_MD)

    # Cell 2: dataset paths. Use a resolver because Kaggle mounts attached
    # datasets inconsistently as either `/kaggle/input/<slug>/` or
    # `/kaggle/input/datasets/<owner>/<slug>/`.
    src = _join_source(nb["cells"][2])
    old_paths_block = (
        'COMP_DIR = Path("/kaggle/input/competitions/playground-series-s6e4")\n'
        'PRED_DIR = Path("/kaggle/input/datasets/wguesdon/ps6e4-30model-oof-predictions")\n'
        'CODE_DIR = Path("/kaggle/input/datasets/wguesdon/ps6e4-training-code-30")'
    )
    new_paths_block = (
        'COMP_DIR = Path("/kaggle/input/competitions/playground-series-s6e4")\n'
        '\n'
        '\n'
        'def _resolve_dataset(slug: str, owner: str = "wguesdon") -> Path:\n'
        '    """Find the mount point for an attached Kaggle dataset.\n'
        '\n'
        '    Kaggle mounts datasets at different paths depending on the dataset\n'
        '    age and notebook config. Try the common locations and return the\n'
        '    first that exists.\n'
        '    """\n'
        '    candidates = [\n'
        '        Path(f"/kaggle/input/{slug}"),\n'
        '        Path(f"/kaggle/input/datasets/{owner}/{slug}"),\n'
        '    ]\n'
        '    for path in candidates:\n'
        '        if path.exists():\n'
        '            return path\n'
        '    raise FileNotFoundError(\n'
        '        f"Dataset {owner}/{slug} not mounted. Tried: " + ", ".join(str(p) for p in candidates)\n'
        '    )\n'
        '\n'
        '\n'
        'PRED_DIR = _resolve_dataset("ps6e4-12th-place-oof-predictions")\n'
        'CODE_DIR = _resolve_dataset("ps6e4-training-code-41")\n'
        'print(f"PRED_DIR resolved to: {PRED_DIR}")\n'
        'print(f"CODE_DIR resolved to: {CODE_DIR}")'
    )
    if old_paths_block not in src:
        raise RuntimeError("Could not find original paths block in cell 2")
    src = src.replace(old_paths_block, new_paths_block)
    _set_source(nb["cells"][2], src)

    # Cell 13: workflow overview
    _set_source(nb["cells"][13], WORKFLOW_MD)

    # Cell 14: model inventory table
    _set_source(nb["cells"][14], _build_inventory_md())

    # Cell 16: MODEL_NAMES list
    _set_source(nb["cells"][16], MODEL_NAMES_CODE)

    # Bulk text edits across remaining cells. Order matters: longer strings first.
    bulk_edits: list[tuple[str, str]] = [
        ("LGB Stacker 30m", "LGB Stacker 41m"),
        ("LGB stacker (all 30 models)", "LGB stacker (all 41 models)"),
        ("LGB Stacker (all 30 models)", "LGB Stacker (all 41 models)"),
        ("LightGBM stacker on all 30 models", "LightGBM stacker on all 41 models"),
        ("LGB stacking on all 30 models", "LGB stacking on all 41 models"),
        ("LGB stacker on all 30 models", "LGB stacker on all 41 models"),
        ("Equal 30m", "Equal 41m"),
        ("Rank 30m", "Rank 41m"),
        ("Equal weight all 30 models", "Equal weight all 41 models"),
        ("all 30 models' OOF", "all 41 models' OOF"),
        ("all 30 prediction files", "all 41 prediction files"),
        ("90 features = 30 models x 3 classes", "123 features = 41 models x 3 classes"),
        ("how to train all 30 models", "how to train all 41 models"),
        ("30 models ensembled", "41 models ensembled"),
        ("`ps6e4-training-code`", "`ps6e4-training-code-41`"),
    ]

    skip_cells = {0, 2, 13, 14, 16}
    for i, cell in enumerate(nb["cells"]):
        if i in skip_cells:
            continue
        old = _join_source(cell)
        new = _replace_in_cell(old, bulk_edits)
        if new != old:
            _set_source(cell, new)

    # Quick sanity: confirm no stale "30 models" wording survives
    for i, cell in enumerate(nb["cells"]):
        text = _join_source(cell)
        for needle in ("30 models", "30-Model", "30 Model"):
            if needle in text:
                print(f"WARN cell {i} still contains '{needle}'")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1))
    shutil.copy2(OUT_NB, REPO / "notebooks" / "ps6e4_41model_ensemble.ipynb")

    metadata = {
        "id": "wguesdon/ps6e4-12th-place-41-model-ensemble-with-stacking",
        "title": "PS6E4 12th Place: 41-Model Ensemble with Stacking",
        "code_file": OUT_NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["playground-series-s6e4"],
        "dataset_sources": [
            "wguesdon/ps6e4-12th-place-oof-predictions",
            "wguesdon/ps6e4-training-code-41",
        ],
        "kernel_sources": [],
    }
    OUT_META.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {OUT_NB}")
    print(f"Wrote {OUT_META}")
    print(f"Cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
