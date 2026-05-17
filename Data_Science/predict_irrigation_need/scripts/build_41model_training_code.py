"""Build the 41 model training code Kaggle dataset folder.

Stages every training and ensembling script used to produce the 41 model v15
ensemble. Includes both the local Inspiron / SageMaker-style scripts under
`scripts/` and the SageMaker entry-point scripts under `aws/`.

Run:
    uv run python Playground_Series/PS6E4/scripts/build_41model_training_code.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Files to copy. Source is relative to repo root, destination is the basename
# in the dataset folder (no nesting; Kaggle datasets are flat).
FILES: list[tuple[str, str]] = [
    # AWS SageMaker entry points (v3 pipeline boosters and depth variants)
    ("aws/train_xgb.py", "train_xgb.py"),
    ("aws/train_lgb.py", "train_lgb.py"),
    ("aws/train_cat.py", "train_cat.py"),
    ("aws/train_xgb_v5.py", "train_xgb_v5.py"),
    ("aws/train_lgb_v5.py", "train_lgb_v5.py"),
    ("aws/train_cat_v5.py", "train_cat_v5.py"),
    ("aws/train_xgb_digit.py", "train_xgb_digit.py"),
    ("aws/train_lgb_digit.py", "train_lgb_digit.py"),
    ("aws/train_cat_digit.py", "train_cat_digit.py"),
    # Neural networks (v3 fix pipeline + mahog pipeline)
    ("aws/train_nn_v3.py", "train_nn_v3.py"),
    # OTE pipeline GBDTs and diversity (Inspiron / SageMaker)
    ("scripts/train_xgb_ote.py", "train_xgb_ote.py"),
    ("scripts/train_lgb_ote.py", "train_lgb_ote.py"),
    ("scripts/train_cat_ote.py", "train_cat_ote.py"),
    ("scripts/train_xgb_ote_magic.py", "train_xgb_ote_magic.py"),
    ("scripts/train_lgb_ote_shallow.py", "train_lgb_ote_shallow.py"),
    ("scripts/train_lgb_ote_deep.py", "train_lgb_ote_deep.py"),
    ("scripts/train_lr_ote.py", "train_lr_ote.py"),
    ("scripts/train_lr_elastic_ote.py", "train_lr_elastic_ote.py"),
    ("scripts/train_knn_ote.py", "train_knn_ote.py"),
    ("scripts/train_extratrees_ote.py", "train_extratrees_ote.py"),
    # cuML diversity models on g4dn
    ("scripts/train_rf_ote_cuml.py", "train_rf_ote_cuml.py"),
    ("scripts/train_svm_ote_cuml.py", "train_svm_ote_cuml.py"),
    ("aws/train_gnb_ote_cuml.py", "train_gnb_ote_cuml.py"),
    ("aws/train_knn_ote_cuml.py", "train_knn_ote_cuml.py"),
    ("aws/train_knn5_lite_cuml.py", "train_knn5_lite_cuml.py"),
    ("aws/train_lr_ote_cuml.py", "train_lr_ote_cuml.py"),
    ("aws/train_lr_l1_ote_cuml.py", "train_lr_l1_ote_cuml.py"),
    # Kaggle GPU XGB shallow (notebook source)
    ("notebooks/xgb_ote_shallow_gpu/xgb_ote_shallow_gpu.py", "xgb_ote_shallow_gpu.py"),
    # Ensemble level 2
    ("scripts/train_ensemble.py", "train_ensemble.py"),
    ("scripts/ensemble_v4.py", "ensemble_v4.py"),
    # Launcher reference (single representative example)
    ("scripts/launch_ensemble_v15.py", "launch_ensemble_v15.py"),
]


def main() -> None:
    """Copy each script into the dataset folder and write metadata.

    Raises:
        FileNotFoundError: if any source script is missing on disk.
    """
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "kaggle_datasets" / "ps6e4-training-code-41"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied: list[str] = []
    for rel_src, dst_name in FILES:
        src = repo / rel_src
        if not src.exists():
            missing.append(str(src))
            continue
        dst = out_dir / dst_name
        shutil.copy2(src, dst)
        copied.append(dst_name)

    if missing:
        raise FileNotFoundError("Missing source files:\n  " + "\n  ".join(missing))

    metadata = {
        "title": "PS6E4 Training Code (41 Models, 12th Place)",
        "id": "wguesdon/ps6e4-training-code-41",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Staged {len(copied)} scripts in {out_dir}:")
    for name in sorted(copied):
        print(f"  {name}")


if __name__ == "__main__":
    main()
