"""Build the 41 model OOF predictions Kaggle dataset folder.

Copies oof_*.npy and pred_*.npy for the v15 model list (the 12th place winning
submission) plus irrigation_prediction.csv into a dataset upload folder. Also
writes the dataset-metadata.json.

Run:
    uv run python Playground_Series/PS6E4/scripts/build_41model_dataset.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# v15 model list (the 41 models in the LB 0.98082 winning submission)
V15_MODELS: list[str] = [
    # v3 pipeline GBDTs (14)
    "xgb_v2", "xgb_s43", "xgb_s44",
    "xgb-2026-04-03-11-12-38-123", "xgb-2026-04-03-11-12-41-698",
    "lgb_v2", "lgb_s43", "lgb_s44",
    "cat_v2", "cat_s43", "cat_s44",
    "cat-2026-04-03-12-44-01-719", "cat-2026-04-03-12-44-05-363",
    "cat-2026-04-03-13-29-13-994",
    # v3 pipeline mixed (7)
    "realmlp_mahog", "lgb_v5", "xgb_v5",
    "realmlp_v3fix", "tabm_v3fix",
    "lgb_digit", "cat_digit",
    # OTE pipeline GBDTs (6)
    "xgb_ote", "cat_ote", "lgb_ote",
    "xgb_ote_s43", "xgb_ote_s44", "xgb_ote_magic",
    # OTE diversity (3)
    "lr_ote", "knn_ote", "et_ote",
    # LGB OTE depth variants (2)
    "lgb_ote_shallow", "lgb_ote_deep",
    # cuML diversity (4)
    "rf_ote", "svm_ote", "gnb_ote", "knn15_ote",
    # cuML LR / lite KNN (3)
    "lr_cuml", "lr_l1_cuml", "knn5_lite",
    # Late additions to v15 (2)
    "xgb_ote_shallow_gpu", "lr_elastic",
]


def main() -> None:
    """Stage all dataset files into the upload folder.

    Verifies each `oof_<model>.npy` and `pred_<model>.npy` exists in the local
    predictions directory and copies it into the dataset folder.

    Raises:
        SystemExit: if any required file is missing.
    """
    assert len(V15_MODELS) == 41, f"expected 41 models, got {len(V15_MODELS)}"

    repo = Path(__file__).resolve().parents[1]
    pred_dir = repo / "predictions"
    out_dir = repo / "kaggle_datasets" / "ps6e4-41model-oof-predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for model in V15_MODELS:
        for kind in ("oof", "pred"):
            src = pred_dir / f"{kind}_{model}.npy"
            if not src.exists():
                missing.append(str(src))
                continue
            dst = out_dir / f"{kind}_{model}.npy"
            shutil.copy2(src, dst)

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    irrig_src = repo / "kaggle_datasets" / "ps6e4-30model-oof-predictions" / "irrigation_prediction.csv"
    if irrig_src.exists():
        shutil.copy2(irrig_src, out_dir / "irrigation_prediction.csv")
    else:
        print(f"warning: irrigation_prediction.csv not found at {irrig_src}")

    metadata = {
        "title": "PS6E4 41-Model OOF Predictions (12th Place)",
        "id": "wguesdon/ps6e4-41model-oof-predictions",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    files = sorted(p.name for p in out_dir.iterdir())
    print(f"Staged {len(files)} files in {out_dir}:")
    for name in files:
        print(f"  {name}")


if __name__ == "__main__":
    main()
