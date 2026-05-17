#!/usr/bin/env python
"""Retroactively register models and ensembles into experiments.db.

Adds all the models trained since 2026-04-10 that weren't logged:
- 2 LGB OTE variants (shallow, deep)
- 7 cuML models (rf, svm, gnb, knn15, knn5_lite, lr, lr_l1)
- 3 ensembles (v11, v12, v13, v14_blend, v14_v11_seed3)

Usage: uv run python Playground_Series/PS6E4/scripts/update_model_db.py
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "experiments.db"
NOW = datetime.now().isoformat()

# (name, version, model_type, level, cv, lb_public, lb_private, notes)
ENTRIES = [
    # Wave 0: LGB OTE variants on SageMaker
    ("lgb_ote_shallow", 1, "lightgbm", 1, 0.97048, None, None,
     "depth=2, leaves=4, n_est=3000, lr=0.03. Weak learner for stacker diversity."),
    ("lgb_ote_deep", 1, "lightgbm", 1, 0.97660, None, None,
     "depth=12, leaves=256, n_est=1500, strong regularization."),

    # Wave 1: cuML on g4dn GPU via custom RAPIDS image
    ("rf_ote", 1, "cuml_rf", 1, 0.96005, None, None,
     "cuML RandomForest, 2x OTE augmentation, 2000 trees, max_depth=16."),
    ("svm_ote", 1, "cuml_svm", 1, 0.96154, None, None,
     "cuML SVM RBF, C=10.0, probability=True, no 4x augmentation."),
    ("gnb_ote", 1, "cuml_nb", 1, 0.90860, None, None,
     "cuML GaussianNB + OTE + StandardScaler. Fast, independence assumption."),
    ("knn15_ote", 1, "cuml_knn", 1, 0.71937, None, None,
     "cuML KNN k=15 on 332 OTE features. Curse of dimensionality hurt this one."),

    # Wave 2: cuML LR and KNN variants
    ("lr_cuml", 1, "cuml_lr", 1, 0.95522, None, None,
     "cuML LogisticRegression L2 C=0.1. Stronger reg than sklearn LR C=1.0."),
    ("lr_l1_cuml", 1, "cuml_lr", 1, 0.95554, None, None,
     "cuML LR L1 C=0.1 (Lasso). Sparse feature selection."),
    ("knn5_lite", 1, "cuml_knn", 1, 0.90700, None, None,
     "cuML KNN k=5 without digit features. Big improvement over knn15."),

    # Ensembles
    ("ensemble_v11_lgb", 1, "ensemble_stacker", 2, 0.98061, 0.98072, None,
     "LGB stacker on 30 models + log_bias thresholds."),
    ("ensemble_v12_lgb", 1, "ensemble_stacker", 2, 0.98051, 0.98074, None,
     "LGB stacker on 32 models (30 + lgb_ote_shallow/deep) + log_bias."),
    ("ensemble_v12_greedy", 1, "ensemble_greedy", 2, 0.98086, 0.97979, None,
     "Greedy forward selection 9m + diffevol thresholds."),
    ("ensemble_v13_lgb", 1, "ensemble_stacker", 2, 0.98058, 0.98042, None,
     "LGB stacker on 41 models (32 + 7 cuML) + log_bias."),
    ("ensemble_v13_lgb_gfs", 1, "ensemble_stacker", 2, 0.98045, 0.98047, None,
     "LGB stacker on v13 greedy-selected subset (9m)."),
    ("ensemble_v14_blend_v11v12", 1, "ensemble_blend", 2, 0.98055, 0.98056, None,
     "Average v11+v12 LGB stacker OOF probabilities + log_bias thresholds."),
    ("ensemble_v14_v11_seed3", 1, "ensemble_stacker", 2, 0.98061, None, None,
     "LGB stacker on v11 pool with 3-seed averaging (42, 43, 44)."),
]


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check which names already exist
    existing = {r[0] for r in cur.execute("SELECT name FROM models").fetchall()}

    added = 0
    skipped = 0
    for entry in ENTRIES:
        name, version, mtype, level, cv, lb_pub, lb_priv, notes = entry
        if name in existing:
            # Update CV/LB if provided (idempotent-ish)
            cur.execute(
                "UPDATE models SET cv_score=?, lb_score_public=?, lb_score_private=?, notes=? WHERE name=?",
                (cv, lb_pub, lb_priv, notes, name),
            )
            skipped += 1
            print(f"  Updated existing: {name}")
            continue

        cur.execute(
            """INSERT INTO models
               (name, version, model_type, level, cv_score, metric,
                lb_score_public, lb_score_private, notes, created_at)
               VALUES (?, ?, ?, ?, ?, 'balanced_accuracy', ?, ?, ?, ?)""",
            (name, version, mtype, level, cv, lb_pub, lb_priv, notes, NOW),
        )
        added += 1
        print(f"  Added: {name}  CV={cv}  LB={lb_pub}")

    conn.commit()
    print(f"\nAdded {added} new, updated {skipped} existing. Total rows: "
          f"{cur.execute('SELECT COUNT(*) FROM models').fetchone()[0]}")

    # Show leaderboard of ensembles
    print("\n=== Ensemble leaderboard ===")
    rows = cur.execute(
        """SELECT name, cv_score, lb_score_public FROM models
           WHERE model_type LIKE 'ensemble%'
           ORDER BY COALESCE(lb_score_public, 0) DESC, cv_score DESC"""
    ).fetchall()
    for name, cv, lb in rows:
        print(f"  {name:40s}  CV: {cv:.5f}  LB: {lb if lb else '---':>7}")

    conn.close()


if __name__ == "__main__":
    main()
