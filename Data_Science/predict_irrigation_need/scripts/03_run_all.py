"""Run all baseline GBDT models.

Launches XGBoost, LightGBM, and CatBoost training sequentially
or prints commands for parallel execution.

Usage:
    # Run all sequentially:
    uv run python Playground_Series/PS6E4/scripts/03_run_all.py

    # Run in parallel (3 terminals):
    uv run python Playground_Series/PS6E4/scripts/03a_train_xgb.py
    uv run python Playground_Series/PS6E4/scripts/03b_train_lgb.py
    uv run python Playground_Series/PS6E4/scripts/03c_train_cat.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_script(script_name: str):
    """Run a training script as a subprocess."""
    script = SCRIPTS_DIR / script_name
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {script_name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRIPTS_DIR),
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent.parent / "tabml")},
    )
    if result.returncode != 0:
        print(f"  {script_name} FAILED with exit code {result.returncode}")
    return result.returncode


def show_leaderboard():
    """Display the current model leaderboard."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "tabml"))
    from tabml.tracking import ModelTracker

    db_path = str(SCRIPTS_DIR.parent / "experiments.db")
    tracker = ModelTracker(db_path=db_path)
    lb = tracker.get_leaderboard()
    if len(lb) > 0:
        print(f"\n{'=' * 60}")
        print("  MODEL LEADERBOARD")
        print(f"{'=' * 60}")
        cols = ["name", "model_type", "cv_score", "cv_std", "training_time_seconds"]
        display_cols = [c for c in cols if c in lb.columns]
        print(lb[display_cols].to_string(index=False))
    else:
        print("\nNo models logged yet.")


if __name__ == "__main__":
    scripts = [
        "03a_train_xgb.py",
        "03b_train_lgb.py",
        "03c_train_cat.py",
    ]

    for script in scripts:
        rc = run_script(script)
        if rc != 0:
            print(f"Stopping due to failure in {script}")
            break

    show_leaderboard()
