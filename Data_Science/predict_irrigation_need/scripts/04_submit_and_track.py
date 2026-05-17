#!/usr/bin/env python
"""Submit predictions to Kaggle and update the local model tracker.

Usage:
    # Submit a single model
    PYTHONPATH=/mnt/data/Github/tabml uv run python scripts/04_submit_and_track.py \
        --model xgb --cv-score 0.9624

    # Submit all models with results JSON from AWS
    PYTHONPATH=/mnt/data/Github/tabml uv run python scripts/04_submit_and_track.py \
        --from-aws

    # Just update LB score for an existing model
    PYTHONPATH=/mnt/data/Github/tabml uv run python scripts/04_submit_and_track.py \
        --model xgb --lb-score 0.9600

    # Show leaderboard
    PYTHONPATH=/mnt/data/Github/tabml uv run python scripts/04_submit_and_track.py \
        --leaderboard
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "tabml"))

from tabml.tracking import ModelTracker

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = BASE_DIR / "predictions"
DATA_DIR = BASE_DIR / "data" / "raw"
AWS_OUTPUT_DIR = BASE_DIR / "output" / "aws"
DB_PATH = str(BASE_DIR / "experiments.db")

COMPETITION = "playground-series-s6e4"
TARGET_ORDER = ["Low", "Medium", "High"]


def load_test_ids() -> pd.Series:
    """Load test set IDs."""
    test = pd.read_csv(DATA_DIR / "test.csv")
    return test["id"]


def make_submission(pred_path: str, output_path: str) -> str:
    """Convert probability predictions to submission CSV.

    Args:
        pred_path: Path to .npy file with shape (n_samples, 3) probabilities.
        output_path: Path to write submission CSV.

    Returns:
        Path to the written submission CSV.
    """
    preds = np.load(pred_path)
    test_ids = load_test_ids()

    le = LabelEncoder()
    le.fit(TARGET_ORDER)

    labels = le.inverse_transform(preds.argmax(axis=1))
    submission = pd.DataFrame({"id": test_ids, "Irrigation_Need": labels})
    submission.to_csv(output_path, index=False)
    print(f"  Submission saved: {output_path} ({len(submission)} rows)")
    return output_path


def submit_to_kaggle(csv_path: str, message: str) -> float:
    """Submit CSV to Kaggle and return the public LB score.

    Args:
        csv_path: Path to submission CSV.
        message: Submission description message.

    Returns:
        The public leaderboard score.
    """
    print(f"  Submitting to Kaggle: {message}")
    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", COMPETITION,
            "-f", csv_path,
            "-m", message,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Submit error: {result.stderr}")
        raise RuntimeError(f"Kaggle submit failed: {result.stderr}")
    print(f"  {result.stdout.strip()}")

    # Wait for scoring
    print("  Waiting for Kaggle scoring...", end="", flush=True)
    for _ in range(30):
        time.sleep(10)
        print(".", end="", flush=True)
        score = get_latest_lb_score()
        if score is not None:
            print(f" done! LB={score:.5f}")
            return score
    print(" timeout (check manually)")
    return None


def get_latest_lb_score() -> float:
    """Get the most recent submission's public score from Kaggle.

    Returns:
        The public score as a float, or None if not yet scored.
    """
    result = subprocess.run(
        [
            "kaggle", "competitions", "submissions",
            "-c", COMPETITION,
            "--csv",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    import io
    df = pd.read_csv(io.StringIO(result.stdout))
    if df.empty:
        return None

    latest = df.iloc[0]
    score = latest.get("publicScore")
    if score is not None and str(score) != "" and str(score) != "None":
        return float(score)
    return None


def log_model_from_results(results_path: str, tracker: ModelTracker) -> str:
    """Log a model to the tracker from its results JSON.

    Args:
        results_path: Path to the _results.json file from training.
        tracker: ModelTracker instance.

    Returns:
        The model name that was logged.
    """
    with open(results_path) as f:
        results = json.load(f)

    model_short = results["model"]
    model_type = results.get("model_type", model_short)
    cv_score = results["cv_balanced_accuracy"]
    fold_scores = results["fold_scores"]
    best_params = results.get("best_params", {})
    n_features = results.get("n_features", 0)
    version = 100  # AWS v100 series

    name = f"{model_short}_v{version}"

    oof_path = str(PRED_DIR / f"oof_{model_short}.npy")
    pred_path = str(PRED_DIR / f"pred_{model_short}.npy")

    # Check if OOF exists (may have been renamed from AWS format)
    for pattern in [f"{model_short}_oof.npy", f"oof_{model_short}.npy"]:
        candidate = PRED_DIR / pattern
        if candidate.exists():
            oof_path = str(candidate)
            break

    for pattern in [f"{model_short}_test.npy", f"pred_{model_short}.npy"]:
        candidate = PRED_DIR / pattern
        if candidate.exists():
            pred_path = str(candidate)
            break

    try:
        tracker.log_model(
            name=name,
            version=version,
            model_type=model_type,
            cv_score=cv_score,
            fold_scores=fold_scores,
            metric="balanced_accuracy",
            params=best_params,
            feature_group="aws_engineered",
            n_features=n_features,
            oof_path=oof_path,
            test_pred_path=pred_path,
            notes=f"AWS SageMaker training, {results.get('n_trials', '?')} Optuna trials",
            level=2,
        )
        print(f"  Logged {name}: CV={cv_score:.5f}")
    except Exception as e:
        print(f"  Warning: could not log {name}: {e}")

    return name


def process_aws_results(tracker: ModelTracker):
    """Process all AWS training results: log to tracker, submit to Kaggle."""
    # Look for results JSONs in predictions/ and output/aws/
    results_files = list(PRED_DIR.glob("*_results.json")) + list(
        AWS_OUTPUT_DIR.glob("*_results.json")
    )

    if not results_files:
        print("No results JSON files found. Run './run.sh sync' first.")
        return

    for results_path in results_files:
        print(f"\nProcessing: {results_path.name}")
        with open(results_path) as f:
            results = json.load(f)

        model_short = results["model"]
        name = log_model_from_results(str(results_path), tracker)

        # Find test predictions and submit
        pred_file = None
        for pattern in [
            f"{model_short}_test.npy",
            f"pred_{model_short}.npy",
        ]:
            candidate = PRED_DIR / pattern
            if candidate.exists():
                pred_file = candidate
                break

        if pred_file is None:
            print(f"  No test predictions found for {model_short}, skipping submission")
            continue

        # Create submission CSV
        sub_dir = BASE_DIR / "submissions"
        sub_dir.mkdir(exist_ok=True)
        csv_path = str(sub_dir / f"sub_{name}.csv")
        make_submission(str(pred_file), csv_path)

        # Submit to Kaggle
        cv_score = results["cv_balanced_accuracy"]
        lb_score = submit_to_kaggle(
            csv_path, f"{name} CV={cv_score:.5f}"
        )

        # Update tracker with LB score
        if lb_score is not None:
            tracker.update_lb_score(name, lb_score, "public")
            print(f"  Updated tracker: {name} LB={lb_score:.5f}")


def show_leaderboard(tracker: ModelTracker):
    """Display the model leaderboard."""
    df = tracker.get_leaderboard()
    if df.empty:
        print("No models in tracker yet.")
        return

    cols = ["name", "model_type", "cv_score", "cv_std", "lb_score_public", "n_features", "level"]
    available = [c for c in cols if c in df.columns]
    print("\n" + df[available].to_string(index=False))
    print(f"\n{len(df)} models tracked")


def main():
    parser = argparse.ArgumentParser(description="Submit to Kaggle and update tracker")
    parser.add_argument("--model", help="Model short name (e.g. xgb, lgb, cat)")
    parser.add_argument("--cv-score", type=float, help="CV score to log")
    parser.add_argument("--lb-score", type=float, help="Manually set LB score")
    parser.add_argument("--from-aws", action="store_true", help="Process all AWS results")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--submit-only", action="store_true", help="Submit without logging")
    args = parser.parse_args()

    tracker = ModelTracker(db_path=DB_PATH)

    if args.leaderboard:
        show_leaderboard(tracker)
        return

    if args.from_aws:
        process_aws_results(tracker)
        show_leaderboard(tracker)
        return

    if args.model and args.lb_score:
        tracker.update_lb_score(args.model, args.lb_score, "public")
        print(f"Updated {args.model} LB={args.lb_score:.5f}")
        show_leaderboard(tracker)
        return

    if args.model:
        pred_file = PRED_DIR / f"pred_{args.model}.npy"
        if not pred_file.exists():
            pred_file = PRED_DIR / f"{args.model}_test.npy"
        if not pred_file.exists():
            print(f"No prediction file found for {args.model}")
            return

        sub_dir = BASE_DIR / "submissions"
        sub_dir.mkdir(exist_ok=True)
        name = args.model
        csv_path = str(sub_dir / f"sub_{name}.csv")
        make_submission(str(pred_file), csv_path)

        lb_score = submit_to_kaggle(csv_path, f"{name} CV={args.cv_score or '?'}")

        if lb_score is not None and not args.submit_only:
            tracker.update_lb_score(name, lb_score, "public")
            print(f"Updated tracker: {name} LB={lb_score:.5f}")

        show_leaderboard(tracker)


if __name__ == "__main__":
    main()
