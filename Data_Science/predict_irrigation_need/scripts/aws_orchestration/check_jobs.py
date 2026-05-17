#!/usr/bin/env python
"""Check SageMaker ps6e4 job status and download predictions from completed jobs.

Usage:
    uv run python Playground_Series/PS6E4/scripts/check_jobs.py
    uv run python Playground_Series/PS6E4/scripts/check_jobs.py --download
    uv run python Playground_Series/PS6E4/scripts/check_jobs.py --max-results 30
"""
import argparse
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION = "us-east-1"
# Match jobs by these prefixes. The pytorch-training entries are cuML jobs
# that launched before base_job_name was set in the launch script.
JOB_PREFIXES = ("ps6e4", "pytorch-training-2026-04-10-23-58")
PRED_DIR = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4/predictions")


def get_jobs(sm_client, max_results=20):
    """Fetch recent SageMaker training jobs matching the prefix."""
    jobs = sm_client.list_training_jobs(
        MaxResults=max_results,
        SortBy="CreationTime",
        SortOrder="Descending",
    )
    return [
        j for j in jobs["TrainingJobSummaries"]
        if any(p in j["TrainingJobName"] for p in JOB_PREFIXES)
    ]


def print_status(sm_client, max_results=20):
    """Print a status table of recent jobs."""
    now = datetime.now(timezone.utc)
    jobs = get_jobs(sm_client, max_results)

    print(f"\n{'Status':<8} {'Job Name':<55} {'Instance':<18} {'Info'}")
    print("-" * 110)

    for j in jobs:
        name = j["TrainingJobName"]
        status = j["TrainingJobStatus"]
        detail = sm_client.describe_training_job(TrainingJobName=name)
        inst = detail["ResourceConfig"]["InstanceType"]
        elapsed_h = (now - detail["CreationTime"]).total_seconds() / 3600
        runtime_s = detail.get("TrainingTimeInSeconds", 0)

        if status == "Completed":
            print(f"  DONE  {name:<55} {inst:<18} runtime={runtime_s // 60}m")
        elif status == "Failed":
            reason = detail.get("FailureReason", "")[:80]
            print(f"  FAIL  {name:<55} {inst:<18} {reason}")
        elif status == "InProgress":
            secondary = detail.get("SecondaryStatus", "")
            print(f"  RUN   {name:<55} {inst:<18} {secondary} ({elapsed_h:.1f}h)")
        elif status == "Stopped":
            print(f"  STOP  {name:<55} {inst:<18}")
        else:
            print(f"  {status:<6} {name:<55} {inst:<18}")

    print()


def download_predictions(sm_client, s3_client, max_results=20):
    """Download oof/pred .npy files from completed jobs not yet in predictions/."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in PRED_DIR.glob("*.npy")}

    jobs = get_jobs(sm_client, max_results)
    completed = [
        j for j in jobs if j["TrainingJobStatus"] == "Completed"
    ]

    downloaded = 0
    for j in completed:
        name = j["TrainingJobName"]
        detail = sm_client.describe_training_job(TrainingJobName=name)
        artifact_uri = detail.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        if not artifact_uri:
            continue

        bucket, key = artifact_uri.replace("s3://", "").split("/", 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "model.tar.gz")
            try:
                s3_client.download_file(bucket, key, tar_path)
            except Exception as e:
                print(f"  Skipping {name}: {e}")
                continue

            with tarfile.open(tar_path, "r:gz") as tar:
                npy_members = [m for m in tar.getmembers() if m.name.endswith(".npy")]
                if not npy_members:
                    continue

                # Check if all files already exist
                all_exist = all(
                    os.path.basename(m.name) in existing for m in npy_members
                )
                if all_exist:
                    continue

                tar.extractall(tmpdir, members=npy_members, filter="data")

                for m in npy_members:
                    src = os.path.join(tmpdir, m.name)
                    dst = PRED_DIR / os.path.basename(m.name)
                    if dst.name not in existing:
                        shutil.copy2(src, str(dst))
                        print(f"  Downloaded: {dst.name}  (from {name})")
                        downloaded += 1

    if downloaded == 0:
        print("  No new predictions to download.")
    else:
        print(f"\n  Downloaded {downloaded} file(s) to {PRED_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Check ps6e4 SageMaker jobs")
    parser.add_argument(
        "--download", action="store_true",
        help="Download predictions from completed jobs",
    )
    parser.add_argument(
        "--max-results", type=int, default=20,
        help="Max recent jobs to check (default: 20)",
    )
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    print_status(sm, args.max_results)

    if args.download:
        s3 = boto3.client("s3", region_name=REGION)
        print("Checking for new predictions to download...")
        download_predictions(sm, s3, args.max_results)


if __name__ == "__main__":
    main()
