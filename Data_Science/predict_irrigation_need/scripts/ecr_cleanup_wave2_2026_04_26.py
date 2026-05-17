"""ECR cleanup wave 2: delete repos for finished competitions.

Deletes 7 repos for competitions or experiments that are done. Based on
explicit user approval on 2026-04-26.

  - kaggle-ps4e11-* (4 repos): PS4E11 finished, last push 2026-03-10
  - kaggle-training/protein-localization: comp finished
  - kaggle-training/cibmtr-nn: comp finished
  - kaggle-aimo3-gemma4: empty repo, never used

Expected saving: ~$1.72/month.

Irreversible. Idempotent: re-running after completion is a no-op.
"""
import boto3

REPOS_TO_DELETE = [
    "kaggle-ps4e11-xgb",
    "kaggle-ps4e11-cat",
    "kaggle-ps4e11-lgb",
    "kaggle-ps4e11-autogluon",
    "kaggle-training/protein-localization",
    "kaggle-training/cibmtr-nn",
    "kaggle-aimo3-gemma4",
]


def main():
    ecr = boto3.client("ecr", region_name="us-east-1")
    print(f"Deleting {len(REPOS_TO_DELETE)} repos...\n")

    for repo in REPOS_TO_DELETE:
        try:
            ecr.delete_repository(repositoryName=repo, force=True)
            print(f"  DELETED  {repo}")
        except ecr.exceptions.RepositoryNotFoundException:
            print(f"  skip     {repo}  (already gone)")
        except Exception as e:
            print(f"  FAIL     {repo}  {e}")

    # Verify
    print("\n=== Verification ===")
    remaining = {r["repositoryName"] for r in ecr.describe_repositories()["repositories"]}
    leftover = [r for r in REPOS_TO_DELETE if r in remaining]
    if leftover:
        print(f"  WARNING: still present: {leftover}")
    else:
        print("  All target repos removed.")
    print(f"  Total repos remaining: {len(remaining)}")


if __name__ == "__main__":
    main()
