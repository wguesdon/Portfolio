"""ECR cleanup: Tier A (untagged in kaggle-training/*) + Tier B (delete PS6E4 repos).

Run from /mnt/data/Github/Kaggle. Requires AWS creds in env.

Tier A: delete all untagged images in repos prefixed with "kaggle-training".
        Reusable base images. Tagged images (latest, v3, etc.) are preserved.

Tier B: delete entire repos prefixed with "kaggle-ps6e4". Comp ends 2026-04-30.
        Final picks are locked, no further experiments planned.

Both operations are irreversible. Script is idempotent: re-running after
completion does nothing.
"""
import boto3


def get_repos(ecr):
    """Return list of repo names in the current account/region."""
    paginator = ecr.get_paginator("describe_repositories")
    repos = []
    for page in paginator.paginate():
        repos.extend(r["repositoryName"] for r in page["repositories"])
    return repos


def list_untagged(ecr, repo):
    """Return imageDigest list of untagged images in repo."""
    paginator = ecr.get_paginator("describe_images")
    digests = []
    for page in paginator.paginate(repositoryName=repo):
        for img in page["imageDetails"]:
            if not img.get("imageTags"):
                digests.append(img["imageDigest"])
    return digests


def batch_delete(ecr, repo, digests):
    """Delete images in chunks of 100 (ECR batch limit). Returns total deleted bytes."""
    if not digests:
        return 0, 0
    total_deleted = 0
    failed = 0
    for i in range(0, len(digests), 100):
        chunk = digests[i : i + 100]
        resp = ecr.batch_delete_image(
            repositoryName=repo,
            imageIds=[{"imageDigest": d} for d in chunk],
        )
        total_deleted += len(resp.get("imageIds", []))
        failed += len(resp.get("failures", []))
        for f in resp.get("failures", []):
            print(f"    FAIL {repo} {f['imageId'].get('imageDigest', '?')[:19]}: {f['failureReason']}")
    return total_deleted, failed


def tier_a(ecr, repos):
    """Delete untagged images in kaggle-training/* repos."""
    targets = [r for r in repos if r.startswith("kaggle-training")]
    print(f"\n=== Tier A: delete untagged images in {len(targets)} kaggle-training/* repos ===")
    grand_deleted = 0
    grand_failed = 0
    for repo in sorted(targets):
        digests = list_untagged(ecr, repo)
        if not digests:
            print(f"  {repo}: 0 untagged (skip)")
            continue
        deleted, failed = batch_delete(ecr, repo, digests)
        print(f"  {repo}: deleted {deleted}/{len(digests)} untagged ({failed} failed)")
        grand_deleted += deleted
        grand_failed += failed
    print(f"\n  Tier A total: {grand_deleted} images deleted, {grand_failed} failures")


def tier_b(ecr, repos):
    """Delete entire kaggle-ps6e4-* repos."""
    targets = [r for r in repos if r.startswith("kaggle-ps6e4")]
    print(f"\n=== Tier B: delete {len(targets)} kaggle-ps6e4-* repos entirely ===")
    for repo in sorted(targets):
        try:
            ecr.delete_repository(repositoryName=repo, force=True)
            print(f"  {repo}: DELETED")
        except ecr.exceptions.RepositoryNotFoundException:
            print(f"  {repo}: not found (already deleted)")


def main():
    ecr = boto3.client("ecr", region_name="us-east-1")
    repos = get_repos(ecr)
    print(f"Found {len(repos)} ECR repositories")

    tier_a(ecr, repos)
    tier_b(ecr, repos)

    # Verify
    print("\n=== Verification ===")
    repos_after = get_repos(ecr)
    print(f"  Repos remaining: {len(repos_after)} (was {len(repos)})")
    ps6e4_remaining = [r for r in repos_after if r.startswith("kaggle-ps6e4")]
    if ps6e4_remaining:
        print(f"  WARNING: PS6E4 repos still present: {ps6e4_remaining}")
    else:
        print("  All PS6E4 repos removed.")


if __name__ == "__main__":
    main()
