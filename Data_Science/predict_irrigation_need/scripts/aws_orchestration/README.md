# AWS orchestration (reference only)

These scripts were used during the competition to fan out training jobs on AWS SageMaker, manage ECR images, and run the showcase pipeline on EC2. They reference IAM roles, S3 buckets, ECR repositories, and SageMaker training images that are specific to my AWS account, so they will not run as-is for anyone else.

They are kept here as a record of the orchestration layer behind the 30 to 41 model ensemble. The runnable portfolio artefact is the showcase notebook at `notebooks/ps6e4_showcase_v3.ipynb`.

| Script | Purpose |
|---|---|
| `launch_cpu_models.py`, `launch_cuml_*`, `launch_ensemble_v*`, `launch_pseudo.py`, `launch_rf_cuml.py` | Submit SageMaker training jobs for each base model family. |
| `ecr_cleanup_*.py` | Prune old container images from ECR after the competition. |
| `check_jobs.py` | Poll SageMaker for in-flight job statuses. |
| `inspiron_run.sh`, `run_showcase_ec2.sh` | Wrapper scripts to run training on the local Inspiron workstation and on a GPU EC2 host respectively. |
| `update_model_db.py` | Sync the local `experiments.db` SQLite store with completed SageMaker jobs. |
