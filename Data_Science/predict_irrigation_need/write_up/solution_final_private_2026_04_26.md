# PS6E4 Private Post Mortem

**Status:** picks locked, comp deadline 2026-04-30.
**Final LB:** 0.98081 (Pick 1, v15 LGB stacker).
**Date drafted:** 2026-04-26.

This is the internal version. It includes AWS cost, infra pain, what failed, and lessons for future Playground Series comps. The public solution post is in `solution_final_public_2026_04_26.md`.

## 1. Final picks (locked on Kaggle)

| Pick | File | Method | Models | CV | Public LB | Gap |
|-----:|------|--------|------:|----:|----------:|----:|
| 1 | `sub_v15_lgb_stack_log_bias.csv` | LGB stack + log bias | 41 | 0.98045 | 0.98081 | +0.00036 |
| 2 | `sub_v11_lgb_stack_30m.csv`      | LGB stack + log bias | 30 | 0.98061 | 0.98072 | +0.00011 |

Both are LGB stackers with log bias threshold tuning. Pick 1 has the highest public LB. Pick 2 has the tighter CV to LB gap and the highest CV among proven LB stackers.

## 2. Submission ledger

The full LB ladder during the comp.

| Submission | Method | Models | CV | LB | Gap |
|-----------|--------|------:|----:|----:|----:|
| **v15 LGB stacker**    | stacker + log_bias | 41 | 0.98045 | **0.98081** | +0.00036 |
| **v11 LGB stacker**    | stacker + log_bias | 30 | 0.98061 | 0.98072 | +0.00011 |
| v12 LGB stacker        | stacker + log_bias | 32 | 0.98051 | 0.98074 | +0.00023 |
| v17 LGB stacker        | stacker + log_bias | 43 | 0.98057 | 0.98041 | -0.00016 |
| v13 LGB stacker        | stacker + log_bias | 41 | 0.98058 | 0.98042 | -0.00016 |
| v9 LGB stacker         | stacker + log_bias | 27 | 0.98045 | 0.98042 | -0.00003 |
| v10 LGB stacker        | stacker + log_bias | 29 | 0.98046 | 0.98036 | -0.00010 |
| v6 greedy 3m           | greedy diffevol    |  3 | 0.98067 | 0.98023 | -0.00044 |
| v7 LGB stacker         | stacker + log_bias | 24 | 0.98054 | 0.98008 | -0.00046 |
| v6 LGB stacker         | stacker + log_bias | 22 | 0.98038 | 0.97995 | -0.00043 |
| v7 greedy 7m           | greedy diffevol    |  7 | 0.98084 | 0.97988 | -0.00096 |
| v10 greedy 9m          | greedy diffevol    |  9 | 0.98086 | 0.97979 | -0.00107 |
| v8 greedy 8m           | greedy diffevol    |  8 | 0.98081 | 0.97973 | -0.00108 |
| v4d LGB stacker        | stacker + log_bias | 21 | 0.98026 | 0.97972 | -0.00054 |
| v5 LGB stacker         | stacker + log_bias | 22 | 0.98029 | 0.97962 | -0.00067 |

The pattern is unambiguous.

- LGB stackers cluster around a narrow CV to LB gap.
- Greedy methods at any pool size show large negative gaps.
- Adding diversity helps stackers up to a saturation point near 30 to 41 models. Past that, marginal CV gains do not translate to LB.

## 3. Key inflection points

| Date | Event | Effect |
|------|-------|--------|
| 2026-04-08 | OTE pipeline ported from emanuellcs | Top 3 single models came from it. v6 LB 0.98023, +0.00051 over previous best. |
| 2026-04-10 | Added ExtraTrees + LR + KNN to the pool | v11 LB 0.98072. The 3 weak models added stacker signal. |
| 2026-04-11 | Added LGB shallow + LGB deep + xgb_ote_shallow_gpu | v15 LB 0.98081, current best. |
| 2026-04-12 | Added 7 cuML models (RF, SVM, GNB, KNN, LR variants) | v13 LB 0.98042. Hurt LB. |
| 2026-04-13 | v17 with 43 models. CV +0.00012 over v15 | LB 0.98041. Confirmed saturation. |

## 4. Infra lessons

### What worked

- **EC2 spot jumpbox for ECR push.** Built and pushed Docker images from a c5.4xlarge spot in us-east-1. Push took ~5 min vs 25+ min from home network. The build often took 10 min, push was the bottleneck without it.
- **SageMaker spot for all training.** Default `use_spot_instances=False` was used for sub 8 hour jobs because spot setup overhead and interruption risk were not worth it at this scale. Spot was used selectively for very long jobs.
- **AutoGluon Deep Learning Container** as the base for all GBDT and ensemble jobs. `763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.2-gpu-py311`. Wide enough to run sklearn, LGB, XGB, CatBoost, custom Python without rebuilding.
- **Reusable per algo ECR repos** under `kaggle-training-*`. One built image per family, reused across PS6E4 / PS4E11 / future comps.
- **Inspiron i7 12 core 62 GB as a free CPU job runner.** XGB shallow CPU finished at CV 0.97938, matching the Kaggle GPU run within 0.0001. Confirmed the "XGB hist CPU equals GPU" claim on this dataset.

### What hurt

- **RAPIDS cuML custom image build.** Could not pip install cuML on the PyTorch DLC because of numba and cudf version conflicts. Had to build from `nvcr.io/nvidia/rapidsai/base:26.02-cuda12-py3.12`. Fix: `USER root` in the Dockerfile, since SageMaker mounts `/opt/ml/model` as root owned and the RAPIDS base runs as UID 1001 by default. Lost ~1 day of debugging.
- **Kaggle P100 not supported by current PyTorch.** GNN training crashed with `sm_60 not supported`. Moved to SageMaker T4. Added cost and a long debug cycle.
- **TabPFN hit Kaggle 12 hour kernel limit.** Only 1 of 5 subsamples of fold 1 finished. T4 16 GB cannot fit 126K rows at once for predict_proba; needed batched inference at 5K rows. Even with that, the full 5 fold loop did not fit. Should have moved to SageMaker GPU with a larger card.
- **Multi arch manifest list confusion during ECR cleanup.** Untagged platform sub manifests cannot be deleted while the parent tag exists. ECR error message is misleading. Resolution: leave them, they are not orphans.

### AWS cost summary

Month to date through 2026-04-26.

| Service | Cost |
|---------|----:|
| SageMaker training | $248.58 |
| Tax | $62.61 |
| S3 | $26.88 |
| EC2 Other (NAT, EIP, EBS) | $12.67 |
| EC2 Compute | $10.75 |
| ECR | $8.65 |
| Other | ~$5 |
| **Total** | **$375.27** |

Daily idle was running about $2.62 to $3.69, almost entirely storage.

After the 2026-04-26 ECR cleanup (Tier A + Tier B), ECR storage cost dropped from $40.57/month to $7.05/month, a saving of $33.52/month.

The full breakdown of cost by phase was approximately:

- ~$80 on initial v3 pipeline build, base GBDT runs, and Optuna sweeps.
- ~$110 on OTE pipeline, multi seed runs, ensemble v6 to v9.
- ~$70 on v11 to v15 ensemble waves, including the GNN move from Kaggle to SageMaker.
- ~$40 on v13 to v17 stacker saturation experiments and cuML diversity wave (most of which did not move LB).
- ~$15 on the failed pseudo labelling run.
- ~$60 on tax, storage, and EC2.

## 5. What did not work

### Models that hurt LB

- **GNN (GraphSAGE, CV 0.97073).** Added to v5, hurt LB by 0.00010. Single model too weak relative to GBDT, no clean way for stacker to use it.
- **RealMLP on OTE features (CV 0.96509).** Required `fillna(0)` after OTE transform. Zero is a meaningful value here. Should have used median imputation per column. Never retried with the fix.
- **cuML diversity wave (RF, SVM, GNB, KNN, LR).** v13 added 7 cuML models. CV rose 0.00007, LB fell 0.00032. Stacker overweighted them on CV folds where they happened to align with majority class.

### Methods that overfit CV

- **Greedy forward selection.** Every greedy run had a higher CV than the matched stacker, and a worse LB. Pattern held across pool sizes from 3 to 9 models.
- **Hill climbing weight optimisation among only the 3 GBDT models.** Collapsed to 100% CatBoost. Useless without weak diverse models in the pool.

### Approaches that wasted compute

- **Pseudo labelling LGB.** 58 min on c5.9xlarge. CV 0.95390. Two failures: lost the 4x OTE shuffle augmentation, and used label agreement as the confidence filter rather than max probability. Should have kept 4x OTE concat and used `max_proba > 0.9`.
- **v16 correlation pruned stacker.** All our base models correlate above 0.995. Pruning at 0.99 or 0.995 produced an 11 model pool with worse CV than the full 43 model stacker.
- **v17 ensemble.** +0.00012 CV vs v15, -0.00040 LB. Confirmed the stacker was at saturation around 30 to 41 models.
- **TabPFN on Kaggle.** Two days of license, secret, and accelerator setup; crashed on the 12 hour limit. Architecturally still the most likely lever for future LB gains, but only with proper subsampling and on SageMaker GPU.

## 6. Lessons for the next Playground Series

1. **Lock down a per algorithm Docker image early.** Reuse it across comps. `USER root` and explicit CUDA version pinning saved hours when reused. New custom images cost roughly half a day each.
2. **Never run a greedy ensemble at production stage.** Use it as a sanity check on the model pool but never submit. Its CV inflation is structural.
3. **Stop adding models past 30 to 40 in a stacker.** The stacker saturates. Past that point each new model costs more in compute than it returns in LB.
4. **Trust 5 fold CV on full train over 20% public LB.** Anything below 0.0005 LB delta is noise. Public LB is for sanity, not selection.
5. **Pseudo labelling needs the same feature pipeline as the base model.** Lost augmentation steps silently kill performance. Verify the script reproduces the base CV before adding pseudo data.
6. **TabPFN and FT Transformer for diversity.** When stacker saturates on GBDT plus light NN, only true architectural diversity moves LB. Pre allocate SageMaker GPU budget for these from day one. Do not rely on Kaggle kernels for them.
7. **Original dataset is gold.** Target encodings derived from a clean original dataset are zero leakage and consistently top of feature importance. Always check whether the synthetic data has a clean ancestor.
8. **Magic formula discovery is worth a dedicated day.** Ours came mid comp. If discovered earlier, every model would have included those features from v1.
9. **Inspiron is a real free worker.** A 12 core local box matches a c5.9xlarge for any single XGB or LGB run, and hist CPU matches GPU on this dataset class. Reserve SageMaker for jobs that genuinely need GPU or massive RAM.
10. **Hold an honest "stop list".** Track methods that did not work in this file, not in the chat. Saves rerunning failed experiments months later.

## 7. Cleanup checklist for after 2026-04-30

- [x] ECR Tier A: delete untagged in `kaggle-training/*`. Done 2026-04-26.
- [x] ECR Tier B: delete `kaggle-ps6e4-*` repos. Done 2026-04-26.
- [ ] S3: archive `s3://kaggle-ps6e4/` to Glacier or delete. Currently 10.4 GiB. Costs ~$0.27/month while live, ~$0.04/month in Glacier, $0 if deleted.
- [ ] Local: delete `predictions/` artefacts not needed for the public notebook. Currently ~2 GB.
- [ ] Inspiron: clear `~/Documents/kaggle/PS6E4/` working dir.
- [ ] Lambda functions / EventBridge rules from the SageMaker auto stop scripts. Verify none remain.
- [ ] CloudWatch logs older than 30 days. Set retention to 7 days on `/aws/sagemaker/TrainingJobs` log group.
- [ ] Upload final two CSVs to a private archive bucket with comp metadata before deleting the working bucket.

## 8. Files of record

- `STRATEGY.md` — pre comp plan.
- `STRATEGY_UPDATE_MAGIC_FORMULA.md` — pivot when magic formula was discovered.
- `FINAL_SUBMISSION_STRATEGY.md` — original submission strategy doc.
- `docs/final_submission_strategy.md` — detailed pick rationale.
- `SESSION_RESUME.md` — running session log through 2026-04-13.
- `experiments.db` — SQLite of all 51 trained models with CV, params, and S3 paths.
- `predictions/` — OOF and test predictions for every model in the pool.
- `submissions/` — every CSV submitted to Kaggle, named by version.

## 9. Reproducibility

To rerun the v15 final pick from scratch:

1. Pull this repo. Set `KAGGLE_API_TOKEN` and AWS credentials in `.env`.
2. Train the 41 base models. Each has its own `scripts/train_*.py`. Costs about $200 in SageMaker.
3. Sync OOF and test predictions to `s3://<your-bucket>/ensemble-data/predictions/`.
4. Edit `scripts/train_ensemble.py` to list the 41 model names.
5. Launch the ensemble job (see `SESSION_RESUME.md` "How to run ensemble on SageMaker"). Choose `c5.9xlarge`, expect ~45 min.
6. Pull the LGB stacker output, run `scripts/log_bias_threshold_tune.py` on its OOF, apply to test predictions, write CSV.
7. Submit via `kaggle competitions submit`.

End to end clean rerun is under 12 hours of training time and roughly $250 in compute.
