# Final Submission Strategy

Kaggle allows 2 selected submissions for the private leaderboard. The private LB determines the final ranking.

## Selection Strategy

Pick one submission that maximizes public LB (trust the test set signal) and one that maximizes CV (trust the training signal). If they disagree on the private LB, we hedge our bets.

### Selection 1: Best Public LB

**Notebook v8: 14-model GBDT ensemble with hardcoded thresholds**
- Public LB: **0.97884**
- CV: 0.97933
- 14 GBDT models (XGB + LGB + CatBoost, seeds 42/43/44/45, feature strategies)
- Equal weights, differential evolution thresholds [2.605, 0.508, 0.656]
- Source: Kaggle notebook v8

Why: This is our highest public LB score. The thresholds were pre-computed locally and reproduce exactly. No stochastic optimization at submission time means no variance. Pure GBDT models with no NN component performed best on the public test set.

### Selection 2: Best CV

**15-model blend (14 GBDT + RealMLP) with differential evolution**
- Public LB: 0.97848
- CV: **0.97943**
- 15 models: 14 GBDT + Mahog's RealMLP (CV 0.97802)
- Equal weights, differential evolution thresholds [3.428, 0.705, 0.914]
- Source: sub_15model_realmlp_diffevol.csv

Why: Highest CV score. The NN model adds architectural diversity that may generalize better to the private test set even though it scored lower on the public portion. If the private test has a different distribution from the public test, diversity wins.

## All Submissions Ranked

| Rank | Submission | CV | Public LB | Selected |
|------|-----------|------|-----------|----------|
| 1 | Notebook v8 (14 GBDT, hardcoded thresholds) | 0.97933 | **0.97884** | Yes (best LB) |
| 2 | 14-model blend + diffevol | 0.97930 | 0.97876 | No |
| 3 | 15-model + RealMLP + diffevol | **0.97943** | 0.97848 | Yes (best CV) |
| 4 | LR stacker L3 + diffevol | 0.97911 | 0.97828 | No |
| 5 | CatBoost blend + diffevol | 0.97906 | 0.97823 | No |
| 6 | Notebook v9 (longer search) | higher | 0.97807 | No |
| 7 | Notebook v7 (live search) | - | 0.97793 | No |

## Observations

- Higher CV does not always mean higher LB. The v9 notebook (longest optimization) had the highest CV but scored 0.97807 on LB (overfit to OOF).
- The gap between best CV (0.97943) and best LB (0.97884) is 0.00059. This suggests mild overfitting in the threshold optimization step.
- GBDT-only ensembles outperform GBDT+NN on public LB so far. This may reverse on private LB if the distribution differs.
- Equal weights outperform learned weights. Hill climbing gives CatBoost 100% weight (models too correlated). Equal weights preserve diversity.

## Pending: Final Ensemble After Training Completes

When all v4, v5, and NN training jobs complete, we will build one more ensemble. If it scores higher than either selection above, we replace the weaker pick. Plan:

1. Download all new OOF/test predictions
2. Build ensemble with 20+ models (v3 + v4 + v5 GBDT + 4 NN + RealMLP Mahog)
3. Run ensemble_v3.py (log-space bias + differential evolution)
4. Submit and compare
5. Update selections if improved

## Risk Assessment

The safest pick is the 14-model GBDT ensemble (Selection 1). It uses no NN models, no stochastic optimization at submission time, and simple equal weights. The thresholds are aggressive (High class multiplied by 2.6x) which could hurt if the private test has fewer ambiguous High samples.

Selection 2 hedges against this risk by including NN diversity and different thresholds.
