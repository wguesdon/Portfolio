# Final Submission Selection Strategy

Competition allows 2 final submissions. Both are scored on the private leaderboard. The best of the two determines final ranking.

## Public/Private LB split: 20/80

The public LB is only **20%** of the test set. The private LB (80%) determines the final ranking. This has important implications:

- Public LB score has high variance. A delta of 0.00002 between two submissions is essentially noise at the 20% sample size.
- CV (5-fold on 630K rows) is a much more reliable estimator of private LB performance than public LB.
- A submission with a slightly lower public LB but tighter CV-LB gap is likely safer than one with a higher public LB that overfits to CV or public quirks.

**Strategy implications:**
1. **Prefer stackers over greedy.** Stackers are regularized by their own internal CV training, so their reported CV is conservative and closer to generalization performance. Greedy forward selection tends to overfit to specific 5-fold CV quirks, as shown by the consistent negative gap pattern (high CV, lower LB).
2. **Trust CV when LB differences are tiny.** If two submissions are within 0.0005 on public LB, go with the one whose method historically has smaller CV-LB gaps.
3. **Don't chase public LB.** A public LB improvement of ~0.0002 means almost nothing on private. What matters is which method produced it.
4. **Both final picks should be stackers.** Method diversity (stacker + greedy) sounds appealing but greedy has consistently underperformed on LB despite higher CV. Better to hedge via two different stackers (different model pool, different stacker type) than to pair a reliable method with an overfit one.

## Current picks (as of 2026-04-12)

### Pick 1: v12 LGB stacker 32m (LB 0.98074, CV 0.98051, gap +0.00023)

File: `sub_v12_lgb_stack_log_bias.csv`

Best public LB score achieved. 32 models: 30 original + lgb_ote_shallow + lgb_ote_deep. Standard method: LGB stacker with log-bias threshold tuning. Positive gap (LB > CV) means public test was slightly favorable, but CV is reliable.

### Pick 2: v11 LGB stacker 30m (LB 0.98072, CV 0.98061, gap -0.00011)

File: `sub_v11_lgb_stack_30m.csv`

2nd best public LB, and the highest CV among all top-LB stackers. 30 models (no cuML, no shallow/deep LGB). v11 strictly dominates v13 as a Pick 2: v13 has LB 0.98042 and CV 0.98058, so v11 is better on both dimensions.

**Why these two:** They hedge via model pool diversity at the top of the leaderboard.
- v12 = 32 models (30 + lgb_ote_shallow + lgb_ote_deep), LB 0.98074 / CV 0.98051.
- v11 = 30 models (original pool), LB 0.98072 / CV 0.98061.
- Both are LGB stackers with log-bias threshold tuning. Same method, slightly different pools.
- This pair is safer than pairing with v13: v11 beats v13 on both CV and LB, so there's no reason to prefer v13.

**Why not v13 (41 models with 7 cuML + 2 LGB):** Adding the 7 cuML diversity models (rf, svm, gnb, knn15, knn5_lite, lr_l2, lr_l1) raised CV marginally (0.98058 vs 0.98051) but dropped public LB by 0.00032 (0.98042 vs 0.98074). The cuML models are weak individually (0.72-0.96) and the stacker may have overweighted them slightly in CV but they don't generalize. v11 and v12 share the same stronger core and their LB consistency is reassuring.

---

## Previous picks (superseded, kept for history)

### OLD Pick 1: v11 LGB stacker 30m (LB 0.98072, CV 0.98061, gap -0.00011)

File: `sub_v11_lgb_stack_30m.csv`

Best public LB score. Uses all 30 models including 6 OTE variants plus 3 diversity
models (ExtraTrees, Logistic Regression, KNN). The stacker learns per-sample weights
across the full pool. The negative gap (LB > CV) is unusual and may reflect a favorable
public test split. This is the aggressive pick that goes for the win.

### Pick 2: v9 LGB stacker 27m (LB 0.98042, CV 0.98045, gap 0.00003)

File: `sub_v9_lgb_stack_27m.csv`

Insurance pick. Near-zero gap (0.00003) makes this the most trustworthy submission.
Uses 27 models without the 3 weak diversity models. If the weak models (et_ote at
CV 0.962, lr_ote at 0.937, knn_ote at 0.776) add noise rather than signal on
private test, v9 will outperform v11.

## Why these two

- Both are LGB stackers with log-bias threshold tuning, the method that has consistently
  produced the best LB scores throughout this competition.
- v11 maximizes public LB score. v9 maximizes reliability (tightest gap).
- They differ in model pool (30 vs 27). The 3 extra models are weak but algorithmically
  diverse (bagging, linear, instance-based). If the stacker correctly learns to downweight
  them, v11 wins. If they introduce noise, v9 wins.
- Greedy methods were considered for Pick 2 but rejected. Greedy has consistently
  underperformed stackers on LB despite higher CV, suggesting greedy CV is overly
  optimistic. Sacrificing 0.00019 LB (v9 vs v6 greedy) for a method-diversity hedge
  is not worth it given the pattern.

## Full submission history

| Submission | Method | Models | CV | LB | Gap |
|-----------|--------|--------|-----|-----|-----|
| **v11 LGB stacker** | **stacker+log_bias** | **30** | **0.98061** | **0.98072** | **-0.00011** |
| **v9 LGB stacker** | **stacker+log_bias** | **27** | **0.98045** | **0.98042** | **0.00003** |
| v10 LGB stacker | stacker+log_bias | 29 | 0.98046 | 0.98036 | 0.00010 |
| v6 greedy 3m | greedy+diffevol | 3 | 0.98067 | 0.98023 | 0.00044 |
| blend v9+v6 | CSV blend | 2 | n/a | 0.98023 | n/a |
| v7 LGB stacker | stacker+log_bias | 24 | 0.98054 | 0.98008 | 0.00046 |
| v6 LGB stacker | stacker+log_bias | 22 | 0.98038 | 0.97995 | 0.00043 |
| v7 greedy 7m | greedy+diffevol | 7 | 0.98084 | 0.97988 | 0.00096 |
| v10 greedy 9m | greedy+diffevol | 9 | 0.98086 | 0.97979 | 0.00107 |
| v8 greedy 8m | greedy+diffevol | 8 | 0.98081 | 0.97973 | 0.00108 |
| v4d LGB stacker | stacker+log_bias | 21 | 0.98026 | 0.97972 | 0.00054 |
| v5 LGB stacker | stacker+log_bias | 22 | 0.98029 | 0.97962 | 0.00067 |

## Key patterns

- LGB stackers consistently have tighter gaps than greedy selections.
- More models help the stacker: v11 30m (0.98072) > v9 27m (0.98042) > v7 24m (0.98008).
- More models hurt greedy: v6 3m (0.98023) > v7 7m (0.97988) > v10 9m (0.97979).
- The OTE feature pipeline was the single biggest improvement driver.
- Adding weak but algorithmically diverse models (ExtraTrees, LR, KNN) helped the stacker
  despite their low individual CV scores. The stacker learns to use them selectively.

## Action required

Select v11 and v9 as the two final submissions on the Kaggle competition page before April 30, 2026.
