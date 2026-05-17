# Next-Competition Compute Strategy

Lessons learned from PS6E4 on how to structure compute for future Kaggle competitions, ranked by cost-effectiveness. Target: competitive ensemble while spending as close to $0 as possible.

## Resource tiers

| Tier | Cost | Compute | Max runtime | Best for |
|------|------|---------|-------------|----------|
| **Kaggle CPU notebook** | Free, unlimited | ~4 CPU / 30 GB RAM | 12h per run | LGB/XGB/CAT hist, LR, stacker jobs |
| **Kaggle GPU notebook** | Free, 30h/week T4 | T4 16 GB + 4 CPU | 12h per run | NN models, TabPFN, cuML |
| **Inspiron (home CPU)** | Free, unlimited | 12 CPU / 62 GB RAM | no limit | Overnight long jobs, OTE 4x augmentation |
| **Inspiron GPU** | n/a | - | - | No GPU on this box |
| **vast.ai / RunPod GPU** | $0.20-0.70/hr | T4 / 3090 / 4090 | as long as you pay | Fast GPU bursts when Kaggle quota exhausted |
| **SageMaker** | $0.50-1.50/hr | 36 CPU c5.9xl, T4 g4dn.xl | 12h per job | Only when parallelism or deadline matters |

## Core insight: tree models on CPU are free

**XGBoost `tree_method='hist'`** and **LightGBM** give identical results on CPU and GPU. GPU is just faster. For a Kaggle competition, the quality is unchanged and you save all the money.

**CatBoost CPU vs GPU differ slightly** in tree construction, but both are strong. Pick one and stay consistent.

**Verified on PS6E4:** xgb_ote_shallow with same params
- Inspiron CPU: fold 1 = 0.97824
- Kaggle GPU: fold 1 = 0.97831
- Gap: 0.00007 (well within CV noise)

## Recommended workflow

### Phase 1: Baseline (Day 1)

Push 3 Kaggle CPU notebooks in parallel:
- `lgb_baseline` — LightGBM with baseline FE and hyperparams
- `xgb_baseline` — XGBoost hist
- `cat_baseline` — CatBoost CPU

Cost: $0. Runtime: 2-4h each (within 12h limit). Can run 4+ kernels simultaneously.

### Phase 2: Feature engineering + OTE variants (Days 2-3)

For strong FE candidates: OTE, magic features, digit extraction.
- Run each as a separate Kaggle CPU notebook if it fits in 12h
- If 4x OTE augmentation exceeds 12h: Inspiron overnight
- Collect OOF + test predictions as kernel outputs

### Phase 3: NN and TabPFN (Days 2-3, parallel to phase 2)

- TabPFN on Kaggle GPU (~30 min per fold for tabular)
- RealMLP / TabM / FT-Transformer on Kaggle GPU (~2-4h per run)
- Stay within 30h/week GPU budget

### Phase 4: Ensemble (Days 4-6)

Run ensemble stacker jobs as:
- Kaggle CPU notebook (if model pool is small, 2-4h)
- Inspiron overnight (if pool is large or you want multiple variants)

The ensemble itself rarely needs 36-core SageMaker. 12 cores is plenty.

### Phase 5: Final crunch (Last 24h)

**Only here** is SageMaker worth considering:
- You need to launch 8+ training jobs in parallel and can't wait 12h per Kaggle queue
- You need a urgent same-day turnaround on a 4-hour CPU job that would take 12h on Inspiron
- You're testing a risky new approach and want to fail fast

Budget: $20-50 for the final sprint is reasonable; beyond that you're buying diminishing returns.

## What this would have cost on PS6E4

Actual spend: ~$320 ($241 SageMaker + tax, S3, EC2, ECR).

Projected spend on free tiers:
| Model type | Actual | Free-tier alternative |
|-----------|--------|----------------------|
| LGB/XGB/CAT (30 variants) | ~$60 | $0 — Kaggle CPU |
| cuML wave 1 + 2 (7 models) | ~$20 | $0 — Kaggle GPU quota |
| RealMLP, TabM, GNN | ~$30 | $0 — Kaggle GPU |
| LGB pseudo-labeling | $3 | $0 — Inspiron overnight |
| XGB OTE magic (2h 50m on c5.9xl) | $8 | $0 — Inspiron overnight (~10h) |
| 7 ensemble stackers | ~$20 | $0 — Kaggle CPU or Inspiron |
| cuML Docker image build | $3 ECR + $10 EC2 | $0 — Kaggle GPU has RAPIDS preinstalled |
| Failed cuML retries | ~$15 | $0 — the image bug would have failed locally too, saving the retries |
| Tax | $53 | $0 |
| **Estimated realistic saving** | | **~$220** |

The remaining $100 would still have been SageMaker if we wanted parallel batch training, but we could probably have done the whole competition for under $50.

## Practical setup

### Kaggle CLI

Already configured in this repo. Use `uv run kaggle` from the project root with the `.env` loaded.

```bash
# Push a new notebook version
uv run kaggle kernels push -p Playground_Series/<comp>/notebooks/<kernel>/

# Check status
uv run kaggle kernels status wguesdon/<kernel-slug>

# Fetch outputs after complete
uv run kaggle kernels output wguesdon/<kernel-slug> -p <local-dir>/
```

See `docs/kaggle_notebook_workflow.md` for full guide.

### Inspiron

SSH + tmux + uv venv. Queue jobs sequentially using `while tmux has-session -t <prev>; do sleep 60; done` pattern.

See `docs/inspiron_workflow.md` for full guide.

### vast.ai / RunPod fallback

For when Kaggle GPU quota is exhausted mid-week. `docs/vastai_runpod_workflow.md` has the recipe. Budget $5-20 for a few hours of RTX 3090.

## Checklist for day 1 of next competition

- [ ] Create competition folder and `data/raw/` with train/test CSVs
- [ ] Push 3 baseline notebooks to Kaggle CPU (LGB, XGB, CAT)
- [ ] Fetch individual CV scores after 12h
- [ ] Register in `experiments.db` via `update_model_db.py` pattern
- [ ] Plan ensemble sweep only after baselines land
- [ ] Reserve SageMaker budget for final 24h, not day 1

## Checklist for reducing SageMaker dependency

- [ ] Is there a Kaggle CPU notebook alternative? → use it
- [ ] Is this an overnight job? → Inspiron instead
- [ ] Is this a GPU job? → Kaggle GPU (T4) or vast.ai spot ($0.30/hr)
- [ ] Am I under deadline pressure? → SageMaker only if hours-not-days matter
- [ ] Is this a risky experiment? → Kaggle CPU with 12h cap, fail fast for free
