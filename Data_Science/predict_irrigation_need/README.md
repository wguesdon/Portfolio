# Predicting Irrigation Need (Kaggle Playground Series S6E4)

3 class classification of irrigation need (Low, Medium, High) from soil, weather, and field features. The metric was balanced accuracy on a heavily imbalanced target (Low 58.7%, Medium 38.0%, High 3.3%).

## Results

| Submission | CV | Public LB |
|---|---:|---:|
| `sub_v15_lgb_stack_log_bias.csv` (Pick 1) | 0.98045 | 0.98081 |
| `sub_v11_lgb_stack_30m.csv` (Pick 2) | 0.98061 | 0.98072 |

Both picks are LightGBM stackers with log bias threshold tuning on top of a 30 to 41 model base pool.

## Blog post

A full write up of the OOF ensembling method used in this solution is published here:

[OOF ensemble method for PS6E4](https://willguesdon.com/posts/ps6e4-oof-ensemble-method)

## How to reproduce

The canonical, runnable artefact in this repository is the showcase notebook at `notebooks/ps6e4_showcase_v3.ipynb`. It trains a 3 model GBDT ensemble (XGBoost, LightGBM, CatBoost) end to end and reproduces the core ideas of the final solution on a single machine. The full 30 to 41 model ensemble required AWS SageMaker and is not reproducible from this repository alone.

### 1. Clone and set up the environment

```bash
git clone https://github.com/wguesdon/Portfolio.git
cd Portfolio/Data_Science/predict_irrigation_need
uv sync
```

This pins Python 3.11 to 3.12 and installs XGBoost, LightGBM, CatBoost, feature-engine, scikit-learn, and the plotting stack.

### 2. Download the competition data

The notebook expects `train.csv`, `test.csv`, and the original `irrigation_prediction.csv` under `data/raw/` when run locally.

```bash
mkdir -p data/raw
uv run kaggle competitions download -c playground-series-s6e4 -p data/raw
unzip -o data/raw/playground-series-s6e4.zip -d data/raw
uv run kaggle datasets download -d wguesdon/ps6e4-irrigation-14-model-predictions -p data/raw
unzip -o data/raw/ps6e4-irrigation-14-model-predictions.zip -d data/raw
```

You will need Kaggle API credentials in `~/.kaggle/kaggle.json` (see [Kaggle API docs](https://www.kaggle.com/docs/api)) and you must have accepted the competition rules on the website first.

### 3. Run the showcase notebook

```bash
uv run jupyter lab notebooks/ps6e4_showcase_v3.ipynb
```

Or execute headless:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/ps6e4_showcase_v3.ipynb --output ps6e4_showcase_v3.executed.ipynb
```

The notebook also detects when it is running on Kaggle and adapts its input paths automatically.

## Approach summary

1. Recovered the deterministic generator rule (the "magic formula") for the original 10K dataset and used distance to its decision boundaries as the dominant feature.
2. Engineered roughly 260 features per model, organised in seven groups: magic score features, domain interactions, original dataset target encodings, 2 way interaction target encodings, digit extraction, frequency encodings, and threshold flags.
3. Diversified the base pool across algorithm family, seed, and feature subset (XGBoost, LightGBM, CatBoost, ExtraTrees, RealMLP, KNN, LR, SVM, TabPFN, GNN).
4. Stacked with a LightGBM meta learner over 5 fold out of fold predictions, with log bias threshold tuning for the rare High class.
5. Trusted 5 fold CV over a 20% public leaderboard. Greedy forward selection consistently overfit CV, so the final picks were stacker outputs rather than hand picked blends.

## Repository layout

```
predict_irrigation_need/
├── README.md
├── pyproject.toml
├── STRATEGY.md, STRATEGY_UPDATE_MAGIC_FORMULA.md, FINAL_SUBMISSION_STRATEGY.md
├── docs/                       # workflow notes, DAG, SageMaker and Kaggle kernel guides
├── notebooks/
│   ├── ps6e4_showcase_v3.ipynb # canonical end-to-end notebook (runnable)
│   └── ps6e4_showcase_v3.py    # paired jupytext source for easy diffing
├── scripts/
│   ├── 01_eda.py               # exploratory data analysis
│   ├── 02_feature_engineering.py
│   ├── 03a_train_xgb.py        # per-family trainers used by the full pipeline
│   ├── 03b_train_lgb.py
│   ├── 03c_train_cat.py
│   ├── 03_run_all.py
│   ├── 04_submit_and_track.py
│   ├── train_*_ote.py          # original-target-encoded base model trainers
│   ├── ensemble_v*.py          # blending and pruning utilities
│   └── aws_orchestration/      # SageMaker / ECR / EC2 launchers (reference only)
└── write_up/                   # public and private solution write ups
```

Heavyweight artefacts (pickled model weights, CatBoost info, the `experiments.db` SQLite store, raw data, submissions, predictions) are excluded from this portfolio copy. The scripts under `scripts/aws_orchestration/` reference AWS resources that are specific to my account and are kept as reference rather than to be run by readers.

## Tooling

- Python 3.11 to 3.12 managed with `uv`
- XGBoost, LightGBM, CatBoost, scikit-learn, feature-engine
- AWS SageMaker training jobs for the GPU base models in the full pipeline
- 5 fold stratified cross validation throughout
