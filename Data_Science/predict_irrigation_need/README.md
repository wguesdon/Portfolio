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
├── main.py
├── pyproject.toml
├── STRATEGY.md
├── STRATEGY_UPDATE_MAGIC_FORMULA.md
├── FINAL_SUBMISSION_STRATEGY.md
├── docs/         # workflow notes, DAG, SageMaker and Kaggle kernel guides
├── notebooks/    # showcase notebooks, 22 to 41 model ensembles, GNN training
├── scripts/      # EDA, feature engineering, per model trainers, ensemble launchers
└── write_up/     # public and private solution write ups
```

Heavyweight artefacts (pickled model weights, CatBoost info, the experiments database, raw data, submissions, predictions, AWS infrastructure) live in the source workspace and are excluded from this portfolio copy.

## Tooling

- Python 3.11 managed with `uv`
- XGBoost, LightGBM, CatBoost, scikit-learn, cuML, TabPFN, RealMLP
- AWS SageMaker training jobs for the GPU base models
- 5 fold stratified cross validation throughout
