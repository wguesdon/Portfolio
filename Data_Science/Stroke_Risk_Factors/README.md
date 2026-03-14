# Stroke Risk Factor Identification

## Introduction

This project identifies risk factors associated with stroke using four complementary
analytical approaches on the [Kaggle Healthcare Stroke Dataset](https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data)
(43,400 observations).

## Methods

| Option | Approach | Key Technique |
|--------|----------|---------------|
| 1 | Supervised learning | Logistic Regression, Random Forest, Gradient Boosting with SHAP |
| 2 | Unsupervised learning | PCA + K-Means clustering with stroke susceptibility profiling |
| 3 | Causal inference | Inverse Probability Weighting (IPW) with bootstrap CIs |
| 4 | Ensemble ranking | Unified feature importance combining all methods |

**Class imbalance** (1.8% stroke) is handled via cost-sensitive learning
(`class_weight="balanced"`, `scale_pos_weight`) and threshold tuning on the
precision-recall curve — deliberately avoiding SMOTE (see `SMOTE_LIMITATIONS.md`).

## Results

All four approaches converge on the same top risk factors:

1. Age
2. Average Glucose Level
3. Hypertension
4. Heart Disease
5. Smoking History (former/current)

## Reproduction

```bash
# Local
uv sync
uv run quarto render report.qmd --output-dir output

# Container
podman build -t stroke-risk .
podman run --rm -v ./:/project:Z stroke-risk
```

## Project Structure

```
├── ASSIGNMENT.md              # Original assignment brief
├── SMOTE_LIMITATIONS.md       # Why we avoid SMOTE + modern alternatives
├── Data/
│   ├── train_2v.csv           # Training dataset (43,400 rows)
│   ├── test_2v.csv            # Test dataset (no labels)
│   └── data_dictionary.csv    # Variable definitions
├── report.qmd                 # Full analysis (Quarto notebook)
├── output/                    # Generated figures (15 PNGs) + report.html
├── pyproject.toml             # Python dependencies (uv)
├── Containerfile              # Podman container for reproducibility
└── README.md
```
