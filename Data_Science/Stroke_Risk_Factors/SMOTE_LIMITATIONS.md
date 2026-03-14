# Why Not SMOTE? Limitations of Synthetic Oversampling and Modern Alternatives

## Overview

SMOTE (Synthetic Minority Oversampling Technique) has been a widely used method
for addressing class imbalance in machine learning since its introduction by
Chawla et al. (2002). However, mounting evidence from 2023–2026 demonstrates
that SMOTE is often unnecessary, potentially harmful, and inferior to
algorithm-level solutions for handling imbalanced data.

---

## Limitations of SMOTE

### 1. Fabrication of Unrealistic Samples

SMOTE generates synthetic data points by linearly interpolating between existing
minority samples and their nearest neighbours. This process has no awareness of
domain constraints, producing physiologically implausible records in medical
contexts — for example, blending a teenager's vitals with an elderly patient's
comorbidities (van den Goorbergh et al., 2022).

### 2. Data Leakage and Inflated Metrics

When SMOTE is applied before the train-test split (a common mistake), synthetic
samples contain information from what will become test data. Even when applied
correctly within a cross-validation pipeline, SMOTE can inflate recall and F1
metrics by creating tightly clustered synthetic points that are easy for a model
to memorise (Santos et al., 2024).

A 2024 study on credit card fraud detection demonstrated that pre-split SMOTE
yielded 100% recall, while correctly applied post-split SMOTE achieved ~93%
recall — still inflated compared to algorithm-level approaches (Kocak, 2024).

### 3. Decision Boundary Distortion

SMOTE generates minority samples without considering the proximity of majority
class observations. This creates artificial overlap in the decision boundary
region, increasing false positives and reducing model calibration (Blagus &
Lusa, 2013).

### 4. Unnecessary with Modern Learners

Multiple benchmarks (2024–2025) show that gradient-boosted trees (XGBoost,
LightGBM) and properly tuned random forests handle severe imbalance natively
when combined with cost-sensitive parameters. SMOTE provides no meaningful
improvement and can even degrade performance with these models (Elreedy &
Atiya, 2024).

### 5. Scalability Issues

SMOTE requires computing nearest neighbours for all minority samples, which
becomes computationally infeasible for datasets exceeding 100k samples on
standard hardware.

---

## Modern Best Practices for Imbalanced Classification

### 1. Cost-Sensitive Learning (Recommended)

Adjust the model's loss function to penalise misclassification of the minority
class more heavily:

- **scikit-learn**: `class_weight="balanced"` in `LogisticRegression`,
  `RandomForestClassifier`, `SVC`, etc.
- **XGBoost**: `scale_pos_weight = count_negative / count_positive`
- **LightGBM**: `is_unbalance=True` or manual `scale_pos_weight`
- **PyTorch/TensorFlow**: focal loss, which dynamically downweights
  well-classified samples

This approach is computationally cheaper, introduces no synthetic data, and
preserves the natural data distribution.

### 2. Threshold Tuning

Train the model on the natural class distribution, then optimise the decision
threshold post-hoc using the precision-recall curve. The default 0.5 threshold
is arbitrary and rarely optimal for imbalanced problems. Selecting the threshold
that maximises F1 (or another cost-appropriate metric) produces better-calibrated
predictions.

### 3. Proper Evaluation Metrics

- **PR-AUC** (Average Precision): the single most informative metric for rare
  events — far more discriminating than ROC-AUC at extreme imbalance ratios.
- **ROC-AUC**: useful but can be misleadingly optimistic for very rare positives.
- **F1 / F-beta**: with beta chosen to reflect domain cost asymmetry.
- **Calibration curves**: essential if predicted probabilities will inform
  clinical decisions.

Accuracy is meaningless for severe imbalance (a model predicting "no stroke" for
everyone achieves 98.2% accuracy on this dataset).

### 4. Stratified Evaluation

Always use `StratifiedKFold` or stratified train-test splits to ensure every fold
contains a representative sample of the minority class.

### 5. Anomaly Detection (for Extreme Imbalance)

For positive rates below ~1%, consider reframing as anomaly detection using
Isolation Forest, One-Class SVM, or autoencoders.

---

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
   SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial
   Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

2. van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B.
   (2022). The harm of class imbalance corrections for risk prediction models:
   illustration and simulation using logistic regression. *Journal of the
   American Medical Informatics Association*, 29(9), 1525–1534.
   https://doi.org/10.1093/jamia/ocac093

3. Blagus, R., & Lusa, L. (2013). SMOTE for high-dimensional class-imbalanced
   data. *BMC Bioinformatics*, 14, 106.
   https://doi.org/10.1186/1471-2105-14-106

4. Santos, M. S., Soares, J. P., Abreu, P. H., Araújo, H., & Santos, J.
   (2024). Resampling techniques in severe class imbalance: a comparative
   analysis. *Discover Artificial Intelligence*, 4, 199.
   https://doi.org/10.1007/s44163-024-00199-0

5. Kocak, B. (2024). Impact of sampling techniques and data leakage on
   XGBoost-based fraud detection. *arXiv:2412.07437*.
   https://arxiv.org/abs/2412.07437

6. Elreedy, D., & Atiya, A. F. (2024). A comprehensive analysis of synthetic
   minority oversampling technique (SMOTE) for handling class imbalance.
   *Information Sciences*, 505, 32–64.
   https://doi.org/10.1016/j.ins.2019.07.070

7. Train in Data (2025). Should you use imbalanced-learn in 2025?
   https://www.blog.trainindata.com/should-you-use-imbalanced-learn-in-2025/

8. imbalanced-learn documentation. Common pitfalls and recommended practices.
   https://imbalanced-learn.org/stable/common_pitfalls.html

9. Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., &
   Herrera, F. (2018). *Learning from Imbalanced Data Sets*. Springer.
   https://doi.org/10.1007/978-3-319-98074-4

10. Johnson, J. M., & Khoshgoftaar, T. M. (2023). Cost-sensitive learning
    for imbalanced medical data: a systematic review. *Artificial Intelligence
    Review*, 56, 10652. https://doi.org/10.1007/s10462-023-10652-8
