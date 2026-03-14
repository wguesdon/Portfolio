# Data Science & Bioinformatics Portfolio

A collection of Data Science, and Bioinformatics projects.

## Contact

William Guesdon |
[Personal website](https://willguesdon.com/) | [LinkedIn](https://www.linkedin.com/in/william-guesdon/) | [Kaggle](https://www.kaggle.com/wguesdon) | <a href="mailto:wguesdon@gmail.com">Email</a>

---

## Bioinformatics

### [nf-core RNA-seq Pipeline](Bioinformatics/nfcore_rnaseq_airway)

**End-to-end differential expression** — RNA-seq analysis using nf-core/rnaseq and nf-core/differentialabundance on AWS Batch Spot instances. Identifies ~4,000 differentially expressed genes in airway smooth muscle cells treated with dexamethasone, with pathway enrichment analysis.

### [BCR Repertoire Analysis](Bioinformatics/BCR_Repertoire)

**COVID-19 immune repertoire** — B-cell receptor repertoire analysis using nf-core/airrflow on amplicon-based IGH sequencing data from COVID-19 patients. Processes 6 samples through pRESTO, IgBLAST, and Immcantation to identify convergent antibody sequences across disease severity groups.

### [Microbial Strain Analysis](Bioinformatics/Microbial_Strain_Analysis)

**Interactive dashboard** — EDA and Streamlit dashboard for a bacterial strain database linking taxonomy, growth parameters, and media composition. Automated profiling with Sweetviz and interactive Plotly visualizations.

### [Tidyplots for Bioinformatics](Bioinformatics/tidyplots)

**Publication-ready figures** — Tutorial demonstrating tidyplots on the canonical airway RNA-seq dataset. Produces volcano plots, MA plots, and heatmaps in a containerized R environment with DESeq2 analysis.

### [ggpubr Publication Figures](Bioinformatics/ggpubr_figures)

**Journal-quality R visualizations** — Practical guide to publication-ready figures using ggpubr and ggprism with Nature Publishing Group colour palettes and statistical annotations.

### [Matplotlib Publication Figures](Bioinformatics/matplotlib_figures)

**Journal-quality Python visualizations** — Tutorial on creating publication-ready figures with matplotlib, seaborn, and SciencePlots using NPG styling and significance brackets via statannotations.

### [R Basics Tutorial](Bioinformatics/R_basics_tutorial)

**Beginner-friendly** — Quarto tutorial introducing R fundamentals for bioinformatics: variables, data types, and statistical analysis with the Tidyverse.

---
## Data Science

### [Obesity Risk Prediction](Data_Science/Obesity_Risk_Prediction)

**Top 1% on Kaggle** (30/3,587) — Multi-class classification predicting obesity risk from lifestyle and anthropometric features. Ensemble of LightGBM, XGBoost, and CatBoost trained on AWS SageMaker, achieving ~91.3% accuracy across 7 obesity categories.

### [CIBMTR HCT Survival Prediction](Data_Science/CIBMTR_HCT)

**Fairness-aware survival modelling** — Predicts outcomes for ~28,800 allogeneic hematopoietic cell transplant patients. Gradient boosting ensemble (XGBoost, LightGBM, CatBoost) optimized for equitable performance across racial groups using stratified concordance index.

### [Mental Health Depression Prediction](Data_Science/Mental_Health_Depression)

**Kaggle Playground Series** — EDA and predictive modelling on mental health survey data covering sleep patterns, stress levels, and depression indicators. Includes AWS SageMaker training pipelines and interactive Quarto reports.

### [VinBigData Chest X-ray Abnormalities Detection](https://github.com/66Days-group-learners/VinBigData_Chest_X-ray_Abnormalities_Detection)

**Top 7% on Kaggle** — Team project to detect and localize abnormalities on chest X-rays. Managed the GPU-powered virtual machine used to train and improve the model.

### [NYC Airbnb Price Prediction](Data_Science/NYC_Airbnb)

**R² = 0.622 with LightGBM** — Exploratory analysis and price prediction for ~49,000 NYC Airbnb listings. Location-based features are the strongest predictors, outperforming baseline approaches without manual price-range splitting.

### [Stroke Risk Factors](Data_Science/Stroke_Risk_Factors)

**Four complementary analytical approaches** — Identifies stroke risk factors using supervised learning, unsupervised clustering, causal inference, and ensemble ranking on 43,400 healthcare observations. Handles class imbalance via cost-sensitive learning.

### [Ph.D. Stipends Analysis](Data_Science/PhD_Stipends)

**Stipend trends over time** — Data cleaning and analysis of Ph.D. student stipends by university and department. Key findings: higher US stipends, no increase with seniority, significant drop around the 2008 crisis, and disparities between departments.

### [Bioinformatics Job Market Analysis](Data_Science/Bioinformatics_Job_Market_Analysis)

**1,677 job postings analysed** — Explores bioinformatics job market trends from December 2024 postings across the USA, UK, and Ireland. Identifies in-demand skills (clinical expertise, R, Python), salary distributions (median $75.5k), geographic hotspots, and skill co-occurrence patterns.

### [SARS-CoV-2 Vaccine Coverage](Data_Science/Analyse_SARS_CoV_2_vaccines_coverage)

**Global vaccine comparison** — Compares vaccine coverage between countries and explores the correlation between coverage and vaccine hesitancy.

### [Python Basics Tutorial](Data_Science/Python_basics_tutorial)

**Beginner-friendly** — Quarto tutorial introducing Python fundamentals for data science: variables, data types, statistical analysis with Pandas, Matplotlib, and Scikit-learn.


