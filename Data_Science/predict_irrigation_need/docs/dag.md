# PS6E4 Model and Ensembling DAG

Mermaid version. Renders natively on GitHub, GitLab, Notion, and Obsidian. For Kaggle, render the `dag.dot` Graphviz file to PNG and upload as an image.

```mermaid
flowchart TD
    Data["Raw Data<br/>Train 630K rows, Test 270K rows<br/>20 features, 3 classes"]

    Data --> FE1["OTE Pipeline<br/>4x shuffle concat<br/>Leave one out target encoding<br/>Digit extraction"]
    Data --> FE2["v3 Pipeline<br/>Magic ratios + domain features<br/>Standard target encoding<br/>2 way interactions"]

    FE1 --> OTE_TOP["Top OTE Boosters<br/>LGB OTE 0.97942<br/>XGB OTE 0.97938<br/>XGB OTE shallow GPU 0.97927<br/>CAT OTE 0.97919<br/>XGB OTE magic 0.97910<br/>XGB OTE seeds 43, 44<br/>LGB OTE shallow / deep"]

    FE1 --> OTE_DIV["OTE Diversity Models<br/>ExtraTrees 0.96156<br/>cuML SVM RBF 0.96154<br/>cuML RF 0.96005<br/>LR ElasticNet 0.95568<br/>LR L1 0.95554<br/>LR L2 0.95522<br/>cuML GNB 0.90860<br/>KNN k=15 0.71937"]

    FE2 --> V3_GBDT["v3 Gradient Boosters<br/>7 CatBoost variants<br/>4 LightGBM variants<br/>6 XGBoost variants<br/>CV range 0.97131 to 0.97834"]

    FE2 --> V3_NN["v3 Neural Networks<br/>RealMLP mahog 0.97802<br/>RealMLP v3fix 0.97108<br/>TabM v3fix 0.97053"]

    OTE_TOP --> STACK["LightGBM Stacker<br/>41 base model OOF probabilities<br/>5 fold StratifiedKFold seed 42<br/>CV 0.98045"]
    OTE_DIV --> STACK
    V3_GBDT --> STACK
    V3_NN --> STACK

    STACK --> LOGBIAS["Log Bias Correction<br/>Per class additive shift on log proba<br/>Tuned on OOF for balanced accuracy"]

    LOGBIAS --> SUB["Final Submission v15<br/>Public LB 0.98081<br/>Private LB 0.98082<br/>12 of 457"]

    classDef data fill:#e8e8e8,stroke:#333,stroke-width:2px;
    classDef fe fill:#cfe2ff,stroke:#0d6efd,stroke-width:2px;
    classDef strong fill:#fff3cd,stroke:#664d03,stroke-width:2px;
    classDef diversity fill:#d1e7dd,stroke:#0a3622,stroke-width:2px;
    classDef stack fill:#ffe5d0,stroke:#c2410c,stroke-width:3px;
    classDef final fill:#90ee90,stroke:#0a3622,stroke-width:3px;

    class Data data;
    class FE1,FE2 fe;
    class OTE_TOP,V3_GBDT strong;
    class OTE_DIV,V3_NN diversity;
    class STACK,LOGBIAS stack;
    class SUB final;
```

## Models Excluded from the Stacker

These were trained but did not enter the final ensemble.

| Model | CV | Reason excluded |
|-------|-----|-----------------|
| GraphSAGE GNN | 0.97073 | hurt LB when added |
| RealMLP on OTE features | 0.96509 | fillna(0) breaks MLP on NaN heavy OTE |
| XGB Residual | 0.96633 | formula correction has low ceiling |
| cat_utaazu (public OOF) | 0.97872 | baked in bias from non matching folds |
| ravi_baseline (public) | 0.97850 | KFold not StratifiedKFold |
| AutoGluon | 0.96430 | no custom feature engineering |
| TabPFN | n/a | hit Kaggle 12h GPU limit on fold 1 of 5 |
| Pseudo labeling LGB | 0.95390 | lost 4x OTE shuffle aug, 99.9 percent label agreement adds no signal |
