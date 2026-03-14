#!/usr/bin/env python
"""
XGBoost trainer for obesity risk prediction.

Best params from Kaggle competition notebook:
  CV Accuracy ~0.908 (10-fold), ensemble contribution weight: 1
"""
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from category_encoders import MEstimateEncoder
from xgboost import XGBClassifier

from base_trainer import BaseTrainer

CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]


class XGBoostTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "xgb"

    def get_default_params(self) -> dict:
        """Best params from competition notebook (Optuna-tuned)."""
        return {
            "grow_policy": "depthwise",
            "n_estimators": 982,
            "learning_rate": 0.050053726931263504,
            "gamma": 0.5354391952653927,
            "subsample": 0.7060590452456204,
            "colsample_bytree": 0.37939433412123275,
            "max_depth": 23,
            "min_child_weight": 21,
            "reg_lambda": 9.150224029846654e-08,
            "reg_alpha": 5.671063656994295e-08,
            "booster": "gbtree",
            "objective": "multi:softprob",
            "device": "cuda",  # XGBoost 2.x: device=cuda replaces tree_method=gpu_hist
            "verbosity": 0,
            "seed": self.random_state,
            "num_class": 7,
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.8),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "booster": "gbtree",
            "objective": "multi:softprob",
            "device": "cuda",
            "verbosity": 0,
            "seed": self.random_state,
            "num_class": 7,
        }

    def create_pipeline(self, params: dict) -> Pipeline:
        return Pipeline([
            ("encoder", MEstimateEncoder(cols=CATEGORICAL_COLS)),
            ("model", XGBClassifier(**params)),
        ])


if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.run()
