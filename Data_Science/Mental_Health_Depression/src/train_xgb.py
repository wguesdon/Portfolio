#!/usr/bin/env python
"""
XGBoost trainer for mental health depression prediction (PS4E11).

Binary classification: Depression (0/1).
GPU instance: ml.g4dn.xlarge (device=cuda)
"""
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from category_encoders import MEstimateEncoder
from xgboost import XGBClassifier

from base_trainer import BaseTrainer

# MEstimateEncoder handles high-cardinality categoricals (City, Profession, Degree)
CATEGORICAL_COLS = ["Profession", "Degree", "City"]


class XGBoostTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "xgb"

    def get_default_params(self) -> dict:
        """Reasonable defaults for binary depression classification with GPU."""
        return {
            "grow_policy": "depthwise",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "gamma": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_depth": 6,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "reg_alpha": 0.1,
            "booster": "gbtree",
            "objective": "binary:logistic",
            "device": "cuda",  # XGBoost 2.x: device=cuda (no tree_method=gpu_hist)
            "verbosity": 0,
            "seed": self.random_state,
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "booster": "gbtree",
            "objective": "binary:logistic",
            "device": "cuda",
            "verbosity": 0,
            "seed": self.random_state,
        }

    def create_pipeline(self, params: dict) -> Pipeline:
        return Pipeline([
            ("encoder", MEstimateEncoder(cols=CATEGORICAL_COLS)),
            ("model", XGBClassifier(**params)),
        ])


if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.run()
