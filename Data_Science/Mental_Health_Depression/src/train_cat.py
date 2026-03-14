#!/usr/bin/env python
"""
CatBoost trainer for mental health depression prediction (PS4E11).

Binary classification: Depression (0/1).
GPU instance: ml.g5.xlarge (task_type=GPU)
CatBoost handles Profession, Degree, City natively — no encoding needed.
"""
import numpy as np
import optuna
from catboost import CatBoostClassifier

from base_trainer import BaseTrainer

# CatBoost handles these natively (no encoding required)
CATEGORICAL_COLS = ["Profession", "Degree", "City"]


class CatBoostTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "cat"

    def get_default_params(self) -> dict:
        """Reasonable defaults for binary depression classification with GPU."""
        return {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 0.5,
            "random_seed": self.random_state,
            "verbose": False,
            "task_type": "GPU",
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": self.random_state,
            "verbose": False,
            "task_type": "GPU",
            "loss_function": "Logloss",
            "eval_metric": "Accuracy",
        }

    def create_pipeline(self, params: dict):
        model = CatBoostClassifier(**params, cat_features=CATEGORICAL_COLS)

        class _CatPipeline:
            """Thin wrapper so CatBoost works with the base_trainer pipeline interface."""
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def predict(self, X):
                return self.model.predict(X).flatten()

            def predict_proba(self, X):
                return self.model.predict_proba(X)

        return _CatPipeline(model)


if __name__ == "__main__":
    trainer = CatBoostTrainer()
    trainer.run()
