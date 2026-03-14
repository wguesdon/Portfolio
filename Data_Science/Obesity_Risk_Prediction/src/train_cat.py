#!/usr/bin/env python
"""
CatBoost trainer for obesity risk prediction.

Best params from Kaggle competition notebook:
  CV Accuracy ~0.910 (10-fold), ensemble contribution weight: 1
"""
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from base_trainer import BaseTrainer

# CatBoost handles these natively — no encoding needed
CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]


class CatBoostTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "cat"

    def get_default_params(self) -> dict:
        """Best params from competition notebook (Optuna-tuned)."""
        return {
            "iterations": 1000,
            "learning_rate": 0.13762007048684638,
            "depth": 5,
            "l2_leaf_reg": 5.285199432056192,
            "bagging_temperature": 0.6029582154263095,
            "random_seed": self.random_state,
            "verbose": False,
            "task_type": "GPU",
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": self.random_state,
            "verbose": False,
            "task_type": "GPU",
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
        }

    def create_pipeline(self, params: dict) -> Pipeline:
        # CatBoost accepts cat_features index or column names.
        # Since pipeline transforms X into a numpy array, we pass indices.
        # Workaround: use a single-step pipeline so X stays as DataFrame.
        model = CatBoostClassifier(**params, cat_features=CATEGORICAL_COLS)

        class _CatPipeline:
            """Thin wrapper to expose fit/predict/predict_proba on a DataFrame."""
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
