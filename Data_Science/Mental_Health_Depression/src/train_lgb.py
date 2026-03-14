#!/usr/bin/env python
"""
LightGBM trainer for mental health depression prediction (PS4E11).

Binary classification: Depression (0/1).
CV Accuracy target: ~0.940+
"""
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder
from lightgbm import LGBMClassifier

from base_trainer import BaseTrainer

NUMERICAL_COLS = [
    "Age", "Sleep_Hours", "Work_Pressure", "Academic_Pressure", "CGPA",
    "Study_Satisfaction", "Job_Satisfaction", "Work_Study_Hours",
    "Financial_Stress", "is_student", "Gender_Male", "Suicidal_Thoughts",
    "Family_History", "Dietary_Score",
]
CATEGORICAL_COLS = ["Profession", "Degree", "City"]


class LightGBMTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "lgb"

    def get_default_params(self) -> dict:
        """Reasonable defaults for binary depression classification."""
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "num_leaves": 63,
            "max_depth": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 5,
            "min_child_samples": 20,
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 5,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

    def create_pipeline(self, params: dict) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ]
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**params, verbose=-1)),
        ])


if __name__ == "__main__":
    trainer = LightGBMTrainer()
    trainer.run()
