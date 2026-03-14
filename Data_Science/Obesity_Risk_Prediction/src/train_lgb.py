#!/usr/bin/env python
"""
LightGBM trainer for obesity risk prediction.

Best params from Kaggle competition (moazeldsokyx, adapted):
  CV Accuracy ~0.909 (10-fold), ensemble contribution weight: 3
"""
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder
from lightgbm import LGBMClassifier

from base_trainer import BaseTrainer

# Raw feature groups — matches notebook's dynamically-computed column splits
NUMERICAL_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS",
]


class LightGBMTrainer(BaseTrainer):

    @property
    def model_name(self) -> str:
        return "lgb"

    def get_default_params(self) -> dict:
        """Best params from competition notebook (moazeldsokyx)."""
        return {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,  # notebook's best_params use 42 explicitly
            "num_class": 7,
            "learning_rate": 0.030962211546832760,
            "n_estimators": 500,
            "lambda_l1": 0.009667446568254372,
            "lambda_l2": 0.04018641437301800,
            "max_depth": 10,
            "colsample_bytree": 0.40977129346872643,
            "subsample": 0.9535797422450176,
            "min_child_samples": 26,
        }

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": self.random_state,
            "num_class": 7,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "num_leaves": trial.suggest_int("num_leaves", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 1.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 1.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
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
