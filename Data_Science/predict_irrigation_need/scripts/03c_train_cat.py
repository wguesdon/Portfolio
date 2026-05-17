"""CatBoost training for PS6E4.

Uses native categorical handling (no manual target encoding needed).
GPU-accelerated when available.

Usage:
    uv run python Playground_Series/PS6E4/scripts/03c_train_cat.py
"""

import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool

from train_base import PS6E4Trainer, FEAT_DIR

import json
import pandas as pd


class CatBoostTrainer(PS6E4Trainer):
    """CatBoost multiclass trainer with native categoricals and GPU."""

    def __init__(self, version: int, feature_groups: list[str], n_trials: int = 50):
        super().__init__(feature_groups=feature_groups, n_trials=n_trials)
        self._version = version
        self.use_gpu = self._detect_gpu()
        self.cat_feature_indices = []

    def _detect_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  GPU detected: {result.stdout.strip()}")
                return True
        except Exception:
            pass
        return False

    @property
    def model_name(self) -> str:
        return f"cat_v{self._version}"

    @property
    def model_type(self) -> str:
        return "catboost"

    def _get_cat_feature_indices(self, X: pd.DataFrame) -> list[int]:
        """Identify categorical columns by name pattern."""
        with open(FEAT_DIR / "metadata.json") as f:
            meta = json.load(f)
        cat_cols = meta["cat_cols"]
        indices = []
        for i, col in enumerate(X.columns):
            if col in cat_cols:
                indices.append(i)
        return indices

    def get_search_space(self, trial: optuna.Trial) -> dict:
        params = {
            "loss_function": "MultiClass",
            "classes_count": 3,
            "auto_class_weights": "Balanced",
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": self.seed,
            "verbose": 0,
        }
        if self.use_gpu:
            params["task_type"] = "GPU"
        return params

    def create_model(self, params: dict):
        return CatBoostClassifier(**params)

    def fit_model(self, model, X_train, y_train, X_val, y_val):
        if not self.cat_feature_indices:
            self.cat_feature_indices = self._get_cat_feature_indices(
                X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
            )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=self.cat_feature_indices if self.cat_feature_indices else None,
            early_stopping_rounds=100,
            verbose=0,
        )

    def predict_proba(self, model, X) -> np.ndarray:
        return model.predict_proba(X)

    def get_final_params(self, best_params: dict) -> dict:
        params = best_params.copy()
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3
        params["auto_class_weights"] = "Balanced"
        params["random_seed"] = self.seed
        params["verbose"] = 0
        params["iterations"] = 5000
        if self.use_gpu:
            params["task_type"] = "GPU"
        return params

    def save_model(self, model, path: str):
        model.save_model(f"{path}.cbm")


if __name__ == "__main__":
    # CatBoost uses raw categoricals (no TE needed).
    # Include raw_categorical_encoded so CatBoost can use native handling.
    trainer = CatBoostTrainer(
        version=100,
        feature_groups=[
            "raw_numeric",
            "raw_categorical_encoded",
            "snap",
            "domain",
            "boolean",
            "orig_priors",
        ],
        n_trials=50,
    )
    trainer.run()
