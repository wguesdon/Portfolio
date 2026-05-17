"""XGBoost training for PS6E4.

Trains multiple XGBoost variants with different feature sets.
Uses GPU when available.

Usage:
    uv run python Playground_Series/PS6E4/scripts/03a_train_xgb.py
"""

import numpy as np
import optuna
import xgboost as xgb

from train_base import PS6E4Trainer


class XGBoostTrainer(PS6E4Trainer):
    """XGBoost multiclass trainer with GPU support."""

    def __init__(self, version: int, feature_groups: list[str], n_trials: int = 50):
        super().__init__(feature_groups=feature_groups, n_trials=n_trials)
        self._version = version
        self.use_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Check if CUDA GPU is available for XGBoost."""
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
        return f"xgb_v{self._version}"

    @property
    def model_type(self) -> str:
        return "xgboost"

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": self.seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "device": "cuda" if self.use_gpu else "cpu",
            "verbosity": 0,
        }

    def create_model(self, params: dict):
        return xgb.XGBClassifier(**params)

    def fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def predict_proba(self, model, X) -> np.ndarray:
        return model.predict_proba(X)

    def get_final_params(self, best_params: dict) -> dict:
        params = best_params.copy()
        params["objective"] = "multi:softprob"
        params["num_class"] = 3
        params["eval_metric"] = "mlogloss"
        params["random_state"] = self.seed
        params["n_jobs"] = -1
        params["tree_method"] = "hist"
        params["device"] = "cuda" if self.use_gpu else "cpu"
        params["verbosity"] = 0
        params["n_estimators"] = 5000
        params["early_stopping_rounds"] = 200
        return params


if __name__ == "__main__":
    # v100: Baseline with core features
    trainer = XGBoostTrainer(
        version=100,
        feature_groups=[
            "raw_numeric",
            "raw_categorical_encoded",
            "snap",
            "domain",
            "frequency",
            "boolean",
            "orig_priors",
        ],
        n_trials=50,
    )
    trainer.run()
