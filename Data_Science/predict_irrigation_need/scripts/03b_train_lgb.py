"""LightGBM training for PS6E4.

Leaf-wise growth for complementary predictions to XGBoost.
CPU-based (LightGBM GPU build is optional).

Usage:
    uv run python Playground_Series/PS6E4/scripts/03b_train_lgb.py
"""

import numpy as np
import optuna
import lightgbm as lgb

from train_base import PS6E4Trainer


class LightGBMTrainer(PS6E4Trainer):
    """LightGBM multiclass trainer."""

    def __init__(self, version: int, feature_groups: list[str], n_trials: int = 50):
        super().__init__(feature_groups=feature_groups, n_trials=n_trials)
        self._version = version

    @property
    def model_name(self) -> str:
        return f"lgb_v{self._version}"

    @property
    def model_type(self) -> str:
        return "lightgbm"

    def get_search_space(self, trial: optuna.Trial) -> dict:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def create_model(self, params: dict):
        return lgb.LGBMClassifier(**params)

    def fit_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

    def predict_proba(self, model, X) -> np.ndarray:
        return model.predict_proba(X)

    def get_final_params(self, best_params: dict) -> dict:
        params = best_params.copy()
        params["objective"] = "multiclass"
        params["num_class"] = 3
        params["metric"] = "multi_logloss"
        params["is_unbalance"] = True
        params["random_state"] = self.seed
        params["n_jobs"] = -1
        params["verbosity"] = -1
        params["n_estimators"] = 5000
        return params


if __name__ == "__main__":
    trainer = LightGBMTrainer(
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
