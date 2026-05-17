"""Base trainer for PS6E4: shared CV, OOF, and tracking logic.

All model training scripts inherit from PS6E4Trainer. It handles:
- Loading the feature store
- Stratified 5-fold CV (same folds for all models)
- OOF prediction saving as (n_samples, 3) arrays
- Test prediction saving as (n_samples, 3) arrays
- SQLite model tracking via tabml.tracking.ModelTracker
- Optuna Phase 1 + CV Phase 2 training pattern

Usage:
    This module is imported by the individual model scripts (03a, 03b, 03c, ...).
    It is not run directly.
"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Add tabml to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent / "tabml"))

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FEAT_DIR = DATA_DIR / "features"
PRED_DIR = BASE_DIR / "predictions"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

PRED_DIR.mkdir(parents=True, exist_ok=True)

# Competition constants
TARGET = "Irrigation_Need"
TARGET_ORDER = ["Low", "Medium", "High"]
METRIC = "balanced_accuracy"
N_FOLDS = 5
SEED = 42
N_CLASSES = 3


def load_feature_store(feature_groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load features from the parquet feature store.

    Args:
        feature_groups: List of feature group names to load
            (e.g. ["raw_numeric", "snap", "domain"]).

    Returns:
        Tuple of (X_train, X_test, y_train_encoded, label_encoder_classes).
    """
    with open(FEAT_DIR / "metadata.json") as f:
        meta = json.load(f)
    n_train = meta["n_train"]

    frames = []
    for group in feature_groups:
        path = FEAT_DIR / f"{group}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature group '{group}' not found at {path}")
        df = pd.read_parquet(path)
        frames.append(df)

    all_features = pd.concat(frames, axis=1)

    # Remove any duplicate columns
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    X_train = all_features.iloc[:n_train].reset_index(drop=True)
    X_test = all_features.iloc[n_train:].reset_index(drop=True)

    # Load target
    target_df = pd.read_parquet(FEAT_DIR / "target.parquet")
    le = LabelEncoder()
    le.fit(TARGET_ORDER)
    y_train = le.transform(target_df[TARGET])

    return X_train, X_test, y_train, le.classes_


class PS6E4Trainer(ABC):
    """Base trainer for PS6E4 competition models.

    Implements the 2-phase training pattern from the AWS BaseTrainer,
    adapted for multiclass classification with balanced accuracy metric.
    Runs locally or on AWS SageMaker.

    Attributes:
        n_folds: Number of CV folds.
        seed: Random state for reproducibility.
        n_trials: Number of Optuna trials for Phase 1.
        feature_groups: List of feature store groups to load.
    """

    def __init__(
        self,
        feature_groups: list[str],
        n_trials: int = 50,
        n_folds: int = N_FOLDS,
        seed: int = SEED,
    ):
        self.feature_groups = feature_groups
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.seed = seed

        self.oof_preds = None
        self.test_preds = None
        self.fold_scores = []
        self.best_params = None
        self.training_time = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short model identifier (e.g. 'xgb_v100')."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model family (e.g. 'xgboost', 'lightgbm', 'catboost')."""
        pass

    @abstractmethod
    def get_search_space(self, trial: optuna.Trial) -> dict:
        """Return Optuna search space."""
        pass

    @abstractmethod
    def create_model(self, params: dict):
        """Create a model instance with given parameters."""
        pass

    @abstractmethod
    def fit_model(self, model, X_train, y_train, X_val, y_val):
        """Fit model with validation set for early stopping."""
        pass

    @abstractmethod
    def predict_proba(self, model, X) -> np.ndarray:
        """Return class probabilities as (n_samples, 3) array."""
        pass

    @abstractmethod
    def get_final_params(self, best_params: dict) -> dict:
        """Prepare parameters for final CV training."""
        pass

    def save_model(self, model, path: str):
        """Save model to disk. Override for non-picklable models."""
        joblib.dump(model, f"{path}.pkl")

    def _compute_metric(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute balanced accuracy from probability predictions."""
        y_pred = y_proba.argmax(axis=1)
        return balanced_accuracy_score(y_true, y_pred)

    def run_optuna(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> dict:
        """Phase 1: Optuna hyperparameter tuning on a single split."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed
        )
        print(f"  Tuning split: train={len(X_train)}, val={len(X_val)}")

        def objective(trial):
            params = self.get_search_space(trial)
            model = self.create_model(params)
            self.fit_model(model, X_train, y_train, X_val, y_val)
            proba = self.predict_proba(model, X_val)
            return self._compute_metric(y_val, proba)

        study = optuna.create_study(direction="maximize")

        def callback(study, trial):
            if trial.number % 10 == 0:
                print(f"  [Trial {trial.number}/{self.n_trials}] Best: {study.best_value:.5f}")

        study.optimize(objective, n_trials=self.n_trials, callbacks=[callback])

        print(f"  Best tuning score: {study.best_value:.5f}")
        return study.best_params

    def train_with_cv(
        self,
        X: pd.DataFrame,
        X_test: pd.DataFrame,
        y: np.ndarray,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase 2: K-fold CV with best params. Saves OOF and test predictions."""
        final_params = self.get_final_params(params)
        kfold = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        self.oof_preds = np.zeros((len(X), N_CLASSES))
        self.test_preds = np.zeros((len(X_test), N_CLASSES))
        self.fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\n  --- Fold {fold + 1}/{self.n_folds} ---")

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = self.create_model(final_params)
            self.fit_model(model, X_tr, y_tr, X_val, y_val)

            # OOF predictions (n_samples, 3)
            self.oof_preds[val_idx] = self.predict_proba(model, X_val)

            # Test predictions (averaged across folds)
            self.test_preds += self.predict_proba(model, X_test) / self.n_folds

            fold_score = self._compute_metric(y_val, self.oof_preds[val_idx])
            self.fold_scores.append(fold_score)
            print(f"  Fold {fold + 1} balanced_accuracy: {fold_score:.5f}")

            # Save fold model
            fold_dir = MODEL_DIR / "level2"
            fold_dir.mkdir(parents=True, exist_ok=True)
            self.save_model(model, str(fold_dir / f"{self.model_name}_fold{fold}"))

        cv_score = self._compute_metric(y, self.oof_preds)
        cv_std = np.std(self.fold_scores)
        print(f"\n  CV balanced_accuracy: {cv_score:.5f} (+/- {cv_std:.5f})")

        return self.oof_preds, self.test_preds

    def save_predictions(self):
        """Save OOF and test predictions as .npy files."""
        np.save(PRED_DIR / f"oof_{self.model_name}.npy", self.oof_preds)
        np.save(PRED_DIR / f"pred_{self.model_name}.npy", self.test_preds)
        print(f"  Saved: oof_{self.model_name}.npy {self.oof_preds.shape}")
        print(f"  Saved: pred_{self.model_name}.npy {self.test_preds.shape}")

    def log_to_tracker(self, y: np.ndarray):
        """Log results to the SQLite model tracker."""
        from tabml.tracking import ModelTracker

        db_path = str(BASE_DIR / "experiments.db")
        tracker = ModelTracker(db_path=db_path)

        cv_score = self._compute_metric(y, self.oof_preds)
        tracker.log_model(
            name=self.model_name,
            version=int(self.model_name.split("_v")[-1]) if "_v" in self.model_name else 0,
            model_type=self.model_type,
            cv_score=cv_score,
            fold_scores=self.fold_scores,
            metric=METRIC,
            params=self.best_params,
            feature_group=",".join(self.feature_groups),
            n_features=self.oof_preds.shape[1] if self.oof_preds is not None else None,
            oof_path=str(PRED_DIR / f"oof_{self.model_name}.npy"),
            test_pred_path=str(PRED_DIR / f"pred_{self.model_name}.npy"),
            level=2,
            training_time_seconds=self.training_time,
        )
        print(f"  Logged to tracker: {self.model_name} (CV={cv_score:.5f})")

    def run(self):
        """Full training pipeline: load data, tune, train CV, save."""
        print(f"{'=' * 60}")
        print(f"  {self.model_name.upper()} TRAINING")
        print(f"{'=' * 60}")

        start = time.time()

        # Load features
        print(f"\nLoading features: {self.feature_groups}")
        X_train, X_test, y_train, classes = load_feature_store(self.feature_groups)
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Classes: {classes}")

        # Phase 1: Optuna
        print(f"\n{'=' * 40}")
        print(f"  PHASE 1: Optuna ({self.n_trials} trials)")
        print(f"{'=' * 40}")
        self.best_params = self.run_optuna(X_train, y_train)

        # Phase 2: CV
        print(f"\n{'=' * 40}")
        print(f"  PHASE 2: {self.n_folds}-fold CV")
        print(f"{'=' * 40}")
        self.train_with_cv(X_train, X_test, y_train, self.best_params)

        self.training_time = time.time() - start

        # Save
        print("\nSaving predictions...")
        self.save_predictions()
        self.log_to_tracker(y_train)

        cv_score = self._compute_metric(y_train, self.oof_preds)
        print(f"\n{'=' * 60}")
        print(f"  {self.model_name.upper()} COMPLETE")
        print(f"  CV balanced_accuracy: {cv_score:.5f}")
        print(f"  Time: {self.training_time:.0f}s")
        print(f"{'=' * 60}")
