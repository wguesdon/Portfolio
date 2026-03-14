"""
ODST Pairwise Ranking Neural Network for CIBMTR HCT survival prediction.

Faithful reproduction of the 0.692 LB notebook's PRL-NN architecture:
- Categorical embeddings (16-dim) → projection (112-dim)
- ODST layer (Neural Oblivious Decision Trees) from pytorch_tabular
- All-pairs pairwise ranking loss with proper censoring masks
- Race-group variance penalty for equity across groups
- Auxiliary MSE on log(efs_time) for multi-task regularization
- Stochastic Weight Averaging (SWA)

Trains on AWS SageMaker with GPU, saves fold weights for Kaggle inference.
"""
import functools
import json
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_tabular.models.common.layers import ODST
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HYPERPARAMS = json.loads(os.environ.get("SM_HPS", "{}"))
N_FOLDS = int(HYPERPARAMS.get("n_folds", 5))
SEED = int(HYPERPARAMS.get("random_state", 42))
MAX_EPOCHS = int(HYPERPARAMS.get("max_epochs", 60))

# Tuned hyperparams from the 0.692 notebook
EMBEDDING_DIM = 16
PROJECTION_DIM = 112
HIDDEN_DIM = 56
LR = 0.06464861983337984
DROPOUT = 0.05463240181423116
AUX_WEIGHT = 0.26545778308743806
MARGIN = 0.2588153271003354
WEIGHT_DECAY = 0.0002773544957610778
BATCH_SIZE = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── Feature types ─────────────────────────────────────────────────────────────
CATEGORICAL_VARIABLES = [
    'dri_score', 'graft_type', 'prod_type', 'prim_disease_hct',
    'psych_disturb', 'diabetes', 'arrhythmia', 'vent_hist', 'renal_issue',
    'pulm_moderate', 'pulm_severe', 'obesity', 'hepatic_mild',
    'hepatic_severe', 'peptic_ulcer', 'rheum_issue', 'cardiac',
    'prior_tumor', 'mrd_hct', 'tbi_status', 'cyto_score',
    'cyto_score_detail', 'ethnicity', 'race_group', 'sex_match',
    'donor_related', 'cmv_status', 'tce_imm_match', 'tce_match',
    'tce_div_match', 'melphalan_dose', 'rituximab', 'gvhd_proph',
    'in_vivo_tcd', 'conditioning_intensity',
]

HLA_COLUMNS = [
    'hla_match_a_low', 'hla_match_a_high', 'hla_match_b_low',
    'hla_match_b_high', 'hla_match_c_low', 'hla_match_c_high',
    'hla_match_dqb1_low', 'hla_match_dqb1_high', 'hla_match_drb1_low',
    'hla_match_drb1_high', 'hla_nmdp_6', 'hla_low_res_6', 'hla_high_res_6',
    'hla_low_res_8', 'hla_high_res_8', 'hla_low_res_10', 'hla_high_res_10',
]

OTHER_NUMERICAL = ['year_hct', 'donor_age', 'age_at_hct',
                   'comorbidity_score', 'karnofsky_score']


def get_feature_types(df):
    """Identify categorical and numerical columns from the dataframe.

    Categorical = object dtype or 2 < nunique < 25.
    Numerical = everything else minus ID, efs, efs_time.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (categorical_cols, numerical_cols).
    """
    cat_cols = [
        col for col in df.columns
        if (df[col].dtype == "object") or (2 < df[col].nunique() < 25)
    ]
    rmv = {"ID", "efs", "efs_time"}
    features = [c for c in df.columns if c not in rmv]
    num_cols = [c for c in features if c not in cat_cols]
    return cat_cols, num_cols


def preprocess_fold(train_df, val_df, cat_cols, num_cols):
    """Preprocess a single CV fold: label-encode categoricals, scale numericals.

    Args:
        train_df: Training fold DataFrame.
        val_df: Validation fold DataFrame.
        cat_cols: List of categorical column names.
        num_cols: List of numerical column names.

    Returns:
        Tuple of (X_cat_train, X_cat_val, X_num_train, X_num_val, transformers,
                  scaler, imputer).
    """
    # Handle unseen categories in val
    for col in cat_cols:
        unseen = ~val_df[col].isin(train_df[col])
        if unseen.any():
            val_df = val_df.copy()
            val_df.loc[unseen, col] = np.nan

    # Label encode
    transformers = []
    X_cat_train_list = []
    X_cat_val_list = []
    for col in cat_cols:
        le = LabelEncoder()
        train_vals = train_df[col].fillna("__missing__").astype(str)
        val_vals = val_df[col].fillna("__missing__").astype(str)
        le.fit(train_vals)
        # Handle unseen in val
        val_mapped = val_vals.map(
            lambda x, _le=le: x if x in _le.classes_ else "__missing__")
        if "__missing__" not in le.classes_:
            le.classes_ = np.append(le.classes_, "__missing__")
        X_cat_train_list.append(le.transform(train_vals))
        X_cat_val_list.append(le.transform(val_mapped))
        transformers.append(le)

    X_cat_train = np.column_stack(X_cat_train_list)
    X_cat_val = np.column_stack(X_cat_val_list)

    # Numerical: impute + scale (with missing indicators)
    imputer = SimpleImputer(strategy='mean', add_indicator=True)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(
        imputer.fit_transform(train_df[num_cols].values))
    X_num_val = scaler.transform(imputer.transform(val_df[num_cols].values))

    return (X_cat_train, X_cat_val,
            X_num_train.astype(np.float32), X_num_val.astype(np.float32),
            transformers, scaler, imputer)


def make_dataloader(X_cat, X_num, efs_time, efs, batch_size, shuffle=False):
    """Create a DataLoader with log-transformed efs_time.

    Args:
        X_cat: Categorical features (integer-encoded).
        X_num: Numerical features (scaled).
        efs_time: Event/censoring times (will be log-transformed).
        efs: Event indicators.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        PyTorch DataLoader.
    """
    ds = TensorDataset(
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(efs_time, dtype=torch.float32).log(),
        torch.tensor(efs, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


# ── Model ─────────────────────────────────────────────────────────────────────
class CatEmbeddings(nn.Module):
    """Embedding module for categorical features with projection.

    Args:
        projection_dim: Output dimension after projection.
        categorical_cardinality: Number of unique categories per feature.
        embedding_dim: Embedding dimension per feature.
    """
    def __init__(self, projection_dim, categorical_cardinality, embedding_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embedding_dim)
            for card in categorical_cardinality
        ])
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * len(categorical_cardinality), projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x_cat):
        embs = [self.embeddings[i](x_cat[:, i])
                for i in range(x_cat.shape[1])]
        x = torch.cat(embs, dim=1)
        return self.projection(x)


class ODSTNN(nn.Module):
    """Neural network with ODST layer for tabular data.

    Architecture: CatEmbeddings → concat(cat_proj, numerical) → ODST → BN →
    Dropout → Linear → risk score. Auxiliary head for efs_time regression.

    Args:
        continuous_dim: Number of continuous features.
        categorical_cardinality: List of cardinalities per categorical feature.
        embedding_dim: Embedding dimension.
        projection_dim: Categorical projection dimension.
        hidden_dim: ODST output dimension.
        dropout: Dropout probability.
    """
    def __init__(self, continuous_dim, categorical_cardinality, embedding_dim,
                 projection_dim, hidden_dim, dropout=0.05):
        super().__init__()
        self.embeddings = CatEmbeddings(
            projection_dim, categorical_cardinality, embedding_dim)
        self.mlp = nn.Sequential(
            ODST(projection_dim + continuous_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_cat, x_cont):
        x = self.embeddings(x_cat)
        x = torch.cat([x, x_cont], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)
        return self.out(x).squeeze(1), x


@functools.lru_cache
def combinations(N):
    """Compute all 2-combinations of indices 0..N-1 on GPU.

    Args:
        N: Number of elements.

    Returns:
        Tensor of shape (C(N,2), 2) with all pairs.
    """
    ind = torch.arange(N)
    comb = torch.combinations(ind, r=2)
    return comb.cuda()


class LitODSTNN(pl.LightningModule):
    """PyTorch Lightning module for ODST pairwise ranking NN.

    Implements the full training loop with:
    - All-pairs pairwise ranking loss with censoring masks
    - Race-group variance penalty
    - Auxiliary MSE loss on log(efs_time)
    - Stratified CI evaluation

    Args:
        continuous_dim: Number of continuous features.
        categorical_cardinality: List of cardinalities.
        embedding_dim: Embedding dimension.
        projection_dim: Categorical projection dimension.
        hidden_dim: ODST output dimension.
        lr: Learning rate.
        dropout: Dropout probability.
        weight_decay: L2 regularization.
        aux_weight: Weight for auxiliary MSE loss.
        margin: Margin for pairwise ranking loss.
        race_index: Index of race_group in categorical features.
    """
    def __init__(self, continuous_dim, categorical_cardinality, embedding_dim,
                 projection_dim, hidden_dim, lr, dropout, weight_decay,
                 aux_weight, margin, race_index):
        super().__init__()
        self.save_hyperparameters()

        self.model = ODSTNN(
            continuous_dim=continuous_dim,
            categorical_cardinality=categorical_cardinality,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.targets = []

        # Auxiliary head for efs_time regression
        self.aux_cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, 1),
        )

    def forward(self, x_cat, x_cont):
        x, emb = self.model(x_cat, x_cont)
        return x, emb

    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        aux_pred = self.aux_cls(emb).squeeze(1)

        # Main pairwise ranking loss + race penalty
        loss, race_loss = self._get_full_loss(efs, x_cat, y, y_hat)

        # Auxiliary MSE on log(efs_time), only for events
        aux_loss = F.mse_loss(aux_pred, y, reduction='none')
        aux_mask = efs == 1
        if aux_mask.sum() > 0:
            aux_loss = (aux_loss * aux_mask).sum() / aux_mask.sum()
        else:
            aux_loss = aux_loss.mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("race_loss", race_loss, on_epoch=True, prog_bar=True,
                 on_step=False)
        return loss + aux_loss * self.hparams.aux_weight

    def _get_full_loss(self, efs, x_cat, y, y_hat):
        """Compute pairwise ranking loss + race variance penalty."""
        loss = self._calc_loss(y, y_hat, efs)
        race_loss = self._get_race_losses(efs, x_cat, y, y_hat)
        loss = loss + 0.1 * race_loss
        return loss, race_loss

    def _get_race_losses(self, efs, x_cat, y, y_hat):
        """Compute std of per-race losses for equity."""
        races = torch.unique(x_cat[:, self.hparams.race_index])
        race_losses = []
        for race in races:
            ind = x_cat[:, self.hparams.race_index] == race
            if ind.sum() < 2:
                continue
            race_losses.append(self._calc_loss(y[ind], y_hat[ind], efs[ind]))
        if len(race_losses) < 2:
            return torch.tensor(0.0, device=y.device)
        race_loss_mean = sum(race_losses) / len(race_losses)
        variance = sum((r - race_loss_mean) ** 2 for r in race_losses) / len(
            race_losses)
        return torch.sqrt(variance + 1e-8)

    def _calc_loss(self, y, y_hat, efs):
        """All-pairs pairwise ranking loss with censoring masks.

        For all pairs (i, j) with at least one event, compute margin-based
        hinge loss with proper masking for censored observations.
        """
        N = y.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=y.device)

        comb = combinations(N)
        # Keep pairs with at least one event
        has_event = (efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)
        comb = comb[has_event]

        if len(comb) == 0:
            return torch.tensor(0.0, device=y.device)

        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = y[comb[:, 0]]
        y_right = y[comb[:, 1]]

        # +1 if left > right (left survived longer), -1 otherwise
        target = 2 * (y_left > y_right).int() - 1
        loss = F.relu(-target.float() * (pred_left - pred_right)
                      + self.hparams.margin)

        # Mask invalid comparisons involving censored observations
        mask = self._get_mask(comb, efs, y_left, y_right)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y.device)
        return (loss.double() * mask.double()).sum() / mask.sum()

    def _get_mask(self, comb, efs, y_left, y_right):
        """Create mask for valid survival pairs.

        Invalid: censored patient appears to outlive event patient, but we
        can't know (censored time >= event time when censored has no event).
        """
        left_outlived = y_left >= y_right
        left_event_right_cens = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        mask = left_outlived & left_event_right_cens

        right_outlived = y_right >= y_left
        right_event_left_cens = (efs[comb[:, 1]] == 1) & (efs[comb[:, 0]] == 0)
        mask = mask | (right_outlived & right_event_left_cens)

        return ~mask

    def validation_step(self, batch, batch_idx):
        x_cat, x_cont, y, efs = batch
        y_hat, emb = self(x_cat, x_cont)
        loss, race_loss = self._get_full_loss(efs, x_cat, y, y_hat)
        self.targets.append([y, y_hat.detach(), efs,
                             x_cat[:, self.hparams.race_index]])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.targets:
            return
        y = torch.cat([t[0] for t in self.targets]).cpu().numpy()
        y_hat = torch.cat([t[1] for t in self.targets]).cpu().numpy()
        efs = torch.cat([t[2] for t in self.targets]).cpu().numpy()
        races = torch.cat([t[3] for t in self.targets]).cpu().numpy()

        # Compute stratified CI
        cis = []
        for race in np.unique(races):
            mask = races == race
            if mask.sum() < 5:
                continue
            ci = self._harrell_ci(np.exp(y[mask]), y_hat[mask], efs[mask])
            cis.append(ci)
        metric = float(np.mean(cis) - np.std(cis)) if cis else 0.0
        self.log("cindex", metric, on_epoch=True, prog_bar=True)
        self.targets.clear()

    @staticmethod
    def _harrell_ci(event_times, scores, events):
        """Harrell C-index. Higher score = longer survival."""
        events = events.astype(bool)
        ev_times = event_times[events]
        ev_scores = scores[events]
        concordant = permissible = 0.0
        batch = 500
        for i in range(0, len(ev_times), batch):
            t_i = ev_times[i:i + batch, None]
            s_i = ev_scores[i:i + batch, None]
            mask = event_times[None, :] > t_i
            concordant += float(((scores[None, :] > s_i) & mask).sum())
            concordant += 0.5 * float(((scores[None, :] == s_i) & mask).sum())
            permissible += float(mask.sum())
        return concordant / permissible if permissible > 0 else 0.5

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=45, eta_min=6e-3),
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pl.seed_everything(SEED)
    print("=" * 60)
    print("CIBMTR HCT — ODST Pairwise Ranking NN v4")
    print("=" * 60)
    print(f"Hyperparams: emb={EMBEDDING_DIM}, proj={PROJECTION_DIM}, "
          f"hidden={HIDDEN_DIM}")
    print(f"  lr={LR:.4f}, dropout={DROPOUT:.4f}, margin={MARGIN:.4f}")
    print(f"  aux_weight={AUX_WEIGHT:.4f}, wd={WEIGHT_DECAY:.6f}")
    print(f"  batch={BATCH_SIZE}, epochs={MAX_EPOCHS}")

    # ── Load data ─────────────────────────────────────────────────
    train_raw = pd.read_csv(INPUT_DIR / "train.csv")
    test_raw = pd.read_csv(INPUT_DIR / "test.csv")
    print(f"\nTrain: {train_raw.shape}  Test: {test_raw.shape}")

    # Minimal preprocessing (matching 0.692 notebook)
    for df in [train_raw, test_raw]:
        df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna("Unknown")
        df[OTHER_NUMERICAL] = df[OTHER_NUMERICAL].fillna(
            df[OTHER_NUMERICAL].median())
        df['year_hct'] = df['year_hct'] - 2000

    # Fill test efs/efs_time for consistent processing
    test_raw['efs_time'] = 1.0
    test_raw['efs'] = 1

    cat_cols, num_cols = get_feature_types(train_raw)
    print(f"Categorical: {len(cat_cols)}  Numerical: {len(num_cols)}")
    race_index = cat_cols.index("race_group")
    print(f"Race index in cat_cols: {race_index}")

    # ── CV ────────────────────────────────────────────────────────
    efs_time_arr = train_raw['efs_time'].values
    efs_arr = train_raw['efs'].values
    race_group = train_raw['race_group'].values

    strat = (train_raw['race_group'].astype(str)
             + (train_raw['age_at_hct'] == 0.044).astype(str))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(train_raw))
    test_preds = np.zeros(len(test_raw))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_raw, strat)):
        print(f"\n{'=' * 40} Fold {fold + 1}/{N_FOLDS} {'=' * 40}")

        train_fold = train_raw.iloc[tr_idx].copy()
        val_fold = train_raw.iloc[val_idx].copy()

        # Preprocess
        (X_cat_tr, X_cat_val, X_num_tr, X_num_val,
         transformers, scaler, imputer) = preprocess_fold(
            train_fold, val_fold, cat_cols, num_cols)

        dl_train = make_dataloader(
            X_cat_tr, X_num_tr,
            train_fold['efs_time'].values, train_fold['efs'].values,
            BATCH_SIZE, shuffle=True)
        dl_val = make_dataloader(
            X_cat_val, X_num_val,
            val_fold['efs_time'].values, val_fold['efs'].values,
            BATCH_SIZE, shuffle=False)

        # Model
        model = LitODSTNN(
            continuous_dim=X_num_tr.shape[1],
            categorical_cardinality=[len(t.classes_) for t in transformers],
            embedding_dim=EMBEDDING_DIM,
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            lr=LR,
            dropout=DROPOUT,
            weight_decay=WEIGHT_DECAY,
            aux_weight=AUX_WEIGHT,
            margin=MARGIN,
            race_index=race_index,
        )

        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Model parameters: {n_params:,}")

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss", save_top_k=1, mode="min")
        trainer = pl.Trainer(
            accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
            max_epochs=MAX_EPOCHS,
            log_every_n_steps=6,
            enable_progress_bar=True,
            callbacks=[
                checkpoint_cb,
                LearningRateMonitor(logging_interval='epoch'),
                StochasticWeightAveraging(
                    swa_lrs=1e-5, swa_epoch_start=45, annealing_epochs=15),
            ],
        )
        trainer.fit(model, dl_train, dl_val)

        # Predict OOF
        model.eval().to(DEVICE)
        with torch.no_grad():
            oof_pred, _ = model(
                torch.tensor(X_cat_val, dtype=torch.long).to(DEVICE),
                torch.tensor(X_num_val, dtype=torch.float32).to(DEVICE),
            )
        oof_preds[val_idx] = oof_pred.cpu().numpy()

        # Predict test (reprocess with train_fold)
        test_fold = test_raw.copy()
        (_, X_cat_test, _, X_num_test,
         _, _, _) = preprocess_fold(train_fold, test_fold, cat_cols, num_cols)
        with torch.no_grad():
            test_pred, _ = model(
                torch.tensor(X_cat_test, dtype=torch.long).to(DEVICE),
                torch.tensor(X_num_test, dtype=torch.float32).to(DEVICE),
            )
        test_preds += test_pred.cpu().numpy()

        # Fold CI
        fold_ci = _stratified_ci(
            efs_time_arr[val_idx], oof_preds[val_idx],
            efs_arr[val_idx], race_group[val_idx])
        fold_scores.append(fold_ci)
        print(f"  Fold {fold + 1} Stratified CI = {fold_ci:.4f}")

        # Save fold weights
        torch.save(model.state_dict(), OUTPUT_DIR / f"nn_fold{fold}.pt")

        # Save fold preprocessing for inference
        fold_meta = {
            "transformers_classes": [t.classes_.tolist() for t in transformers],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        with open(OUTPUT_DIR / f"nn_fold{fold}_preproc.json", "w") as f:
            json.dump(fold_meta, f)

    # ── OOF score ─────────────────────────────────────────────────
    # Negate predictions (model outputs survival score, higher = longer survival)
    oof_risk = -oof_preds
    test_risk = -test_preds

    oof_ci = _stratified_ci(efs_time_arr, oof_preds, efs_arr, race_group)
    print(f"\n{'=' * 60}")
    print(f"OOF Stratified CI: {oof_ci:.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # ── Save artifacts ────────────────────────────────────────────
    np.save(OUTPUT_DIR / "nn_oof_rank.npy", oof_risk)
    np.save(OUTPUT_DIR / "nn_oof_cls.npy", oof_preds)
    np.save(OUTPUT_DIR / "nn_test_rank.npy", test_risk)
    np.save(OUTPUT_DIR / "nn_test_cls.npy", test_preds)

    metadata = {
        "model": "odst_pairwise_ranking_nn_v4",
        "n_folds": N_FOLDS,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "race_index": race_index,
        "embedding_dim": EMBEDDING_DIM,
        "projection_dim": PROJECTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "oof_ci": oof_ci,
        "fold_scores": [float(s) for s in fold_scores],
        "score_direction": "risk (higher = higher risk, negated from model)",
    }
    with open(OUTPUT_DIR / "nn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {OUTPUT_DIR}")


def _stratified_ci(efs_time, scores, efs, race_group):
    """Compute stratified CI (mean - std across race groups).

    Args:
        efs_time: Event/censoring times.
        scores: Survival scores (higher = longer survival).
        efs: Event indicators.
        race_group: Race group labels.

    Returns:
        Stratified concordance index.
    """
    cis = []
    for race in np.unique(race_group):
        mask = race_group == race
        if mask.sum() < 5:
            continue
        ci = LitODSTNN._harrell_ci(efs_time[mask], scores[mask], efs[mask])
        cis.append(ci)
    return float(np.mean(cis) - np.std(cis))


if __name__ == "__main__":
    main()
