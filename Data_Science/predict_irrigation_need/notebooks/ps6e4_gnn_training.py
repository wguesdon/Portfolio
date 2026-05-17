# %% [markdown]
# # PS6E4 — GraphSAGE GNN for Irrigation Prediction
# Based on ravi20076's notebook (adapted from cdeotte's S6E3 starter).
# Trains a 2-layer GraphSAGE with categorical embeddings using 5-fold CV.

# %% Check GPU and install compatible PyTorch BEFORE any torch import
import subprocess, sys, os

# Detect GPU via nvidia-smi before importing torch
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True).strip()
    gpu_name, cc_str = smi.split(", ")
    cc_major = int(cc_str.split(".")[0])
    print(f"[GPU Check] {gpu_name}, compute capability {cc_str}")
except Exception as e:
    gpu_name, cc_major = "unknown", 7
    print(f"[GPU Check] nvidia-smi failed ({e}), assuming cc >= 7.0")

if cc_major < 7:
    print(f"[GPU Check] P100 detected (sm_60). Reinstalling PyTorch with cu118 for sm_60 support...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "torch==2.2.2+cu118", "torchvision==0.17.2+cu118",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    ])
    print("[GPU Check] PyTorch reinstalled. Installing torch-geometric...")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"])

# %%
# ============================================================
# GNN — Irrigation Need (3-class: Low / Medium / High)
# GraphSAGE + Categorical Embeddings | StratifiedKFold CV
# ============================================================

# ============================================================
# SECTION 1 — IMPORTS & CONFIG
# ============================================================
import gc
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from cuml.neighbors import NearestNeighbors as cumlNN
    USE_CUML = True
    print("[Init] cuML NearestNeighbors available")
except (ImportError, RuntimeError):
    USE_CUML = False
    print("[Init] cuML not available, will use sklearn NearestNeighbors")
from sklearn.neighbors import NearestNeighbors as sklearnNN

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e4/train.csv"
TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e4/test.csv"
VERSION_NB = "v1"
PATIENCE   = 20
EPOCHS     = 100

# ----------------------------
# FIXED CONFIG
# ----------------------------
SEED         = 42
N_FOLDS      = 5
K            = 8               # KNN neighbors for graph
VAL_EVERY    = 1
BATCH_SIZE   = 8192
INFER_BATCH  = 16384
FANOUTS      = [6, 4]
GRAPH_NUM_MULTIPLIER = 3.0
USE_AMP      = True
RARE_MIN     = 25

TARGET       = "Irrigation_Need"
LABEL_MAP    = {"Low": 0, "Medium": 1, "High": 2}
LABEL_INV    = {0: "Low", 1: "Medium", 2: "High"}
N_CLASSES    = 3

NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"
]
CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"
]

# Derived feature names added during feature engineering
CAT_PROXY = [f"{c}__cat"     for c in NUMS]
CAT_RARE  = [f"{c}__is_rare" for c in NUMS]
ALL_CATS  = CATS + CAT_PROXY + CAT_RARE   # node categorical features
ALL_NUMS  = NUMS[:]                        # node numeric features

GRAPH_CAT_COLS = CATS[:]   # used for KNN graph construction
GRAPH_NUM_COLS = NUMS[:]

# Multi-GPU: use all available GPUs, fall back to CPU
N_GPUS  = torch.cuda.device_count()
DEVICE  = torch.device("cuda:0" if N_GPUS >= 1 else "cpu")
print(f"Detected {N_GPUS} GPU(s). Primary device: {DEVICE}")

# Reproducibility
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# SECTION 2 — PREPROCESSING
# ============================================================

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    - Numeric: coerce, median-fill (train median applied to both)
    - Categorical: strip whitespace, fill missing
    - Target: label-encode to int
    """
    print("\n[Preprocess] Starting...")
    tr = train_df.copy()
    te = test_df.copy()

    # Numeric
    for c in NUMS:
        tr[c] = pd.to_numeric(tr[c], errors="coerce").astype(np.float32)
        te[c] = pd.to_numeric(te[c], errors="coerce").astype(np.float32)
        med = float(np.nanmedian(tr[c].values))
        med = med if np.isfinite(med) else 0.0
        tr[c] = tr[c].fillna(med)
        te[c] = te[c].fillna(med)

    # Categorical
    for c in CATS:
        tr[c] = tr[c].astype(str).str.strip().fillna("missing")
        te[c] = te[c].astype(str).str.strip().fillna("missing")

    # Target
    y = tr[TARGET].map(LABEL_MAP).values.astype(np.int64)

    print(f"[Preprocess] Train shape: {tr.shape} | Test shape: {te.shape}")
    print(f"[Preprocess] Target distribution: { {LABEL_INV[k]: int((y==k).sum()) for k in range(N_CLASSES)} }")
    return tr, te, y


# ============================================================
# SECTION 3 — FEATURE ENGINEERING
# ============================================================

def _build_snapper(train_series: pd.Series):
    """
    Fit a rare-value snapper on train. Returns a transform function.
    Rare = appears fewer than RARE_MIN times.
    Rare values are snapped to the nearest frequent value.
    """
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = s.value_counts(dropna=False)
    frequent = np.sort(
        np.array([v for v in vc[vc >= RARE_MIN].index if pd.notna(v)], dtype=np.float32)
    )
    if frequent.size == 0:
        frequent = np.sort(s.dropna().unique().astype(np.float32))

    freq_set = set(frequent.tolist())

    def transform(series):
        x = pd.to_numeric(series, errors="coerce").astype(np.float32).values
        is_nan  = np.isnan(x)
        is_rare = np.ones(len(x), dtype=np.int32)

        for i, v in enumerate(x):
            if not np.isnan(v) and float(v) in freq_set:
                is_rare[i] = 0

        x_snapped = x.copy()
        snap_idx  = np.where((~is_nan) & (is_rare == 1))[0]
        if snap_idx.size > 0 and frequent.size > 0:
            v     = x[snap_idx]
            pos   = np.clip(np.searchsorted(frequent, v), 0, len(frequent) - 1)
            left  = np.clip(pos - 1, 0, len(frequent) - 1)
            right = pos
            nearest = np.where(
                np.abs(v - frequent[right]) <= np.abs(v - frequent[left]),
                frequent[right], frequent[left]
            )
            x_snapped[snap_idx] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    For each numeric column: add __cat (snapped value as string) and __is_rare flag.
    Snappers are fit on train only.
    """
    print("\n[Feature Engineering] Building rare-snap features...")
    tr = train_df.copy()
    te = test_df.copy()

    for col in NUMS:
        snapper = _build_snapper(tr[col])
        tr_snap, tr_rare = snapper(tr[col])
        te_snap, te_rare = snapper(te[col])

        tr[f"{col}__cat"]     = pd.Series(tr_snap).astype(str).values
        te[f"{col}__cat"]     = pd.Series(te_snap).astype(str).values
        tr[f"{col}__is_rare"] = pd.Series(tr_rare).astype(str).values
        te[f"{col}__is_rare"] = pd.Series(te_rare).astype(str).values

    for df in (tr, te):
        for c in ALL_CATS:
            df[c] = df[c].astype(str).fillna("missing")

    print(f"[Feature Engineering] Node categorical features : {len(ALL_CATS)}")
    print(f"[Feature Engineering] Node numeric features     : {len(ALL_NUMS)}")
    return tr, te


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Integer-encode all categorical node features using a shared vocab
    built from train + test (no unseen categories at inference).
    Returns integer matrices and per-feature cardinalities.
    """
    print("\n[Encode] Encoding categorical node features...")
    tr_codes, te_codes, cardinalities = [], [], []

    for c in ALL_CATS:
        all_vals = pd.concat(
            [train_df[c].astype(str), test_df[c].astype(str)], ignore_index=True
        )
        mapping = {v: i for i, v in enumerate(all_vals.unique())}
        tr_codes.append(train_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        te_codes.append(test_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        cardinalities.append(len(mapping))

    Xc_tr = np.stack(tr_codes, axis=1)
    Xc_te = np.stack(te_codes, axis=1)
    print(f"[Encode] Categorical matrix — train: {Xc_tr.shape} | test: {Xc_te.shape}")
    return Xc_tr, Xc_te, cardinalities


def scale_numerics(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """StandardScale numeric node features, fit on train."""
    print("\n[Scale] Scaling numeric node features...")
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(
        train_df[ALL_NUMS].values.astype(np.float32)
    ).astype(np.float32)
    Xn_te = scaler.transform(
        test_df[ALL_NUMS].values.astype(np.float32)
    ).astype(np.float32)
    print(f"[Scale] Numeric matrix — train: {Xn_tr.shape} | test: {Xn_te.shape}")
    return Xn_tr, Xn_te


def build_knn_graph(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = K):
    """
    Build KNN graph on combined train+test using cuML.
    Graph features: OHE of base categoricals + standardized numerics * multiplier.
    Returns neighbors array of shape [n_total, k].
    """
    print(f"\n[Graph] Building KNN graph (k={k}) on {len(train_df) + len(test_df):,} nodes with cuML...")

    graph_cat = pd.concat(
        [train_df[GRAPH_CAT_COLS].astype(str), test_df[GRAPH_CAT_COLS].astype(str)],
        ignore_index=True
    )
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    X_cat_ohe = ohe.fit_transform(graph_cat).astype(np.float32)

    graph_num_tr = train_df[GRAPH_NUM_COLS].copy()
    graph_num_te = test_df[GRAPH_NUM_COLS].copy()
    for c in GRAPH_NUM_COLS:
        graph_num_tr[c] = pd.to_numeric(graph_num_tr[c], errors="coerce").fillna(0).astype(np.float32)
        graph_num_te[c] = pd.to_numeric(graph_num_te[c], errors="coerce").fillna(0).astype(np.float32)

    num_scaler = StandardScaler()
    X_num_tr = num_scaler.fit_transform(graph_num_tr.values.astype(np.float32))
    X_num_te = num_scaler.transform(graph_num_te.values.astype(np.float32))
    X_num    = np.vstack([X_num_tr, X_num_te]).astype(np.float32) * GRAPH_NUM_MULTIPLIER

    X_graph = np.concatenate([X_cat_ohe, X_num], axis=1).astype(np.float32)
    print(f"[Graph] Graph feature matrix shape: {X_graph.shape}")

    # Try cuML first (GPU accelerated), fall back to sklearn (CPU)
    if USE_CUML:
        try:
            print("[Graph] Trying cuML GPU kNN...")
            knn = cumlNN(n_neighbors=k)
            knn.fit(X_graph)
            _, idx = knn.kneighbors(X_graph)
            neighbors = idx.astype(np.int32)
            print("[Graph] cuML kNN succeeded")
        except RuntimeError as e:
            print(f"[Graph] cuML failed ({e}), falling back to sklearn...")
            knn = sklearnNN(n_neighbors=k, algorithm="auto", n_jobs=-1)
            knn.fit(X_graph)
            _, idx = knn.kneighbors(X_graph)
            neighbors = idx.astype(np.int32)
    else:
        print("[Graph] Using sklearn CPU kNN...")
        knn = sklearnNN(n_neighbors=k, algorithm="auto", n_jobs=-1)
        knn.fit(X_graph)
        _, idx = knn.kneighbors(X_graph)
        neighbors = idx.astype(np.int32)

    print(f"[Graph] Neighbors matrix shape: {neighbors.shape}")
    del X_graph, X_cat_ohe, X_num, idx, knn
    gc.collect()

    return neighbors


# ============================================================
# SECTION 4 — MODEL DEFINITION
# ============================================================

def _emb_dim(cardinality: int) -> int:
    return int(np.clip(round(1.8 * (cardinality ** 0.25)), 4, 24))


class CatEmbed(nn.Module):
    def __init__(self, cardinalities):
        super().__init__()
        self.embs = nn.ModuleList()
        self.out_dim = 0
        for card in cardinalities:
            card = max(2, int(card))
            d = _emb_dim(card)
            self.embs.append(nn.Embedding(card, d))
            self.out_dim += d
        for e in self.embs:
            nn.init.normal_(e.weight, 0.0, 0.02)

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)


class IrrigationGNN(nn.Module):
    """
    2-layer GraphSAGE with categorical embeddings.
    Residual connections + LayerNorm between SAGE layers.
    Output: 3-class logits.
    """
    def __init__(self, num_in: int, cardinalities: list, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cat    = CatEmbed(cardinalities)
        in_dim      = num_in + self.cat.out_dim

        self.lin_in = nn.Linear(in_dim, hidden)
        self.conv1  = SAGEConv(hidden, hidden)
        self.conv2  = SAGEConv(hidden, hidden)
        self.norm1  = nn.LayerNorm(hidden)
        self.norm2  = nn.LayerNorm(hidden)
        self.drop   = dropout

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, data):
        x = torch.cat([data.x_num, self.cat(data.x_cat)], dim=1)
        x = F.dropout(F.relu(self.lin_in(x)), p=self.drop, training=self.training)

        x1 = F.relu(self.norm1(self.conv1(x,  data.edge_index)))
        x1 = F.dropout(x1, p=self.drop, training=self.training)
        x  = x + 0.5 * x1

        x2 = F.relu(self.norm2(self.conv2(x, data.edge_index)))
        x2 = F.dropout(x2, p=self.drop, training=self.training)
        x  = x + 0.5 * x2

        return self.head(x)   # [N, 3]


# ============================================================
# SECTION 5 — SUBGRAPH SAMPLING UTILITIES
# ============================================================

_global_pos = None   # reused buffer, initialised per-fold inside fit()


def _build_subgraph(seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu, fanouts, device, offset=0):
    global _global_pos
    seed_nodes = np.asarray(seed_nodes, dtype=np.int32)
    frontier   = seed_nodes
    collected  = [seed_nodes]

    for hop, fanout in enumerate(fanouts):
        nbr   = neighbors[frontier]
        start = (offset + hop) % nbr.shape[1]
        cols  = (np.arange(fanout) + start) % nbr.shape[1]
        frontier = np.unique(nbr[:, cols].reshape(-1))
        collected.append(frontier)

    nodes = np.unique(np.concatenate(collected))
    m     = len(nodes)
    _global_pos[nodes] = np.arange(m, dtype=np.int32)

    sub_nbr      = neighbors[nodes]
    dst_local    = _global_pos[sub_nbr]
    mask         = dst_local >= 0
    src_l        = np.repeat(np.arange(m, dtype=np.int64), sub_nbr.shape[1])[mask.reshape(-1)]
    dst_l        = dst_local[mask].astype(np.int64)
    edge_index   = torch.tensor(np.vstack([src_l, dst_l]), dtype=torch.long, device=device)

    # Convert numpy indices to torch for compatibility with numpy 2.x + older PyTorch
    nodes_t = torch.from_numpy(nodes.astype(np.int64))
    batch = Data(
        x_num      = x_num_cpu[nodes_t].to(device, non_blocking=True),
        x_cat      = x_cat_cpu[nodes_t].to(device, non_blocking=True),
        y          = y_cpu[nodes_t].to(device, non_blocking=True),
        edge_index = edge_index,
    )
    batch.seed_local = torch.tensor(_global_pos[seed_nodes], dtype=torch.long, device=device)
    _global_pos[nodes] = -1
    return batch


def _seed_batches(seed_nodes, batch_size, shuffle):
    arr = np.asarray(seed_nodes, dtype=np.int32).copy()
    if shuffle:
        np.random.shuffle(arr)
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]


@torch.no_grad()
def _predict_nodes(model, seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu,
                   fanouts, batch_size, device, offset=0):
    model.eval()
    out = np.zeros((len(seed_nodes), N_CLASSES), dtype=np.float32)
    pos = 0
    for batch_seeds in _seed_batches(seed_nodes, batch_size, shuffle=False):
        batch  = _build_subgraph(batch_seeds, neighbors, x_num_cpu, x_cat_cpu,
                                  y_cpu, fanouts, device, offset)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(USE_AMP and device.type == "cuda")):
            logits = model(batch)
        probs = F.softmax(logits[batch.seed_local], dim=1).float().cpu().numpy()
        out[pos:pos + len(batch_seeds)] = probs
        pos += len(batch_seeds)
        del batch, logits, probs
    return out   # [n_seeds, 3]


# ============================================================
# SECTION 6 — SKLEARN-STYLE WRAPPER
# ============================================================

class IrrigationGNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-style wrapper for the 3-class GNN.

    Parameters
    ----------
    hidden      : hidden dimension for SAGE layers
    dropout     : dropout rate
    lr          : AdamW learning rate
    weight_decay: AdamW weight decay
    epochs      : max training epochs
    patience    : early stopping patience (on val log-loss, lower = better)
    batch_size  : mini-batch seed size during training
    infer_batch : mini-batch seed size during inference
    fanouts     : neighbourhood fanouts per SAGE hop
    device      : torch device
    use_amp     : mixed precision
    """
    def __init__(
        self,
        hidden       = 128,
        dropout      = 0.20,
        lr           = 1e-3,
        weight_decay = 3e-4,
        epochs       = EPOCHS,
        patience     = PATIENCE,
        batch_size   = BATCH_SIZE,
        infer_batch  = INFER_BATCH,
        fanouts      = FANOUTS,
        device       = DEVICE,
        use_amp      = USE_AMP,
    ):
        self.hidden       = hidden
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.patience     = patience
        self.batch_size   = batch_size
        self.infer_batch  = infer_batch
        self.fanouts      = fanouts
        self.device       = device
        self.use_amp      = use_amp

        self.model_        = None
        self.cardinalities_= None

    def fit(
        self,
        train_idx,
        val_idx,
        y_all,
        neighbors,
        x_num_cpu,
        x_cat_cpu,
        y_cpu,
        cardinalities,
        class_weights,
    ):
        """
        Train on train_idx nodes, validate on val_idx nodes.
        All node features live in the shared x_num_cpu / x_cat_cpu tensors (train+test).

        Parameters
        ----------
        train_idx     : np.ndarray of global node indices for this fold's train split
        val_idx       : np.ndarray of global node indices for this fold's val split
        y_all         : np.ndarray, integer labels for all training nodes (-1 for test)
        neighbors     : KNN neighbors array [n_total, K]
        x_num_cpu     : pinned CPU tensor [n_total, n_num_features]
        x_cat_cpu     : pinned CPU tensor [n_total, n_cat_features]
        y_cpu         : pinned CPU tensor [n_total] (float, -1 for test)
        cardinalities : list of ints, cardinality per categorical feature
        class_weights : torch.Tensor of shape [N_CLASSES], balanced class weights
        """
        global _global_pos
        n_all = x_num_cpu.shape[0]
        _global_pos = np.full(n_all, -1, dtype=np.int32)

        self.cardinalities_ = cardinalities

        # Build model; wrap with DataParallel if multiple GPUs available
        core_model = IrrigationGNN(
            num_in       = x_num_cpu.shape[1],
            cardinalities= cardinalities,
            hidden       = self.hidden,
            dropout      = self.dropout,
        )
        if N_GPUS > 1:
            core_model = nn.DataParallel(core_model)
        self.model_ = core_model.to(self.device)

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )
        opt     = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # torch.amp.GradScaler added in 2.3; use torch.cuda.amp.GradScaler for 2.2 compat
        scaler  = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))

        best_val_loss = float("inf")
        best_state    = None
        bad_epochs    = 0

        print(f"    Model parameters : {sum(p.numel() for p in self.model_.parameters()):,}")
        print(f"    Training nodes   : {len(train_idx):,}  |  Validation nodes: {len(val_idx):,}")
        print(f"    Max epochs: {self.epochs}  |  Patience: {self.patience}\n")

        for epoch in range(1, self.epochs + 1):
            # ---- train ----
            self.model_.train()
            epoch_losses = []
            offset = epoch % K

            for batch_seeds in _seed_batches(train_idx, self.batch_size, shuffle=True):
                batch = _build_subgraph(
                    batch_seeds, neighbors, x_num_cpu, x_cat_cpu,
                    y_cpu, self.fanouts, self.device, offset
                )
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(self.use_amp and self.device.type == "cuda")):
                    logits = self.model_(batch)
                    loss   = loss_fn(logits[batch.seed_local], batch.y[batch.seed_local].long())
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                epoch_losses.append(loss.item())
                del batch, logits, loss

            # ---- validate ----
            val_probs = _predict_nodes(
                self.model_, val_idx, neighbors, x_num_cpu, x_cat_cpu,
                y_cpu, self.fanouts, self.infer_batch, self.device, offset
            )
            y_val_true = y_all[val_idx]
            val_loss   = log_loss(y_val_true, val_probs, labels=[0, 1, 2])
            val_bal_acc = balanced_accuracy_score(y_val_true, np.argmax(val_probs, axis=1))

            print(
                f"    Epoch {epoch:04d} | "
                f"Train Loss: {np.mean(epoch_losses):.5f} | "
                f"Val Log-Loss: {val_loss:.5f} | "
                f"Val Bal-Acc: {val_bal_acc:.5f}"
            )

            # ---- early stopping ----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs    = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"\n    [Early Stop] No improvement for {self.patience} epochs. Stopping at epoch {epoch}.")
                    break

        print(f"\n    [Best] Val Log-Loss: {best_val_loss:.5f}")
        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu):
        """Returns softmax probabilities of shape [n_seeds, N_CLASSES]."""
        return _predict_nodes(
            self.model_, seed_nodes, neighbors, x_num_cpu, x_cat_cpu,
            y_cpu, self.fanouts, self.infer_batch, self.device
        )

    def predict(self, seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu):
        """Returns integer class predictions."""
        probs = self.predict_proba(seed_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu)
        return np.argmax(probs, axis=1)


# ============================================================
# SECTION 7 — LOAD DATA & BUILD GRAPH
# ============================================================

# %%
print("\n" + "="*60)
print("Loading data...")
print("="*60)

train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"Raw train: {train_raw.shape} | Raw test: {test_raw.shape}")

train_pre, test_pre, y_train = preprocess(train_raw, test_raw)
train_fe,  test_fe           = engineer_features(train_pre, test_pre)

Xc_train, Xc_test, cat_cardinalities = encode_categoricals(train_fe, test_fe)
Xn_train, Xn_test                    = scale_numerics(train_fe, test_fe)

neighbors = build_knn_graph(train_fe, test_fe, k=K)

n_train = len(train_fe)
n_test  = len(test_fe)
n_all   = n_train + n_test

# Combine train + test node features into shared tensors
Xn_all = np.vstack([Xn_train, Xn_test])
Xc_all = np.vstack([Xc_train, Xc_test])

y_all_np = np.concatenate([y_train, np.full(n_test, -1, dtype=np.int64)])

x_num_cpu = torch.tensor(Xn_all, dtype=torch.float32).pin_memory()
x_cat_cpu = torch.tensor(Xc_all, dtype=torch.long).pin_memory()
y_cpu     = torch.tensor(y_all_np.astype(np.float32), dtype=torch.float32).pin_memory()

print(f"\n[Tensors] x_num: {tuple(x_num_cpu.shape)} | x_cat: {tuple(x_cat_cpu.shape)}")

# Class weights (balanced) from training labels
classes       = np.arange(N_CLASSES)
cw_values     = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = torch.tensor(cw_values, dtype=torch.float32)
print(f"\n[Class Weights] { {LABEL_INV[i]: round(float(w), 4) for i, w in enumerate(cw_values)} }")


# ============================================================
# SECTION 8 — CROSS-VALIDATION TRAINING
# ============================================================

# %%
print("\n" + "="*60)
print("Starting 5-Fold Cross-Validation")
print("="*60)

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs  = np.zeros((n_train, N_CLASSES), dtype=np.float32)
pred_probs = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
test_nodes = np.arange(n_train, n_all, dtype=np.int32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train), 1):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold} / {N_FOLDS}")
    print(f"{'='*60}\n")

    t_fold = time.time()

    clf = IrrigationGNNClassifier()
    clf.fit(
        train_idx    = tr_idx,
        val_idx      = va_idx,
        y_all        = y_all_np,
        neighbors    = neighbors,
        x_num_cpu    = x_num_cpu,
        x_cat_cpu    = x_cat_cpu,
        y_cpu        = y_cpu,
        cardinalities= cat_cardinalities,
        class_weights= class_weights,
    )

    # OOF predictions
    oof_probs[va_idx] = clf.predict_proba(
        va_idx, neighbors, x_num_cpu, x_cat_cpu, y_cpu
    )

    # Test predictions (averaged across folds)
    pred_probs += clf.predict_proba(
        test_nodes, neighbors, x_num_cpu, x_cat_cpu, y_cpu
    ) / N_FOLDS

    fold_bal_acc = balanced_accuracy_score(y_train[va_idx], np.argmax(oof_probs[va_idx], axis=1))
    fold_logloss = log_loss(y_train[va_idx], oof_probs[va_idx], labels=[0, 1, 2])

    print(f"\n  [Fold {fold} Result] Bal-Acc: {fold_bal_acc:.5f} | Log-Loss: {fold_logloss:.5f} | Time: {time.time()-t_fold:.1f}s")
    print(f"\n{'='*60}\n\n")

    del clf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SECTION 9 — OOF EVALUATION
# ============================================================

# %%
oof_preds    = np.argmax(oof_probs, axis=1)
cv_bal_acc   = balanced_accuracy_score(y_train, oof_preds)
cv_logloss   = log_loss(y_train, oof_probs, labels=[0, 1, 2])

print("="*60)
print("  FINAL OOF EVALUATION")
print("="*60)
print(f"  CV Balanced Accuracy : {cv_bal_acc:.5f}")
print(f"  CV Log-Loss          : {cv_logloss:.5f}")
print("="*60)


# ============================================================
# SECTION 10 — SUBMISSION
# ============================================================

# %%
print("\n[Submission] Generating predictions...")

test_preds      = np.argmax(pred_probs, axis=1)
test_labels     = [LABEL_INV[p] for p in test_preds]

submission = pd.DataFrame({
    "id"           : test_raw["id"],
    TARGET         : test_labels,
})
submission.to_csv(f"submission_gnn_{VERSION_NB}.csv", index=False)

np.save(f"oof_gnn_{VERSION_NB}.npy",  oof_probs.astype(np.float32))
np.save(f"pred_gnn_{VERSION_NB}.npy", pred_probs.astype(np.float32))

print(f"\n  Saved: submission_gnn_{VERSION_NB}.csv")
print(f"  Saved: oof_gnn_{VERSION_NB}.npy")
print(f"  Saved: pred_gnn_{VERSION_NB}.npy")
print(f"\n  Prediction distribution:")

for label, count in pd.Series(test_labels).value_counts().items():
    print(f"    {label}: {count:,}")


# ============================================================
# SECTION 11 — SAVE WITH OUR LABEL ENCODING
# ============================================================

# %%
# Reorder to our label encoding: High=0, Low=1, Medium=2
# GNN internal: Low=0, Medium=1, High=2
# Mapping: our[0]=High=their[2], our[1]=Low=their[0], our[2]=Medium=their[1]
oof_reordered = oof_probs[:, [2, 0, 1]]
pred_reordered = pred_probs[:, [2, 0, 1]]
np.save("oof_gnn.npy", oof_reordered.astype(np.float32))
np.save("pred_gnn.npy", pred_reordered.astype(np.float32))
print(f"Saved oof_gnn.npy {oof_reordered.shape} and pred_gnn.npy {pred_reordered.shape}")
print("Label order: High=0, Low=1, Medium=2")

print("\nDone.")
