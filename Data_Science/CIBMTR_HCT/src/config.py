"""
CIBMTR — Equity in post-HCT Survival Predictions
Central configuration: paths, constants, colour palettes.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
EDA_DIR    = OUTPUT_DIR / "eda"
MODEL_DIR  = OUTPUT_DIR / "model"

TRAIN_FILE  = DATA_DIR / "train.csv"
TEST_FILE   = DATA_DIR / "test.csv"
SUBMIT_FILE = MODEL_DIR / "submission.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5
DPI          = 150

# ── Target columns ────────────────────────────────────────────────────────────
EFS_COL      = "efs"        # binary: 1=event, 0=censored
TIME_COL     = "efs_time"   # continuous: months
GROUP_COL    = "race_group" # stratification column
ID_COL       = "ID"

# ── Colour palettes ───────────────────────────────────────────────────────────
RACE_PALETTE = {
    "White":                                    "#2196F3",
    "Black or African-American":                "#E63946",
    "Asian":                                    "#2A9D8F",
    "Hispanic or Latino":                       "#F4A261",
    "Native Hawaiian or other Pacific Islander":"#9B5DE5",
    "American Indian or Alaska Native":         "#F15BB5",
    "More than one race":                       "#8B8B8B",
}

PALETTE = ["#2196F3", "#E63946", "#2A9D8F", "#F4A261", "#9B5DE5", "#F15BB5", "#8B8B8B"]

MODEL_COLORS = {
    "XGBoost":  "#F4A261",
    "LightGBM": "#2A9D8F",
    "CatBoost": "#9B5DE5",
    "Ensemble": "#E63946",
}

# ── Categorical features ──────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "dri_score", "psych_disturb", "cyto_score", "diabetes", "tbi_status",
    "arrhythmia", "graft_type", "vent_hist", "renal_issue", "pulm_severe",
    "prim_disease_hct", "cmv_status", "tce_imm_match", "rituximab",
    "prod_type", "cyto_score_detail", "conditioning_intensity", "ethnicity",
    "obesity", "mrd_hct", "in_vivo_tcd", "tce_match", "hepatic_severe",
    "prior_tumor", "peptic_ulcer", "gvhd_proph", "rheum_issue", "sex_match",
    "hepatic_mild", "tce_div_match", "donor_related", "melphalan_dose",
    "cardiac", "pulm_moderate", "race_group",
]

NUMERICAL_COLS = [
    "hla_match_c_high", "hla_high_res_8", "hla_low_res_6", "hla_high_res_6",
    "hla_high_res_10", "hla_match_dqb1_high", "hla_nmdp_6", "hla_match_c_low",
    "hla_match_drb1_low", "hla_match_dqb1_low", "hla_match_a_high", "donor_age",
    "hla_match_b_low", "age_at_hct", "hla_match_a_low", "hla_match_b_high",
    "comorbidity_score", "karnofsky_score", "hla_low_res_8", "hla_match_drb1_high",
    "hla_low_res_10", "year_hct",
]
