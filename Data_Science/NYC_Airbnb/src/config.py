"""
Project-wide paths and constants.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "model"

DATA_FILE = DATA_DIR / "AB_NYC_2019.csv"

RANDOM_STATE = 42
N_FOLDS = 5

# Times Square — centre of Manhattan for distance feature
MANHATTAN_CENTER_LAT = 40.7580
MANHATTAN_CENTER_LON = -73.9855

# ── Plotting ──────────────────────────────────────────────────────────────────
DPI = 150

BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

BOROUGH_PALETTE = {
    "Manhattan":     "#E63946",
    "Brooklyn":      "#457B9D",
    "Queens":        "#2A9D8F",
    "Bronx":         "#E9C46A",
    "Staten Island": "#F4A261",
}

ROOM_PALETTE = {
    "Entire home/apt": "#2196F3",
    "Private room":    "#FF5722",
    "Shared room":     "#4CAF50",
}

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
