"""
Feature engineering for Mental Health Depression Prediction (PS4E11).

Binary classification: predict Depression (0=no, 1=yes).

Key data characteristics:
- 140K train / 93K test samples
- Structural missing values: professionals lack academic cols, students lack work cols
- Noisy synthetic values in Sleep Duration and Dietary Habits
- 26% missing Profession (students don't have one)
- Class imbalance: ~82% non-depressed, 18% depressed
"""
import re

import numpy as np
import pandas as pd


# Columns returned after feature engineering
NUMERICAL_COLS = [
    "Age",
    "Sleep_Hours",         # cleaned from noisy "Sleep Duration" strings
    "Work_Pressure",       # 0 for students (N/A)
    "Academic_Pressure",   # 0 for working professionals (N/A)
    "CGPA",                # 0 for working professionals (N/A)
    "Study_Satisfaction",  # 0 for working professionals (N/A)
    "Job_Satisfaction",    # 0 for students (N/A)
    "Work_Study_Hours",    # renamed from "Work/Study Hours"
    "Financial_Stress",
]

BINARY_COLS = [
    "is_student",
    "Gender_Male",
    "Suicidal_Thoughts",
    "Family_History",
    "Dietary_Score",       # ordinal: 0=unhealthy, 1=moderate, 2=healthy
]

CATEGORICAL_COLS = [
    "Profession",  # model-specific encoding (OHE / MEstimate / native)
    "Degree",
    "City",
]

FEATURE_COLUMNS = NUMERICAL_COLS + BINARY_COLS + CATEGORICAL_COLS

# Dietary habits mapping (valid values only; garbage → NaN → fillna(1))
_DIET_MAP = {
    "healthy": 2,
    "moderate": 1,
    "unhealthy": 0,
}


def _parse_sleep_hours(val: str) -> float:
    """
    Extract numeric sleep hours from noisy Sleep Duration values.

    Valid examples: "7-8 hours", "More than 8 hours", "Less than 5 hours", "8 hours"
    Garbage examples: "Indore", "No", "Moderate", "45" (> 16 h discarded)
    Returns np.nan for unrecognizable values.
    """
    if not isinstance(val, str):
        return np.nan
    s = val.strip().lower()
    # Remove unit words
    s = re.sub(r'\s*hours?\s*', '', s).strip()

    if "more than 8" in s or s == ">8":
        return 9.0
    if "less than 5" in s or "than 5" in s:
        return 4.0

    # "X-Y" range → midpoint
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0 < lo < 16 and 0 < hi < 16:
            return (lo + hi) / 2.0

    # Single number
    m = re.fullmatch(r'(\d+(?:\.\d+)?)', s)
    if m:
        v = float(m.group(1))
        if 0 < v < 16:
            return v

    return np.nan


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a train or test DataFrame.

    All transformations are self-contained (no fit/state) so they can be
    applied identically to train and test inside the SageMaker container.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Student / Professional flag                                         #
    # ------------------------------------------------------------------ #
    df["is_student"] = (df["Working Professional or Student"] == "Student").astype(int)

    # ------------------------------------------------------------------ #
    # Binary encodings                                                   #
    # ------------------------------------------------------------------ #
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    df["Suicidal_Thoughts"] = (
        df["Have you ever had suicidal thoughts ?"] == "Yes"
    ).astype(int)
    df["Family_History"] = (
        df["Family History of Mental Illness"] == "Yes"
    ).astype(int)

    # ------------------------------------------------------------------ #
    # Dietary habits: clean noisy values → ordinal score                 #
    # ------------------------------------------------------------------ #
    df["Dietary_Score"] = (
        df["Dietary Habits"].str.strip().str.lower().map(_DIET_MAP)
    )
    df["Dietary_Score"] = df["Dietary_Score"].fillna(1).astype(int)

    # ------------------------------------------------------------------ #
    # Sleep Duration: parse and clean                                     #
    # ------------------------------------------------------------------ #
    df["Sleep_Hours"] = df["Sleep Duration"].apply(_parse_sleep_hours)
    sleep_median = df["Sleep_Hours"].median()
    if np.isnan(sleep_median):
        sleep_median = 7.0
    df["Sleep_Hours"] = df["Sleep_Hours"].fillna(sleep_median)

    # ------------------------------------------------------------------ #
    # Structural missing values                                           #
    # Students: Work Pressure, Job Satisfaction → 0 (N/A)               #
    # Professionals: Academic Pressure, CGPA, Study Satisfaction → 0     #
    # ------------------------------------------------------------------ #
    df["Work_Pressure"] = df["Work Pressure"].fillna(0)
    df["Job_Satisfaction"] = df["Job Satisfaction"].fillna(0)
    df["Academic_Pressure"] = df["Academic Pressure"].fillna(0)
    df["Study_Satisfaction"] = df["Study Satisfaction"].fillna(0)
    df["CGPA"] = df["CGPA"].fillna(0)

    # ------------------------------------------------------------------ #
    # Rename for column-name safety (no spaces or special chars)         #
    # ------------------------------------------------------------------ #
    df["Work_Study_Hours"] = df["Work/Study Hours"]
    df["Financial_Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())

    # ------------------------------------------------------------------ #
    # Profession: students have no profession — fill with "Student"      #
    # Working professionals with missing Profession → "Unknown"          #
    # ------------------------------------------------------------------ #
    df["Profession"] = df["Profession"].copy()
    student_mask = df["Working Professional or Student"] == "Student"
    df.loc[student_mask & df["Profession"].isna(), "Profession"] = "Student"
    df["Profession"] = df["Profession"].fillna("Unknown")

    df["Degree"] = df["Degree"].fillna("Unknown")
    df["City"] = df["City"].fillna("Unknown")

    return df


def get_feature_columns() -> list[str]:
    """Return columns to use for model training (no id, no target)."""
    return FEATURE_COLUMNS
