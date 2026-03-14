"""
Feature engineering for Obesity Risk Prediction (PS4E2).

BMI is near-deterministic for the target class in this dataset.
All other features add marginal but meaningful signal.
"""
import numpy as np
import pandas as pd


# Ordinal mappings for categorical variables
CAEC_ORDER = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
CALC_ORDER = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MTRANS_ORDER = {
    "Walking": 0,
    "Bike": 1,
    "Public_Transportation": 2,
    "Motorbike": 3,
    "Automobile": 4,
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a train or test DataFrame.

    Args:
        df: Raw DataFrame with original competition columns.

    Returns:
        DataFrame with all original and engineered columns.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Core anthropometric feature: BMI                                     #
    # This is the strongest predictor — near-deterministic for class       #
    # ------------------------------------------------------------------ #
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # ------------------------------------------------------------------ #
    # Age groups                                                           #
    # ------------------------------------------------------------------ #
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 20, 30, 45, 200],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(int)

    # ------------------------------------------------------------------ #
    # Activity and lifestyle scores                                        #
    # ------------------------------------------------------------------ #
    # Activity score: exercise frequency minus screen time (net active time)
    df["Activity_Score"] = df["FAF"] - df["TUE"]

    # Water intake category (CH2O: 1=<1L, 2=1-2L, 3=>2L)
    df["Water_Category"] = pd.cut(
        df["CH2O"],
        bins=[0, 1.5, 2.5, 10],
        labels=[0, 1, 2],
        right=True,
    ).astype(int)

    # Meal frequency normalized
    df["Meal_Freq"] = df["NCP"].round().astype(int).clip(1, 4)

    # ------------------------------------------------------------------ #
    # Binary encodings                                                     #
    # ------------------------------------------------------------------ #
    df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    df["SMOKE_bin"] = (df["SMOKE"] == "yes").astype(int)
    df["SCC_bin"] = (df["SCC"] == "yes").astype(int)
    df["FAVC_bin"] = (df["FAVC"] == "yes").astype(int)
    df["family_hist_bin"] = (df["family_history_with_overweight"] == "yes").astype(int)

    # ------------------------------------------------------------------ #
    # Ordinal encodings for categorical variables                          #
    # ------------------------------------------------------------------ #
    df["CAEC_ord"] = df["CAEC"].map(CAEC_ORDER).fillna(0).astype(int)
    df["CALC_ord"] = df["CALC"].map(CALC_ORDER).fillna(0).astype(int)
    df["MTRANS_ord"] = df["MTRANS"].map(MTRANS_ORDER).fillna(2).astype(int)

    # ------------------------------------------------------------------ #
    # Interaction features                                                 #
    # ------------------------------------------------------------------ #
    # Physical activity modifying BMI effect
    df["BMI_x_FAF"] = df["BMI"] * df["FAF"]

    # Age and BMI interaction (older + higher BMI = higher risk)
    df["Age_x_BMI"] = df["Age"] * df["BMI"]

    # Weight-to-height ratio (alternative to BMI, correlated but distinct)
    df["Weight_Height_Ratio"] = df["Weight"] / df["Height"]

    # ------------------------------------------------------------------ #
    # BMI-derived bins (explicit class boundaries based on WHO)           #
    # ------------------------------------------------------------------ #
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25.0, 27.5, 30.0, 35.0, 40.0, 200],
        labels=[0, 1, 2, 3, 4, 5, 6],
        right=True,
    ).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """
    Return the list of feature columns to use for training.

    Returns:
        List of column names (excludes id, target, and raw categoricals
        that have been encoded).
    """
    return [
        # Raw numeric features
        "Age",
        "Height",
        "Weight",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
        # Engineered features
        "BMI",
        "BMI_Category",
        "Age_Group",
        "Activity_Score",
        "Water_Category",
        "Meal_Freq",
        "Weight_Height_Ratio",
        "BMI_x_FAF",
        "Age_x_BMI",
        # Binary encodings
        "Gender_Male",
        "SMOKE_bin",
        "SCC_bin",
        "FAVC_bin",
        "family_hist_bin",
        # Ordinal encodings
        "CAEC_ord",
        "CALC_ord",
        "MTRANS_ord",
    ]
