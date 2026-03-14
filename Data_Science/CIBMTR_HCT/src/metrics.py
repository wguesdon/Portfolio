"""
CIBMTR — Evaluation Metrics

Stratified Concordance Index: mean(C-index per race group) - std(C-index per race group)
"""
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index


def concordance_index_score(event_times, predicted_scores, event_observed):
    """
    C-index for survival time predictions.
    lifelines.concordance_index expects: higher predicted_scores → longer survival.
    Our models predict log(efs_time), so higher = longer survival → pass directly.
    """
    return float(concordance_index(
        event_times=event_times,
        predicted_scores=predicted_scores,
        event_observed=event_observed,
    ))


def stratified_concordance_index(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    event_col: str,
    pred_col: str,
) -> dict:
    """
    Compute stratified C-index across race groups.

    Score = mean(C-index per group) - std(C-index per group)

    Returns a dict with per-group and aggregate scores.
    """
    group_scores = {}
    for group, gdf in df.groupby(group_col, observed=True):
        if len(gdf) < 5:
            continue
        ci = concordance_index_score(
            event_times=gdf[time_col].values,
            predicted_scores=gdf[pred_col].values,
            event_observed=gdf[event_col].values,
        )
        group_scores[group] = ci

    scores = list(group_scores.values())
    macro_ci = float(np.mean(scores))
    std_ci   = float(np.std(scores))
    stratified_ci = macro_ci - std_ci

    # Overall (micro) C-index
    micro_ci = concordance_index_score(
        event_times=df[time_col].values,
        predicted_scores=df[pred_col].values,
        event_observed=df[event_col].values,
    )

    return {
        "stratified_ci": stratified_ci,
        "macro_ci": macro_ci,
        "std_ci": std_ci,
        "micro_ci": micro_ci,
        **{f"ci_{str(g).lower().replace(' ', '_').replace('/', '_')}": v
           for g, v in group_scores.items()},
    }
