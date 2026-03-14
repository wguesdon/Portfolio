"""Reusable Plotly figures for EDA and Streamlit app."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color-blind friendly palette
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#999999",
]


# ── Taxonomy ─────────────────────────────────────────────────────────────────

def phylum_bar(phylum_counts: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = phylum_counts.head(top_n)
    fig = px.bar(
        df, x="count", y="phylum", orientation="h",
        color="pct", color_continuous_scale="Blues",
        labels={"count": "Number of strains", "phylum": "Phylum", "pct": "% of total"},
        title=f"Top {top_n} Phyla by Strain Count",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def class_treemap(strains: pd.DataFrame, top_n: int = 30) -> go.Figure:
    top_classes = strains["class"].value_counts().head(top_n).index
    df = strains[strains["class"].isin(top_classes)].copy()
    fig = px.treemap(
        df, path=["phylum", "class"],
        title=f"Taxonomic Hierarchy (top {top_n} classes)",
        color_discrete_sequence=PALETTE,
    )
    return fig


def o2_pie(o2_counts: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        o2_counts, names="o2_tol", values="count",
        title="O₂ Tolerance Distribution",
        color_discrete_sequence=PALETTE,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


# ── Growth parameters ─────────────────────────────────────────────────────────

def temp_violin(strains: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        strains, y="temp_opt", x="o2_tol", color="o2_tol",
        box=True, points="outliers",
        color_discrete_sequence=PALETTE,
        labels={"temp_opt": "Optimal Temperature (°C)", "o2_tol": "O₂ Tolerance"},
        title="Optimal Temperature by O₂ Tolerance",
    )
    fig.update_layout(showlegend=False)
    return fig


def ph_violin(strains: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        strains, y="ph_opt", x="o2_tol", color="o2_tol",
        box=True, points="outliers",
        color_discrete_sequence=PALETTE,
        labels={"ph_opt": "Optimal pH", "o2_tol": "O₂ Tolerance"},
        title="Optimal pH by O₂ Tolerance",
    )
    fig.update_layout(showlegend=False)
    return fig


def temp_ph_scatter(strains: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        strains, x="temp_opt", y="ph_opt", color="o2_tol",
        facet_col="o2_tol", facet_col_wrap=3,
        opacity=0.5, size_max=6,
        color_discrete_sequence=PALETTE,
        labels={"temp_opt": "Optimal Temp (°C)", "ph_opt": "Optimal pH"},
        title="Temperature vs pH Optima by O₂ Tolerance",
    )
    return fig


def growth_range_box(strains: pd.DataFrame, top_phyla_n: int = 10) -> go.Figure:
    top = strains["phylum"].value_counts().head(top_phyla_n).index
    df = strains[strains["phylum"].isin(top)].copy()
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    fig = px.box(
        df, x="phylum", y="temp_range", color="phylum",
        color_discrete_sequence=PALETTE,
        labels={"temp_range": "Temperature Range (°C)", "phylum": "Phylum"},
        title=f"Temperature Range Distribution (top {top_phyla_n} phyla)",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Growth Parameter Correlation Matrix", height=500)
    return fig


# ── Media & compounds ─────────────────────────────────────────────────────────

def compound_freq_bar(compound_freq: pd.DataFrame, top_n: int = 25) -> go.Figure:
    df = compound_freq.head(top_n)
    fig = px.bar(
        df, x="media_count", y="compound_name", orientation="h",
        color="media_count", color_continuous_scale="Teal",
        labels={"media_count": "Number of media", "compound_name": "Compound"},
        title=f"Top {top_n} Most Common Compounds Across Media",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def media_richness_hist(media_richness: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        media_richness, x="compound_count", nbins=50,
        color_discrete_sequence=[PALETTE[0]],
        labels={"compound_count": "Compounds per Medium"},
        title="Distribution of Media Compound Richness",
    )
    fig.update_layout(bargap=0.05)
    return fig


def media_per_strain_hist(media_per_strain: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        media_per_strain, x="media_count", nbins=40,
        color_discrete_sequence=[PALETTE[2]],
        labels={"media_count": "Media per Strain"},
        title="Distribution of Media per Strain",
    )
    fig.update_layout(bargap=0.05)
    return fig


# ── Phylum × O₂ heatmap ───────────────────────────────────────────────────────

def phylum_o2_heatmap(phylum_o2: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = (
        phylum_o2.groupby("phylum")["count"].sum()
        .nlargest(top_n).index
    )
    df = phylum_o2[phylum_o2["phylum"].isin(top)]
    pivot = df.pivot(index="phylum", columns="o2_tol", values="count").fillna(0)
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto",
        color_continuous_scale="Blues",
        title=f"Strain Count by Phylum × O₂ Tolerance (top {top_n})",
        labels={"x": "O₂ Tolerance", "y": "Phylum", "color": "Strains"},
    )
    return fig
