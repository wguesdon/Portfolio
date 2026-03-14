#!/usr/bin/env python
"""
Publication-ready EDA plots for PS4E11 — Exploring Mental Health Data.
Generates all figures used in the Quarto report.
"""

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # force raster backend before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    after_stat,
    coord_flip,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_bar,
    geom_col,
    geom_errorbar,
    geom_hline,
    geom_jitter,
    geom_label,
    geom_point,
    geom_text,
    geom_tile,
    geom_violin,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_fill_gradient2,
    scale_x_continuous,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_minimal,
    coord_cartesian,
    geom_boxplot,
    geom_density,
    scale_fill_brewer,
    annotate,
    position_dodge,
    position_stack,
    position_fill,
    scale_y_discrete,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent  # Mental_Health_Depression/
DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette & theme ────────────────────────────────────────────────────────────
CLR_NO = "#5B8DB8"   # blue — no depression
CLR_YES = "#E05C6B"  # rose — depression
PALETTE = {0: CLR_NO, 1: CLR_YES}
PALETTE_STR = {"No Depression": CLR_NO, "Depression": CLR_YES}

FONT = "DejaVu Sans"

def pub_theme():
    return (
        theme_minimal()
        + theme(
            text=element_text(family=FONT, size=11),
            plot_title=element_text(size=14, face="bold", margin={"b": 10}),
            plot_subtitle=element_text(size=11, color="#555555", margin={"b": 12}),
            plot_caption=element_text(size=9, color="#888888", margin={"t": 8}),
            axis_title=element_text(size=11, face="bold"),
            axis_text=element_text(size=10),
            legend_title=element_text(size=10, face="bold"),
            legend_text=element_text(size=10),
            panel_grid_major=element_line(color="#E8E8E8", size=0.5),
            panel_grid_minor=element_blank(),
            strip_text=element_text(size=10, face="bold"),
            strip_background=element_rect(fill="#F0F4F8", color="white"),
        )
    )

DPI = 600   # high-quality retina — 600 DPI for maximum sharpness
W, H = 12, 7


def save(p, name, width=W, height=H):
    # Draw via matplotlib directly — bypasses plotnine's save, which can drop DPI
    fig = p.draw()
    fig.set_size_inches(width, height)
    fig.savefig(str(OUT_DIR / name), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Data loading & cleaning ────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")

    # ── Clean Sleep Duration → numeric hours ──────────────────────────────────
    SLEEP_MAP = {
        "Less than 5 hours": 4.0,
        "5-6 hours": 5.5,
        "6-7 hours": 6.5,
        "6-8 hours": 7.0,
        "7-8 hours": 7.5,
        "8 hours": 8.0,
        "8-9 hours": 8.5,
        "More than 8 hours": 9.0,
        "9-11 hours": 10.0,
    }

    def parse_sleep(s):
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        if s in SLEEP_MAP:
            return SLEEP_MAP[s]
        # parse "X-Y hours" → midpoint
        m = re.match(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*hours?$", s, re.I)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        # single numeric
        m = re.match(r"^(\d+(?:\.\d+)?)\s*hours?$", s, re.I)
        if m:
            return float(m.group(1))
        return np.nan

    df["Sleep_Hours"] = df["Sleep Duration"].apply(parse_sleep)

    # ── Clean Dietary Habits ──────────────────────────────────────────────────
    DIET_MAP = {"Healthy": "Healthy", "Moderate": "Moderate", "Unhealthy": "Unhealthy"}
    df["Diet"] = df["Dietary Habits"].map(DIET_MAP)

    # ── Readable labels ───────────────────────────────────────────────────────
    df["Status"] = df["Depression"].map({0: "No Depression", 1: "Depression"})
    df["Work_Status"] = df["Working Professional or Student"]
    df["Suicidal"] = df["Have you ever had suicidal thoughts ?"]
    df["Family_History"] = df["Family History of Mental Illness"]
    df["Financial_Stress"] = df["Financial Stress"]
    df["Work_Study_Hours"] = df["Work/Study Hours"]

    # Age groups
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[17, 24, 30, 39, 49, 60],
        labels=["18–24", "25–30", "31–39", "40–49", "50–60"],
    )

    return df


# ── Figure 1: Target distribution ─────────────────────────────────────────────
def fig_target(df):
    counts = df["Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Pct"] = counts["Count"] / counts["Count"].sum() * 100
    counts["Label"] = counts.apply(lambda r: f"{r['Count']:,}\n({r['Pct']:.1f}%)", axis=1)

    p = (
        ggplot(counts, aes("Status", "Count", fill="Status"))
        + geom_col(width=0.55, show_legend=False)
        + geom_text(aes(label="Label"), va="bottom", size=11, nudge_y=500)
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(
            labels=lambda lst: [f"{int(x/1000)}k" for x in lst],
            expand=(0, 0, 0.15, 0),
        )
        + labs(
            title="Target Distribution",
            subtitle="Dataset is imbalanced — only 18.2 % of respondents report depression",
            x="",
            y="Number of respondents",
            caption="PS4E11 · train.csv · n = 140,700",
        )
        + pub_theme()
    )
    save(p, "fig01_target_distribution.png", width=7, height=5)


# ── Figure 2: Depression rate by gender & work status ─────────────────────────
def fig_gender_workstatus(df):
    rate = (
        df.groupby(["Work_Status", "Gender"])["Depression"]
        .mean()
        .reset_index()
        .rename(columns={"Depression": "Rate"})
    )
    rate["Rate_pct"] = rate["Rate"] * 100
    rate["Label"] = rate["Rate_pct"].map(lambda x: f"{x:.1f}%")
    # Encode label y-position in data so nudge_y isn't needed alongside position_dodge
    rate["Label_y"] = rate["Rate_pct"] + 0.4

    p = (
        ggplot(rate, aes("Gender", "Rate_pct", fill="Work_Status"))
        + geom_col(position=position_dodge(width=0.7), width=0.6)
        + geom_text(
            aes(y="Label_y", label="Label"),
            position=position_dodge(width=0.7),
            va="bottom",
            size=9,
        )
        + scale_fill_manual(values={"Working Professional": "#4A90D9", "Student": "#E8A838"})
        + scale_y_continuous(
            labels=lambda lst: [f"{x:.0f}%" for x in lst],
            expand=(0, 0, 0.1, 0),
        )
        + labs(
            title="Depression Rate by Gender and Work Status",
            subtitle="Students show higher depression rates than working professionals across both genders",
            x="",
            y="Depression rate (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig02_gender_workstatus.png", width=8, height=5)


# ── Figure 3: Depression rate by age group ────────────────────────────────────
def fig_age(df):
    age_rate = (
        df.groupby(["Age_Group", "Status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    age_total = df.groupby("Age_Group", observed=True).size().reset_index(name="Total")
    age_rate = age_rate.merge(age_total, on="Age_Group")
    age_rate["Pct"] = age_rate["Count"] / age_rate["Total"] * 100

    p = (
        ggplot(age_rate, aes("Age_Group", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.6)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + annotate("text", x=4.9, y=19.5, label="Overall avg 18.2%",
                   size=9, color="#555555", ha="right")
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Prevalence by Age Group",
            subtitle="Younger adults (18–30) show disproportionately higher depression rates",
            x="Age group",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig03_age_groups.png", width=9, height=5)


# ── Figure 4: Sleep hours distribution ────────────────────────────────────────
def fig_sleep(df):
    sleep_df = df.dropna(subset=["Sleep_Hours"]).copy()
    # Bin to sensible hours
    bins = [0, 4, 5, 6, 7, 8, 9, 15]
    labels = ["<4h", "4–5h", "5–6h", "6–7h", "7–8h", "8–9h", ">9h"]
    sleep_df["Sleep_Bin"] = pd.cut(sleep_df["Sleep_Hours"], bins=bins, labels=labels)

    rate = (
        sleep_df.groupby(["Sleep_Bin", "Status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    total = sleep_df.groupby("Sleep_Bin", observed=True).size().reset_index(name="Total")
    rate = rate.merge(total, on="Sleep_Bin")
    rate["Pct"] = rate["Count"] / rate["Total"] * 100

    p = (
        ggplot(rate, aes("Sleep_Bin", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.65)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Rate by Sleep Duration",
            subtitle="Very short (<4 h) and very long (>9 h) sleep are linked to higher depression rates",
            x="Sleep duration",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv · noisy values excluded",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig04_sleep_duration.png", width=9, height=5)


# ── Figure 5: Financial stress ────────────────────────────────────────────────
def fig_financial_stress(df):
    fs = df.dropna(subset=["Financial_Stress"]).copy()
    fs["FS"] = fs["Financial_Stress"].astype(int).astype(str)

    rate = (
        fs.groupby(["FS", "Status"])
        .size()
        .reset_index(name="Count")
    )
    total = fs.groupby("FS").size().reset_index(name="Total")
    rate = rate.merge(total, on="FS")
    rate["Pct"] = rate["Count"] / rate["Total"] * 100

    p = (
        ggplot(rate, aes("FS", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.6)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + annotate("text", x="5", y=20.5, label="Overall avg",
                   size=9, color="#555555", ha="right")
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Rate by Financial Stress Level",
            subtitle="Strong monotonic relationship — highest stress (5) shows ~2.5× the depression rate of lowest (1)",
            x="Financial stress (1 = low, 5 = high)",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig05_financial_stress.png", width=9, height=5)


# ── Figure 6: Suicidal thoughts & Family History ──────────────────────────────
def fig_risk_factors(df):
    records = []
    for col, label in [("Suicidal", "Suicidal thoughts"), ("Family_History", "Family history of mental illness")]:
        grp = (
            df.groupby([col, "Status"])
            .size()
            .reset_index(name="Count")
        )
        total = df.groupby(col).size().reset_index(name="Total")
        grp = grp.merge(total, on=col)
        grp["Pct"] = grp["Count"] / grp["Total"] * 100
        grp["Factor"] = label
        grp = grp.rename(columns={col: "Response"})
        records.append(grp)

    combined = pd.concat(records, ignore_index=True)

    p = (
        ggplot(combined, aes("Response", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.55)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + facet_wrap("Factor", scales="free_x")
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Rate by Key Risk Factors",
            subtitle="Prior suicidal thoughts are the single strongest predictor; family history shows moderate elevation",
            x="",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig06_risk_factors.png", width=10, height=5)


# ── Figure 7: Top professions by depression rate ──────────────────────────────
def fig_professions(df):
    prof = df[df["Work_Status"] == "Working Professional"].copy()
    prof_rate = (
        prof.groupby("Profession")["Depression"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Rate", "count": "N"})
    )
    # Keep only professions with ≥ 500 respondents
    prof_rate = prof_rate[prof_rate["N"] >= 500].copy()
    prof_rate["Rate_pct"] = prof_rate["Rate"] * 100
    prof_rate = prof_rate.sort_values("Rate_pct", ascending=True).tail(20)

    p = (
        ggplot(prof_rate, aes("reorder(Profession, Rate_pct)", "Rate_pct"))
        + geom_col(aes(fill="Rate_pct"), width=0.7, show_legend=False)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + geom_text(aes(label="Rate_pct.round(1).astype(str) + '%'"),
                    ha="left", size=8.5, nudge_y=0.2)
        + coord_flip()
        + scale_fill_gradient2(
            low=CLR_NO, mid="#F5C842", high=CLR_YES,
            midpoint=18.2, limits=(0, 40),
        )
        + scale_y_continuous(
            labels=lambda lst: [f"{x:.0f}%" for x in lst],
            expand=(0, 0, 0.08, 0),
        )
        + labs(
            title="Depression Rate by Profession (Top 20)",
            subtitle="Dashed line = overall average (18.2%). Creative and high-pressure roles show elevated rates.",
            x="",
            y="Depression rate (%)",
            caption="PS4E11 · Working Professionals only · min. 500 respondents",
        )
        + pub_theme()
        + theme(panel_grid_major_y=element_blank())
    )
    save(p, "fig07_professions.png", width=10, height=8)


# ── Figure 8: Work/Study hours vs depression ──────────────────────────────────
def fig_work_hours(df):
    wh = df.copy()
    wh["Hours_Bin"] = pd.cut(
        wh["Work_Study_Hours"],
        bins=[-1, 2, 4, 6, 8, 10, 12],
        labels=["0–2", "2–4", "4–6", "6–8", "8–10", "10–12"],
    )
    rate = (
        wh.groupby(["Hours_Bin", "Work_Status", "Status"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    total = wh.groupby(["Hours_Bin", "Work_Status"], observed=True).size().reset_index(name="Total")
    rate = rate.merge(total, on=["Hours_Bin", "Work_Status"])
    rate["Pct"] = rate["Count"] / rate["Total"] * 100

    p = (
        ggplot(rate, aes("Hours_Bin", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.65)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + facet_wrap("Work_Status")
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Rate by Work / Study Hours",
            subtitle="Longer hours are associated with higher depression, especially for students",
            x="Hours per day",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig08_work_hours.png", width=11, height=5)


# ── Figure 9: Academic & Work Pressure ────────────────────────────────────────
def fig_pressure(df):
    records = []
    for group, col, label in [
        ("Student", "Academic Pressure", "Academic Pressure (Students)"),
        ("Working Professional", "Work Pressure", "Work Pressure (Professionals)"),
    ]:
        sub = df[df["Work_Status"] == group].dropna(subset=[col]).copy()
        sub["PressureVal"] = sub[col].astype(int).astype(str)
        rate = (
            sub.groupby(["PressureVal", "Status"])
            .size()
            .reset_index(name="Count")
        )
        total = sub.groupby("PressureVal").size().reset_index(name="Total")
        rate = rate.merge(total, on="PressureVal")
        rate["Pct"] = rate["Count"] / rate["Total"] * 100
        rate["Group"] = label
        records.append(rate)

    combined = pd.concat(records, ignore_index=True)

    p = (
        ggplot(combined, aes("PressureVal", "Pct", fill="Status"))
        + geom_col(position=position_stack(), width=0.6)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + facet_wrap("Group", scales="free_x")
        + scale_fill_manual(values=PALETTE_STR)
        + scale_y_continuous(labels=lambda lst: [f"{x:.0f}%" for x in lst])
        + labs(
            title="Depression Rate by Pressure Level",
            subtitle="Both academic and work pressure show a near-linear relationship with depression",
            x="Pressure score (1 = low, 5 = high)",
            y="Proportion (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top")
    )
    save(p, "fig09_pressure.png", width=11, height=5)


# ── Figure 10: Correlation heatmap (numeric features) ─────────────────────────
def fig_correlation(df):
    num_cols = [
        "Age", "Academic Pressure", "Work Pressure", "CGPA",
        "Study Satisfaction", "Job Satisfaction", "Sleep_Hours",
        "Work_Study_Hours", "Financial_Stress", "Depression",
    ]
    corr = df[num_cols].corr().round(2)

    # Melt for ggplot
    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="Variable", value_name="Correlation")
        .rename(columns={"index": "Feature"})
    )
    # Shorten labels
    short = {
        "Academic Pressure": "Acad. Pressure",
        "Work Pressure": "Work Pressure",
        "Study Satisfaction": "Study Satisf.",
        "Job Satisfaction": "Job Satisf.",
        "Sleep_Hours": "Sleep (h)",
        "Work_Study_Hours": "Work/Study h",
        "Financial_Stress": "Financial Stress",
        "Depression": "Depression",
    }
    corr_long["Feature"] = corr_long["Feature"].replace(short)
    corr_long["Variable"] = corr_long["Variable"].replace(short)
    corr_long["Label"] = corr_long["Correlation"].map(lambda x: f"{x:.2f}")

    order = list(reversed([short.get(c, c) for c in num_cols]))
    corr_long["Feature"] = pd.Categorical(corr_long["Feature"], categories=order)
    corr_long["Variable"] = pd.Categorical(
        corr_long["Variable"], categories=[short.get(c, c) for c in num_cols]
    )

    p = (
        ggplot(corr_long, aes("Variable", "Feature", fill="Correlation"))
        + geom_tile(color="white", size=0.3)
        + geom_text(aes(label="Label"), size=7.5, color="black")
        + scale_fill_gradient2(
            low="#3B82C4", mid="white", high="#E05C6B",
            midpoint=0, limits=(-1, 1),
            name="r",
        )
        + labs(
            title="Pearson Correlation Matrix — Numeric Features",
            subtitle="Financial stress and pressure scores are most correlated with depression",
            x="",
            y="",
            caption="PS4E11 · train.csv · pairwise Pearson r",
        )
        + pub_theme()
        + theme(
            axis_text_x=element_text(angle=35, ha="right"),
            panel_grid_major=element_blank(),
        )
    )
    save(p, "fig10_correlation_heatmap.png", width=10, height=8)


# ── Figure 11: Overall missingness overview ───────────────────────────────────
def fig_missing_overview(df):
    """Horizontal bar: % missing for every column that has any missing values."""
    raw_cols = [c for c in df.columns
                if c not in ("Status", "Work_Status", "Suicidal", "Family_History",
                             "Financial_Stress", "Work_Study_Hours", "Sleep_Hours",
                             "Diet", "Age_Group")]
    miss_pct = df[raw_cols].isnull().mean() * 100
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=True).reset_index()
    miss_pct.columns = ["Feature", "Missing_pct"]
    miss_pct["Label"] = miss_pct["Missing_pct"].map(lambda x: f"{x:.1f}%")
    miss_pct["Label_x"] = miss_pct["Missing_pct"] + 0.8

    # Colour by missingness magnitude
    miss_pct["Category"] = pd.cut(
        miss_pct["Missing_pct"],
        bins=[-1, 5, 30, 101],
        labels=["Low (<5%)", "Moderate (5–30%)", "High (>30%)"],
    )

    p = (
        ggplot(miss_pct, aes("reorder(Feature, Missing_pct)", "Missing_pct", fill="Category"))
        + geom_col(width=0.7)
        + geom_text(aes(x="reorder(Feature, Missing_pct)", y="Label_x", label="Label"),
                    ha="left", size=9)
        + coord_flip()
        + scale_fill_manual(
            values={"Low (<5%)": "#8BBFD4", "Moderate (5–30%)": "#E8A838", "High (>30%)": "#E05C6B"},
            name="",
        )
        + scale_y_continuous(
            labels=lambda lst: [f"{x:.0f}%" for x in lst],
            expand=(0, 0, 0.12, 0),
        )
        + labs(
            title="Missing Values — All Features",
            subtitle="Six features have >30% missing; missingness is structural, not random",
            x="",
            y="Missing (%)",
            caption="PS4E11 · train.csv · n = 140,700",
        )
        + pub_theme()
        + theme(legend_position="top", panel_grid_major_y=element_blank())
    )
    save(p, "fig11_missing_overview.png", width=10, height=6)


# ── Figure 12: Missingness pattern matrix ────────────────────────────────────
def fig_missing_matrix(df):
    """
    Presence/absence heatmap for the structurally missing columns.
    Rows = a stratified sample; columns = key features.
    Sorted by Work_Status so the two blocks appear clearly.
    """
    focus = [
        "Academic Pressure", "CGPA", "Study Satisfaction",
        "Work Pressure", "Job Satisfaction", "Profession",
    ]
    SAMPLE = 2000
    rng = np.random.default_rng(42)

    pros = df[df["Work_Status"] == "Working Professional"].sample(SAMPLE // 2, random_state=42)
    stus = df[df["Work_Status"] == "Student"].sample(SAMPLE // 2, random_state=42)
    sample = pd.concat([pros, stus]).reset_index(drop=True)
    sample["Row"] = range(len(sample))
    sample["Group"] = sample["Work_Status"]

    # Melt to long: 1 = present, 0 = missing
    melted = (
        sample[["Row", "Group"] + focus]
        .melt(id_vars=["Row", "Group"], var_name="Feature", value_name="Value")
    )
    melted["Present"] = melted["Value"].notna().map({True: "Present", False: "Missing"})

    # Feature order: student cols first, then professional cols
    feat_order = [
        "Academic Pressure", "CGPA", "Study Satisfaction",
        "Profession", "Work Pressure", "Job Satisfaction",
    ]
    melted["Feature"] = pd.Categorical(melted["Feature"], categories=feat_order)

    p = (
        ggplot(melted, aes("Feature", "Row", fill="Present"))
        + geom_tile()
        + scale_fill_manual(
            values={"Present": "#4A90D9", "Missing": "#F2D4D7"},
            name="",
        )
        + annotate("text", x=2, y=SAMPLE * 0.75, label="Students",
                   size=10, color="white", fontweight="bold")
        + annotate("text", x=2, y=SAMPLE * 0.25, label="Professionals",
                   size=10, color="white", fontweight="bold")
        + annotate("segment", x=0.5, xend=6.5, y=SAMPLE // 2, yend=SAMPLE // 2,
                   color="white", size=1.2, linetype="dashed")
        + scale_y_continuous(labels=lambda _: [], breaks=[])
        + labs(
            title="Missingness Pattern Matrix",
            subtitle="Blue = value present · Pink = missing. The two blocks reveal perfectly structured missingness",
            x="",
            y=f"Respondents (n = {SAMPLE:,} sample)",
            caption="PS4E11 · train.csv · stratified sample (1,000 per group)",
        )
        + pub_theme()
        + theme(
            panel_grid_major=element_blank(),
            axis_text_x=element_text(angle=25, ha="right"),
        )
    )
    save(p, "fig12_missing_matrix.png", width=10, height=7)


# ── Figure 13: Structural missingness by work status (% per group) ────────────
def fig_missing_by_group(df):
    """Dodge bar: % missing per feature, split by work status."""
    focus_cols = [
        "Academic Pressure", "CGPA", "Study Satisfaction",
        "Work Pressure", "Job Satisfaction", "Profession",
    ]
    miss = (
        df.groupby("Work_Status")[focus_cols]
        .apply(lambda g: g.isnull().mean() * 100)
        .reset_index()
        .melt(id_vars="Work_Status", var_name="Feature", value_name="Missing_pct")
    )
    miss["Label_y"] = miss["Missing_pct"] + 1.5

    p = (
        ggplot(miss, aes("Feature", "Missing_pct", fill="Work_Status"))
        + geom_col(position=position_dodge(width=0.7), width=0.6)
        + geom_text(
            aes(y="Label_y", label="Missing_pct.round(0).astype(int).astype(str) + '%'"),
            position=position_dodge(width=0.7),
            va="bottom", size=8.5,
        )
        + coord_flip()
        + scale_fill_manual(values={"Working Professional": "#4A90D9", "Student": "#E8A838"})
        + scale_y_continuous(
            labels=lambda lst: [f"{x:.0f}%" for x in lst],
            expand=(0, 0, 0.12, 0),
        )
        + labs(
            title="Structural Missing Data by Work Status",
            subtitle="Each group is missing exactly the other group's features — missingness is perfectly partitioned",
            x="",
            y="Missing (%)",
            fill="",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="top", panel_grid_major_y=element_blank())
    )
    save(p, "fig13_missing_by_group.png", width=10, height=5)


# ── Figure 14: Depression rate — missing vs present ───────────────────────────
def fig_missing_vs_target(df):
    """
    For each feature with structural missingness, compare the depression rate
    when the value is missing vs when it is present.
    Reveals whether missingness itself is predictive.
    """
    focus_cols = [
        "Academic Pressure", "CGPA", "Study Satisfaction",
        "Work Pressure", "Job Satisfaction", "Profession",
    ]
    records = []
    for col in focus_cols:
        for is_missing, label in [(True, "Missing"), (False, "Present")]:
            mask = df[col].isna() if is_missing else df[col].notna()
            sub = df[mask]
            if len(sub) == 0:
                continue
            dep_rate = sub["Depression"].mean() * 100
            n = len(sub)
            records.append({"Feature": col, "Missingness": label,
                            "Rate": dep_rate, "N": n})

    rate_df = pd.DataFrame(records)
    rate_df["Label"] = rate_df["Rate"].map(lambda x: f"{x:.1f}%")
    rate_df["Label_y"] = rate_df["Rate"] + 0.4

    p = (
        ggplot(rate_df, aes("Missingness", "Rate", fill="Missingness"))
        + geom_col(width=0.55)
        + geom_text(aes(y="Label_y", label="Label"), va="bottom", size=9)
        + geom_hline(yintercept=18.2, linetype="dashed", color="#555555", size=0.6)
        + facet_wrap("Feature", nrow=2)
        + scale_fill_manual(values={"Missing": "#F2A65A", "Present": "#4A90D9"}, name="")
        + scale_y_continuous(
            labels=lambda lst: [f"{x:.0f}%" for x in lst],
            expand=(0, 0, 0.15, 0),
        )
        + labs(
            title="Is Missingness Predictive of Depression?",
            subtitle="Dashed line = overall average (18.2%). Missing values align perfectly with one work-status group.",
            x="",
            y="Depression rate (%)",
            caption="PS4E11 · train.csv",
        )
        + pub_theme()
        + theme(legend_position="none", axis_text_x=element_text(size=10))
    )
    save(p, "fig14_missing_vs_target.png", width=12, height=6)


# ── Figure 15: Model & ensemble OOF comparison ────────────────────────────────
def fig_model_comparison():
    import json
    artifacts = ROOT / "artifacts"
    ensemble_dir = ROOT / "ensemble"

    records = []
    for model, label in [("lgb", "LightGBM"), ("xgb", "XGBoost"),
                          ("cat", "CatBoost"), ("autogluon", "AutoGluon")]:
        p = artifacts / model / f"{model}_results.json"
        if p.exists():
            r = json.loads(p.read_text())
            records.append({"Model": label, "OOF Accuracy": r["cv_accuracy_mean"],
                            "Type": "Base model"})

    ep = ensemble_dir / "ensemble_results.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        for method, info in e.get("all_results", {}).items():
            records.append({"Model": method.replace("_", " ").title(),
                            "OOF Accuracy": info["score"], "Type": "Ensemble"})
        # highlight best
        best = max(records, key=lambda x: x["OOF Accuracy"])
        for r in records:
            if r["Model"] == best["Model"]:
                r["Type"] = "Best"

    plot_df = pd.DataFrame(records).sort_values("OOF Accuracy")
    plot_df["Label"] = plot_df["OOF Accuracy"].map(lambda x: f"{x:.5f}")
    plot_df["Label_x"] = plot_df["OOF Accuracy"] + 0.0003

    p = (
        ggplot(plot_df, aes("reorder(Model, OOF_Accuracy)", "OOF_Accuracy", fill="Type"))
        + geom_col(width=0.65)
        + geom_text(aes(x="reorder(Model, OOF_Accuracy)", y="Label_x", label="Label"),
                    ha="left", size=9)
        + coord_flip()
        + scale_fill_manual(
            values={"Base model": "#5B8DB8", "Ensemble": "#7BC67E", "Best": "#E05C6B"},
            name="",
        )
        + scale_y_continuous(
            limits=(0.93, 0.975),
            labels=lambda lst: [f"{x:.3f}" for x in lst],
            expand=(0, 0, 0.06, 0),
        )
        + labs(
            title="Model & Ensemble OOF Accuracy (10-fold CV)",
            subtitle="Stacking all four models achieves the highest out-of-fold score",
            x="",
            y="OOF Accuracy",
            caption="PS4E11 · train.csv · higher is better",
        )
        + pub_theme()
        + theme(legend_position="top", panel_grid_major_y=element_blank())
    )
    # rename column for aes
    plot_df = plot_df.rename(columns={"OOF Accuracy": "OOF_Accuracy"})
    p = (
        ggplot(plot_df, aes("reorder(Model, OOF_Accuracy)", "OOF_Accuracy", fill="Type"))
        + geom_col(width=0.65)
        + geom_text(aes(x="reorder(Model, OOF_Accuracy)", y="Label_x", label="Label"),
                    ha="left", size=9)
        + coord_flip()
        + scale_fill_manual(
            values={"Base model": "#5B8DB8", "Ensemble": "#7BC67E", "Best": "#E05C6B"},
            name="",
        )
        + scale_y_continuous(
            limits=(0.93, 0.975),
            labels=lambda lst: [f"{x:.3f}" for x in lst],
            expand=(0, 0, 0.06, 0),
        )
        + labs(
            title="Model & Ensemble OOF Accuracy (10-fold CV)",
            subtitle="Stacking all four models achieves the highest out-of-fold accuracy",
            x="",
            y="OOF Accuracy",
            caption="PS4E11 · train.csv · higher is better",
        )
        + pub_theme()
        + theme(legend_position="top", panel_grid_major_y=element_blank())
    )
    save(p, "fig15_model_comparison.png", width=10, height=6)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  Train: {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("\nGenerating figures...")
    fig_target(df)
    fig_gender_workstatus(df)
    fig_age(df)
    fig_sleep(df)
    fig_financial_stress(df)
    fig_risk_factors(df)
    fig_professions(df)
    fig_work_hours(df)
    fig_pressure(df)
    fig_correlation(df)
    fig_missing_overview(df)
    fig_missing_matrix(df)
    fig_missing_by_group(df)
    fig_missing_vs_target(df)
    fig_model_comparison()

    print(f"\nAll figures saved to: {OUT_DIR}")
