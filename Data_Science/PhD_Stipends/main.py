import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

DATA_DIR = "Data"
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_US = "#CC6666"
COLOR_NON_US = "#9999CC"


# ---------------------------------------------------------------------------
# Load & clean data
# ---------------------------------------------------------------------------

df = pd.read_csv(f"{DATA_DIR}/phd-stipends.csv")

df.rename(columns={
    "Overall Pay": "Overall_Pay",
    "LW Ratio": "LW_Ratio",
    "Academic Year": "Academic_Year",
    "Program Year": "Program_Year",
    "12 M Gross Pay": "Gross_Pay_12M",
    "9 M Gross Pay": "Gross_Pay_9M",
    "3 M Gross Pay": "Gross_Pay_3M",
}, inplace=True)

# Drop high-missing columns
df.drop(columns=["Fees", "Gross_Pay_9M", "Comments", "Gross_Pay_3M"], errors="ignore", inplace=True)

# Convert salary columns: strip punctuation then cast to float
for col in ("Overall_Pay", "Gross_Pay_12M"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")

# Normalise Program_Year to integer
program_year_map = {"1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th and up": "6"}
df["Program_Year"] = df["Program_Year"].replace(program_year_map)
df["Seniority_Year"] = pd.to_numeric(df["Program_Year"], errors="coerce")

# Academic_Year: keep only the start year, parse to datetime
df["Academic_Year"] = df["Academic_Year"].astype(str).str.replace(r"-\d{4}$", "", regex=True)
df["Academic_Year"] = pd.to_datetime(df["Academic_Year"], format="%Y", errors="coerce")

# Join university location (sheet index 2)
univ_locations = pd.read_excel(f"{DATA_DIR}/univ_location.xlsx", sheet_name=2)
df = df.merge(univ_locations, on="University", how="inner")

# Department name normalisation
dept_patterns = [
    (r".*bio.*|.*cell.*|.*molecular.*|.*neuro.*|.*genetics.*", "biology"),
    (r".*aero.*|.*space.*", "aerospace"),
    (r".*health.*|.*medical.*|.*clinical.*|.*pharmacy.*", "health"),
    (r".*chemical.*|.*chemistry.*", "chemistry"),
    (r".*comput.*|^computational.*", "computer science"),
    (r".*psych.*", "psychology"),
    (r".*communication.*", "communication"),
    (r".*physics.*", "physics"),
    (r".*agri.*|.*crop.*|.*plant.*", "agriculture"),
    (r".*engineering.*|.*mechanical.*", "engineering"),
    (r".*business.*", "business"),
    (r".*math.*", "mathematics"),
    (r".*crim.*", "criminology"),
    (r".*earth.*|.*geo.*", "geoscience"),
    (r".*english.*", "english"),
]

df["department_cleaned"] = df["Department"].str.lower()
df = df[df["department_cleaned"].notna() & (df["department_cleaned"] != "n/a")]

for pattern, replacement in dept_patterns:
    df["department_cleaned"] = df["department_cleaned"].str.replace(pattern, replacement, regex=True)

top_dept = (
    df.groupby("department_cleaned")
    .size()
    .reset_index(name="n")
    .query("n >= 50")["department_cleaned"]
    .tolist()
)

# Filter observations
df = df[
    df["Overall_Pay"].notna()
    & (df["Overall_Pay"] <= 50000)
    & df["LW_Ratio"].notna()
    & df["Program_Year"].notna()
]


# ---------------------------------------------------------------------------
# Figure 1: Overall Pay & LW Ratio by location (pay > $15k)
# ---------------------------------------------------------------------------

df_high = df[df["Overall_Pay"] > 15000]
palette = {"us": COLOR_US, "non_us": COLOR_NON_US}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(x="location", y="Overall_Pay", hue="location", data=df_high, palette=palette, legend=False, ax=axes[0])
axes[0].set_title("Overall Pay by location")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

sns.boxplot(x="location", y="LW_Ratio", hue="location", data=df_high, palette=palette, legend=False, ax=axes[1])
axes[1].set_title("Living Wage Ratio by location")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_pay_by_location.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/fig1_pay_by_location.png")


# ---------------------------------------------------------------------------
# Figure 2: Count of students below/above $15k by location
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, threshold, label in [
    (axes[0], "below", "Students under $15K"),
    (axes[1], "above", "Students over $15K"),
]:
    subset = df[df["Overall_Pay"] <= 15000] if threshold == "below" else df[df["Overall_Pay"] > 15000]
    counts = subset["location"].value_counts().reindex(["us", "non_us"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[COLOR_US, COLOR_NON_US], width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val), ha="center")
    ax.set_title(label)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_counts_by_location.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/fig2_counts_by_location.png")


# ---------------------------------------------------------------------------
# Figure 3: Pay and LW Ratio by program year
# ---------------------------------------------------------------------------

import numpy as np

df_high = df[df["Overall_Pay"] > 15000].copy()
df_high["log_Overall_Pay"] = np.log10(df_high["Overall_Pay"])

year_order = [str(i) for i in range(1, 7)]
year_colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]
year_palette = {y: c for y, c in zip(year_order, year_colors)}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

counts = df_high["Program_Year"].value_counts().reindex(year_order, fill_value=0)
axes[0].bar(counts.index, counts.values, color=[year_palette.get(y, "#999") for y in counts.index])
axes[0].set_title("Respondents per program year")

sns.boxplot(x="Program_Year", y="log_Overall_Pay", hue="Program_Year", data=df_high, order=year_order,
            palette=year_palette, legend=False, ax=axes[1])
axes[1].set_title("log10(Overall Pay) by program year")

sns.boxplot(x="Program_Year", y="LW_Ratio", hue="Program_Year", data=df_high, order=year_order,
            palette=year_palette, legend=False, ax=axes[2])
axes[2].set_title("LW Ratio by program year")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_pay_by_seniority.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/fig3_pay_by_seniority.png")


# ---------------------------------------------------------------------------
# Figure 4: Mean pay & LW ratio over time by location
# ---------------------------------------------------------------------------

df_time = (
    df[df["Overall_Pay"] > 15000]
    .dropna(subset=["Academic_Year"])
)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for loc, color in [("us", COLOR_US), ("non_us", COLOR_NON_US)]:
    subset = df_time[df_time["location"] == loc].groupby("Academic_Year")
    mean_pay = subset["Overall_Pay"].mean()
    mean_lw = subset["LW_Ratio"].mean()
    axes[0].plot(mean_pay.index, mean_pay.values, label=loc, color=color, linewidth=2)
    axes[1].plot(mean_lw.index, mean_lw.values, label=loc, color=color, linewidth=2)

for ax in axes:
    ax.axvline(pd.Timestamp("2013-01-01"), linestyle="dotted", color="black", linewidth=0.8)
    ax.axvline(pd.Timestamp("2015-01-01"), linestyle="dotted", color="black", linewidth=0.8)
    ax.legend()

axes[0].set_title("Mean Overall Pay over time")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].set_title("Mean LW Ratio over time")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_pay_over_time.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/fig4_pay_over_time.png")


# ---------------------------------------------------------------------------
# Figure 5: Mean pay & LW ratio by department (US vs non-US)
# ---------------------------------------------------------------------------

df_dept = df[(df["department_cleaned"].isin(top_dept)) & (df["Overall_Pay"] > 15000)]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for row, (loc, color, title_suffix) in enumerate([
    ("us", COLOR_US, "US universities"),
    ("non_us", COLOR_NON_US, "non-US universities"),
]):
    subset = df_dept[df_dept["location"] == loc]

    pay_by_dept = (
        subset.groupby("department_cleaned")["Overall_Pay"].mean()
        .sort_values()
    )
    lw_by_dept = (
        subset.groupby("department_cleaned")["LW_Ratio"].mean()
        .sort_values()
    )

    axes[row, 0].barh(pay_by_dept.index, pay_by_dept.values, color=color)
    axes[row, 0].set_title(f"Mean Overall Pay — {title_suffix}")
    axes[row, 0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[row, 1].barh(lw_by_dept.index, lw_by_dept.values, color=color)
    axes[row, 1].set_title(f"Mean LW Ratio — {title_suffix}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_pay_by_department.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/fig5_pay_by_department.png")
