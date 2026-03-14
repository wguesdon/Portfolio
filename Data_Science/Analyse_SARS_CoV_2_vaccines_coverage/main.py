import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = "Data"
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color blind friendly palette
cbPalette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
sns.set_palette(cbPalette)


# --- Load data ---

population = pd.read_csv(f"{DATA_DIR}/population_by_country_2020.csv")
population = population[["Country (or dependency)", "Population (2020)", "Med. Age", "Urban Pop %", "World Share"]].copy()
population.rename(columns={
    "Country (or dependency)": "Country",
    "Population (2020)": "Population",
    "Med. Age": "Median_age",
    "Urban Pop %": "Urban_pop_percentage",
}, inplace=True)

vaccine = pd.read_csv(f"{DATA_DIR}/country_vaccinations.csv")
vaccine = vaccine[["country", "date", "total_vaccinations", "people_fully_vaccinated", "vaccines"]]
vaccine.rename(columns={"country": "Country"}, inplace=True)
vaccine = vaccine.groupby("Country")["people_fully_vaccinated"].max()

vaccine_manufacturer = pd.read_csv(f"{DATA_DIR}/country_vaccinations_by_manufacturer.csv")
vaccine_manufacturer.rename(columns={"location": "Country"}, inplace=True)
vaccine_manufacturer = vaccine_manufacturer.groupby("Country")["total_vaccinations"].max()

border = pd.read_csv(f"{DATA_DIR}/Border_status.csv")


# --- Q1: Highest & lowest % of population vaccinated ---

merged = pd.merge(vaccine, population, on="Country")
merged["percentage_vaccinated"] = merged["people_fully_vaccinated"] / merged["Population"] * 100

high_coverage = (
    merged.loc[(merged["percentage_vaccinated"] >= 40) & (merged["percentage_vaccinated"] < 100)]
    .sort_values("percentage_vaccinated", ascending=False)
    .head(20)
)

low_coverage = (
    merged.loc[merged["percentage_vaccinated"] < 5]
    .sort_values("percentage_vaccinated", ascending=True)
    .head(20)
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(
    x="percentage_vaccinated", y="Country", data=high_coverage, color="#E69F00",
    order=high_coverage.sort_values("percentage_vaccinated", ascending=False).Country,
    ax=axes[0],
)
axes[0].set_title("Top 20 countries by vaccination coverage (>=40%)")
axes[0].set_xlabel("% Fully Vaccinated")

sns.barplot(
    x="percentage_vaccinated", y="Country", data=low_coverage, color="#56B4E9",
    order=low_coverage.sort_values("percentage_vaccinated", ascending=False).Country,
    ax=axes[1],
)
axes[1].set_title("Bottom 20 countries by vaccination coverage (<5%)")
axes[1].set_xlabel("% Fully Vaccinated")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/q1_vaccination_coverage.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/q1_vaccination_coverage.png")


# --- Q2: Vaccination coverage vs border status ---

merged_mfr = pd.merge(vaccine_manufacturer, population, on="Country")
merged_mfr["percentage_vaccinated"] = merged_mfr["total_vaccinations"] / merged_mfr["Population"] * 100

df_border = pd.merge(merged_mfr, border, on="Country")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(
    x="Border_status_tourist", y="percentage_vaccinated", data=df_border,
    showmeans=True, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"},
    ax=axes[0],
)
axes[0].set_title("Vaccination coverage per border status (tourists)")

sns.boxplot(
    x="Border_status_resident", y="percentage_vaccinated", data=df_border,
    showmeans=True, meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"},
    ax=axes[1],
)
axes[1].set_title("Vaccination coverage per border status (residents)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/q2_border_status.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/q2_border_status.png")
