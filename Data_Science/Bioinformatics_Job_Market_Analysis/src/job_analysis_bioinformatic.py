"""Bioinformatics job market analysis pipeline.

Analyzes bioinformatics job postings to extract insights on
required skills, salary distributions, locations, and skill co-occurrence.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "26-12-2024_bioinformatics_jobs_data.xlsx"


# ------------------------------------
# 1. Text Preprocessing & Data Loading
# ------------------------------------
def unify_r_and_statistics_variations(text):
    """Replace various 'R' patterns with 'r' and unify 'statistical'/'statistics'.

    Args:
        text: Raw text string from a job description.

    Returns:
        Text with unified R and statistics tokens.
    """
    r_patterns = [
        r"\b[rR]\b",
        r"\b[rR][-\s]?based\b",
        r"\b[rR][-\s]?programming\b",
    ]
    text_updated = text
    for pattern in r_patterns:
        text_updated = re.sub(pattern, " r ", text_updated, flags=re.IGNORECASE)

    text_updated = re.sub(
        r"\bstatistical(s)?\b", " statistics ", text_updated, flags=re.IGNORECASE
    )
    return text_updated


def load_data(file_path, sheet_name="Sheet1"):
    """Load Excel data into a pandas DataFrame.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name of the sheet to parse.

    Returns:
        DataFrame with job posting data.
    """
    job_data = pd.ExcelFile(file_path)
    df = job_data.parse(sheet_name)
    return df


# ---------------------
# 2. Skills Analysis
# ---------------------
def analyze_skills(df, skill_list, text_col="Job description"):
    """Count occurrences of specified keywords in job descriptions.

    Uses binary CountVectorizer so each keyword is counted at most once per posting.
    Unifies variations of 'R' and 'statistics' before counting.

    Args:
        df: DataFrame with job postings.
        skill_list: List of skill keywords to search for.
        text_col: Column containing job description text.

    Returns:
        DataFrame with columns ['Skill', 'Count'] sorted by frequency.
    """
    if text_col not in df.columns:
        print(f"Warning: '{text_col}' not found in DataFrame. Returning empty result.")
        return pd.DataFrame({"Skill": skill_list, "Count": [0] * len(skill_list)})

    corpus = df[text_col].fillna("").astype(str).str.lower()
    corpus = corpus.apply(unify_r_and_statistics_variations)

    vectorizer = CountVectorizer(
        vocabulary=[kw.lower() for kw in skill_list],
        token_pattern=r"(?u)\b\w+\b",
        binary=True,
    )

    skills_matrix = vectorizer.fit_transform(corpus)
    skills_count = (
        pd.DataFrame(skills_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        .sum(axis=0)
        .sort_values(ascending=False)
    )

    skills_count_df = skills_count.reset_index()
    skills_count_df.columns = ["Skill", "Count"]
    return skills_count_df


def plot_skill_distribution(skills_count_df, title="Keyword Frequency", filename="skills_distribution.png"):
    """Plot a bar chart showing skill/keyword frequency.

    Args:
        skills_count_df: DataFrame with 'Skill' and 'Count' columns.
        title: Chart title.
        filename: Output filename for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(skills_count_df["Skill"], skills_count_df["Count"], color="skyblue", edgecolor="black")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Skills", fontsize=12)
    ax.set_ylabel("Frequency in Job Postings", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


# ----------------------------
# 3. Location & Country Trends
# ----------------------------
def analyze_locations(df, column="Location", top_n=10):
    """Return the top N most frequent entries in a specified column.

    Args:
        df: DataFrame with job postings.
        column: Column to aggregate.
        top_n: Number of top entries to return.

    Returns:
        Series with value counts.
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in DataFrame.")
        return pd.Series(dtype=int)
    return df[column].value_counts().head(top_n)


# -----------------
# 4. Salary Analysis
# -----------------
def analyze_salaries(df, min_col="Min salary", max_col="Max salary"):
    """Compute average salary from min/max salary columns.

    Args:
        df: DataFrame with job postings.
        min_col: Column name for minimum salary.
        max_col: Column name for maximum salary.

    Returns:
        DataFrame subset with valid salary data and an 'Average salary' column.
    """
    if min_col not in df.columns or max_col not in df.columns:
        print(f"Warning: Salary columns '{min_col}' or '{max_col}' not found.")
        return pd.DataFrame(columns=[min_col, max_col, "Average salary"])

    salary_data = df.dropna(subset=[min_col, max_col]).copy()
    salary_data[min_col] = pd.to_numeric(salary_data[min_col], errors="coerce")
    salary_data[max_col] = pd.to_numeric(salary_data[max_col], errors="coerce")
    salary_data.dropna(subset=[min_col, max_col], inplace=True)
    salary_data["Average salary"] = (salary_data[min_col] + salary_data[max_col]) / 2
    return salary_data


def plot_salary_distribution(salary_data, title="Distribution of Average Salaries", filename="salary_distribution.png"):
    """Plot a histogram of average salary distribution.

    Args:
        salary_data: DataFrame with 'Average salary' column.
        title: Chart title.
        filename: Output filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    salary_data["Average salary"].plot(kind="hist", bins=20, edgecolor="black", ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Average Salary", fontsize=12)
    ax.set_ylabel("Number of Job Postings", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def analyze_salaries_by_countries(
    df,
    countries=None,
    country_column="Input Country",
    min_col="Min salary",
    max_col="Max salary",
):
    """Analyze and plot salary distributions per country.

    Args:
        df: DataFrame with job postings.
        countries: List of countries to analyze (None = all).
        country_column: Column with country names.
        min_col: Minimum salary column.
        max_col: Maximum salary column.
    """
    if countries is None:
        countries = df[country_column].dropna().unique()

    for country in countries:
        country_data = df[df[country_column].str.lower() == country.lower()].copy()
        if country_data.empty:
            print(f"No data for country: {country}")
            continue

        salary_data = analyze_salaries(country_data, min_col, max_col)
        if salary_data.empty:
            print(f"No valid salary data for country: {country}")
        else:
            print(f"\n--- Salary analysis for {country} ---")
            print(salary_data[["Min salary", "Max salary", "Average salary"]].describe())
            safe_name = country.lower().replace(" ", "_")
            plot_salary_distribution(
                salary_data,
                title=f"Salary Distribution in {country}",
                filename=f"salary_distribution_{safe_name}.png",
            )


# --------------------------------
# 5. Additional Plotting Functions
# --------------------------------
def plot_job_type_distribution(df, job_type_col="Job type", filename="job_type_distribution.png"):
    """Plot a bar chart of job type counts.

    Args:
        df: DataFrame with job postings.
        job_type_col: Column with job type data.
        filename: Output filename.
    """
    if job_type_col not in df.columns:
        print(f"Warning: Column '{job_type_col}' not found in DataFrame.")
        return

    job_type_counts = df[job_type_col].value_counts(dropna=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(job_type_counts.index.astype(str), job_type_counts.values, color="lightgreen", edgecolor="black")
    ax.set_title("Job Type Distribution", fontsize=16)
    ax.set_xlabel("Job Type", fontsize=12)
    ax.set_ylabel("Number of Postings", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_company_rating_distribution(df, rating_col="Company rating", filename="company_rating_distribution.png"):
    """Plot a histogram of company ratings.

    Args:
        df: DataFrame with job postings.
        rating_col: Column with company rating data.
        filename: Output filename.
    """
    if rating_col not in df.columns:
        print(f"Warning: Column '{rating_col}' not found in DataFrame.")
        return

    ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()
    if ratings.empty:
        print("No valid numeric ratings found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(ratings, bins=10, color="orange", edgecolor="black")
    ax.set_title("Company Rating Distribution", fontsize=16)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_location_distribution(df, location_col="Location", top_n=10, filename="location_distribution.png"):
    """Plot a bar chart of the top N most frequent locations.

    Args:
        df: DataFrame with job postings.
        location_col: Column with location data.
        top_n: Number of top locations to show.
        filename: Output filename.
    """
    if location_col not in df.columns:
        print(f"Warning: Column '{location_col}' not found in DataFrame.")
        return

    top_locations = df[location_col].value_counts().head(top_n)
    if top_locations.empty:
        print("No valid location data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_locations.index.astype(str), top_locations.values, color="lightblue", edgecolor="black")
    ax.set_title(f"Top {top_n} Locations", fontsize=16)
    ax.set_xlabel("Location", fontsize=12)
    ax.set_ylabel("Number of Postings", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


# ---------------------------------------------
# 6. Skill Co-occurrence & Radar (Spider) Plots
# ---------------------------------------------
def create_skill_presence_columns(df, skill_list, text_col="Job description"):
    """Mark skill presence (1/0) for each job posting.

    Args:
        df: DataFrame with job postings.
        skill_list: List of skills to detect.
        text_col: Column with job description text.

    Returns:
        DataFrame with new binary columns for each skill.
    """
    if text_col not in df.columns:
        print(f"[Warning] Column '{text_col}' not found in DataFrame.")
        return df

    df["_temp_text"] = (
        df[text_col].fillna("").astype(str).str.lower().apply(unify_r_and_statistics_variations)
    )

    for skill in skill_list:
        skill_lower = skill.lower()
        df[skill] = df["_temp_text"].apply(lambda x, s=skill_lower: 1 if s in x else 0)

    df.drop(columns=["_temp_text"], inplace=True)
    return df


def build_cooccurrence_matrix(df, skill_list):
    """Build a co-occurrence matrix from binary skill columns.

    Args:
        df: DataFrame with binary skill presence columns.
        skill_list: List of skill column names.

    Returns:
        DataFrame with co-occurrence counts (skill x skill).
    """
    skill_df = df[skill_list].copy()
    return skill_df.T.dot(skill_df)


def plot_spider_for_skill(co_occ_matrix, skill_list, center_skill, filename=None):
    """Create a radar chart for a single skill's co-occurrence with others.

    Args:
        co_occ_matrix: Co-occurrence matrix DataFrame.
        skill_list: List of all skills.
        center_skill: Skill to place at the center of the radar.
        filename: Output filename (auto-generated if None).
    """
    if center_skill not in skill_list:
        print(f"[Warning] '{center_skill}' is not in the skill list.")
        return

    if filename is None:
        filename = f"spider_{center_skill.lower().replace(' ', '_')}.png"

    row_data = co_occ_matrix.loc[center_skill, skill_list].copy()
    row_data.drop(center_skill, inplace=True)

    labels = row_data.index.tolist()
    values = row_data.values.tolist()

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="red", linewidth=2)
    ax.fill(angles, values, color="red", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"Co-occurrence of '{center_skill}' with Other Skills", y=1.08)
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_multi_skill_spider(co_occ_matrix, skill_list, filename="spider_multi_skill.png"):
    """Plot multiple skills on one radar chart.

    Args:
        co_occ_matrix: Co-occurrence matrix DataFrame.
        skill_list: List of skills to overlay.
        filename: Output filename.
    """
    labels = skill_list
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for skill in skill_list:
        row_data = co_occ_matrix.loc[skill, :].copy()
        row_data.loc[skill] = 0
        values = row_data.values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=skill)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Skill-to-Skill Co-occurrence Radar Chart", y=1.08)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


# ----------------------------------
# 7. Main Script
# ----------------------------------
def main():
    """Run the full bioinformatics job market analysis pipeline."""
    # 1. Load the dataset
    df = load_data(INPUT_FILE)

    # 2. Define keywords
    all_keywords = [
        "r", "python", "nextflow", "nf-core", "Snakemake",
        "clinical", "immunology", "oncology", "Immuno-oncology", "proteomic",
        "machine learning", "ML", "genAI", "LLM", "genomic", "NGS",
        "XGBoost", "scikit-learn", "regression", "PyTorch", "TensorFlow",
        "Neural Network", "GPT", "AWS", "cloud",
        "statistics", "spss", "Flow Cytometry",
    ]

    # 3. Skills analysis
    skills_count_df = analyze_skills(df, all_keywords)
    print("\nSkills Analysis (R & 'statistics' variations unified):")
    print(skills_count_df)

    # 4. Plot skill frequency
    plot_skill_distribution(skills_count_df, title="Skills Distribution in Bioinformatics Jobs")

    # 5. Job trends by location
    print("\nTop 10 Locations:")
    location_counts = analyze_locations(df, column="Location", top_n=10)
    print(location_counts)

    # 6. Job trends by country
    print("\nTop 10 Countries:")
    country_counts = analyze_locations(df, column="Input Country", top_n=10)
    print(country_counts)

    # 7. Overall salary analysis
    salary_data = analyze_salaries(df, min_col="Min salary", max_col="Max salary")
    if not salary_data.empty:
        plot_salary_distribution(salary_data)
    else:
        print("No valid salary data to analyze.")

    # 7.1. Salary analysis by country
    countries_to_analyze = ["USA", "Ireland", "UK"]
    analyze_salaries_by_countries(
        df,
        countries=countries_to_analyze,
        country_column="Input Country",
        min_col="Min salary",
        max_col="Max salary",
    )

    # 8. Create R and Python columns
    text_col = "Job description"
    if text_col in df.columns:
        df["_temp_unified"] = (
            df[text_col].fillna("").astype(str).str.lower().apply(unify_r_and_statistics_variations)
        )
        df["R"] = df["_temp_unified"].apply(lambda x: "Yes" if re.search(r"\br\b", x) else "No")
        df["Python"] = df["_temp_unified"].apply(lambda x: "Yes" if "python" in x else "No")
        df.drop(columns=["_temp_unified"], inplace=True)
    else:
        df["R"] = "No"
        df["Python"] = "No"

    # 9. Export modified dataset
    output_csv_path = DATA_DIR / "modified_jobs_data.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nModified dataset saved to: {output_csv_path}")

    # 10. Create filtered datasets
    r_df = df[df["R"] == "Yes"].copy()
    python_df = df[df["Python"] == "Yes"].copy()

    r_csv_path = DATA_DIR / "modified_jobs_data_R_yes.csv"
    python_csv_path = DATA_DIR / "modified_jobs_data_Python_yes.csv"

    r_df.to_csv(r_csv_path, index=False)
    python_df.to_csv(python_csv_path, index=False)

    print(f"R=Yes dataset: {r_csv_path} (Rows: {r_df.shape[0]})")
    print(f"Python=Yes dataset: {python_csv_path} (Rows: {python_df.shape[0]})")

    # 11. Plot distributions
    plot_job_type_distribution(df)
    plot_company_rating_distribution(df)
    plot_location_distribution(df)

    # 12. Skill co-occurrence & spider plots
    skills_of_interest = ["statistics", "python", "r", "spss", "ml", "cloud"]
    df = create_skill_presence_columns(df, skills_of_interest, text_col=text_col)
    cooccurrence_matrix = build_cooccurrence_matrix(df, skills_of_interest)
    print("\nCo-occurrence matrix:\n", cooccurrence_matrix)

    plot_spider_for_skill(cooccurrence_matrix, skills_of_interest, center_skill="python")
    plot_spider_for_skill(cooccurrence_matrix, skills_of_interest, center_skill="r")
    plot_spider_for_skill(cooccurrence_matrix, skills_of_interest, center_skill="statistics")
    plot_multi_skill_spider(cooccurrence_matrix, skills_of_interest)


if __name__ == "__main__":
    main()
