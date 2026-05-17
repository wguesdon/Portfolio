"""Convert ps6e4_showcase_v3.py (percent-format) to a beautified .ipynb notebook."""

import re
from pathlib import Path
import nbformat

SRC = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4/notebooks/ps6e4_showcase_v3.py")
DST = Path("/mnt/data/Github/Kaggle/Playground_Series/PS6E4/notebooks/ps6e4_showcase_v3.ipynb")

# ---------------------------------------------------------------------------
# 1. Parse .py into cells
# ---------------------------------------------------------------------------

raw = SRC.read_text()
# Split on lines that start with "# %%"
blocks = re.split(r"^# %%", raw, flags=re.MULTILINE)

cells = []
for i, block in enumerate(blocks):
    if i == 0 and not block.strip():
        continue  # skip empty preamble before first # %%

    is_markdown = block.startswith(" [markdown]")
    if is_markdown:
        block = block[len(" [markdown]"):]
        # Strip the leading "# " prefix from each line
        lines = block.split("\n")
        cleaned = []
        for line in lines:
            if line.startswith("# "):
                cleaned.append(line[2:])
            elif line == "#":
                cleaned.append("")
            else:
                cleaned.append(line)
        text = "\n".join(cleaned).strip()
        cells.append(("markdown", text))
    else:
        code = block.strip()
        if code:
            cells.append(("code", code))

# ---------------------------------------------------------------------------
# 2. Beautify markdown cells
# ---------------------------------------------------------------------------

TITLE_HTML = """\
<div style="padding: 30px; background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); border-radius: 10px; margin: 10px 0;">
<h1 style="color: white; margin: 0; font-size: 2em;">PS6E4: Predicting Irrigation Need</h1>
<h3 style="color: #ecf0f1; margin: 10px 0 0 0; font-weight: normal;">End-to-End Pipeline with 3-Model Ensemble</h3>
<hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
<p style="color: #bdc3c7; margin: 0;">Competition: Playground Series S6E4 | Metric: Balanced Accuracy | GPU: T4</p>
</div>"""

SECTION_COLORS = {
    "1": "#2980b9",   # Setup - blue
    "2": "#2980b9",   # Load Data - blue
    "3": "#2980b9",   # Constants - blue
    "4": "#27ae60",   # EDA - green
    "5": "#f39c12",   # Feature Eng - orange
    "6": "#e74c3c",   # Model Training - red
    "7": "#8e44ad",   # Ensemble - purple
    "8": "#2980b9",   # Submission - blue
}

SECTION_SUBTITLES = {
    "1": "Install and import all required libraries",
    "2": "Read competition CSVs and the original 10K dataset",
    "3": "Define column groups, constants, and encode target",
    "4": "Visualize target distribution, feature distributions, and key patterns",
    "5": "Seven-stage pipeline: magic formula, domain, feature-engine, TE priors, interactions",
    "6": "XGBoost, LightGBM, CatBoost with 5-fold CV and original data injection",
    "7": "Hill climbing weights, log-bias tuning, differential evolution thresholds",
    "8": "Generate and save the final prediction file",
}


def make_section_header(title, section_num):
    """Create an HTML-styled section header."""
    color = SECTION_COLORS.get(section_num, "#2980b9")
    subtitle = SECTION_SUBTITLES.get(section_num, "")
    bg_map = {
        "#2980b9": "#eef5fc",
        "#27ae60": "#eafaf1",
        "#f39c12": "#fef9e7",
        "#e74c3c": "#fdedec",
        "#8e44ad": "#f4ecf7",
    }
    bg = bg_map.get(color, "#eef5fc")
    sub_html = f'\n<p style="color: #7f8c8d; margin: 5px 0 0 0;">{subtitle}</p>' if subtitle else ""
    return f"""\
<div style="padding: 20px; border-left: 5px solid {color}; background: linear-gradient(to right, {bg}, #ffffff); margin: 10px 0;">
<h2 style="color: #2c3e50; margin: 0;">{title}</h2>{sub_html}
</div>"""


def make_subsection_header(title):
    """Create an HTML-styled subsection header."""
    return f"""\
<div style="padding: 12px 20px; border-left: 4px solid #95a5a6; background-color: #f8f9fa; margin: 10px 0;">
<h3 style="color: #2c3e50; margin: 0;">{title}</h3>
</div>"""


def make_info_box(text):
    """Green info box for observations."""
    return f"""\
<div style="padding: 15px; background-color: #eafaf1; border-left: 4px solid #27ae60; border-radius: 4px; margin: 10px 0;">
{text}
</div>"""


def make_warning_box(text):
    """Orange warning/note box."""
    return f"""\
<div style="padding: 15px; background-color: #fef9e7; border-left: 4px solid #f39c12; border-radius: 4px; margin: 10px 0;">
{text}
</div>"""


def make_approach_box(text):
    """Blue styled box for approach/pipeline info."""
    return f"""\
<div style="padding: 15px 20px; background-color: #eef5fc; border-left: 4px solid #2980b9; border-radius: 4px; margin: 10px 0;">
{text}
</div>"""


def make_summary_card(text):
    """Styled card for summary table."""
    return f"""\
<div style="padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border: 1px solid #dee2e6; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Summary</h2>
{text}
</div>"""


def beautify_markdown(text, index):
    """Apply HTML beautification to a markdown cell."""

    # Title cell (first markdown)
    if index == 0:
        # Extract approach list
        approach_lines = []
        key_insight = ""
        for line in text.split("\n"):
            if line.startswith("- "):
                approach_lines.append(line)
            if line.startswith("**Key insight:**"):
                key_insight = line

        approach_md = "\n".join(approach_lines)
        approach_html = make_approach_box(
            "<strong>Approach:</strong>\n\n" + approach_md
        )
        insight_html = ""
        if key_insight:
            insight_html = "\n\n" + make_warning_box(
                f"<strong>Key Insight:</strong> {key_insight.replace('**Key insight:** ', '')}"
            )

        return TITLE_HTML + "\n\n" + approach_html + insight_html

    # Summary cell (last section)
    if text.strip().startswith("## Summary"):
        table_lines = []
        in_table = False
        for line in text.split("\n"):
            if line.startswith("|"):
                table_lines.append(line)
                in_table = True
            elif in_table:
                break
        table_md = "\n".join(table_lines)
        return make_summary_card(table_md)

    # Major section headers: "## N. Title"
    m = re.match(r"^## (\d+)\.\s+(.*)", text.strip().split("\n")[0])
    if m:
        section_num = m.group(1)
        title = f"{section_num}. {m.group(2)}"
        rest_lines = text.strip().split("\n")[1:]
        rest = "\n".join(rest_lines).strip()

        header_html = make_section_header(title, section_num)

        if rest:
            # Check if rest has bullet points (approach/pipeline info)
            if any(line.strip().startswith("- ") or line.strip().startswith("1.") for line in rest_lines if line.strip()):
                rest_html = make_approach_box(rest)
            elif "| Model |" in rest or "|----" in rest:
                # Table content
                rest_html = make_approach_box(rest)
            else:
                rest_html = make_info_box(rest)
            return header_html + "\n\n" + rest_html
        return header_html

    # Subsection headers: "### N.M Title"
    m = re.match(r"^### ([\d.]+)\s+(.*)", text.strip().split("\n")[0])
    if m:
        title = f"{m.group(1)} {m.group(2)}"
        rest_lines = text.strip().split("\n")[1:]
        rest = "\n".join(rest_lines).strip()
        header_html = make_subsection_header(title)
        if rest:
            # Check for code blocks
            if "```" in rest:
                rest_html = make_approach_box(rest)
            else:
                rest_html = make_info_box(rest)
            return header_html + "\n\n" + rest_html
        return header_html

    # Observation blocks (start with **Observations:** or similar)
    if text.strip().startswith("**Observ"):
        return make_info_box(text)

    # Lines that start with "The" or "Most" (commentary after plots)
    if (text.strip().startswith("The ") or text.strip().startswith("Most ")):
        return make_info_box(text)

    # Default: return as-is
    return text


beautified_cells = []
md_index = 0
for cell_type, content in cells:
    if cell_type == "markdown":
        content = beautify_markdown(content, md_index)
        md_index += 1
    beautified_cells.append((cell_type, content))

# ---------------------------------------------------------------------------
# 3. Insert additional EDA cells after section 4.4
# ---------------------------------------------------------------------------

# Find the index of the cell after the magic formula plot (section 4.4 code + observation)
# We need to find where section 4.4 observations end, before section 5
insert_idx = None
for i, (ct, content) in enumerate(beautified_cells):
    if ct == "markdown" and "5. Feature Engineering" in content:
        insert_idx = i
        break

if insert_idx is None:
    raise ValueError("Could not find section 5 marker for insertion point")

new_eda_cells = [
    ("markdown", make_subsection_header("4.5 Categorical Feature Distribution by Class")),
    ("code", """\
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, col in enumerate(CAT_COLS):
    ax = axes[i // 4, i % 4]
    ct = pd.crosstab(train[col], train[TARGET], normalize='index')[CLASS_ORDER]
    ct.plot(kind='bar', stacked=True, color=CLASS_PALETTE, ax=ax, legend=False, edgecolor='white', linewidth=0.5)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    if i == 0:
        ax.legend(CLASS_ORDER, title='Class', fontsize=8, loc='upper right')
fig.suptitle('Class Proportions by Categorical Feature', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""),
    ("markdown", make_info_box(
        "<strong>Key Finding:</strong> Crop Growth Stage is the strongest categorical separator. "
        "Harvest and Sowing stages have very different class proportions compared to other stages. "
        "Mulching also shows clear differentiation, consistent with the magic formula."
    )),
    ("markdown", make_subsection_header("4.6 Pairwise Scatter of Top 3 Numeric Features")),
    ("code", """\
top3 = ['Soil_Moisture', 'Temperature_C', 'Rainfall_mm']
sample = train.sample(10000, random_state=42)
g = sns.pairplot(
    sample, vars=top3, hue=TARGET, hue_order=CLASS_ORDER,
    palette=CLASS_COLORS, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10},
    height=3,
)
g.figure.suptitle('Pairwise Feature Scatter (10K Sample)', fontweight='bold', y=1.02)
plt.show()"""),
    ("markdown", make_info_box(
        "<strong>Key Finding:</strong> Soil Moisture provides the clearest separation in scatter space. "
        "High irrigation samples cluster at low soil moisture values. "
        "Temperature and Rainfall show overlapping distributions but still contribute to separation when combined."
    )),
]

beautified_cells = beautified_cells[:insert_idx] + new_eda_cells + beautified_cells[insert_idx:]

# ---------------------------------------------------------------------------
# 4. Build notebook with nbformat
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.10.0",
}

for cell_type, content in beautified_cells:
    if cell_type == "markdown":
        nb.cells.append(nbformat.v4.new_markdown_cell(content))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(content))

nbformat.write(nb, str(DST))
print(f"Notebook written to {DST}")
print(f"Total cells: {len(nb.cells)}")
md_count = sum(1 for c in nb.cells if c.cell_type == "markdown")
code_count = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"  Markdown: {md_count}")
print(f"  Code: {code_count}")
