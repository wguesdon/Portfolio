"""Render the PS6E4 model and ensembling DAG as a PNG.

Pure matplotlib so it does not depend on a system Graphviz install.

Outputs:
    docs/dag.png
    docs/dag.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Layout coords roughly matching graphviz top to bottom
# x in [0, 14], y in [0, 12]


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.6,
    fontsize: float = 9.0,
) -> tuple[float, float, float, float]:
    """Draw a rounded box and return its bbox as (x0, y0, x1, y1).

    Args:
        ax: matplotlib axes to draw on.
        xy: (x_center, y_center) of the box.
        width: box width in data coords.
        height: box height in data coords.
        text: text to render inside the box.
        facecolor: fill color.
        edgecolor: stroke color.
        linewidth: stroke width.
        fontsize: text font size.

    Returns:
        (x0, y0, x1, y1) bounding box in data coords.
    """
    x, y = xy
    x0, y0 = x - width / 2, y - height / 2
    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans",
    )
    return x0, y0, x0 + width, y0 + height


def _arrow(
    ax: plt.Axes,
    src_box: tuple[float, float, float, float],
    dst_box: tuple[float, float, float, float],
    color: str = "#555555",
    linewidth: float = 1.4,
) -> None:
    """Draw an arrow from the bottom edge of src_box to the top edge of dst_box.

    Args:
        ax: matplotlib axes.
        src_box: source bbox (x0, y0, x1, y1).
        dst_box: destination bbox (x0, y0, x1, y1).
        color: arrow color.
        linewidth: arrow line width.
    """
    sx = (src_box[0] + src_box[2]) / 2
    sy = src_box[1]
    dx = (dst_box[0] + dst_box[2]) / 2
    dy = dst_box[3]
    arrow = FancyArrowPatch(
        (sx, sy),
        (dx, dy),
        arrowstyle="-|>",
        mutation_scale=14,
        color=color,
        linewidth=linewidth,
        connectionstyle="arc3,rad=0.0",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def render(out_dir: Path) -> None:
    """Render the DAG to PNG and SVG inside out_dir.

    Args:
        out_dir: directory to write dag.png and dag.svg to.
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=160)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.4, 13)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors
    c_data = ("#e8e8e8", "#333333")
    c_fe = ("#cfe2ff", "#0d6efd")
    c_strong = ("#fff3cd", "#664d03")
    c_diversity = ("#d1e7dd", "#0a3622")
    c_stack = ("#ffe5d0", "#c2410c")
    c_final = ("#a8e6a3", "#0a3622")

    # Row 1: Data
    data_text = (
        "Raw Data\n"
        "Train 630K rows, Test 270K rows\n"
        "20 features, 3 classes\n"
        "Low 58.7%, Medium 37.9%, High 3.3%"
    )
    data_box = _box(ax, (8, 11), 5.5, 1.3, data_text, *c_data, fontsize=10)

    # Row 2: Feature pipelines
    fe1_text = (
        "OTE Pipeline\n"
        "4x shuffle concat\n"
        "Leave one out target encoding\n"
        "Digit extraction"
    )
    fe2_text = (
        "v3 Pipeline\n"
        "Magic ratios + domain features\n"
        "Standard target encoding\n"
        "2 way interactions"
    )
    fe1_box = _box(ax, (4.5, 9.1), 4.6, 1.3, fe1_text, *c_fe, fontsize=9.5)
    fe2_box = _box(ax, (11.5, 9.1), 4.6, 1.3, fe2_text, *c_fe, fontsize=9.5)

    # Row 3: Base model groups
    ote_top_text = (
        "Top OTE Boosters\n"
        "LGB OTE              0.97942\n"
        "XGB OTE              0.97938\n"
        "XGB OTE shallow GPU  0.97927\n"
        "CAT OTE              0.97919\n"
        "XGB OTE magic        0.97910\n"
        "XGB OTE seeds 43, 44\n"
        "LGB OTE shallow / deep"
    )
    ote_div_text = (
        "OTE Diversity Models\n"
        "ExtraTrees       0.96156\n"
        "cuML SVM RBF     0.96154\n"
        "cuML RF          0.96005\n"
        "LR ElasticNet    0.95568\n"
        "LR L1            0.95554\n"
        "LR L2            0.95522\n"
        "cuML GaussianNB  0.90860\n"
        "KNN k=15         0.71937"
    )
    v3_gbdt_text = (
        "v3 Gradient Boosters\n"
        "7 CatBoost variants\n"
        "  0.97786 to 0.97834\n"
        "4 LightGBM variants\n"
        "  0.97131 to 0.97701\n"
        "6 XGBoost variants\n"
        "  0.97185 to 0.97561"
    )
    v3_nn_text = (
        "v3 Neural Networks\n"
        "RealMLP mahog   0.97802\n"
        "RealMLP v3fix   0.97108\n"
        "TabM v3fix      0.97053"
    )
    ote_top_box = _box(ax, (2.0, 6.5), 3.6, 2.6, ote_top_text, *c_strong, fontsize=8.5)
    ote_div_box = _box(ax, (6.4, 6.5), 3.6, 2.6, ote_div_text, *c_diversity, fontsize=8.5)
    v3_gbdt_box = _box(ax, (10.6, 6.5), 3.6, 2.6, v3_gbdt_text, *c_strong, fontsize=8.5)
    v3_nn_box = _box(ax, (14.0, 6.5), 3.4, 2.6, v3_nn_text, *c_diversity, fontsize=8.5)

    # Row 4: Stacker
    stack_text = (
        "LightGBM Stacker\n"
        "41 base model OOF probabilities\n"
        "5 fold StratifiedKFold seed 42\n"
        "CV 0.98045"
    )
    stack_box = _box(ax, (8, 3.6), 6.0, 1.3, stack_text, *c_stack, fontsize=10, linewidth=2.0)

    # Row 5: Log bias
    logbias_text = (
        "Log Bias Correction\n"
        "Per class additive shift on log proba\n"
        "Tuned on OOF for balanced accuracy"
    )
    logbias_box = _box(ax, (8, 2.0), 6.0, 1.0, logbias_text, *c_stack, fontsize=10, linewidth=2.0)

    # Row 6: Final
    sub_text = (
        "Final Submission v15\n"
        "Public LB  0.98081     Private LB  0.98082\n"
        "12 of 457"
    )
    sub_box = _box(ax, (8, 0.6), 6.0, 0.95, sub_text, *c_final, fontsize=10.5, linewidth=2.0)

    # Arrows
    _arrow(ax, data_box, fe1_box)
    _arrow(ax, data_box, fe2_box)
    _arrow(ax, fe1_box, ote_top_box)
    _arrow(ax, fe1_box, ote_div_box)
    _arrow(ax, fe2_box, v3_gbdt_box)
    _arrow(ax, fe2_box, v3_nn_box)
    _arrow(ax, ote_top_box, stack_box)
    _arrow(ax, ote_div_box, stack_box)
    _arrow(ax, v3_gbdt_box, stack_box)
    _arrow(ax, v3_nn_box, stack_box)
    _arrow(ax, stack_box, logbias_box, linewidth=1.8)
    _arrow(ax, logbias_box, sub_box, linewidth=1.8)

    # Title
    ax.text(
        8,
        12.85,
        "PS6E4 Irrigation Need: Models and Ensembling DAG",
        ha="center",
        va="bottom",
        fontsize=14,
        weight="bold",
    )

    # Legend, placed under the title above the data box
    legend_handles = [
        mpatches.Patch(facecolor=c_data[0], edgecolor=c_data[1], label="Data"),
        mpatches.Patch(facecolor=c_fe[0], edgecolor=c_fe[1], label="Feature pipeline"),
        mpatches.Patch(facecolor=c_strong[0], edgecolor=c_strong[1], label="Strong base models"),
        mpatches.Patch(facecolor=c_diversity[0], edgecolor=c_diversity[1], label="Diversity base models"),
        mpatches.Patch(facecolor=c_stack[0], edgecolor=c_stack[1], label="Meta learner"),
        mpatches.Patch(facecolor=c_final[0], edgecolor=c_final[1], label="Final submission"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        fontsize=9,
        frameon=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "dag.png"
    svg_path = out_dir / "dag.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = here.parent / "docs"
    render(out)
