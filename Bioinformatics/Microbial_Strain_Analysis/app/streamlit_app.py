"""
Microbial Strain Analysis — Interactive Streamlit Application

Run with:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from microbial.data.preprocessor import get_data
from microbial.analysis.stats import (
    phylum_summary, class_summary, o2_summary,
    top_compounds, media_per_strain, strains_per_phylum_o2,
    correlation_matrix,
)
from microbial.viz.plots import (
    phylum_bar, class_treemap, o2_pie,
    temp_violin, ph_violin, temp_ph_scatter,
    growth_range_box, correlation_heatmap,
    compound_freq_bar, media_richness_hist,
    media_per_strain_hist, phylum_o2_heatmap,
    PALETTE,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Microbial Strain Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.insight {
    background: #f0f7ff;
    border-left: 4px solid #0072B2;
    padding: 0.6rem 0.9rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load():
    return get_data()


data = load()
strains         = data["strains"]
strain_media    = data["strain_media"]
media_compounds = data["media_compounds"]
compounds       = data["compounds"]
compound_freq   = data["compound_freq"]
media_richness  = data["media_richness"]
swm             = data["strains_with_media"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🦠 Microbial Strain Explorer")
st.sidebar.markdown("---")
st.sidebar.header("Filters")

all_phyla = sorted(strains["phylum"].dropna().unique())
selected_phyla = st.sidebar.multiselect(
    "Phylum", all_phyla, placeholder="All phyla", key="phylum_filter"
)

all_classes = sorted(strains["class"].dropna().unique())
selected_classes = st.sidebar.multiselect(
    "Class", all_classes, placeholder="All classes", key="class_filter"
)

all_o2 = sorted(strains["o2_tol"].dropna().unique())
selected_o2 = st.sidebar.multiselect(
    "O₂ Tolerance", all_o2, placeholder="All", key="o2_filter"
)

temp_min_val = float(strains["temp_opt"].min())
temp_max_val = float(strains["temp_opt"].max())
temp_range = st.sidebar.slider(
    "Optimal Temperature (°C)",
    min_value=temp_min_val, max_value=temp_max_val,
    value=(temp_min_val, temp_max_val), step=1.0,
    key="temp_filter",
)

ph_min_val = float(strains["ph_opt"].min())
ph_max_val = float(strains["ph_opt"].max())
ph_range = st.sidebar.slider(
    "Optimal pH",
    min_value=ph_min_val, max_value=ph_max_val,
    value=(ph_min_val, ph_max_val), step=0.1,
    key="ph_filter",
)

if st.sidebar.button("Reset all filters", use_container_width=True):
    for key in ["phylum_filter", "class_filter", "o2_filter", "temp_filter", "ph_filter"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("Glossary"):
    st.markdown("""
**Phylum / Class** — levels of taxonomic classification.
A phylum is a broad group; a class is a subdivision within it.

**O₂ Tolerance**
- *Aerobe* — grows in the presence of oxygen
- *Obligate anaerobe* — cannot survive in oxygen
- *Facultative anaerobe* — can grow with or without oxygen

**Temp opt / pH opt** — the temperature (°C) or pH at which the
strain grows best.

**Growth medium** — the nutrient mixture used to cultivate a strain
in the lab.

**Compound** — a chemical ingredient in a growth medium.
""")

st.sidebar.markdown("---")
st.sidebar.caption("Data: Microbial strain bioinformatics challenge dataset")

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    strains["temp_opt"].between(*temp_range)
    & strains["ph_opt"].between(*ph_range)
)
if selected_phyla:
    mask &= strains["phylum"].isin(selected_phyla)
if selected_classes:
    mask &= strains["class"].isin(selected_classes)
if selected_o2:
    mask &= strains["o2_tol"].isin(selected_o2)

filtered = strains[mask].copy()
filter_active = bool(
    selected_phyla or selected_classes or selected_o2
    or temp_range != (temp_min_val, temp_max_val)
    or ph_range != (ph_min_val, ph_max_val)
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_overview, tab_taxonomy,
 tab_growth, tab_media, tab_strain, tab_data) = st.tabs([
    "Overview", "Taxonomy",
    "Growth Parameters", "Media & Compounds", "Strain Lookup", "Data Explorer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: KEY FINDINGS
# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.title("Database Overview")

    if filter_active:
        st.info(f"Filters active — showing **{len(filtered):,}** of {len(strains):,} strains.")

    n_total = len(strains)
    n_filt  = len(filtered)
    delta   = f"{n_filt - n_total:+,}" if filter_active else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Strains",        f"{n_filt:,}", delta=delta)
    c2.metric("Phyla",          f"{filtered['phylum'].nunique()}")
    c3.metric("Classes",        f"{filtered['class'].nunique()}")
    c4.metric("O₂ categories",  f"{filtered['o2_tol'].nunique()}")
    c5.metric("Total media",    f"{strain_media['medium_id'].nunique():,}")

    # ── Contextual insights (dynamic, based on filtered data) ─────────────────
    phylum_counts = phylum_summary(filtered)
    o2_counts     = o2_summary(filtered)
    top_phylum    = phylum_counts.iloc[0]
    aerobe_row    = o2_counts[o2_counts["o2_tol"] == "aerobe"]
    aerobe_pct    = aerobe_row["pct"].values[0] if not aerobe_row.empty else 0
    median_temp   = filtered["temp_opt"].median()

    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(
            f'<div class="insight">Most common phylum: <strong>{top_phylum["phylum"].title()}</strong> '
            f'({top_phylum["pct"]:.0f}% of {n_filt:,} strains)</div>',
            unsafe_allow_html=True,
        )
    with ins2:
        st.markdown(
            f'<div class="insight">Aerobic strains: <strong>{aerobe_pct:.0f}%</strong> '
            f'— reflecting a lab cultivation bias toward oxygen-tolerant organisms</div>',
            unsafe_allow_html=True,
        )
    with ins3:
        st.markdown(
            f'<div class="insight">Median optimal temperature: <strong>{median_temp:.0f}°C</strong> '
            f'— consistent with a mesophile-dominated dataset</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Top Phyla")
        st.plotly_chart(phylum_bar(phylum_counts, top_n=12), use_container_width=True, key="ov_phylum")
    with col_r:
        st.subheader("O₂ Tolerance")
        st.plotly_chart(o2_pie(o2_counts), use_container_width=True, key="ov_o2")

    st.subheader("Temperature vs pH — where do strains live?")
    st.caption("Each point is one strain. Colour shows O₂ tolerance. Hover for details.")
    st.plotly_chart(temp_ph_scatter(filtered), use_container_width=True, key="ov_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
with tab_taxonomy:
    st.title("Taxonomic Distribution")

    st.subheader("Phylum → Class hierarchy")
    st.caption("Click a phylum to zoom in. Click the centre to zoom out.")
    st.plotly_chart(class_treemap(filtered, top_n=30), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Phyla")
        n_top = st.slider("Top N", 5, 30, 15, key="phylum_n")
        st.plotly_chart(phylum_bar(phylum_summary(filtered), top_n=n_top), use_container_width=True, key="tax_phylum")

    with col_r:
        st.subheader("Classes")
        n_top_class = st.slider("Top N", 5, 30, 15, key="class_n")
        class_counts = class_summary(filtered)
        fig = px.bar(
            class_counts.head(n_top_class),
            x="count", y="class", orientation="h",
            color="pct", color_continuous_scale="Blues",
            labels={"count": "Strains", "class": "Class"},
            title=f"Top {n_top_class} Classes",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, key="tax_class")

    st.subheader("Phylum × O₂ Tolerance")
    st.caption("How many strains of each phylum fall into each oxygen tolerance category.")
    phylum_o2 = strains_per_phylum_o2(
        filtered.merge(strain_media[["strain_id", "medium_id"]], on="strain_id", how="left")
    )
    n_heatmap = st.slider("Top N phyla", 5, 25, 12, key="heatmap_n")
    st.plotly_chart(phylum_o2_heatmap(phylum_o2, top_n=n_heatmap), use_container_width=True, key="tax_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: GROWTH PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with tab_growth:
    st.title("Growth Parameter Analysis")

    st.caption(
        "Growth parameters describe the environmental conditions each strain prefers. "
        "Comparing these across groups reveals ecological niches."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Optimal Temperature by O₂ Tolerance")
        st.plotly_chart(temp_violin(filtered), use_container_width=True, key="gp_temp")
    with col_r:
        st.subheader("Optimal pH by O₂ Tolerance")
        st.plotly_chart(ph_violin(filtered), use_container_width=True, key="gp_ph")

    st.subheader("Temperature Range by Phylum")
    st.caption("The span between minimum and maximum tolerated temperature — a measure of thermal flexibility.")
    n_phyla_box = st.slider("Top N phyla", 5, 20, 10, key="growth_box_n")
    st.plotly_chart(growth_range_box(filtered, top_phyla_n=n_phyla_box), use_container_width=True, key="gp_range")

    st.subheader("Correlation between growth parameters")
    st.caption("Values close to +1 or −1 indicate a strong relationship between two parameters.")
    corr = correlation_matrix(filtered)
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True, key="gp_corr")

    with st.expander("Summary statistics by O₂ tolerance"):
        df_tmp = filtered.copy()
        df_tmp["temp_range"] = df_tmp["temp_max"] - df_tmp["temp_min"]
        df_tmp["ph_range"]   = df_tmp["ph_max"] - df_tmp["ph_min"]
        st.dataframe(
            df_tmp.groupby("o2_tol")[
                ["temp_min", "temp_opt", "temp_max", "temp_range",
                 "ph_min", "ph_opt", "ph_max", "ph_range"]
            ].agg(["mean", "median", "std"]).round(2),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MEDIA & COMPOUNDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_media:
    st.title("Media & Compound Analysis")

    # Filtered media
    filtered_media     = strain_media[strain_media["strain_id"].isin(filtered["strain_id"])]
    filtered_media_ids = filtered_media["medium_id"].unique()
    filtered_mc        = media_compounds[media_compounds["medium_id"].isin(filtered_media_ids)]
    filtered_cf = (
        filtered_mc.merge(compounds, on="compound_id", how="left")
        .groupby(["compound_id", "compound_name"])["medium_id"]
        .nunique()
        .rename("media_count")
        .reset_index()
        .sort_values("media_count", ascending=False)
        .reset_index(drop=True)
    )
    filtered_mr = media_richness[media_richness["medium_id"].isin(filtered_media_ids)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media in selection",      f"{len(filtered_media_ids):,}")
    c2.metric("Distinct compounds",      f"{filtered_cf['compound_id'].nunique():,}")
    c3.metric("Strain–media links",      f"{len(filtered_media):,}")
    c4.metric("Median compounds/medium", f"{filtered_mr['compound_count'].median():.0f}")

    st.markdown("---")

    # ── Most common compounds ──────────────────────────────────────────────
    st.subheader("Most common compounds across media")
    st.caption("How many different media formulations include each compound.")
    n_compounds = st.slider("Top N compounds", 10, 50, 25, key="cpd_n")
    st.plotly_chart(compound_freq_bar(filtered_cf, top_n=n_compounds), use_container_width=True, key="med_cpd")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Compound richness per medium")
        st.caption("Distribution of how many compounds each medium contains.")
        st.plotly_chart(media_richness_hist(filtered_mr), use_container_width=True, key="med_rich")
    with col_r:
        st.subheader("Media per strain")
        st.caption("Distribution of how many different media each strain can grow in.")
        mps = media_per_strain(filtered_media)
        st.plotly_chart(media_per_strain_hist(mps), use_container_width=True, key="med_mps")

    st.markdown("---")

    # ── Medium profile ─────────────────────────────────────────────────────
    st.subheader("Medium profile")
    st.caption("Pick a medium ID to inspect its compound composition and the strains that use it.")

    available_media = sorted(filtered_media_ids.tolist())
    selected_medium = st.selectbox("Select medium", available_media, key="medium_sel")

    if selected_medium is not None:
        med_compounds = (
            media_compounds[media_compounds["medium_id"] == selected_medium]
            .merge(compounds, on="compound_id", how="left")
            [["compound_id", "compound_name"]]
        )
        med_strains = (
            strain_media[strain_media["medium_id"] == selected_medium]
            .merge(filtered[["strain_id", "phylum", "class", "o2_tol", "temp_opt", "ph_opt"]],
                   on="strain_id", how="inner")
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Medium {selected_medium}** contains **{len(med_compounds)}** compounds:")
            st.dataframe(med_compounds, use_container_width=True, height=300)
        with col_r:
            st.markdown(f"**{len(med_strains)}** strains (in current filter) use this medium:")
            st.dataframe(med_strains, use_container_width=True, height=300)

    st.markdown("---")

    # ── Compound search ────────────────────────────────────────────────────
    with st.expander("Search compounds"):
        query = st.text_input("Compound name contains…")
        result = (
            filtered_cf[filtered_cf["compound_name"].str.contains(query, case=False, na=False)]
            if query else filtered_cf
        )
        total_media_n = media_compounds["medium_id"].nunique()
        result = result.copy()
        result["% of all media"] = (result["media_count"] / total_media_n * 100).round(1)
        st.dataframe(result, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: STRAIN LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
with tab_strain:
    st.title("Strain Lookup")
    st.caption(
        "Search for a specific strain by ID, or browse strains in the current filter "
        "and click one to see its full profile."
    )

    col_search, col_browse = st.columns([1, 2])

    with col_search:
        strain_id_input = st.text_input("Enter strain ID", placeholder="e.g. 26")

    with col_browse:
        # Let user pick from a dropdown of filtered strains
        strain_options = filtered["strain_id"].astype(str).tolist()
        selected_from_list = st.selectbox(
            f"Or pick from filtered strains ({len(strain_options):,})",
            [""] + strain_options,
            key="strain_browse",
        )

    # Resolve which strain to show
    strain_id_to_show = None
    if strain_id_input.strip():
        try:
            strain_id_to_show = int(strain_id_input.strip())
        except ValueError:
            st.warning("Please enter a numeric strain ID.")
    elif selected_from_list:
        strain_id_to_show = int(selected_from_list)

    if strain_id_to_show is not None:
        row = strains[strains["strain_id"] == strain_id_to_show]
        if row.empty:
            st.error(f"Strain ID {strain_id_to_show} not found in the database.")
        else:
            row = row.iloc[0]

            st.markdown("---")
            st.subheader(f"Strain {strain_id_to_show}")

            # Taxonomy card
            col_tax, col_growth, col_o2 = st.columns(3)
            with col_tax:
                st.markdown("**Taxonomy**")
                st.markdown(f"- Phylum: `{row['phylum']}`")
                st.markdown(f"- Class: `{row['class']}`")
            with col_growth:
                st.markdown("**Growth Parameters**")
                st.markdown(f"- Temperature: {row['temp_min']}–{row['temp_max']}°C "
                            f"(optimum **{row['temp_opt']}°C**)")
                st.markdown(f"- pH: {row['ph_min']}–{row['ph_max']} "
                            f"(optimum **{row['ph_opt']}**)")
            with col_o2:
                st.markdown("**O₂ Tolerance**")
                st.markdown(f"- `{row['o2_tol']}`")

            st.markdown("---")

            # Media used by this strain
            strain_media_rows = strain_media[strain_media["strain_id"] == strain_id_to_show]
            st.markdown(f"**Compatible media ({len(strain_media_rows)})**")

            if strain_media_rows.empty:
                st.info("No media recorded for this strain.")
            else:
                media_ids = strain_media_rows["medium_id"].tolist()

                # Compound composition of all its media combined
                strain_cpds = (
                    media_compounds[media_compounds["medium_id"].isin(media_ids)]
                    .merge(compounds, on="compound_id", how="left")
                    .groupby(["compound_id", "compound_name"])["medium_id"]
                    .nunique()
                    .rename("media_using_compound")
                    .reset_index()
                    .sort_values("media_using_compound", ascending=False)
                )

                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown(f"Media IDs: {', '.join(map(str, media_ids[:20]))}"
                                + (" …" if len(media_ids) > 20 else ""))
                    st.markdown(f"Distinct compounds across all media: **{len(strain_cpds)}**")

                with col_r:
                    # Richness of each medium
                    med_rich = (
                        media_compounds[media_compounds["medium_id"].isin(media_ids)]
                        .groupby("medium_id")["compound_id"].nunique()
                        .reset_index(name="compounds")
                    )
                    fig = px.bar(
                        med_rich.sort_values("compounds", ascending=False),
                        x="medium_id", y="compounds",
                        labels={"medium_id": "Medium ID", "compounds": "# Compounds"},
                        title="Compounds per medium",
                        color_discrete_sequence=[PALETTE[0]],
                    )
                    fig.update_layout(xaxis_type="category", height=300)
                    st.plotly_chart(fig, use_container_width=True, key="sl_med_rich")

                # Top compounds bar
                fig2 = px.bar(
                    strain_cpds.head(20),
                    x="media_using_compound", y="compound_name", orientation="h",
                    labels={"media_using_compound": "Media containing compound",
                            "compound_name": "Compound"},
                    title="Top 20 compounds across this strain's media",
                    color_discrete_sequence=[PALETTE[2]],
                )
                fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
                st.plotly_chart(fig2, use_container_width=True, key="sl_cpd")

                # How does this strain compare to its phylum?
                st.markdown(f"**How does this strain compare to other *{row['phylum']}*?**")
                phylum_strains = strains[strains["phylum"] == row["phylum"]]

                col_l, col_r = st.columns(2)
                with col_l:
                    fig3 = px.histogram(
                        phylum_strains, x="temp_opt",
                        nbins=30, color_discrete_sequence=[PALETTE[0]],
                        title=f"Temp opt distribution in {row['phylum']}",
                        labels={"temp_opt": "Optimal temperature (°C)"},
                    )
                    fig3.add_vline(x=row["temp_opt"], line_dash="dash",
                                   line_color="red",
                                   annotation_text=f"This strain ({row['temp_opt']}°C)",
                                   annotation_position="top right")
                    st.plotly_chart(fig3, use_container_width=True, key="sl_temp")
                with col_r:
                    fig4 = px.histogram(
                        phylum_strains, x="ph_opt",
                        nbins=30, color_discrete_sequence=[PALETTE[2]],
                        title=f"pH opt distribution in {row['phylum']}",
                        labels={"ph_opt": "Optimal pH"},
                    )
                    fig4.add_vline(x=row["ph_opt"], line_dash="dash",
                                   line_color="red",
                                   annotation_text=f"This strain ({row['ph_opt']})",
                                   annotation_position="top right")
                    st.plotly_chart(fig4, use_container_width=True, key="sl_ph")
    else:
        st.info("Enter a strain ID or pick one from the dropdown above to see its profile.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.title("Data Explorer")
    st.caption("Browse and download any of the four underlying tables.")

    dataset_choice = st.selectbox(
        "Dataset",
        ["Strains (filtered)", "Strain–Media links", "Media–Compound links", "Compounds"],
    )

    if dataset_choice == "Strains (filtered)":
        df_show = filtered
    elif dataset_choice == "Strain–Media links":
        df_show = strain_media[strain_media["strain_id"].isin(filtered["strain_id"])]
    elif dataset_choice == "Media–Compound links":
        media_in_filter = strain_media[
            strain_media["strain_id"].isin(filtered["strain_id"])
        ]["medium_id"]
        df_show = media_compounds[media_compounds["medium_id"].isin(media_in_filter)]
    else:
        df_show = compounds

    st.caption(f"{len(df_show):,} rows × {df_show.shape[1]} columns")
    st.dataframe(df_show, use_container_width=True, height=500)

    csv = df_show.to_csv(index=False).encode()
    st.download_button(
        "Download as CSV",
        data=csv,
        file_name=f"{dataset_choice.lower().replace(' ', '_').replace('–','')}.csv",
        mime="text/csv",
    )
