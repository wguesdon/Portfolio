import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import plotly.io as pio

from microbial.data.preprocessor import get_data
from microbial.analysis.stats import (
    missing_summary, numeric_summary,
    phylum_summary, class_summary, o2_summary,
    correlation_matrix, top_compounds,
    strains_per_phylum_o2, media_per_strain,
)
from microbial.viz.plots import (
    phylum_bar, class_treemap, o2_pie,
    temp_violin, ph_violin, temp_ph_scatter,
    growth_range_box, correlation_heatmap,
    compound_freq_bar, media_richness_hist,
    media_per_strain_hist, phylum_o2_heatmap,
)

data = get_data()
strains         = data["strains"]
strain_media    = data["strain_media"]
media_compounds = data["media_compounds"]
compounds       = data["compounds"]
swm             = data["strains_with_media"]
media_richness  = data["media_richness"]
compound_freq   = data["compound_freq"]
