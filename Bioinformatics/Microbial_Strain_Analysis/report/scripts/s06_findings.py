phylum_counts = phylum_summary(strains)
o2_counts = o2_summary(strains)
mps = media_per_strain(strain_media)

aerobe_pct = o2_counts.loc[o2_counts["o2_tol"] == "aerobe", "pct"].values[0]
unique_cpd_pct = (compound_freq["media_count"] == 1).mean() * 100

findings = [
    ("Taxonomic dominance",
     f"Proteobacteria account for {phylum_counts.iloc[0]['pct']:.1f}% of strains; "
     f"the top 3 phyla cover {phylum_counts.head(3)['pct'].sum():.1f}% of the dataset."),
    ("Aerobic bias",
     f"{aerobe_pct:.1f}% of strains are aerobic, reflecting historical cultivation bias."),
    ("Temperature niches",
     "Clear separation between mesophiles (~37°C) and thermophiles (>55°C). "
     "Anaerobes show a slightly lower mean optimum temperature."),
    ("pH breadth",
     "Aerobic strains have broader pH tolerance on average. "
     "Extreme acidophiles and alkaliphiles are rare but present."),
    ("Media complexity",
     f"Median medium contains {media_richness['compound_count'].median():.0f} compounds. "
     "A small set of ubiquitous compounds (Distilled water, Yeast extract, NaCl) "
     "appear in the vast majority of media."),
    ("Compound specificity",
     f"{unique_cpd_pct:.1f}% of compounds are unique to a single medium, "
     "suggesting high media specificity."),
]

from IPython.display import display, Markdown

md = "\n".join(f"**{title}**\n\n{detail}\n" for title, detail in findings)
display(Markdown(md))
