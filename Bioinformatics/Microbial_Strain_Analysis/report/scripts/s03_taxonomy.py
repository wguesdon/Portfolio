phylum_counts = phylum_summary(strains)
o2_counts = o2_summary(strains)
phylum_o2 = strains_per_phylum_o2(swm)

print("Phylum count:", strains["phylum"].nunique())
top3 = phylum_counts.head(3)
print(f"Top 3 phyla cover {top3['pct'].sum():.1f}% of strains:")
display(top3)  # noqa: F821

phylum_bar(phylum_counts, top_n=15).show()
