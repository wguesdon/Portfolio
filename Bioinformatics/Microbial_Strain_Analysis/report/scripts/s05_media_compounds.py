mps = media_per_strain(strain_media)
mr = media_richness["compound_count"]

compound_freq_bar(compound_freq, top_n=25).show()
media_richness_hist(media_richness).show()
media_per_strain_hist(mps).show()

print(f"Median compounds per medium:  {mr.median():.0f}")
print(f"Mean compounds per medium:    {mr.mean():.1f}")
print(f"Max:  {mr.max()}   Min:  {mr.min()}")
print(f"\nStrains with 1 medium:  {(mps['media_count'] == 1).sum():,}")
print(f"Strains with >5 media:  {(mps['media_count'] > 5).sum():,}")

total_media = media_compounds["medium_id"].nunique()
cf = compound_freq.copy()
cf["pct_media"] = (cf["media_count"] / total_media * 100).round(1)
print("\nTop 10 most ubiquitous compounds:")
display(cf.head(10)[["compound_name", "media_count", "pct_media"]])  # noqa: F821
