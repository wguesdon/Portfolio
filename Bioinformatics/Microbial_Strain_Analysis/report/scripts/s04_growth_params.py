df_tmp = strains.copy()
df_tmp["temp_range"] = df_tmp["temp_max"] - df_tmp["temp_min"]
df_tmp["ph_range"] = df_tmp["ph_max"] - df_tmp["ph_min"]

temp_violin(strains).show()
ph_violin(strains).show()
temp_ph_scatter(strains).show()
growth_range_box(strains, top_phyla_n=10).show()

print("Temperature stats by O₂ tolerance:")
display(  # noqa: F821
    df_tmp.groupby("o2_tol")[["temp_min", "temp_opt", "temp_max", "temp_range"]]
    .agg(["mean", "median"]).round(2)
)

print("\npH stats by O₂ tolerance:")
display(  # noqa: F821
    df_tmp.groupby("o2_tol")[["ph_min", "ph_opt", "ph_max", "ph_range"]]
    .agg(["mean", "median"]).round(2)
)
