import numpy as np

corr = correlation_matrix(strains)
correlation_heatmap(corr).show()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
pairs = (
    corr.where(mask)
    .stack()
    .reset_index()
    .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"})
    .assign(abs_r=lambda d: d["r"].abs())
    .sort_values("abs_r", ascending=False)
    .head(8)
)
print("Strongest correlations:")
display(pairs[["var1", "var2", "r"]])  # noqa: F821
