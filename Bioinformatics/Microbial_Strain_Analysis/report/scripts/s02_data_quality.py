from microbial.data.loader import load_strains

raw = load_strains()
n_raw = len(raw)
n_clean = len(strains)
removed = n_raw - n_clean

print(f"Raw rows:  {n_raw:,}")
print(f"Retained:  {n_clean:,}  ({n_clean/n_raw*100:.1f}%)")
print(f"Removed:   {removed:,}  ({removed/n_raw*100:.1f}%) — incomplete growth parameters\n")

ms = missing_summary(raw)
display(ms if not ms.empty else "No missing values.")  # noqa: F821

print("\nNumeric summary (clean strains):")
display(numeric_summary(strains))  # noqa: F821
