import pandas as pd

summary = {
    "Strains (after cleaning)": len(strains),
    "Phyla": strains["phylum"].nunique(),
    "Classes": strains["class"].nunique(),
    "O₂ tolerance categories": strains["o2_tol"].nunique(),
    "Total media": strain_media["medium_id"].nunique(),
    "Total compounds": compounds["compound_id"].nunique(),
    "Strain–media links": len(strain_media),
    "Media–compound links": len(media_compounds),
}
pd.Series(summary).to_frame("Value")
