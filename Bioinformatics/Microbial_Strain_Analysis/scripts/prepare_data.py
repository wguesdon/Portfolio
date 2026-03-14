#!/usr/bin/env python3
"""
Prepare and validate all datasets.

Usage:
    uv run python scripts/prepare_data.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from microbial.data.preprocessor import get_data
from microbial.analysis.stats import (
    missing_summary, numeric_summary, phylum_summary,
    o2_summary, top_compounds,
)


def main():
    print("Loading and processing data...")
    data = get_data()

    strains = data["strains"]
    compound_freq = data["compound_freq"]
    media_richness = data["media_richness"]
    strain_compounds = data["strain_compounds"]

    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Strains (clean):       {len(strains):,}")
    print(f"Phyla:                 {strains['phylum'].nunique()}")
    print(f"Classes:               {strains['class'].nunique()}")
    print(f"Media:                 {data['strain_media']['medium_id'].nunique():,}")
    print(f"Compounds:             {data['compounds']['compound_id'].nunique():,}")
    print(f"Strain-media links:    {len(data['strain_media']):,}")
    print(f"Media-compound links:  {len(data['media_compounds']):,}")

    print(f"\n{'='*50}")
    print("MISSING VALUES (raw strains)")
    print(f"{'='*50}")
    raw_strains = __import__(
        "microbial.data.loader", fromlist=["load_strains"]
    ).load_strains()
    ms = missing_summary(raw_strains)
    print(ms.to_string() if not ms.empty else "No missing values.")

    print(f"\n{'='*50}")
    print("NUMERIC SUMMARY (clean strains)")
    print(f"{'='*50}")
    print(numeric_summary(strains).to_string())

    print(f"\n{'='*50}")
    print("O₂ TOLERANCE")
    print(f"{'='*50}")
    print(o2_summary(strains).to_string(index=False))

    print(f"\n{'='*50}")
    print("TOP 10 PHYLA")
    print(f"{'='*50}")
    print(phylum_summary(strains).head(10).to_string(index=False))

    print(f"\n{'='*50}")
    print("TOP 10 COMPOUNDS")
    print(f"{'='*50}")
    print(top_compounds(compound_freq, 10).to_string(index=False))

    print("\nDone. All data validated successfully.")


if __name__ == "__main__":
    main()
