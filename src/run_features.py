"""
run_features.py — Academic feature computation

Loads kanombe_buildings.geojson, computes Project Illiya features with:
  - include_extended=True
  - include_orientation=True
  - include_multiscale=True

Then prints summary statistics for:
  - orientation_entropy
  - density_ratio

And saves the full output to kanombe_features_academic.csv
"""

import sys

import pandas as pd

from feature import compute_features, load_geojson


INPUT_GEOJSON = "kanombe_buildings.geojson"
OUTPUT_CSV = "kanombe_features_academic.csv"


def main() -> None:
    # Fail fast with a clear message if extraction didn't produce the GeoJSON.
    try:
        gdf = load_geojson(INPUT_GEOJSON)
    except FileNotFoundError:
        print(f"[run_features] ERROR: Missing {INPUT_GEOJSON}. Run kanombe_extraction.py first.")
        sys.exit(1)

    features = compute_features(
        gdf,
        include_extended=True,
        include_orientation=True,
        include_multiscale=True,
    )

    # Defensive checks so we fail clearly if upstream feature names change.
    needed_cols = ["orientation_entropy", "density_ratio"]
    missing = [c for c in needed_cols if c not in features.columns]
    if missing:
        print(f"[run_features] ERROR: Missing columns in output: {missing}")
        sys.exit(1)

    # Required prints.
    total_buildings = len(features)
    mean_orientation_entropy = float(features["orientation_entropy"].mean())
    std_density_ratio = float(features["density_ratio"].std())

    print(f"Total Buildings: {total_buildings}")
    print(f"Mean Orientation Entropy: {mean_orientation_entropy}")
    print(f"Standard Deviation of Density Ratio: {std_density_ratio}")

    # Print requested describe() stats specifically for the two columns.
    print("\nDescribe() — orientation_entropy & density_ratio")
    print(features[["orientation_entropy", "density_ratio"]].describe().to_string())

    # Save full dataset.
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

