"""Create a manual audit CSV template from a sampled grid GeoPackage."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


AUDIT_COLUMNS = [
    "audit_label",
    "label_quality",
    "dominant_fabric_note",
    "dsm_note",
    "vegetation_note",
    "network_note",
    "audit_source_layers",
    "audit_reviewer",
    "audit_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-gpkg", required=True)
    parser.add_argument("--layer", default="audit_sample")
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gdf = gpd.read_file(args.sample_gpkg, layer=args.layer)
    keep = [
        "grid_id",
        "district",
        "built_intensity_bin",
        "building_count_prelim",
        "building_coverage_prelim",
        "road_density_prelim_m_per_km2",
        "lcz_weak_label",
        "lcz_weak_confidence",
        "lcz_probability_mean",
        "lcz_mixed_flag",
        "sample_stratum",
        "sample_role",
    ]
    df = gdf[[column for column in keep if column in gdf.columns]].copy()
    for column in AUDIT_COLUMNS:
        df[column] = ""
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} out={out_csv}")


if __name__ == "__main__":
    main()
