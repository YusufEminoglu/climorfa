"""Convert Mert's edited audit GeoPackage back into the CSV shape that
validate_manual_audit.py expects, for the manual_audit_sheet.csv slot."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd


FIXED_FIELDS = [
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

AUDIT_FIELDS = [
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
    parser.add_argument("--edited-gpkg", required=True)
    parser.add_argument("--edited-layer", default="audit_sample_v1_editable")
    parser.add_argument("--out-csv", default="data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gdf = gpd.read_file(args.edited_gpkg, layer=args.edited_layer)
    df = gdf[[column for column in [*FIXED_FIELDS, *AUDIT_FIELDS] if column in gdf.columns]].copy()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} out={out_csv}")


if __name__ == "__main__":
    main()
