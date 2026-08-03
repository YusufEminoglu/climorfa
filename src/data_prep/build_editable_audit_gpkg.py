"""Add empty audit columns to the priority-queue GeoPackage so Mert can edit
the attribute table directly in QGIS instead of a separate CSV."""

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
    parser.add_argument("--priority-gpkg", default="data/04_training_labels/audit_sample_v1_priority_queue.gpkg")
    parser.add_argument("--priority-layer", default="audit_sample_v1_priority_queue")
    parser.add_argument("--out-gpkg", default="data/04_training_labels/audit_sample_v1_editable.gpkg")
    parser.add_argument("--out-layer", default="audit_sample_v1_editable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gdf = gpd.read_file(args.priority_gpkg, layer=args.priority_layer)
    for column in AUDIT_COLUMNS:
        gdf[column] = ""
    gdf = gdf.sort_values("audit_priority_rank").reset_index(drop=True)
    out_gpkg = Path(args.out_gpkg)
    if out_gpkg.exists():
        out_gpkg.unlink()
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, layer=args.out_layer, driver="GPKG")
    print(f"rows={len(gdf)} out={out_gpkg} layer={args.out_layer}")


if __name__ == "__main__":
    main()
