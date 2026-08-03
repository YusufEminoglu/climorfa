"""Build coastline distance and coastal-band variables for grid cells."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from osgeo import ogr, osr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shoreline-gpkg", required=True, help="Path to shoreline GeoPackage (e.g. from OSM coastline download)")
    parser.add_argument("--shoreline-layer", default="lines")
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--out-shoreline-gpkg", default="data/02_interim/shoreline_epsg5253.gpkg")
    parser.add_argument("--out-shoreline-layer", default="shoreline_epsg5253")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_coastal_context.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_coastal_context")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_coastal_context.csv")
    parser.add_argument("--assumed-epsg", type=int, default=5253)
    return parser.parse_args()


def make_srs(epsg: int) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def copy_shoreline_with_srs(
    source_layer: ogr.Layer,
    out_path: Path,
    out_layer_name: str,
    srs: osr.SpatialReference,
    source_path: str,
) -> list[ogr.Geometry]:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        ds = driver.Open(str(out_path), update=1)
        if ds is None:
            raise RuntimeError(f"Could not open existing {out_path}")
        existing = ds.GetLayerByName(out_layer_name)
        if existing is not None:
            ds.DeleteLayer(out_layer_name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds = driver.CreateDataSource(str(out_path))
    if ds is None:
        raise RuntimeError(f"Could not create {out_path}")

    out_layer = ds.CreateLayer(out_layer_name, srs=srs, geom_type=ogr.wkbMultiLineString)
    out_layer.CreateField(ogr.FieldDefn("source", ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn("crs_note", ogr.OFTString))
    out_layer.CreateField(ogr.FieldDefn("src_fid", ogr.OFTInteger))

    out_defn = out_layer.GetLayerDefn()
    geometries: list[ogr.Geometry] = []
    source_layer.ResetReading()
    for feature in source_layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom_clone = geom.Clone()
        geometries.append(geom_clone.Clone())

        out_feature = ogr.Feature(out_defn)
        out_feature.SetField("source", source_path)
        out_feature.SetField("crs_note", "input CRS metadata missing; coordinates assumed EPSG:5253")
        out_feature.SetField("src_fid", int(feature.GetFID()))
        out_feature.SetGeometry(geom_clone)
        out_layer.CreateFeature(out_feature)
        out_feature = None

    out_layer.SyncToDisk()
    ds = None
    return geometries


def band_from_distance(distance_m: float) -> str:
    if distance_m <= 0:
        return "shoreline_cell"
    if distance_m <= 500:
        return "coastal_0_500m"
    if distance_m <= 1000:
        return "coastal_500_1000m"
    if distance_m <= 2000:
        return "coastal_1_2km"
    if distance_m <= 5000:
        return "transition_2_5km"
    return "inland_5km_plus"


def min_distance(geom: ogr.Geometry, coast_geoms: list[ogr.Geometry]) -> float:
    return min(geom.Distance(coast) for coast in coast_geoms)


def build_stats(grid_layer: ogr.Layer, coast_geoms: list[ogr.Geometry]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    for feature in grid_layer:
        grid_id = feature.GetField("grid_id")
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom_clone = geom.Clone()
        centroid = geom_clone.Centroid()
        cell_min_distance = float(min_distance(geom_clone, coast_geoms))
        centroid_distance = float(min_distance(centroid, coast_geoms))
        stats[grid_id] = {
            "coast_min_distance_m": cell_min_distance,
            "coast_centroid_distance_m": centroid_distance,
            "coast_intersects_flag": int(cell_min_distance <= 0.001),
            "coastal_2km_flag": int(cell_min_distance <= 2000),
            "coastal_5km_flag": int(cell_min_distance <= 5000),
            "coastal_band": band_from_distance(cell_min_distance),
        }
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if isinstance(value, float):
        return ogr.FieldDefn(name, ogr.OFTReal)
    field = ogr.FieldDefn(name, ogr.OFTString)
    field.SetWidth(40)
    return field


def write_grid_output(
    grid_layer: ogr.Layer,
    stats: dict[str, dict[str, Any]],
    out_path: Path,
    out_layer_name: str,
) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        out_ds = driver.Open(str(out_path), update=1)
        if out_ds is None:
            raise RuntimeError(f"Could not open existing output {out_path}")
        existing = out_ds.GetLayerByName(out_layer_name)
        if existing is not None:
            out_ds.DeleteLayer(out_layer_name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")

    out_layer = out_ds.CreateLayer(out_layer_name, srs=grid_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
    in_defn = grid_layer.GetLayerDefn()
    for i in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(i))

    fields = list(next(iter(stats.values())).keys())
    sample = next(iter(stats.values()))
    for field in fields:
        out_layer.CreateField(create_field(field, sample[field]))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        out_feature = ogr.Feature(out_defn)
        for i in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(i).GetName(), feature.GetField(i))
        grid_id = feature.GetField("grid_id")
        row = stats.get(grid_id, {})
        for field in fields:
            value = row.get(field)
            if value is None:
                continue
            out_feature.SetField(field, value)
        geom = feature.GetGeometryRef()
        if geom is not None:
            out_feature.SetGeometry(geom.Clone())
        out_layer.CreateFeature(out_feature)
        out_feature = None
    out_layer.SyncToDisk()
    out_ds = None


def write_csv(out_path: Path, stats: dict[str, dict[str, Any]]) -> None:
    fields = list(next(iter(stats.values())).keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_id", *fields])
        writer.writeheader()
        for grid_id, row in stats.items():
            writer.writerow({"grid_id": grid_id, **row})


def main() -> None:
    args = parse_args()
    assumed_srs = make_srs(args.assumed_epsg)
    shoreline_ds, shoreline_layer = open_layer(args.shoreline_gpkg, args.shoreline_layer)
    grid_ds, grid_layer = open_layer(args.grid_gpkg, args.grid_layer)

    coast_geoms = copy_shoreline_with_srs(
        source_layer=shoreline_layer,
        out_path=Path(args.out_shoreline_gpkg),
        out_layer_name=args.out_shoreline_layer,
        srs=assumed_srs,
        source_path=args.shoreline_gpkg,
    )
    if not coast_geoms:
        raise RuntimeError("No coastline geometries found")

    stats = build_stats(grid_layer, coast_geoms)
    write_grid_output(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    bands: dict[str, int] = {}
    for row in stats.values():
        bands[row["coastal_band"]] = bands.get(row["coastal_band"], 0) + 1

    print(f"shoreline_features={len(coast_geoms)}")
    print(f"grid_cells={len(stats)}")
    print(f"bands={bands}")
    print(f"out_shoreline={args.out_shoreline_gpkg}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    shoreline_ds = None
    grid_ds = None


if __name__ == "__main__":
    main()
