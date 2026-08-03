"""Aggregate explicit-motorway road-class sensitivity around grid centroids."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from osgeo import ogr

ogr.UseExceptions()


DEFAULT_RADII = [250.0, 400.0, 800.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--roads-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--road-type-field", default="YOLTIP")
    parser.add_argument("--motorway-value", default="OTOYOL")
    parser.add_argument("--study-area-gpkg", default="data/02_interim/study_area_fua.gpkg")
    parser.add_argument("--study-area-layer", default="study_area_fua")
    parser.add_argument(
        "--out-gpkg",
        default="data/03_processed/grid_road_network_reach_road_class_sensitivity.gpkg",
    )
    parser.add_argument(
        "--out-layer",
        default="grid_250m_road_network_reach_road_class_sensitivity",
    )
    parser.add_argument(
        "--out-csv",
        default="data/03_processed/grid_250m_road_network_reach_road_class_sensitivity.csv",
    )
    parser.add_argument("--radii-m", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument("--buffer-segments", type=int, default=24)
    return parser.parse_args()


def radius_label(radius_m: float) -> str:
    if float(radius_m).is_integer():
        return f"{int(radius_m)}m"
    return f"{str(radius_m).replace('.', 'p')}m"


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def read_union_geometry(layer: ogr.Layer) -> ogr.Geometry:
    union_geom: ogr.Geometry | None = None
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        cloned = geom.Clone()
        union_geom = cloned if union_geom is None else union_geom.Union(cloned)
    layer.ResetReading()
    if union_geom is None or union_geom.IsEmpty():
        raise RuntimeError("Study area geometry is empty")
    return union_geom


def safe_intersection(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        inter = a.Intersection(b)
    except RuntimeError:
        inter = a.MakeValid().Intersection(b.MakeValid())
    if inter is None or inter.IsEmpty():
        return None
    return inter


def empty_metrics(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_road_type_known_length_m": 0.0,
        f"{prefix}_motorway_length_m": 0.0,
        f"{prefix}_motorway_density_m_per_km2": 0.0,
        f"{prefix}_road_excl_motorway_length_m": 0.0,
        f"{prefix}_road_excl_motorway_density_m_per_km2": 0.0,
        f"{prefix}_motorway_share_of_road_length": None,
    }


def aggregate_buffer(
    roads_layer: ogr.Layer,
    context_geom: ogr.Geometry,
    prefix: str,
    road_type_field: str,
    motorway_value: str,
) -> dict[str, Any]:
    row = empty_metrics(prefix)
    context_area = float(context_geom.Area()) if context_geom is not None and not context_geom.IsEmpty() else 0.0
    if context_area <= 0:
        return row

    total_length = 0.0
    roads_layer.SetSpatialFilter(context_geom)
    roads_layer.ResetReading()
    for road_feature in roads_layer:
        road_geom = road_feature.GetGeometryRef()
        if road_geom is None or road_geom.IsEmpty():
            continue
        inter = safe_intersection(context_geom, road_geom)
        if inter is None:
            continue
        length = float(inter.Length())
        if length <= 0:
            continue
        total_length += length
        road_type_raw = road_feature.GetField(road_type_field)
        road_type = str(road_type_raw).strip().upper() if road_type_raw not in (None, "") else ""
        if road_type:
            row[f"{prefix}_road_type_known_length_m"] += length
        if road_type == motorway_value.strip().upper():
            row[f"{prefix}_motorway_length_m"] += length
        else:
            row[f"{prefix}_road_excl_motorway_length_m"] += length
    roads_layer.SetSpatialFilter(None)
    roads_layer.ResetReading()

    km2 = context_area / 1_000_000.0
    row[f"{prefix}_motorway_density_m_per_km2"] = row[f"{prefix}_motorway_length_m"] / km2
    row[f"{prefix}_road_excl_motorway_density_m_per_km2"] = (
        row[f"{prefix}_road_excl_motorway_length_m"] / km2
    )
    if total_length > 0:
        row[f"{prefix}_motorway_share_of_road_length"] = row[f"{prefix}_motorway_length_m"] / total_length
    return row


def aggregate(
    grid_layer: ogr.Layer,
    roads_layer: ogr.Layer,
    study_area_geom: ogr.Geometry,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for index, grid_feature in enumerate(grid_layer, start=1):
        if index % 500 == 0:
            print(f"processed_road_class_sensitivity_cells={index}/{total}")
        grid_id = str(grid_feature.GetField("grid_id"))
        geom = grid_feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        centroid = geom.Centroid()
        row: dict[str, Any] = {}
        for radius_m in args.radii_m:
            prefix = f"network_reach_{radius_label(radius_m)}"
            buffer_geom = centroid.Buffer(float(radius_m), int(args.buffer_segments))
            context_geom = safe_intersection(buffer_geom, study_area_geom)
            if context_geom is None:
                context_geom = ogr.Geometry(ogr.wkbPolygon)
            row.update(
                aggregate_buffer(
                    roads_layer=roads_layer,
                    context_geom=context_geom,
                    prefix=prefix,
                    road_type_field=args.road_type_field,
                    motorway_value=args.motorway_value,
                )
            )
        stats[grid_id] = row
    grid_layer.ResetReading()
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if isinstance(value, float) or value is None:
        return ogr.FieldDefn(name, ogr.OFTReal)
    field = ogr.FieldDefn(name, ogr.OFTString)
    field.SetWidth(80)
    return field


def write_gpkg(grid_layer: ogr.Layer, stats: dict[str, dict[str, Any]], out_path: Path, out_layer_name: str) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        driver.DeleteDataSource(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")

    out_layer = out_ds.CreateLayer(out_layer_name, srs=grid_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
    in_defn = grid_layer.GetLayerDefn()
    for index in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(index))

    fields = list(next(iter(stats.values())).keys())
    sample_values: dict[str, Any] = {}
    for field in fields:
        for row in stats.values():
            value = row.get(field)
            if value is not None:
                sample_values[field] = value
                break
    for field in fields:
        out_layer.CreateField(create_field(field, sample_values.get(field)))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        out_feature = ogr.Feature(out_defn)
        for index in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(index).GetName(), feature.GetField(index))
        grid_id = str(feature.GetField("grid_id"))
        for field in fields:
            value = stats.get(grid_id, {}).get(field)
            if value is not None:
                out_feature.SetField(field, value)
        geom = feature.GetGeometryRef()
        if geom is not None:
            out_feature.SetGeometry(geom.Clone())
        out_layer.CreateFeature(out_feature)
        out_feature = None
    out_layer.SyncToDisk()
    out_ds = None
    grid_layer.ResetReading()


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
    grid_ds, grid_layer = open_layer(args.grid_gpkg, args.grid_layer)
    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)
    study_ds, study_layer = open_layer(args.study_area_gpkg, args.study_area_layer)

    road_defn = roads_layer.GetLayerDefn()
    road_fields = {road_defn.GetFieldDefn(index).GetName() for index in range(road_defn.GetFieldCount())}
    if args.road_type_field not in road_fields:
        raise RuntimeError(f"Road type field not found: {args.road_type_field}")

    study_area_geom = read_union_geometry(study_layer)
    stats = aggregate(grid_layer, roads_layer, study_area_geom, args)
    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    print(f"grid_cells={len(stats)}")
    print(f"road_type_field={args.road_type_field}")
    print(f"motorway_value={args.motorway_value}")
    for radius_m in args.radii_m:
        prefix = f"network_reach_{radius_label(radius_m)}"
        motorway_length = sum(float(row[f"{prefix}_motorway_length_m"]) for row in stats.values())
        excl_length = sum(float(row[f"{prefix}_road_excl_motorway_length_m"]) for row in stats.values())
        motorway_cells = sum(1 for row in stats.values() if float(row[f"{prefix}_motorway_length_m"]) > 0)
        print(f"{prefix}_motorway_cells={motorway_cells}")
        print(f"{prefix}_summed_motorway_length_m={motorway_length:.2f}")
        print(f"{prefix}_summed_road_excl_motorway_length_m={excl_length:.2f}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    roads_ds = None
    study_ds = None


if __name__ == "__main__":
    main()
