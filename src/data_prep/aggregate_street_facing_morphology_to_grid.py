"""Aggregate street-facing morphology proxy metrics to 250 m grid cells."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

from osgeo import ogr

ogr.UseExceptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--buildings-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--buildings-layer", default="buildings_fua")
    parser.add_argument("--roads-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_street_facing_morphology.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_street_facing_morphology")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_street_facing_morphology.csv")
    parser.add_argument("--street-buffer-m", type=float, default=20.0)
    return parser.parse_args()


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def safe_intersection(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        inter = a.Intersection(b)
    except RuntimeError:
        inter = a.MakeValid().Intersection(b.MakeValid())
    if inter is None or inter.IsEmpty():
        return None
    return inter


def iter_line_parts(geom: ogr.Geometry | None) -> Iterable[ogr.Geometry]:
    if geom is None or geom.IsEmpty():
        return
    geom_name = geom.GetGeometryName().upper()
    if geom_name in {"LINESTRING", "LINEARRING"}:
        yield geom
        return
    if geom_name in {"MULTILINESTRING", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_line_parts(child)


def iter_polygon_parts(geom: ogr.Geometry | None) -> Iterable[ogr.Geometry]:
    if geom is None or geom.IsEmpty():
        return
    geom_name = geom.GetGeometryName().upper()
    if geom_name in {"POLYGON", "CURVEPOLYGON"}:
        yield geom
        return
    if geom_name in {"MULTIPOLYGON", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_polygon_parts(child)


def geometry_union(geoms: list[ogr.Geometry]) -> ogr.Geometry | None:
    if not geoms:
        return None
    collection = ogr.Geometry(ogr.wkbMultiPolygon)
    for geom in geoms:
        for part in iter_polygon_parts(geom):
            collection.AddGeometry(part.Clone())
    if collection.GetGeometryCount() == 0:
        return None
    try:
        union = collection.UnionCascaded()
    except (AttributeError, RuntimeError):
        union = geoms[0].Clone()
        for geom in geoms[1:]:
            union = union.Union(geom)
    if union is None or union.IsEmpty():
        return None
    return union


def polygon_area(geom: ogr.Geometry | None) -> float:
    return sum(float(part.Area()) for part in iter_polygon_parts(geom))


def empty_row(street_buffer_m: float) -> dict[str, Any]:
    return {
        "street_frontage_buffer_m": street_buffer_m,
        "street_frontage_road_length_m": 0.0,
        "street_frontage_buffer_area_m2": 0.0,
        "street_frontage_buffer_coverage_share": 0.0,
        "street_frontage_building_count": 0,
        "street_frontage_building_area_m2": 0.0,
        "street_frontage_building_area_share_of_buffer": None,
        "street_frontage_open_buffer_share": None,
        "street_frontage_building_edge_m": 0.0,
        "street_frontage_edge_m_per_road_km": None,
        "street_frontage_continuity_proxy": None,
        "street_frontage_gap_proxy": None,
        "street_frontage_building_density_per_buffer_ha": None,
    }


def road_context_for_cell(
    cell_geom: ogr.Geometry,
    roads_layer: ogr.Layer,
    street_buffer_m: float,
) -> tuple[float, ogr.Geometry | None]:
    road_length_m = 0.0
    road_buffers: list[ogr.Geometry] = []
    roads_layer.SetSpatialFilter(cell_geom)
    roads_layer.ResetReading()
    for road_feature in roads_layer:
        road_geom = road_feature.GetGeometryRef()
        if road_geom is None or road_geom.IsEmpty():
            continue
        inter = safe_intersection(cell_geom, road_geom)
        if inter is None:
            continue
        length = float(inter.Length())
        if length <= 0:
            continue
        road_length_m += length
        for line in iter_line_parts(inter):
            if line.Length() <= 0:
                continue
            buffered = line.Buffer(street_buffer_m)
            if buffered is None or buffered.IsEmpty():
                continue
            clipped = safe_intersection(cell_geom, buffered)
            if clipped is not None:
                road_buffers.append(clipped)
    roads_layer.SetSpatialFilter(None)
    return road_length_m, geometry_union(road_buffers)


def building_context_for_street_buffer(
    street_buffer_geom: ogr.Geometry,
    buildings_layer: ogr.Layer,
) -> tuple[int, float, float]:
    building_count = 0
    building_area_m2 = 0.0
    building_edge_m = 0.0
    buildings_layer.SetSpatialFilter(street_buffer_geom)
    buildings_layer.ResetReading()
    for building_feature in buildings_layer:
        building_geom = building_feature.GetGeometryRef()
        if building_geom is None or building_geom.IsEmpty():
            continue
        building_in_buffer = safe_intersection(street_buffer_geom, building_geom)
        if building_in_buffer is None:
            continue
        area = polygon_area(building_in_buffer)
        if area <= 0:
            continue
        building_count += 1
        building_area_m2 += area
        for part in iter_polygon_parts(building_geom):
            boundary = part.Boundary()
            boundary_in_buffer = safe_intersection(street_buffer_geom, boundary)
            if boundary_in_buffer is not None:
                building_edge_m += float(boundary_in_buffer.Length())
    buildings_layer.SetSpatialFilter(None)
    return building_count, building_area_m2, building_edge_m


def finalize_row(
    row: dict[str, Any],
    cell_area_m2: float,
    road_length_m: float,
    street_buffer_area_m2: float,
    building_count: int,
    building_area_m2: float,
    building_edge_m: float,
) -> dict[str, Any]:
    row["street_frontage_road_length_m"] = road_length_m
    row["street_frontage_buffer_area_m2"] = street_buffer_area_m2
    row["street_frontage_buffer_coverage_share"] = street_buffer_area_m2 / cell_area_m2 if cell_area_m2 else 0.0
    row["street_frontage_building_count"] = building_count
    row["street_frontage_building_area_m2"] = building_area_m2
    row["street_frontage_building_edge_m"] = building_edge_m
    if street_buffer_area_m2 > 0:
        area_share = building_area_m2 / street_buffer_area_m2
        row["street_frontage_building_area_share_of_buffer"] = min(max(area_share, 0.0), 1.0)
        row["street_frontage_open_buffer_share"] = max(0.0, 1.0 - row["street_frontage_building_area_share_of_buffer"])
        row["street_frontage_building_density_per_buffer_ha"] = building_count / (street_buffer_area_m2 / 10_000.0)
    if road_length_m > 0:
        row["street_frontage_edge_m_per_road_km"] = building_edge_m / (road_length_m / 1000.0)
        continuity = building_edge_m / (2.0 * road_length_m)
        if math.isfinite(continuity):
            row["street_frontage_continuity_proxy"] = min(max(continuity, 0.0), 1.0)
            row["street_frontage_gap_proxy"] = 1.0 - row["street_frontage_continuity_proxy"]
    return row


def aggregate(
    grid_layer: ogr.Layer,
    buildings_layer: ogr.Layer,
    roads_layer: ogr.Layer,
    street_buffer_m: float,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for index, grid_feature in enumerate(grid_layer, start=1):
        if index % 1000 == 0:
            print(f"processed_street_facing_grid_cells={index}/{total}")
        grid_id = str(grid_feature.GetField("grid_id"))
        cell_geom_ref = grid_feature.GetGeometryRef()
        if cell_geom_ref is None or cell_geom_ref.IsEmpty():
            continue
        cell_geom = cell_geom_ref.Clone()
        cell_area_m2 = float(grid_feature.GetField("cell_area_m2") or cell_geom.Area())
        row = empty_row(street_buffer_m)

        road_length_m, street_buffer_geom = road_context_for_cell(cell_geom, roads_layer, street_buffer_m)
        if street_buffer_geom is None:
            stats[grid_id] = finalize_row(row, cell_area_m2, road_length_m, 0.0, 0, 0.0, 0.0)
            continue

        street_buffer_area_m2 = polygon_area(street_buffer_geom)
        building_count, building_area_m2, building_edge_m = building_context_for_street_buffer(
            street_buffer_geom,
            buildings_layer,
        )
        stats[grid_id] = finalize_row(
            row=row,
            cell_area_m2=cell_area_m2,
            road_length_m=road_length_m,
            street_buffer_area_m2=street_buffer_area_m2,
            building_count=building_count,
            building_area_m2=building_area_m2,
            building_edge_m=building_edge_m,
        )
    grid_layer.ResetReading()
    buildings_layer.ResetReading()
    roads_layer.ResetReading()
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if name.endswith("_count"):
        return ogr.FieldDefn(name, ogr.OFTInteger)
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
        row = stats.get(grid_id, {})
        for field in fields:
            value = row.get(field)
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
    buildings_ds, buildings_layer = open_layer(args.buildings_gpkg, args.buildings_layer)
    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)

    stats = aggregate(
        grid_layer=grid_layer,
        buildings_layer=buildings_layer,
        roads_layer=roads_layer,
        street_buffer_m=args.street_buffer_m,
    )
    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    road_cells = sum(1 for row in stats.values() if row["street_frontage_road_length_m"] > 0)
    street_building_cells = sum(1 for row in stats.values() if row["street_frontage_building_count"] > 0)
    total_road_length = sum(float(row["street_frontage_road_length_m"]) for row in stats.values())
    total_building_edge = sum(float(row["street_frontage_building_edge_m"]) for row in stats.values())
    print(f"grid_cells={len(stats)}")
    print(f"street_buffer_m={args.street_buffer_m}")
    print(f"road_cells={road_cells}")
    print(f"street_frontage_building_cells={street_building_cells}")
    print(f"street_frontage_road_length_m={total_road_length:.2f}")
    print(f"street_frontage_building_edge_m={total_building_edge:.2f}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    buildings_ds = None
    roads_ds = None


if __name__ == "__main__":
    main()
