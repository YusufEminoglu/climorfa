"""Aggregate footprint morphometrics and open-space fragmentation to grid cells."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable

from osgeo import ogr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--buildings-gpkg", default="data/02_interim/buildings_roads_fua.gpkg")
    parser.add_argument("--buildings-layer", default="buildings_fua")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_building_morphometrics.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_building_morphometrics")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_building_morphometrics.csv")
    parser.add_argument("--min-open-patch-area-m2", type=float, default=25.0)
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


def safe_difference(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        diff = a.Difference(b)
    except RuntimeError:
        diff = a.MakeValid().Difference(b.MakeValid())
    if diff is None or diff.IsEmpty():
        return None
    return diff


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
            collection.AddGeometry(part)
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


def gini(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value > 0)
    n = len(clean)
    if n == 0:
        return None
    total = sum(clean)
    if total <= 0:
        return None
    weighted = sum((index + 1) * value for index, value in enumerate(clean))
    return (2 * weighted) / (n * total) - (n + 1) / n


def weighted_mean_std(values: list[float], weights: list[float]) -> tuple[float | None, float | None]:
    total_weight = sum(weights)
    if total_weight <= 0:
        return None, None
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    variance = max(sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight, 0.0)
    return mean, math.sqrt(variance)


def empty_row() -> dict[str, Any]:
    return {
        "morph_building_count": 0,
        "morph_building_area_m2": 0.0,
        "morph_building_area_share": 0.0,
        "morph_building_perimeter_m": 0.0,
        "morph_building_perimeter_density_m_per_ha": 0.0,
        "morph_mean_building_area_m2": None,
        "morph_median_building_area_m2": None,
        "morph_building_area_cv": None,
        "morph_building_area_gini": None,
        "morph_building_compactness_aw_mean": None,
        "morph_building_compactness_aw_std": None,
        "morph_open_space_area_m2": 0.0,
        "morph_open_space_share": 0.0,
        "morph_open_space_patch_count": 0,
        "morph_open_space_patch_density_per_ha": 0.0,
        "morph_open_space_largest_patch_m2": 0.0,
        "morph_open_space_largest_patch_share": None,
        "morph_open_space_fragmentation_index": 0.0,
    }


def compactness(area: float, perimeter: float) -> float | None:
    if area <= 0 or perimeter <= 0:
        return None
    return 4.0 * math.pi * area / (perimeter * perimeter)


def finalize_row(
    row: dict[str, Any],
    cell_geom: ogr.Geometry,
    cell_area_m2: float,
    building_geoms: list[ogr.Geometry],
    building_areas: list[float],
    building_perimeters: list[float],
    min_open_patch_area_m2: float,
) -> dict[str, Any]:
    row["morph_building_count"] = len(building_areas)
    row["morph_building_area_m2"] = sum(building_areas)
    row["morph_building_area_share"] = row["morph_building_area_m2"] / cell_area_m2 if cell_area_m2 else 0.0
    row["morph_building_perimeter_m"] = sum(building_perimeters)
    row["morph_building_perimeter_density_m_per_ha"] = (
        row["morph_building_perimeter_m"] / (cell_area_m2 / 10_000.0) if cell_area_m2 else 0.0
    )

    if building_areas:
        sorted_areas = sorted(building_areas)
        mean_area = sum(sorted_areas) / len(sorted_areas)
        row["morph_mean_building_area_m2"] = mean_area
        mid = len(sorted_areas) // 2
        if len(sorted_areas) % 2:
            row["morph_median_building_area_m2"] = sorted_areas[mid]
        else:
            row["morph_median_building_area_m2"] = (sorted_areas[mid - 1] + sorted_areas[mid]) / 2.0
        if mean_area > 0 and len(sorted_areas) > 1:
            variance = sum((area - mean_area) ** 2 for area in sorted_areas) / len(sorted_areas)
            row["morph_building_area_cv"] = math.sqrt(variance) / mean_area
        row["morph_building_area_gini"] = gini(sorted_areas)

        comp_values: list[float] = []
        comp_weights: list[float] = []
        for area, perimeter in zip(building_areas, building_perimeters):
            comp = compactness(area, perimeter)
            if comp is not None:
                comp_values.append(comp)
                comp_weights.append(area)
        mean_comp, std_comp = weighted_mean_std(comp_values, comp_weights)
        row["morph_building_compactness_aw_mean"] = mean_comp
        row["morph_building_compactness_aw_std"] = std_comp

    building_union = geometry_union(building_geoms)
    open_geom = safe_difference(cell_geom, building_union) if building_union is not None else cell_geom.Clone()
    patch_areas = [
        float(part.Area())
        for part in iter_polygon_parts(open_geom)
        if float(part.Area()) >= min_open_patch_area_m2
    ]
    open_area = sum(patch_areas)
    row["morph_open_space_area_m2"] = open_area
    row["morph_open_space_share"] = open_area / cell_area_m2 if cell_area_m2 else 0.0
    row["morph_open_space_patch_count"] = len(patch_areas)
    row["morph_open_space_patch_density_per_ha"] = len(patch_areas) / (cell_area_m2 / 10_000.0) if cell_area_m2 else 0.0
    if patch_areas:
        largest = max(patch_areas)
        row["morph_open_space_largest_patch_m2"] = largest
        row["morph_open_space_largest_patch_share"] = largest / open_area if open_area > 0 else None
        row["morph_open_space_fragmentation_index"] = (
            len(patch_areas) * (1.0 - row["morph_open_space_largest_patch_share"])
            if row["morph_open_space_largest_patch_share"] is not None
            else 0.0
        )
    return row


def aggregate(
    grid_layer: ogr.Layer,
    buildings_layer: ogr.Layer,
    min_open_patch_area_m2: float,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for index, grid_feature in enumerate(grid_layer, start=1):
        if index % 1000 == 0:
            print(f"processed_morph_grid_cells={index}/{total}")
        grid_id = str(grid_feature.GetField("grid_id"))
        cell_geom_ref = grid_feature.GetGeometryRef()
        if cell_geom_ref is None or cell_geom_ref.IsEmpty():
            continue
        cell_geom = cell_geom_ref.Clone()
        cell_area_m2 = float(grid_feature.GetField("cell_area_m2") or cell_geom.Area())
        row = empty_row()
        building_geoms: list[ogr.Geometry] = []
        building_areas: list[float] = []
        building_perimeters: list[float] = []

        buildings_layer.SetSpatialFilter(cell_geom)
        buildings_layer.ResetReading()
        for building_feature in buildings_layer:
            building_geom = building_feature.GetGeometryRef()
            if building_geom is None or building_geom.IsEmpty():
                continue
            inter = safe_intersection(cell_geom, building_geom)
            if inter is None:
                continue
            area = float(inter.Area())
            if area <= 0:
                continue
            perimeter = float(inter.Boundary().Length())
            building_geoms.append(inter.Clone())
            building_areas.append(area)
            building_perimeters.append(perimeter)
        buildings_layer.SetSpatialFilter(None)

        stats[grid_id] = finalize_row(
            row=row,
            cell_geom=cell_geom,
            cell_area_m2=cell_area_m2,
            building_geoms=building_geoms,
            building_areas=building_areas,
            building_perimeters=building_perimeters,
            min_open_patch_area_m2=min_open_patch_area_m2,
        )
    grid_layer.ResetReading()
    buildings_layer.ResetReading()
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if name.endswith("_count"):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if isinstance(value, (int, float)) or value is None:
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

    stats = aggregate(
        grid_layer=grid_layer,
        buildings_layer=buildings_layer,
        min_open_patch_area_m2=args.min_open_patch_area_m2,
    )
    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    built_cells = sum(1 for row in stats.values() if row["morph_building_count"] > 0)
    fragmented_open_cells = sum(1 for row in stats.values() if row["morph_open_space_patch_count"] > 1)
    print(f"grid_cells={len(stats)}")
    print(f"built_cells={built_cells}")
    print(f"fragmented_open_space_cells={fragmented_open_cells}")
    print(f"min_open_patch_area_m2={args.min_open_patch_area_m2}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    buildings_ds = None


if __name__ == "__main__":
    main()
