"""Aggregate building footprint, floor-count, and height-proxy metrics to grid cells."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from osgeo import ogr


TYPE_FIELD_MAP = {
    "MESKEN": "mesken",
    "İŞYERİ": "isyeri",
    "ISYERI": "isyeri",
    "KAMU": "kamu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--buildings-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--buildings-layer", default="buildings_fua")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_building_floor_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_building_floor_features")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_building_floor_features.csv")
    parser.add_argument("--floor-height-m", type=float, default=3.2)
    parser.add_argument("--floor-count-max-valid", type=float, default=60.0)
    return parser.parse_args()


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def safe_intersection_area(a: ogr.Geometry, b: ogr.Geometry) -> float:
    try:
        inter = a.Intersection(b)
    except RuntimeError:
        a_valid = a.MakeValid()
        b_valid = b.MakeValid()
        inter = a_valid.Intersection(b_valid)
    if inter is None or inter.IsEmpty():
        return 0.0
    return float(inter.Area())


def empty_row() -> dict[str, Any]:
    row: dict[str, Any] = {
        "building_count_exact": 0,
        "building_area_exact_m2": 0.0,
        "building_coverage_exact": 0.0,
        "building_density_exact_per_km2": 0.0,
        "floor_count_valid_buildings": 0,
        "floor_count_missing_or_zero_buildings": 0,
        "floor_count_outlier_buildings": 0,
        "floor_count_max_raw": None,
        "floor_count_max_valid": None,
        "floor_count_aw_mean": None,
        "floor_count_aw_std": None,
        "height_proxy_aw_mean_m": None,
        "height_proxy_max_m": None,
        "floor_area_proxy_m2": 0.0,
        "floor_area_ratio_proxy": 0.0,
        "built_volume_proxy_m3": 0.0,
        "underground_floor_aw_mean": None,
        "lowrise_1_2_area_share": None,
        "midrise_3_5_area_share": None,
        "upper_midrise_6_10_area_share": None,
        "highrise_11_plus_area_share": None,
        "type_mesken_area_share": None,
        "type_isyeri_area_share": None,
        "type_kamu_area_share": None,
        "dominant_yapitip": "",
    }
    return row


def finalize_row(row: dict[str, Any], cell_area_m2: float, floor_height_m: float) -> dict[str, Any]:
    building_area = row.pop("_building_area", 0.0)
    valid_floor_area = row.pop("_valid_floor_area", 0.0)
    floor_sum = row.pop("_floor_sum", 0.0)
    floor_sum2 = row.pop("_floor_sum2", 0.0)
    underground_sum = row.pop("_underground_sum", 0.0)
    low_area = row.pop("_low_area", 0.0)
    mid_area = row.pop("_mid_area", 0.0)
    upper_mid_area = row.pop("_upper_mid_area", 0.0)
    high_area = row.pop("_high_area", 0.0)
    type_areas = {
        "mesken": row.pop("_type_mesken_area", 0.0),
        "isyeri": row.pop("_type_isyeri_area", 0.0),
        "kamu": row.pop("_type_kamu_area", 0.0),
    }

    row["building_area_exact_m2"] = building_area
    row["building_coverage_exact"] = building_area / cell_area_m2 if cell_area_m2 else 0.0
    row["building_density_exact_per_km2"] = (
        row["building_count_exact"] / (cell_area_m2 / 1_000_000.0) if cell_area_m2 else 0.0
    )

    if valid_floor_area > 0:
        mean = floor_sum / valid_floor_area
        variance = max((floor_sum2 / valid_floor_area) - mean * mean, 0.0)
        row["floor_count_aw_mean"] = mean
        row["floor_count_aw_std"] = math.sqrt(variance)
        row["height_proxy_aw_mean_m"] = mean * floor_height_m
        if row["floor_count_max_valid"] is not None:
            row["height_proxy_max_m"] = row["floor_count_max_valid"] * floor_height_m
        row["floor_area_proxy_m2"] = floor_sum
        row["floor_area_ratio_proxy"] = floor_sum / cell_area_m2 if cell_area_m2 else 0.0
        row["built_volume_proxy_m3"] = floor_sum * floor_height_m
        row["underground_floor_aw_mean"] = underground_sum / valid_floor_area
        row["lowrise_1_2_area_share"] = low_area / valid_floor_area
        row["midrise_3_5_area_share"] = mid_area / valid_floor_area
        row["upper_midrise_6_10_area_share"] = upper_mid_area / valid_floor_area
        row["highrise_11_plus_area_share"] = high_area / valid_floor_area

    if building_area > 0:
        for key, area in type_areas.items():
            row[f"type_{key}_area_share"] = area / building_area
        dominant = max(type_areas.items(), key=lambda item: item[1])[0]
        row["dominant_yapitip"] = dominant if type_areas[dominant] > 0 else ""

    return row


def aggregate(
    grid_layer: ogr.Layer,
    buildings_layer: ogr.Layer,
    floor_height_m: float,
    floor_count_max_valid: float,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for idx, grid_feature in enumerate(grid_layer, start=1):
        if idx % 1000 == 0:
            print(f"processed_grid_cells={idx}/{total}")
        grid_id = grid_feature.GetField("grid_id")
        cell_geom = grid_feature.GetGeometryRef()
        if cell_geom is None or cell_geom.IsEmpty():
            continue
        cell_geom = cell_geom.Clone()
        cell_area_m2 = float(grid_feature.GetField("cell_area_m2") or cell_geom.Area())
        row = empty_row()
        row["_building_area"] = 0.0
        row["_valid_floor_area"] = 0.0
        row["_floor_sum"] = 0.0
        row["_floor_sum2"] = 0.0
        row["_underground_sum"] = 0.0
        row["_low_area"] = 0.0
        row["_mid_area"] = 0.0
        row["_upper_mid_area"] = 0.0
        row["_high_area"] = 0.0
        row["_type_mesken_area"] = 0.0
        row["_type_isyeri_area"] = 0.0
        row["_type_kamu_area"] = 0.0

        buildings_layer.SetSpatialFilter(cell_geom)
        buildings_layer.ResetReading()
        for building_feature in buildings_layer:
            building_geom = building_feature.GetGeometryRef()
            if building_geom is None or building_geom.IsEmpty():
                continue
            area = safe_intersection_area(cell_geom, building_geom)
            if area <= 0:
                continue

            row["building_count_exact"] += 1
            row["_building_area"] += area

            yapi_tip = building_feature.GetField("YAPITIP")
            type_key = TYPE_FIELD_MAP.get(str(yapi_tip).upper(), "")
            if type_key:
                row[f"_type_{type_key}_area"] += area

            floors = building_feature.GetField("ZEMINUSTUK")
            underground = building_feature.GetField("ZEMINALTIK")
            if floors is not None:
                row["floor_count_max_raw"] = (
                    float(floors)
                    if row["floor_count_max_raw"] is None
                    else max(float(row["floor_count_max_raw"]), float(floors))
                )

            if floors is None or floors <= 0:
                row["floor_count_missing_or_zero_buildings"] += 1
                continue
            if floors > floor_count_max_valid:
                row["floor_count_outlier_buildings"] += 1
                continue

            floors_f = float(floors)
            row["floor_count_valid_buildings"] += 1
            row["floor_count_max_valid"] = (
                floors_f if row["floor_count_max_valid"] is None else max(row["floor_count_max_valid"], floors_f)
            )
            row["_valid_floor_area"] += area
            row["_floor_sum"] += area * floors_f
            row["_floor_sum2"] += area * floors_f * floors_f
            row["_underground_sum"] += area * float(underground or 0)

            if floors_f <= 2:
                row["_low_area"] += area
            elif floors_f <= 5:
                row["_mid_area"] += area
            elif floors_f <= 10:
                row["_upper_mid_area"] += area
            else:
                row["_high_area"] += area

        stats[grid_id] = finalize_row(row, cell_area_m2, floor_height_m)
        buildings_layer.SetSpatialFilter(None)
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if isinstance(value, float) or value is None:
        return ogr.FieldDefn(name, ogr.OFTReal)
    field = ogr.FieldDefn(name, ogr.OFTString)
    field.SetWidth(40)
    return field


def write_gpkg(
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
        for i in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(i).GetName(), feature.GetField(i))
        grid_id = feature.GetField("grid_id")
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
        floor_height_m=args.floor_height_m,
        floor_count_max_valid=args.floor_count_max_valid,
    )
    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    built_cells = sum(1 for row in stats.values() if row["building_count_exact"] > 0)
    valid_floor_cells = sum(1 for row in stats.values() if row["floor_count_valid_buildings"] > 0)
    outlier_buildings = sum(int(row["floor_count_outlier_buildings"]) for row in stats.values())
    print(f"grid_cells={len(stats)}")
    print(f"built_cells={built_cells}")
    print(f"valid_floor_cells={valid_floor_cells}")
    print(f"outlier_building_intersections={outlier_buildings}")
    print(f"floor_height_m={args.floor_height_m}")
    print(f"floor_count_max_valid={args.floor_count_max_valid}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    buildings_ds = None


if __name__ == "__main__":
    main()
