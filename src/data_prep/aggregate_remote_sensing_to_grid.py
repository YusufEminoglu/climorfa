"""Aggregate remote-sensing and surface model rasters to the 250 m analysis grid."""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, ogr


CONTINUOUS_PRODUCTS = [
    {
        "path": "data/02_interim/rasters/sentinel2_indices_fua.tif",
        "bands": {
            1: "s2_ndvi",
            2: "s2_ndwi",
            3: "s2_ndbi",
        },
        "stats": ("mean", "std"),
    },
    {
        "path": "data/02_interim/rasters/landsat_lst_fua.tif",
        "bands": {
            1: "lst_c_median",
            2: "lst_obs_count",
        },
        "stats": ("mean", "std"),
    },
    {
        "path": "data/02_interim/rasters/dynamic_world_fua.tif",
        "bands": {
            1: "dw_water_prob",
            2: "dw_trees_prob",
            3: "dw_grass_prob",
            4: "dw_flooded_vegetation_prob",
            5: "dw_crops_prob",
            6: "dw_shrub_scrub_prob",
            7: "dw_built_prob",
            8: "dw_bare_prob",
        },
        "stats": ("mean", "std"),
    },
    {
        "path": "data/02_interim/rasters/surface_model_fua.tif",
        "bands": {
            1: "dsm_elevation_m",
        },
        "stats": ("mean", "std", "min", "max", "range"),
        "valid_min": -10,
    },
]

WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument(
        "--worldcover-raster",
        default="data/02_interim/rasters/worldcover_fua.tif",
    )
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_remote_sensing_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_remote_sensing")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_remote_sensing_features.csv")
    return parser.parse_args()


def open_grid(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open grid {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Grid layer not found: {layer_name}")
    return ds, layer


def build_grid_index_layer(grid_layer: ogr.Layer) -> tuple[ogr.DataSource, ogr.Layer, dict[int, str]]:
    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("grid_index")
    srs = grid_layer.GetSpatialRef()
    mem_layer = mem_ds.CreateLayer("grid_index", srs=srs, geom_type=ogr.wkbPolygon)
    mem_layer.CreateField(ogr.FieldDefn("grid_num", ogr.OFTInteger))

    grid_id_by_num: dict[int, str] = {}
    out_defn = mem_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for num, feature in enumerate(grid_layer, start=1):
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        out_feature = ogr.Feature(out_defn)
        out_feature.SetField("grid_num", num)
        out_feature.SetGeometry(geom.Clone())
        mem_layer.CreateFeature(out_feature)
        grid_id_by_num[num] = feature.GetField("grid_id")
        out_feature = None
    return mem_ds, mem_layer, grid_id_by_num


def rasterize_grid(grid_layer: ogr.Layer, template_ds: gdal.Dataset) -> np.ndarray:
    cols = template_ds.RasterXSize
    rows = template_ds.RasterYSize
    mem_raster = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Int32)
    mem_raster.SetGeoTransform(template_ds.GetGeoTransform())
    mem_raster.SetProjection(template_ds.GetProjection())
    band = mem_raster.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(mem_raster, [1], grid_layer, options=["ATTRIBUTE=grid_num"])
    return band.ReadAsArray()


def init_stats(grid_id_by_num: dict[int, str]) -> dict[str, dict[str, Any]]:
    return {grid_id: {} for grid_id in grid_id_by_num.values()}


def clean_float(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def add_continuous_stats(
    stats: dict[str, dict[str, Any]],
    grid_id_by_num: dict[int, str],
    grid_num_arr: np.ndarray,
    raster_ds: gdal.Dataset,
    band_num: int,
    prefix: str,
    stat_names: tuple[str, ...],
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> None:
    band = raster_ds.GetRasterBand(band_num)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    valid = grid_num_arr > 0
    if np.issubdtype(arr.dtype, np.floating):
        valid &= np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    if valid_min is not None:
        valid &= arr >= valid_min
    if valid_max is not None:
        valid &= arr <= valid_max

    nums = grid_num_arr[valid].astype(np.int64, copy=False)
    values = arr[valid].astype(np.float64, copy=False)
    max_num = max(grid_id_by_num)
    counts = np.bincount(nums, minlength=max_num + 1)
    sums = np.bincount(nums, weights=values, minlength=max_num + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts

    if "std" in stat_names:
        sums2 = np.bincount(nums, weights=values * values, minlength=max_num + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = sums2 / counts - means * means
        variance = np.maximum(variance, 0)
        stds = np.sqrt(variance)
    else:
        stds = None

    mins = maxs = None
    if "min" in stat_names or "max" in stat_names or "range" in stat_names:
        mins = np.full(max_num + 1, np.inf, dtype=np.float64)
        maxs = np.full(max_num + 1, -np.inf, dtype=np.float64)
        np.minimum.at(mins, nums, values)
        np.maximum.at(maxs, nums, values)

    for grid_num, grid_id in grid_id_by_num.items():
        count = int(counts[grid_num])
        stats[grid_id][f"{prefix}_valid_px"] = count
        if count == 0:
            for name in stat_names:
                stats[grid_id][f"{prefix}_{name}"] = None
            continue
        if "mean" in stat_names:
            stats[grid_id][f"{prefix}_mean"] = clean_float(means[grid_num])
        if "std" in stat_names and stds is not None:
            stats[grid_id][f"{prefix}_std"] = clean_float(stds[grid_num])
        if "min" in stat_names and mins is not None:
            stats[grid_id][f"{prefix}_min"] = clean_float(mins[grid_num])
        if "max" in stat_names and maxs is not None:
            stats[grid_id][f"{prefix}_max"] = clean_float(maxs[grid_num])
        if "range" in stat_names and mins is not None and maxs is not None:
            stats[grid_id][f"{prefix}_range"] = clean_float(maxs[grid_num] - mins[grid_num])

    del arr, valid, nums, values, counts, sums
    gc.collect()


def aggregate_continuous_products(
    stats: dict[str, dict[str, Any]],
    grid_id_by_num: dict[int, str],
    grid_index_layer: ogr.Layer,
) -> None:
    for product in CONTINUOUS_PRODUCTS:
        raster_path = product["path"]
        ds = gdal.Open(raster_path)
        if ds is None:
            raise RuntimeError(f"Could not open raster {raster_path}")
        print(f"aggregating_continuous={raster_path}")
        grid_num_arr = rasterize_grid(grid_index_layer, ds)
        for band_num, prefix in product["bands"].items():
            add_continuous_stats(
                stats=stats,
                grid_id_by_num=grid_id_by_num,
                grid_num_arr=grid_num_arr,
                raster_ds=ds,
                band_num=band_num,
                prefix=prefix,
                stat_names=product["stats"],
                valid_min=product.get("valid_min"),
                valid_max=product.get("valid_max"),
            )
        del grid_num_arr
        ds = None
        gc.collect()


def aggregate_worldcover(
    stats: dict[str, dict[str, Any]],
    grid_id_by_num: dict[int, str],
    grid_index_layer: ogr.Layer,
    raster_path: str,
) -> None:
    ds = gdal.Open(raster_path)
    if ds is None:
        raise RuntimeError(f"Could not open WorldCover raster {raster_path}")
    print(f"aggregating_categorical={raster_path}")
    grid_num_arr = rasterize_grid(grid_index_layer, ds)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    valid = grid_num_arr > 0
    if nodata is not None:
        valid &= arr != nodata
    nums = grid_num_arr[valid].astype(np.int64, copy=False)
    values = arr[valid].astype(np.int16, copy=False)
    max_num = max(grid_id_by_num)
    valid_counts = np.bincount(nums, minlength=max_num + 1)
    majority_count = np.zeros(max_num + 1, dtype=np.int64)
    majority_class = np.zeros(max_num + 1, dtype=np.int16)

    class_counts_by_name: dict[str, np.ndarray] = {}
    for code, name in WORLDCOVER_CLASSES.items():
        class_counts = np.bincount(nums[values == code], minlength=max_num + 1)
        class_counts_by_name[name] = class_counts
        update = class_counts > majority_count
        majority_count[update] = class_counts[update]
        majority_class[update] = code

    for grid_num, grid_id in grid_id_by_num.items():
        count = int(valid_counts[grid_num])
        stats[grid_id]["worldcover_valid_px"] = count
        if count == 0:
            stats[grid_id]["worldcover_majority_class"] = None
            stats[grid_id]["worldcover_majority_share"] = None
        else:
            stats[grid_id]["worldcover_majority_class"] = int(majority_class[grid_num])
            stats[grid_id]["worldcover_majority_share"] = float(majority_count[grid_num] / count)
        for name, class_counts in class_counts_by_name.items():
            stats[grid_id][f"worldcover_share_{name}"] = (
                float(class_counts[grid_num] / count) if count else None
            )

    del grid_num_arr, arr, valid, nums, values, valid_counts, majority_count, majority_class
    gc.collect()


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        field = ogr.FieldDefn(name, ogr.OFTInteger)
    else:
        field = ogr.FieldDefn(name, ogr.OFTReal)
    return field


def write_gpkg(
    grid_layer: ogr.Layer,
    out_path: Path,
    out_layer_name: str,
    stats: dict[str, dict[str, Any]],
    stat_fields: list[str],
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

    sample_values: dict[str, Any] = {}
    for field in stat_fields:
        for row in stats.values():
            value = row.get(field)
            if value is not None:
                sample_values[field] = value
                break
    for field in stat_fields:
        out_layer.CreateField(create_field(field, sample_values.get(field, 0.0)))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        out_feature = ogr.Feature(out_defn)
        for i in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(i).GetName(), feature.GetField(i))
        grid_id = feature.GetField("grid_id")
        row = stats.get(grid_id, {})
        for field in stat_fields:
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, int):
                out_feature.SetField(field, int(value))
            else:
                out_feature.SetField(field, float(value))
        geom = feature.GetGeometryRef()
        if geom is not None:
            out_feature.SetGeometry(geom.Clone())
        out_layer.CreateFeature(out_feature)
        out_feature = None
    out_layer.SyncToDisk()
    out_ds = None


def write_csv(out_path: Path, stats: dict[str, dict[str, Any]], stat_fields: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_id", *stat_fields])
        writer.writeheader()
        for grid_id, row in stats.items():
            writer.writerow({"grid_id": grid_id, **{field: row.get(field) for field in stat_fields}})


def main() -> None:
    args = parse_args()
    grid_ds, grid_layer = open_grid(args.grid_gpkg, args.grid_layer)
    mem_ds, mem_layer, grid_id_by_num = build_grid_index_layer(grid_layer)
    stats = init_stats(grid_id_by_num)

    aggregate_continuous_products(stats, grid_id_by_num, mem_layer)
    aggregate_worldcover(stats, grid_id_by_num, mem_layer, args.worldcover_raster)

    stat_fields = sorted(next(iter(stats.values())).keys())
    write_gpkg(grid_layer, Path(args.out_gpkg), args.out_layer, stats, stat_fields)
    write_csv(Path(args.out_csv), stats, stat_fields)

    print(f"grid_cells={len(stats)}")
    print(f"feature_fields={len(stat_fields)}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    mem_ds = None
    grid_ds = None


if __name__ == "__main__":
    main()
