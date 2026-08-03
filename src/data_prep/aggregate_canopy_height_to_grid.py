"""Aggregate ETH 2020 canopy-height raster to the 250 m grid."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal, ogr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument(
        "--canopy-raster",
        default="data/02_interim/rasters/canopy_height_fua.tif",
    )
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_canopy_height_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_canopy_height_features")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_canopy_height_features.csv")
    parser.add_argument("--pixel-area-m2", type=float, default=100.0)
    return parser.parse_args()


def open_grid(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open grid {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Grid layer not found: {layer_name}")
    return ds, layer


def build_grid_index_layer(grid_layer: ogr.Layer) -> tuple[ogr.DataSource, ogr.Layer, dict[int, str], dict[str, float]]:
    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("grid_index")
    srs = grid_layer.GetSpatialRef()
    mem_layer = mem_ds.CreateLayer("grid_index", srs=srs, geom_type=ogr.wkbPolygon)
    mem_layer.CreateField(ogr.FieldDefn("grid_num", ogr.OFTInteger))

    grid_id_by_num: dict[int, str] = {}
    cell_area_by_grid_id: dict[str, float] = {}
    out_defn = mem_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for num, feature in enumerate(grid_layer, start=1):
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        grid_id = feature.GetField("grid_id")
        out_feature = ogr.Feature(out_defn)
        out_feature.SetField("grid_num", num)
        out_feature.SetGeometry(geom.Clone())
        mem_layer.CreateFeature(out_feature)
        grid_id_by_num[num] = grid_id
        cell_area_by_grid_id[grid_id] = float(feature.GetField("cell_area_m2") or geom.Area())
        out_feature = None
    return mem_ds, mem_layer, grid_id_by_num, cell_area_by_grid_id


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


def percentile_by_group(nums: np.ndarray, values: np.ndarray, grid_id_by_num: dict[int, str], q: float) -> dict[int, float]:
    out: dict[int, float] = {}
    order = np.argsort(nums, kind="mergesort")
    nums_sorted = nums[order]
    values_sorted = values[order]
    starts = np.flatnonzero(np.r_[True, nums_sorted[1:] != nums_sorted[:-1]])
    ends = np.r_[starts[1:], len(nums_sorted)]
    for start, end in zip(starts, ends):
        grid_num = int(nums_sorted[start])
        if grid_num in grid_id_by_num:
            out[grid_num] = float(np.percentile(values_sorted[start:end], q))
    return out


def aggregate(
    ds: gdal.Dataset,
    grid_num_arr: np.ndarray,
    grid_id_by_num: dict[int, str],
    cell_area_by_grid_id: dict[str, float],
    pixel_area_m2: float,
) -> dict[str, dict[str, Any]]:
    height_band = ds.GetRasterBand(1)
    sd_band = ds.GetRasterBand(2)
    height = height_band.ReadAsArray().astype(np.float64)
    sd = sd_band.ReadAsArray().astype(np.float64)
    height_nodata = height_band.GetNoDataValue()
    sd_nodata = sd_band.GetNoDataValue()

    valid = grid_num_arr > 0
    valid &= np.isfinite(height)
    if height_nodata is not None:
        valid &= height != height_nodata
    valid &= height >= 0
    valid &= height < 255

    nums = grid_num_arr[valid].astype(np.int64, copy=False)
    values = height[valid]
    max_num = max(grid_id_by_num)
    counts = np.bincount(nums, minlength=max_num + 1)
    sums = np.bincount(nums, weights=values, minlength=max_num + 1)
    sums2 = np.bincount(nums, weights=values * values, minlength=max_num + 1)
    gt2 = np.bincount(nums[values >= 2], minlength=max_num + 1)
    gt5 = np.bincount(nums[values >= 5], minlength=max_num + 1)
    gt10 = np.bincount(nums[values >= 10], minlength=max_num + 1)
    volume_gt2 = np.bincount(nums[values >= 2], weights=values[values >= 2] * pixel_area_m2, minlength=max_num + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / counts
        variances = sums2 / counts - means * means
    stds = np.sqrt(np.maximum(variances, 0))
    p95 = percentile_by_group(nums, values, grid_id_by_num, 95)
    maxes = np.full(max_num + 1, -np.inf)
    np.maximum.at(maxes, nums, values)

    sd_valid = valid.copy()
    sd_valid &= np.isfinite(sd)
    if sd_nodata is not None:
        sd_valid &= sd != sd_nodata
    sd_valid &= sd >= 0
    sd_nums = grid_num_arr[sd_valid].astype(np.int64, copy=False)
    sd_values = sd[sd_valid]
    sd_counts = np.bincount(sd_nums, minlength=max_num + 1)
    sd_sums = np.bincount(sd_nums, weights=sd_values, minlength=max_num + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_means = sd_sums / sd_counts

    stats: dict[str, dict[str, Any]] = {}
    for grid_num, grid_id in grid_id_by_num.items():
        count = int(counts[grid_num])
        row: dict[str, Any] = {
            "canopy_height_valid_px": count,
            "canopy_height_mean_m": None,
            "canopy_height_std_m": None,
            "canopy_height_p95_m": None,
            "canopy_height_max_m": None,
            "canopy_cover_gt2m_share": None,
            "canopy_cover_gt5m_share": None,
            "canopy_cover_gt10m_share": None,
            "canopy_area_gt2m_m2": 0.0,
            "canopy_volume_gt2m_proxy_m3": 0.0,
            "canopy_volume_gt2m_proxy_m3_per_ha": 0.0,
            "canopy_height_sd_mean_m": None,
            "canopy_uncertainty_valid_px": int(sd_counts[grid_num]),
        }
        if count > 0:
            cell_area = cell_area_by_grid_id.get(grid_id, 0.0)
            row["canopy_height_mean_m"] = float(means[grid_num])
            row["canopy_height_std_m"] = float(stds[grid_num])
            row["canopy_height_p95_m"] = p95.get(grid_num)
            row["canopy_height_max_m"] = float(maxes[grid_num])
            row["canopy_cover_gt2m_share"] = float(gt2[grid_num] / count)
            row["canopy_cover_gt5m_share"] = float(gt5[grid_num] / count)
            row["canopy_cover_gt10m_share"] = float(gt10[grid_num] / count)
            row["canopy_area_gt2m_m2"] = float(gt2[grid_num] * pixel_area_m2)
            row["canopy_volume_gt2m_proxy_m3"] = float(volume_gt2[grid_num])
            row["canopy_volume_gt2m_proxy_m3_per_ha"] = (
                float(volume_gt2[grid_num] / (cell_area / 10_000.0)) if cell_area else 0.0
            )
        if sd_counts[grid_num] > 0:
            row["canopy_height_sd_mean_m"] = float(sd_means[grid_num])
        stats[grid_id] = row

    del height, sd, valid, nums, values
    gc.collect()
    return stats


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    return ogr.FieldDefn(name, ogr.OFTReal)


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
        out_layer.CreateField(create_field(field, sample_values.get(field, 0.0)))

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
    grid_ds, grid_layer = open_grid(args.grid_gpkg, args.grid_layer)
    mem_ds, mem_layer, grid_id_by_num, cell_area_by_grid_id = build_grid_index_layer(grid_layer)
    canopy_ds = gdal.Open(args.canopy_raster)
    if canopy_ds is None:
        raise RuntimeError(f"Could not open canopy raster {args.canopy_raster}")
    grid_num_arr = rasterize_grid(mem_layer, canopy_ds)
    stats = aggregate(canopy_ds, grid_num_arr, grid_id_by_num, cell_area_by_grid_id, args.pixel_area_m2)

    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    cells_with_canopy = sum(1 for row in stats.values() if row["canopy_height_valid_px"] > 0)
    cells_gt2 = sum(1 for row in stats.values() if (row["canopy_cover_gt2m_share"] or 0) > 0)
    print(f"grid_cells={len(stats)}")
    print(f"cells_with_valid_canopy_pixels={cells_with_canopy}")
    print(f"cells_with_canopy_gt2m={cells_gt2}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    mem_ds = None
    grid_ds = None


if __name__ == "__main__":
    main()
