"""Aggregate weak LCZ raster labels to grid cells."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", required=True)
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--lcz-raster", required=True)
    parser.add_argument("--out-gpkg", required=True)
    parser.add_argument("--out-layer", default="grid_250m_lcz")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--mixed-threshold", type=float, default=0.7)
    return parser.parse_args()


def build_grid_index_layer(grid_layer: ogr.Layer) -> tuple[ogr.DataSource, ogr.Layer, dict[int, str]]:
    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("grid_index")
    srs = grid_layer.GetSpatialRef()
    mem_layer = mem_ds.CreateLayer("grid_index", srs=srs, geom_type=ogr.wkbPolygon)
    idx_field = ogr.FieldDefn("grid_num", ogr.OFTInteger)
    mem_layer.CreateField(idx_field)

    grid_id_by_num = {}
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


def mode(values: np.ndarray) -> tuple[int | None, float]:
    if values.size == 0:
        return None, 0.0
    counts = Counter(values.tolist())
    label, count = counts.most_common(1)[0]
    return int(label), float(count) / float(values.size)


def write_output(
    grid_ds: ogr.DataSource,
    grid_layer: ogr.Layer,
    out_path: Path,
    out_layer_name: str,
    stats: dict[str, dict[str, float | int | None]],
) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        out_ds = driver.Open(str(out_path), update=1)
        existing = out_ds.GetLayerByName(out_layer_name) if out_ds else None
        if existing is not None:
            out_ds.DeleteLayer(out_layer_name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        out_ds = driver.Open(str(out_path), update=1)
    if out_ds is None:
        raise RuntimeError(f"Could not open output {out_path}")

    srs = grid_layer.GetSpatialRef()
    out_layer = out_ds.CreateLayer(out_layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    in_defn = grid_layer.GetLayerDefn()
    for i in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(i))
    for name, ftype in [
        ("lcz_weak_label", ogr.OFTInteger),
        ("lcz_weak_confidence", ogr.OFTReal),
        ("lcz_probability_mean", ogr.OFTReal),
        ("lcz_valid_pixels", ogr.OFTInteger),
        ("lcz_mixed_flag", ogr.OFTInteger),
    ]:
        out_layer.CreateField(ogr.FieldDefn(name, ftype))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        out_feature = ogr.Feature(out_defn)
        for i in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(i).GetName(), feature.GetField(i))
        grid_id = feature.GetField("grid_id")
        row = stats.get(grid_id, {})
        if row.get("lcz_weak_label") is not None:
            out_feature.SetField("lcz_weak_label", int(row["lcz_weak_label"]))
        out_feature.SetField("lcz_weak_confidence", float(row.get("lcz_weak_confidence", 0.0)))
        out_feature.SetField("lcz_probability_mean", float(row.get("lcz_probability_mean", 0.0)))
        out_feature.SetField("lcz_valid_pixels", int(row.get("lcz_valid_pixels", 0)))
        out_feature.SetField("lcz_mixed_flag", int(row.get("lcz_mixed_flag", 1)))
        out_feature.SetGeometry(feature.GetGeometryRef().Clone())
        out_layer.CreateFeature(out_feature)
        out_feature = None
    out_layer.SyncToDisk()
    out_ds = None


def main() -> None:
    args = parse_args()
    grid_ds = ogr.Open(args.grid_gpkg)
    if grid_ds is None:
        raise RuntimeError(f"Could not open grid {args.grid_gpkg}")
    grid_layer = grid_ds.GetLayerByName(args.grid_layer)
    if grid_layer is None:
        raise RuntimeError(f"Grid layer not found {args.grid_layer}")

    lcz_ds = gdal.Open(args.lcz_raster)
    if lcz_ds is None:
        raise RuntimeError(f"Could not open LCZ raster {args.lcz_raster}")

    lcz_arr = lcz_ds.GetRasterBand(1).ReadAsArray()
    prob_arr = lcz_ds.GetRasterBand(3).ReadAsArray()
    lcz_nodata = lcz_ds.GetRasterBand(1).GetNoDataValue()
    prob_nodata = lcz_ds.GetRasterBand(3).GetNoDataValue()

    mem_ds, mem_layer, grid_id_by_num = build_grid_index_layer(grid_layer)
    grid_num_arr = rasterize_grid(mem_layer, lcz_ds)

    stats = {}
    for grid_num, grid_id in grid_id_by_num.items():
        mask = grid_num_arr == grid_num
        if lcz_nodata is not None:
            mask &= lcz_arr != lcz_nodata
        labels = lcz_arr[mask]
        label, confidence = mode(labels)
        prob_mask = mask.copy()
        if prob_nodata is not None:
            prob_mask &= prob_arr != prob_nodata
        prob_values = prob_arr[prob_mask]
        prob_mean = float(np.mean(prob_values)) if prob_values.size else 0.0
        stats[grid_id] = {
            "lcz_weak_label": label,
            "lcz_weak_confidence": confidence,
            "lcz_probability_mean": prob_mean,
            "lcz_valid_pixels": int(labels.size),
            "lcz_mixed_flag": int(confidence < args.mixed_threshold),
        }

    out_path = Path(args.out_gpkg)
    write_output(grid_ds, grid_layer, out_path, args.out_layer, stats)

    import csv

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "grid_id",
                "lcz_weak_label",
                "lcz_weak_confidence",
                "lcz_probability_mean",
                "lcz_valid_pixels",
                "lcz_mixed_flag",
            ],
        )
        writer.writeheader()
        for grid_id, row in stats.items():
            writer.writerow({"grid_id": grid_id, **row})

    labelled = sum(1 for row in stats.values() if row["lcz_weak_label"] is not None)
    mixed = sum(1 for row in stats.values() if row["lcz_mixed_flag"] == 1)
    print(f"grid_cells={len(stats)} labelled={labelled} mixed={mixed}")
    print(f"out_gpkg={out_path}")
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
