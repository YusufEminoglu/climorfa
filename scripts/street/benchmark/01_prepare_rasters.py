"""
CLIMORFA Street — WP0.4 SOLWEIG Benchmark Tile Preparation
Prepares 1 m and 2 m DEM, Building DSM, and Tree DSM rasters for the Alsancak benchmark tile.

Spatial extent (EPSG:5253):
- Inner 500x500 m: [512350, 4255850, 512850, 4256350]
- Buffered 700x700 m: [512250, 4255750, 512950, 4256450]
"""

import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import pyogrio

def prepare_benchmark_rasters():
    # Paths
    project_root = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
    buildings_path = os.path.join(project_root, "data/02_interim/akilli_sehir_fua.gpkg")
    dsm_5m_path = os.path.join(project_root, "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif")
    eth_canopy_path = os.path.join(project_root, "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif")
    out_dir = os.path.join(project_root, "data/street/benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # Coordinates (EPSG:5253)
    xmin_inner, ymin_inner = 512350.0, 4255850.0
    xmax_inner, ymax_inner = 512850.0, 4256350.0

    buffer_m = 100.0
    xmin_buf = xmin_inner - buffer_m  # 512250.0
    ymin_buf = ymin_inner - buffer_m  # 4255750.0
    xmax_buf = xmax_inner + buffer_m  # 512950.0
    ymax_buf = ymax_inner + buffer_m  # 4256450.0

    width_m = xmax_buf - xmin_buf     # 700.0 m
    height_m = ymax_buf - ymin_buf   # 700.0 m

    print("=== Step 1: Building Footprints & Floor Statistics ===")
    bbox_buf = (xmin_buf, ymin_buf, xmax_buf, ymax_buf)
    bld_gdf = pyogrio.read_dataframe(buildings_path, layer="buildings_fua", bbox=bbox_buf)
    print(f"Loaded {len(bld_gdf)} buildings in 700x700 m buffered window.")

    # Calculate missing / zero floor share
    floors_raw = bld_gdf["ZEMINUSTUK"].values
    missing_or_zero = np.isnan(floors_raw) | (floors_raw <= 0)
    missing_or_zero_cnt = int(np.sum(missing_or_zero))
    missing_or_zero_share = float(missing_or_zero_cnt / len(bld_gdf)) if len(bld_gdf) > 0 else 0.0

    print(f"Missing or <=0 floors: {missing_or_zero_cnt}/{len(bld_gdf)} ({missing_or_zero_share * 100:.2f}%)")

    # Replace missing or 0 floors with 1 floor
    floors_clean = np.where(missing_or_zero, 1.0, floors_raw).astype(np.float32)
    bld_gdf["clean_floors"] = floors_clean
    bld_gdf["building_height"] = floors_clean * 3.0

    # Also check inner 500 m stats
    bbox_inner = (xmin_inner, ymin_inner, xmax_inner, ymax_inner)
    bld_inner = pyogrio.read_dataframe(buildings_path, layer="buildings_fua", bbox=bbox_inner)
    fl_inner_raw = bld_inner["ZEMINUSTUK"].values
    m_z_inner = np.isnan(fl_inner_raw) | (fl_inner_raw <= 0)
    print(f"Inner 500x500 m buildings: {len(bld_inner)}, missing/zero: {int(np.sum(m_z_inner))}/{len(bld_inner)}")
    print(f"Inner mean floors: {np.where(m_z_inner, 1.0, fl_inner_raw).mean():.2f}")

    shapes_with_height = [
        (geom, float(ht)) for geom, ht in zip(bld_gdf.geometry.values, bld_gdf["building_height"].values)
        if geom is not None and not geom.is_empty
    ]

    tile_meta = {
        "is_synthetic": False,
        "crs": "EPSG:5253",
        "inner_500m_bbox": [xmin_inner, ymin_inner, xmax_inner, ymax_inner],
        "buffered_700m_bbox": [xmin_buf, ymin_buf, xmax_buf, ymax_buf],
        "buffer_m": buffer_m,
        "building_count_buffered": len(bld_gdf),
        "building_count_inner": len(bld_inner),
        "missing_or_zero_floor_count": missing_or_zero_cnt,
        "missing_or_zero_floor_share": missing_or_zero_share,
        "mean_floors_buffered": float(floors_clean.mean()),
        "max_floors_buffered": float(floors_clean.max()),
        "mean_height_m_buffered": float(bld_gdf["building_height"].mean()),
    }

    resolutions = [1.0, 2.0]

    for res in resolutions:
        res_str = f"{int(res)}m"
        print(f"\n=== Step 2: Building {res_str} Rasters ===")
        cols = int(round(width_m / res))
        rows = int(round(height_m / res))
        dst_transform = from_origin(xmin_buf, ymax_buf, res, res)
        dst_crs = rasterio.crs.CRS.from_epsg(5253)

        # 1. Resample DEM from 5 m DSM
        dem = np.zeros((rows, cols), dtype=np.float32)
        with rasterio.open(dsm_5m_path) as src_dsm:
            reproject(
                source=rasterio.band(src_dsm, 1),
                destination=dem,
                src_transform=src_dsm.transform,
                src_crs=src_dsm.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src_dsm.nodata,
                dst_nodata=-32767.0
            )

        # Fill any sea / water nodata in DEM with 0.0 m (sea level)
        sea_nodata_mask = (dem <= -9999.0) | np.isnan(dem)
        dem[sea_nodata_mask] = 0.0
        print(f"[{res_str}] DEM min: {dem.min():.2f} m, mean: {dem.mean():.2f} m, max: {dem.max():.2f} m")

        # 2. Rasterize buildings
        bld_height_grid = rasterize(
            shapes=shapes_with_height,
            out_shape=(rows, cols),
            transform=dst_transform,
            fill=0.0,
            dtype=np.float32,
            all_touched=False,
        )

        bld_mask = bld_height_grid > 0.0
        building_dsm = dem.copy()
        building_dsm[bld_mask] += bld_height_grid[bld_mask]
        print(f"[{res_str}] Building DSM min: {building_dsm.min():.2f} m, mean: {building_dsm.mean():.2f} m, max: {building_dsm.max():.2f} m")
        print(f"[{res_str}] Building footprint coverage: {100.0 * np.sum(bld_mask) / bld_mask.size:.2f}%")

        # 3. Resample ETH Canopy Height
        canopy = np.zeros((rows, cols), dtype=np.float32)
        with rasterio.open(eth_canopy_path) as src_canopy:
            reproject(
                source=rasterio.band(src_canopy, 1),
                destination=canopy,
                src_transform=src_canopy.transform,
                src_crs=src_canopy.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src_canopy.nodata,
                dst_nodata=-9999.0
            )

        # Set 0 for nodata, inside buildings, and heights < 2 m
        canopy_invalid = (canopy <= -9999.0) | np.isnan(canopy) | (canopy < 2.0)
        canopy[canopy_invalid] = 0.0
        canopy[bld_mask] = 0.0  # Zero inside buildings

        tree_px = np.sum(canopy >= 2.0)
        tree_share = 100.0 * tree_px / canopy.size
        print(f"[{res_str}] Trees min: {canopy.min():.2f} m, mean (valid trees): {canopy[canopy >= 2.0].mean() if tree_px > 0 else 0:.2f} m, max: {canopy.max():.2f} m")
        print(f"[{res_str}] Tree cover share: {tree_share:.2f}% ({tree_px} pixels)")

        # Write out GeoTIFFs
        meta = {
            "driver": "GTiff",
            "height": rows,
            "width": cols,
            "count": 1,
            "dtype": np.float32,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": None
        }

        dem_out = os.path.join(out_dir, f"DEM_{res_str}.tif")
        bld_dsm_out = os.path.join(out_dir, f"Building_DSM_{res_str}.tif")
        trees_out = os.path.join(out_dir, f"Trees_{res_str}.tif")

        with rasterio.open(dem_out, "w", **meta) as dst:
            dst.write(dem, 1)
        with rasterio.open(bld_dsm_out, "w", **meta) as dst:
            dst.write(building_dsm, 1)
        with rasterio.open(trees_out, "w", **meta) as dst:
            dst.write(canopy, 1)

        print(f"Saved: {dem_out}")
        print(f"Saved: {bld_dsm_out}")
        print(f"Saved: {trees_out}")

    meta_json_path = os.path.join(out_dir, "tile_metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(tile_meta, f, indent=2)
    print(f"Saved metadata: {meta_json_path}")

if __name__ == "__main__":
    prepare_benchmark_rasters()
