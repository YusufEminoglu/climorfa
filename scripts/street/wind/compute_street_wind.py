"""Compute street-level pedestrian wind screening for all 427 core tiles in Izmir FUA.

Source reference:
  Plugin file: plugin/wind.js
  Origin: qgis_plugins/zero2geospatial3d/web/src/core/wind.js
  Software: 02Geo3D v1.2.3 (MIT License, author Yusuf Eminoglu)

DISCLAIMER:
  Geometric screening model (wake deficit, channelling, corner factor), NOT CFD.

Workflow:
  1. Load 1200 m buffered rasters (600 x 600 px at 2 m) per tile:
     - Building DSM and DEM -> heights above ground = max(0, building_dsm - dem)
     - Tree DSM -> optical porosity = 0.5 for trees >= 2.0 m (plugin default: runners.js line 655)
  2. Precompute 16 wind direction sectors (22.5 deg intervals: 0, 22.5, ..., 337.5 deg).
     Work on the full 1200 m buffered tile, save the inner 1000 m (500 x 500 px) to:
     data/street/wind/ratio16/<tile>.tif (16 bands, float32, DEFLATE pred 3).
  3. For both summer design days:
     - met_DESIGN_IMBAT_2022-07-05_hourly.txt
     - met_DESIGN_CALM_2023-07-26_hourly.txt
     Calculate hourly pedestrian wind (06:00 to 21:00, 16 bands):
     U_ped = ratio(interpolated nearest sectors) * U10 * OPEN_PEDESTRIAN_FACTOR
     Save to data/street/wind/<DAY>/<tile>.tif (16 bands, float32, DEFLATE pred 3).
  4. Write band-hour metadata tables and virtual mosaics (VRTs).
  5. Enforce memory check: free RAM >= 3 GB, tile-by-tile, single worker, float32.
"""

from __future__ import annotations
import gc
import json
import os
import sys
import time
from pathlib import Path

# Ensure PROJ and GDAL paths are properly configured for rasterio/pyproj
conda_proj = Path("C:/Users/yusuf/miniforge3/envs/climorfa-cpu/Library/share/proj")
if conda_proj.exists():
    os.environ["PROJ_LIB"] = str(conda_proj)
    os.environ["PROJ_DATA"] = str(conda_proj)

conda_gdal = Path("C:/Users/yusuf/miniforge3/envs/climorfa-cpu/Library/share/gdal")
if conda_gdal.exists():
    os.environ["GDAL_DATA"] = str(conda_gdal)

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

from osgeo import gdal
gdal.UseExceptions()

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path("C:/Users/yusuf/climorfa_work").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.street.wind.wind_screening import wind_ratio, OPEN_PEDESTRIAN_FACTOR

SECTOR_ANGLES = np.array([k * 22.5 for k in range(16)], dtype=np.float32)


def check_ram(min_free_gb: float = 3.0):
    """Enforce memory constraint: stop if free RAM < 3 GB."""
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    if free_gb < min_free_gb:
        raise MemoryError(
            f"Available RAM ({free_gb:.2f} GB) is below safety threshold ({min_free_gb:.2f} GB)!"
        )
    return free_gb


def load_tile_geometry(tile_id: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load building DSM, DEM, and Tree DSM; return heights, porosity, x0, y0."""
    b_path = PROJECT_ROOT / f"data/street/tiles/2m/building_dsm/{tile_id}.tif"
    d_path = PROJECT_ROOT / f"data/street/tiles/2m/dem/{tile_id}.tif"
    t_path = PROJECT_ROOT / f"data/street/tiles/2m/tree_dsm/{tile_id}.tif"

    with rasterio.open(b_path) as fb, rasterio.open(d_path) as fd, rasterio.open(t_path) as ft:
        b = fb.read(1)
        d = fd.read(1)
        t = ft.read(1)
        # Parse x0, y0 from tile id: T_{x0/1000}_{y0/1000}
        parts = tile_id.split("_")
        x0 = float(parts[1]) * 1000.0
        y0 = float(parts[2]) * 1000.0

    # Building height above ground
    h_bldg = np.maximum(0.0, b - d).astype(np.float32)

    heights = np.zeros_like(h_bldg, dtype=np.float32)
    porosity = np.zeros_like(h_bldg, dtype=np.float32)

    # 1. Buildings (solid, porosity = 0)
    bldg_mask = h_bldg > 0.0
    heights[bldg_mask] = h_bldg[bldg_mask]
    porosity[bldg_mask] = 0.0

    # 2. Trees (optical porosity = 0.5 for trees >= 2.0 m, runners.js line 655)
    tree_mask = (t >= 2.0) & (~bldg_mask)
    heights[tree_mask] = t[tree_mask]
    porosity[tree_mask] = 0.5

    return heights, porosity, x0, y0


def interpolate_ratio_at_direction(ratio16: np.ndarray, wd_deg: float) -> np.ndarray:
    """Linearly interpolate ratio array between two nearest 22.5 deg sectors.

    ratio16: (16, rows, cols) array
    wd_deg: meteorological wind direction in degrees
    """
    s = (float(wd_deg) % 360.0) / 22.5
    k0 = int(np.floor(s)) % 16
    k1 = (k0 + 1) % 16
    f = float(s - np.floor(s))

    r0 = ratio16[k0]
    r1 = ratio16[k1]
    return (1.0 - f) * r0 + f * r1


def write_geotiff_16band(
    out_path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:5253",
    band_names: list[str] | None = None,
):
    """Write 16-band float32 GeoTIFF with DEFLATE + Predictor 3, tiled 256x256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, rows, cols = data.shape

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        if band_names:
            for b, name in enumerate(band_names, 1):
                dst.set_band_description(b, name)


def generate_band_hour_tables(wind_dir: Path):
    """Generate band-hour documentation tables for ratio16 and hourly wind."""
    # 1. Sector table for ratio16
    sector_rows = []
    for k in range(16):
        ang = k * 22.5
        min_ang = (ang - 11.25) % 360.0
        max_ang = (ang + 11.25) % 360.0
        sector_rows.append({
            "band": k + 1,
            "sector_index": k,
            "direction_deg": ang,
            "sector_span": f"{min_ang:.2f} - {max_ang:.2f} deg",
            "variable": "Pedestrian Wind Ratio",
            "units": "dimensionless (ratio to open-terrain pedestrian wind)",
            "dtype": "float32",
            "model": "02Geo3D wind.js screening model",
        })
    pd.DataFrame(sector_rows).to_csv(wind_dir / "ratio16/band_sector_table.csv", index=False)

    # 2. Hourly band table for design days (matching WP2.1)
    hourly_rows = []
    for b, it in enumerate(range(6, 22), 1):
        utc = (it - 3) % 24
        hourly_rows.append({
            "band": b,
            "local_time_lst": f"{it:02d}:00",
            "utc_time": f"{utc:02d}:00",
            "wind_variable": "Pedestrian Wind Speed (U_ped)",
            "wind_units": "m/s",
            "wind_dtype": "float32",
            "screening_model": "02Geo3D wind.js screening (ratio x U10 x OPEN_PEDESTRIAN_FACTOR)",
        })
    df_hours = pd.DataFrame(hourly_rows)
    df_hours.to_csv(wind_dir / "band_hour_table.csv", index=False)
    for d in ["DESIGN_IMBAT_2022-07-05", "DESIGN_CALM_2023-07-26"]:
        (wind_dir / d).mkdir(parents=True, exist_ok=True)
        df_hours.to_csv(wind_dir / d / "band_hour_table.csv", index=False)


def build_virtual_mosaics(wind_dir: Path, tile_ids: list[str]):
    """Build GDAL VRT mosaics for ratio16 and both design days."""
    layers = ["ratio16", "DESIGN_IMBAT_2022-07-05", "DESIGN_CALM_2023-07-26"]
    for layer in layers:
        layer_dir = wind_dir / layer
        tifs = [str(layer_dir / f"{tid}.tif") for tid in tile_ids if (layer_dir / f"{tid}.tif").exists()]
        vrt_path = wind_dir / f"{layer}.vrt"
        if tifs:
            print(f"Building VRT mosaic: {vrt_path} ({len(tifs)} tiles)...")
            vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
            gdal.BuildVRT(str(vrt_path), tifs, options=vrt_options)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="City-wide Street-level Wind Screening Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tiles to process")
    parser.add_argument("--tiles", nargs="+", default=None, help="Specific tile IDs to process")
    args = parser.parse_args()

    print("=" * 80)
    print("WP2.2 — City-wide Street-level Wind Screening Pipeline")
    print("Agent: agy3 (laptop, CPU) | Model: 02Geo3D wind.js screening")
    print("=" * 80)

    # Output directory
    wind_dir = PROJECT_ROOT / "data/street/wind"
    wind_dir.mkdir(parents=True, exist_ok=True)
    (wind_dir / "ratio16").mkdir(parents=True, exist_ok=True)
    (wind_dir / "DESIGN_IMBAT_2022-07-05").mkdir(parents=True, exist_ok=True)
    (wind_dir / "DESIGN_CALM_2023-07-26").mkdir(parents=True, exist_ok=True)

    # 1. Load core tiles
    tiles_gdf = gpd.read_file(PROJECT_ROOT / "data/street/grid/tiles_core.gpkg")
    if args.tiles:
        tile_ids = [t for t in args.tiles if t in set(tiles_gdf["tile_id"])]
    else:
        tile_ids = sorted(tiles_gdf["tile_id"].tolist())

    if args.limit:
        tile_ids = tile_ids[:args.limit]

    n_tiles = len(tile_ids)
    print(f"Targeting {n_tiles} core tiles (out of {len(tiles_gdf)} total in grid).")

    # 2. Load met forcing for both design days (06..21 h)
    met_imbat = pd.read_csv(
        PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt",
        sep=r"\s+",
    )
    met_calm = pd.read_csv(
        PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt",
        sep=r"\s+",
    )

    imbat_hours = met_imbat[(met_imbat["it"] >= 6) & (met_imbat["it"] <= 21)].copy()
    calm_hours = met_calm[(met_calm["it"] >= 6) & (met_calm["it"] <= 21)].copy()
    assert len(imbat_hours) == 16, f"Expected 16 hours for IMBAT, got {len(imbat_hours)}"
    assert len(calm_hours) == 16, f"Expected 16 hours for CALM, got {len(calm_hours)}"

    # Generate metadata tables
    generate_band_hour_tables(wind_dir)

    # Band descriptions
    ratio_band_names = [f"sector_{k:02d}_{ang:05.1f}deg" for k, ang in enumerate(SECTOR_ANGLES)]
    hourly_band_names = [f"hour_{int(row['it']):02d}_{int(row['it']):02d}00LST" for _, row in imbat_hours.iterrows()]

    print(f"Memory check: Initial free RAM = {check_ram():.2f} GB.")

    # JIT warmup
    print("Warming up Numba JIT compiler...")
    _dummy_h = np.zeros((20, 20), dtype=np.float32)
    _dummy_h[5:15, 5:15] = 10.0
    _ = wind_ratio(_dummy_h, cell=2.0, wind_from_deg=270.0)

    print(f"Processing {n_tiles} tiles sequentially (one worker, float32)...")

    tile_stats = []
    t_start_all = time.perf_counter()

    for idx, tid in enumerate(tile_ids, 1):
        t0 = time.perf_counter()

        # Enforce memory safety rule
        free_ram = check_ram(min_free_gb=3.0)

        # Load tile geometry (600x600 px at 2m)
        heights, porosity, x0, y0 = load_tile_geometry(tid)

        # Precompute 16 sectors on the full 1200 m buffered grid
        ratio16_full = np.empty((16, 600, 600), dtype=np.float32)
        for k, ang in enumerate(SECTOR_ANGLES):
            ratio16_full[k] = wind_ratio(
                heights=heights,
                porosity=porosity,
                cell=2.0,
                wind_from_deg=float(ang),
            )

        # Slice inner 1000 m (500 x 500 px)
        ratio16_inner = ratio16_full[:, 50:550, 50:550]
        inner_transform = from_origin(x0, y0 + 1000.0, 2.0, 2.0)

        # Save precomputed ratio16
        out_ratio16 = wind_dir / f"ratio16/{tid}.tif"
        write_geotiff_16band(
            out_path=out_ratio16,
            data=ratio16_inner,
            transform=inner_transform,
            band_names=ratio_band_names,
        )

        # Compute hourly pedestrian wind for IMBAT design day (16 bands)
        wind_imbat = np.empty((16, 500, 500), dtype=np.float32)
        for b_idx, (_, row) in enumerate(imbat_hours.iterrows()):
            u10 = float(row["Wind"])
            wd = float(row["Wd"])
            r_hour = interpolate_ratio_at_direction(ratio16_inner, wd)
            wind_imbat[b_idx] = r_hour * u10 * OPEN_PEDESTRIAN_FACTOR

        out_imbat = wind_dir / f"DESIGN_IMBAT_2022-07-05/{tid}.tif"
        write_geotiff_16band(
            out_path=out_imbat,
            data=wind_imbat,
            transform=inner_transform,
            band_names=hourly_band_names,
        )

        # Compute hourly pedestrian wind for CALM design day (16 bands)
        wind_calm = np.empty((16, 500, 500), dtype=np.float32)
        for b_idx, (_, row) in enumerate(calm_hours.iterrows()):
            u10 = float(row["Wind"])
            wd = float(row["Wd"])
            r_hour = interpolate_ratio_at_direction(ratio16_inner, wd)
            wind_calm[b_idx] = r_hour * u10 * OPEN_PEDESTRIAN_FACTOR

        out_calm = wind_dir / f"DESIGN_CALM_2023-07-26/{tid}.tif"
        write_geotiff_16band(
            out_path=out_calm,
            data=wind_calm,
            transform=inner_transform,
            band_names=hourly_band_names,
        )

        # Sanity metrics on inner ratio
        valid_r = ratio16_inner[~np.isnan(ratio16_inner)]
        r_min = float(np.min(valid_r)) if len(valid_r) > 0 else 0.0
        r_max = float(np.max(valid_r)) if len(valid_r) > 0 else 0.0
        r_mean = float(np.mean(valid_r)) if len(valid_r) > 0 else 0.0
        sheltered_share = float(np.mean(valid_r < 0.5)) if len(valid_r) > 0 else 0.0
        nan_share = float(np.isnan(ratio16_inner[0]).mean())

        elapsed = time.perf_counter() - t0
        tile_stats.append({
            "tile_id": tid,
            "runtime_sec": elapsed,
            "free_ram_gb": free_ram,
            "ratio_min": r_min,
            "ratio_max": r_max,
            "ratio_mean": r_mean,
            "sheltered_share": sheltered_share,
            "nan_share": nan_share,
        })

        if idx % 25 == 0 or idx == n_tiles or idx <= 5:
            avg_t = (time.perf_counter() - t_start_all) / idx
            eta_sec = avg_t * (n_tiles - idx)
            print(
                f"[{idx:03d}/{n_tiles}] {tid}: time={elapsed:.2f}s (avg={avg_t:.2f}s) | "
                f"RAM={free_ram:.2f}GB | r_mean={r_mean:.3f} | ETA={eta_sec/60:.1f}min"
            )

        # Explicit garbage collection
        del heights, porosity, ratio16_full, ratio16_inner, wind_imbat, wind_calm
        gc.collect()

    t_total = time.perf_counter() - t_start_all
    avg_runtime = t_total / n_tiles
    print("-" * 80)
    print(f"Processed all {n_tiles} tiles in {t_total:.1f} s ({t_total/60:.2f} min).")
    print(f"Average runtime per tile: {avg_runtime:.3f} s.")

    # Save execution metrics CSV & JSON
    stats_df = pd.DataFrame(tile_stats)
    out_stats_csv = PROJECT_ROOT / "outputs/street/wind/tile_execution_stats.csv"
    out_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_stats_csv, index=False)

    summary_metrics = {
        "num_tiles": n_tiles,
        "total_runtime_sec": t_total,
        "avg_runtime_sec": avg_runtime,
        "min_runtime_sec": float(stats_df["runtime_sec"].min()),
        "max_runtime_sec": float(stats_df["runtime_sec"].max()),
        "min_free_ram_gb": float(stats_df["free_ram_gb"].min()),
        "overall_ratio_min": float(stats_df["ratio_min"].min()),
        "overall_ratio_max": float(stats_df["ratio_max"].max()),
        "overall_ratio_mean": float(stats_df["ratio_mean"].mean()),
        "mean_sheltered_share": float(stats_df["sheltered_share"].mean()),
        "mean_building_coverage": float(stats_df["nan_share"].mean()),
    }
    with open(PROJECT_ROOT / "outputs/street/wind/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Build Virtual Mosaics (VRT)
    build_virtual_mosaics(wind_dir, tile_ids)
    print("=" * 80)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
