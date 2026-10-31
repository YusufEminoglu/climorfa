"""
CLIMORFA Street — WP2.1 Citywide SOLWEIG-GPU Driver & Pilot Runner
==================================================================
Runs citywide SOLWEIG-GPU microclimate thermal comfort simulations across
Izmir urban-core tiles (EPSG:5253, 2 m resolution, 1000 m inner + 100 m buffer).

Simulations are run for two summer design days:
  1. IMBAT: 2022-07-05 (typical hot day with NW/W sea breeze)
  2. CALM : 2023-07-26 (extreme heatwave with low wind)

Key pipeline design:
  - SVF (153 sky patches) and walls/aspect are computed ONCE per tile and reused
    for both design days (~60% time savings on Day 2).
  - Scratch rasters are staged with canonical EPSG:5253 CRS.
  - Final outputs are cropped to the inner 1000 m (500 x 500 px, 100 m buffer stripped).
  - Extracted hours: 06:00–21:00 local time (16 bands).
  - Output formats:
      * Tmrt   : float32, DEFLATE pred 3, nodata -9999.0
      * UTCI   : float32, DEFLATE pred 3, nodata -9999.0
      * Shadow : uint8 (sunlit % x 100), DEFLATE, nodata 255
      * SVF    : float32, DEFLATE pred 3, nodata -9999.0 (single band, once per tile)
  - Resumable: automatically checks and validates existing GeoTIFFs.
  - Progress logging: writes checkpoint log to logs/wp/WP2.1_progress.csv.
  - Disk management: cleans up intermediate ray-tracing scratch tensors after each tile.
"""

import os
import sys
import time
import shutil
import argparse
import datetime
import gc
import json
import glob
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import rasterio
from rasterio.windows import Window, transform as win_transform
import torch

# Ensure GDAL exceptions are enabled
gdal.UseExceptions()

# Design day configurations
DESIGN_DAYS = {
    "IMBAT": {
        "date": "2022-07-05",
        "met_file": "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt",
        "desc": "Typical summer day with strong afternoon Imbat sea breeze",
    },
    "CALM": {
        "date": "2023-07-26",
        "met_file": "data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt",
        "desc": "Extreme heatwave day with calm synoptic conditions",
    },
}

# Pilot tiles definition
PILOT_TILES = [
    {
        "tile_id": "T_0512_4255",
        "archetype": "Alsancak (Central Konak/Alsancak)",
        "lcz": "LCZ 2 (Compact mid-rise)",
        "reason": "Urban core baseline benchmark: flat alluvial ground, high building coverage (26.9%), deep canyons, and street trees.",
    },
    {
        "tile_id": "T_0508_4248",
        "archetype": "Karabağlar hillside (Cennetçeşme / Salih Omurtak)",
        "lcz": "LCZ 3 / LCZ 7 (Informal dense low-rise on steep slope)",
        "reason": "Extreme topographic relief: DEM range 60.1–237.4 m (177.3 m relief, mean 152.5 m), dense informal hillside settlement (24.0% coverage, 2–3 floors). Tests complex sloped shadow casting.",
    },
    {
        "tile_id": "T_0503_4262",
        "archetype": "Çiğli AOSB (İzmir Atatürk Organize Sanayi Bölgesi)",
        "lcz": "LCZ 8 (Large low-rise industrial)",
        "reason": "Industrial manufacturing hub: flat alluvial coastal plain (DEM -1.0 to 4.0 m, mean 1.6 m), expansive low-rise industrial plants/hangars (28.6% coverage, mean height 8.4 m), 35 intersecting LCZ 8 cells.",
    },
]


def get_canonical_epsg5253_wkt():
    """Return canonical WKT for EPSG:5253 (TUREF / TM27)."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(5253)
    return srs.ExportToWkt()


def validate_geotiff(filepath, expected_bands, expected_shape=(500, 500)):
    """Check if a GeoTIFF exists, is readable, and has expected bands and dimensions."""
    if not os.path.isfile(filepath):
        return False
    try:
        with rasterio.open(filepath) as src:
            if src.count != expected_bands:
                return False
            if src.shape != expected_shape:
                return False
            # Check first and last band can be read
            _ = src.read(1)
            if expected_bands > 1:
                _ = src.read(expected_bands)
        return True
    except Exception:
        return False


def stage_tile_inputs(tile_id, tiles_base_dir, prep_dir, wkt_5253):
    """
    Stage building_dsm, dem, and tree_dsm for a single tile into prep_dir
    with standardized filenames and canonical EPSG:5253 projection.
    """
    bld_src = os.path.join(tiles_base_dir, "building_dsm", f"{tile_id}.tif")
    dem_src = os.path.join(tiles_base_dir, "dem", f"{tile_id}.tif")
    tree_src = os.path.join(tiles_base_dir, "tree_dsm", f"{tile_id}.tif")

    for src_path, name in [(bld_src, "Building_DSM"), (dem_src, "DEM"), (tree_src, "Trees")]:
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Missing required tile raster: {src_path}")
        
        dst_folder = os.path.join(prep_dir, name)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, f"{name}_0_0.tif")

        # Copy and ensure canonical EPSG:5253 CRS
        src_ds = gdal.Open(src_path)
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.CreateCopy(dst_path, src_ds)
        dst_ds.SetProjection(wkt_5253)
        dst_ds.FlushCache()
        dst_ds = None
        src_ds = None


def crop_and_save_svf(raw_svf_path, output_svf_path, wkt_5253):
    """Crop the 600x600 SVF raster to inner 500x500 px and save compressed GeoTIFF."""
    os.makedirs(os.path.dirname(output_svf_path), exist_ok=True)
    with rasterio.open(raw_svf_path) as src:
        win = Window(col_off=50, row_off=50, width=500, height=500)
        arr = src.read(1, window=win)
        out_trans = win_transform(win, src.transform)

        profile = {
            "driver": "GTiff",
            "height": 500,
            "width": 500,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:5253",
            "transform": out_trans,
            "nodata": -9999.0,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(output_svf_path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
            dst.set_band_description(1, "Sky View Factor (SVF, 0 to 1)")


def crop_and_save_comfort_layers(
    work_output_dir,
    tile_id,
    day_name,
    output_base_dir,
    met_file_path,
):
    """
    Crop 24-band simulation rasters to inner 500x500 px, extract hours 06:00–21:00 (16 bands),
    validate physical sanity at 15:00 LST, and save compressed GeoTIFFs.
    """
    raw_tmrt_path = os.path.join(work_output_dir, "0_0", "TMRT_0_0.tif")
    raw_utci_path = os.path.join(work_output_dir, "0_0", "UTCI_0_0.tif")
    raw_shadow_path = os.path.join(work_output_dir, "0_0", "Shadow_0_0.tif")

    for p in [raw_tmrt_path, raw_utci_path, raw_shadow_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing raw simulation output: {p}")

    win = Window(col_off=50, row_off=50, width=500, height=500)

    # 1. Read raw arrays (hours 6 to 21 -> band indices 7 to 22 in 1-based index)
    with rasterio.open(raw_tmrt_path) as src_t, \
         rasterio.open(raw_utci_path) as src_u, \
         rasterio.open(raw_shadow_path) as src_s:

        out_trans = win_transform(win, src_t.transform)
        # Bands 7 through 22 (16 bands: 06:00 to 21:00 inclusive)
        band_indices = list(range(7, 23))
        tmrt_16 = src_t.read(band_indices, window=win)    # shape (16, 500, 500)
        utci_16 = src_u.read(band_indices, window=win)    # shape (16, 500, 500)
        shadow_16 = src_s.read(band_indices, window=win)  # shape (16, 500, 500)

    # 2. Sanity check at 15:00 LST (index 9 in 16-band array, corresponding to hour 15)
    # Read Ta at 15:00 from met file
    df_met = pd.read_csv(met_file_path, sep=r"\s+")
    ta_15 = float(df_met[df_met["it"] == 15]["Td"].values[0])

    tmrt_15 = tmrt_16[9]
    shadow_15 = shadow_16[9]

    # In solweig_gpu: shadow >= 0.5 is sunlit, < 0.5 is shaded
    sunlit_mask = shadow_15 >= 0.5
    shaded_mask = shadow_15 < 0.5

    t_sun = tmrt_15[sunlit_mask]
    t_shd = tmrt_15[shaded_mask]

    sanity = {
        "day": day_name,
        "hour": 15,
        "ta": ta_15,
        "sunlit_px": int(sunlit_mask.sum()),
        "shaded_px": int(shaded_mask.sum()),
        "sunlit_pct": round(float(sunlit_mask.mean() * 100), 1),
        "tmrt_sunlit_median": round(float(np.median(t_sun)), 2) if len(t_sun) > 0 else None,
        "tmrt_sunlit_5th": round(float(np.percentile(t_sun, 5)), 2) if len(t_sun) > 0 else None,
        "tmrt_sunlit_95th": round(float(np.percentile(t_sun, 95)), 2) if len(t_sun) > 0 else None,
        "tmrt_shaded_median": round(float(np.median(t_shd)), 2) if len(t_shd) > 0 else None,
        "tmrt_shaded_5th": round(float(np.percentile(t_shd, 5)), 2) if len(t_shd) > 0 else None,
        "tmrt_shaded_95th": round(float(np.percentile(t_shd, 95)), 2) if len(t_shd) > 0 else None,
        "viol_high_85c": int((tmrt_15 > 85.0).sum()),
        "viol_low_ta10": int((tmrt_15 < (ta_15 - 10.0)).sum()),
    }

    # 3. Write final GeoTIFFs
    out_tmrt_dir = os.path.join(output_base_dir, day_name, "tmrt")
    out_utci_dir = os.path.join(output_base_dir, day_name, "utci")
    out_shadow_dir = os.path.join(output_base_dir, day_name, "shadow")

    os.makedirs(out_tmrt_dir, exist_ok=True)
    os.makedirs(out_utci_dir, exist_ok=True)
    os.makedirs(out_shadow_dir, exist_ok=True)

    out_tmrt_path = os.path.join(out_tmrt_dir, f"{tile_id}.tif")
    out_utci_path = os.path.join(out_utci_dir, f"{tile_id}.tif")
    out_shadow_path = os.path.join(out_shadow_dir, f"{tile_id}.tif")

    float_profile = {
        "driver": "GTiff",
        "height": 500,
        "width": 500,
        "count": 16,
        "dtype": "float32",
        "crs": "EPSG:5253",
        "transform": out_trans,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Tmrt
    with rasterio.open(out_tmrt_path, "w", **float_profile) as dst:
        for b_idx in range(16):
            h = b_idx + 6
            dst.write(tmrt_16[b_idx].astype(np.float32), b_idx + 1)
            dst.set_band_description(b_idx + 1, f"Tmrt_{h:02d}00_LST")

    # UTCI (NaN values converted to nodata -9999.0)
    utci_clean = np.nan_to_num(utci_16, nan=-9999.0).astype(np.float32)
    with rasterio.open(out_utci_path, "w", **float_profile) as dst:
        for b_idx in range(16):
            h = b_idx + 6
            dst.write(utci_clean[b_idx], b_idx + 1)
            dst.set_band_description(b_idx + 1, f"UTCI_{h:02d}00_LST")

    # Shadow (uint8: 0 to 100, where 100 = full sun, 0 = full shade, nodata = 255)
    shadow_uint8 = np.clip(np.round(shadow_16 * 100.0), 0, 100).astype(np.uint8)
    shadow_profile = {
        "driver": "GTiff",
        "height": 500,
        "width": 500,
        "count": 16,
        "dtype": "uint8",
        "crs": "EPSG:5253",
        "transform": out_trans,
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_shadow_path, "w", **shadow_profile) as dst:
        for b_idx in range(16):
            h = b_idx + 6
            dst.write(shadow_uint8[b_idx], b_idx + 1)
            dst.set_band_description(b_idx + 1, f"Shadow_{h:02d}00_LST")

    return sanity


def write_progress_record(csv_path, record):
    """Append a checkpoint record to the progress CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)


def write_band_hour_table(output_csv_path):
    """Write metadata band-to-hour translation table."""
    records = []
    for b in range(1, 17):
        lst_hour = b + 5  # 1 -> 6, 16 -> 21
        utc_hour = lst_hour - 3
        records.append({
            "band": b,
            "local_time_lst": f"{lst_hour:02d}:00",
            "utc_time": f"{utc_hour:02d}:00",
            "tmrt_variable": "Mean Radiant Temperature (Tmrt)",
            "tmrt_units": "°C",
            "tmrt_dtype": "float32",
            "utci_variable": "Universal Thermal Climate Index (UTCI)",
            "utci_units": "°C",
            "utci_dtype": "float32",
            "shadow_variable": "Sunlit Surface Fraction",
            "shadow_units": "% (0=shaded, 100=sunlit)",
            "shadow_dtype": "uint8",
        })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Wrote band-to-hour table: {output_csv_path}")


def build_vrts(output_base_dir, days=["IMBAT", "CALM"]):
    """Build VRT virtual mosaics for all generated layers and days."""
    print("\nBuilding VRTs...")
    # 1. SVF VRT
    svf_files = sorted(glob.glob(os.path.join(output_base_dir, "svf", "*.tif")))
    if svf_files:
        svf_vrt = os.path.join(output_base_dir, "svf.vrt")
        vrt_ds = gdal.BuildVRT(svf_vrt, svf_files)
        vrt_ds.FlushCache()
        vrt_ds = None
        print(f"  Built SVF VRT ({len(svf_files)} tiles): {svf_vrt}")

    # 2. Per-day layers (tmrt, utci, shadow)
    for day in days:
        for lyr in ["tmrt", "utci", "shadow"]:
            files = sorted(glob.glob(os.path.join(output_base_dir, day, lyr, "*.tif")))
            if files:
                vrt_path = os.path.join(output_base_dir, day, f"{lyr}.vrt")
                vrt_ds = gdal.BuildVRT(vrt_path, files)
                vrt_ds.FlushCache()
                vrt_ds = None
                print(f"  Built {day}/{lyr}.vrt ({len(files)} tiles): {vrt_path}")


def process_single_tile(
    tile_id,
    days,
    tiles_base_dir,
    output_base_dir,
    scratch_base_dir,
    progress_csv_path,
    force=False,
):
    """
    Execute full pipeline for a single tile across requested days.
    Reuses walls, aspect, and SVF between days.
    """
    import solweig_gpu

    wkt_5253 = get_canonical_epsg5253_wkt()
    tile_scratch = os.path.join(scratch_base_dir, tile_id)
    prep_dir = os.path.join(tile_scratch, "prep")
    
    # Check if all days are already completed
    days_to_run = []
    for d in days:
        t_ok = validate_geotiff(os.path.join(output_base_dir, d, "tmrt", f"{tile_id}.tif"), 16)
        u_ok = validate_geotiff(os.path.join(output_base_dir, d, "utci", f"{tile_id}.tif"), 16)
        s_ok = validate_geotiff(os.path.join(output_base_dir, d, "shadow", f"{tile_id}.tif"), 16)
        if force or not (t_ok and u_ok and s_ok):
            days_to_run.append(d)

    svf_out_path = os.path.join(output_base_dir, "svf", f"{tile_id}.tif")
    svf_exists = validate_geotiff(svf_out_path, 1)

    if not days_to_run and (svf_exists and not force):
        print(f"[{tile_id}] All requested outputs exist and validate. Skipping.")
        return []

    print(f"\n=======================================================")
    print(f"  Processing Tile: {tile_id} (Days to run: {days_to_run})")
    print(f"=======================================================")

    sanity_reports = []

    try:
        # Step A: Stage rasters
        if os.path.exists(tile_scratch):
            shutil.rmtree(tile_scratch)
        os.makedirs(prep_dir, exist_ok=True)
        os.makedirs(os.path.join(prep_dir, "metfiles"), exist_ok=True)

        print(f"[{tile_id}] Staging inputs with EPSG:5253 CRS...")
        stage_tile_inputs(tile_id, tiles_base_dir, prep_dir, wkt_5253)

        # Step B: Walls & Aspect
        print(f"[{tile_id}] Computing walls and aspect...")
        t0 = time.time()
        solweig_gpu.run_walls_aspect(prep_dir)
        t_walls = time.time() - t0
        print(f"[{tile_id}] Walls/aspect completed in {t_walls:.2f} s")

        # Step C: SVF Calculation
        print(f"[{tile_id}] Computing standalone SVF (153 patches)...")
        t0 = time.time()
        solweig_gpu.calculate_svf(prep_dir, patch_option=2, overwrite=True)
        t_svf = time.time() - t0
        print(f"[{tile_id}] SVF completed in {t_svf:.2f} s")

        # Save cropped SVF
        raw_svf = os.path.join(prep_dir, "SVF", "SkyViewFactor_0_0.tif")
        crop_and_save_svf(raw_svf, svf_out_path, wkt_5253)
        # Also copy to per-day svf folders for convenience
        for d in days:
            d_svf_path = os.path.join(output_base_dir, d, "svf", f"{tile_id}.tif")
            os.makedirs(os.path.dirname(d_svf_path), exist_ok=True)
            shutil.copy(svf_out_path, d_svf_path)

        # Step D: Run requested days
        for day_name in days_to_run:
            day_cfg = DESIGN_DAYS[day_name]
            met_source = day_cfg["met_file"]
            date_str = day_cfg["date"]

            day_work_dir = os.path.join(tile_scratch, f"work_{day_name}")
            os.makedirs(day_work_dir, exist_ok=True)

            # Copy met file
            target_met = os.path.join(prep_dir, "metfiles", "metfile_0_0.txt")
            shutil.copy(met_source, target_met)

            print(f"\n[{tile_id}] Running {day_name} ({date_str}) [reusing SVF & walls]...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            t_start_iso = datetime.datetime.now().isoformat()
            t_start = time.time()

            solweig_gpu.run_utci_tiles(
                base_path=day_work_dir,
                preprocess_dir=prep_dir,
                selected_date_str=date_str,
                save_tmrt=True,
                save_svf=False,  # SVF already saved from standalone stage
                save_shadow=True,
                save_ta=False,
                save_wind=False,
            )

            t_elapsed = round(time.time() - t_start, 2)
            t_end_iso = datetime.datetime.now().isoformat()
            peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

            print(f"[{tile_id}] {day_name} completed in {t_elapsed:.2f} s. Peak VRAM: {peak_vram} MB")

            # Post-process, crop, validate sanity
            sanity = crop_and_save_comfort_layers(
                work_output_dir=os.path.join(day_work_dir, "output_folder"),
                tile_id=tile_id,
                day_name=day_name,
                output_base_dir=output_base_dir,
                met_file_path=met_source,
            )
            sanity["tile_id"] = tile_id
            sanity["seconds"] = t_elapsed
            sanity["peak_vram_mb"] = peak_vram
            sanity_reports.append(sanity)

            # Log to progress CSV
            write_progress_record(
                progress_csv_path,
                {
                    "tile": tile_id,
                    "day": day_name,
                    "start": t_start_iso,
                    "end": t_end_iso,
                    "seconds": t_elapsed,
                    "peak_vram_mb": peak_vram,
                    "status": "OK",
                },
            )

        print(f"[{tile_id}] Successfully finished all stages.")

    except Exception as e:
        print(f"[{tile_id}] ERROR: {str(e)}")
        # Log failure
        write_progress_record(
            progress_csv_path,
            {
                "tile": tile_id,
                "day": "FAILED",
                "start": datetime.datetime.now().isoformat(),
                "end": datetime.datetime.now().isoformat(),
                "seconds": 0.0,
                "peak_vram_mb": 0.0,
                "status": f"ERROR: {str(e)}",
            },
        )
        raise e

    finally:
        # Clean up scratch folder immediately to save disk space
        if os.path.exists(tile_scratch):
            shutil.rmtree(tile_scratch)
        gc.collect()
        torch.cuda.empty_cache()

    return sanity_reports


def run_pilot():
    """Execute Step 1 (Pilot on 3 representative tiles)."""
    print("\n=======================================================")
    print("  CLIMORFA Street — WP2.1 STEP 1 (PILOT EXECUTION)")
    print("=======================================================\n")

    tiles_base_dir = "data/street/tiles/2m"
    output_base_dir = "data/street/solweig"
    scratch_base_dir = "outputs/street/solweig/_scratch"
    progress_csv_path = "logs/wp/WP2.1_progress.csv"
    pilot_json_path = "outputs/street/solweig/pilot_summary.json"
    band_hour_csv_path = "data/street/solweig/band_hour_table.csv"

    os.makedirs("outputs/street/solweig", exist_ok=True)
    os.makedirs("logs/wp", exist_ok=True)

    # 1. Write band-to-hour translation table
    write_band_hour_table(band_hour_csv_path)
    write_band_hour_table("outputs/street/solweig/band_hour_table.csv")

    all_pilot_reports = []

    for pilot_info in PILOT_TILES:
        tid = pilot_info["tile_id"]
        print(f"\n>>> Running Pilot Tile: {tid} ({pilot_info['archetype']})")
        print(f"    Rationale: {pilot_info['reason']}")

        reports = process_single_tile(
            tile_id=tid,
            days=["IMBAT", "CALM"],
            tiles_base_dir=tiles_base_dir,
            output_base_dir=output_base_dir,
            scratch_base_dir=scratch_base_dir,
            progress_csv_path=progress_csv_path,
            force=True,  # ensure fresh benchmark for pilot
        )
        for r in reports:
            r["archetype"] = pilot_info["archetype"]
            r["lcz"] = pilot_info["lcz"]
        all_pilot_reports.extend(reports)

    # 2. Build VRTs for pilot tiles
    build_vrts(output_base_dir, days=["IMBAT", "CALM"])

    # 3. Save pilot summary JSON
    with open(pilot_json_path, "w", encoding="utf-8") as f:
        json.dump(all_pilot_reports, f, indent=2)
    print(f"\nSaved pilot summary report: {pilot_json_path}")

    return all_pilot_reports


def run_full(tile_list_file="data/street/grid/tiles_core.gpkg", days=["IMBAT", "CALM"]):
    """Execute Full Run (427 tiles x 2 days)."""
    import geopandas as gpd

    print("\n=======================================================")
    print("  CLIMORFA Street — WP2.1 FULL 427-TILE CITYWIDE RUN")
    print("=======================================================\n")

    gdf = gpd.read_file(tile_list_file)
    tile_ids = gdf["tile_id"].tolist()
    print(f"Loaded {len(tile_ids)} tiles from {tile_list_file}")

    tiles_base_dir = "data/street/tiles/2m"
    output_base_dir = "data/street/solweig"
    scratch_base_dir = "outputs/street/solweig/_scratch"
    progress_csv_path = "logs/wp/WP2.1_progress.csv"
    band_hour_csv_path = "data/street/solweig/band_hour_table.csv"

    write_band_hour_table(band_hour_csv_path)

    for idx, tid in enumerate(tile_ids):
        print(f"\n[{idx+1}/{len(tile_ids)}] Tile {tid}")
        try:
            process_single_tile(
                tile_id=tid,
                days=days,
                tiles_base_dir=tiles_base_dir,
                output_base_dir=output_base_dir,
                scratch_base_dir=scratch_base_dir,
                progress_csv_path=progress_csv_path,
                force=False,
            )
        except Exception as e:
            print(f"Continuing to next tile after error on {tid}: {e}")

    # Build final VRTs
    build_vrts(output_base_dir, days=days)
    print("\nFull citywide simulation completed.")


def main():
    parser = argparse.ArgumentParser(description="CLIMORFA SOLWEIG-GPU Runner")
    parser.add_argument("--pilot", action="store_true", help="Execute Step 1 (Pilot on 3 tiles only)")
    parser.add_argument("--full", action="store_true", help="Execute Step 2 (Full 427-tile citywide run)")
    parser.add_argument("--tiles", type=str, default=None, help="Comma-separated tile IDs to run")
    parser.add_argument("--days", type=str, default="IMBAT,CALM", help="Comma-separated days to run")
    parser.add_argument("--build-vrts", action="store_true", help="Build VRTs and exit")
    args = parser.parse_args()

    days = [d.strip() for d in args.days.split(",")]

    if args.build_vrts:
        build_vrts("data/street/solweig", days=days)
        return

    if args.pilot:
        run_pilot()
    elif args.full:
        run_full(days=days)
    elif args.tiles:
        tile_ids = [t.strip() for t in args.tiles.split(",")]
        for tid in tile_ids:
            process_single_tile(
                tile_id=tid,
                days=days,
                tiles_base_dir="data/street/tiles/2m",
                output_base_dir="data/street/solweig",
                scratch_base_dir="outputs/street/solweig/_scratch",
                progress_csv_path="logs/wp/WP2.1_progress.csv",
                force=False,
            )
        build_vrts("data/street/solweig", days=days)
    else:
        print("Please specify --pilot, --full, or --tiles <tile1,tile2>")


if __name__ == "__main__":
    main()
