"""
CLIMORFA Street — WP3.6c Step 2: Build 2 m Land-Cover Rasters for 40-Tile Subset
================================================================================
Constructs standardized 2 m land-cover GeoTIFF rasters on the 1200x1200m tile grid
(600x600 px, EPSG:5253, DEFLATE) for the stratified 40-tile sensitivity subset.

Conforms strictly to SOLWEIG-GPU / UMEP class conventions:
  - Class 1: Paved (roads_fua buffered by road class + WorldCover 50/60 outside buildings)
  - Class 2: Buildings (building_dsm - dem > 0.5 m)
  - Class 5: Grass / vegetation (WorldCover 10/20/30/40/90 + tree canopy tree_dsm >= 2 m ground)
  - Class 6: Bare soil (reserved/fallback in UMEP)
  - Class 7: Water (WorldCover 80 + sea mask landmask == 0)

Outputs:
  - data/street/sens/landcover/inputs/<tile_id>.tif (uint8)
  - outputs/street/sens/landcover/qa_landcover_shares.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Set PROJ_DATA to avoid PROJ database lookup warnings in conda env
if "PROJ_DATA" not in os.environ and os.path.exists("/home/ye/miniforge3/envs/climorfa-wsl/share/proj"):
    os.environ["PROJ_DATA"] = "/home/ye/miniforge3/envs/climorfa-wsl/share/proj"

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import shapely

PROJECT_ROOT = Path("/mnt/c/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()

# Road class half-widths (meters) based on municipal and geometric design standards
ROAD_HALF_WIDTHS = {
    "OTOYOL": 12.5,     # Motorway: divided carriageway + shoulders
    "BULVAR": 8.0,      # Boulevard / major arterial: multi-lane
    "CADDE": 5.0,       # Avenue / collector: 2-4 lanes with sidewalks
    "SOKAK": 3.5,       # Street: standard residential street / lane
    "MEYDAN": 8.0,      # Square / plaza: open pedestrian/paved space
    "KÜME EVLER": 3.0,  # Cluster / rural access lane
    "DEFAULT": 4.0,     # Default for unclassified or missing road class
}

NODATA_ELEV = -9999.0


def ensure_worldcover_vrt(vrt_path: Path, raw_tiles_dir: Path) -> Path:
    """Ensure the ESA WorldCover 2021 VRT mosaic exists."""
    if vrt_path.exists():
        return vrt_path
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_tiles_dir.glob("*.tif"))
    if not raw_files:
        raise FileNotFoundError(f"No WorldCover GeoTIFFs found in {raw_tiles_dir}")

    # Build VRT using gdalbuildvrt CLI
    cmd = f"gdalbuildvrt {vrt_path} " + " ".join(str(f) for f in raw_files)
    ret = os.system(cmd)
    if ret != 0 or not vrt_path.exists():
        raise RuntimeError(f"Failed to create WorldCover VRT at {vrt_path}")
    return vrt_path


def build_tile_landcover(
    tile_id: str,
    roads_gdf_or_path: str | Path,
    wc_vrt_path: Path,
    out_dir: Path,
    ps_index_dir: Path,
    force: bool = False,
) -> dict:
    """
    Build a single 2 m landcover raster for tile_id and return QA metrics.
    """
    out_path = out_dir / f"{tile_id}.tif"
    if out_path.exists() and not force:
        print(f"[{tile_id}] Landcover input already exists. Loading for QA...")
        with rasterio.open(out_path) as s:
            lc = s.read(1)
            transform = s.transform
            shape = s.shape
    else:
        dem_path = PROJECT_ROOT / f"data/street/tiles/2m/dem/{tile_id}.tif"
        bld_path = PROJECT_ROOT / f"data/street/tiles/2m/building_dsm/{tile_id}.tif"
        tree_path = PROJECT_ROOT / f"data/street/tiles/2m/tree_dsm/{tile_id}.tif"
        lm_path = PROJECT_ROOT / f"data/street/tiles/2m/landmask/{tile_id}.tif"

        for p in (dem_path, bld_path, tree_path, lm_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required tile raster: {p}")

        with rasterio.open(dem_path) as s_dem:
            dem = s_dem.read(1)
            transform = s_dem.transform
            shape = s_dem.shape
            bounds = s_dem.bounds
            crs = s_dem.crs

        with rasterio.open(bld_path) as s_bld:
            bld_dsm = s_bld.read(1)

        with rasterio.open(tree_path) as s_tree:
            tree_dsm = s_tree.read(1)

        with rasterio.open(lm_path) as s_lm:
            landmask = s_lm.read(1)

        # 1. Reproject ESA WorldCover (nearest neighbour) to 600x600 tile grid
        wc_arr = np.zeros(shape, dtype=np.uint8)
        with rasterio.open(wc_vrt_path) as src_wc:
            reproject(
                source=rasterio.band(src_wc, 1),
                destination=wc_arr,
                src_transform=src_wc.transform,
                src_crs=src_wc.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )

        # 2. Base mapping from WorldCover classes
        # Class 5: Vegetation (10 Tree, 20 Shrub, 30 Grass, 40 Crop, 90 Wetland)
        # Class 1: Built-up / bare outside buildings (50 Built-up, 60 Bare/sparse)
        # Class 7: Water (80 Water)
        lc = np.zeros(shape, dtype=np.uint8)
        lc[np.isin(wc_arr, [10, 20, 30, 40, 90])] = 5
        lc[np.isin(wc_arr, [50, 60])] = 1
        lc[wc_arr == 80] = 7

        # Fallback for any unmapped pixels
        lc[lc == 0] = 5

        # 3. Tree canopy cells (tree_dsm >= 2.0 m) count as grass ground beneath
        tree_mask = (tree_dsm >= 2.0) & (tree_dsm != NODATA_ELEV)
        lc[tree_mask] = 5

        # 4. Roads from roads_fua buffered by road class
        bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        df_roads = pyogrio.read_dataframe(
            str(roads_gdf_or_path),
            layer="roads_fua",
            bbox=bbox,
            columns=["YOLTIP", "SERITSAYIS"],
        )

        buffered_geoms = []
        for geom, yt, s in zip(df_roads.geometry, df_roads["YOLTIP"], df_roads["SERITSAYIS"]):
            if geom is None or geom.is_empty:
                continue
            yt_str = str(yt).upper() if yt is not None else "DEFAULT"
            base_hw = ROAD_HALF_WIDTHS.get(yt_str, ROAD_HALF_WIDTHS["DEFAULT"])
            try:
                lanes = int(s) if s is not None and not np.isnan(s) else 0
            except Exception:
                lanes = 0
            hw = max(base_hw, lanes * 1.75) if lanes > 0 else base_hw
            buffered_geoms.append(shapely.buffer(geom, hw, cap_style=1, join_style=1))

        if buffered_geoms:
            road_mask = rasterize(
                [(g, 1) for g in buffered_geoms],
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ) == 1
            lc[road_mask] = 1

        # 5. Buildings: building_dsm - dem > 0.5 m
        bld_h = bld_dsm - dem
        bld_mask = (bld_h > 0.5) & (dem != NODATA_ELEV) & (bld_dsm != NODATA_ELEV)
        lc[bld_mask] = 2

        # 6. Sea mask: landmask == 0 -> class 7 (Water)
        lc[landmask == 0] = 7

        # Write output raster
        out_profile = {
            "driver": "GTiff",
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 1,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(lc, 1)
            dst.set_band_description(1, "SOLWEIG Landcover (1=Paved, 2=Buildings, 5=Grass, 6=Bare, 7=Water)")

    # Compute QA metrics
    total_px = lc.size
    n_paved = int(np.sum(lc == 1))
    n_bld = int(np.sum(lc == 2))
    n_grass = int(np.sum(lc == 5))
    n_bare = int(np.sum(lc == 6))
    n_water = int(np.sum(lc == 7))

    share_paved = n_paved / total_px
    share_bld = n_bld / total_px
    share_grass = n_grass / total_px
    share_bare = n_bare / total_px
    share_water = n_water / total_px

    # Independent check: Pedestrian Space (PS) inside inner 500x500 px
    ps_file = ps_index_dir / f"{tile_id}.tif"
    ps_paved_share = np.nan
    ps_grass_share = np.nan
    n_ps_cells = 0
    if ps_file.exists():
        with rasterio.open(ps_file) as s_ps:
            ps_arr = s_ps.read(1)
        # Inner 500x500 window from (600, 600)
        lc_inner = lc[50:550, 50:550]
        ps_mask = ps_arr > 0
        n_ps_cells = int(np.sum(ps_mask))
        if n_ps_cells > 0:
            ps_paved_share = float(np.mean(lc_inner[ps_mask] == 1))
            ps_grass_share = float(np.mean(lc_inner[ps_mask] == 5))

    # Independent check: WorldCover 10m native class shares on inner tile
    # Reproject or sample WorldCover on inner tile bounds
    wc_built_share = np.nan
    wc_veg_share = np.nan
    wc_water_share = np.nan
    try:
        # Load cropped wc_arr if available
        wc_arr = np.zeros(shape, dtype=np.uint8)
        with rasterio.open(wc_vrt_path) as src_wc:
            reproject(
                source=rasterio.band(src_wc, 1),
                destination=wc_arr,
                src_transform=src_wc.transform,
                src_crs=src_wc.crs,
                dst_transform=transform,
                dst_crs="EPSG:5253",
                resampling=Resampling.nearest,
            )
        wc_inner = wc_arr[50:550, 50:550]
        n_in = wc_inner.size
        wc_built_share = float(np.sum(np.isin(wc_inner, [50, 60])) / n_in)
        wc_veg_share = float(np.sum(np.isin(wc_inner, [10, 20, 30, 40, 90])) / n_in)
        wc_water_share = float(np.sum(wc_inner == 80) / n_in)
    except Exception as e:
        print(f"[{tile_id}] Error sampling WorldCover inner: {e}")

    return {
        "tile_id": tile_id,
        "total_pixels": total_px,
        "share_paved": round(share_paved, 4),
        "share_buildings": round(share_bld, 4),
        "share_grass": round(share_grass, 4),
        "share_bare": round(share_bare, 4),
        "share_water": round(share_water, 4),
        "built_and_paved_share_2m": round(share_paved + share_bld, 4),
        "wc_built_share_10m": round(wc_built_share, 4),
        "wc_veg_share_10m": round(wc_veg_share, 4),
        "wc_water_share_10m": round(wc_water_share, 4),
        "n_ps_cells": n_ps_cells,
        "ps_paved_share": round(ps_paved_share, 4),
        "ps_grass_share": round(ps_grass_share, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="CLIMORFA Street — Land-Cover Tile Builder (WP3.6c)")
    parser.add_argument("--tiles-csv", default=str(PROJECT_ROOT / "data/street/analysis/sens/subset_40_tiles.csv"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data/street/sens/landcover/inputs"))
    parser.add_argument("--qa-csv", default=str(PROJECT_ROOT / "outputs/street/sens/landcover/qa_landcover_shares.csv"))
    parser.add_argument("--roads-gpkg", default=str(PROJECT_ROOT / "data/02_interim/akilli_sehir_fua.gpkg"))
    parser.add_argument("--worldcover-dir", default=str(PROJECT_ROOT / "data/01_raw/gee/esa_worldcover_2021/tiles"))
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing output tiles")

    args = parser.parse_args()

    tiles_df = pd.read_csv(args.tiles_csv)
    tiles = tiles_df["tile_id"].tolist()
    print(f"Loaded {len(tiles)} tiles from {args.tiles_csv}")

    vrt_path = PROJECT_ROOT / "outputs/street/sens/landcover/worldcover_mosaic.vrt"
    ensure_worldcover_vrt(vrt_path, Path(args.worldcover_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ps_index_dir = PROJECT_ROOT / "data/street/analysis/ps_index"

    t0 = time.time()
    records = []
    for idx, row in tiles_df.iterrows():
        tid = row["tile_id"]
        dist = row["district"]
        dom_lcz = int(row["dom_lcz"])

        print(f"[{idx+1:02d}/{len(tiles)}] Building land-cover raster for {tid} ({dist}, LCZ {dom_lcz})...")
        rec = build_tile_landcover(
            tile_id=tid,
            roads_gdf_or_path=args.roads_gpkg,
            wc_vrt_path=vrt_path,
            out_dir=out_dir,
            ps_index_dir=ps_index_dir,
            force=args.force,
        )
        rec["district"] = dist
        rec["dom_lcz"] = dom_lcz
        records.append(rec)

    df_qa = pd.DataFrame(records)
    # Order columns logically
    cols = [
        "tile_id", "district", "dom_lcz", "total_pixels",
        "share_paved", "share_buildings", "share_grass", "share_bare", "share_water",
        "built_and_paved_share_2m", "wc_built_share_10m", "wc_veg_share_10m", "wc_water_share_10m",
        "n_ps_cells", "ps_paved_share", "ps_grass_share"
    ]
    df_qa = df_qa[cols]

    qa_csv_path = Path(args.qa_csv)
    qa_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_qa.to_csv(qa_csv_path, index=False)
    print(f"\nWrote QA table with {len(df_qa)} tiles to {qa_csv_path}")

    elapsed = time.time() - t0
    print(f"All 40 tiles built in {elapsed:.1f} s ({elapsed/len(tiles):.2f} s/tile)")

    # Print summary by LCZ
    print("\n--- Summary Class Shares by Dominant LCZ ---")
    grp = df_qa.groupby("dom_lcz")[["share_paved", "share_buildings", "share_grass", "share_water", "ps_paved_share"]].mean()
    print(grp.to_string())


if __name__ == "__main__":
    main()
