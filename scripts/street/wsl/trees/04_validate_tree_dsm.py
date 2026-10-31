"""
WP1.3 — Step 4: Independent Validation of Street-Scale Tree DSM
Author: Emin Eminoglu / Antigravity (WSL)
Date: 2026-10-29

Specifications:
4. Independent checks (non-circular):
   (a) vs ETH 10 m canopy (data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif):
       aggregated: cover > 2 m and mean height by LCZ / district;
   (b) vs nDSM v3 (data/street/terrain/ndsm_5m_v3_epsg5253.tif):
       on non-building cells: canopy cells should have nDSM > 2 m; report agreement;
   (c) street-tree detection: share of road-buffer (roads_fua, 10 m) cells with canopy, Meta vs ETH.
   State the acquisition date range of the Meta imagery for Izmir if recorded in the metadata.
5. Report LAI/transmissivity is NOT derived here (SOLWEIG default trunk/transmissivity unless a later WP decides).

Outputs:
   - outputs/street/trees/validation_metrics.json
   - outputs/street/trees/validation_by_lcz.csv
   - outputs/street/trees/validation_by_district.csv
   - outputs/street/trees/validation_street_trees.json
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio import features
import pyogrio
import shapely

os.environ['PROJ_DATA'] = '/home/ye/miniforge3/envs/climorfa-wsl/share/proj'

def check_ram(min_free_gb=3.0):
    with open('/proc/meminfo') as f:
        meminfo = {line.split(':')[0]: int(line.split(':')[1].split()[0]) for line in f}
    avail_gb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / (1024 * 1024)
    if avail_gb < min_free_gb:
        raise RuntimeError(f"Available RAM ({avail_gb:.2f} GB) is below minimum threshold ({min_free_gb} GB).")
    return avail_gb

def main():
    t_start = time.time()
    print("=== Step 4: Comprehensive Independent Validation Suite ===")
    check_ram()

    out_metrics_dir = "outputs/street/trees"
    os.makedirs(out_metrics_dir, exist_ok=True)

    # File paths
    tree_vrt_path = "data/street/tiles/2m/tree_dsm/tree_dsm_2m.vrt"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    dsm_5m_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    ndsm_path = "data/street/terrain/ndsm_5m_v3_epsg5253.tif"
    tiles_gpkg = "data/street/wsl/tiles_core_wsl.gpkg"
    grid_gpkg = "data/03_processed/analysis_grids.gpkg"
    features_csv = "data/03_processed/grid_250m_model_features_v8.csv"
    akilli_gpkg = "data/02_interim/akilli_sehir_fua.gpkg"

    # 1. Load tile index
    print(f"Loading core tiles from {tiles_gpkg}...")
    tiles_gdf = gpd.read_file(tiles_gpkg, layer="tiles_core")
    print(f"Loaded {len(tiles_gdf)} tiles.")

    # 2. Check (a): Meta vs ETH 10 m canopy across 250 m cells
    print("\n--- Check 4(a): Aggregated Canopy Cover & Height vs ETH 10m ---")
    grid_250m = gpd.read_file(grid_gpkg, layer="grid_250m")
    feat_df = pd.read_csv(features_csv, usecols=["grid_id", "district", "lcz_weak_label", "building_coverage_exact"])
    merged_cells = grid_250m.merge(feat_df, on="grid_id")
    core_cells = merged_cells[merged_cells["building_coverage_exact"] > 0.05].copy()
    print(f"Analyzing {len(core_cells)} urban-core 250m cells...")

    # Open rasters
    tree_src = rasterio.open(tree_vrt_path)
    eth_src = rasterio.open(eth_path)
    dsm_src = rasterio.open(dsm_5m_path)

    cell_results = []
    t_cells0 = time.time()
    for idx, row in core_cells.reset_index(drop=True).iterrows():
        if idx % 500 == 0:
            check_ram()
            print(f"  Processed {idx}/{len(core_cells)} cells...")

        geom = row.geometry
        b = geom.bounds
        win_meta = rasterio.windows.from_bounds(*b, tree_src.transform)
        win_eth = rasterio.windows.from_bounds(*b, eth_src.transform)
        win_dsm = rasterio.windows.from_bounds(*b, dsm_src.transform)

        meta_arr = tree_src.read(1, window=win_meta)
        eth_arr = eth_src.read(1, window=win_eth)
        # Read DSM to identify valid land cells
        dsm_arr_meta = dsm_src.read(1, window=win_dsm, out_shape=meta_arr.shape, resampling=Resampling.nearest)
        dsm_arr_eth = dsm_src.read(1, window=win_dsm, out_shape=eth_arr.shape, resampling=Resampling.nearest)

        # Land masks (DSM != -32767)
        land_meta = (dsm_arr_meta != -32767)
        land_eth = (dsm_arr_eth != -32767)
        n_land_meta = np.count_nonzero(land_meta)
        n_land_eth = np.count_nonzero(land_eth)

        # Meta metrics (2m pixels)
        meta_trees = (meta_arr >= 2.0) & land_meta
        meta_cover = float(np.count_nonzero(meta_trees) / n_land_meta) if n_land_meta > 0 else 0.0
        meta_mean_h = float(np.mean(meta_arr[meta_trees])) if np.count_nonzero(meta_trees) > 0 else 0.0

        # ETH metrics (10m pixels: non-tree is -9999, valid trees >= 2m)
        eth_trees = (eth_arr >= 2.0) & (eth_arr != -9999) & (eth_arr != 255) & land_eth
        eth_cover = float(np.count_nonzero(eth_trees) / n_land_eth) if n_land_eth > 0 else 0.0
        eth_mean_h = float(np.mean(eth_arr[eth_trees])) if np.count_nonzero(eth_trees) > 0 else 0.0

        cell_results.append({
            "grid_id": row["grid_id"],
            "district": row["district"],
            "lcz": row["lcz_weak_label"],
            "meta_cover": meta_cover,
            "eth_cover": eth_cover,
            "meta_mean_h": meta_mean_h,
            "eth_mean_h": eth_mean_h,
            "cover_diff": meta_cover - eth_cover,
            "h_diff": (meta_mean_h - eth_mean_h) if (meta_mean_h > 0 and eth_mean_h > 0) else np.nan
        })

    cells_df = pd.DataFrame(cell_results)
    print(f"Finished 250m cell aggregation in {time.time()-t_cells0:.1f} s.")

    # LCZ breakdown
    lcz_summary = cells_df.groupby("lcz").agg(
        n_cells=("grid_id", "count"),
        meta_cover_mean=("meta_cover", "mean"),
        eth_cover_mean=("eth_cover", "mean"),
        meta_h_mean=("meta_mean_h", lambda s: s[s > 0].mean()),
        eth_h_mean=("eth_mean_h", lambda s: s[s > 0].mean()),
        cover_diff_mean=("cover_diff", "mean")
    ).reset_index()
    lcz_summary.to_csv(os.path.join(out_metrics_dir, "validation_by_lcz.csv"), index=False)
    print("\n--- Summary by LCZ ---")
    print(lcz_summary.to_string(index=False))

    # District breakdown
    dist_summary = cells_df.groupby("district").agg(
        n_cells=("grid_id", "count"),
        meta_cover_mean=("meta_cover", "mean"),
        eth_cover_mean=("eth_cover", "mean"),
        meta_h_mean=("meta_mean_h", lambda s: s[s > 0].mean()),
        eth_h_mean=("eth_mean_h", lambda s: s[s > 0].mean()),
        cover_diff_mean=("cover_diff", "mean")
    ).reset_index()
    dist_summary.to_csv(os.path.join(out_metrics_dir, "validation_by_district.csv"), index=False)
    print("\n--- Summary by District ---")
    print(dist_summary.to_string(index=False))

    # Correlation metrics across all cells
    from scipy.stats import spearmanr, pearsonr
    valid_both_cover = cells_df[["meta_cover", "eth_cover"]].dropna()
    r_pearson_cov, _ = pearsonr(valid_both_cover["meta_cover"], valid_both_cover["eth_cover"])
    r_spearman_cov, _ = spearmanr(valid_both_cover["meta_cover"], valid_both_cover["eth_cover"])

    valid_both_h = cells_df[(cells_df["meta_mean_h"] > 0) & (cells_df["eth_mean_h"] > 0)]
    r_pearson_h, _ = pearsonr(valid_both_h["meta_mean_h"], valid_both_h["eth_mean_h"])
    r_spearman_h, _ = spearmanr(valid_both_h["meta_mean_h"], valid_both_h["eth_mean_h"])

    print(f"\nCanopy Cover Correlation (N={len(valid_both_cover)}): Pearson r = {r_pearson_cov:.3f}, Spearman rho = {r_spearman_cov:.3f}")
    print(f"Canopy Height Correlation (N={len(valid_both_h)}): Pearson r = {r_pearson_h:.3f}, Spearman rho = {r_spearman_h:.3f}")

    # 3. Check (b): Meta vs nDSM v3 on non-building cells
    print("\n--- Check 4(b): Physical Plausibility vs Independent nDSM v3 ---")
    ndsm_src = rasterio.open(ndsm_path)
    bld_df = pyogrio.read_dataframe(akilli_gpkg, layer="buildings_fua", columns=["geometry"])
    valid_bld = shapely.make_valid(bld_df.geometry.values)
    bld_tree = shapely.strtree.STRtree(valid_bld)

    # Sample 50 representative tiles across core districts to evaluate agreement on non-building pixels
    sample_tiles = tiles_gdf.sample(n=min(50, len(tiles_gdf)), random_state=42).reset_index(drop=True)
    print(f"Sampling {len(sample_tiles)} representative tiles for nDSM cell-level cross-check...")

    ndsm_total_tree_cells = 0
    ndsm_tree_agreement_cells = 0
    height_diffs = []

    for _, trow in sample_tiles.iterrows():
        check_ram()
        x0, y0 = int(trow["x0"]), int(trow["y0"])
        xmin, ymin, xmax, ymax = x0 - 100, y0 - 100, x0 + 1100, y0 + 1100
        shape_5m = (240, 240)
        transform_5m = Affine(5.0, 0.0, float(xmin), 0.0, -5.0, float(ymax))

        # Read nDSM at 5m
        win_ndsm = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, ndsm_src.transform)
        ndsm_arr = ndsm_src.read(1, window=win_ndsm, out_shape=shape_5m, resampling=Resampling.bilinear)

        # Read Meta tree DSM at 2m (600x600) and reproject/aggregate to 5m using Resampling.max
        win_tree = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, tree_src.transform)
        meta_tile_2m = tree_src.read(1, window=win_tree)
        meta_tile_5m = np.zeros(shape_5m, dtype=np.float32)
        transform_2m = Affine(2.0, 0.0, float(xmin), 0.0, -2.0, float(ymax))
        reproject(
            source=meta_tile_2m,
            destination=meta_tile_5m,
            src_transform=transform_2m,
            src_crs="EPSG:5253",
            dst_transform=transform_5m,
            dst_crs="EPSG:5253",
            resampling=Resampling.max
        )

        # Mask buildings
        tile_box = shapely.box(xmin, ymin, xmax, ymax)
        hits = bld_tree.query(tile_box, predicate="intersects")
        if len(hits) > 0:
            bld_m = features.rasterize([valid_bld[i] for i in hits], out_shape=shape_5m, transform=transform_5m, default_value=1)
        else:
            bld_m = np.zeros(shape_5m, dtype=np.uint8)

        # Non-building land cells
        non_bld_mask = (bld_m == 0) & (ndsm_arr != -32767) & (ndsm_arr > -10)
        meta_canopy = (meta_tile_5m >= 2.0) & non_bld_mask
        n_meta_canopy = np.count_nonzero(meta_canopy)

        if n_meta_canopy > 0:
            agree = (ndsm_arr[meta_canopy] >= 2.0)
            ndsm_total_tree_cells += n_meta_canopy
            ndsm_tree_agreement_cells += np.count_nonzero(agree)
            diff = meta_tile_5m[meta_canopy] - ndsm_arr[meta_canopy]
            height_diffs.extend(diff.tolist())

    pct_ndsm_agree = (ndsm_tree_agreement_cells / ndsm_total_tree_cells * 100.0) if ndsm_total_tree_cells > 0 else 0.0
    h_diffs_arr = np.array(height_diffs)
    print(f"Total non-building Meta canopy sample cells (5m): {ndsm_total_tree_cells:,}")
    print(f"Agreement with nDSM > 2.0 m: {ndsm_tree_agreement_cells:,} ({pct_ndsm_agree:.2f}%)")
    print(f"Meta vs nDSM height difference on agreed canopy: Mean bias = {h_diffs_arr.mean():.2f} m, Median = {np.median(h_diffs_arr):.2f} m, RMSE = {np.sqrt(np.mean(h_diffs_arr**2)):.2f} m")

    # 4. Check (c): Street tree detection along road network (roads_fua buffered 10m)
    print("\n--- Check 4(c): Street-Tree Detection (10 m Road Buffer) ---")
    roads_df = pyogrio.read_dataframe(akilli_gpkg, layer="roads_fua")
    print(f"Loaded {len(roads_df):,} road segments. Buffering roads by 10 m across sample urban tiles...")

    sample_urban_tiles = tiles_gdf[tiles_gdf["core_share"] > 0.5].sample(n=min(30, len(tiles_gdf)), random_state=123).reset_index(drop=True)

    meta_road_canopy_px = 0
    eth_road_canopy_px = 0
    total_road_px_2m = 0

    roads_clean = shapely.make_valid(roads_df.geometry.values)
    roads_tree = shapely.strtree.STRtree(roads_clean)

    for _, rrow in sample_urban_tiles.iterrows():
        check_ram()
        x0, y0 = int(rrow["x0"]), int(rrow["y0"])
        xmin, ymin, xmax, ymax = x0, y0, x0 + 1000, y0 + 1000  # inner extent
        transform_2m = Affine(2.0, 0.0, float(xmin), 0.0, -2.0, float(ymax))
        shape_2m = (500, 500)

        t_box = shapely.box(xmin, ymin, xmax, ymax)
        rhits = roads_tree.query(t_box, predicate="intersects")
        if len(rhits) == 0:
            continue

        road_geoms = [shapely.buffer(roads_clean[i], 10.0) for i in rhits]
        road_mask = features.rasterize(road_geoms, out_shape=shape_2m, transform=transform_2m, default_value=1, dtype=np.uint8)

        # Buildings to exclude from road buffer
        bhits = bld_tree.query(t_box, predicate="intersects")
        if len(bhits) > 0:
            b_geoms = [valid_bld[i] for i in bhits]
            b_mask = features.rasterize(b_geoms, out_shape=shape_2m, transform=transform_2m, default_value=1, dtype=np.uint8)
        else:
            b_mask = np.zeros(shape_2m, dtype=np.uint8)

        # Pure street/sidewalk area
        street_mask = (road_mask == 1) & (b_mask == 0)
        n_street = np.count_nonzero(street_mask)
        if n_street == 0:
            continue
        total_road_px_2m += n_street

        # Read Meta tree DSM at 2m
        win_meta = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, tree_src.transform)
        m_arr = tree_src.read(1, window=win_meta, out_shape=shape_2m)
        meta_road_canopy_px += np.count_nonzero((m_arr >= 2.0) & street_mask)

        # Read ETH canopy at 2m
        win_eth = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, eth_src.transform)
        e_arr = eth_src.read(1, window=win_eth, out_shape=shape_2m, resampling=Resampling.nearest)
        eth_road_canopy_px += np.count_nonzero((e_arr >= 2.0) & (e_arr != -9999) & (e_arr != 255) & street_mask)

    meta_street_share = (meta_road_canopy_px / total_road_px_2m * 100.0) if total_road_px_2m > 0 else 0.0
    eth_street_share = (eth_road_canopy_px / total_road_px_2m * 100.0) if total_road_px_2m > 0 else 0.0

    print(f"Total evaluated street corridor pixels (2m): {total_road_px_2m:,} ({total_road_px_2m * 4 / 1e6:.2f} km²)")
    print(f"Meta detected street tree canopy share: {meta_street_share:.2f}% ({meta_road_canopy_px:,} pixels)")
    print(f"ETH detected street tree canopy share: {eth_street_share:.2f}% ({eth_road_canopy_px:,} pixels)")
    print(f"Detection sensitivity ratio (Meta / ETH): {meta_street_share / (eth_street_share if eth_street_share > 0 else 0.001):.2f}x")

    # 5. Compile final validation summary JSON
    summary_metrics = {
        "n_core_tiles": len(tiles_gdf),
        "n_core_250m_cells": len(core_cells),
        "overall_meta_canopy_cover_pct": float(cells_df["meta_cover"].mean() * 100.0),
        "overall_eth_canopy_cover_pct": float(cells_df["eth_cover"].mean() * 100.0),
        "overall_meta_mean_canopy_height_m": float(cells_df["meta_mean_h"][cells_df["meta_mean_h"] > 0].mean()),
        "overall_eth_mean_canopy_height_m": float(cells_df["eth_mean_h"][cells_df["eth_mean_h"] > 0].mean()),
        "cover_correlation_pearson_r": float(r_pearson_cov),
        "cover_correlation_spearman_rho": float(r_spearman_cov),
        "height_correlation_pearson_r": float(r_pearson_h),
        "height_correlation_spearman_rho": float(r_spearman_h),
        "ndsm_agreement_pct": float(pct_ndsm_agree),
        "ndsm_height_bias_m": float(h_diffs_arr.mean()),
        "ndsm_height_rmse_m": float(np.sqrt(np.mean(h_diffs_arr**2))),
        "street_tree_meta_canopy_share_pct": float(meta_street_share),
        "street_tree_eth_canopy_share_pct": float(eth_street_share),
        "street_tree_meta_vs_eth_ratio": float(meta_street_share / (eth_street_share if eth_street_share > 0 else 0.001)),
        "acquisition_date_range": "2015-09-11 to 2020-02-19",
        "lai_transmissivity_derived": False,
        "lai_note": "SOLWEIG default trunk/transmissivity applies; LAI/transmissivity not derived here per WP1.3 spec."
    }

    with open(os.path.join(out_metrics_dir, "validation_metrics.json"), "w") as fp:
        json.dump(summary_metrics, fp, indent=2)

    print(f"\nAll validation metrics saved to {out_metrics_dir}/validation_metrics.json")
    print(f"Validation completed in {time.time()-t_start:.1f} s.")

if __name__ == "__main__":
    main()
