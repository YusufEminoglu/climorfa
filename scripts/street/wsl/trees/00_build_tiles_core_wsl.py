"""
WP1.3 — Step 0: Generate tiles_core_wsl.gpkg per grid_spec.md
Author: Emin Eminoglu / Antigravity (WSL)
Date: 2026-10-29

Specifications from grid_spec.md:
- CRS: EPSG:5253
- Inner tile extent: 1000 m x 1000 m, corners at integer multiples of 1000 m.
- Urban-core mask: 250 m grid cells with building_coverage_exact > 0.05
  (data/03_processed/grid_250m_model_features_v8.csv joined to data/03_processed/analysis_grids.gpkg layer grid_250m).
- In-scope tile: inner extent intersects the urban-core mask.
- Output: data/street/wsl/tiles_core_wsl.gpkg (inner polygons + tile_id + core_share).
"""

import os
import sys
import time
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box

os.environ['PROJ_DATA'] = '/home/ye/miniforge3/envs/climorfa-wsl/share/proj'

def main():
    t0 = time.time()
    out_path = "data/street/wsl/tiles_core_wsl.gpkg"
    print(f"=== Generating Core Tile Index: {out_path} ===")

    # 1. Load 250m grid and features
    grid_path = "data/03_processed/analysis_grids.gpkg"
    feat_path = "data/03_processed/grid_250m_model_features_v8.csv"
    
    print(f"Loading 250m grid from {grid_path}...")
    gdf = gpd.read_file(grid_path, layer="grid_250m")
    print(f"Loaded {len(gdf)} grid cells. CRS: {gdf.crs}")

    print(f"Loading features from {feat_path}...")
    df_feat = pd.read_csv(feat_path, usecols=["grid_id", "building_coverage_exact"])
    
    # 2. Merge and filter core mask
    merged = gdf.merge(df_feat, on="grid_id")
    core_cells = merged[merged["building_coverage_exact"] > 0.05].copy()
    n_core = len(core_cells)
    area_km2 = core_cells.geometry.area.sum() / 1e6
    print(f"Urban-core mask cells (building_coverage_exact > 0.05): {n_core:,} cells ({area_km2:.2f} km²)")
    assert n_core == 3647, f"Expected 3,647 core cells per grid_spec.md, got {n_core}"

    # 3. Form spatial union of core cells for intersection testing
    print("Computing urban core union...")
    core_union = core_cells.geometry.union_all()

    # 4. Determine bounding grid in multiples of 1000m
    bounds = core_cells.total_bounds
    min_x = int(np.floor(bounds[0] / 1000.0) * 1000)
    min_y = int(np.floor(bounds[1] / 1000.0) * 1000)
    max_x = int(np.ceil(bounds[2] / 1000.0) * 1000)
    max_y = int(np.ceil(bounds[3] / 1000.0) * 1000)
    print(f"Grid coordinate range: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]")

    # 5. Test each 1000m tile
    tile_records = []
    for x in range(min_x, max_x, 1000):
        for y in range(min_y, max_y, 1000):
            tile_box = box(x, y, x + 1000, y + 1000)
            if tile_box.intersects(core_union):
                inter = tile_box.intersection(core_union)
                core_share = float(inter.area / 1e6)  # 1000x1000 m2
                tile_id = f"T_{x//1000:04d}_{y//1000:04d}"
                tile_records.append({
                    "tile_id": tile_id,
                    "x0": x,
                    "y0": y,
                    "x1": x + 1000,
                    "y1": y + 1000,
                    "core_share": core_share,
                    "geometry": tile_box
                })

    tiles_gdf = gpd.GeoDataFrame(tile_records, crs=gdf.crs)
    tiles_gdf = tiles_gdf.sort_values(by=["tile_id"]).reset_index(drop=True)
    print(f"Total in-scope tiles: {len(tiles_gdf)}")
    print(f"Total tile footprint area: {len(tiles_gdf) * 1.0:.2f} km²")
    print(f"Mean core share: {tiles_gdf['core_share'].mean()*100:.1f}%")

    # Save to GPKG
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tiles_gdf.to_file(out_path, layer="tiles_core", driver="GPKG")
    print(f"Saved tile index to {out_path} ({os.path.getsize(out_path):,} bytes)")
    print(f"Finished in {time.time()-t0:.2f} s.")

if __name__ == "__main__":
    main()
