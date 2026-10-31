"""
WP3.0c Master Production Script: Scale-Ladder Grids and Context Layers
Author: Antigravity (Laptop / climorfa-cpu)
Date: 2026-10-30

Specifications:
- Generates data/street/analysis/grid_{050,100,250,500}.gpkg
- Target area: inner extents of the 427 core tiles (data/street/grid/tiles_core.gpkg)
- CRS: EPSG:5253 (TUREF / TM27)
- Memory rules: tile by tile, float32, single worker, halt if free RAM < 3 GB
- Schema-B fields:
    g<L>_id, tile_id, lcz_primary_major, lcz_primary_share, lcz_weak250,
    lst_med, pop, reach, district, geometry
"""

import os
import sys
import time
import argparse
import psutil
import pyproj

# Ensure PROJ data directory is set correctly for pyproj and rasterio
proj_dir = pyproj.datadir.get_data_dir()
os.environ['PROJ_DATA'] = proj_dir
os.environ['PROJ_LIB'] = proj_dir

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import rasterio
from rasterio import features, Affine
from rasterio.warp import reproject, Resampling
import pyogrio


def check_ram(tag=""):
    vm = psutil.virtual_memory()
    free_gb = vm.available / 1e9
    if free_gb < 3.0:
        raise MemoryError(
            f"CRITICAL: Free RAM {free_gb:.2f} GB is below 3.0 GB threshold! Halting."
        )
    return free_gb


def main():
    parser = argparse.ArgumentParser(description="WP3.0c Grid Context Builder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tiles for testing")
    parser.add_argument("--out-suffix", type=str, default="", help="Suffix for output GPKG filenames")
    args = parser.parse_args()

    t_start = time.time()
    print("=== WP3.0c: Scale-Ladder Grids and Context Layers ===")
    if args.limit:
        print(f"*** RUNNING IN TEST MODE (LIMIT: {args.limit} TILES) ***")
    initial_ram = check_ram("Start")
    print(f"Initial available RAM: {initial_ram:.2f} GB")

    # Paths
    tiles_core_path = "data/street/grid/tiles_core.gpkg"
    akilli_sehir_path = "data/02_interim/akilli_sehir_fua.gpkg"
    lcz_raster_path = "data/01_raw/gee/lcz/izmir_fua_rub_lcz_latest.tif"
    lst_raster_path = "data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif"
    pop_gpkg_path = "data/01_raw/population/mahalle_population_controls_2024.gpkg"
    reach_csv_path = "data/03_processed/grid_250m_road_network_reach_features.csv"
    weak_lcz_path = "data/03_processed/grid_lcz_weak_labels.gpkg"
    analysis_grids_path = "data/03_processed/analysis_grids.gpkg"

    out_dir = "data/street/analysis"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load core tiles
    print("\n--- Step 1: Loading Core Tiles ---")
    gdf_tiles = pyogrio.read_dataframe(tiles_core_path, layer="tiles_core")
    print(f"Loaded {len(gdf_tiles)} core tiles from {tiles_core_path}.")
    if args.limit:
        gdf_tiles = gdf_tiles.iloc[:args.limit].copy()
        print(f"Limiting to first {len(gdf_tiles)} tiles for testing.")
    else:
        assert len(gdf_tiles) == 427, f"Expected 427 core tiles, got {len(gdf_tiles)}"

    # 2. Load Reach Features and map to new g250_id by centroid
    print("\n--- Step 2: Preparing Reach Centrality (network_reach_400m_road_length_m) ---")
    df_reach = pd.read_csv(
        reach_csv_path,
        usecols=["grid_id", "network_reach_400m_road_length_m"]
    )
    gdf_ag250 = gpd.read_file(analysis_grids_path, layer="grid_250m")
    merged_reach = gdf_ag250.merge(df_reach, on="grid_id", how="left")
    
    # Calculate centroid and new g250_id formula: f"{x0//250}_{y0//250}"
    c_x = merged_reach.geometry.centroid.x.values
    c_y = merged_reach.geometry.centroid.y.values
    new_g250_ids = [f"{int(x)//250}_{int(y)//250}" for x, y in zip(c_x, c_y)]
    merged_reach["new_g250_id"] = new_g250_ids
    
    reach_dict = dict(zip(merged_reach["new_g250_id"], merged_reach["network_reach_400m_road_length_m"]))
    print(f"Mapped {len(reach_dict)} reach values to new g250_ids.")

    # 3. Load Weak LCZ Labels and Districts
    print("\n--- Step 3: Preparing Weak LCZ Labels and District Mapping ---")
    gdf_weak = gpd.read_file(weak_lcz_path, layer="grid_250m_lcz")
    c_x_w = gdf_weak.geometry.centroid.x.values
    c_y_w = gdf_weak.geometry.centroid.y.values
    gdf_weak["new_g250_id"] = [f"{int(x)//250}_{int(y)//250}" for x, y in zip(c_x_w, c_y_w)]
    
    weak_lcz_dict = dict(zip(gdf_weak["new_g250_id"], gdf_weak["lcz_weak_label"]))
    district_dict = dict(zip(gdf_weak["new_g250_id"], gdf_weak["district"]))
    print(f"Mapped {len(weak_lcz_dict)} weak LCZ labels and {len(district_dict)} districts.")

    # 4. Load Mahalle Population Controls & Building Footprints for Areal Weighting
    print("\n--- Step 4: Preparing Population Dasymetric Weights ---")
    gdf_pop = gpd.read_file(pop_gpkg_path).to_crs("EPSG:5253")
    gdf_pop["pop_total"] = gdf_pop["pop_total"].fillna(0.0)
    mahalle_sindex = gdf_pop.sindex

    # Total building footprint area per mahalle across the FUA from buildings_fua
    print("Calculating building footprint areas across FUA from buildings_fua...")
    bld_fua = pyogrio.read_dataframe(
        akilli_sehir_path, layer="buildings_fua", columns=["SHAPE_Area"]
    )
    bld_fua["centroid"] = bld_fua.geometry.centroid
    bld_fua = bld_fua.set_geometry("centroid")
    bld_fua_joined = gpd.sjoin(
        bld_fua[["SHAPE_Area", "centroid"]],
        gdf_pop[["join_key", "geometry"]],
        predicate="within"
    )
    bld_area_total = bld_fua_joined.groupby("join_key")["SHAPE_Area"].sum().to_dict()

    # Core buildings footprint area
    tiles_union = gdf_tiles.union_all()
    bld_core = bld_fua[bld_fua.geometry.within(tiles_union)]
    bld_core_joined = gpd.sjoin(
        bld_core[["SHAPE_Area", "centroid"]],
        gdf_pop[["join_key", "geometry"]],
        predicate="within"
    )
    bld_area_core = bld_core_joined.groupby("join_key")["SHAPE_Area"].sum().to_dict()
    del bld_fua, bld_core, bld_fua_joined, bld_core_joined

    # 5. Load Road Network for Pedestrian Space (PS) Mask
    print("\n--- Step 5: Indexing Road Network (roads_fua) ---")
    roads_df = pyogrio.read_dataframe(
        akilli_sehir_path, layer="roads_fua", columns=["geom"]
    )
    roads_tree = shapely.STRtree(roads_df.geometry.values)
    print(f"Loaded {len(roads_df)} road segments and built spatial index.")

    # 6. Pass 1: Count Building Cells per Mahalle in Core Tiles
    print("\n--- Step 6: Pass 1 - Counting Building Pixels per Mahalle in Core Tiles ---")
    t_p1 = time.time()
    n_mahalles = len(gdf_pop)
    mahalle_bld_px = np.zeros(n_mahalles + 1, dtype=np.int64)

    for idx, row in gdf_tiles.iterrows():
        if idx % 50 == 0:
            check_ram(f"Pass1 Tile {idx}/{len(gdf_tiles)}")
            print(f"Pass 1: Processed {idx}/{len(gdf_tiles)} tiles ({time.time()-t_p1:.1f}s)...")
        
        tid = row["tile_id"]
        x0, y0 = int(row["x0"]), int(row["y0"])

        with rasterio.open(f"data/street/tiles/2m/dem/{tid}.tif") as s_dem, \
             rasterio.open(f"data/street/tiles/2m/building_dsm/{tid}.tif") as s_bld, \
             rasterio.open(f"data/street/tiles/2m/landmask/{tid}.tif") as s_lm:
            dem = s_dem.read(1)[50:550, 50:550].astype(np.float32)
            bld = s_bld.read(1)[50:550, 50:550].astype(np.float32)
            lm = s_lm.read(1)[50:550, 50:550].astype(np.float32)

        bld_mask = (lm == 1.0) & ((bld - dem) > 0.5)
        if not np.any(bld_mask):
            continue

        tile_box = box(x0, y0, x0 + 1000, y0 + 1000)
        cand_indices = list(mahalle_sindex.intersection(tile_box.bounds))
        cand_mahalles = gdf_pop.iloc[cand_indices]
        cand_mahalles = cand_mahalles[cand_mahalles.intersects(tile_box)]

        if len(cand_mahalles) > 0:
            tile_transform = Affine(2.0, 0.0, x0, 0.0, -2.0, y0 + 1000)
            shapes = [(geom, i + 1) for i, geom in zip(cand_mahalles.index, cand_mahalles.geometry)]
            tile_mahalle = features.rasterize(
                shapes, out_shape=(500, 500), transform=tile_transform, fill=0
            )
            bc = np.bincount(tile_mahalle[bld_mask], minlength=n_mahalles + 1)
            mahalle_bld_px += bc

    print(f"Pass 1 complete in {time.time()-t_p1:.2f}s. Total building pixels: {mahalle_bld_px.sum():,}")

    # Compute per-mahalle dasymetric population weights
    print("\n--- Step 7: Computing Dasymetric Population Weight per Building Cell ---")
    pop_weights = np.zeros(n_mahalles + 1, dtype=np.float32)
    mahalle_district_map = {}

    for idx, row in gdf_pop.iterrows():
        m_idx = idx + 1
        jkey = row["join_key"]
        p_total = float(row["pop_total"])
        mahalle_district_map[m_idx] = row.get("district_norm") or row.get("district_name") or "unknown"

        a_total = bld_area_total.get(jkey, 0.0)
        a_core = bld_area_core.get(jkey, 0.0)
        core_frac = (a_core / a_total) if a_total > 0.0 else 1.0
        p_core = p_total * core_frac

        n_px = mahalle_bld_px[m_idx]
        if n_px > 0:
            pop_weights[m_idx] = float(p_core / n_px)
        else:
            pop_weights[m_idx] = 0.0

    print(f"Calculated population weights for {len(gdf_pop)} mahalles.")

    # 8. Pass 2: Main Production Loop across All 427 Tiles
    print("\n--- Step 8: Pass 2 - Main Processing and Grid Assembly ---")
    t_p2 = time.time()

    src_lcz = rasterio.open(lcz_raster_path)
    src_lst = rasterio.open(lst_raster_path)

    # Output storage lists for each level
    levels = {
        "050": {"L": 50, "s": 25, "records": []},
        "100": {"L": 100, "s": 50, "records": []},
        "250": {"L": 250, "s": 125, "records": []},
        "500": {"L": 500, "s": 250, "records": []},
    }

    for idx, row in gdf_tiles.iterrows():
        check_ram(f"Pass2 Tile {idx}/{len(gdf_tiles)}")
        if idx % 25 == 0:
            elapsed = time.time() - t_p2
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(gdf_tiles) - idx) / rate if rate > 0 else 0
            print(f"Pass 2: Tile {idx}/{len(gdf_tiles)} ({row['tile_id']}) | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

        tid = row["tile_id"]
        x0, y0 = int(row["x0"]), int(row["y0"])
        tile_box = box(x0, y0, x0 + 1000, y0 + 1000)
        tile_transform = Affine(2.0, 0.0, x0, 0.0, -2.0, y0 + 1000)

        # 1. Read DSM rasters
        with rasterio.open(f"data/street/tiles/2m/dem/{tid}.tif") as s_dem, \
             rasterio.open(f"data/street/tiles/2m/building_dsm/{tid}.tif") as s_bld, \
             rasterio.open(f"data/street/tiles/2m/landmask/{tid}.tif") as s_lm:
            dem = s_dem.read(1)[50:550, 50:550].astype(np.float32)
            bld = s_bld.read(1)[50:550, 50:550].astype(np.float32)
            lm = s_lm.read(1)[50:550, 50:550].astype(np.float32)

        # Building mask
        bld_mask = (lm == 1.0) & ((bld - dem) > 0.5)

        # 2. Roads buffer & rasterize
        road_query_box = box(x0 - 20, y0 - 20, x0 + 1020, y0 + 1020)
        cand_road_indices = roads_tree.query(road_query_box, predicate="intersects")
        if len(cand_road_indices) > 0:
            sub_roads = roads_df.iloc[cand_road_indices].geometry.buffer(15.0)
            road_buf_mask = features.rasterize(
                [(g, 1) for g in sub_roads],
                out_shape=(500, 500),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8
            )
        else:
            road_buf_mask = np.zeros((500, 500), dtype=np.uint8)

        # 3. PS Mask (exact pre-analysis lock definition)
        # landmask = 1, building_dsm - dem <= 0.5 m, <= 15 m from roads_fua
        ps_mask = (lm == 1.0) & ((bld - dem) <= 0.5) & (road_buf_mask == 1)

        # 4. LCZ Raster Reprojection into Tile (nearest)
        tile_lcz = np.full((500, 500), -9999, dtype=np.int32)
        reproject(
            source=rasterio.band(src_lcz, 1),
            destination=tile_lcz,
            dst_transform=tile_transform,
            dst_crs="EPSG:5253",
            resampling=Resampling.nearest,
            dst_nodata=-9999
        )

        # 5. LST Raster Resampling into Tile (bilinear)
        tile_lst = np.full((500, 500), -9999.0, dtype=np.float32)
        reproject(
            source=rasterio.band(src_lst, 1),
            destination=tile_lst,
            dst_transform=tile_transform,
            dst_crs=src_lst.crs,
            resampling=Resampling.bilinear,
            dst_nodata=-9999.0
        )

        # 6. Rasterize Mahalles & Compute Pixel Population
        cand_m_indices = list(mahalle_sindex.intersection(tile_box.bounds))
        cand_mahalles = gdf_pop.iloc[cand_m_indices]
        cand_mahalles = cand_mahalles[cand_mahalles.intersects(tile_box)]

        if len(cand_mahalles) > 0:
            m_shapes = [(geom, i + 1) for i, geom in zip(cand_mahalles.index, cand_mahalles.geometry)]
            tile_mahalle = features.rasterize(
                m_shapes, out_shape=(500, 500), transform=tile_transform, fill=0
            )
            # Assign population to building cells
            tile_pop = np.where(bld_mask, pop_weights[tile_mahalle], 0.0).astype(np.float32)
        else:
            tile_mahalle = np.zeros((500, 500), dtype=np.int32)
            tile_pop = np.zeros((500, 500), dtype=np.float32)

        # Determine tile fallback district from dominant intersecting mahalle
        tile_district = "unknown"
        if len(cand_mahalles) > 0:
            m_inter_areas = [geom.intersection(tile_box).area for geom in cand_mahalles.geometry]
            max_m_idx = np.argmax(m_inter_areas)
            tile_district = cand_mahalles.iloc[max_m_idx].get("district_norm") or \
                            cand_mahalles.iloc[max_m_idx].get("district_name") or "unknown"

        # 7. Aggregate for each grid level
        for lvl_name, lvl_cfg in levels.items():
            L = lvl_cfg["L"]
            s = lvl_cfg["s"]
            n_blocks = 1000 // L

            for j in range(n_blocks):
                cx0 = x0 + j * L
                col_slice = slice(j * s, (j + 1) * s)

                for k in range(n_blocks):
                    cy0 = y0 + k * L
                    row_slice = slice(500 - (k + 1) * s, 500 - k * s)

                    # ID
                    cell_id = f"{cx0 // L}_{cy0 // L}"

                    # PS and LCZ
                    sub_ps = ps_mask[row_slice, col_slice]
                    sub_lcz = tile_lcz[row_slice, col_slice]

                    if np.any(sub_ps):
                        ps_lcz_vals = sub_lcz[sub_ps]
                        valid_lcz = ps_lcz_vals[ps_lcz_vals > 0]
                        if len(valid_lcz) > 0:
                            u_vals, u_counts = np.unique(valid_lcz, return_counts=True)
                            max_u = np.argmax(u_counts)
                            lcz_primary_major = int(u_vals[max_u])
                            lcz_primary_share = float(u_counts[max_u] / len(ps_lcz_vals))
                        else:
                            lcz_primary_major = np.nan
                            lcz_primary_share = np.nan
                    else:
                        lcz_primary_major = np.nan
                        lcz_primary_share = np.nan

                    # LST
                    sub_lst = tile_lst[row_slice, col_slice]
                    valid_lst = sub_lst[sub_lst > -9000]
                    if len(valid_lst) > 0:
                        lst_med = float(np.median(valid_lst))
                    else:
                        lst_med = np.nan

                    # Population
                    sub_pop = tile_pop[row_slice, col_slice]
                    pop_val = float(np.sum(sub_pop))

                    # Context mappings (reach, weak LCZ, district)
                    if L in [50, 100, 250]:
                        p250_id = f"{cx0 // 250}_{cy0 // 250}"
                        reach_val = reach_dict.get(p250_id, np.nan)
                        lcz_weak = weak_lcz_dict.get(p250_id, np.nan)
                        dist_val = district_dict.get(p250_id, tile_district)
                    else:  # L == 500
                        cids = [f"{(cx0 + dx)//250}_{(cy0 + dy)//250}" for dx in [0, 250] for dy in [0, 250]]
                        # Reach: area-weighted mean of 250m cells
                        sub_reaches = [reach_dict[c] for c in cids if c in reach_dict and not np.isnan(reach_dict[c])]
                        reach_val = float(np.mean(sub_reaches)) if len(sub_reaches) > 0 else np.nan
                        
                        # Weak LCZ: majority of constituent 250m cells
                        sub_weaks = [weak_lcz_dict[c] for c in cids if c in weak_lcz_dict and not np.isnan(weak_lcz_dict[c])]
                        if len(sub_weaks) > 0:
                            vals_w, counts_w = np.unique(sub_weaks, return_counts=True)
                            lcz_weak = float(vals_w[np.argmax(counts_w)])
                        else:
                            lcz_weak = np.nan

                        # District: majority of constituent 250m cells
                        sub_dists = [district_dict[c] for c in cids if c in district_dict and district_dict[c] not in [None, "unknown"]]
                        if len(sub_dists) > 0:
                            vals_d, counts_d = np.unique(sub_dists, return_counts=True)
                            dist_val = str(vals_d[np.argmax(counts_d)])
                        else:
                            dist_val = tile_district

                    geom = box(cx0, cy0, cx0 + L, cy0 + L)

                    lvl_cfg["records"].append({
                        f"g{lvl_name}_id": cell_id,
                        "tile_id": tid,
                        "lcz_primary_major": lcz_primary_major,
                        "lcz_primary_share": lcz_primary_share,
                        "lcz_weak250": lcz_weak,
                        "lst_med": lst_med,
                        "pop": pop_val,
                        "reach": reach_val,
                        "district": dist_val,
                        "geometry": geom
                    })

    src_lcz.close()
    src_lst.close()
    print(f"\nPass 2 complete in {time.time()-t_p2:.2f}s!")

    # 9. Build and Write GeoPackages
    print("\n--- Step 9: Writing GeoPackage Files ---")
    for lvl_name, lvl_cfg in levels.items():
        t_w = time.time()
        out_gpkg = os.path.join(out_dir, f"grid_{lvl_name}{args.out_suffix}.gpkg")
        print(f"Creating GeoDataFrame for grid_{lvl_name} ({len(lvl_cfg['records'])} rows)...")
        gdf_out = gpd.GeoDataFrame(lvl_cfg["records"], crs="EPSG:5253")
        
        # Enforce column ordering per schema B
        col_id = f"g{lvl_name}_id"
        ordered_cols = [
            col_id, "tile_id", "lcz_primary_major", "lcz_primary_share",
            "lcz_weak250", "lst_med", "pop", "reach", "district", "geometry"
        ]
        gdf_out = gdf_out[ordered_cols]

        print(f"Writing to {out_gpkg} (layer: grid)...")
        pyogrio.write_dataframe(gdf_out, out_gpkg, layer="grid")
        size_mb = os.path.getsize(out_gpkg) / (1024 * 1024)
        print(f"Saved {out_gpkg} ({size_mb:.2f} MB) in {time.time()-t_w:.2f}s.")

    total_time = time.time() - t_start
    print(f"\n=== WP3.0c Generation Complete in {total_time:.2f}s ({total_time/60:.2f} min) ===")


if __name__ == "__main__":
    main()
