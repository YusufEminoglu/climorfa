"""
WP1.3 — Step 2 & 3: Generate 2 m Tree DSM Tiles for CLIMORFA Urban Core
Author: Emin Eminoglu / Antigravity (WSL)
Date: 2026-10-29

Specifications:
- Source: Meta / WRI High-Resolution Canopy Height Maps (Tolan et al. 2024, RSE; CC BY 4.0)
- Grid: Authoritative street-scale tiling grid (grid_spec.md)
  - Tile index: data/street/wsl/tiles_core_wsl.gpkg (537 tiles)
  - Tile dimensions: 1000 m inner + 100 m buffer = 1200 m x 1200 m (600 x 600 px at 2 m)
  - CRS: EPSG:5253 (TUREF / TM27), float32, nodata=-9999.0
  - Format: GeoTIFF, tiled 256x256, DEFLATE + predictor 3
- Processing rules:
  - Reproject to EPSG:5253, resample to 2 m using Resampling.max
    (Justification: SOLWEIG CDSM represents canopy top obstruction height; max pooling preserves
    canopy crown summits and prevents thinning/underestimation of narrow street trees).
  - Mask buildings (data/02_interim/akilli_sehir_fua.gpkg layer buildings_fua buffered +1.0 m) to 0.0.
  - Set heights < 2.0 m to 0.0.
  - Set sea cells (DSM == -32767) to 0.0.
  - Strictly tile-by-tile, single worker, float32, RAM check threshold = 3.0 GB free.
- Outputs:
  - data/street/tiles/2m/tree_dsm/<tile_id>.tif
  - data/street/tiles/2m/tree_dsm/tree_dsm_2m.vrt
"""

import os
import sys
import time
import subprocess
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio import features
import geopandas as gpd
import pyogrio
import shapely
from shapely.strtree import STRtree

os.environ['PROJ_DATA'] = '/home/ye/miniforge3/envs/climorfa-wsl/share/proj'

def check_ram(min_free_gb=3.0):
    with open('/proc/meminfo') as f:
        meminfo = {line.split(':')[0]: int(line.split(':')[1].split()[0]) for line in f}
    avail_gb = meminfo.get('MemAvailable', meminfo.get('MemFree', 0)) / (1024 * 1024)
    if avail_gb < min_free_gb:
        raise RuntimeError(f"Available RAM ({avail_gb:.2f} GB) is below minimum threshold ({min_free_gb} GB). Aborting.")
    return avail_gb

def main():
    t_start = time.time()
    print("=== Step 2 & 3: Generate 2 m Tree DSM Tiles ===")
    avail_ram = check_ram()
    print(f"Initial available RAM: {avail_ram:.2f} GB (Safety threshold: 3.0 GB)")

    # Paths
    tiles_gpkg = "data/street/wsl/tiles_core_wsl.gpkg"
    meta_vrt_path = "data/street/wsl/meta_chm/meta_chm_raw.vrt"
    bld_gpkg = "data/02_interim/akilli_sehir_fua.gpkg"
    dsm_5m_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    out_dir = "data/street/tiles/2m/tree_dsm"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load tile index
    print(f"Loading tile index from {tiles_gpkg}...")
    tiles_gdf = gpd.read_file(tiles_gpkg, layer="tiles_core")
    n_tiles = len(tiles_gdf)
    print(f"Loaded {n_tiles} core tiles to process.")

    # 2. Load and index building footprints
    print(f"Loading building footprints from {bld_gpkg}...")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(bld_gpkg, layer="buildings_fua", columns=["geometry"])
    print(f"  Loaded {len(bld_df):,} buildings in {time.time()-t0:.2f} s. Cleaning geometries...")
    # Clean invalid geometries
    valid_geoms = shapely.make_valid(bld_df.geometry.values)
    print("  Building STRtree spatial index...")
    bld_tree = STRtree(valid_geoms)
    print(f"  STRtree built in {time.time()-t0:.2f} s.")

    # 3. Open source rasters
    print(f"Opening Meta CHM raw VRT: {meta_vrt_path}...")
    raw_vrt_src = rasterio.open(meta_vrt_path)
    # WarpedVRT reprojects to EPSG:5253 using Resampling.max
    warped_vrt = WarpedVRT(raw_vrt_src, crs="EPSG:5253", resampling=Resampling.max)
    
    print(f"Opening 5 m DSM reference: {dsm_5m_path}...")
    dsm_src = rasterio.open(dsm_5m_path)

    # GeoTIFF creation profile
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": 600,
        "height": 600,
        "count": 1,
        "crs": "EPSG:5253",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3
    }

    # Statistics accumulators
    total_tree_cells = 0
    total_valid_cells = 0
    max_tree_height_all = 0.0
    sum_tree_height = 0.0

    print(f"\nProcessing {n_tiles} tiles tile-by-tile (float32, 1 worker)...")
    t_loop = time.time()
    for idx, row in tiles_gdf.iterrows():
        if idx % 50 == 0:
            avail = check_ram()
            elapsed = time.time() - t_loop
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Progress: {idx}/{n_tiles} tiles ({idx/n_tiles*100:.1f}%) | "
                  f"Rate: {rate:.1f} tiles/s | Avail RAM: {avail:.2f} GB")

        tile_id = row["tile_id"]
        x0, y0 = int(row["x0"]), int(row["y0"])
        xmin, ymin = x0 - 100, y0 - 100
        xmax, ymax = x0 + 1100, y0 + 1100
        out_tif = os.path.join(out_dir, f"{tile_id}.tif")

        # Tile affine transform (2m resolution, 600x600 pixels)
        transform = Affine(2.0, 0.0, float(xmin), 0.0, -2.0, float(ymax))
        profile["transform"] = transform

        # (a) Read warped CHM (Resampling.max from ~1.19m to 2.0m)
        win_vrt = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, warped_vrt.transform)
        chm_tile = warped_vrt.read(1, window=win_vrt, out_shape=(600, 600)).astype(np.float32)

        # (b) Building footprint masking (buffered +1.0 m)
        tile_box = shapely.box(xmin, ymin, xmax, ymax)
        bld_hits = bld_tree.query(tile_box, predicate="intersects")
        if len(bld_hits) > 0:
            hit_geoms = valid_geoms[bld_hits]
            # Buffer by 1.0 m
            buffered_geoms = [shapely.buffer(g, 1.0) for g in hit_geoms if not g.is_empty]
            if len(buffered_geoms) > 0:
                bld_mask = features.rasterize(
                    buffered_geoms,
                    out_shape=(600, 600),
                    transform=transform,
                    default_value=1,
                    all_touched=False,
                    dtype=np.uint8
                )
                chm_tile[bld_mask == 1] = 0.0

        # (c) Cutoff heights < 2.0 m to 0.0
        chm_tile[chm_tile < 2.0] = 0.0
        chm_tile[np.isnan(chm_tile)] = 0.0

        # (d) Sea masking: DSM == -32767
        win_dsm = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, dsm_src.transform)
        dsm_tile = dsm_src.read(1, window=win_dsm, out_shape=(600, 600), resampling=Resampling.nearest)
        sea_mask = (dsm_tile == -32767)
        chm_tile[sea_mask] = 0.0

        # Accumulate metrics
        tree_pixels = chm_tile >= 2.0
        n_tree = int(np.count_nonzero(tree_pixels))
        total_tree_cells += n_tree
        total_valid_cells += int(np.count_nonzero(~sea_mask))
        if n_tree > 0:
            tile_max = float(chm_tile.max())
            if tile_max > max_tree_height_all:
                max_tree_height_all = tile_max
            sum_tree_height += float(chm_tile[tree_pixels].sum())

        # Write GeoTIFF
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(chm_tile, 1)

    print(f"\nAll {n_tiles} tiles written successfully in {time.time()-t_loop:.1f} s.")
    mean_tree_h = sum_tree_height / total_tree_cells if total_tree_cells > 0 else 0.0
    tree_cover_pct = (total_tree_cells / total_valid_cells * 100.0) if total_valid_cells > 0 else 0.0
    print(f"Summary across all urban-core tiles:")
    print(f"  Valid land pixels (2 m): {total_valid_cells:,} ({total_valid_cells * 4 / 1e6:.2f} km²)")
    print(f"  Canopy pixels (h >= 2 m): {total_tree_cells:,} ({total_tree_cells * 4 / 1e6:.2f} km²)")
    print(f"  Overall Urban-Core Tree Cover: {tree_cover_pct:.2f}%")
    print(f"  Mean Canopy Height: {mean_tree_h:.2f} m")
    print(f"  Maximum Canopy Height: {max_tree_height_all:.2f} m")

    # Close sources
    warped_vrt.close()
    raw_vrt_src.close()
    dsm_src.close()

    # 4. Build output VRT
    vrt_out = os.path.join(out_dir, "tree_dsm_2m.vrt")
    vrt_sym = os.path.join(out_dir, "tree_dsm.vrt")
    print(f"\nBuilding VRT: {vrt_out}...")
    cmd = f"gdalbuildvrt -overwrite {vrt_out} {out_dir}/T_*.tif"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error building VRT: {res.stderr}")
    else:
        print(f"Built {vrt_out} ({os.path.getsize(vrt_out):,} bytes).")

    # Also symlink / copy to tree_dsm.vrt
    if os.path.exists(vrt_out):
        if os.path.exists(vrt_sym):
            os.remove(vrt_sym)
        os.symlink(os.path.basename(vrt_out), vrt_sym)
        print(f"Linked {vrt_sym} -> {os.path.basename(vrt_out)}")

    print(f"Step 2 & 3 completed in {time.time()-t_start:.1f} s.")

if __name__ == "__main__":
    main()
