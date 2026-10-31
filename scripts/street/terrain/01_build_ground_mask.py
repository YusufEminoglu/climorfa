"""
WP1.1 Ground Mask Generation for Izmir FUA 5m DSM
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
1. Exclude (a) building footprints from data/02_interim/akilli_sehir_fua.gpkg/buildings_fua buffered +5 m
2. Exclude (b) vegetation: ETH canopy (> 2 m) resampled nearest to 5 m, buffered 1 pixel (3x3 dilation)
3. Exclude (c) nodata (where DSM == -32767)
4. Keep roads (roads_fua) outside raw building footprints as preferred ground anchors.
5. Save output to data/street/terrain/ground_mask_5m.tif
"""

import os
import sys
import time
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
import pyogrio

def main():
    t_start = time.time()
    print("=== Step 1: Building Ground Mask for Izmir FUA (5m) ===")
    
    # Paths
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    gpkg_path = "data/02_interim/akilli_sehir_fua.gpkg"
    out_mask_path = "data/street/terrain/ground_mask_5m.tif"
    
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    
    # 1. Read DSM metadata and nodata
    print(f"Reading DSM metadata from {dsm_path}...")
    with rasterio.open(dsm_path) as src_dsm:
        profile = src_dsm.profile.copy()
        shape = src_dsm.shape
        transform = src_dsm.transform
        crs = src_dsm.crs
        nodata_val = src_dsm.nodata
        dsm = src_dsm.read(1)
        
    print(f"DSM dimensions: {shape[1]}x{shape[0]}, CRS: {crs}, Nodata: {nodata_val}")
    valid_dsm = (dsm != nodata_val)
    n_valid = int(np.sum(valid_dsm))
    n_total = dsm.size
    print(f"Total cells: {n_total:,}, Valid DSM cells: {n_valid:,} ({n_valid/n_total*100:.2f}%)")
    
    # 2. Vector Building Footprints (Raw and Buffered +5m)
    print(f"Reading buildings from {gpkg_path} (layer: buildings_fua)...")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(gpkg_path, layer="buildings_fua", columns=["geom"])
    print(f"Loaded {len(bld_df):,} buildings in {time.time()-t0:.2f} s")
    
    # Rasterize raw building footprints
    print("Rasterizing raw building footprints...")
    t0 = time.time()
    raw_bld_mask = features.rasterize(
        ((g, 1) for g in bld_df.geometry if g is not None and not g.is_empty),
        out_shape=shape,
        transform=transform,
        dtype=np.uint8,
        fill=0
    )
    print(f"Raw building rasterization complete in {time.time()-t0:.2f} s ({np.sum(raw_bld_mask==1):,} pixels)")
    
    # Buffer buildings by +5m and rasterize
    print("Buffering buildings by +5.0 m...")
    t0 = time.time()
    buffered_bld_geoms = bld_df.geometry.buffer(5.0)
    print(f"Vector buffer complete in {time.time()-t0:.2f} s")
    
    print("Rasterizing buffered (+5m) building footprints...")
    t0 = time.time()
    buff_bld_mask = features.rasterize(
        ((g, 1) for g in buffered_bld_geoms if g is not None and not g.is_empty),
        out_shape=shape,
        transform=transform,
        dtype=np.uint8,
        fill=0
    )
    del buffered_bld_geoms
    del bld_df
    print(f"Buffered building rasterization complete in {time.time()-t0:.2f} s ({np.sum(buff_bld_mask==1):,} pixels)")
    
    # 3. Roads Layer (roads_fua)
    print(f"Reading roads from {gpkg_path} (layer: roads_fua)...")
    t0 = time.time()
    roads_df = pyogrio.read_dataframe(gpkg_path, layer="roads_fua", columns=["geom"])
    print(f"Loaded {len(roads_df):,} road segments in {time.time()-t0:.2f} s")
    
    print("Rasterizing road network...")
    t0 = time.time()
    road_mask = features.rasterize(
        ((g, 1) for g in roads_df.geometry if g is not None and not g.is_empty),
        out_shape=shape,
        transform=transform,
        dtype=np.uint8,
        fill=0
    )
    del roads_df
    print(f"Road rasterization complete in {time.time()-t0:.2f} s ({np.sum(road_mask==1):,} pixels)")
    
    # 4. ETH Canopy Height Model (> 2m, resampled nearest to 5m, buffered 1 pixel)
    print(f"Resampling ETH canopy from {eth_path} to 5m grid...")
    t0 = time.time()
    eth_5m = np.full(shape, -9999.0, dtype=np.float32)
    with rasterio.open(eth_path) as src_eth:
        reproject(
            source=rasterio.band(src_eth, 1),
            destination=eth_5m,
            src_transform=src_eth.transform,
            src_crs=src_eth.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            src_nodata=src_eth.nodata,
            dst_nodata=-9999.0
        )
    print(f"ETH canopy resampled in {time.time()-t0:.2f} s")
    
    veg_raw = (eth_5m > 2.0)
    del eth_5m
    print(f"Vegetation (>2m) raw pixels: {np.sum(veg_raw):,}")
    
    print("Buffering vegetation mask by 1 pixel (3x3 dilation)...")
    t0 = time.time()
    veg_buff = binary_dilation(veg_raw, structure=np.ones((3, 3), dtype=bool))
    del veg_raw
    print(f"Buffered vegetation pixels: {np.sum(veg_buff):,} in {time.time()-t0:.2f} s")
    
    # 5. Combine Ground Mask Logic
    print("Synthesizing final ground mask...")
    # Exclude: (a) building buff 5m, (b) veg buff 1px, (c) nodata
    excluded_objects = (buff_bld_mask == 1) | veg_buff
    del buff_bld_mask
    del veg_buff
    
    ground = valid_dsm & (~excluded_objects)
    del excluded_objects
    
    # Keep roads as preferred ground anchors outside raw building footprints
    road_anchors = (road_mask == 1) & (raw_bld_mask == 0) & valid_dsm
    del road_mask
    del raw_bld_mask
    del valid_dsm
    del dsm
    
    # Preferred ground anchors added back to ground
    final_ground_mask = (ground | road_anchors).astype(np.uint8)
    del ground
    
    n_ground = int(np.sum(final_ground_mask == 1))
    n_anchors = int(np.sum(road_anchors))
    print(f"Final ground anchor cells: {n_ground:,} ({n_ground/n_valid*100:.2f}% of valid land cells)")
    print(f"Road anchor cells added back: {n_anchors:,}")
    
    # 6. Save to GeoTIFF
    print(f"Writing ground mask to {out_mask_path}...")
    profile.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "count": 1,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })
    
    with rasterio.open(out_mask_path, "w", **profile) as dst:
        dst.write(final_ground_mask, 1)
        
    print(f"Ground mask successfully created in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
