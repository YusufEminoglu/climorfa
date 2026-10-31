"""
WP1.1 DTM and nDSM Generation for Izmir FUA (5m)
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
1. Interpolate masked surface to a DTM over full FUA extent.
2. Uses GDAL FillNodata (maxSearchDist=250, smoothingIterations=1) with deep forest canopy compensation.
3. Strict enforcement: DTM <= DSM everywhere (cap DTM = min(DTM, DSM)), preventing pits.
4. Compute nDSM = DSM - DTM (>= 0 everywhere).
5. Outputs:
   - data/street/terrain/dtm_5m_epsg5253.tif
   - data/street/terrain/ndsm_5m_epsg5253.tif
"""

import os
import sys
import time
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from osgeo import gdal

# Suppress GDAL future warnings
gdal.UseExceptions()

def main():
    t_start = time.time()
    print("=== Step 2: Full FUA Bare-Earth DTM & nDSM Interpolation (5m) ===")
    
    # Paths
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    mask_path = "data/street/terrain/ground_mask_5m.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    out_dtm_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    out_ndsm_path = "data/street/terrain/ndsm_5m_epsg5253.tif"
    
    os.makedirs(os.path.dirname(out_dtm_path), exist_ok=True)
    
    # 1. Read DSM and Ground Mask
    print(f"Reading DSM from {dsm_path}...")
    t0 = time.time()
    with rasterio.open(dsm_path) as src_dsm:
        profile = src_dsm.profile.copy()
        shape = src_dsm.shape
        transform = src_dsm.transform
        crs = src_dsm.crs
        nodata_val = float(src_dsm.nodata)
        dsm = src_dsm.read(1)
        
    print(f"Reading ground mask from {mask_path}...")
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1)
        
    print(f"Data loaded in {time.time()-t0:.2f} s. Grid size: {shape[1]}x{shape[0]}")
    valid = (dsm != nodata_val)
    n_valid = int(np.sum(valid))
    print(f"Valid land pixels: {n_valid:,} / {dsm.size:,}")
    
    # 2. Setup in-memory dataset for GDAL FillNodata
    print("Preparing GDAL in-memory datasets for FillNodata interpolation...")
    t0 = time.time()
    arr = np.where(mask == 1, dsm.astype(np.float32), 0.0)
    
    drv = gdal.GetDriverByName("MEM")
    ds_target = drv.Create("", shape[1], shape[0], 1, gdal.GDT_Float32)
    ds_target.GetRasterBand(1).WriteArray(arr)
    del arr
    
    ds_mask = drv.Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    ds_mask.GetRasterBand(1).WriteArray(mask)
    del mask
    
    # 3. Execute GDAL FillNodata (maxSearchDist=250, smoothingIterations=1)
    print("Executing GDAL FillNodata (maxSearchDist=250 pixels / 1250 m, smoothingIterations=1)...")
    t_fill = time.time()
    gdal.FillNodata(
        ds_target.GetRasterBand(1),
        ds_mask.GetRasterBand(1),
        maxSearchDist=250,
        smoothingIterations=1
    )
    dtm_filled = ds_target.GetRasterBand(1).ReadAsArray()
    print(f"GDAL FillNodata completed in {time.time()-t_fill:.2f} s")
    
    # 4. Handle any remaining deep mountain forest gaps (> 1250 m from ground anchors)
    unfilled = valid & (dtm_filled == 0.0) & (dsm != 0.0)
    n_unfilled = int(np.sum(unfilled))
    print(f"Unfilled pixels in deep forest interior: {n_unfilled:,} ({n_unfilled/n_valid*100:.3f}%)")
    
    if n_unfilled > 0:
        print("Compensating deep forest gaps using ETH canopy height model...")
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
        canopy_adj = np.where(eth_5m > 0, eth_5m, 0.0)
        dtm_filled[unfilled] = dsm[unfilled] - canopy_adj[unfilled]
        del eth_5m
        del canopy_adj
        
    # 5. Strict enforcement: DTM <= DSM everywhere (cap DTM = min(DTM, DSM))
    print("Applying strict physical constraint: DTM <= DSM everywhere (capping)...")
    dsm_f32 = dsm.astype(np.float32)
    del dsm
    dtm = np.minimum(dtm_filled, dsm_f32)
    del dtm_filled
    
    # Set nodata outside valid FUA terrain
    dtm[~valid] = nodata_val
    
    # Compute nDSM = DSM - DTM
    print("Computing nDSM (nDSM = DSM - DTM)...")
    ndsm = np.where(valid, dsm_f32 - dtm, nodata_val)
    del dsm_f32
    
    # 6. Verification Metrics
    print("--- Verification Metrics on Full FUA Grid ---")
    valid_dtm = dtm[valid]
    valid_ndsm = ndsm[valid]
    print(f"Valid pixels: {len(valid_dtm):,}")
    print(f"DTM range: [{np.min(valid_dtm):.2f}, {np.max(valid_dtm):.2f}] m, median: {np.median(valid_dtm):.2f} m")
    print(f"nDSM range: [{np.min(valid_ndsm):.2f}, {np.max(valid_ndsm):.2f}] m, median: {np.median(valid_ndsm):.2f} m")
    print(f"Percentiles of nDSM: p50={np.percentile(valid_ndsm, 50):.2f} m, p75={np.percentile(valid_ndsm, 75):.2f} m, p90={np.percentile(valid_ndsm, 90):.2f} m, p95={np.percentile(valid_ndsm, 95):.2f} m, p99={np.percentile(valid_ndsm, 99):.2f} m")
    print(f"nDSM == 0.0 (ground) count: {np.sum(valid_ndsm == 0.0):,} ({np.mean(valid_ndsm == 0.0)*100:.2f}%)")
    print(f"nDSM > 0.0 (objects) count: {np.sum(valid_ndsm > 0.0):,} ({np.mean(valid_ndsm > 0.0)*100:.2f}%)")
    print(f"DTM <= DSM check: {np.all(valid_ndsm >= 0.0)} (100.0% compliant, min nDSM = {np.min(valid_ndsm):.4f} m)")
    del valid_dtm
    del valid_ndsm
    
    # 7. Write DTM and nDSM GeoTIFFs
    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": nodata_val,
        "count": 1,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 3
    })
    
    print(f"Writing DTM to {out_dtm_path}...")
    t0 = time.time()
    with rasterio.open(out_dtm_path, "w", **out_profile) as dst:
        dst.write(dtm.astype(np.float32), 1)
    print(f"DTM saved in {time.time()-t0:.2f} s")
    del dtm
    
    print(f"Writing nDSM to {out_ndsm_path}...")
    t0 = time.time()
    with rasterio.open(out_ndsm_path, "w", **out_profile) as dst:
        dst.write(ndsm.astype(np.float32), 1)
    print(f"nDSM saved in {time.time()-t0:.2f} s")
    del ndsm
    
    print(f"Step 2 complete in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
