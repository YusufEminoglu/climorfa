"""
Test Regional Bare-Earth Trend T Generation
Purely derived from bare open DSM cells (no FABDEM)
Checks memory, speed, and accuracy across FUA
"""

import os
import sys
import time
import psutil
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio import features
from scipy.ndimage import gaussian_filter, zoom
from osgeo import gdal
import pyogrio

gdal.UseExceptions()

def check_ram(tag=""):
    vm = psutil.virtual_memory()
    free_gb = vm.available / 1e9
    print(f"[RAM Check {tag}] Available: {free_gb:.2f} GB ({vm.percent}% used)")
    if free_gb < 3.0:
        raise MemoryError(f"CRITICAL: Free RAM {free_gb:.2f} GB is below 3.0 GB threshold! Halting.")
    return free_gb

def main():
    print("=== Testing Regional Trend T Generation ===")
    check_ram("Start")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    bld_path = "data/02_interim/akilli_sehir_fua.gpkg"
    
    t0 = time.time()
    with rasterio.open(dsm_path) as s_dsm:
        shape = s_dsm.shape
        trans = s_dsm.transform
        crs = s_dsm.crs
        nodata = float(s_dsm.nodata)
        dsm = s_dsm.read(1).astype(np.float32)
        
    print(f"Loaded DSM {shape} in {time.time()-t0:.2f} s")
    valid_land = (dsm != nodata) & (dsm >= -1.0)
    check_ram("DSM Loaded")
    
    # 2. Building mask
    print("Rasterizing building mask (+2.5 m buffer)...")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(bld_path, layer="buildings_fua", columns=["geom"])
    buffered = bld_df.geometry.buffer(2.5)
    del bld_df
    bld_mask = features.rasterize(
        ((g, 1) for g in buffered if g is not None and not g.is_empty),
        out_shape=shape,
        transform=trans,
        dtype=np.uint8,
        fill=0
    )
    del buffered
    print(f"Building mask ready in {time.time()-t0:.2f} s ({np.sum(bld_mask==1):,} cells)")
    check_ram("Building Mask Ready")
    
    # 3. ETH Canopy
    print("Loading ETH Canopy...")
    t0 = time.time()
    with rasterio.open(eth_path) as s_eth:
        c_10m = s_eth.read(1)
        c_nodata = float(s_eth.nodata)
    c_valid = (c_10m != c_nodata) & (c_10m > 0)
    c_f32 = np.where(c_valid, c_10m, 0.0).astype(np.float32)
    del c_10m
    c_rep = np.repeat(np.repeat(c_f32, 2, axis=0), 2, axis=1)
    del c_f32
    canopy_5m = np.zeros(shape, dtype=np.float32)
    mh, mw = min(shape[0], c_rep.shape[0]), min(shape[1], c_rep.shape[1])
    canopy_5m[:mh, :mw] = c_rep[:mh, :mw]
    if shape[0] > mh:
        canopy_5m[mh:, :] = canopy_5m[mh-1:mh, :]
    if shape[1] > mw:
        canopy_5m[:, mw:] = canopy_5m[:, mw-1:mw]
    del c_rep
    print(f"Canopy ready in {time.time()-t0:.2f} s")
    check_ram("Canopy Ready")
    
    # Bare cells mask: non-building, canopy < 1.5 m, valid land
    bare_cells = (bld_mask == 0) & (canopy_5m < 1.5) & valid_land
    n_bare = int(np.sum(bare_cells))
    n_land = int(np.sum(valid_land))
    print(f"Initial bare cells: {n_bare:,} / {n_land:,} ({n_bare/n_land*100:.2f}% of land)")
    
    # Free building mask and canopy array before coarse block aggregation
    del bld_mask
    del canopy_5m
    check_ram("Freed Masks")
    
    # 4. Block aggregation of bare cells at 100m (step=20)
    step = 20
    h_c = (shape[0] + step - 1) // step
    w_c = (shape[1] + step - 1) // step
    pad_y = (step - shape[0] % step) % step
    pad_x = (step - shape[1] % step) % step
    
    print(f"Aggregating to {step*5} m coarse grid ({h_c} x {w_c})...")
    t0 = time.time()
    
    # Create padded bare DSM array
    dsm_bare = np.where(bare_cells, dsm, np.nan)
    del bare_cells
    del dsm
    d_pad = np.pad(dsm_bare, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    del dsm_bare
    
    blks = d_pad.reshape(h_c, step, w_c, step).transpose(0, 2, 1, 3).reshape(h_c, w_c, step*step)
    del d_pad
    
    # Use 15th percentile of bare cells in each block (robust against any edge vegetation)
    # Using np.nanpercentile along axis -1
    # To optimize speed, we can compute nanmin or nanpercentile
    trend_coarse = np.nanpercentile(blks, 15, axis=-1)
    del blks
    print(f"Coarse trend aggregated in {time.time()-t0:.2f} s. Valid blocks: {np.sum(~np.isnan(trend_coarse)):,} / {h_c*w_c:,}")
    
    # 5. Fill nodata gaps at coarse level
    t0 = time.time()
    val_c = ~np.isnan(trend_coarse)
    drv = gdal.GetDriverByName("MEM")
    ds_t = drv.Create("", w_c, h_c, 1, gdal.GDT_Float32)
    ds_t.GetRasterBand(1).WriteArray(np.nan_to_num(trend_coarse, nan=0.0))
    ds_m = drv.Create("", w_c, h_c, 1, gdal.GDT_Byte)
    ds_m.GetRasterBand(1).WriteArray(val_c.astype(np.uint8))
    del val_c
    del trend_coarse
    
    gdal.FillNodata(ds_t.GetRasterBand(1), ds_m.GetRasterBand(1), maxSearchDist=60, smoothingIterations=1)
    trend_filled = ds_t.GetRasterBand(1).ReadAsArray()
    del ds_t
    del ds_m
    
    # Smooth with Gaussian filter sigma=1.5 coarse cells = 150 m
    trend_smooth = gaussian_filter(trend_filled, sigma=1.5)
    del trend_filled
    print(f"Coarse trend filled and smoothed in {time.time()-t0:.2f} s")
    
    # 6. Check elevation at evaluation windows
    print("\n--- Checking Trend Elevation at Key Windows ---")
    windows = {
        "Alsancak (Dense flat plain)": (512350, 4255850, 512850, 4256350),
        "Cigli Plain (Coastal flat)": (502000, 4263000, 504000, 4265000),
        "Karabaglar (Hillside)": (508000, 4247000, 509000, 4248000),
        "Buca (Hillside)": (511000, 4249000, 512000, 4250000),
        "Bayrakli hill": (514000, 4260000, 515000, 4261000),
    }
    
    for wname, bbox in windows.items():
        # Map bbox to coarse grid coordinates
        # origin = (trans[2], trans[5]), pixel size = (trans[0]*step, trans[4]*step)
        col0 = int((bbox[0] - trans[2]) / (trans[0] * step))
        col1 = int((bbox[2] - trans[2]) / (trans[0] * step))
        row0 = int((bbox[3] - trans[5]) / (trans[4] * step))
        row1 = int((bbox[1] - trans[5]) / (trans[4] * step))
        r_min, r_max = min(row0, row1), max(row0, row1)
        c_min, c_max = min(col0, col1), max(col0, col1)
        
        sub = trend_smooth[r_min:r_max+1, c_min:c_max+1]
        print(f"  {wname:30s}: Trend elev range = [{np.min(sub):.2f} .. {np.max(sub):.2f}] m, Mean = {np.mean(sub):.2f} m")
        
    check_ram("Finish")
    print("Trend test completed successfully!")

if __name__ == "__main__":
    main()
