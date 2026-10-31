"""
WP1.1d Step 2: Full FUA Bare-Earth DTM v3 and nDSM v3 Production (Definitive Formulation)
- Median coarse regional trend (100 m) eliminating downhill bias
- Sub-pixel exact full-FUA trend upsampling (zero tile edge seams)
- Slope-adaptive residual thresholding (1.0 m on plains, 2.5 m on hillsides)
- Non-circular bare open ground preservation (DSM kept on bare hillside cells)
- Memory-safe 2000x2000 tiled processing with 300 m overlap
Author: Yusuf Eminoglu / Antigravity
"""

import os
import sys
import time
import psutil
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    sobel,
    percentile_filter,
    distance_transform_edt,
    zoom
)
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

def build_coarse_trend(dsm, bare_mask, step=20, sigma=1.5):
    """
    Build regional ground trend T purely from bare DSM cells (100m resolution, median).
    """
    shape = dsm.shape
    h_c = (shape[0] + step - 1) // step
    w_c = (shape[1] + step - 1) // step
    pad_y = (step - shape[0] % step) % step
    pad_x = (step - shape[1] % step) % step
    
    dsm_bare = np.where(bare_mask, dsm, np.nan)
    d_pad = np.pad(dsm_bare, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    del dsm_bare
    
    blks = d_pad.reshape(h_c, step, w_c, step).transpose(0, 2, 1, 3).reshape(h_c, w_c, step * step)
    del d_pad
    
    trend_coarse = np.nanmedian(blks, axis=-1)
    del blks
    
    val_c = ~np.isnan(trend_coarse)
    drv = gdal.GetDriverByName("MEM")
    ds_t = drv.Create("", w_c, h_c, 1, gdal.GDT_Float32)
    ds_t.GetRasterBand(1).WriteArray(np.nan_to_num(trend_coarse, nan=0.0))
    ds_m = drv.Create("", w_c, h_c, 1, gdal.GDT_Byte)
    ds_m.GetRasterBand(1).WriteArray(val_c.astype(np.uint8))
    del val_c, trend_coarse
    
    gdal.FillNodata(ds_t.GetRasterBand(1), ds_m.GetRasterBand(1), maxSearchDist=60, smoothingIterations=1)
    trend_filled = ds_t.GetRasterBand(1).ReadAsArray()
    del ds_t, ds_m
    
    trend_smooth = gaussian_filter(trend_filled, sigma=sigma)
    del trend_filled
    
    scale_y = shape[0] / h_c
    scale_x = shape[1] / w_c
    T_full = zoom(trend_smooth, (scale_y, scale_x), order=1)[:shape[0], :shape[1]]
    del trend_smooth
    return T_full

def process_tile(dsm_buf, bld_buf, can_buf, T_buf, bld_dil_buf, valid_buf):
    """
    Processes a buffered tile and returns bare-earth DTM.
    """
    forest_canopy = (can_buf > 3.0) & (dsm_buf > 50.0)
    dsm_adj = np.where(forest_canopy, np.maximum(dsm_buf - can_buf, 0.0), dsm_buf)
    
    R_buf = dsm_adj - T_buf
    nb_mask = (bld_buf == 0) & (can_buf < 1.5) & valid_buf
    
    h, w = dsm_buf.shape
    pad_y = (4 - h % 4) % 4
    pad_x = (4 - w % 4) % 4
    R_pad = np.pad(np.where(nb_mask, R_buf, np.nan), ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    h4, w4 = R_pad.shape[0] // 4, R_pad.shape[1] // 4
    R_blks = R_pad.reshape(h4, 4, w4, 4).transpose(0, 2, 1, 3).reshape(h4, w4, 16)
    del R_pad
    c_min = np.nanmin(R_blks, axis=-1)
    del R_blks
    
    val_c = ~np.isnan(c_min)
    drv = gdal.GetDriverByName("MEM")
    ds_rt = drv.Create("", w4, h4, 1, gdal.GDT_Float32)
    ds_rt.GetRasterBand(1).WriteArray(np.nan_to_num(c_min, nan=0.0))
    ds_rm = drv.Create("", w4, h4, 1, gdal.GDT_Byte)
    ds_rm.GetRasterBand(1).WriteArray(val_c.astype(np.uint8))
    del val_c, c_min
    
    gdal.FillNodata(ds_rt.GetRasterBand(1), ds_rm.GetRasterBand(1), maxSearchDist=60, smoothingIterations=0)
    c_filled = ds_rt.GetRasterBand(1).ReadAsArray()
    del ds_rt, ds_rm
    
    c_p10 = percentile_filter(c_filled, percentile=10, size=4, mode='nearest')
    del c_filled
    
    low_env = zoom(c_p10, (h / h4, w / w4), order=1)[:h, :w]
    del c_p10
    
    dx_t = sobel(T_buf, axis=1) / (8.0 * 5.0)
    dy_t = sobel(T_buf, axis=0) / (8.0 * 5.0)
    trend_slp = np.degrees(np.arctan(np.sqrt(dx_t**2 + dy_t**2)))
    del dx_t, dy_t
    
    delta_thresh = np.where(trend_slp < 2.0, 1.0, 2.5)
    ground_cand = nb_mask & (R_buf <= low_env + delta_thresh)
    del low_env, delta_thresh
    
    ds_it = drv.Create("", w, h, 1, gdal.GDT_Float32)
    ds_it.GetRasterBand(1).WriteArray(np.where(ground_cand, R_buf, 0.0).astype(np.float32))
    
    fill_mask = np.where(valid_buf, ground_cand.astype(np.uint8), 1)
    del ground_cand
    ds_im = drv.Create("", w, h, 1, gdal.GDT_Byte)
    ds_im.GetRasterBand(1).WriteArray(fill_mask)
    del fill_mask
    
    gdal.FillNodata(ds_it.GetRasterBand(1), ds_im.GetRasterBand(1), maxSearchDist=250, smoothingIterations=1)
    R_interp = ds_it.GetRasterBand(1).ReadAsArray()
    del ds_it, ds_im
    
    dtm_recon = T_buf + R_interp
    del R_interp
    
    dtm_smooth = uniform_filter(gaussian_filter(dtm_recon, sigma=1.8), size=3)
    del dtm_recon
    
    # Step 1d & 1e:
    # On flat coastal plain (trend_slp < 2.0 deg): use dtm_smooth inside built-up and plain
    # On hillsides (trend_slp >= 2.0 deg):
    # On bare open cells (outside bld_dilated and can_buf < 1.0 m): keep raw DSM
    # Inside built-up envelope (bld_dilated): use dtm_smooth
    is_hill_bare = (trend_slp >= 2.0) & (~bld_dil_buf) & (can_buf < 1.0) & valid_buf
    del trend_slp
    
    dtm_out = np.where(is_hill_bare, dsm_buf, dtm_smooth)
    del is_hill_bare, dtm_smooth
    
    dtm_out = np.minimum(dtm_out, dsm_buf)
    return dtm_out

def main():
    t_start = time.time()
    print("=== WP1.1d Step 2: Full FUA Bare-Earth DTM v3 Production (Definitive) ===")
    check_ram("Start")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    bld_path = "data/02_interim/akilli_sehir_fua.gpkg"
    out_dtm = "data/street/terrain/dtm_5m_v3_epsg5253.tif"
    out_ndsm = "data/street/terrain/ndsm_5m_v3_epsg5253.tif"
    
    os.makedirs(os.path.dirname(out_dtm), exist_ok=True)
    
    # Read DSM profile
    with rasterio.open(dsm_path) as src_dsm:
        profile = src_dsm.profile.copy()
        shape = src_dsm.shape
        trans = src_dsm.transform
        crs = src_dsm.crs
        nodata = float(src_dsm.nodata)
        dsm = src_dsm.read(1).astype(np.float32)
        
    valid_land = (dsm != nodata) & (dsm >= -1.0)
    print(f"Loaded DSM: {shape} ({np.sum(valid_land):,} land cells)")
    check_ram("DSM Loaded")
    
    # Rasterize building mask
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
    print(f"Building mask ready in {time.time()-t0:.2f} s")
    check_ram("Building Mask Ready")
    
    # Built-up envelope: building dilated by 10 m (2 pixels)
    print("Computing full-FUA built-up envelope (10 m dilation)...")
    t0 = time.time()
    bld_dilated = (gaussian_filter(bld_mask.astype(np.float32), sigma=2.0) > 0.02)
    print(f"Built-up envelope ready in {time.time()-t0:.2f} s")
    
    # Load ETH canopy
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
    
    # Bare cells mask
    bare_mask = (bld_mask == 0) & (canopy_5m < 1.5) & valid_land
    
    # Build 100m regional trend T with exact sub-pixel alignment
    print("Building 100m Regional Bare Trend T (Median, exact full-FUA alignment)...")
    t0 = time.time()
    T_full = build_coarse_trend(dsm, bare_mask, step=20, sigma=1.5)
    del bare_mask
    print(f"Regional trend ready in {time.time()-t0:.2f} s")
    check_ram("Trend Ready")
    
    # Prepare output raster profiles
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    # Tiled Processing: 2000 x 2000 cells with 300 m (60 px) overlap
    tile_size = 2000
    overlap = 60
    
    n_rows, n_cols = shape
    row_starts = list(range(0, n_rows, tile_size))
    col_starts = list(range(0, n_cols, tile_size))
    total_tiles = len(row_starts) * len(col_starts)
    
    print(f"\nProcessing full FUA in {total_tiles} tiles ({tile_size}x{tile_size} with {overlap*5}m overlap)...")
    
    with rasterio.open(out_dtm, "w", **out_profile) as dst_dtm, rasterio.open(out_ndsm, "w", **out_profile) as dst_ndsm:
        tile_idx = 0
        for r_start in row_starts:
            for c_start in col_starts:
                tile_idx += 1
                t_tile = time.time()
                
                # Interior unbuffered window
                r_end = min(r_start + tile_size, n_rows)
                c_end = min(c_start + tile_size, n_cols)
                
                # Buffered window
                buf_r0 = max(0, r_start - overlap)
                buf_r1 = min(n_rows, r_end + overlap)
                buf_c0 = max(0, c_start - overlap)
                buf_c1 = min(n_cols, c_end + overlap)
                
                # Slices inside buffered window corresponding to interior
                off_r0 = r_start - buf_r0
                off_r1 = off_r0 + (r_end - r_start)
                off_c0 = c_start - buf_c0
                off_c1 = off_c0 + (c_end - c_start)
                
                # Extract tile arrays
                dsm_buf = dsm[buf_r0:buf_r1, buf_c0:buf_c1]
                val_buf = valid_land[buf_r0:buf_r1, buf_c0:buf_c1]
                
                # If tile is entirely nodata/sea, fill nodata directly
                if not np.any(val_buf):
                    empty_tile = np.full((r_end - r_start, c_end - c_start), nodata, dtype=np.float32)
                    zero_tile = np.full((r_end - r_start, c_end - c_start), nodata, dtype=np.float32)
                    win_out = Window(c_start, r_start, c_end - c_start, r_end - r_start)
                    dst_dtm.write(empty_tile, 1, window=win_out)
                    dst_ndsm.write(zero_tile, 1, window=win_out)
                    continue
                    
                bld_buf = bld_mask[buf_r0:buf_r1, buf_c0:buf_c1]
                can_buf = canopy_5m[buf_r0:buf_r1, buf_c0:buf_c1]
                T_buf = T_full[buf_r0:buf_r1, buf_c0:buf_c1]
                bld_dil_buf = bld_dilated[buf_r0:buf_r1, buf_c0:buf_c1]
                
                # Process tile
                dtm_buf = process_tile(dsm_buf, bld_buf, can_buf, T_buf, bld_dil_buf, val_buf)
                
                # Crop interior
                dtm_int = dtm_buf[off_r0:off_r1, off_c0:off_c1]
                dsm_int = dsm_buf[off_r0:off_r1, off_c0:off_c1]
                val_int = val_buf[off_r0:off_r1, off_c0:off_c1]
                
                # Strict capping: DTM <= DSM everywhere
                dtm_int = np.minimum(dtm_int, dsm_int)
                ndsm_int = np.maximum(dsm_int - dtm_int, 0.0)
                
                # Set nodata
                dtm_int[~val_int] = nodata
                ndsm_int[~val_int] = nodata
                
                # Write to disk
                win_out = Window(c_start, r_start, c_end - c_start, r_end - r_start)
                dst_dtm.write(dtm_int.astype(np.float32), 1, window=win_out)
                dst_ndsm.write(ndsm_int.astype(np.float32), 1, window=win_out)
                
                del dtm_buf, dsm_buf, bld_buf, can_buf, T_buf, bld_dil_buf, val_buf, dtm_int, dsm_int, ndsm_int, val_int
                
                if tile_idx % 4 == 0 or tile_idx == total_tiles:
                    free_gb = check_ram(f"Tile {tile_idx}/{total_tiles}")
                    print(f"  Tile {tile_idx:2d}/{total_tiles} [{r_start}:{r_end}, {c_start}:{c_end}] written in {time.time()-t_tile:.2f} s (RAM free: {free_gb:.2f} GB)")
                    
    del dsm, bld_mask, canopy_5m, T_full, bld_dilated, valid_land
    
    print(f"\nAll {total_tiles} tiles processed successfully in {time.time()-t_start:.2f} s")
    print(f"DTM v3 saved to: {out_dtm} ({os.path.getsize(out_dtm)/(1024*1024):.2f} MB)")
    print(f"nDSM v3 saved to: {out_ndsm} ({os.path.getsize(out_ndsm)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
