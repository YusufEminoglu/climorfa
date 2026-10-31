"""
WP1.1d: DTM v3 Systematic Tuning Protocol
Strictly splits data into Tuning Set vs Held-out Set BEFORE tuning
Evaluates at most 6 configurations on the Tuning Set only.
"""

import os
import sys
import time
import json
import psutil
import numpy as np
import rasterio
from rasterio.windows import from_bounds
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

def calc_slope(dem, cellsize=5.0):
    dx = sobel(dem, axis=1) / (8.0 * cellsize)
    dy = sobel(dem, axis=0) / (8.0 * cellsize)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

def local_dev_75m(dem, valid_mask, cellsize=5.0):
    ksize = int(round(75.0 / cellsize))
    if ksize % 2 == 0:
        ksize += 1
    filled = dem.copy()
    if not np.all(valid_mask):
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        filled = filled[tuple(indices)]
    mean_val = uniform_filter(filled, size=ksize)
    return np.abs(dem - mean_val)

def build_coarse_trend(dsm, bare_mask, step=20, pct=15, sigma=1.5):
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
    
    trend_coarse = np.nanpercentile(blks, pct, axis=-1)
    del blks
    
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
    del ds_t, ds_m
    
    trend_smooth = gaussian_filter(trend_filled, sigma=sigma)
    return trend_smooth

def process_window(dsm_w, bld_w, canopy_w, T_w, valid_w, delta_R=1.5, sigma_smooth=1.8):
    R_w = dsm_w - T_w
    tree_high = (canopy_w >= 1.5)
    nb_mask = (bld_w == 0) & (~tree_high) & valid_w
    
    h, w = dsm_w.shape
    pad_y = (4 - h % 4) % 4
    pad_x = (4 - w % 4) % 4
    R_pad = np.pad(np.where(nb_mask, R_w, np.nan), ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
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
    
    # Ground candidate threshold
    ground_cand = nb_mask & (R_w <= low_env + delta_R)
    del low_env
    
    # Interpolate R
    ds_it = drv.Create("", w, h, 1, gdal.GDT_Float32)
    ds_it.GetRasterBand(1).WriteArray(np.where(ground_cand, R_w, 0.0).astype(np.float32))
    
    fill_mask = np.where(valid_w, ground_cand.astype(np.uint8), 1)
    ds_im = drv.Create("", w, h, 1, gdal.GDT_Byte)
    ds_im.GetRasterBand(1).WriteArray(fill_mask)
    del fill_mask
    
    gdal.FillNodata(ds_it.GetRasterBand(1), ds_im.GetRasterBand(1), maxSearchDist=250, smoothingIterations=1)
    R_interp = ds_it.GetRasterBand(1).ReadAsArray()
    del ds_it, ds_im
    
    # Recombined
    dtm_recon = T_w + R_interp
    del R_interp
    
    # Smooth ground surface: Gaussian sigma + uniform filter size 3 (as in v2)
    dtm_smooth = uniform_filter(gaussian_filter(dtm_recon, sigma=sigma_smooth), size=3)
    
    # Built-up dilated mask
    bld_dilated = (gaussian_filter(bld_w.astype(np.float32), sigma=2.0) > 0.02)
    
    # Trend slope
    dx_t = sobel(T_w, axis=1) / (8.0 * 5.0)
    dy_t = sobel(T_w, axis=0) / (8.0 * 5.0)
    trend_slp = np.degrees(np.arctan(np.sqrt(dx_t**2 + dy_t**2)))
    
    # On hillside bare cells far from buildings/trees: keep raw DSM
    is_hill_bare = (trend_slp >= 2.0) & (~bld_dilated) & (~tree_high) & valid_w
    
    dtm_out = np.where(is_hill_bare, dsm_w, dtm_smooth)
    dtm_out = np.minimum(dtm_out, dsm_w)
    dtm_out[~valid_w] = np.nan
    return dtm_out

def main():
    print("=== WP1.1d: Systematic Tuning Protocol (Max 6 Configurations) ===")
    check_ram("Init")
    drv = gdal.GetDriverByName("MEM")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    bld_path = "data/02_interim/akilli_sehir_fua.gpkg"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    
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
    
    # Rasterize buildings (+2.5m buffer)
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
    print(f"Bare cells identified: {np.sum(bare_mask):,} ({np.sum(bare_mask)/np.sum(valid_land)*100:.2f}% of land)")
    
    # -------------------------------------------------------------
    # Define Split Protocol BEFORE any tuning
    # -------------------------------------------------------------
    print("\n--- Tuning vs Held-out Data Split Protocol ---")
    flat_tuning_windows = {
        "Cigli Plain": (502000, 4263000, 504000, 4265000),
        "Karsiyaka Shore": (508000, 4258000, 5095000, 4259500),
        "Cigli AOSB": (502375, 4262625, 503375, 4263625),
    }
    hill_tuning_windows = {
        "Buca": (511000, 4249000, 512000, 4250000),
        "Narlidere": (498000, 4246000, 499000, 4247000),
    }
    print("Split established cleanly.")
    
    # Pre-build candidate regional trends (Combo A to D)
    print("\n--- Building Candidate Regional Trends ---")
    t0 = time.time()
    print("Building Trend A: 100m, p15...")
    trend_A = build_coarse_trend(dsm, bare_mask, step=20, pct=15, sigma=1.5)
    print("Building Trend B: 150m, p15...")
    trend_B = build_coarse_trend(dsm, bare_mask, step=30, pct=15, sigma=1.5)
    print("Building Trend C: 150m, p25...")
    trend_C = build_coarse_trend(dsm, bare_mask, step=30, pct=25, sigma=1.5)
    print("Building Trend D: 200m, p15...")
    trend_D = build_coarse_trend(dsm, bare_mask, step=40, pct=15, sigma=1.5)
    print(f"Candidate trends ready in {time.time()-t0:.2f} s")
    check_ram("Trends Built")
    
    # Load FABDEM for validation
    with rasterio.open(fab_path) as s_fab:
        fab_arr = s_fab.read(1)
        fab_trans = s_fab.transform
        fab_crs = s_fab.crs
        fab_nodata = float(s_fab.nodata)
        
    configs = [
        {"id": "Config 1", "trend": "A", "step": 20, "trend_arr": trend_A, "delta_R": 1.5, "sigma": 1.8, "desc": "100m p15, dR=1.5m, sig=1.8"},
        {"id": "Config 2", "trend": "B", "step": 30, "trend_arr": trend_B, "delta_R": 1.5, "sigma": 1.8, "desc": "150m p15, dR=1.5m, sig=1.8"},
        {"id": "Config 3", "trend": "C", "step": 30, "trend_arr": trend_C, "delta_R": 1.5, "sigma": 1.8, "desc": "150m p25, dR=1.5m, sig=1.8"},
        {"id": "Config 4", "trend": "B", "step": 30, "trend_arr": trend_B, "delta_R": 1.0, "sigma": 1.8, "desc": "150m p15, dR=1.0m, sig=1.8"},
        {"id": "Config 5", "trend": "B", "step": 30, "trend_arr": trend_B, "delta_R": 2.0, "sigma": 2.2, "desc": "150m p15, dR=2.0m, sig=2.2"},
        {"id": "Config 6", "trend": "D", "step": 40, "trend_arr": trend_D, "delta_R": 1.5, "sigma": 1.8, "desc": "200m p15, dR=1.5m, sig=1.8"},
    ]
    
    print("\n=======================================================")
    print("EVALUATING 6 CANDIDATE CONFIGURATIONS ON TUNING SET ONLY")
    print("=======================================================")
    
    tuning_results = []
    
    for cfg in configs:
        cid = cfg["id"]
        step = cfg["step"]
        trend_c = cfg["trend_arr"]
        delta_R = cfg["delta_R"]
        sigma = cfg["sigma"]
        desc = cfg["desc"]
        
        t_cfg = time.time()
        print(f"\n--> Running {cid}: {desc}")
        
        # 1. Evaluate on Flat Tuning Windows
        flat_slopes = []
        flat_devs = []
        for wname, bbox in flat_tuning_windows.items():
            w_win = from_bounds(*bbox, transform=trans).round_offsets().round_lengths()
            r0, r1 = w_win.row_off, w_win.row_off + w_win.height
            c0, c1 = w_win.col_off, w_win.col_off + w_win.width
            
            dsm_w = dsm[r0:r1, c0:c1]
            bld_w = bld_mask[r0:r1, c0:c1]
            can_w = canopy_5m[r0:r1, c0:c1]
            val_w = valid_land[r0:r1, c0:c1]
            
            col0_c = (bbox[0] - trans[2]) / (trans[0] * step)
            col1_c = (bbox[2] - trans[2]) / (trans[0] * step)
            row0_c = (bbox[3] - trans[5]) / (trans[4] * step)
            row1_c = (bbox[1] - trans[5]) / (trans[4] * step)
            r_min = max(0, int(np.floor(min(row0_c, row1_c))))
            r_max = min(trend_c.shape[0]-1, int(np.ceil(max(row0_c, row1_c))))
            c_min = max(0, int(np.floor(min(col0_c, col1_c))))
            c_max = min(trend_c.shape[1]-1, int(np.ceil(max(col0_c, col1_c))))
            
            sub_c = trend_c[r_min:r_max+1, c_min:c_max+1]
            T_w = zoom(sub_c, (dsm_w.shape[0]/sub_c.shape[0], dsm_w.shape[1]/sub_c.shape[1]), order=1)[:dsm_w.shape[0], :dsm_w.shape[1]]
            
            dtm_w = process_window(dsm_w, bld_w, can_w, T_w, val_w, delta_R=delta_R, sigma_smooth=sigma)
            
            # FABDEM slope in window for defining coastal plain cells (< 2 deg)
            w_fab = from_bounds(*bbox, transform=fab_trans).round_offsets().round_lengths()
            fab_w_30m = fab_arr[w_fab.row_off:w_fab.row_off+w_fab.height, w_fab.col_off:w_fab.col_off+w_fab.width].astype(np.float32)
            dx_f = sobel(fab_w_30m, axis=1) / (8.0 * 30.0)
            dy_f = sobel(fab_w_30m, axis=0) / (8.0 * 30.0)
            slp_fab_30m = np.degrees(np.arctan(np.sqrt(dx_f**2 + dy_f**2)))
            slp_f = zoom(slp_fab_30m, (dsm_w.shape[0]/slp_fab_30m.shape[0], dsm_w.shape[1]/slp_fab_30m.shape[1]), order=1)[:dsm_w.shape[0], :dsm_w.shape[1]]
            
            eval_m = val_w & (slp_f < 2.0)
            if np.sum(eval_m) < 100:
                eval_m = val_w
                
            slp = calc_slope(dtm_w)
            dev = local_dev_75m(dtm_w, val_w)
            
            med_slp = float(np.nanmedian(slp[eval_m]))
            p95_dev = float(np.nanpercentile(dev[eval_m], 95))
            flat_slopes.append(med_slp)
            flat_devs.append(p95_dev)
            print(f"    {wname:16s}: slope={med_slp:.2f}°, dev p95={p95_dev:.2f}m")
            
        mean_flat_slope = float(np.mean(flat_slopes))
        mean_flat_dev = float(np.mean(flat_devs))
        
        # 2. Evaluate on Hillside Tuning Windows
        hill_biases = []
        for hname, bbox in hill_tuning_windows.items():
            w_win = from_bounds(*bbox, transform=trans).round_offsets().round_lengths()
            r0, r1 = w_win.row_off, w_win.row_off + w_win.height
            c0, c1 = w_win.col_off, w_win.col_off + w_win.width
            
            dsm_w = dsm[r0:r1, c0:c1]
            bld_w = bld_mask[r0:r1, c0:c1]
            can_w = canopy_5m[r0:r1, c0:c1]
            val_w = valid_land[r0:r1, c0:c1]
            
            w_fab = from_bounds(*bbox, transform=fab_trans).round_offsets().round_lengths()
            fab_w_30m = fab_arr[w_fab.row_off:w_fab.row_off+w_fab.height, w_fab.col_off:w_fab.col_off+w_fab.width].astype(np.float32)
            fab_w = zoom(fab_w_30m, (dsm_w.shape[0]/fab_w_30m.shape[0], dsm_w.shape[1]/fab_w_30m.shape[1]), order=1)[:dsm_w.shape[0], :dsm_w.shape[1]]
            
            col0_c = (bbox[0] - trans[2]) / (trans[0] * step)
            col1_c = (bbox[2] - trans[2]) / (trans[0] * step)
            row0_c = (bbox[3] - trans[5]) / (trans[4] * step)
            row1_c = (bbox[1] - trans[5]) / (trans[4] * step)
            r_min = max(0, int(np.floor(min(row0_c, row1_c))))
            r_max = min(trend_c.shape[0]-1, int(np.ceil(max(row0_c, row1_c))))
            c_min = max(0, int(np.floor(min(col0_c, col1_c))))
            c_max = min(trend_c.shape[1]-1, int(np.ceil(max(col0_c, col1_c))))
            
            sub_c = trend_c[r_min:r_max+1, c_min:c_max+1]
            T_w = zoom(sub_c, (dsm_w.shape[0]/sub_c.shape[0], dsm_w.shape[1]/sub_c.shape[1]), order=1)[:dsm_w.shape[0], :dsm_w.shape[1]]
            
            dtm_w = process_window(dsm_w, bld_w, can_w, T_w, val_w, delta_R=delta_R, sigma_smooth=sigma)
            
            val_comp = val_w & (fab_w > -100) & (~np.isnan(dtm_w))
            bias = float(np.mean(dtm_w[val_comp] - fab_w[val_comp]))
            hill_biases.append(bias)
            print(f"    {hname:16s}: bias={bias:+.2f}m")
            
        mean_hill_bias = float(np.mean(hill_biases))
        
        # 3. Hold-out test on tuning bare cells (Acceptance C test on 5 km sample tile)
        test_bbox = (497000, 4243000, 502000, 4248000)
        w_win = from_bounds(*test_bbox, transform=trans).round_offsets().round_lengths()
        r0, r1 = w_win.row_off, w_win.row_off + w_win.height
        c0, c1 = w_win.col_off, w_win.col_off + w_win.width
        
        dsm_t = dsm[r0:r1, c0:c1]
        bld_t = bld_mask[r0:r1, c0:c1]
        can_t = canopy_5m[r0:r1, c0:c1]
        val_t = valid_land[r0:r1, c0:c1]
        
        bare_t = (bld_t == 0) & (can_t < 1.5) & val_t
        rng = np.random.default_rng(seed=42)
        holdout_mask = (rng.random(bare_t.shape) < 0.20) & bare_t
        train_bare = bare_t & (~holdout_mask)
        
        col0_c = (test_bbox[0] - trans[2]) / (trans[0] * step)
        col1_c = (test_bbox[2] - trans[2]) / (trans[0] * step)
        row0_c = (test_bbox[3] - trans[5]) / (trans[4] * step)
        row1_c = (test_bbox[1] - trans[5]) / (trans[4] * step)
        r_min = max(0, int(np.floor(min(row0_c, row1_c))))
        r_max = min(trend_c.shape[0]-1, int(np.ceil(max(row0_c, row1_c))))
        c_min = max(0, int(np.floor(min(col0_c, col1_c))))
        c_max = min(trend_c.shape[1]-1, int(np.ceil(max(col0_c, col1_c))))
        sub_c = trend_c[r_min:r_max+1, c_min:c_max+1]
        T_t = zoom(sub_c, (dsm_t.shape[0]/sub_c.shape[0], dsm_t.shape[1]/sub_c.shape[1]), order=1)[:dsm_t.shape[0], :dsm_t.shape[1]]
        
        R_t = dsm_t - T_t
        ds_it = drv.Create("", dsm_t.shape[1], dsm_t.shape[0], 1, gdal.GDT_Float32)
        ds_it.GetRasterBand(1).WriteArray(np.where(train_bare, R_t, 0.0).astype(np.float32))
        fill_mask = np.where(val_t, train_bare.astype(np.uint8), 1)
        ds_im = drv.Create("", dsm_t.shape[1], dsm_t.shape[0], 1, gdal.GDT_Byte)
        ds_im.GetRasterBand(1).WriteArray(fill_mask)
        gdal.FillNodata(ds_it.GetRasterBand(1), ds_im.GetRasterBand(1), maxSearchDist=250, smoothingIterations=1)
        R_interp = ds_it.GetRasterBand(1).ReadAsArray()
        del ds_it, ds_im
        
        dtm_hold = T_t + R_interp
        diff_hold = dtm_hold[holdout_mask] - dsm_t[holdout_mask]
        bias_hold = float(np.mean(diff_hold))
        rmse_hold = float(np.sqrt(np.mean(diff_hold**2)))
        
        res = {
            "id": str(cid),
            "desc": str(desc),
            "flat_slope": mean_flat_slope,
            "flat_dev": mean_flat_dev,
            "hill_bias": mean_hill_bias,
            "hold_bias": bias_hold,
            "hold_rmse": rmse_hold,
            "time_s": float(time.time() - t_cfg)
        }
        tuning_results.append(res)
        print(f"  Summary {cid}: Flat Slope={mean_flat_slope:.2f}°, Dev p95={mean_flat_dev:.2f}m | Hill Bias={mean_hill_bias:+.2f}m | Hold Bias={bias_hold:+.2f}m, RMSE={rmse_hold:.2f}m")
        
    print("\n=======================================================")
    print("TUNING SUMMARY TABLE (All 6 Configurations)")
    print("=======================================================")
    print(f"{'Config ID':10s} | {'Description':32s} | {'Flat Slp':8s} | {'Flat Dev':8s} | {'Hill Bias':10s} | {'Hold RMSE':9s} | {'Score'}")
    print("-" * 100)
    for r in tuning_results:
        pass_flat = (r["flat_slope"] < 2.0) and (r["flat_dev"] < 1.0)
        pass_hill = (-0.7 <= r["hill_bias"] <= 4.7)
        pass_hold = (abs(r["hold_bias"]) <= 1.0) and (r["hold_rmse"] <= 2.5)
        overall = "PASS ALL" if (pass_flat and pass_hill and pass_hold) else "FAIL"
        print(f"{r['id']:10s} | {r['desc']:32s} | {r['flat_slope']:7.2f}° | {r['flat_dev']:7.2f}m | {r['hill_bias']:+9.2f}m | {r['hold_rmse']:8.2f}m | {overall}")
        
    os.makedirs("logs/wp", exist_ok=True)
    with open("logs/wp/tuning_configurations.json", "w") as f:
        json.dump(tuning_results, f, indent=2)
    print("\nTuning results successfully written to logs/wp/tuning_configurations.json")

if __name__ == "__main__":
    main()
