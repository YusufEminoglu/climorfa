"""
WP1.1d Step 3: Comprehensive Non-Circular Validation Suite for DTM v3
Evaluates Acceptance Criteria A, B, C, D, E
Reports raw numbers, PASS/FAIL for all tests
Author: Yusuf Eminoglu / Antigravity
"""

import os
import sys
import time
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio import features
from rasterio.warp import reproject, Resampling
from scipy.ndimage import (
    uniform_filter,
    sobel,
    gaussian_filter,
    distance_transform_edt,
    zoom
)
from osgeo import gdal
import pyogrio

gdal.UseExceptions()

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

def main():
    t_start = time.time()
    print("=== WP1.1d Step 3: Comprehensive Non-Circular Validation Suite for DTM v3 ===")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    dtm_v1_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    dtm_v2_path = "data/street/terrain/dtm_5m_v2_epsg5253.tif"
    dtm_v3_path = "data/street/terrain/dtm_5m_v3_epsg5253.tif"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    shore_path = "data/02_interim/shoreline_osm_epsg5253.gpkg"
    bld_path = "data/02_interim/akilli_sehir_fua.gpkg"
    
    validation_results = {}
    
    # -------------------------------------------------------------
    # 1. Acceptance A: Flat Cores (5 Coastal Plain Windows)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ACCEPTANCE A: Flat Cores (5 Coastal Plain Windows)")
    print("="*70)
    
    windows_A = {
        "Alsancak (Dense core) [HELD-OUT]": (512350, 4255850, 512850, 4256350, True),
        "Bayrakli Shore [HELD-OUT]": (513000, 4257000, 514500, 4258500, True),
        "Cigli Plain (Coastal flat) [Tuning]": (502000, 4263000, 504000, 4265000, False),
        "Karsiyaka Shore [Tuning]": (508000, 4258000, 5095000, 4259500, False),
        "Cigli AOSB (Industrial) [Tuning]": (502375, 4262625, 503375, 4263625, False),
    }
    
    acc_A_table = []
    with rasterio.open(dsm_path) as s_dsm, rasterio.open(dtm_v1_path) as s_v1, rasterio.open(dtm_v2_path) as s_v2, rasterio.open(dtm_v3_path) as s_v3, rasterio.open(fab_path) as s_fab:
        for wname, (minx, miny, maxx, maxy, is_heldout) in windows_A.items():
            bbox = (minx, miny, maxx, maxy)
            win_5m = from_bounds(*bbox, transform=s_dsm.transform).round_offsets().round_lengths()
            dsm_w = s_dsm.read(1, window=win_5m).astype(np.float32)
            v1_w = s_v1.read(1, window=win_5m).astype(np.float32)
            v2_w = s_v2.read(1, window=win_5m).astype(np.float32)
            v3_w = s_v3.read(1, window=win_5m).astype(np.float32)
            valid_w = (dsm_w != s_dsm.nodata) & (dsm_w >= -1.0) & (v3_w != s_v3.nodata)
            
            # 30m FABDEM slope
            win_fab = from_bounds(*bbox, transform=s_fab.transform).round_offsets().round_lengths()
            fab_w = s_fab.read(1, window=win_fab).astype(np.float32)
            dx_f = sobel(fab_w, axis=1) / (8.0 * 30.0)
            dy_f = sobel(fab_w, axis=0) / (8.0 * 30.0)
            fab_slp_30m = np.degrees(np.arctan(np.sqrt(dx_f**2 + dy_f**2)))
            fab_slp_5m = zoom(fab_slp_30m, (dsm_w.shape[0]/fab_slp_30m.shape[0], dsm_w.shape[1]/fab_slp_30m.shape[1]), order=1)[:dsm_w.shape[0], :dsm_w.shape[1]]
            
            eval_mask = valid_w & (fab_slp_5m < 2.0)
            if np.sum(eval_mask) < 100:
                eval_mask = valid_w
                
            slp_v1 = calc_slope(v1_w)
            slp_v2 = calc_slope(v2_w)
            slp_v3 = calc_slope(v3_w)
            
            dev_v1 = local_dev_75m(v1_w, valid_w)
            dev_v2 = local_dev_75m(v2_w, valid_w)
            dev_v3 = local_dev_75m(v3_w, valid_w)
            
            s1_med, d1_p95 = float(np.median(slp_v1[eval_mask])), float(np.percentile(dev_v1[eval_mask], 95))
            s2_med, d2_p95 = float(np.median(slp_v2[eval_mask])), float(np.percentile(dev_v2[eval_mask], 95))
            s3_med, d3_p95 = float(np.median(slp_v3[eval_mask])), float(np.percentile(dev_v3[eval_mask], 95))
            
            pass_slp = s3_med < 2.0
            pass_dev = d3_p95 < 1.0
            st = "PASS" if (pass_slp and pass_dev) else "FAIL"
            
            entry = {
                "window": wname,
                "is_heldout": is_heldout,
                "v1_slope_med": s1_med,
                "v1_dev_p95": d1_p95,
                "v2_slope_med": s2_med,
                "v2_dev_p95": d2_p95,
                "v3_slope_med": s3_med,
                "v3_dev_p95": d3_p95,
                "status": st
            }
            acc_A_table.append(entry)
            print(f"  {wname:36s}: slope={s3_med:.2f}° (v1={s1_med:.2f}°, v2={s2_med:.2f}°) | dev p95={d3_p95:.2f}m (v1={d1_p95:.2f}m, v2={d2_p95:.2f}m) => {st}")
            
    validation_results["acceptance_A"] = acc_A_table
    
    # -------------------------------------------------------------
    # 2. Acceptance D: Built-up Hillside Windows
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ACCEPTANCE D: Built-up Hillside Windows")
    print("="*70)
    
    windows_D = {
        "Karabaglar [HELD-OUT]": (508000, 4247000, 509000, 4248000, True),
        "Bayrakli hill [HELD-OUT]": (514000, 4260000, 515000, 4261000, True),
        "Buca [Tuning]": (511000, 4249000, 512000, 4250000, False),
        "Narlidere [Tuning]": (498000, 4246000, 499000, 4247000, False),
    }
    
    acc_D_table = []
    with rasterio.open(dtm_v1_path) as s_v1, rasterio.open(dtm_v2_path) as s_v2, rasterio.open(dtm_v3_path) as s_v3, rasterio.open(fab_path) as s_fab:
        for hname, (minx, miny, maxx, maxy, is_heldout) in windows_D.items():
            bbox = (minx, miny, maxx, maxy)
            win_5m = from_bounds(*bbox, transform=s_v3.transform).round_offsets().round_lengths()
            v1_w = s_v1.read(1, window=win_5m).astype(np.float32)
            v2_w = s_v2.read(1, window=win_5m).astype(np.float32)
            v3_w = s_v3.read(1, window=win_5m).astype(np.float32)
            
            win_fab = from_bounds(*bbox, transform=s_fab.transform).round_offsets().round_lengths()
            fab_w = s_fab.read(1, window=win_fab).astype(np.float32)
            fab_5m = zoom(fab_w, (v3_w.shape[0]/fab_w.shape[0], v3_w.shape[1]/fab_w.shape[1]), order=1)[:v3_w.shape[0], :v3_w.shape[1]]
            
            val = (v3_w != s_v3.nodata) & (fab_5m > -100) & (v1_w != s_v1.nodata) & (v2_w != s_v2.nodata)
            
            bias_v1 = float(np.mean(v1_w[val] - fab_5m[val]))
            bias_v2 = float(np.mean(v2_w[val] - fab_5m[val]))
            bias_v3 = float(np.mean(v3_w[val] - fab_5m[val]))
            rmse_v3 = float(np.sqrt(np.mean((v3_w[val] - fab_5m[val])**2)))
            
            # Acceptance D: v3 - FABDEM bias within (+1.3...+2.7 m) +/- 2 m => [-0.7 m, +4.7 m]
            st = "PASS" if (-0.7 <= bias_v3 <= 4.7) else "FAIL"
            entry = {
                "window": hname,
                "is_heldout": is_heldout,
                "v1_bias": bias_v1,
                "v2_bias": bias_v2,
                "v3_bias": bias_v3,
                "v3_rmse": rmse_v3,
                "status": st
            }
            acc_D_table.append(entry)
            print(f"  {hname:26s}: v3-FAB bias={bias_v3:+.2f}m (v1={bias_v1:+.2f}m, v2={bias_v2:+.2f}m) | rmse={rmse_v3:.2f}m | req [-0.7..+4.7m] => {st}")
            
    validation_results["acceptance_D"] = acc_D_table
    
    # -------------------------------------------------------------
    # 3. Acceptance B: Bare Open Cells Departure per Slope Class
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ACCEPTANCE B: Bare Open Cells Departure (|v3 - FABDEM| - |DSM - FABDEM|)")
    print("="*70)
    
    with rasterio.open(fab_path) as s_fab:
        fab_30m = s_fab.read(1).astype(np.float32)
        trans_30m = s_fab.transform
        crs_30m = s_fab.crs
        nodata_fab = float(s_fab.nodata)
        shape_30m = fab_30m.shape
        
    dx = sobel(fab_30m, axis=1) / (8.0 * 30.0)
    dy = sobel(fab_30m, axis=0) / (8.0 * 30.0)
    slope_30m = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    print("Aggregating DSM, DTM v1, DTM v2, DTM v3, and ETH to 30m FABDEM grid...")
    dsm_30m = np.full(shape_30m, -9999.0, dtype=np.float32)
    v1_30m = np.full(shape_30m, -9999.0, dtype=np.float32)
    v2_30m = np.full(shape_30m, -9999.0, dtype=np.float32)
    v3_30m = np.full(shape_30m, -9999.0, dtype=np.float32)
    eth_30m = np.full(shape_30m, -9999.0, dtype=np.float32)
    
    with rasterio.open(dsm_path) as s:
        reproject(rasterio.band(s, 1), dsm_30m, src_transform=s.transform, src_crs=s.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s.nodata, dst_nodata=-9999.0)
    with rasterio.open(dtm_v1_path) as s:
        reproject(rasterio.band(s, 1), v1_30m, src_transform=s.transform, src_crs=s.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s.nodata, dst_nodata=-9999.0)
    with rasterio.open(dtm_v2_path) as s:
        reproject(rasterio.band(s, 1), v2_30m, src_transform=s.transform, src_crs=s.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s.nodata, dst_nodata=-9999.0)
    with rasterio.open(dtm_v3_path) as s:
        reproject(rasterio.band(s, 1), v3_30m, src_transform=s.transform, src_crs=s.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s.nodata, dst_nodata=-9999.0)
    with rasterio.open(eth_path) as s:
        reproject(rasterio.band(s, 1), eth_30m, src_transform=s.transform, src_crs=s.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s.nodata, dst_nodata=-9999.0)
        
    val_30 = (fab_30m != nodata_fab) & (fab_30m >= 0) & (dsm_30m > -100) & (v1_30m > -100) & (v2_30m > -100) & (v3_30m > -100)
    bare_30 = val_30 & (eth_30m < 1.0) & (np.abs(dsm_30m - v1_30m) < 1.0)
    
    slope_bins = [
        ("0-2", 0.0, 2.0),
        ("2-5", 2.0, 5.0),
        ("5-15", 5.0, 15.0),
        (">15", 15.0, 90.0)
    ]
    
    acc_B_table = []
    for sname, slo, shi in slope_bins:
        m = bare_30 & (slope_30m >= slo) & (slope_30m < shi)
        n = int(np.sum(m))
        bias_dsm = float(np.mean(dsm_30m[m] - fab_30m[m]))
        bias_v1 = float(np.mean(v1_30m[m] - fab_30m[m]))
        bias_v2 = float(np.mean(v2_30m[m] - fab_30m[m]))
        rmse_v2 = float(np.sqrt(np.mean((v2_30m[m] - fab_30m[m])**2)))
        bias_v3 = float(np.mean(v3_30m[m] - fab_30m[m]))
        rmse_v3 = float(np.sqrt(np.mean((v3_30m[m] - fab_30m[m])**2)))
        departure = float(np.abs(bias_v3 - bias_dsm))
        
        st = "PASS" if departure <= 1.0 else "FAIL"
        entry = {
            "slope_class": sname,
            "n_cells": n,
            "rawDSM_bias": bias_dsm,
            "v1_bias": bias_v1,
            "v2_bias": bias_v2,
            "v2_rmse": rmse_v2,
            "v3_bias": bias_v3,
            "v3_rmse": rmse_v3,
            "departure": departure,
            "status": st
        }
        acc_B_table.append(entry)
        print(f"  Slope {sname:6s} (N={n:,}): rawDSM-FAB={bias_dsm:+.2f}m | v1-FAB={bias_v1:+.2f}m | v2-FAB={bias_v2:+.2f}m | v3-FAB={bias_v3:+.2f}m (rmse={rmse_v3:.2f}m) | |diff|={departure:.2f}m => {st}")
        
    validation_results["acceptance_B"] = acc_B_table
    
    # -------------------------------------------------------------
    # 4. Acceptance C: 20% Stratified Hold-out Test
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ACCEPTANCE C: 20% Stratified Hold-out Rebuild Test")
    print("="*70)
    
    rng = np.random.default_rng(seed=42)
    acc_C_table = []
    drv = gdal.GetDriverByName("MEM")
    
    regions = [
        ("0-2 (Flat plain)", (501000, 4264000, 505000, 4268000), 0.0, 2.0),
        ("2-5 (Gentle slope)", (515000, 4252000, 519000, 4256000), 2.0, 5.0),
        ("5-15 (Moderate slope)", (508000, 4242000, 512000, 4246000), 5.0, 15.0),
        (">15 (Steep mountains)", (496000, 4241000, 500000, 4245000), 15.0, 90.0),
    ]
    
    with rasterio.open(dsm_path) as s_dsm, rasterio.open(dtm_v3_path) as s_v3:
        for rname, bbox, slo, shi in regions:
            w_5m = from_bounds(*bbox, transform=s_dsm.transform).round_offsets().round_lengths()
            dsm_r = s_dsm.read(1, window=w_5m).astype(np.float32)
            v3_r = s_v3.read(1, window=w_5m).astype(np.float32)
            val_r = (dsm_r != s_dsm.nodata) & (dsm_r >= -1.0) & (v3_r != s_v3.nodata)
            
            bld_df = pyogrio.read_dataframe(bld_path, layer="buildings_fua", bbox=bbox)
            buffered = [g.buffer(2.5) for g in bld_df.geometry if g is not None and not g.is_empty]
            del bld_df
            bld_m = np.zeros(dsm_r.shape, dtype=np.uint8)
            if len(buffered) > 0:
                trans_r = rasterio.windows.transform(w_5m, s_dsm.transform)
                bld_m = features.rasterize(((g, 1) for g in buffered), out_shape=dsm_r.shape, transform=trans_r, dtype=np.uint8, fill=0)
            del buffered
            
            with rasterio.open(eth_path) as s_eth:
                w_eth = from_bounds(*bbox, transform=s_eth.transform).round_offsets().round_lengths()
                c_10m = s_eth.read(1, window=w_eth).astype(np.float32)
            c_5m = zoom(c_10m, (dsm_r.shape[0]/c_10m.shape[0], dsm_r.shape[1]/c_10m.shape[1]), order=1)[:dsm_r.shape[0], :dsm_r.shape[1]]
            del c_10m
            
            bare_r = val_r & (bld_m == 0) & (c_5m < 1.5)
            del bld_m, c_5m
            
            slp_r = calc_slope(v3_r)
            bare_class = bare_r & (slp_r >= slo) & (slp_r < shi)
            n_class = int(np.sum(bare_class))
            if n_class < 100:
                bare_class = bare_r
                n_class = int(np.sum(bare_class))
                
            holdout_m = (rng.random(dsm_r.shape) < 0.20) & bare_class
            train_m = bare_r & (~holdout_m)
            n_hold = int(np.sum(holdout_m))
            
            ds_it = drv.Create("", dsm_r.shape[1], dsm_r.shape[0], 1, gdal.GDT_Float32)
            ds_it.GetRasterBand(1).WriteArray(np.where(train_m, dsm_r, 0.0).astype(np.float32))
            f_mask = np.where(val_r, train_m.astype(np.uint8), 1)
            ds_im = drv.Create("", dsm_r.shape[1], dsm_r.shape[0], 1, gdal.GDT_Byte)
            ds_im.GetRasterBand(1).WriteArray(f_mask)
            del f_mask
            
            gdal.FillNodata(ds_it.GetRasterBand(1), ds_im.GetRasterBand(1), maxSearchDist=250, smoothingIterations=1)
            d_rebuilt = ds_it.GetRasterBand(1).ReadAsArray()
            del ds_it, ds_im
            
            diff_hold = d_rebuilt[holdout_m] - dsm_r[holdout_m]
            del d_rebuilt
            
            bias_h = float(np.mean(diff_hold))
            rmse_h = float(np.sqrt(np.mean(diff_hold**2)))
            
            st = "PASS" if (abs(bias_h) <= 1.0 and rmse_h <= 2.5) else "FAIL"
            entry = {
                "slope_class": rname,
                "n_withheld": n_hold,
                "residual_bias": bias_h,
                "residual_rmse": rmse_h,
                "status": st
            }
            acc_C_table.append(entry)
            print(f"  Slope {rname:24s} (N={n_hold:,}): Bias={bias_h:+.2f}m (req +/-1.0m) | RMSE={rmse_h:.2f}m (req <=2.5m) => {st}")
            
    validation_results["acceptance_C"] = acc_C_table
    
    # -------------------------------------------------------------
    # 5. Acceptance E: Sea-Level Plausibility
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ACCEPTANCE E: Sea-Level Elevation Plausibility (50 m Shoreline Buffer)")
    print("="*70)
    
    shore_gdf = pyogrio.read_dataframe(shore_path)
    shore_buffer = shore_gdf.geometry.buffer(50.0)
    del shore_gdf
    
    with rasterio.open(dtm_v3_path) as s_v3:
        shore_mask = features.rasterize(
            ((g, 1) for g in shore_buffer if g is not None and not g.is_empty),
            out_shape=s_v3.shape,
            transform=s_v3.transform,
            dtype=np.uint8,
            fill=0
        )
        del shore_buffer
        v3_arr = s_v3.read(1)
        valid_v3 = (v3_arr != s_v3.nodata)
        
    shore_cells = valid_v3 & (shore_mask == 1)
    del shore_mask, valid_v3
    n_shore = int(np.sum(shore_cells))
    elev_shore = v3_arr[shore_cells]
    del v3_arr, shore_cells
    
    med_elev = float(np.median(elev_shore))
    mean_elev = float(np.mean(elev_shore))
    p10_elev = float(np.percentile(elev_shore, 10))
    p90_elev = float(np.percentile(elev_shore, 90))
    min_elev = float(np.min(elev_shore))
    max_elev = float(np.max(elev_shore))
    
    st = "PASS" if (0.0 <= med_elev <= 3.0 and min_elev >= -1.0) else "FAIL"
    acc_E = {
        "n_shore_cells": n_shore,
        "median_elev": med_elev,
        "mean_elev": mean_elev,
        "p10_elev": p10_elev,
        "p90_elev": p90_elev,
        "min_elev": min_elev,
        "max_elev": max_elev,
        "status": st
    }
    validation_results["acceptance_E"] = acc_E
    print(f"  Shoreline Buffer (N={n_shore:,} land cells):")
    print(f"    Median Elevation: {med_elev:.2f} m (req 0..3 m)")
    print(f"    Mean Elevation:   {mean_elev:.2f} m")
    print(f"    10th Percentile:  {p10_elev:.2f} m")
    print(f"    90th Percentile:  {p90_elev:.2f} m")
    print(f"    Range:            [{min_elev:.2f} .. {max_elev:.2f}] m")
    print(f"    Status:           {st}")
    
    # -------------------------------------------------------------
    # 6. Overall FUA vs FABDEM Comparison
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("OVERALL FUA vs FABDEM 30 m REFERENCE COMPARISON")
    print("="*70)
    
    d_v1 = v1_30m[val_30] - fab_30m[val_30]
    d_v2 = v2_30m[val_30] - fab_30m[val_30]
    d_v3 = v3_30m[val_30] - fab_30m[val_30]
    
    fua_metrics = {
        "n_valid_30m": int(np.sum(val_30)),
        "v1_bias": float(np.mean(d_v1)),
        "v1_rmse": float(np.sqrt(np.mean(d_v1**2))),
        "v1_mae": float(np.mean(np.abs(d_v1))),
        "v2_bias": float(np.mean(d_v2)),
        "v2_rmse": float(np.sqrt(np.mean(d_v2**2))),
        "v2_mae": float(np.mean(np.abs(d_v2))),
        "v3_bias": float(np.mean(d_v3)),
        "v3_rmse": float(np.sqrt(np.mean(d_v3**2))),
        "v3_mae": float(np.mean(np.abs(d_v3))),
    }
    validation_results["overall_fua"] = fua_metrics
    print(f"  FUA Comparison Cells (30 m): N = {fua_metrics['n_valid_30m']:,}")
    print(f"  DTM v1 vs FABDEM: Bias = {fua_metrics['v1_bias']:+.2f} m | RMSE = {fua_metrics['v1_rmse']:.2f} m | MAE = {fua_metrics['v1_mae']:.2f} m")
    print(f"  DTM v2 vs FABDEM: Bias = {fua_metrics['v2_bias']:+.2f} m | RMSE = {fua_metrics['v2_rmse']:.2f} m | MAE = {fua_metrics['v2_mae']:.2f} m")
    print(f"  DTM v3 vs FABDEM: Bias = {fua_metrics['v3_bias']:+.2f} m | RMSE = {fua_metrics['v3_rmse']:.2f} m | MAE = {fua_metrics['v3_mae']:.2f} m")
    
    os.makedirs("logs/wp", exist_ok=True)
    with open("logs/wp/validation_results_v3.json", "w") as f:
        json.dump(validation_results, f, indent=2)
    print("\nValidation metrics written to logs/wp/validation_results_v3.json")

if __name__ == "__main__":
    main()
