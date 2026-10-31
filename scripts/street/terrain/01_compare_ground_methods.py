"""
WP1.1b Step 1 & 2: Compare Ground Extraction Methods Across Izmir Evaluation Windows
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Implements and benchmarks:
1. Method (a): Progressive Morphological Filter (PMF, Zhang et al. 2003, IEEE TGRS 41:872-882)
   - Windows growing from 10 m to 75 m
   - Slope-dependent elevation threshold: dh_max = s * (w_k - w_{k-1}) * c + dh0
2. Method (b): Moving-Window Low Percentile (p10, 75 m window) of non-building cells + Gaussian smoothing (sigma=15 m)
3. Method (c): Wide-Corridor Anchor Hybrid (rejecting narrow SOKAK, keeping only wide BULVAR/OTOYOL)
4. Baseline: WP1.1 DTM v1

Evaluates against:
- FABDEM v1-2 (Hawker et al. 2022, 30 m bare-earth Copernicus GLO-30)
- Slope distribution (median slope, p95 slope)
- Local deviation from 75 m mean p95
- Acceptance criteria in flat coastal plains: median slope < 2 deg, dev p95 < 1.0 m
- Physical capping compliance: DTM <= DSM everywhere
"""

import os
import sys
import time
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio import features
from scipy.ndimage import (
    grey_erosion,
    grey_dilation,
    gaussian_filter,
    uniform_filter,
    sobel,
    minimum_filter,
    zoom
)
from scipy.interpolate import griddata, NearestNDInterpolator
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
    # Replace nodata with nearest valid land value before convolution to prevent edge explosion
    from scipy.ndimage import distance_transform_edt
    filled = dem.copy()
    if not np.all(valid_mask):
        indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
        filled = filled[tuple(indices)]
    mean_val = uniform_filter(filled, size=ksize)
    dev = np.abs(dem - mean_val)
    return dev

def run_method_a_pmf(dsm, non_bld_mask, cellsize=5.0, max_win_m=75.0, s=0.08, dh0=0.3, dhmax=2.5, sigma_m=15.0):
    """
    Progressive Morphological Filter (Zhang et al. 2003)
    """
    win_sizes_m = [10.0, 20.0, 35.0, 50.0, max_win_m]
    win_sizes_px = [int(round(w / cellsize)) for w in win_sizes_m]
    win_sizes_px = [w if w % 2 == 1 else w + 1 for w in win_sizes_px]
    
    is_ground = non_bld_mask.copy().astype(bool)
    last_z = np.where(non_bld_mask, dsm.astype(np.float32), 1e6)
    last_w = 1
    
    for w in win_sizes_px:
        # Opening
        eroded = grey_erosion(last_z, size=(w, w))
        z_open = grey_dilation(eroded, size=(w, w))
        dh = last_z - z_open
        
        # Elevation threshold
        dh_t = s * (w - last_w) * cellsize + dh0
        dh_t = min(dh_t, dhmax)
        
        # Mark non-ground
        non_ground = (dh > dh_t) | (~non_bld_mask)
        is_ground[non_ground] = False
        
        last_z = np.where(is_ground, last_z, z_open)
        last_w = w
        
    # Interpolate ground using GDAL FillNodata
    dtm_ground = np.where(is_ground, last_z, 0.0)
    drv = gdal.GetDriverByName("MEM")
    ds_target = drv.Create("", dsm.shape[1], dsm.shape[0], 1, gdal.GDT_Float32)
    ds_target.GetRasterBand(1).WriteArray(dtm_ground)
    ds_mask = drv.Create("", dsm.shape[1], dsm.shape[0], 1, gdal.GDT_Byte)
    ds_mask.GetRasterBand(1).WriteArray(is_ground.astype(np.uint8))
    
    gdal.FillNodata(ds_target.GetRasterBand(1), ds_mask.GetRasterBand(1), maxSearchDist=100, smoothingIterations=0)
    dtm_filled = ds_target.GetRasterBand(1).ReadAsArray()
    
    # Smooth with Gaussian
    dtm_smooth = gaussian_filter(dtm_filled, sigma=sigma_m / cellsize)
    dtm_final = np.minimum(dtm_smooth, dsm)
    return dtm_final

def run_method_b_low_percentile(dsm, non_bld_mask, cellsize=5.0, win_m=75.0, pct=10, sigma_m=15.0):
    """
    Moving-Window Low Percentile (p10, 75 m) of non-building cells + Gaussian smoothing
    """
    radius_px = int(round((win_m / 2.0) / cellsize))
    step = max(3, int(round(25.0 / cellsize))) # 25 m coarse sampling
    ny, nx = dsm.shape
    coarse_y, coarse_x, coarse_z = [], [], []
    
    for y in range(0, ny, step):
        for x in range(0, nx, step):
            y0, y1 = max(0, y - radius_px), min(ny, y + radius_px + 1)
            x0, x1 = max(0, x - radius_px), min(nx, x + radius_px + 1)
            patch = dsm[y0:y1, x0:x1]
            valid = non_bld_mask[y0:y1, x0:x1]
            vals = patch[valid]
            if len(vals) >= 3:
                coarse_y.append(y)
                coarse_x.append(x)
                coarse_z.append(np.percentile(vals, pct))
                
    if len(coarse_z) == 0:
        return dsm
        
    grid_y, grid_x = np.mgrid[0:ny, 0:nx]
    dtm_interp = griddata((coarse_y, coarse_x), coarse_z, (grid_y, grid_x), method="linear")
    near_interp = NearestNDInterpolator((coarse_y, coarse_x), coarse_z)
    nan_mask = np.isnan(dtm_interp)
    dtm_interp[nan_mask] = near_interp(grid_y[nan_mask], grid_x[nan_mask])
    
    dtm_smooth = gaussian_filter(dtm_interp, sigma=sigma_m / cellsize)
    dtm_final = np.minimum(dtm_smooth, dsm)
    return dtm_final

def run_method_c_wide_roads(dsm, non_bld_mask, wide_roads_mask, cellsize=5.0, win_m=75.0, sigma_m=15.0):
    """
    Method (c): Low-percentile ground augmented with wide-corridor road anchors only
    """
    # Non-building plus wide roads as preferred anchors
    anchors = non_bld_mask | wide_roads_mask
    return run_method_b_low_percentile(dsm, anchors, cellsize=cellsize, win_m=win_m, pct=10, sigma_m=sigma_m)

def main():
    t_start = time.time()
    print("=== WP1.1b Step 1 & 2: Ground Extraction Method Comparison ===")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    v1_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    gpkg_path = "data/02_interim/akilli_sehir_fua.gpkg"
    
    windows = {
        "Alsancak (Dense core)": (512350, 4255850, 512850, 4256350, True, "LCZ 1/2"),
        "Bornova (Mid-rise plain)": (517500, 4257500, 519000, 4259000, False, "LCZ 2/5"),
        "Karabaglar (Hillside)": (508500, 4248000, 510000, 4249500, False, "LCZ 3"),
        "Cigli Plain (Coastal flat)": (502000, 4263000, 504000, 4265000, True, "LCZ 8/open"),
        "Karsiyaka Shore (Coastal)": (508000, 4258000, 5095000, 4259500, True, "LCZ 2/shore"),
        "Bayrakli Shore (Coastal)": (513000, 4257000, 514500, 4258500, True, "LCZ 1/shore"),
        "Cigli AOSB (Industrial)": (502375, 4262625, 503375, 4263625, True, "LCZ 8"),
        "Buca (Compact low-rise)": (514375, 4249125, 515375, 4250125, False, "LCZ 3"),
    }
    
    results = []
    
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(fab_path) as src_fab, rasterio.open(v1_path) as src_v1:
        for name, (minx, miny, maxx, maxy, is_flat_coastal, lcz_tag) in windows.items():
            bbox = (minx, miny, maxx, maxy)
            win_dsm = from_bounds(*bbox, transform=src_dsm.transform).round_offsets().round_lengths()
            trans_dsm = rasterio.windows.transform(win_dsm, src_dsm.transform)
            
            dsm = src_dsm.read(1, window=win_dsm).astype(np.float32)
            v1_dtm = src_v1.read(1, window=win_dsm).astype(np.float32)
            valid_land = (dsm > -100)
            
            # FABDEM
            win_fab = from_bounds(*bbox, transform=src_fab.transform).round_offsets().round_lengths()
            fab = src_fab.read(1, window=win_fab).astype(np.float32)
            fab_slp = calc_slope(fab, cellsize=30.0)
            fab_slp_5m = zoom(fab_slp, (dsm.shape[0] / fab_slp.shape[0], dsm.shape[1] / fab_slp.shape[1]), order=1)
            
            # Buildings
            bld = pyogrio.read_dataframe(gpkg_path, layer="buildings_fua", bbox=bbox)
            valid_bld = [g.buffer(2.5) for g in bld.geometry if g is not None and not g.is_empty]
            bld_mask = np.zeros(dsm.shape, dtype=np.uint8)
            if len(valid_bld) > 0:
                bld_mask = features.rasterize(((g, 1) for g in valid_bld), out_shape=dsm.shape, transform=trans_dsm, dtype=np.uint8, fill=0)
            non_bld_mask = (bld_mask == 0) & valid_land
            
            # Roads: extract wide roads (BULVAR, OTOYOL, or >= 3 lanes)
            roads = pyogrio.read_dataframe(gpkg_path, layer="roads_fua", bbox=bbox)
            wide_geoms = []
            for _, r in roads.iterrows():
                yt = str(r.get("YOLTIP", "")).upper()
                lanes = r.get("SERITSAYIS", 0)
                try:
                    lanes = int(lanes)
                except:
                    lanes = 0
                if yt in ["BULVAR", "OTOYOL"] or lanes >= 3:
                    if r.geometry is not None and not r.geometry.is_empty:
                        wide_geoms.append(r.geometry)
            wide_road_mask = np.zeros(dsm.shape, dtype=np.uint8)
            if len(wide_geoms) > 0:
                wide_road_mask = features.rasterize(((g, 1) for g in wide_geoms), out_shape=dsm.shape, transform=trans_dsm, dtype=np.uint8, fill=0)
            wide_road_mask = (wide_road_mask == 1) & valid_land
            
            # Execute methods
            dtm_a = run_method_a_pmf(dsm, non_bld_mask, cellsize=5.0, max_win_m=75.0, s=0.08, dh0=0.3, dhmax=2.5, sigma_m=15.0)
            dtm_b = run_method_b_low_percentile(dsm, non_bld_mask, cellsize=5.0, win_m=75.0, pct=10, sigma_m=15.0)
            dtm_c = run_method_c_wide_roads(dsm, non_bld_mask, wide_road_mask, cellsize=5.0, win_m=75.0, sigma_m=15.0)
            
            method_dict = {
                "WP1.1 (v1)": v1_dtm,
                "Method (a) PMF": dtm_a,
                "Method (b) Low-Pct": dtm_b,
                "Method (c) Wide-Road": dtm_c,
            }
            
            for mname, m_dtm in method_dict.items():
                # Define evaluation mask: for coastal flat plains, evaluate on FABDEM slope < 2 deg if available
                if is_flat_coastal:
                    eval_mask = valid_land & (fab_slp_5m < 2.0)
                    if np.sum(eval_mask) < 100:
                        eval_mask = valid_land
                else:
                    eval_mask = valid_land
                    
                slp = calc_slope(m_dtm)
                dev = local_dev_75m(m_dtm, valid_land)
                
                slp_med = float(np.median(slp[eval_mask]))
                slp_p95 = float(np.percentile(slp[eval_mask], 95))
                dev_p95 = float(np.percentile(dev[eval_mask], 95))
                
                # FABDEM comparison
                scale_y = fab.shape[0] / m_dtm.shape[0]
                scale_x = fab.shape[1] / m_dtm.shape[1]
                dtm_at_fab = zoom(m_dtm, (scale_y, scale_x), order=1)
                fab_eval = (fab > -100) & (dtm_at_fab > -100)
                diff = dtm_at_fab[fab_eval] - fab[fab_eval]
                bias_fab = float(np.mean(diff))
                rmse_fab = float(np.sqrt(np.mean(diff**2)))
                
                # Capping compliance
                cap_ok = float(np.mean(m_dtm[valid_land] <= dsm[valid_land] + 1e-4) * 100.0)
                
                # Acceptance
                if is_flat_coastal:
                    accept_str = "PASS" if (slp_med < 2.0 and dev_p95 < 1.0) else "FAIL"
                else:
                    accept_str = "N/A (hilly)"
                    
                results.append({
                    "window": name,
                    "is_flat": is_flat_coastal,
                    "lcz": lcz_tag,
                    "method": mname,
                    "elev_med": float(np.median(m_dtm[valid_land])),
                    "elev_min": float(np.min(m_dtm[valid_land])),
                    "elev_max": float(np.max(m_dtm[valid_land])),
                    "slope_med": slp_med,
                    "slope_p95": slp_p95,
                    "dev_p95": dev_p95,
                    "bias_fab": bias_fab,
                    "rmse_fab": rmse_fab,
                    "cap_pct": cap_ok,
                    "acceptance": accept_str
                })
                
    # Summary Table
    print("\n" + "="*120)
    print("### GROUND EXTRACTION METHOD COMPARISON TABLE ACROSS EVALUATION WINDOWS")
    print("="*120)
    headers = [
        "Window", "LCZ", "Method", "Elev Med (m)", "Slope Med", "Slope p95", "Dev p95 (m)",
        "FABDEM Bias", "FABDEM RMSE", "DTM <= DSM", "Acceptance"
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"]*len(headers)) + "|")
    for r in results:
        line = [
            r["window"],
            r["lcz"],
            r["method"],
            f"{r['elev_med']:.1f} m",
            f"{r['slope_med']:.2f}°",
            f"{r['slope_p95']:.2f}°",
            f"{r['dev_p95']:.2f} m",
            f"{r['bias_fab']:+.2f} m",
            f"{r['rmse_fab']:.2f} m",
            f"{r['cap_pct']:.1f}%",
            f"**{r['acceptance']}**" if "PASS" in r['acceptance'] else r['acceptance']
        ]
        print("| " + " | ".join(line) + " |")
        
    print(f"\nStep 1 & 2 comparison completed in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
