"""
Test DTM v3 Pipeline Prototype on Izmir FUA
Checks all acceptance criteria before writing full raster
"""

import os
import sys
import time
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
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

def main():
    print("=== Testing DTM v3 Pipeline ===")
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    gpkg_bld = "data/02_interim/akilli_sehir_fua.gpkg"
    
    t0 = time.time()
    with rasterio.open(dsm_path) as s_dsm:
        profile = s_dsm.profile.copy()
        shape = s_dsm.shape
        trans = s_dsm.transform
        crs = s_dsm.crs
        nodata = float(s_dsm.nodata)
        dsm = s_dsm.read(1).astype(np.float32)
        
    print(f"Loaded DSM {shape} in {time.time()-t0:.2f} s")
    valid = (dsm != nodata) & (dsm >= -1.0)
    
    # 2. Building mask
    print("Rasterizing building mask (+2.5m buffer)...")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(gpkg_bld, layer="buildings_fua", columns=["geom"])
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
    
    # 3. ETH Canopy
    print("Loading ETH Canopy...")
    t0 = time.time()
    with rasterio.open(eth_path) as s_eth:
        canopy_10m = s_eth.read(1)
        canopy_nodata = float(s_eth.nodata)
    canopy_valid = (canopy_10m != canopy_nodata) & (canopy_10m > 0)
    canopy_f32 = np.where(canopy_valid, canopy_10m, 0.0).astype(np.float32)
    del canopy_10m
    canopy_rep = np.repeat(np.repeat(canopy_f32, 2, axis=0), 2, axis=1)
    del canopy_f32
    canopy_5m = np.zeros(shape, dtype=np.float32)
    mh = min(shape[0], canopy_rep.shape[0])
    mw = min(shape[1], canopy_rep.shape[1])
    canopy_5m[:mh, :mw] = canopy_rep[:mh, :mw]
    if shape[0] > mh:
        canopy_5m[mh:, :] = canopy_5m[mh-1:mh, :]
    if shape[1] > mw:
        canopy_5m[:, mw:] = canopy_5m[:, mw-1:mw]
    del canopy_rep
    print(f"Canopy ready in {time.time()-t0:.2f} s")
    
    # 4. Warp FABDEM to 5m
    print("Warping FABDEM to 5m DSM grid...")
    t0 = time.time()
    fab_5m = np.full(shape, -9999.0, dtype=np.float32)
    with rasterio.open(fab_path) as s_fab:
        reproject(
            rasterio.band(s_fab, 1),
            fab_5m,
            src_transform=s_fab.transform,
            src_crs=s_fab.crs,
            dst_transform=trans,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            src_nodata=s_fab.nodata,
            dst_nodata=-9999.0
        )
    print(f"FABDEM warped in {time.time()-t0:.2f} s")
    
    # 5. Regional Offset (DSM - FABDEM on bare cells)
    print("Computing regional offset (DSM - FABDEM)...")
    t0 = time.time()
    # Bare cells for offset estimation: non-building, canopy < 1.5m, valid land
    # To be conservative and avoid any street canyon edges:
    non_bld = (bld_mask == 0) & (canopy_5m < 1.5) & valid & (fab_5m > -100)
    
    # Sample offset at 50 m grid (factor of 10)
    step = 10
    h_c, w_c = (shape[0] + step - 1) // step, (shape[1] + step - 1) // step
    diff_raw = np.where(non_bld, dsm - fab_5m, np.nan)
    
    # Block aggregation to 50m
    pad_y = (step - shape[0] % step) % step
    pad_x = (step - shape[1] % step) % step
    d_pad = np.pad(diff_raw, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    del diff_raw
    d_blks = d_pad.reshape(h_c, step, w_c, step).transpose(0, 2, 1, 3).reshape(h_c, w_c, step*step)
    del d_pad
    offset_coarse = np.nanmedian(d_blks, axis=-1)
    del d_blks
    
    # Fill nodata gaps at 50m level
    val_coarse = ~np.isnan(offset_coarse)
    drv = gdal.GetDriverByName("MEM")
    ds_ot = drv.Create("", w_c, h_c, 1, gdal.GDT_Float32)
    ds_ot.GetRasterBand(1).WriteArray(np.nan_to_num(offset_coarse, nan=2.0))
    ds_om = drv.Create("", w_c, h_c, 1, gdal.GDT_Byte)
    ds_om.GetRasterBand(1).WriteArray(val_coarse.astype(np.uint8))
    del val_coarse
    del offset_coarse
    
    gdal.FillNodata(ds_ot.GetRasterBand(1), ds_om.GetRasterBand(1), maxSearchDist=60, smoothingIterations=1)
    offset_filled = ds_ot.GetRasterBand(1).ReadAsArray()
    
    # Smooth regional offset with Gaussian filter (sigma = 4 cells = 200 m)
    offset_smooth = gaussian_filter(offset_filled, sigma=4.0)
    del offset_filled
    
    # Upsample offset back to 5m
    scale_y = shape[0] / h_c
    scale_x = shape[1] / w_c
    offset_5m = zoom(offset_smooth, (scale_y, scale_x), order=1)[:shape[0], :shape[1]]
    del offset_smooth
    print(f"Regional offset computed in {time.time()-t0:.2f} s. Mean offset: {np.mean(offset_5m[valid]):.2f} m")
    
    # 6. Trend surface T and Residual R
    T = fab_5m + offset_5m
    del offset_5m
    R = dsm - T
    
    # In mountain forest interiors (elev > 50m, canopy > 3m), adjust DSM to remove dense canopy
    forest_canopy = (canopy_5m > 3.0) & (dsm > 50.0)
    R = np.where(forest_canopy, np.maximum(R - canopy_5m, 0.0), R)
    
    # 7. Local low-envelope of R across non-building cells (20 m block min + 80 m p10)
    print("Computing distance to buildings and local low-envelope of R...")
    t0 = time.time()
    dist_bld = distance_transform_edt(bld_mask == 0) * 5.0
    tree_high = (canopy_5m >= 2.0)
    nb_mask = (bld_mask == 0) & (~tree_high) & valid
    del canopy_5m
    del dist_bld
    
    pad_y4 = (4 - shape[0] % 4) % 4
    pad_x4 = (4 - shape[1] % 4) % 4
    R_pad = np.pad(np.where(nb_mask, R, np.nan), ((0, pad_y4), (0, pad_x4)), mode='constant', constant_values=np.nan)
    h4, w4 = R_pad.shape[0] // 4, R_pad.shape[1] // 4
    R_blks = R_pad.reshape(h4, 4, w4, 4).transpose(0, 2, 1, 3).reshape(h4, w4, 16)
    del R_pad
    c_min = np.nanmin(R_blks, axis=-1)
    del R_blks
    
    # Fill gaps at 20m
    val_c = ~np.isnan(c_min)
    ds_rt = drv.Create("", w4, h4, 1, gdal.GDT_Float32)
    ds_rt.GetRasterBand(1).WriteArray(np.nan_to_num(c_min, nan=0.0))
    ds_rm = drv.Create("", w4, h4, 1, gdal.GDT_Byte)
    ds_rm.GetRasterBand(1).WriteArray(val_c.astype(np.uint8))
    del val_c
    del c_min
    
    gdal.FillNodata(ds_rt.GetRasterBand(1), ds_rm.GetRasterBand(1), maxSearchDist=60, smoothingIterations=0)
    c_filled = ds_rt.GetRasterBand(1).ReadAsArray()
    c_p10 = percentile_filter(c_filled, percentile=10, size=4, mode='nearest')
    del c_filled
    
    low_env = zoom(c_p10, (shape[0]/h4, shape[1]/w4), order=1)[:shape[0], :shape[1]]
    del c_p10
    print(f"Low-envelope computed in {time.time()-t0:.2f} s")
    
    # 8. Ground candidate rejection:
    # Ground candidate threshold: R must be <= low_env + 2.2 m
    # This keeps true bare ground (R ~ 0) while strictly rejecting street trees (R >= 4m) and roof overhangs
    ground_cand = nb_mask & (R <= low_env + 2.2)
    del nb_mask
    del low_env
    n_gc = int(np.sum(ground_cand))
    print(f"Ground candidates: {n_gc:,} ({n_gc/np.sum(valid)*100:.2f}% of land)")
    
    # 9. Interpolate R over excluded cells
    print("Interpolating residual R over excluded cells via GDAL FillNodata...")
    t0 = time.time()
    ds_it = drv.Create("", shape[1], shape[0], 1, gdal.GDT_Float32)
    ds_it.GetRasterBand(1).WriteArray(np.where(ground_cand, R, 0.0).astype(np.float32))
    del R
    
    # Don't search into sea
    fill_mask = np.where(valid, ground_cand.astype(np.uint8), 1)
    del ground_cand
    ds_im = drv.Create("", shape[1], shape[0], 1, gdal.GDT_Byte)
    ds_im.GetRasterBand(1).WriteArray(fill_mask)
    del fill_mask
    
    gdal.FillNodata(ds_it.GetRasterBand(1), ds_im.GetRasterBand(1), maxSearchDist=250, smoothingIterations=1)
    R_interp = ds_it.GetRasterBand(1).ReadAsArray()
    print(f"FillNodata completed in {time.time()-t0:.2f} s")
    
    # 10. Recombine: DTM = T + R_interp
    print("Recombining DTM = T + R_interp...")
    dtm_recon = T + R_interp
    del T
    del R_interp
    
    # Smooth ground surface for built-up and flat coastal plains (sigma <= 10m spec)
    dtm_smooth = uniform_filter(gaussian_filter(dtm_recon, sigma=1.8), size=3)
    
    # Slope of regional terrain
    dx_f = sobel(fab_5m, axis=1) / (8.0 * 5.0)
    dy_f = sobel(fab_5m, axis=0) / (8.0 * 5.0)
    fab_slp = np.degrees(np.arctan(np.sqrt(dx_f**2 + dy_f**2)))
    del dx_f
    del dy_f
    
    # On flat coastal plains (< 2.0 deg) and inside built-up areas:
    # use dtm_smooth to eliminate tractor furrows and building eaves
    # On hillsides and mountains (>= 2.0 deg):
    # keep bare open ground (canopy < 2m) as raw DSM (Step 1d: DSM is ground there)
    # under buildings and trees, use dtm_smooth
    # Built-up mask
    bld_dilated = (gaussian_filter(bld_mask.astype(np.float32), sigma=2.0) > 0.02)
    is_hill_bare = (fab_slp >= 2.5) & (~bld_dilated) & (~tree_high) & valid
    del fab_slp
    del bld_dilated
    del tree_high
    del bld_mask
    
    dtm_v3 = np.where(is_hill_bare, dsm, dtm_smooth)
    del is_hill_bare
    del dtm_smooth
    del dtm_recon
    
    # Strict physical capping: DTM <= DSM everywhere (Step 1f)
    dtm_v3 = np.minimum(dtm_v3, dsm)
    dtm_v3[~valid] = nodata
    
    print("DTM v3 synthesized successfully. Running Acceptance Checks...")
    
    # ------------------------------------------------------------------
    # Acceptance Checks
    # ------------------------------------------------------------------
    # A. 5 Coastal Plain Windows
    print("\n--- Acceptance A: Flat Cores (5 Coastal Plain Windows) ---")
    windows = {
        "Alsancak (Dense core)": (512350, 4255850, 512850, 4256350),
        "Cigli Plain (Coastal flat)": (502000, 4263000, 504000, 4265000),
        "Karsiyaka Shore plain": (508000, 4258000, 5095000, 4259500),
        "Bayrakli Shore plain": (513000, 4257000, 514500, 4258500),
        "Cigli AOSB (Industrial)": (502375, 4262625, 503375, 4263625),
    }
    
    def calc_slp(arr, v_m):
        filled = arr.copy()
        if not np.all(v_m):
            idx = distance_transform_edt(~v_m, return_distances=False, return_indices=True)
            filled = filled[tuple(idx)]
        dx = sobel(filled, axis=1) / (8.0 * 5.0)
        dy = sobel(filled, axis=0) / (8.0 * 5.0)
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    def calc_dev(arr, v_m):
        ksize = 15
        filled = arr.copy()
        if not np.all(v_m):
            idx = distance_transform_edt(~v_m, return_distances=False, return_indices=True)
            filled = filled[tuple(idx)]
        return np.abs(arr - uniform_filter(filled, size=ksize))

    for wname, bbox in windows.items():
        w_5m = from_bounds(*bbox, transform=trans).round_offsets().round_lengths()
        d_sub = dtm_v3[w_5m.row_off:w_5m.row_off+w_5m.height, w_5m.col_off:w_5m.col_off+w_5m.width]
        v_sub = valid[w_5m.row_off:w_5m.row_off+w_5m.height, w_5m.col_off:w_5m.col_off+w_5m.width]
        
        with rasterio.open(fab_path) as src_fab_w:
            w_fab = from_bounds(*bbox, transform=src_fab_w.transform).round_offsets().round_lengths()
            fab_w = src_fab_w.read(1, window=w_fab).astype(np.float32)
            dx_f = sobel(fab_w, axis=1) / (8.0 * 30.0)
            dy_f = sobel(fab_w, axis=0) / (8.0 * 30.0)
            slp_fab_30m = np.degrees(np.arctan(np.sqrt(dx_f**2 + dy_f**2)))
            slp_f = zoom(slp_fab_30m, (d_sub.shape[0] / slp_fab_30m.shape[0], d_sub.shape[1] / slp_fab_30m.shape[1]), order=1)[:d_sub.shape[0], :d_sub.shape[1]]
        
        eval_m = v_sub & (slp_f < 2.0)
        if np.sum(eval_m) < 100:
            eval_m = v_sub
            
        slp_v3 = calc_slp(d_sub, v_sub)
        dev_v3 = calc_dev(d_sub, v_sub)
        
        s_med = np.median(slp_v3[eval_m])
        d_p95 = np.percentile(dev_v3[eval_m], 95)
        st = "PASS" if (s_med < 2.0 and d_p95 < 1.0) else "FAIL"
        print(f"  {wname:28s}: slope={s_med:.2f}° (req<2.0), dev p95={d_p95:.2f}m (req<1.0) => {st}")
        
    # D. Built-up Hillside Windows
    print("\n--- Acceptance D: Built-up Hillside Windows ---")
    hill_windows = {
        "Karabaglar": (508000, 4247000, 509000, 4248000),
        "Buca": (511000, 4249000, 512000, 4250000),
        "Bayrakli hill": (514000, 4260000, 515000, 4261000),
        "Narlidere": (498000, 4246000, 499000, 4247000),
    }
    for hname, bbox in hill_windows.items():
        w_5m = from_bounds(*bbox, transform=trans).round_offsets().round_lengths()
        d_sub = dtm_v3[w_5m.row_off:w_5m.row_off+w_5m.height, w_5m.col_off:w_5m.col_off+w_5m.width]
        f_sub = fab_5m[w_5m.row_off:w_5m.row_off+w_5m.height, w_5m.col_off:w_5m.col_off+w_5m.width]
        v_sub = valid[w_5m.row_off:w_5m.row_off+w_5m.height, w_5m.col_off:w_5m.col_off+w_5m.width] & (f_sub > -100)
        
        bias = np.mean(d_sub[v_sub] - f_sub[v_sub])
        st = "PASS" if (-0.7 <= bias <= 4.7) else "FAIL"
        print(f"  {hname:28s}: v3 - FAB bias={bias:+.2f}m (req -0.7..+4.7m) => {st}")

    # B. Bare open cells per slope class
    print("\n--- Acceptance B: Bare Open Cells Departure ---")
    with rasterio.open(eth_path) as s_eth, rasterio.open("data/street/terrain/dtm_5m_epsg5253.tif") as s_v1:
        v1_arr = s_v1.read(1)
    
    # 30m FABDEM slope
    with rasterio.open(fab_path) as s_fab:
        fab_30m = s_fab.read(1)
        trans_30m = s_fab.transform
        crs_30m = s_fab.crs
        nodata_fab = float(s_fab.nodata)
        
    dx = sobel(fab_30m, axis=1) / (8.0 * 30.0)
    dy = sobel(fab_30m, axis=0) / (8.0 * 30.0)
    slope_30m = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    dsm_30m = np.full(fab_30m.shape, -9999.0, dtype=np.float32)
    v1_30m = np.full(fab_30m.shape, -9999.0, dtype=np.float32)
    v3_30m = np.full(fab_30m.shape, -9999.0, dtype=np.float32)
    eth_30m = np.full(fab_30m.shape, -9999.0, dtype=np.float32)
    
    reproject(dsm, dsm_30m, src_transform=trans, src_crs=crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=nodata, dst_nodata=-9999.0)
    reproject(v1_arr, v1_30m, src_transform=trans, src_crs=crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=nodata, dst_nodata=-9999.0)
    del v1_arr
    reproject(dtm_v3, v3_30m, src_transform=trans, src_crs=crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=nodata, dst_nodata=-9999.0)
    
    with rasterio.open(eth_path) as s_eth:
        reproject(rasterio.band(s_eth, 1), eth_30m, src_transform=s_eth.transform, src_crs=s_eth.crs, dst_transform=trans_30m, dst_crs=crs_30m, resampling=Resampling.average, src_nodata=s_eth.nodata, dst_nodata=-9999.0)

    val_30 = (fab_30m != nodata_fab) & (fab_30m >= 0) & (dsm_30m > -100) & (v1_30m > -100) & (v3_30m > -100)
    bare_30 = val_30 & (eth_30m < 1.0) & (np.abs(dsm_30m - v1_30m) < 1.0)
    
    bins = [("0-2", 0, 2), ("2-5", 2, 5), ("5-15", 5, 15), (">15", 15, 90)]
    for name, lo, hi in bins:
        m = bare_30 & (slope_30m >= lo) & (slope_30m < hi)
        n = np.sum(m)
        bias_dsm = np.mean(dsm_30m[m] - fab_30m[m])
        bias_v3 = np.mean(v3_30m[m] - fab_30m[m])
        rmse_v3 = np.sqrt(np.mean((v3_30m[m] - fab_30m[m])**2))
        departure = np.abs(bias_v3 - bias_dsm)
        st = "PASS" if departure <= 1.0 else "FAIL"
        print(f"  Slope {name:6s} (N={n:,}): rawDSM-FAB={bias_dsm:+.2f}m | v3-FAB={bias_v3:+.2f}m (rmse={rmse_v3:.2f}m) | |diff|={departure:.2f}m (req<=1.0m) => {st}")

if __name__ == "__main__":
    main()
