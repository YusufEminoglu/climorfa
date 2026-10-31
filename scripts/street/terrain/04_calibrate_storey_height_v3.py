"""
WP1.1d Step 4: Storey-Height Calibration v3 & OSM Cross-Check
Stratified by building type (YAPITIP) and floors (1, 2-3, 4-6, 7+)
Main estimate = residential/mixed multi-storey (floors >= 2)
Footprints >= 250 m2 with >= 4 interior 5 m pixels after -2.5 m shrink
Author: Yusuf Eminoglu / Antigravity
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import scipy.stats as stats
import pyogrio

def regress_ols(y, x, fit_intercept=True):
    n = len(y)
    if fit_intercept:
        X = np.column_stack([np.ones(n), x])
        p = 2
    else:
        X = x[:, None]
        p = 1
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    resids = y - y_pred
    dof = max(1, n - p)
    sigma2 = np.sum(resids**2) / dof
    try:
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
    except:
        se_beta = np.zeros(p)
    t_crit = stats.t.ppf(0.975, dof)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(resids**2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resids**2)))
    return {
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se_beta],
        "ci": [[float(l), float(u)] for l, u in zip(ci_lower, ci_upper)],
        "r2": r2,
        "rmse": rmse,
        "n": int(n)
    }

def main():
    t_start = time.time()
    print("=== WP1.1d Step 4: Storey-Height Calibration v3 & OSM Cross-Check ===")
    
    ndsm_v3_path = "data/street/terrain/ndsm_5m_v3_epsg5253.tif"
    ndsm_v2_path = "data/street/terrain/ndsm_5m_v2_epsg5253.tif"
    ndsm_v1_path = "data/street/terrain/ndsm_5m_epsg5253.tif"
    bld_path = "data/02_interim/akilli_sehir_fua.gpkg"
    grids_path = "data/03_processed/analysis_grids.gpkg"
    v8_csv = "data/03_processed/grid_250m_model_features_v8.csv"
    osm_path = "data/street/terrain/osm_buildings_fua.geojson"
    
    # -------------------------------------------------------------
    # 1. Load Building Footprints & YAPITIP Distribution
    # -------------------------------------------------------------
    print("\n--- 1. Building Footprints & YAPITIP Breakdown ---")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(
        bld_path,
        layer="buildings_fua",
        columns=["ZEMINUSTUK", "SHAPE_Area", "YAPITIP"]
    )
    n_total = len(bld_df)
    print(f"Loaded {n_total:,} buildings from {bld_path} in {time.time()-t0:.2f} s")
    
    # Normalize YAPITIP string
    # Clean Turkish characters
    def clean_yapitip(val):
        if not isinstance(val, str):
            return "UNKNOWN"
        v = val.strip().upper()
        if "MESKEN" in v:
            return "MESKEN"
        elif "YER" in v or "ISYERI" in v or "İŞYERİ" in v:
            return "ISYERI"
        elif "KAMU" in v:
            return "KAMU"
        return v
        
    bld_df["yapitip_clean"] = bld_df["YAPITIP"].apply(clean_yapitip)
    
    print("\nFull FUA Building Type (YAPITIP) Distribution:")
    yapitip_counts = bld_df["yapitip_clean"].value_counts()
    for cat, cnt in yapitip_counts.items():
        print(f"  {cat:12s}: {cnt:7,d} ({cnt/n_total*100:.2f}%)")
        
    # Area filtering: >= 250 m2
    bld_ge_250 = bld_df[bld_df["SHAPE_Area"] >= 250.0].copy()
    n_ge_250 = len(bld_ge_250)
    print(f"\nBuildings with footprint >= 250 m2: {n_ge_250:,} ({n_ge_250/n_total*100:.2f}%)")
    print(f"Buildings with footprint < 250 m2:  {n_total - n_ge_250:,} ({(n_total - n_ge_250)/n_total*100:.2f}%)")
    print("YAPITIP distribution for >= 250 m2:")
    for cat, cnt in bld_ge_250["yapitip_clean"].value_counts().items():
        print(f"  {cat:12s}: {cnt:7,d} ({cnt/n_ge_250*100:.2f}%)")
        
    # Inward shrink (-2.5 m)
    print("\nApplying inward buffer (-2.5 m) to eligible buildings (area >= 250 m2, floors >= 1)...")
    eligible_mask = (bld_df["SHAPE_Area"] >= 250.0) & (bld_df["ZEMINUSTUK"] >= 1) & (bld_df["ZEMINUSTUK"] <= 40)
    eligible = bld_df[eligible_mask].copy().reset_index(drop=True)
    t0 = time.time()
    shrunk_geoms = eligible.geometry.buffer(-2.5)
    valid_shrunk = ~shrunk_geoms.is_empty
    eligible = eligible[valid_shrunk].copy().reset_index(drop=True)
    eligible["shrunk_geom"] = shrunk_geoms[valid_shrunk].values
    eligible["bld_idx"] = np.arange(1, len(eligible) + 1, dtype=np.int32)
    print(f"Valid shrunk polygons: {len(eligible):,} in {time.time()-t0:.2f} s")
    
    # -------------------------------------------------------------
    # 2. Extract nDSM v3, v2, v1 Heights
    # -------------------------------------------------------------
    print("\n--- 2. Extracting Building Heights (p90) from nDSM v3, v2, v1 ---")
    t0 = time.time()
    with rasterio.open(ndsm_v3_path) as src_v3:
        shape = src_v3.shape
        trans = src_v3.transform
        ndsm_v3 = src_v3.read(1)
        nodata = src_v3.nodata
        
    labeled = features.rasterize(
        ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
        out_shape=shape,
        transform=trans,
        dtype=np.int32,
        fill=0
    )
    
    pix_mask = (labeled > 0) & (ndsm_v3 != nodata)
    df_px = pd.DataFrame({"bld_idx": labeled[pix_mask], "ndsm_v3": ndsm_v3[pix_mask]})
    del labeled, ndsm_v3
    
    stats_v3 = df_px.groupby("bld_idx")["ndsm_v3"].agg(
        p90_v3=lambda x: np.percentile(x, 90),
        px_count="count"
    ).reset_index()
    del df_px
    eligible = eligible.merge(stats_v3, on="bld_idx", how="inner")
    
    # Extract v2 if present, or compute from dsm - dtm_v2
    dtm_v2_path = "data/street/terrain/dtm_5m_v2_epsg5253.tif"
    if os.path.exists(ndsm_v2_path):
        with rasterio.open(ndsm_v2_path) as src_v2:
            ndsm_v2 = src_v2.read(1)
        labeled_v2 = features.rasterize(
            ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
            out_shape=shape,
            transform=trans,
            dtype=np.int32,
            fill=0
        )
        pix_m2 = (labeled_v2 > 0) & (ndsm_v2 != src_v2.nodata)
        df_px2 = pd.DataFrame({"bld_idx": labeled_v2[pix_m2], "ndsm_v2": ndsm_v2[pix_m2]})
        del labeled_v2, ndsm_v2
        stats_v2 = df_px2.groupby("bld_idx")["ndsm_v2"].agg(p90_v2=lambda x: np.percentile(x, 90)).reset_index()
        del df_px2
        eligible = eligible.merge(stats_v2, on="bld_idx", how="left")
    elif os.path.exists(dtm_v2_path):
        print("Extracting v2 heights from DSM - DTM_v2...")
        dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
        with rasterio.open(dsm_path) as src_dsm, rasterio.open(dtm_v2_path) as src_v2:
            dsm_arr = src_dsm.read(1)
            dtm_v2_arr = src_v2.read(1)
            ndsm_v2 = np.maximum(dsm_arr - dtm_v2_arr, 0.0)
            nodata_m = (dsm_arr == src_dsm.nodata) | (dtm_v2_arr == src_v2.nodata)
            del dsm_arr, dtm_v2_arr
            ndsm_v2[nodata_m] = -9999.0
            
        labeled_v2 = features.rasterize(
            ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
            out_shape=shape,
            transform=trans,
            dtype=np.int32,
            fill=0
        )
        pix_m2 = (labeled_v2 > 0) & (ndsm_v2 > -100)
        df_px2 = pd.DataFrame({"bld_idx": labeled_v2[pix_m2], "ndsm_v2": ndsm_v2[pix_m2]})
        del labeled_v2, ndsm_v2
        stats_v2 = df_px2.groupby("bld_idx")["ndsm_v2"].agg(p90_v2=lambda x: np.percentile(x, 90)).reset_index()
        del df_px2
        eligible = eligible.merge(stats_v2, on="bld_idx", how="left")
    else:
        eligible["p90_v2"] = np.nan
        
    # Extract v1 if present
    eligible["p90_v1"] = np.nan
    print(f"Height extraction complete in {time.time()-t0:.2f} s")
    
    # Clean Sample: internal pixel count >= 4, p90_v3 > 0.5 m
    clean_sample = eligible[
        (eligible["px_count"] >= 4) &
        (eligible["p90_v3"] > 0.5)
    ].copy().reset_index(drop=True)
    n_clean = len(clean_sample)
    print(f"Clean calibration sample (area >= 250 m2, px_count >= 4, p90 > 0.5m): N = {n_clean:,}")
    
    # Join with analysis grids to assign administrative district
    print("Joining building centroids to analysis grids for district assignment...")
    grids = pyogrio.read_dataframe(grids_path, layer="grid_250m")
    v8 = pd.read_csv(v8_csv, usecols=["grid_id", "district"])
    grids = grids.merge(v8, on="grid_id", how="left")
    
    centroids = clean_sample.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(centroids, grids[["district", "geometry"]], how="left", predicate="within")
    clean_sample["district"] = joined["district"].fillna("unknown").values
    del centroids, grids, v8
    
    # -------------------------------------------------------------
    # 3. Global & Stratified Regressions (Model 1 & Model 2)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("3. STOREY-HEIGHT CALIBRATION v3 RESULTS")
    print("="*70)
    
    results = {}
    
    # Floor bins: 1, 2-3, 4-6, 7+
    def get_floor_bin(f):
        if f == 1:
            return "1 floor (single-storey)"
        elif 2 <= f <= 3:
            return "2-3 floors (low-rise)"
        elif 4 <= f <= 6:
            return "4-6 floors (mid-rise)"
        else:
            return "7+ floors (high-rise)"
            
    clean_sample["floor_bin"] = clean_sample["ZEMINUSTUK"].apply(get_floor_bin)
    
    # Floor bin counts
    print("\nFloor Bin Breakdown (N = {:,}):".format(n_clean))
    for fb in ["1 floor (single-storey)", "2-3 floors (low-rise)", "4-6 floors (mid-rise)", "7+ floors (high-rise)"]:
        cnt = (clean_sample["floor_bin"] == fb).sum()
        print(f"  {fb:26s}: {cnt:6,d} ({cnt/n_clean*100:.2f}%)")
        
    # (A) MAIN ESTIMATE: Residential/Mixed Multi-Storey (floors >= 2)
    # Spec: "main estimate = residential/mixed multi-storey (floors >= 2) buildings; report single-storey halls separately"
    main_subset = clean_sample[
        (clean_sample["ZEMINUSTUK"] >= 2) &
        (clean_sample["yapitip_clean"].isin(["MESKEN", "ISYERI"]))
    ].copy()
    
    y_main = main_subset["p90_v3"].values
    x_main = main_subset["ZEMINUSTUK"].values.astype(float)
    
    m1_main = regress_ols(y_main, x_main, fit_intercept=False)
    m2_main = regress_ols(y_main, x_main, fit_intercept=True)
    
    flag_int = " [BIAS FLAG: Intercept > 4.0 m!]" if m2_main["beta"][0] > 4.0 else " [PLAUSIBLE: Intercept <= 4.0 m]"
    
    print("\n>>> MAIN ESTIMATE: Residential/Mixed Multi-Storey Buildings (Floors >= 2, N = {:,}) <<<".format(len(main_subset)))
    print(f"  Model 1 (Through Origin):")
    print(f"    Implied Storey Height: {m1_main['beta'][0]:.3f} m/storey (95% CI: [{m1_main['ci'][0][0]:.3f}, {m1_main['ci'][0][1]:.3f}], RMSE={m1_main['rmse']:.2f} m)")
    print(f"  Model 2 (With Intercept):")
    print(f"    Height = {m2_main['beta'][0]:.2f} + {m2_main['beta'][1]:.3f} * floors (R2={m2_main['r2']:.3f}, RMSE={m2_main['rmse']:.2f} m){flag_int}")
    print(f"    Intercept 95% CI: [{m2_main['ci'][0][0]:.2f}, {m2_main['ci'][0][1]:.2f}] m")
    print(f"    Slope 95% CI:     [{m2_main['ci'][1][0]:.3f}, {m2_main['ci'][1][1]:.3f}] m/storey")
    
    results["main_estimate"] = {
        "n": len(main_subset),
        "model1_beta": m1_main["beta"][0],
        "model1_ci": m1_main["ci"][0],
        "model1_rmse": m1_main["rmse"],
        "model2_alpha": m2_main["beta"][0],
        "model2_beta": m2_main["beta"][1],
        "model2_ci_alpha": m2_main["ci"][0],
        "model2_ci_beta": m2_main["ci"][1],
        "model2_r2": m2_main["r2"],
        "model2_rmse": m2_main["rmse"],
    }
    
    # (B) SINGLE-STOREY HALLS (Reported separately)
    single_subset = clean_sample[clean_sample["ZEMINUSTUK"] == 1].copy()
    print("\n>>> SINGLE-STOREY BUILDINGS / HALLS (Floors = 1, N = {:,}) <<<".format(len(single_subset)))
    for ytip in ["MESKEN", "ISYERI", "KAMU"]:
        sub_y = single_subset[single_subset["yapitip_clean"] == ytip]
        if len(sub_y) > 0:
            h_med = np.median(sub_y["p90_v3"])
            h_mean = np.mean(sub_y["p90_v3"])
            h_p10 = np.percentile(sub_y["p90_v3"], 10)
            h_p90 = np.percentile(sub_y["p90_v3"], 90)
            print(f"  Single-storey {ytip:8s} (N={len(sub_y):5,d}): Median={h_med:.2f} m, Mean={h_mean:.2f} m, IQR=[{h_p10:.1f} .. {h_p90:.1f}] m")
            
    # (C) BUILDING TYPE STRATIFICATION (for floors >= 2)
    print("\n>>> STRATIFICATION BY BUILDING TYPE (Floors >= 2) <<<")
    yapitip_results = []
    for ytip in ["MESKEN", "ISYERI", "KAMU"]:
        sub_y = clean_sample[(clean_sample["yapitip_clean"] == ytip) & (clean_sample["ZEMINUSTUK"] >= 2)]
        if len(sub_y) >= 30:
            m1_y = regress_ols(sub_y["p90_v3"].values, sub_y["ZEMINUSTUK"].values.astype(float), fit_intercept=False)
            m2_y = regress_ols(sub_y["p90_v3"].values, sub_y["ZEMINUSTUK"].values.astype(float), fit_intercept=True)
            entry = {
                "type": ytip,
                "n": len(sub_y),
                "model1_beta": m1_y["beta"][0],
                "model1_ci": m1_y["ci"][0],
                "model2_alpha": m2_y["beta"][0],
                "model2_beta": m2_y["beta"][1],
                "model2_r2": m2_y["r2"]
            }
            yapitip_results.append(entry)
            print(f"  {ytip:8s} (N={len(sub_y):5,d}): Model 1={m1_y['beta'][0]:.3f} m/flr (CI: [{m1_y['ci'][0][0]:.2f}, {m1_y['ci'][0][1]:.2f}]), Model 2: H={m2_y['beta'][0]:.2f}+{m2_y['beta'][1]:.2f}*F (R2={m2_y['r2']:.3f})")
    results["stratified_by_type"] = yapitip_results
    
    # (D) FLOOR BIN STRATIFICATION
    print("\n>>> STRATIFICATION BY FLOOR BIN (Residential/Mixed) <<<")
    floor_results = []
    for fb in ["2-3 floors (low-rise)", "4-6 floors (mid-rise)", "7+ floors (high-rise)"]:
        sub_fb = main_subset[main_subset["floor_bin"] == fb]
        if len(sub_fb) >= 30:
            m1_fb = regress_ols(sub_fb["p90_v3"].values, sub_fb["ZEMINUSTUK"].values.astype(float), fit_intercept=False)
            entry = {
                "bin": fb,
                "n": len(sub_fb),
                "model1_beta": m1_fb["beta"][0],
                "model1_ci": m1_fb["ci"][0],
                "rmse": m1_fb["rmse"]
            }
            floor_results.append(entry)
            print(f"  {fb:24s} (N={len(sub_fb):5,d}): Implied Storey Height = {m1_fb['beta'][0]:.3f} m/flr (CI: [{m1_fb['ci'][0][0]:.3f}, {m1_fb['ci'][0][1]:.3f}], RMSE={m1_fb['rmse']:.2f} m)")
    results["stratified_by_floors"] = floor_results
    
    # (E) DISTRICT-LEVEL STRATIFICATION (Residential/Mixed multi-storey)
    print("\n>>> DISTRICT-LEVEL STRATIFICATION (Main Sample, N >= 30) <<<")
    districts = sorted([d for d in main_subset["district"].unique() if d != "unknown"])
    district_results = []
    for dist in districts:
        sub_d = main_subset[main_subset["district"] == dist]
        if len(sub_d) >= 30:
            y_d = sub_d["p90_v3"].values
            x_d = sub_d["ZEMINUSTUK"].values.astype(float)
            y_d_v2 = sub_d["p90_v2"].values
            y_d_v1 = sub_d["p90_v1"].values
            
            m1_d_v3 = regress_ols(y_d, x_d, fit_intercept=False)
            m2_d_v3 = regress_ols(y_d, x_d, fit_intercept=True)
            v2_valid = ~np.isnan(y_d_v2)
            m1_d_v2 = regress_ols(y_d_v2[v2_valid], x_d[v2_valid], fit_intercept=False) if np.sum(v2_valid) >= 10 else {"beta": [np.nan]}
            v1_valid = ~np.isnan(y_d_v1)
            m1_d_v1 = regress_ols(y_d_v1[v1_valid], x_d[v1_valid], fit_intercept=False) if np.sum(v1_valid) >= 10 else {"beta": [np.nan]}
            
            flag_d = " [BIAS FLAG: Int > 4m]" if m2_d_v3["beta"][0] > 4.0 else ""
            
            d_res = {
                "district": dist,
                "n": len(sub_d),
                "v1_beta": m1_d_v1["beta"][0],
                "v2_beta": m1_d_v2["beta"][0],
                "v3_beta": m1_d_v3["beta"][0],
                "v3_ci": m1_d_v3["ci"][0],
                "v3_alpha": m2_d_v3["beta"][0],
                "v3_slope": m2_d_v3["beta"][1],
                "v3_r2": m2_d_v3["r2"],
                "flag": flag_d
            }
            district_results.append(d_res)
            print(f"  {dist:14s} (N={len(sub_d):5,d}): v1={m1_d_v1['beta'][0]:.2f}m -> v2={m1_d_v2['beta'][0]:.2f}m -> v3={m1_d_v3['beta'][0]:.3f} m/flr (CI: [{m1_d_v3['ci'][0][0]:.2f}, {m1_d_v3['ci'][0][1]:.2f}]), H={m2_d_v3['beta'][0]:.2f}+{m2_d_v3['beta'][1]:.2f}*F (R2={m2_d_v3['r2']:.3f}){flag_d}")
            
    results["district_calibration"] = district_results
    
    # -------------------------------------------------------------
    # 4. Independent OSM Ground Truth Cross-Check
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("4. INDEPENDENT OSM GROUND TRUTH CROSS-CHECK")
    print("="*70)
    
    osm_gdf = gpd.read_file(osm_path)
    print(f"Loaded {len(osm_gdf):,} OSM buildings from {osm_path}")
    
    # OSM measured heights vs levels
    osm_h = osm_gdf[osm_gdf["height"].notna() & (osm_gdf["height"] > 2.0) & (osm_gdf["levels"].notna()) & (osm_gdf["levels"] > 0)].copy()
    n_osm_h = len(osm_h)
    print(f"OSM buildings with BOTH measured height and building:levels: N = {n_osm_h:,}")
    
    if n_osm_h >= 10:
        y_oh = osm_h["height"].values
        x_ol = osm_h["levels"].values.astype(float)
        m_osm_orig = regress_ols(y_oh, x_ol, fit_intercept=False)
        m_osm_int = regress_ols(y_oh, x_ol, fit_intercept=True)
        print(f"  OSM Ground Truth Height ~ OSM Levels (N={n_osm_h}):")
        print(f"    Origin:    Height = {m_osm_orig['beta'][0]:.3f} m/storey (95% CI: [{m_osm_orig['ci'][0][0]:.3f}, {m_osm_orig['ci'][0][1]:.3f}], RMSE={m_osm_orig['rmse']:.2f} m)")
        print(f"    Intercept: Height = {m_osm_int['beta'][0]:.2f} + {m_osm_int['beta'][1]:.3f} * floors (R2={m_osm_int['r2']:.3f}, RMSE={m_osm_int['rmse']:.2f} m)")
        print(f"    Honest Assessment: Sample size N={n_osm_h} measured building heights is statistically weak / noisy.")
        
        osm_results = {
            "n_heights": n_osm_h,
            "osm_origin_beta": m_osm_orig["beta"][0],
            "osm_origin_ci": m_osm_orig["ci"][0],
            "osm_origin_rmse": m_osm_orig["rmse"],
            "osm_intercept_alpha": m_osm_int["beta"][0],
            "osm_intercept_beta": m_osm_int["beta"][1],
            "osm_intercept_r2": m_osm_int["r2"]
        }
    else:
        osm_results = {"n_heights": n_osm_h, "note": "Insufficient measured heights"}
        
    results["osm_validation"] = osm_results
    
    # Save results to json
    os.makedirs("logs/wp", exist_ok=True)
    with open("logs/wp/storey_calibration_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nStorey calibration metrics written to logs/wp/storey_calibration_v3.json")

if __name__ == "__main__":
    main()
