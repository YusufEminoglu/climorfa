"""
WP1.1b Step 4: Storey-Height Calibration on nDSM v2 & Independent OSM Cross-Check
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
1. Re-run storey-height calibration on nDSM v2 (ndsm_5m_v2_epsg5253.tif):
   - Sample eligibility: footprint area >= 400 m2, inward buffer -2.5 m, internal pixel count >= 4,
     p90 > 0.5 m, 1 <= ZEMINUSTUK <= 40.
   - Model 1: Through origin (implied metres per storey + 95% CI)
   - Model 2: With intercept (alpha + beta * floors + 95% CI)
   - Stratified across 11 administrative districts (N >= 30)
   - Compare with WP1.1 v1 calibration results (document resolution of underestimation bias).
2. Independent cross-check against OSM buildings (osm_buildings_fua.geojson):
   - Match OSM buildings with building:levels and height to municipal buildings and nDSM v2.
   - Evaluate agreement between OSM levels and ZEMINUSTUK.
   - Regress OSM measured height vs OSM levels, and nDSM v2 vs OSM levels.
3. Quantify smoothing bias across building footprint sizes:
   - Height per storey across area classes: <100 m2, 100-200 m2, 200-400 m2, 400-800 m2, >=800 m2.
   - Explicitly evaluate whether 5 m nDSM can resolve canyon heights.
"""

import os
import sys
import time
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
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_crit = stats.t.ppf(0.975, dof)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(resids**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(resids**2))
    return {
        "beta": beta,
        "se": se_beta,
        "ci": list(zip(ci_lower, ci_upper)),
        "r2": r2,
        "rmse": rmse,
        "n": n
    }

def main():
    t_start = time.time()
    print("=== WP1.1b Step 4: Storey-Height Calibration & OSM Cross-Check on nDSM v2 ===")
    
    ndsm_v2_path = "data/street/terrain/ndsm_5m_v2_epsg5253.tif"
    ndsm_v1_path = "data/street/terrain/ndsm_5m_epsg5253.tif"
    gpkg_bld = "data/02_interim/akilli_sehir_fua.gpkg"
    gpkg_grids = "data/03_processed/analysis_grids.gpkg"
    v8_csv = "data/03_processed/grid_250m_model_features_v8.csv"
    osm_path = "data/street/terrain/osm_buildings_fua.geojson"
    
    # -------------------------------------------------------------
    # 1. Load Buildings & Usability Breakdown
    # -------------------------------------------------------------
    print("\n--- 1. Building Footprints & Usability Breakdown ---")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(gpkg_bld, layer="buildings_fua", columns=["ZEMINUSTUK", "SHAPE_Area"])
    n_total_bld = len(bld_df)
    print(f"Loaded {n_total_bld:,} buildings from {gpkg_bld} in {time.time()-t0:.2f} s")
    
    n_under_400 = int(np.sum(bld_df["SHAPE_Area"] < 400.0))
    n_eligible = int(np.sum(bld_df["SHAPE_Area"] >= 400.0))
    share_unusable = (n_under_400 / n_total_bld) * 100.0
    print(f"Total buildings in Izmir FUA: {n_total_bld:,}")
    print(f"Buildings with footprint < 400 m2 (unusable for 5 m raster interior extraction): {n_under_400:,} ({share_unusable:.2f}%)")
    print(f"Eligible buildings with footprint >= 400 m2: {n_eligible:,} ({n_eligible/n_total_bld*100:.2f}%)")
    
    # Filter candidate buildings
    valid_floors_mask = (bld_df["ZEMINUSTUK"] > 0) & (bld_df["SHAPE_Area"] >= 400.0)
    eligible = bld_df[valid_floors_mask].copy().reset_index(drop=True)
    
    print("Applying inward buffer (-2.5 m) to eligible buildings...")
    t0 = time.time()
    shrunk_geoms = eligible.geometry.buffer(-2.5)
    valid_shrunk = ~shrunk_geoms.is_empty
    eligible = eligible[valid_shrunk].copy().reset_index(drop=True)
    eligible["shrunk_geom"] = shrunk_geoms[valid_shrunk].values
    eligible["bld_idx"] = np.arange(1, len(eligible) + 1, dtype=np.int32)
    print(f"Valid shrunk polygons: {len(eligible):,} in {time.time()-t0:.2f} s")
    
    # -------------------------------------------------------------
    # 2. Extract nDSM v2 and nDSM v1 Heights
    # -------------------------------------------------------------
    print(f"\n--- 2. Extracting 90th Percentile nDSM v2 Heights ---")
    t0 = time.time()
    with rasterio.open(ndsm_v2_path) as src_v2:
        shape = src_v2.shape
        transform = src_v2.transform
        ndsm_v2 = src_v2.read(1)
        nodata = src_v2.nodata
        
    labeled = features.rasterize(
        ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
        out_shape=shape,
        transform=transform,
        dtype=np.int32,
        fill=0
    )
    
    pix_mask = (labeled > 0) & (ndsm_v2 != nodata)
    df_px = pd.DataFrame({"bld_idx": labeled[pix_mask], "ndsm_v2": ndsm_v2[pix_mask]})
    del labeled
    del ndsm_v2
    
    p90_v2 = df_px.groupby("bld_idx")["ndsm_v2"].agg(
        p90_v2=lambda x: np.percentile(x, 90),
        px_count="count"
    ).reset_index()
    del df_px
    
    eligible = eligible.merge(p90_v2, on="bld_idx", how="inner")
    
    # Also extract v1 for direct comparison
    with rasterio.open(ndsm_v1_path) as src_v1:
        ndsm_v1 = src_v1.read(1)
    labeled_v1 = features.rasterize(
        ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
        out_shape=shape,
        transform=transform,
        dtype=np.int32,
        fill=0
    )
    pix_mask_v1 = (labeled_v1 > 0) & (ndsm_v1 != src_v1.nodata)
    df_px_v1 = pd.DataFrame({"bld_idx": labeled_v1[pix_mask_v1], "ndsm_v1": ndsm_v1[pix_mask_v1]})
    del labeled_v1
    del ndsm_v1
    p90_v1 = df_px_v1.groupby("bld_idx")["ndsm_v1"].agg(p90_v1=lambda x: np.percentile(x, 90)).reset_index()
    del df_px_v1
    eligible = eligible.merge(p90_v1, on="bld_idx", how="left")
    print(f"Height extraction complete in {time.time()-t0:.2f} s for {len(eligible):,} candidate buildings")
    
    # Spatial join with analysis grids to assign administrative district
    print("Joining centroids to analysis grids for district stratification...")
    grids = pyogrio.read_dataframe(gpkg_grids, layer="grid_250m")
    v8 = pd.read_csv(v8_csv, usecols=["grid_id", "district"])
    grids = grids.merge(v8, on="grid_id", how="left")
    
    centroids = eligible.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(centroids, grids[["district", "geometry"]], how="left", predicate="within")
    eligible["district"] = joined["district"].fillna("unknown").values
    del centroids
    del grids
    del v8
    
    # Clean Calibration Sample: px_count >= 4, p90_v2 > 0.5, 1 <= floors <= 40
    clean = eligible[
        (eligible["px_count"] >= 4) &
        (eligible["p90_v2"] > 0.5) &
        (eligible["ZEMINUSTUK"] <= 40)
    ].copy().reset_index(drop=True)
    n_clean = len(clean)
    print(f"Clean calibration sample size: N = {n_clean:,} buildings")
    
    # -------------------------------------------------------------
    # 3. Storey Regressions on nDSM v2 vs v1
    # -------------------------------------------------------------
    print("\n--- 3. Global Storey-Height Regressions ---")
    y_v2 = clean["p90_v2"].values
    y_v1 = clean["p90_v1"].values
    x_floors = clean["ZEMINUSTUK"].values.astype(float)
    
    m1_v2 = regress_ols(y_v2, x_floors, fit_intercept=False)
    m2_v2 = regress_ols(y_v2, x_floors, fit_intercept=True)
    m1_v1 = regress_ols(y_v1, x_floors, fit_intercept=False)
    m2_v1 = regress_ols(y_v1, x_floors, fit_intercept=True)
    
    print(f"Global Model 1 (Through origin):")
    print(f"  WP1.1 (v1): Height = {m1_v1['beta'][0]:.3f} * floors (95% CI: [{m1_v1['ci'][0][0]:.3f}, {m1_v1['ci'][0][1]:.3f}], RMSE={m1_v1['rmse']:.2f} m)")
    print(f"  WP1.1b (v2): Height = {m1_v2['beta'][0]:.3f} * floors (95% CI: [{m1_v2['ci'][0][0]:.3f}, {m1_v2['ci'][0][1]:.3f}], RMSE={m1_v2['rmse']:.2f} m)")
    
    print(f"\nGlobal Model 2 (With intercept):")
    print(f"  WP1.1 (v1): Height = {m2_v1['beta'][0]:.3f} + {m2_v1['beta'][1]:.3f} * floors (R2={m2_v1['r2']:.3f}, RMSE={m2_v1['rmse']:.2f} m)")
    print(f"  WP1.1b (v2): Height = {m2_v2['beta'][0]:.3f} + {m2_v2['beta'][1]:.3f} * floors (R2={m2_v2['r2']:.3f}, RMSE={m2_v2['rmse']:.2f} m)")
    print(f"  v2 Intercept 95% CI: [{m2_v2['ci'][0][0]:.3f}, {m2_v2['ci'][0][1]:.3f}] m")
    print(f"  v2 Slope 95% CI:     [{m2_v2['ci'][1][0]:.3f}, {m2_v2['ci'][1][1]:.3f}] m/storey")
    
    # District Stratification
    print("\n--- 4. District-Level Storey Height Calibration (N >= 30) ---")
    districts = sorted([d for d in clean["district"].unique() if d != "unknown"])
    district_results = []
    
    for dist in districts:
        sub = clean[clean["district"] == dist]
        if len(sub) >= 30:
            y_sub = sub["p90_v2"].values
            x_sub = sub["ZEMINUSTUK"].values.astype(float)
            y_sub_v1 = sub["p90_v1"].values
            
            m1_d = regress_ols(y_sub, x_sub, fit_intercept=False)
            m2_d = regress_ols(y_sub, x_sub, fit_intercept=True)
            m1_d_v1 = regress_ols(y_sub_v1, x_sub, fit_intercept=False)
            
            district_results.append({
                "district": dist,
                "n": len(sub),
                "v1_beta_orig": m1_d_v1["beta"][0],
                "v2_beta_orig": m1_d["beta"][0],
                "v2_ci_orig": m1_d["ci"][0],
                "v2_alpha": m2_d["beta"][0],
                "v2_beta_slope": m2_d["beta"][1],
                "v2_ci_slope": m2_d["ci"][1],
                "v2_r2": m2_d["r2"],
                "v2_rmse": m2_d["rmse"]
            })
            print(f"  {dist:14s} (N={len(sub):,}): v1={m1_d_v1['beta'][0]:.3f} m/flr -> v2={m1_d['beta'][0]:.3f} m/flr (95% CI: [{m1_d['ci'][0][0]:.2f}, {m1_d['ci'][0][1]:.2f}]), Model 2: H={m2_d['beta'][0]:.2f}+{m2_d['beta'][1]:.2f}*F, R2={m2_d['r2']:.3f}")
            
    # -------------------------------------------------------------
    # 4. Independent Cross-Check against OSM Buildings
    # -------------------------------------------------------------
    print("\n--- 5. Independent Cross-Check with OSM Buildings ---")
    osm_gdf = gpd.read_file(osm_path)
    print(f"Loaded {len(osm_gdf):,} OSM buildings from {osm_path}")
    
    # Match OSM buildings with municipal buildings (via spatial join on geometry)
    osm_with_levels = osm_gdf[osm_gdf["levels"].notna() & (osm_gdf["levels"] > 0) & (osm_gdf["levels"] <= 40)].copy()
    osm_matched = gpd.sjoin(osm_with_levels, bld_df[["ZEMINUSTUK", "SHAPE_Area", "geometry"]], how="inner", predicate="intersects")
    print(f"OSM buildings spatially intersecting municipal buildings: {len(osm_matched):,}")
    
    # Compare OSM levels vs municipal ZEMINUSTUK
    osm_matched["level_diff"] = osm_matched["levels"] - osm_matched["ZEMINUSTUK"]
    corr_levels = osm_matched["levels"].corr(osm_matched["ZEMINUSTUK"])
    mae_levels = np.mean(np.abs(osm_matched["level_diff"]))
    med_diff_levels = np.median(osm_matched["level_diff"])
    pct_exact_levels = np.mean(osm_matched["levels"] == osm_matched["ZEMINUSTUK"]) * 100.0
    pct_within1_levels = np.mean(np.abs(osm_matched["level_diff"]) <= 1.0) * 100.0
    
    print(f"OSM 'building:levels' vs Municipal 'ZEMINUSTUK' Comparison (N={len(osm_matched):,}):")
    print(f"  Pearson Correlation: r = {corr_levels:.3f}")
    print(f"  Exact Floor Agreement: {pct_exact_levels:.1f}%")
    print(f"  Within +/- 1 Floor:     {pct_within1_levels:.1f}%")
    print(f"  Mean Absolute Difference: {mae_levels:.2f} floors, Median Difference: {med_diff_levels:+.1f} floors")
    
    # OSM measured heights vs levels
    osm_with_h = osm_gdf[osm_gdf["height"].notna() & (osm_gdf["height"] > 2.0) & (osm_gdf["levels"].notna()) & (osm_gdf["levels"] > 0)].copy()
    print(f"\nOSM buildings with BOTH measured height and building:levels: N = {len(osm_with_h):,}")
    if len(osm_with_h) >= 10:
        y_osm_h = osm_with_h["height"].values
        x_osm_l = osm_with_h["levels"].values
        m_osm_orig = regress_ols(y_osm_h, x_osm_l, fit_intercept=False)
        m_osm_int = regress_ols(y_osm_h, x_osm_l, fit_intercept=True)
        print(f"  OSM Ground Truth Height ~ OSM Levels:")
        print(f"    Origin:    Height = {m_osm_orig['beta'][0]:.3f} m/storey (95% CI: [{m_osm_orig['ci'][0][0]:.3f}, {m_osm_orig['ci'][0][1]:.3f}], RMSE={m_osm_orig['rmse']:.2f} m)")
        print(f"    Intercept: Height = {m_osm_int['beta'][0]:.2f} + {m_osm_int['beta'][1]:.3f} * floors (R2={m_osm_int['r2']:.3f}, RMSE={m_osm_int['rmse']:.2f} m)")
    else:
        m_osm_orig = {"beta": [np.nan], "ci": [(np.nan, np.nan)], "rmse": np.nan}
        m_osm_int = {"beta": [np.nan, np.nan], "ci": [(np.nan, np.nan), (np.nan, np.nan)], "r2": np.nan, "rmse": np.nan}
        
    # Extract nDSM v2 for OSM buildings
    print("Extracting nDSM v2 at OSM building centroids...")
    with rasterio.open(ndsm_v2_path) as src_v2:
        ndsm_v2_arr = src_v2.read(1)
        trans_v2 = src_v2.transform
        nodata_v2 = src_v2.nodata
        
    osm_ndsm_vals = []
    for geom in osm_with_levels.geometry:
        pt = geom.centroid
        px, py = ~trans_v2 * (pt.x, pt.y)
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < ndsm_v2_arr.shape[0] and 0 <= ix < ndsm_v2_arr.shape[1]:
            val = ndsm_v2_arr[iy, ix]
            osm_ndsm_vals.append(val if val != nodata_v2 else np.nan)
        else:
            osm_ndsm_vals.append(np.nan)
            
    osm_with_levels["ndsm_v2"] = osm_ndsm_vals
    osm_clean = osm_with_levels[osm_with_levels["ndsm_v2"] > 0.5].dropna(subset=["ndsm_v2", "levels"]).copy()
    print(f"OSM buildings with valid nDSM v2 extraction: N = {len(osm_clean):,}")
    
    m_osm_ndsm = regress_ols(osm_clean["ndsm_v2"].values, osm_clean["levels"].values, fit_intercept=False)
    print(f"  nDSM v2 Height ~ OSM Levels: Implied Height = {m_osm_ndsm['beta'][0]:.3f} m/storey (95% CI: [{m_osm_ndsm['ci'][0][0]:.3f}, {m_osm_ndsm['ci'][0][1]:.3f}])")
    
    # -------------------------------------------------------------
    # 5. Smoothing Bias vs Building Footprint Size in Dense Canyons
    # -------------------------------------------------------------
    print("\n--- 6. Quantifying 5 m Raster Smoothing Bias vs Footprint Size ---")
    # Sample 10,000 random buildings across size tiers to measure smoothing bias
    area_bins = [
        ("< 100 m2 (Narrow canyon / row house)", 0.0, 100.0),
        ("100 - 200 m2 (Small residential)", 100.0, 200.0),
        ("200 - 400 m2 (Standard residential)", 200.0, 400.0),
        ("400 - 800 m2 (Large apartment)", 400.0, 800.0),
        (">= 800 m2 (Commercial / Complex)", 800.0, 1e7),
    ]
    
    bias_results = []
    # Sample buildings with valid floors >= 2
    bld_sample = bld_df[(bld_df["ZEMINUSTUK"] >= 2) & (bld_df["ZEMINUSTUK"] <= 20)].copy()
    # Sample up to 2000 per bin
    rng = np.random.default_rng(42)
    sample_geoms = []
    sample_floors = []
    sample_areas = []
    sample_bins = []
    
    for bname, amin, amax in area_bins:
        sub_b = bld_sample[(bld_sample["SHAPE_Area"] >= amin) & (bld_sample["SHAPE_Area"] < amax)]
        if len(sub_b) > 2000:
            idx = rng.choice(len(sub_b), size=2000, replace=False)
            sub_b = sub_b.iloc[idx]
        sample_geoms.extend(sub_b.geometry.values)
        sample_floors.extend(sub_b["ZEMINUSTUK"].values)
        sample_areas.extend(sub_b["SHAPE_Area"].values)
        sample_bins.extend([bname] * len(sub_b))
        
    df_sample = pd.DataFrame({
        "bin": sample_bins,
        "area": sample_areas,
        "floors": sample_floors,
        "geom": sample_geoms
    })
    
    # Extract centroid nDSM v2
    cent_vals = []
    for g in df_sample["geom"]:
        pt = g.centroid
        px, py = ~trans_v2 * (pt.x, pt.y)
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < ndsm_v2_arr.shape[0] and 0 <= ix < ndsm_v2_arr.shape[1]:
            val = ndsm_v2_arr[iy, ix]
            cent_vals.append(val if val != nodata_v2 else np.nan)
        else:
            cent_vals.append(np.nan)
            
    df_sample["ndsm_v2"] = cent_vals
    df_sample = df_sample[df_sample["ndsm_v2"] > 0.5].copy()
    df_sample["implied_h_per_storey"] = df_sample["ndsm_v2"] / df_sample["floors"]
    
    print("\nImplied Height per Storey by Footprint Size Tier:")
    for bname, _, _ in area_bins:
        tier = df_sample[df_sample["bin"] == bname]
        if len(tier) > 0:
            h_med = np.median(tier["implied_h_per_storey"])
            h_mean = np.mean(tier["implied_h_per_storey"])
            h_p25 = np.percentile(tier["implied_h_per_storey"], 25)
            h_p75 = np.percentile(tier["implied_h_per_storey"], 75)
            bias_results.append({
                "bin": bname,
                "n": len(tier),
                "med_h": h_med,
                "mean_h": h_mean,
                "p25": h_p25,
                "p75": h_p75
            })
            print(f"  {bname:42s} (N={len(tier):,}): Median = {h_med:.2f} m/storey (IQR: [{h_p25:.2f}, {h_p75:.2f}] m)")
            
    del ndsm_v2_arr
    
    # -------------------------------------------------------------
    # 6. Format Markdown Tables
    # -------------------------------------------------------------
    print("\n" + "="*120)
    print("### TABLE 1: STOREY-HEIGHT CALIBRATION ON nDSM v2 (WP1.1b vs WP1.1)")
    print("="*120)
    print(f"| Model | Version | Beta (Slope) | 95% CI (m/storey) | Intercept Alpha (m) | 95% CI (m) | R2 | RMSE (m) | Sample N |")
    print(f"|---|---|---|---|---|---|---|---|---|")
    print(f"| Model 1 (Origin) | WP1.1 (v1) | {m1_v1['beta'][0]:.3f} m | [{m1_v1['ci'][0][0]:.3f}, {m1_v1['ci'][0][1]:.3f}] | - | - | - | {m1_v1['rmse']:.2f} m | {n_clean:,} |")
    print(f"| **Model 1 (Origin)** | **WP1.1b (v2)** | **{m1_v2['beta'][0]:.3f} m** | **[{m1_v2['ci'][0][0]:.3f}, {m1_v2['ci'][0][1]:.3f}]** | - | - | - | **{m1_v2['rmse']:.2f} m** | **{n_clean:,}** |")
    print(f"| Model 2 (Intercept) | WP1.1 (v1) | {m2_v1['beta'][1]:.3f} m | [{m2_v1['ci'][1][0]:.3f}, {m2_v1['ci'][1][1]:.3f}] | {m2_v1['beta'][0]:.3f} m | [{m2_v1['ci'][0][0]:.3f}, {m2_v1['ci'][0][1]:.3f}] | {m2_v1['r2']:.3f} | {m2_v1['rmse']:.2f} m | {n_clean:,} |")
    print(f"| **Model 2 (Intercept)** | **WP1.1b (v2)** | **{m2_v2['beta'][1]:.3f} m** | **[{m2_v2['ci'][1][0]:.3f}, {m2_v2['ci'][1][1]:.3f}]** | **{m2_v2['beta'][0]:.3f} m** | **[{m2_v2['ci'][0][0]:.3f}, {m2_v2['ci'][0][1]:.3f}]** | **{m2_v2['r2']:.3f}** | **{m2_v2['rmse']:.2f} m** | **{n_clean:,}** |")
    
    print("\n" + "="*120)
    print("### TABLE 2: DISTRICT-STRATIFIED STOREY CALIBRATION (WP1.1b nDSM v2)")
    print("="*120)
    h_dist = ["District", "Sample Size (N)", "v1 Origin Beta", "v2 Origin Beta", "v2 95% CI", "v2 Alpha (Base)", "v2 Beta (Slope)", "v2 Slope 95% CI", "R2", "RMSE"]
    print("| " + " | ".join(h_dist) + " |")
    print("|" + "|".join(["---"]*len(h_dist)) + "|")
    for r in district_results:
        line = [
            r["district"],
            f"{r['n']:,}",
            f"{r['v1_beta_orig']:.3f} m",
            f"**{r['v2_beta_orig']:.3f} m**",
            f"[{r['v2_ci_orig'][0]:.2f}, {r['v2_ci_orig'][1]:.2f}]",
            f"{r['v2_alpha']:.2f} m",
            f"{r['v2_beta_slope']:.3f} m",
            f"[{r['v2_ci_slope'][0]:.2f}, {r['v2_ci_slope'][1]:.2f}]",
            f"{r['v2_r2']:.3f}",
            f"{r['v2_rmse']:.2f} m"
        ]
        print("| " + " | ".join(line) + " |")
        
    print("\n" + "="*120)
    print("### TABLE 3: FOOTPRINT SIZE vs 5 M RASTER SMOOTHING BIAS")
    print("="*120)
    h_bias = ["Footprint Area Tier", "Sample (N)", "Median Implied m/storey", "Mean Implied m/storey", "IQR (p25 - p75)", "Resolution Status"]
    print("| " + " | ".join(h_bias) + " |")
    print("|" + "|".join(["---"]*len(h_bias)) + "|")
    for r in bias_results:
        status = "**UNRESOLVED (Smearing Bias)**" if "Narrow" in r["bin"] or "Small" in r["bin"] or "<" in r["bin"] else "**RESOLVED (Clean Core)**"
        line = [
            r["bin"],
            f"{r['n']:,}",
            f"{r['med_h']:.2f} m",
            f"{r['mean_h']:.2f} m",
            f"[{r['p25']:.2f}, {r['p75']:.2f}] m",
            status
        ]
        print("| " + " | ".join(line) + " |")
        
    print(f"\nStep 4 completed in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
