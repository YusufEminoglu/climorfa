"""
WP1.1 Storey-Height Calibration & Quicklooks Generation
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
1. Storey-height calibration: for buildings with footprint >= 400 m2,
   compute 90th percentile nDSM inside footprint shrunk 2.5 m.
2. Regress against ZEMINUSTUK:
   - Model 1: Through origin (implied metres per storey + 95% CI)
   - Model 2: With intercept (alpha + beta * floors + 95% CI)
   - By district via centroid spatial join to analysis_grids.gpkg (joined to model_features_v8.csv).
3. Report share of buildings where 5 m nDSM is unusable (< 400 m2).
4. Generate quicklooks:
   - outputs/street/terrain/quicklook_hillshade_dsm_vs_dtm.png (3 windows: Alsancak, Bornova, Karabaglar)
   - outputs/street/terrain/quicklook_storey_calibration.png (scatter, regression fits, district comparison)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
import scipy.stats as stats
from scipy.ndimage import sobel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pyogrio

# Set global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2c3e50",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

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
    dof = n - p
    sigma2 = np.sum(resids**2) / dof
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(var_beta))
    t_crit = stats.t.ppf(0.975, dof)
    ci_lower = beta - t_crit * se_beta
    ci_upper = beta + t_crit * se_beta
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(resids**2)
    r2 = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(resids**2))
    return {
        "beta": beta,
        "se": se_beta,
        "ci": list(zip(ci_lower, ci_upper)),
        "r2": r2,
        "rmse": rmse,
        "n": n
    }

def compute_hillshade(dem, cellsize=5.0, azimuth=315.0, altitude=45.0):
    dx = sobel(dem, axis=1) / (8.0 * cellsize)
    dy = sobel(dem, axis=0) / (8.0 * cellsize)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    
    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)
    
    shaded = np.sin(zenith_rad) * np.sin(slope) + np.cos(zenith_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    hillshade = 255.0 * np.clip(shaded, 0.0, 1.0)
    return hillshade

def main():
    t_start = time.time()
    print("=== Step 4: Storey-Height Calibration & Quicklooks Generation ===")
    
    # Paths
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    dtm_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    ndsm_path = "data/street/terrain/ndsm_5m_epsg5253.tif"
    gpkg_bld = "data/02_interim/akilli_sehir_fua.gpkg"
    gpkg_grids = "data/03_processed/analysis_grids.gpkg"
    v8_csv = "data/03_processed/grid_250m_model_features_v8.csv"
    
    out_dir = "outputs/street/terrain"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Buildings & Calculate Usability Statistics
    print("Loading building footprints and floor counts...")
    t0 = time.time()
    bld_df = pyogrio.read_dataframe(gpkg_bld, layer="buildings_fua", columns=["ZEMINUSTUK", "SHAPE_Area"])
    n_total_bld = len(bld_df)
    print(f"Loaded {n_total_bld:,} buildings in {time.time()-t0:.2f} s")
    
    # Area thresholds
    n_under_400 = int(np.sum(bld_df["SHAPE_Area"] < 400.0))
    n_eligible = int(np.sum(bld_df["SHAPE_Area"] >= 400.0))
    share_unusable_area = (n_under_400 / n_total_bld) * 100.0
    print(f"Building footprint < 400 m2 (unusable for 5m nDSM): {n_under_400:,} ({share_unusable_area:.2f}%)")
    print(f"Building footprint >= 400 m2 (eligible candidate): {n_eligible:,} ({n_eligible/n_total_bld*100:.2f}%)")
    
    # Floor statistics
    valid_floors_mask = (bld_df["ZEMINUSTUK"] > 0) & (bld_df["SHAPE_Area"] >= 400.0)
    eligible = bld_df[valid_floors_mask].copy().reset_index(drop=True)
    
    # Shrunk footprint
    print("Buffering eligible buildings (-2.5 m)...")
    shrunk_geoms = eligible.geometry.buffer(-2.5)
    valid_shrunk = ~shrunk_geoms.is_empty
    eligible = eligible[valid_shrunk].copy().reset_index(drop=True)
    eligible["shrunk_geom"] = shrunk_geoms[valid_shrunk].values
    eligible["bld_idx"] = np.arange(1, len(eligible) + 1, dtype=np.int32)
    n_usable_shrunk = len(eligible)
    
    total_unusable_bld = n_total_bld - n_usable_shrunk
    total_unusable_share = (total_unusable_bld / n_total_bld) * 100.0
    print(f"Final usable buildings for 5m calibration: {n_usable_shrunk:,} / {n_total_bld:,}")
    print(f"Total unusable share (too small or empty core): {total_unusable_share:.2f}%\n")
    
    # 2. Extract 90th percentile nDSM inside shrunk footprint
    print(f"Extracting 90th percentile nDSM from {ndsm_path}...")
    t0 = time.time()
    with rasterio.open(ndsm_path) as src_ndsm:
        shape = src_ndsm.shape
        transform = src_ndsm.transform
        ndsm = src_ndsm.read(1)
        nodata = src_ndsm.nodata
        
    labeled = features.rasterize(
        ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
        out_shape=shape,
        transform=transform,
        dtype=np.int32,
        fill=0
    )
    
    pix_mask = (labeled > 0) & (ndsm != nodata)
    df_px = pd.DataFrame({"bld_idx": labeled[pix_mask], "ndsm": ndsm[pix_mask]})
    del labeled
    del ndsm
    
    p90 = df_px.groupby("bld_idx")["ndsm"].agg(
        p90=lambda x: np.percentile(x, 90),
        px_count="count"
    ).reset_index()
    del df_px
    
    eligible = eligible.merge(p90, on="bld_idx", how="inner")
    print(f"Extraction complete in {time.time()-t0:.2f} s for {len(eligible):,} buildings")
    
    # 3. Spatial Join with Analysis Grids to attach District
    print("Spatially joining building centroids to 250m analysis grids to assign district...")
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
    
    # 4. Clean Calibration Sample
    clean = eligible[
        (eligible["px_count"] >= 4) &
        (eligible["p90"] > 0.5) &
        (eligible["ZEMINUSTUK"] <= 40)
    ].copy()
    print(f"Clean calibration sample (px_count >= 4, p90 > 0.5m, floors <= 40): {len(clean):,} buildings\n")
    
    # 5. OLS Regressions
    y = clean["p90"].values
    x = clean["ZEMINUSTUK"].values.astype(float)
    
    m1 = regress_ols(y, x, fit_intercept=False)
    m2 = regress_ols(y, x, fit_intercept=True)
    
    b_orig = m1["beta"][0]
    ci_orig = m1["ci"][0]
    
    alpha = m2["beta"][0]
    ci_alpha = m2["ci"][0]
    b_slope = m2["beta"][1]
    ci_slope = m2["ci"][1]
    
    print("="*75)
    print("### STOREY-HEIGHT CALIBRATION RESULTS (FULL IZMIR FUA)")
    print("="*75)
    print(f"Sample size N = {len(clean):,} buildings (footprint >= 400 m2)")
    print(f"Unusable building share (< 400 m2): {total_unusable_share:.2f}%\n")
    print(f"MODEL 1 (Through origin: Height = beta * Floors):")
    print(f"  Implied metres per storey: {b_orig:.3f} m/storey")
    print(f"  95% Confidence Interval:  [{ci_orig[0]:.3f}, {ci_orig[1]:.3f}] m/storey (margin: +/- {ci_orig[1]-b_orig:.3f} m)")
    print(f"  RMSE:                     {m1['rmse']:.2f} m\n")
    
    print(f"MODEL 2 (With intercept: Height = alpha + beta * Floors):")
    print(f"  Base offset (alpha):      {alpha:.3f} m (95% CI: [{ci_alpha[0]:.3f}, {ci_alpha[1]:.3f}] m)")
    print(f"  Slope (beta):             {b_slope:.3f} m/storey (95% CI: [{ci_slope[0]:.3f}, {ci_slope[1]:.3f}] m)")
    print(f"  R-squared:                {m2['r2']:.3f}")
    print(f"  RMSE:                     {m2['rmse']:.2f} m\n")
    
    # District Breakdown
    print("--- DISTRICT BREAKDOWN (for districts with N >= 30) ---")
    dist_results = []
    for dist, grp in clean.groupby("district"):
        if len(grp) < 30 or dist == "unknown":
            continue
        y_d = grp["p90"].values
        x_d = grp["ZEMINUSTUK"].values.astype(float)
        m1_d = regress_ols(y_d, x_d, fit_intercept=False)
        m2_d = regress_ols(y_d, x_d, fit_intercept=True)
        dist_results.append({
            "District": dist,
            "N": len(grp),
            "Beta_Origin": m1_d["beta"][0],
            "CI_Origin_Low": m1_d["ci"][0][0],
            "CI_Origin_High": m1_d["ci"][0][1],
            "Alpha": m2_d["beta"][0],
            "CI_Alpha_Low": m2_d["ci"][0][0],
            "CI_Alpha_High": m2_d["ci"][0][1],
            "Beta_Slope": m2_d["beta"][1],
            "CI_Slope_Low": m2_d["ci"][1][0],
            "CI_Slope_High": m2_d["ci"][1][1],
            "R2": m2_d["r2"],
            "RMSE": m2_d["rmse"]
        })
        
    df_dist = pd.DataFrame(dist_results).sort_values("N", ascending=False)
    headers_dist = ["District", "N", "Beta (Origin)", "95% CI (Origin)", "Alpha (Base)", "Beta (Slope)", "95% CI (Slope)", "R2", "RMSE"]
    print("| " + " | ".join(headers_dist) + " |")
    print("|" + "|".join(["---"]*len(headers_dist)) + "|")
    for _, r in df_dist.iterrows():
        line = [
            r["District"],
            f"{int(r['N']):,}",
            f"{r['Beta_Origin']:.3f} m",
            f"[{r['CI_Origin_Low']:.2f}, {r['CI_Origin_High']:.2f}]",
            f"{r['Alpha']:.2f} m",
            f"{r['Beta_Slope']:.3f} m",
            f"[{r['CI_Slope_Low']:.2f}, {r['CI_Slope_High']:.2f}]",
            f"{r['R2']:.3f}",
            f"{r['RMSE']:.2f} m"
        ]
        print("| " + " | ".join(line) + " |")
        
    # 6. Generate Quicklooks
    print("\nGenerating publication-quality quicklooks...")
    
    # Figure 1: Storey Calibration Scatter & District Breakdown
    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
    
    # Subplot A: Scatter & Fits
    ax1 = fig.add_subplot(gs[0])
    # Subsample for clear scatter visualization if large
    scatter_sample = clean.sample(n=min(5000, len(clean)), random_state=42)
    ax1.scatter(
        scatter_sample["ZEMINUSTUK"], scatter_sample["p90"],
        alpha=0.25, color="#2980b9", edgecolors="none", s=18, label=f"Building samples (n={len(scatter_sample):,})"
    )
    
    floor_grid = np.linspace(1, 30, 100)
    # Model 1 fit
    pred_m1 = b_orig * floor_grid
    ci_m1_low = ci_orig[0] * floor_grid
    ci_m1_high = ci_orig[1] * floor_grid
    ax1.plot(floor_grid, pred_m1, color="#e74c3c", lw=2.5, label=f"Model 1 (Origin): {b_orig:.2f} m/fl (CI: {ci_orig[0]:.2f}–{ci_orig[1]:.2f})")
    ax1.fill_between(floor_grid, ci_m1_low, ci_m1_high, color="#e74c3c", alpha=0.15)
    
    # Model 2 fit
    pred_m2 = alpha + b_slope * floor_grid
    ci_m2_low = ci_alpha[0] + ci_slope[0] * floor_grid
    ci_m2_high = ci_alpha[1] + ci_slope[1] * floor_grid
    ax1.plot(floor_grid, pred_m2, color="#27ae60", lw=2.5, linestyle="--", label=f"Model 2 (Intercept): {alpha:.2f}m + {b_slope:.2f}m/fl ($R^2$={m2['r2']:.2f})")
    
    ax1.set_xlabel("Number of Above-Ground Storeys (ZEMINUSTUK)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Extracted 90th Percentile nDSM (m)", fontsize=11, fontweight="bold")
    ax1.set_title("Storey-Height Calibration (Izmir FUA, Footprint >= 400 m²)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(0.5, 25.5)
    ax1.set_ylim(0, 85)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Annotation box
    textstr = "\n".join([
        f"Calibration N = {len(clean):,} bldgs",
        f"Unusable (<400 m²): {total_unusable_share:.1f}%",
        f"Model 1 (origin): {b_orig:.3f} m/storey",
        f"Model 2 (slope):  {b_slope:.3f} m/storey",
        f"Model 2 offset:   {alpha:.2f} m"
    ])
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.95)
    ax1.text(0.96, 0.05, textstr, transform=ax1.transAxes, fontsize=9, verticalalignment="bottom", horizontalalignment="right", bbox=props)
    
    # Subplot B: District Comparison Errorbars
    ax2 = fig.add_subplot(gs[1])
    y_pos = np.arange(len(df_dist))
    ax2.errorbar(
        df_dist["Beta_Origin"], y_pos,
        xerr=[df_dist["Beta_Origin"] - df_dist["CI_Origin_Low"], df_dist["CI_Origin_High"] - df_dist["Beta_Origin"]],
        fmt="o", color="#2980b9", ecolor="#2980b9", elinewidth=2, capsize=4, label="Model 1 (Through origin)"
    )
    ax2.axvline(b_orig, color="#e74c3c", linestyle="--", lw=1.5, label=f"FUA Pooled ({b_orig:.2f} m)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{d} (n={n:,})" for d, n in zip(df_dist["District"], df_dist["N"])], fontsize=9)
    ax2.set_xlabel("Implied Metres per Storey (m/storey)", fontsize=11, fontweight="bold")
    ax2.set_title("Implied Storey Height by District (+95% CI)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(2.2, 3.2)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.invert_yaxis()
    
    plot_path1 = os.path.join(out_dir, "quicklook_storey_calibration.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved storey calibration quicklook: {plot_path1}")
    
    # Figure 2: Hillshade DSM vs DTM for 3 Required Windows
    print("Generating 3-window hillshade DSM vs DTM comparison quicklook...")
    test_wins = [
        ("Alsancak (Dense Urban Core)", (512350, 4255850, 512850, 4256350)),
        ("Bornova (Urban Mid-Rise)", (517500, 4257500, 519000, 4259000)),
        ("Karabaglar (Dense Hillside)", (508500, 4248000, 510000, 4249500))
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(11, 14), dpi=300)
    
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dtm_path) as src_dtm:
        for idx, (title, bbox) in enumerate(test_wins):
            win = from_bounds(*bbox, transform=src_dsm.transform).round_offsets().round_lengths()
            crop_dsm = src_dsm.read(1, window=win).astype(np.float32)
            crop_dtm = src_dtm.read(1, window=win).astype(np.float32)
            
            hs_dsm = compute_hillshade(crop_dsm, cellsize=5.0)
            hs_dtm = compute_hillshade(crop_dtm, cellsize=5.0)
            
            # DSM
            ax_dsm = axes[idx, 0]
            im0 = ax_dsm.imshow(hs_dsm, cmap="gray", vmin=0, vmax=255)
            ax_dsm.set_title(f"{title} — DSM (Surface Model)", fontsize=11, fontweight="bold")
            ax_dsm.axis("off")
            
            # DTM
            ax_dtm = axes[idx, 1]
            im1 = ax_dtm.imshow(hs_dtm, cmap="gray", vmin=0, vmax=255)
            ax_dtm.set_title(f"{title} — DTM (Bare-Earth Model)", fontsize=11, fontweight="bold")
            ax_dtm.axis("off")
            
            # Add label
            ax_dsm.text(0.03, 0.05, f"Window: {bbox[0]},{bbox[1]}", transform=ax_dsm.transAxes, color="white", fontsize=8,
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            ax_dtm.text(0.03, 0.05, "Buildings Removed (DTM <= DSM)", transform=ax_dtm.transAxes, color="#2ecc71", fontsize=8, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            
    plot_path2 = os.path.join(out_dir, "quicklook_hillshade_dsm_vs_dtm.png")
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved hillshade quicklook: {plot_path2}")
    
    print(f"Step 4 complete in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
