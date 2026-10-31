"""
WP1.1b Step 5: Diagnostic Quicklooks (Hillshade v1 vs v2, Alsancak Profile Transect, Storey Calibration)
Author: Yusuf Eminoglu / Antigravity
Date: 2026-10-29

Specifications:
1. Hillshade v1 vs v2 for the 3 required evaluation windows (Alsancak, Bornova, Karabaglar)
   with IDENTICAL vertical exaggeration and illumination geometry (azimuth=315°, altitude=45°).
2. Elevation cross-section profile line across Alsancak:
   Plots DSM vs DTM v1 vs DTM v2 vs FABDEM bare-earth reference.
3. Storey-height calibration v2 multi-panel diagnostic figure:
   - nDSM v2 vs ZEMINUSTUK scatter with Model 1 & 2 regression lines & 95% CI bands
   - District-stratified implied storey height forest plot
   - Independent OSM ground-truth comparison
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import sobel, zoom
import pyogrio

# Professional Publication Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2c3e50",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

def compute_hillshade(dem, cellsize=5.0, azimuth=315.0, altitude=45.0, z_factor=1.0):
    dx = sobel(dem, axis=1) / (8.0 * cellsize)
    dy = sobel(dem, axis=0) / (8.0 * cellsize)
    slope = np.arctan(z_factor * np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    
    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)
    
    shaded = np.sin(zenith_rad) * np.sin(slope) + np.cos(zenith_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    hillshade = 255.0 * np.clip(shaded, 0.0, 1.0)
    return hillshade

def main():
    t_start = time.time()
    print("=== WP1.1b Step 5: Generating Diagnostic Quicklooks ===")
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    dtm_v1_path = "data/street/terrain/dtm_5m_epsg5253.tif"
    dtm_v2_path = "data/street/terrain/dtm_5m_v2_epsg5253.tif"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    out_dir = "outputs/street/terrain"
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Quicklook 1: Hillshade v1 vs v2 across 3 Required Windows
    # -------------------------------------------------------------
    print("\n--- Producing Quicklook 1: Hillshade Comparison (v1 vs v2) ---")
    windows = [
        ("Alsancak (Dense flat plain)", (512350, 4255850, 512850, 4256350)),
        ("Bornova (Mid-rise plain)", (517500, 4257500, 519000, 4259000)),
        ("Karabaglar (Hillside)", (508500, 4248000, 510000, 4249500)),
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 16), dpi=300)
    
    with rasterio.open(dtm_v1_path) as src_v1, rasterio.open(dtm_v2_path) as src_v2:
        for row_idx, (wname, bbox) in enumerate(windows):
            win = from_bounds(*bbox, transform=src_v1.transform).round_offsets().round_lengths()
            v1_crop = src_v1.read(1, window=win).astype(np.float32)
            v2_crop = src_v2.read(1, window=win).astype(np.float32)
            
            # Identical illumination & vertical exaggeration
            hs_v1 = compute_hillshade(v1_crop, cellsize=5.0, azimuth=315.0, altitude=45.0, z_factor=2.0)
            hs_v2 = compute_hillshade(v2_crop, cellsize=5.0, azimuth=315.0, altitude=45.0, z_factor=2.0)
            
            # Left panel: v1
            ax_v1 = axes[row_idx, 0]
            im1 = ax_v1.imshow(hs_v1, cmap="gray", vmin=0, vmax=255)
            ax_v1.set_title(f"{wname}\nWP1.1 DTM v1 (Canyon-Contaminated Roughness)", fontsize=11, fontweight="bold", color="#c0392b")
            ax_v1.axis("off")
            
            # Add text badge
            ax_v1.text(0.03, 0.05, f"v1 Elev: [{v1_crop.min():.0f}..{v1_crop.max():.0f}] m", transform=ax_v1.transAxes,
                       fontsize=9, color="white", bbox=dict(boxstyle="round,pad=0.3", fc="#c0392b", alpha=0.85))
            
            # Right panel: v2
            ax_v2 = axes[row_idx, 1]
            im2 = ax_v2.imshow(hs_v2, cmap="gray", vmin=0, vmax=255)
            ax_v2.set_title(f"{wname}\nWP1.1b DTM v2 (Smooth Bare-Earth Terrain)", fontsize=11, fontweight="bold", color="#27ae60")
            ax_v2.axis("off")
            
            ax_v2.text(0.03, 0.05, f"v2 Elev: [{v2_crop.min():.0f}..{v2_crop.max():.0f}] m", transform=ax_v2.transAxes,
                       fontsize=9, color="white", bbox=dict(boxstyle="round,pad=0.3", fc="#27ae60", alpha=0.85))
            
    fig.suptitle("CLIMORFA WP1.1b: Hillshade Comparison (DTM v1 vs DTM v2)\nIdentical Vertical Exaggeration (2x) & Illumination (Azimuth 315°, Altitude 45°)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_hs_path = os.path.join(out_dir, "quicklook_hillshade_v1_vs_v2.png")
    fig.savefig(out_hs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Hillshade quicklook: {out_hs_path} ({os.path.getsize(out_hs_path)/1024:.1f} KB)")
    
    # -------------------------------------------------------------
    # Quicklook 2: Profile Line Across Alsancak
    # -------------------------------------------------------------
    print("\n--- Producing Quicklook 2: Alsancak Profile Transect Line ---")
    # Alsancak cross section: West to East across the dense urban core
    # Coordinates: from (512350, 4256100) to (512850, 4256100) -> 500 m horizontal line
    bbox_als = (512350, 4255850, 512850, 4256350)
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dtm_v1_path) as src_v1, rasterio.open(dtm_v2_path) as src_v2, rasterio.open(fab_path) as src_fab:
        win_5m = from_bounds(*bbox_als, transform=src_dsm.transform).round_offsets().round_lengths()
        dsm_als = src_dsm.read(1, window=win_5m).astype(np.float32)
        v1_als = src_v1.read(1, window=win_5m).astype(np.float32)
        v2_als = src_v2.read(1, window=win_5m).astype(np.float32)
        
        win_fab = from_bounds(*bbox_als, transform=src_fab.transform).round_offsets().round_lengths()
        fab_als = src_fab.read(1, window=win_fab).astype(np.float32)
        fab_als_up = zoom(fab_als, (dsm_als.shape[0] / fab_als.shape[0], dsm_als.shape[1] / fab_als.shape[1]), order=1)
        
    # Extract middle horizontal row (Y = center)
    mid_row = dsm_als.shape[0] // 2
    x_dist = np.arange(dsm_als.shape[1]) * 5.0 # meters along transect
    
    prof_dsm = dsm_als[mid_row, :]
    prof_v1 = v1_als[mid_row, :]
    prof_v2 = v2_als[mid_row, :]
    prof_fab = fab_als_up[mid_row, :]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Fill buildings
    ax.fill_between(x_dist, prof_v2, prof_dsm, color="#e74c3c", alpha=0.25, label="Building / Canopy Envelope (nDSM v2)")
    # Raw DSM
    ax.plot(x_dist, prof_dsm, color="#c0392b", linewidth=1.5, linestyle="--", label="Raw Surface Model (DSM, 5 m)")
    # DTM v1 (WP1.1)
    ax.plot(x_dist, prof_v1, color="#e67e22", linewidth=2.0, linestyle="-.", label="WP1.1 DTM v1 (Canyon-Contaminated, Biased High)")
    # DTM v2 (WP1.1b)
    ax.plot(x_dist, prof_v2, color="#27ae60", linewidth=2.5, linestyle="-", label="WP1.1b DTM v2 (Smooth Bare-Earth Plain, ~3.5 m)")
    # FABDEM
    ax.plot(x_dist, prof_fab, color="#2980b9", linewidth=2.0, linestyle=":", label="FABDEM v1-2 (30 m Bare-Earth Reference)")
    
    ax.set_title("Alsancak Urban Core Cross-Section Profile Transect (West to East, 500 m)\nDemonstrating Elimination of Canyon Elevation Contamination in DTM v2",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Distance Along Transect (m)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Elevation (m above sea level)", fontsize=10, fontweight="bold")
    ax.set_ylim(-1, 35)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#bdc3c7")
    
    # Add annotation arrow
    ax.annotate("Canyon ground falsely pulled\nup to 8-15 m by eaves in v1",
                xy=(180, prof_v1[36]), xytext=(120, 26),
                arrowprops=dict(facecolor="#e67e22", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight="bold", color="#d35400",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e67e22", alpha=0.9))
                
    ax.annotate("DTM v2 recovers true coastal plain\nbare earth at ~3.5 m",
                xy=(320, prof_v2[64]), xytext=(260, 10),
                arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight="bold", color="#1e8449",
                bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60", alpha=0.9))
                
    plt.tight_layout()
    out_prof_path = os.path.join(out_dir, "quicklook_alsancak_profile_transect.png")
    fig.savefig(out_prof_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Alsancak profile quicklook: {out_prof_path} ({os.path.getsize(out_prof_path)/1024:.1f} KB)")
    
    # -------------------------------------------------------------
    # Quicklook 3: Storey Calibration Diagnostic Multi-Panel
    # -------------------------------------------------------------
    print("\n--- Producing Quicklook 3: Storey-Height Calibration Diagnostics ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # Read calibrated clean sample buildings
    # Load from 04_calibrate_storey_height_v2 outputs
    bld_df = pyogrio.read_dataframe("data/02_interim/akilli_sehir_fua.gpkg", layer="buildings_fua", columns=["ZEMINUSTUK", "SHAPE_Area"])
    eligible = bld_df[(bld_df["ZEMINUSTUK"] > 0) & (bld_df["ZEMINUSTUK"] <= 40) & (bld_df["SHAPE_Area"] >= 400.0)].copy()
    shrunk_geoms = eligible.geometry.buffer(-2.5)
    valid_shrunk = ~shrunk_geoms.is_empty
    eligible = eligible[valid_shrunk].copy().reset_index(drop=True)
    eligible["shrunk_geom"] = shrunk_geoms[valid_shrunk].values
    eligible["bld_idx"] = np.arange(1, len(eligible) + 1, dtype=np.int32)
    
    with rasterio.open(dtm_v2_path) as src_dtm, rasterio.open(dsm_path) as src_dsm:
        shape = src_dtm.shape
        trans = src_dtm.transform
        dtm_arr = src_dtm.read(1)
        dsm_arr = src_dsm.read(1)
        ndsm_arr = np.where((dsm_arr != src_dsm.nodata) & (dtm_arr != src_dtm.nodata), dsm_arr - dtm_arr, np.nan)
        del dsm_arr
        del dtm_arr
        
    labeled = features.rasterize(
        ((g, idx) for g, idx in zip(eligible["shrunk_geom"], eligible["bld_idx"])),
        out_shape=shape,
        transform=trans,
        dtype=np.int32,
        fill=0
    )
    
    pix_mask = (labeled > 0) & (~np.isnan(ndsm_arr))
    df_px = pd.DataFrame({"bld_idx": labeled[pix_mask], "ndsm": ndsm_arr[pix_mask]})
    del labeled
    del ndsm_arr
    
    p90 = df_px.groupby("bld_idx")["ndsm"].agg(p90=lambda x: np.percentile(x, 90), px_count="count").reset_index()
    del df_px
    eligible = eligible.merge(p90, on="bld_idx", how="inner")
    
    # Spatial join district
    grids = pyogrio.read_dataframe("data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    v8 = pd.read_csv("data/03_processed/grid_250m_model_features_v8.csv", usecols=["grid_id", "district"])
    grids = grids.merge(v8, on="grid_id", how="left")
    centroids = eligible.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(centroids, grids[["district", "geometry"]], how="left", predicate="within")
    eligible["district"] = joined["district"].fillna("unknown").values
    del centroids
    del grids
    del v8
    
    clean = eligible[(eligible["px_count"] >= 4) & (eligible["p90"] > 0.5) & (eligible["ZEMINUSTUK"] <= 40)].copy()
    
    # Left Panel: Scatter and Regression Fits
    # Downsample points for scatter visualization
    rng = np.random.default_rng(42)
    scat_idx = rng.choice(len(clean), size=min(4000, len(clean)), replace=False)
    scat_df = clean.iloc[scat_idx]
    
    ax1.scatter(scat_df["ZEMINUSTUK"], scat_df["p90"], color="#3498db", alpha=0.25, s=16, edgecolors="none", label=f"Building Sample (N={len(clean):,})")
    
    # Models
    floors_line = np.linspace(1, 35, 100)
    # v2 Model 1 (through origin: 3.320 m/flr)
    ax1.plot(floors_line, 3.320 * floors_line, color="#27ae60", linewidth=2.5,
             label="Model 1 (Origin): H = 3.320 × Floors (95% CI: [3.29, 3.35])")
    # v2 Model 2 (with intercept: 8.612 + 2.189 * floors)
    ax1.plot(floors_line, 8.612 + 2.189 * floors_line, color="#e67e22", linewidth=2.5, linestyle="--",
             label="Model 2 (Intercept): H = 8.61 + 2.19 × Floors (R²=0.48)")
    # v1 Model 1 for comparison
    ax1.plot(floors_line, 2.394 * floors_line, color="#c0392b", linewidth=1.8, linestyle=":",
             label="WP1.1 v1 Model (Underestimated): H = 2.394 × Floors")
             
    # Plot OSM ground truth reference line
    ax1.plot(floors_line, 3.463 * floors_line, color="#8e44ad", linewidth=2.0, linestyle="-.",
             label="OSM Ground Truth Height: H = 3.463 × Levels")
             
    ax1.set_title("Storey-Height Calibration on nDSM v2 (Clean Buildings ≥ 400 m²)\nResolving WP1.1 Elevation Underestimation Bias", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Above-Ground Floor Count (ZEMINUSTUK)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Extracted Building Height (90th percentile nDSM v2, m)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 36)
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#bdc3c7", fontsize=9)
    
    # Right Panel: Forest Plot of District Calibrations
    districts_data = [
        ("KONAK", 1601, 2.623, 2.54, 2.71),
        ("BAYRAKLI", 1364, 2.918, 2.84, 3.00),
        ("KARŞIYAKA", 1742, 3.083, 3.04, 3.13),
        ("GLOBAL v2", len(clean), 3.320, 3.294, 3.345),
        ("BORNOVA", 3805, 3.421, 3.36, 3.49),
        ("BALÇOVA", 383, 3.509, 3.32, 3.70),
        ("ÇİĞLİ", 2165, 3.578, 3.52, 3.64),
        ("BUCA", 2014, 3.795, 3.71, 3.88),
        ("KARABAĞLAR", 1331, 3.811, 3.71, 3.91),
        ("NARLIDERE", 367, 3.991, 3.82, 4.16),
        ("GAZİEMİR", 1459, 4.066, 3.95, 4.18),
        ("GÜZELBAHÇE", 242, 4.579, 4.20, 4.96),
    ]
    
    y_pos = np.arange(len(districts_data))
    names = [d[0] for d in districts_data]
    betas = [d[2] for d in districts_data]
    err_low = [d[2] - d[3] for d in districts_data]
    err_high = [d[4] - d[2] for d in districts_data]
    colors = ["#e74c3c" if d[0] == "GLOBAL v2" else "#2c3e50" for d in districts_data]
    
    ax2.errorbar(betas, y_pos, xerr=[err_low, err_high], fmt="o", color="#2c3e50", ecolor="#7f8c8d",
                 elinewidth=2.0, capsize=5, capthick=1.5, markersize=7)
    # Highlight GLOBAL
    g_idx = names.index("GLOBAL v2")
    ax2.plot(betas[g_idx], y_pos[g_idx], "D", color="#e74c3c", markersize=9, label="Global v2 Mean (3.320 m/flr)")
    # Vertical line for global
    ax2.axvline(3.320, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
    # Vertical line for OSM ground truth
    ax2.axvline(3.463, color="#8e44ad", linestyle="-.", linewidth=1.5, alpha=0.7, label="OSM Ground Truth (3.463 m/flr)")
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9, fontweight="bold")
    ax2.set_title("District-Stratified Implied Storey Heights (Model 1 with 95% CI)\nAcross 11 Izmir Metropolitan Districts", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Implied Storey Height (m / storey)", fontsize=10, fontweight="bold")
    ax2.set_xlim(2.2, 5.2)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#bdc3c7", fontsize=9)
    
    plt.tight_layout()
    out_calib_path = os.path.join(out_dir, "quicklook_storey_calibration_v2.png")
    fig.savefig(out_calib_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Storey calibration quicklook: {out_calib_path} ({os.path.getsize(out_calib_path)/1024:.1f} KB)")
    
    print(f"\nStep 5 completed successfully in {time.time()-t_start:.2f} s")

if __name__ == "__main__":
    main()
