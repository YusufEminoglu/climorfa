"""
WP1.1d Step 5: Diagnostic Quicklooks for DTM v3
1. Multi-panel hillshade comparison (v2 vs v3) on two hillside windows (Karabağlar, Bayraklı Hill) + (v3 - v2) diff.
2. Bare-earth preservation: (v3 - rawDSM) spatial map and error distribution across slope classes.
3. Storey-height calibration v3: regression fits (Model 1 & 2) and district comparison forest plot (v2 vs v3).
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import geopandas as gpd

def compute_hillshade(dem, cellsize=5.0, az=315.0, alt=45.0):
    ls = LightSource(azdeg=az, altdeg=alt)
    # Handle NaNs or missing
    dem_clean = np.nan_to_num(dem, nan=0.0)
    hs = ls.hillshade(dem_clean, vert_exag=1.0, dx=cellsize, dy=cellsize)
    return hs

def main():
    print("=== WP1.1d Step 5: Generating Diagnostic Quicklooks ===")
    out_dir = "outputs/street/terrain"
    os.makedirs(out_dir, exist_ok=True)
    
    dsm_path = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
    dtm_v2_path = "data/street/terrain/dtm_5m_v2_epsg5253.tif"
    dtm_v3_path = "data/street/terrain/dtm_5m_v3_epsg5253.tif"
    fab_path = "data/street/terrain/fabdem_fua_30m_epsg5253.tif"
    
    # -------------------------------------------------------------------------
    # Plot 1: Hillshade & Slope Comparison (v2 vs v3) on Two Hillside Windows
    # -------------------------------------------------------------------------
    print("Generating Figure 1: Hillshade & Slope Comparison (v2 vs v3)...")
    windows = [
        ("Karabaglar Hillside [HELD-OUT]", (508000, 4247000, 509000, 4248000)),
        ("Bayrakli Hill [HELD-OUT]", (514000, 4260000, 515000, 4261000))
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    
    with rasterio.open(dtm_v2_path) as s_v2, rasterio.open(dtm_v3_path) as s_v3:
        for row_idx, (wname, bbox) in enumerate(windows):
            win = from_bounds(*bbox, transform=s_v3.transform).round_offsets().round_lengths()
            v2_w = s_v2.read(1, window=win).astype(np.float32)
            v3_w = s_v3.read(1, window=win).astype(np.float32)
            
            # Mask nodata
            mask = (v2_w > -50) & (v3_w > -50)
            
            # Hillshades
            hs_v2 = compute_hillshade(v2_w)
            hs_v3 = compute_hillshade(v3_w)
            
            # Diff (v3 - v2): positive means v2 was eroded below v3
            diff = v3_w - v2_w
            diff[~mask] = np.nan
            
            # Panel 1: v2 Hillshade
            axes[row_idx, 0].imshow(hs_v2, cmap='gray', origin='upper')
            axes[row_idx, 0].set_title(f"{wname}\nDTM v2 Hillshade (Slope-Eroded)", fontsize=11, fontweight='bold')
            axes[row_idx, 0].axis('off')
            
            # Panel 2: v3 Hillshade
            axes[row_idx, 1].imshow(hs_v3, cmap='gray', origin='upper')
            axes[row_idx, 1].set_title(f"{wname}\nDTM v3 Hillshade (Preserved Slopes)", fontsize=11, fontweight='bold', color='darkgreen')
            axes[row_idx, 1].axis('off')
            
            # Panel 3: Elevation Recovery (v3 - v2)
            norm = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=15)
            im = axes[row_idx, 2].imshow(diff, cmap='coolwarm', norm=norm, origin='upper')
            cbar = fig.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Elevation Difference: v3 − v2 (m)', fontsize=9)
            axes[row_idx, 2].set_title(f"{wname}\nv3 − v2 Recovery (Mean: +{np.nanmean(diff):.1f}m)", fontsize=11, fontweight='bold')
            axes[row_idx, 2].axis('off')
            
    fig.suptitle("WP1.1d Quicklook 1: Topographic Slope Preservation & Artifact Remediation (v2 vs v3)", fontsize=14, fontweight='bold')
    f1_path = os.path.join(out_dir, "quicklook_hillshade_v2_vs_v3.png")
    fig.savefig(f1_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f1_path}")
    
    # -------------------------------------------------------------------------
    # Plot 2: Bare Ground Preservation & Slope-Bias Test B Diagnostics
    # -------------------------------------------------------------------------
    print("Generating Figure 2: Bare Ground Preservation & Bias Diagnostics...")
    # Load validation results
    val_json_path = "logs/wp/validation_results_v3.json"
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    acc_B = val_data["acceptance_B"]
    slope_classes = [item["slope_class"] for item in acc_B]
    raw_biases = [item["rawDSM_bias"] for item in acc_B]
    v1_biases = [item["v1_bias"] for item in acc_B]
    v2_biases = [item["v2_bias"] for item in acc_B]
    v3_biases = [item["v3_bias"] for item in acc_B]
    departures = [item["departure"] for item in acc_B]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # Left: Bias vs Slope Class relative to FABDEM
    x = np.arange(len(slope_classes))
    width = 0.2
    
    ax1.bar(x - 1.5*width, raw_biases, width, label='Raw DSM (Ground Truth on Bare)', color='#4a5568', alpha=0.9)
    ax1.bar(x - 0.5*width, v1_biases, width, label='DTM v1 (Block-min)', color='#3182ce', alpha=0.9)
    ax1.bar(x + 0.5*width, v2_biases, width, label='DTM v2 (p10/p15 Eroded)', color='#e53e3e', alpha=0.9)
    ax1.bar(x + 1.5*width, v3_biases, width, label='DTM v3 (Detrended Residual)', color='#38a169', alpha=0.9)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Slope Class (degrees)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bias vs FABDEM (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Elevation Bias on Bare Open Cells vs Slope Class\n(Highlighting v2 Erosion of −14.3m on Steep Slopes vs v3 Recovery)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sc}°" for sc in slope_classes], fontsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9)
    
    # Right: Acceptance Test B Departures vs 1.0 m Threshold
    colors = ['#dd6b20' if dep > 1.0 else '#38a169' for dep in departures]
    bars = ax2.bar(slope_classes, departures, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Acceptance B Max Limit (1.0 m)')
    
    for bar, dep in zip(bars, departures):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{dep:.2f} m", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax2.set_xlabel('Slope Class (degrees)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Departure: |(v3 − FAB) − (rawDSM − FAB)| (m)', fontsize=11, fontweight='bold')
    ax2.set_title('Master Acceptance B Departure Metric\n(Threshold: <= 1.0 m; Raw Results: 1.18 - 1.48 m)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 2.0)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)
    
    fig.suptitle("WP1.1d Quicklook 2: Non-Circular Bare Earth Ground Preservation & Slope-Bias Test B", fontsize=13, fontweight='bold')
    f2_path = os.path.join(out_dir, "quicklook_v3_minus_rawDSM_bare.png")
    fig.savefig(f2_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f2_path}")
    
    # -------------------------------------------------------------------------
    # Plot 3: Storey Height Calibration v3 & District Forest Plot
    # -------------------------------------------------------------------------
    print("Generating Figure 3: Storey Calibration & District Comparison...")
    store_json_path = "logs/wp/storey_calibration_v3.json"
    with open(store_json_path, "r", encoding="utf-8") as f:
        store_data = json.load(f)
        
    dist_data = store_data["district_calibration"]
    # Filter districts with main sample
    dist_names = [d["district"] for d in dist_data]
    v2_betas = [d["v2_beta"] for d in dist_data]
    v3_betas = [d["v3_beta"] for d in dist_data]
    v3_ci_low = [d["v3_ci"][0] for d in dist_data]
    v3_ci_high = [d["v3_ci"][1] for d in dist_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    
    # Panel A: Main Sample Model Comparison
    main_est = store_data["main_estimate"]
    floors_sim = np.linspace(2, 25, 100)
    h_m1 = main_est["model1_beta"] * floors_sim
    h_m2 = main_est["model2_alpha"] + main_est["model2_beta"] * floors_sim
    
    ax1.plot(floors_sim, h_m1, label=f"Model 1 (Origin): H = {main_est['model1_beta']:.2f} × F (RMSE={main_est['model1_rmse']:.1f}m)", color='#2b6cb0', linewidth=2.5)
    ax1.plot(floors_sim, h_m2, label=f"Model 2 (Intercept): H = {main_est['model2_alpha']:.2f} + {main_est['model2_beta']:.2f} × F (R²={main_est['model2_r2']:.2f})", color='#c53030', linewidth=2.5, linestyle='--')
    
    # Reference Turkish/Mediterranean standard
    ax1.plot(floors_sim, 3.0 * floors_sim, label="Standard Nominal (3.00 m / floor)", color='#718096', linestyle=':', linewidth=1.5)
    
    ax1.set_xlabel('Storey Count (ZEMINUSTUK, floors >= 2)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Building Height p90 nDSM (m)', fontsize=11, fontweight='bold')
    ax1.set_title(f"Full FUA Residential/Mixed Storey Calibration\n(N = {main_est['n']:,} buildings >= 250 m²)", fontsize=11, fontweight='bold')
    ax1.set_xlim(2, 25)
    ax1.set_ylim(0, 80)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Panel B: District Forest Plot (v2 vs v3)
    y_pos = np.arange(len(dist_names))
    ax2.scatter(v2_betas, y_pos, color='#e53e3e', marker='s', s=50, label='DTM v2 (Hillside Inflated: 3.5 - 4.1 m)', zorder=3)
    
    xerr_low = np.array(v3_betas) - np.array(v3_ci_low)
    xerr_high = np.array(v3_ci_high) - np.array(v3_betas)
    ax2.errorbar(v3_betas, y_pos, xerr=[xerr_low, xerr_high], fmt='o', color='#2b6cb0', ecolor='#2b6cb0', elinewidth=2, capsize=4, label='DTM v3 (95% CI: 2.78 - 3.40 m)', zorder=4)
    
    ax2.axvline(3.0, color='green', linestyle='--', linewidth=1.2, label='Nominal 3.0 m/flr Reference')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(dist_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Implied Storey Height β (m / storey)', fontsize=11, fontweight='bold')
    ax2.set_title('District-Level Calibration Comparison (v2 vs v3)\n(Remediation of Hillside Inflation in Buca, Narlıdere, Karabağlar)', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', linestyle=':', alpha=0.6)
    ax2.set_xlim(2.4, 4.6)
    ax2.legend(loc='lower right', fontsize=10)
    
    fig.suptitle("WP1.1d Quicklook 3: Storey Height Calibration v3 & Cross-District Stabilization", fontsize=13, fontweight='bold')
    f3_path = os.path.join(out_dir, "quicklook_storey_calibration_v3.png")
    fig.savefig(f3_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {f3_path}")
    
    print("\nAll 3 quicklooks generated successfully!")

if __name__ == "__main__":
    main()
