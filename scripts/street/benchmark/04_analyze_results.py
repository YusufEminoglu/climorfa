"""
CLIMORFA Street — WP0.4 Results Analysis & Extrapolation
Performs sanity checks, distribution analysis, and runtime extrapolation:
1. Inner 500x500 m analysis at 15:00 (Tmrt, UTCI, SVF, sunlit vs shaded).
2. Physical plausibility checks (flag Tmrt < Ta - 10 or > 85 C).
3. Urban-core extrapolation for 2 days x 24 h (48 timesteps) at 1 m and 2 m (with and without buffer).
4. Generates publication-ready multi-panel quicklook figure.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_analysis():
    project_root = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
    out_dir = os.path.join(project_root, "outputs/street/benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load benchmark metrics
    metrics_path = os.path.join(out_dir, "benchmark_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        bench_metrics = json.load(f)

    # 2. Coordinates & Inner Window
    xmin_inner, ymin_inner = 512350.0, 4255850.0
    xmax_inner, ymax_inner = 512850.0, 4256350.0
    xmin_buf, ymax_buf = 512250.0, 4256450.0

    # Ta at 15:00
    ta_1500 = 36.0  # Celsius

    analysis_results = {}

    for res_str, res_val in [("2m", 2.0), ("1m", 1.0)]:
        run_folder = os.path.join(out_dir, f"run_{res_str}")
        out_0_0 = os.path.join(run_folder, "output_folder/0_0")
        svf_folder = os.path.join(run_folder, "processed_inputs/SVF")

        tmrt_path = os.path.join(out_0_0, "TMRT_0_0.tif")
        utci_path = os.path.join(out_0_0, "UTCI_0_0.tif")
        shadow_path = os.path.join(out_0_0, "Shadow_0_0.tif")
        svf_path = os.path.join(svf_folder, "SkyViewFactor_0_0.tif")

        # Read 15:00 band (band 16, 1-based index)
        with rasterio.open(tmrt_path) as src:
            tmrt_full = src.read(16)
        with rasterio.open(utci_path) as src:
            utci_full = src.read(16)
        with rasterio.open(shadow_path) as src:
            shadow_full = src.read(16)
        with rasterio.open(svf_path) as src:
            svf_full = src.read(1)

        # Slice inner 500x500 m
        r_start = int(round((ymax_buf - ymax_inner) / res_val))
        r_end = r_start + int(round(500.0 / res_val))
        c_start = int(round((xmin_inner - xmin_buf) / res_val))
        c_end = c_start + int(round(500.0 / res_val))

        tmrt = tmrt_full[r_start:r_end, c_start:c_end]
        utci = utci_full[r_start:r_end, c_start:c_end]
        shadow = shadow_full[r_start:r_end, c_start:c_end]
        svf = svf_full[r_start:r_end, c_start:c_end]

        # Valid outdoor mask (where UTCI is not NaN, meaning not inside buildings)
        outdoor_mask = ~np.isnan(utci)
        sunlit_mask = outdoor_mask & (shadow > 0.5)
        shaded_mask = outdoor_mask & (shadow <= 0.5)

        # Physical plausibility check
        # Flag if Tmrt < Ta - 10 (26.0 C) or Tmrt > 85.0 C on outdoor pixels
        tmrt_outdoor = tmrt[outdoor_mask]
        implausible_low = np.sum(tmrt_outdoor < (ta_1500 - 10.0))
        implausible_high = np.sum(tmrt_outdoor > 85.0)

        def calc_percentiles(arr):
            if len(arr) == 0:
                return {"p05": None, "p50": None, "p95": None, "mean": None, "std": None, "count": 0}
            return {
                "p05": round(float(np.percentile(arr, 5)), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "count": int(len(arr))
            }

        res_data = {
            "resolution": res_str,
            "inner_shape": list(tmrt.shape),
            "total_pixels": tmrt.size,
            "outdoor_pixels": int(np.sum(outdoor_mask)),
            "outdoor_share_pct": round(float(100.0 * np.sum(outdoor_mask) / tmrt.size), 2),
            "sunlit_pixels": int(np.sum(sunlit_mask)),
            "shaded_pixels": int(np.sum(shaded_mask)),
            "physical_checks": {
                "ta_1500_c": ta_1500,
                "implausible_low_count (Tmrt < 26 C)": int(implausible_low),
                "implausible_high_count (Tmrt > 85 C)": int(implausible_high),
                "status": "PASSED" if (implausible_low == 0 and implausible_high == 0) else "FLAGGED"
            },
            "tmrt_outdoor_all": calc_percentiles(tmrt_outdoor),
            "tmrt_sunlit": calc_percentiles(tmrt[sunlit_mask]),
            "tmrt_shaded": calc_percentiles(tmrt[shaded_mask]),
            "utci_outdoor_all": calc_percentiles(utci[outdoor_mask]),
            "utci_sunlit": calc_percentiles(utci[sunlit_mask]),
            "utci_shaded": calc_percentiles(utci[shaded_mask]),
            "svf_outdoor": calc_percentiles(svf[outdoor_mask]),
            "svf_all": calc_percentiles(svf.flatten()),
        }
        analysis_results[res_str] = res_data

        print(f"\n=== Inner 500 m Sanity Checks ({res_str}) at 15:00 ===")
        print(f"Outdoor share: {res_data['outdoor_share_pct']}% ({res_data['outdoor_pixels']} px)")
        print(f"Sunlit: {res_data['sunlit_pixels']} px | Shaded: {res_data['shaded_pixels']} px")
        print(f"Tmrt Sunlit (5/50/95): {res_data['tmrt_sunlit']['p05']} / {res_data['tmrt_sunlit']['p50']} / {res_data['tmrt_sunlit']['p95']} °C")
        print(f"Tmrt Shaded (5/50/95): {res_data['tmrt_shaded']['p05']} / {res_data['tmrt_shaded']['p50']} / {res_data['tmrt_shaded']['p95']} °C")
        print(f"UTCI Sunlit (5/50/95): {res_data['utci_sunlit']['p05']} / {res_data['utci_sunlit']['p50']} / {res_data['utci_sunlit']['p95']} °C")
        print(f"UTCI Shaded (5/50/95): {res_data['utci_shaded']['p05']} / {res_data['utci_shaded']['p50']} / {res_data['utci_shaded']['p95']} °C")
        print(f"SVF Outdoor (5/50/95): {res_data['svf_outdoor']['p05']} / {res_data['svf_outdoor']['p50']} / {res_data['svf_outdoor']['p95']}")
        print(f"Physical plausibility check: {res_data['physical_checks']['status']} (low={implausible_low}, high={implausible_high})")

    # 3. Area A Extrapolation
    print("\n=== Urban Core Area A Extrapolation ===")
    features_csv = os.path.join(project_root, "data/03_processed/grid_250m_model_features_v8.csv")
    grids_gpkg = os.path.join(project_root, "data/03_processed/analysis_grids.gpkg")

    df_feats = pd.read_csv(features_csv)
    gdf_grids = gpd.read_file(grids_gpkg, layer="grid_250m")
    merged = gdf_grids.merge(df_feats[["grid_id", "building_coverage_exact"]], on="grid_id")

    core_cells = merged[merged["building_coverage_exact"] > 0.05]
    n_core_cells = len(core_cells)
    area_core_m2 = n_core_cells * 250.0 * 250.0  # exactly 62,500 m2 per cell
    area_core_km2 = area_core_m2 / 1e6

    print(f"Core cells (building_coverage_exact > 0.05): {n_core_cells}")
    print(f"Total urban-core area A: {area_core_m2:,.0f} m² ({area_core_km2:.2f} km²)")

    # Tile sizing
    tile_inner_m2 = 500.0 * 500.0  # 0.25 km2
    tile_buf_m2 = 700.0 * 700.0    # 0.49 km2
    n_tiles_500m = area_core_m2 / tile_inner_m2  # 911.75 tiles

    # Scaling formulas
    # For a simulation of 2 days x 24 h = 48 timesteps:
    # T_total = T_setup (preprocess + walls + SVF) + 48 * T_hourly_step
    extrapolations = {}

    for res_str in ["2m", "1m"]:
        timings = bench_metrics[f"run_{res_str}"]["timings"]
        t_setup_buffered = timings["preprocess_s"] + timings["walls_aspect_s"] + timings["svf_s"]
        t_hourly_step_buffered = timings["time_per_hourly_step_s"]
        t_48h_buffered_single_tile = t_setup_buffered + 48 * t_hourly_step_buffered

        # Area A with 100m buffers (912 tiles of 700x700m compute area)
        total_time_buffered_s = n_tiles_500m * t_48h_buffered_single_tile
        total_time_buffered_h = total_time_buffered_s / 3600.0

        # Without buffers: pixel count is (500/700)^2 = 25/49 = 0.5102x of buffered tile
        pixel_ratio_nobuf = (500.0 / 700.0) ** 2
        t_setup_nobuf = t_setup_buffered * pixel_ratio_nobuf
        t_hourly_nobuf = t_hourly_step_buffered * pixel_ratio_nobuf
        t_48h_nobuf_single_tile = t_setup_nobuf + 48 * t_hourly_nobuf

        total_time_nobuf_s = n_tiles_500m * t_48h_nobuf_single_tile
        total_time_nobuf_h = total_time_nobuf_s / 3600.0

        extrapolations[res_str] = {
            "single_tile_setup_buffered_s": round(t_setup_buffered, 2),
            "single_tile_hourly_step_buffered_s": round(t_hourly_step_buffered, 3),
            "single_tile_48h_buffered_s": round(t_48h_buffered_single_tile, 2),
            "urban_core_48h_buffered_hours": round(total_time_buffered_h, 2),
            "urban_core_48h_buffered_days": round(total_time_buffered_h / 24.0, 2),
            "single_tile_48h_nobuf_s": round(t_48h_nobuf_single_tile, 2),
            "urban_core_48h_nobuf_hours": round(total_time_nobuf_h, 2),
            "urban_core_48h_nobuf_days": round(total_time_nobuf_h / 24.0, 2),
        }
        print(f"[{res_str}] Single tile 48h (buffered): {t_48h_buffered_single_tile:.1f} s")
        print(f"[{res_str}] Full Urban Core A (48h, WITH 100m buffers): {total_time_buffered_h:.2f} hours ({total_time_buffered_h/24.0:.2f} days)")
        print(f"[{res_str}] Full Urban Core A (48h, WITHOUT buffers):   {total_time_nobuf_h:.2f} hours ({total_time_nobuf_h/24.0:.2f} days)")

    extrapolation_meta = {
        "urban_core_area_m2": area_core_m2,
        "urban_core_area_km2": area_core_km2,
        "core_cell_count": n_core_cells,
        "tile_500m_count": round(n_tiles_500m, 2),
        "simulation_timesteps": 48,
        "extrapolations": extrapolations,
    }

    # Save summary JSON
    summary_data = {
        "benchmark_metrics": bench_metrics,
        "sanity_checks": analysis_results,
        "extrapolation": extrapolation_meta,
    }
    summary_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved analysis summary to: {summary_path}")

    # 4. Generate Quicklook PNG
    generate_quicklook_figure(out_dir, xmin_buf, ymax_buf, xmin_inner, ymin_inner, xmax_inner, ymax_inner)

def generate_quicklook_figure(out_dir, xmin_buf, ymax_buf, xmin_inner, ymin_inner, xmax_inner, ymax_inner):
    """Plot multi-panel quicklook of Tmrt, UTCI, SVF, and Shadow at 15:00."""
    print("\n=== Generating Quicklook PNG ===")
    run_1m = os.path.join(out_dir, "run_1m")
    out_0_0 = os.path.join(run_1m, "output_folder/0_0")
    svf_folder = os.path.join(run_1m, "processed_inputs/SVF")

    with rasterio.open(os.path.join(out_0_0, "TMRT_0_0.tif")) as src:
        tmrt = src.read(16)
    with rasterio.open(os.path.join(out_0_0, "UTCI_0_0.tif")) as src:
        utci = src.read(16)
    with rasterio.open(os.path.join(out_0_0, "Shadow_0_0.tif")) as src:
        shadow = src.read(16)
    with rasterio.open(os.path.join(svf_folder, "SkyViewFactor_0_0.tif")) as src:
        svf = src.read(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), dpi=150)

    # 1. Tmrt
    ax1 = axes[0, 0]
    im1 = ax1.imshow(tmrt, cmap="turbo", vmin=30, vmax=75, origin="upper")
    ax1.set_title("Mean Radiant Temperature ($T_{mrt}$) @ 15:00 LST [°C]\n(1 m resolution, central Alsancak)", fontsize=11, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("$T_{mrt}$ (°C)")

    # 2. UTCI
    ax2 = axes[0, 1]
    # Mask building NaNs as light grey
    utci_plot = np.ma.masked_invalid(utci)
    cmap_utci = plt.colormaps["YlOrRd"].copy()
    cmap_utci.set_bad(color="lightgrey")
    im2 = ax2.imshow(utci_plot, cmap=cmap_utci, vmin=32, vmax=43, origin="upper")
    ax2.set_title("Universal Thermal Climate Index (UTCI) @ 15:00 LST [°C]\n(Outdoor canyon pixels; buildings masked grey)", fontsize=11, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("UTCI (°C)")

    # 3. SVF
    ax3 = axes[1, 0]
    im3 = ax3.imshow(svf, cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
    ax3.set_title("Sky View Factor (SVF) [0 - 1]\n(Vegetation + Building obstructions accounted)", fontsize=11, fontweight="bold")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("SVF [-]")

    # 4. Shadow Map
    ax4 = axes[1, 1]
    im4 = ax4.imshow(shadow, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax4.set_title("Shadow Classification @ 15:00 LST\n(0 = Shadow [Black], 1 = Sunlit [White])", fontsize=11, fontweight="bold")
    cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar4.ax.set_yticklabels(["Shadow", "Sunlit"])

    # Draw inner 500x500 m boundary on all panels
    # Inner box in 1m pixel coordinates: [100, 100] to [600, 600]
    for ax in [ax1, ax2, ax3, ax4]:
        rect = patches.Rectangle((100, 100), 500, 500, linewidth=2.0, edgecolor="red", facecolor="none", linestyle="--", label="Inner 500 m Evaluation Tile")
        ax.add_patch(rect)
        ax.set_xlabel("Easting pixel offset (1 m/px)")
        ax.set_ylabel("Northing pixel offset (1 m/px)")
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)

    plt.suptitle("CLIMORFA Street — WP0.4 SOLWEIG-GPU Benchmark Quicklook\nAlsancak Tile (700x700 m with 100 m buffer) — 2025-07-15 15:00 LST", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    quicklook_png = os.path.join(out_dir, "quicklook_tmrt_svf_benchmark.png")
    plt.savefig(quicklook_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Quicklook saved: {quicklook_png}")

if __name__ == "__main__":
    run_analysis()
