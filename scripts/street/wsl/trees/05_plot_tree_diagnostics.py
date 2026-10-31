"""
WP1.3: Generate diagnostic figures for Tree DSM
Target: outputs/street/trees/
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

def main():
    out_dir = "outputs/street/trees"
    os.makedirs(out_dir, exist_ok=True)

    # Load validation summaries
    lcz_path = os.path.join(out_dir, "validation_by_lcz.csv")
    dist_path = os.path.join(out_dir, "validation_by_district.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics.json")

    if not os.path.exists(lcz_path) or not os.path.exists(dist_path):
        print("Validation tables not yet ready.")
        return

    df_lcz = pd.read_csv(lcz_path)
    df_dist = pd.read_csv(dist_path)
    with open(metrics_path) as fp:
        metrics = json.load(fp)

    # 1. LCZ & District Comparison Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel A: LCZ Canopy Cover
    ax = axes[0, 0]
    # Filter valid LCZ classes
    df_lcz_clean = df_lcz[df_lcz["n_cells"] >= 10].sort_values("lcz")
    x = np.arange(len(df_lcz_clean))
    w = 0.35
    ax.bar(x - w/2, df_lcz_clean["meta_cover_mean"] * 100, width=w, label="Meta 2m CHM", color="#2ca02c")
    ax.bar(x + w/2, df_lcz_clean["eth_cover_mean"] * 100, width=w, label="ETH 10m Canopy", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"LCZ {int(v)}" if v.is_integer() else f"LCZ {v}" for v in df_lcz_clean["lcz"]], rotation=45)
    ax.set_ylabel("Canopy Cover (%)")
    ax.set_title("A) Canopy Cover by Local Climate Zone (>2m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel B: LCZ Mean Height
    ax = axes[0, 1]
    ax.bar(x - w/2, df_lcz_clean["meta_h_mean"], width=w, label="Meta 2m CHM", color="#2ca02c")
    ax.bar(x + w/2, df_lcz_clean["eth_h_mean"], width=w, label="ETH 10m Canopy", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"LCZ {int(v)}" if v.is_integer() else f"LCZ {v}" for v in df_lcz_clean["lcz"]], rotation=45)
    ax.set_ylabel("Mean Canopy Top Height (m)")
    ax.set_title("B) Mean Canopy Height by Local Climate Zone")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel C: District Canopy Cover
    ax = axes[1, 0]
    df_dist_clean = df_dist[df_dist["n_cells"] >= 50].sort_values("meta_cover_mean", ascending=False)
    xd = np.arange(len(df_dist_clean))
    ax.bar(xd - w/2, df_dist_clean["meta_cover_mean"] * 100, width=w, label="Meta 2m CHM", color="#2ca02c")
    ax.bar(xd + w/2, df_dist_clean["eth_cover_mean"] * 100, width=w, label="ETH 10m Canopy", color="#1f77b4")
    ax.set_xticks(xd)
    ax.set_xticklabels(df_dist_clean["district"], rotation=45, ha="right")
    ax.set_ylabel("Canopy Cover (%)")
    ax.set_title("C) Canopy Cover by Core Urban District")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Panel D: Street Tree Sensitivity
    ax = axes[1, 1]
    categories = ["Urban-Core Overall", "Street Corridor (10m Buffer)"]
    meta_vals = [metrics["overall_meta_canopy_cover_pct"], metrics["street_tree_meta_canopy_share_pct"]]
    eth_vals = [metrics["overall_eth_canopy_cover_pct"], metrics["street_tree_eth_canopy_share_pct"]]
    xs = np.arange(len(categories))
    ax.bar(xs - w/2, meta_vals, width=w, label="Meta 2m CHM", color="#2ca02c")
    ax.bar(xs + w/2, eth_vals, width=w, label="ETH 10m Canopy", color="#1f77b4")
    ax.set_xticks(xs)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Canopy Share (%)")
    ax.set_title(f"D) Street-Tree Detection Sensitivity ({metrics['street_tree_meta_vs_eth_ratio']:.1f}x higher in Meta)")
    for i in xs:
        ax.annotate(f"{meta_vals[i]:.1f}%", (i - w/2, meta_vals[i] + 0.5), ha='center', fontweight='bold')
        ax.annotate(f"{eth_vals[i]:.1f}%", (i + w/2, eth_vals[i] + 0.5), ha='center', fontweight='bold')
    ax.set_ylim(0, max(meta_vals) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    p1 = os.path.join(out_dir, "canopy_validation_diagnostics.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Saved {p1}")

    # 2. Detailed Inset Figure: Kültürpark & Street Grid (Alsancak T_0512_4256)
    tile_tif = "data/street/tiles/2m/tree_dsm/T_0512_4256.tif"
    eth_path = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    if os.path.exists(tile_tif):
        with rasterio.open(tile_tif) as src_tile, rasterio.open(eth_path) as src_eth:
            bounds = src_tile.bounds
            win_eth = rasterio.windows.from_bounds(*bounds, src_eth.transform)
            meta_arr = src_tile.read(1)
            eth_arr = src_eth.read(1, window=win_eth, out_shape=meta_arr.shape)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            cmap = plt.cm.viridis.copy()
            cmap.set_under("lightgray")

            im1 = ax1.imshow(meta_arr, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                             vmin=2.0, vmax=25.0, cmap=cmap)
            ax1.set_title("Meta / WRI 2 m Tree DSM (T_0512_4256, Alsancak / Kültürpark)")
            ax1.set_xlabel("Easting (m)")
            ax1.set_ylabel("Northing (m)")
            plt.colorbar(im1, ax=ax1, label="Tree Height (m)")

            im2 = ax2.imshow(eth_arr, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                             vmin=2.0, vmax=25.0, cmap=cmap)
            ax2.set_title("ETH Global 10 m Canopy Height (Resampled)")
            ax2.set_xlabel("Easting (m)")
            ax2.set_ylabel("Northing (m)")
            plt.colorbar(im2, ax=ax2, label="Tree Height (m)")

            plt.tight_layout()
            p2 = os.path.join(out_dir, "kulturpark_alsancak_comparison.png")
            fig.savefig(p2, dpi=200)
            plt.close(fig)
            print(f"Saved {p2}")

if __name__ == "__main__":
    main()
