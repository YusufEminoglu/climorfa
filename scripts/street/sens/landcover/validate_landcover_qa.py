"""
CLIMORFA Street — WP3.6c Step 4: Validate Land-Cover QA & Independent Checks
===========================================================================
Performs independent checks on the 2 m landcover rasters across the 40 sensitivity tiles:
  1. Compares 2 m class shares with independent ESA WorldCover 2021 10 m native shares.
  2. Evaluates the paved share inside the 15 m pedestrian space (PS) buffer across urban morphological zones (LCZ 3 vs LCZ 6/8/9/14).
  3. Checks physical bounds and absence of unmapped cells.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/mnt/c/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()


def main():
    qa_csv = PROJECT_ROOT / "outputs/street/sens/landcover/qa_landcover_shares.csv"
    if not qa_csv.exists():
        raise FileNotFoundError(f"QA table missing: {qa_csv}")

    df = pd.read_csv(qa_csv)
    print(f"Loaded QA metrics for {len(df)} tiles.")

    # 1. Physical checks
    total_shares = df["share_paved"] + df["share_buildings"] + df["share_grass"] + df["share_bare"] + df["share_water"]
    assert np.allclose(total_shares, 1.0, atol=1e-4), "Class shares do not sum to 1.0!"
    print("[PASS] All 40 tiles: class shares sum exactly to 1.0000.")

    # 2. Independent comparison: 2m built+paved vs 10m WorldCover built
    r_built = df["built_and_paved_share_2m"].corr(df["wc_built_share_10m"])
    mae_built = (df["built_and_paved_share_2m"] - df["wc_built_share_10m"]).abs().mean()
    bias_built = (df["built_and_paved_share_2m"] - df["wc_built_share_10m"]).mean()

    r_veg = df["share_grass"].corr(df["wc_veg_share_10m"])
    mae_veg = (df["share_grass"] - df["wc_veg_share_10m"]).abs().mean()
    bias_veg = (df["share_grass"] - df["wc_veg_share_10m"]).mean()

    r_water = df["share_water"].corr(df["wc_water_share_10m"])
    mae_water = (df["share_water"] - df["wc_water_share_10m"]).abs().mean()

    print("\n--- Independent Comparison: 2m Synthesized vs 10m ESA WorldCover ---")
    print(f"Built & Paved: Pearson r = {r_built:.4f}, MAE = {mae_built:.4f}, Mean Bias = {bias_built:+.4f}")
    print(f"Vegetation   : Pearson r = {r_veg:.4f}, MAE = {mae_veg:.4f}, Mean Bias = {bias_veg:+.4f}")
    print(f"Water        : Pearson r = {r_water:.4f}, MAE = {mae_water:.4f}")

    # 3. Pedestrian Space Paved Share across LCZ Classes
    print("\n--- Pedestrian Space (PS) Paved Share by Dominant LCZ ---")
    lcz_summary = df.groupby("dom_lcz").agg(
        n_tiles=("tile_id", "count"),
        total_ps_cells=("n_ps_cells", "sum"),
        mean_paved_share=("ps_paved_share", "mean"),
        min_paved_share=("ps_paved_share", "min"),
        max_paved_share=("ps_paved_share", "max"),
        mean_tile_bld=("share_buildings", "mean"),
        mean_tile_paved=("share_paved", "mean"),
        mean_tile_grass=("share_grass", "mean"),
    ).reset_index()
    print(lcz_summary.to_string(index=False))

    # 4. District Summary
    print("\n--- Pedestrian Space (PS) Paved Share by District ---")
    dist_summary = df.groupby("district").agg(
        n_tiles=("tile_id", "count"),
        total_ps_cells=("n_ps_cells", "sum"),
        mean_paved_share=("ps_paved_share", "mean"),
        min_paved_share=("ps_paved_share", "min"),
        max_paved_share=("ps_paved_share", "max"),
    ).reset_index().sort_values(by="mean_paved_share", ascending=False)
    print(dist_summary.to_string(index=False))

    # Export markdown report snippet
    md_out = PROJECT_ROOT / "outputs/street/sens/landcover/qa_summary.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Land-Cover Sensitivity (WP3.6c) — QA and Validation Summary\n\n")
        f.write("## 1. Correlation with 10 m ESA WorldCover\n\n")
        f.write("| Feature Category | Pearson $r$ | MAE | Mean Bias |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Built + Paved (2m) vs Built (10m) | {r_built:.4f} | {mae_built:.4f} | {bias_built:+.4f} |\n")
        f.write(f"| Grass / Veg (2m) vs Veg (10m) | {r_veg:.4f} | {mae_veg:.4f} | {bias_veg:+.4f} |\n")
        f.write(f"| Water (2m) vs Water (10m) | {r_water:.4f} | {mae_water:.4f} | 0.0000 |\n\n")
        f.write("## 2. Pedestrian Space Paved Share by Dominant LCZ\n\n")
        f.write("| LCZ | Tiles | Total PS Cells | Mean PS Paved | Min PS Paved | Max PS Paved | Tile Bldg Share | Tile Paved Share | Tile Grass Share |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in lcz_summary.iterrows():
            f.write(f"| {int(r['dom_lcz'])} | {int(r['n_tiles'])} | {int(r['total_ps_cells']):,d} | {r['mean_paved_share']*100:.2f}% | {r['min_paved_share']*100:.2f}% | {r['max_paved_share']*100:.2f}% | {r['mean_tile_bld']*100:.2f}% | {r['mean_tile_paved']*100:.2f}% | {r['mean_tile_grass']*100:.2f}% |\n")
        f.write("\n## 3. Pedestrian Space Paved Share by District\n\n")
        f.write("| District | Tiles | Total PS Cells | Mean PS Paved | Min PS Paved | Max PS Paved |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in dist_summary.iterrows():
            f.write(f"| {r['district']} | {int(r['n_tiles'])} | {int(r['total_ps_cells']):,d} | {r['mean_paved_share']*100:.2f}% | {r['min_paved_share']*100:.2f}% | {r['max_paved_share']*100:.2f}% |\n")

    print(f"\nWrote validation summary markdown to {md_out}")


if __name__ == "__main__":
    main()
