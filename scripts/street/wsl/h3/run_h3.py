"""CLIMORFA Street — Work Package 3.3 (H3: surface temperature vs pedestrian heat stress).

Pre-analysis lock (§2 H3):
- Landsat summer LST: median over PS cells per segment, and per 250 m cell.
- Spearman rho(LST, M3) and rho(LST, M1) overall and by LCZ, with 2 km spatial block-bootstrap CIs (1000 resamples).
- Bivariate local Moran (LST vs M1) at 250 m.
- 'Decoupled' segments = top LST tercile and bottom M1 tercile (and vice versa). Report shares by LCZ and audit class.
- The compact-paradox decision rule: counts as observed only if rho(LST, M1) < 0 within LCZ 2/3 with 95% CI excluding 0.
- Every output carries 'PARTIAL (n tiles)' until n = 427.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd

from scripts.street.integrity.synthetic import assert_real
from scripts.street.wsl.h3.bootstrap import (
    SEED,
    N_BOOT,
    assign_2km_blocks,
    run_spearman_analysis,
    evaluate_compact_paradox_rule,
)
from scripts.street.wsl.h3.decoupling import (
    identify_decoupled_segments,
    compute_decoupled_shares_by_lcz,
    load_manual_audit_table,
    link_segments_to_audit,
    compute_decoupled_shares_by_audit,
)
from scripts.street.wsl.h3.moran import (
    aggregate_m1_to_grid250,
    run_bivariate_moran_250m,
)
from scripts.street.wsl.h3.plotting import (
    plot_lst_vs_m1_scatter,
    plot_spearman_forest,
    plot_moran_bivariate_map,
    plot_decoupled_shares,
)

DAYS = ("IMBAT", "CALM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP3.3 — H3: Surface vs Pedestrian Heat Stress Runner")
    parser.add_argument("--segments-path", type=str, default="data/street/analysis/segments.gpkg",
                        help="Path to segments.gpkg")
    parser.add_argument("--metrics-dir", type=str, default="data/street/analysis",
                        help="Directory containing seg_metrics_<DAY>.parquet")
    parser.add_argument("--grid250-path", type=str, default="data/street/analysis/grid_250.gpkg",
                        help="Path to grid_250.gpkg")
    parser.add_argument("--audit-csv", type=str,
                        default="data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv",
                        help="Path to manual audit sheet CSV")
    parser.add_argument("--audit-gpkg", type=str,
                        default="data/04_training_labels/audit_sample_v1.gpkg",
                        help="Path to manual audit GPKG")
    parser.add_argument("--out-dir", type=str, default="outputs/street/analysis/h3",
                        help="Output directory for H3 results")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on synthetic test data in out-dir/synthetic_test")
    parser.add_argument("--poll", action="store_true",
                        help="Poll for missing input files periodically")
    parser.add_argument("--poll-interval", type=int, default=1800,
                        help="Polling interval in seconds (default: 1800 = 30 min)")
    parser.add_argument("--poll-timeout", type=int, default=72000,
                        help="Polling timeout in seconds (default: 72000 = 20 h)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Number of block-bootstrap resamples (default: {N_BOOT})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    return parser.parse_args()


def poll_for_inputs(
    seg_path: Path,
    metrics_dir: Path,
    interval_sec: int = 1800,
    timeout_sec: int = 72000,
) -> bool:
    """Poll every interval_sec until segments and at least one metric file exist or timeout is reached."""
    start_time = time.time()
    print(f"Polling for input files every {interval_sec // 60} min (timeout {timeout_sec // 3600} h)...")

    while True:
        elapsed = time.time() - start_time
        metrics_found = [d for d in DAYS if (metrics_dir / f"seg_metrics_{d}.parquet").exists()]
        if seg_path.exists() and len(metrics_found) > 0:
            print(f"Inputs detected after {elapsed / 60:.1f} min! Days found: {metrics_found}")
            return True

        if elapsed >= timeout_sec:
            print(f"Polling timed out after {elapsed / 3600:.1f} h without inputs.")
            return False

        remaining = timeout_sec - elapsed
        wait = min(interval_sec, remaining)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for inputs ({seg_path.name}, seg_metrics_*.parquet)... Elapsed: {elapsed/60:.1f} min. Sleeping {wait}s.")
        time.sleep(wait)


def run_h3_analysis(
    segments_path: Path,
    metrics_paths: dict[str, Path],
    grid250_path: Optional[Path],
    audit_csv: Path,
    audit_gpkg: Path,
    out_dir: Path,
    is_synthetic: bool = False,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Execute WP3.3 analysis end-to-end across available design days."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Integrity assertions (Rule 6)
    if not is_synthetic:
        assert_real(segments_path)
        for d, mp in metrics_paths.items():
            assert_real(mp)
        if grid250_path and grid250_path.exists():
            assert_real(grid250_path)
    else:
        print("Note: Running in SYNTHETIC mode (is_synthetic=True).")

    # 2. Load segments
    print(f"Loading segments from {segments_path}...")
    if segments_path.suffix == ".parquet":
        df_segs = pd.read_parquet(segments_path)
    else:
        gdf_segs = gpd.read_file(segments_path, layer="segments" if "segments" in gpd.list_layers(segments_path)["name"].values else None)
        mid = gdf_segs.geometry.interpolate(0.5, normalized=True)
        df_segs = pd.DataFrame(gdf_segs.drop(columns="geometry"))
        df_segs["x_mid"] = mid.x.to_numpy()
        df_segs["y_mid"] = mid.y.to_numpy()

    # Assign 2 km blocks if not present
    if "block_2k" not in df_segs.columns:
        df_segs = assign_2km_blocks(df_segs)

    # 3. Load manual audit table (569 points)
    print(f"Loading manual audit table from {audit_csv} and {audit_gpkg}...")
    try:
        audit_table = load_manual_audit_table(audit_csv, audit_gpkg)
        print(f"Loaded {len(audit_table)} audit grid cells with calculated g250_id.")
    except Exception as e:
        print(f"Warning: Could not load manual audit table ({e}). Decoupling by audit class will be skipped.")
        audit_table = None

    # 4. Load 250 m grid if available (or fallback to analysis_grids.gpkg)
    gdf_grid250 = None
    if grid250_path and grid250_path.exists():
        print(f"Loading 250 m grid from {grid250_path}...")
        try:
            gdf_grid250 = gpd.read_file(grid250_path, layer="grid" if "grid" in gpd.list_layers(grid250_path)["name"].values else None)
            print(f"Loaded {len(gdf_grid250)} grid cells from grid_250.gpkg.")
        except Exception as e:
            print(f"Warning: Could not load grid_250.gpkg ({e}).")
    if gdf_grid250 is None:
        fallback_grid_path = Path("data/03_processed/analysis_grids.gpkg")
        if fallback_grid_path.exists():
            print(f"Using 250 m grid from fallback {fallback_grid_path}...")
            try:
                gdf_base = gpd.read_file(fallback_grid_path, layer="grid_250m")
                b = gdf_base.geometry.bounds
                gdf_base["g250_id"] = (b["minx"].astype(int) // 250).astype(str) + "_" + (b["miny"].astype(int) // 250).astype(str)
                # Compute lst_med from segments if available
                if "lst_med" in df_segs.columns and "g250_id" in df_segs.columns:
                    cell_lst = df_segs.groupby("g250_id")["lst_med"].median().reset_index()
                    gdf_base = gdf_base.merge(cell_lst, on="g250_id", how="left")
                gdf_grid250 = gdf_base
                print(f"Loaded {len(gdf_grid250)} 250 m grid cells from analysis_grids.gpkg.")
            except Exception as e:
                print(f"Warning: Could not load fallback grid ({e}). Moran analysis will be skipped.")

    master_summary = {
        "analysis": "CLIMORFA Street WP3.3 (H3: surface vs pedestrian heat stress)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "is_synthetic": is_synthetic,
        "n_boot": n_boot,
        "seed": seed,
        "days_analyzed": [],
        "results_by_day": {},
    }

    # 5. Process each day
    for day, met_path in metrics_paths.items():
        if not met_path.exists():
            print(f"Skipping {day}: metrics file {met_path} does not exist.")
            continue

        print(f"\n==========================================")
        print(f"Analyzing {day} (metrics: {met_path.name})")
        print(f"==========================================")

        df_met = pd.read_parquet(met_path)
        merged = df_segs.merge(df_met, on="seg_id", how="inner", suffixes=("", "_m"))

        # Map M1 and M3
        merged["m1"] = merged["utci_b15"]
        merged["m3"] = merged["tmrt_b15"]

        # Determine partial status and tile count
        is_partial = bool(merged["partial_run"].any()) if "partial_run" in merged.columns else True
        n_tiles = int(merged["n_tiles_at_extraction"].max()) if "n_tiles_at_extraction" in merged.columns else len(merged["tile_id"].unique())

        if is_synthetic:
            label_str = f"SYNTHETIC (planted-effect test, not observed data)"
        elif is_partial or n_tiles < 427:
            label_str = f"PARTIAL ({n_tiles} tiles)"
        else:
            label_str = f"FINAL ({n_tiles} tiles)"

        print(f"Dataset status: {label_str} — {len(merged)} segments merged.")

        # --- A. Spearman Correlation Analysis ---
        print("\n[1/4] Running Spearman correlation analysis with 2 km spatial block bootstrap...")
        # Primary LCZ
        spearman_primary = run_spearman_analysis(
            merged, lcz_col="lcz_primary", pool=True, n_boot=n_boot, seed=seed
        )
        spearman_primary["day"] = day
        nodata_primary = spearman_primary.attrs.get("lcz_nodata_excluded", {})
        print(f"A6: LCZ no-data / missing excluded from by-LCZ statistics (kept in overall): {nodata_primary}")
        spearman_primary["data_status"] = label_str
        spearman_primary["lcz_source"] = "lcz_primary"

        # Sensitivity: weak 250 m labels if present
        spearman_weak = None
        if "lcz_weak250" in merged.columns and merged["lcz_weak250"].notna().sum() > 30:
            spearman_weak = run_spearman_analysis(
                merged, lcz_col="lcz_weak250", pool=True, n_boot=n_boot, seed=seed
            )
            spearman_weak["day"] = day
            spearman_weak["data_status"] = label_str
            spearman_weak["lcz_source"] = "lcz_weak250"
            spearman_all = pd.concat([spearman_primary, spearman_weak], ignore_index=True)
        else:
            spearman_all = spearman_primary

        spearman_out_path = out_dir / f"h3_spearman_{day}.csv"
        spearman_all.to_csv(spearman_out_path, index=False)
        print(f"Saved Spearman correlations to {spearman_out_path}")

        # --- B. Compact Paradox Decision Rule ---
        print("\n[2/4] Evaluating Compact Paradox decision rule on LCZ 2/3...")
        paradox_res = evaluate_compact_paradox_rule(merged, lcz_col="lcz_primary", n_boot=n_boot, seed=seed)
        paradox_res["day"] = day
        paradox_res["data_status"] = label_str

        paradox_out_path = out_dir / f"h3_compact_paradox_{day}.json"
        with open(paradox_out_path, "w", encoding="utf-8") as f:
            json.dump(paradox_res, f, indent=2)

        print(f"Decision rule: {paradox_res['decision']}")
        print(f"  Target: {paradox_res['target_class']} (n={paradox_res['n_segments']} segs, {paradox_res['n_blocks']} blocks)")
        print(f"  rho(LST, M1) = {paradox_res['rho']:.3f} [95% CI: {paradox_res['ci95_lower']:.3f}, {paradox_res['ci95_upper']:.3f}]")
        print(f"  Compact Paradox observed: {paradox_res['compact_paradox_observed']}")

        # --- C. Decoupled Segments Analysis ---
        print("\n[3/4] Identifying decoupled segments and computing shares by LCZ & audit class...")
        df_dec = identify_decoupled_segments(merged)

        # Shares by LCZ
        lcz_shares = compute_decoupled_shares_by_lcz(df_dec, lcz_col="lcz_primary", pool=True)
        lcz_shares["day"] = day
        lcz_shares["data_status"] = label_str
        lcz_shares_out = out_dir / f"h3_decoupled_lcz_{day}.csv"
        lcz_shares.to_csv(lcz_shares_out, index=False)
        print(f"Saved LCZ decoupled shares to {lcz_shares_out}")

        # Shares by Audit Class
        audit_shares = None
        if audit_table is not None:
            df_dec_audit = link_segments_to_audit(df_dec, audit_table)
            audit_shares = compute_decoupled_shares_by_audit(df_dec_audit)
            audit_shares["day"] = day
            audit_shares["data_status"] = label_str
            audit_shares_out = out_dir / f"h3_decoupled_audit_{day}.csv"
            audit_shares.to_csv(audit_shares_out, index=False)
            print(f"Saved audit decoupled shares to {audit_shares_out}")

        # --- D. Bivariate Local Moran (250 m) ---
        moran_summary = {}
        if gdf_grid250 is not None:
            print("\n[4/4] Running Bivariate Local Moran (LST vs M1) at 250 m...")
            grid_annotated = aggregate_m1_to_grid250(merged, gdf_grid250)
            res_grid, moran_summary = run_bivariate_moran_250m(grid_annotated, seed=seed)
            moran_summary["day"] = day
            moran_summary["data_status"] = label_str

            moran_summary_out = out_dir / f"h3_moran_250m_summary_{day}.json"
            with open(moran_summary_out, "w", encoding="utf-8") as f:
                json.dump(moran_summary, f, indent=2)

            # Save cells table
            cells_out = out_dir / f"h3_moran_250m_cells_{day}.parquet"
            # Drop geometry for parquet or save as gpkg/parquet
            df_cells = pd.DataFrame(res_grid.drop(columns="geometry"))
            df_cells.to_parquet(cells_out, index=False)
            print(f"Global Bivariate Moran's I: {moran_summary.get('global_bivariate_moran_I', float('nan')):.3f} "
                  f"(p={moran_summary.get('global_bivariate_moran_p_sim', float('nan')):.3f})")
            print(f"Saved Moran summary to {moran_summary_out}")

            # Plot Moran Map
            moran_map_path = out_dir / f"fig_h3_moran_bivariate_map_{day}.png"
            plot_moran_bivariate_map(res_grid, moran_summary, day, label_str, moran_map_path, is_synthetic=is_synthetic)
            print(f"Saved draft Moran map to {moran_map_path}")

        # --- E. Draft Figures ---
        fig_scatter_path = out_dir / f"fig_h3_lst_vs_m1_scatter_{day}.png"
        plot_lst_vs_m1_scatter(df_dec, day, label_str, fig_scatter_path, is_synthetic=is_synthetic)
        print(f"Saved draft scatter figure to {fig_scatter_path}")

        fig_forest_path = out_dir / f"fig_h3_spearman_by_lcz_{day}.png"
        plot_spearman_forest(spearman_primary, day, label_str, fig_forest_path, is_synthetic=is_synthetic)
        print(f"Saved draft forest plot to {fig_forest_path}")

        if audit_shares is not None:
            fig_shares_path = out_dir / f"fig_h3_decoupled_shares_{day}.png"
            plot_decoupled_shares(lcz_shares, audit_shares, day, label_str, fig_shares_path, is_synthetic=is_synthetic)
            print(f"Saved draft decoupled shares plot to {fig_shares_path}")

        # Store in master summary
        overall_m1_row = spearman_primary[
            (spearman_primary["scope"] == "overall") & (spearman_primary["metric_ped"].str.contains("M1"))
        ].iloc[0]
        overall_m3_row = spearman_primary[
            (spearman_primary["scope"] == "overall") & (spearman_primary["metric_ped"].str.contains("M3"))
        ].iloc[0]

        master_summary["days_analyzed"].append(day)
        master_summary["results_by_day"][day] = {
            "data_status": label_str,
            "n_tiles": n_tiles,
            "n_segments": len(merged),
            "spearman_overall_m1": {
                "rho": float(overall_m1_row["rho"]),
                "ci95": [float(overall_m1_row["ci95_lower"]), float(overall_m1_row["ci95_upper"])],
            },
            "spearman_overall_m3": {
                "rho": float(overall_m3_row["rho"]),
                "ci95": [float(overall_m3_row["ci95_lower"]), float(overall_m3_row["ci95_upper"])],
            },
            "lcz_nodata_excluded_A6": {
                "lcz_primary": nodata_primary.get("lcz_primary"),
                "lcz_weak250": (spearman_weak.attrs.get("lcz_nodata_excluded", {}).get("lcz_weak250")
                                if spearman_weak is not None else None),
                "note": "Excluded from every by-LCZ statistic; kept in the overall Spearman (A6).",
            },
            "compact_paradox": paradox_res,
            "decoupled_citywide_share": float(lcz_shares.loc[lcz_shares["group"] == "ALL", "share_decoupled"].iloc[0]),
            "moran_summary": moran_summary,
        }

    # Save master summary JSON
    summary_path = out_dir / "h3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\n==========================================")
    print(f"H3 Analysis Complete! Master summary saved to {summary_path}")
    print(f"==========================================\n")

    return master_summary


def main():
    args = parse_args()

    seg_path = Path(args.segments_path)
    metrics_dir = Path(args.metrics_dir)
    grid250_path = Path(args.grid250_path)
    audit_csv = Path(args.audit_csv)
    audit_gpkg = Path(args.audit_gpkg)
    out_dir = Path(args.out_dir)

    if args.synthetic:
        # Synthetic test mode
        test_dir = out_dir / "synthetic_test"
        from scripts.street.wsl.h3.synthetic_test import generate_synthetic_h3_dataset
        sp, mp, gp = generate_synthetic_h3_dataset(test_dir, n_segments=1500)
        summary = run_h3_analysis(
            segments_path=sp,
            metrics_paths=mp,
            grid250_path=gp,
            audit_csv=audit_csv,
            audit_gpkg=audit_gpkg,
            out_dir=test_dir,
            is_synthetic=True,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        return

    # Check if inputs exist or poll
    met_paths = {d: metrics_dir / f"seg_metrics_{d}.parquet" for d in DAYS}
    has_inputs = seg_path.exists() and any(p.exists() for p in met_paths.values())

    if not has_inputs:
        if args.poll:
            ok = poll_for_inputs(seg_path, metrics_dir, interval_sec=args.poll_interval, timeout_sec=args.poll_timeout)
            if not ok:
                print("Exiting: input tables were not produced within timeout.")
                sys.exit(1)
        else:
            print(f"Error: input tables not found ({seg_path}, {metrics_dir}/seg_metrics_*.parquet). Use --poll to wait for them.")
            sys.exit(1)

    # Run on real data
    summary = run_h3_analysis(
        segments_path=seg_path,
        metrics_paths=met_paths,
        grid250_path=grid250_path if grid250_path.exists() else None,
        audit_csv=audit_csv,
        audit_gpkg=audit_gpkg,
        out_dir=out_dir,
        is_synthetic=False,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
