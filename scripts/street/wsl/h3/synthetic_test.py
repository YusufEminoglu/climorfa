"""Synthetic test generator and validator for H3 (surface vs pedestrian heat stress).

Adheres strictly to:
- Rule 6: Synthetic data rule (is_synthetic=True, SYNTHETIC in filenames, never described as observed).
- Rule 6b: Non-circular validation (tests that could fail, independent reference checks).
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box
import scipy.stats as stats

from scripts.street.wsl.h3.bootstrap import (
    assign_2km_blocks,
    block_bootstrap_spearman,
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


def generate_synthetic_h3_dataset(
    out_dir: Path,
    n_segments: int = 1200,
    n_grid_cells: int = 64,
    seed: int = 20261030,
) -> tuple[Path, dict[str, Path], Path]:
    """Generate synthetic segments, metrics (IMBAT and CALM), and grid_250 with planted effects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1. Generate 250 m grid polygons (8x8 grid in Izmir EPSG:5253 coordinates)
    x_base = 500000
    y_base = 4250000
    grid_rows = []
    g250_ids = []

    for i in range(8):
        for j in range(8):
            x0 = x_base + i * 250
            y0 = y_base + j * 250
            gid = f"{x0 // 250}_{y0 // 250}"
            g250_ids.append(gid)
            geom = box(x0, y0, x0 + 250, y0 + 250)
            # Spatially clustered LST (high in east, low in west)
            spatial_trend = (i / 8.0) * 8.0 + (j / 8.0) * 4.0
            lst_cell = 38.0 + spatial_trend + rng.normal(0, 1.0)
            grid_rows.append({
                "g250_id": gid,
                "tile_id": f"T_{(x0 // 1000):04d}_{(y0 // 1000):04d}",
                "district": "KONAK" if j < 4 else "BORNOVA",
                "lcz_primary_major": int(rng.choice([2, 3, 5, 6, 8])),
                "lcz_weak250": int(rng.choice([2, 3, 5, 6, 8])),
                "lst_med": float(lst_cell),
                "pop": float(rng.uniform(100, 2000)),
                "reach": float(rng.uniform(500, 5000)),
                "is_synthetic": True,
                "geometry": geom,
            })

    gdf_grid = gpd.GeoDataFrame(grid_rows, crs="EPSG:5253")
    grid_path = out_dir / "grid_250_SYNTHETIC.gpkg"
    gdf_grid.to_file(grid_path, layer="grid", driver="GPKG")

    # 2. Generate Segments
    seg_rows = []
    lcz_choices = [1, 2, 3, 4, 5, 6, 8, 9, 10, 101, 102]
    lcz_probs = [0.03, 0.25, 0.25, 0.05, 0.15, 0.12, 0.05, 0.04, 0.03, 0.02, 0.01]
    lcz_probs = np.array(lcz_probs) / sum(lcz_probs)

    for sid in range(1, n_segments + 1):
        # Pick a random cell
        cell_idx = rng.integers(0, len(gdf_grid))
        cell = gdf_grid.iloc[cell_idx]
        b = cell.geometry.bounds

        # Segment coordinates inside cell
        sx0 = rng.uniform(b[0], b[2] - 20)
        sy0 = rng.uniform(b[1], b[3] - 20)
        sx1 = sx0 + rng.uniform(15, 80)
        sy1 = sy0 + rng.uniform(15, 80)
        line = LineString([(sx0, sy0), (sx1, sy1)])

        lcz_p = int(rng.choice(lcz_choices, p=lcz_probs))
        # Add local variation around cell LST
        lst_seg = float(cell["lst_med"] + rng.normal(0, 0.8))

        seg_rows.append({
            "seg_id": sid,
            "tile_id": cell["tile_id"],
            "district": cell["district"],
            "lcz_primary": lcz_p,
            "lcz_weak250": lcz_p,
            "length_m": float(line.length),
            "n_ps": int(rng.integers(25, 200)),
            "azimuth_deg": float(rng.uniform(0, 180)),
            "hw_ratio": float(rng.uniform(0.3, 3.5)),
            "bh_mean30": float(rng.uniform(4.0, 35.0)),
            "bcover30": float(rng.uniform(0.1, 0.8)),
            "tcover15": float(rng.uniform(0.0, 0.4)),
            "tht15": float(rng.uniform(2.0, 12.0)),
            "dist_coast_m": float(rng.uniform(100, 5000)),
            "elev_m": float(rng.uniform(5, 120)),
            "slope_deg": float(rng.uniform(0, 15)),
            "lst_med": lst_seg,
            "g050_id": f"{int(sx0)//50}_{int(sy0)//50}",
            "g100_id": f"{int(sx0)//100}_{int(sy0)//100}",
            "g250_id": cell["g250_id"],
            "g500_id": f"{int(sx0)//500}_{int(sy0)//500}",
            "is_synthetic": True,
            "geometry": line,
        })

    gdf_segs = gpd.GeoDataFrame(seg_rows, crs="EPSG:5253")
    seg_path = out_dir / "segments_SYNTHETIC.gpkg"
    gdf_segs.to_file(seg_path, layer="segments", driver="GPKG")

    # 3. Generate Metrics per Day with Planted Effects
    # IMBAT:
    # - Overall: rho(LST, M1) is POSITIVE (~ +0.65)
    # - Planted compact paradox in LCZ 2/3: rho(LST, M1) is NEGATIVE (~ -0.55)
    # CALM:
    # - Overall: rho(LST, M1) is POSITIVE (~ +0.60)
    # - LCZ 2/3: rho(LST, M1) is POSITIVE (~ +0.45) -> Paradox NOT observed
    met_paths = {}

    for day in ["IMBAT", "CALM"]:
        met_rows = []
        for _, row in gdf_segs.iterrows():
            lcz_val = row["lcz_primary"]
            lst = row["lst_med"]

            if day == "IMBAT":
                if lcz_val in (2, 3):
                    # Negative correlation planted in LCZ 2/3: deep canyon shading
                    # Higher LST surface, but deeper canyons with lower M1
                    m1 = 44.0 - 0.45 * (lst - 38.0) + rng.normal(0, 0.6)
                else:
                    # Positive correlation overall
                    m1 = 33.0 + 0.50 * (lst - 38.0) + rng.normal(0, 0.8)
                m3 = m1 + rng.normal(15.0, 2.0)
            else:  # CALM
                # In CALM, positive correlation everywhere (calm wind eliminates ventilation contrast)
                if lcz_val in (2, 3):
                    m1 = 36.0 + 0.40 * (lst - 38.0) + rng.normal(0, 0.7)
                else:
                    m1 = 35.0 + 0.50 * (lst - 38.0) + rng.normal(0, 0.8)
                m3 = m1 + rng.normal(16.0, 2.0)

            met_rows.append({
                "seg_id": row["seg_id"],
                "n_ps_valid": row["n_ps"],
                "utci_b06": m1 - 10.0,
                "utci_b15": float(m1),
                "utci_b21": m1 - 8.0,
                "utci_unif_b15": float(m1 + 0.5),
                "tmrt_b15": float(m3),
                "tmrt_b15_p90": float(m3 + 2.0),
                "v10eff_b15": float(rng.uniform(0.5, 4.0)),
                "ratio_med_b15": float(rng.uniform(0.2, 0.9)),
                "svf_med": float(rng.uniform(0.2, 0.9)),
                "shade_frac_10_18": float(rng.uniform(0.1, 0.8)),
                "m2_strong": int(rng.integers(1, 9)),
                "m2_vstrong": int(rng.integers(0, 5)),
                "is_synthetic": True,
                "partial_run": True,
                "n_tiles_at_extraction": 12,
            })

        df_met = pd.DataFrame(met_rows)
        p = out_dir / f"seg_metrics_{day}_SYNTHETIC.parquet"
        df_met.to_parquet(p, index=False)
        met_paths[day] = p

    return seg_path, met_paths, grid_path


def run_synthetic_validation_suite() -> dict:
    """Run full synthetic testing suite including non-circular validation (Rule 6b)."""
    test_dir = Path("outputs/street/analysis/h3/synthetic_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic test dataset...")
    seg_path, met_paths, grid_path = generate_synthetic_h3_dataset(test_dir, n_segments=1500)

    results = {}

    # Load segments and add midpoints and blocks
    gdf_segs = gpd.read_file(seg_path, layer="segments")
    mid = gdf_segs.geometry.interpolate(0.5, normalized=True)
    gdf_segs["x_mid"] = mid.x
    gdf_segs["y_mid"] = mid.y
    gdf_segs = assign_2km_blocks(gdf_segs)

    # -------------------------------------------------------------
    # NON-CIRCULAR VALIDATION CHECKS (Rule 6b)
    # -------------------------------------------------------------
    # Check 1: Independent Mathematical Monotonicity Reference
    # Verify Spearman implementation returns exactly 1.0 for strictly monotonic data
    x_mono = np.linspace(10, 50, 100)
    y_mono = np.exp(x_mono / 10.0)  # Nonlinear strictly increasing
    blks_mono = np.repeat(["b1", "b2", "b3", "b4"], 25)
    rho_mono, _, _, _, _ = block_bootstrap_spearman(x_mono, y_mono, blks_mono, n_boot=50)
    check_1_pass = bool(np.isclose(rho_mono, 1.0, atol=1e-5))
    results["check_1_monotonicity_independent"] = {
        "passed": check_1_pass,
        "rho_measured": rho_mono,
        "expected": 1.0,
    }
    assert check_1_pass, f"Check 1 failed: Spearman monotonicity test failed ({rho_mono})"

    # Check 2: Physical Plausibility Limits
    # Ensure all simulated values lie within physical bounds of summer urban microclimate
    df_imbat_met = pd.read_parquet(met_paths["IMBAT"])
    m1_min = df_imbat_met["utci_b15"].min()
    m1_max = df_imbat_met["utci_b15"].max()
    lst_min = gdf_segs["lst_med"].min()
    lst_max = gdf_segs["lst_med"].max()
    check_2_pass = bool((m1_min >= 15.0) and (m1_max <= 60.0) and (lst_min >= 20.0) and (lst_max <= 70.0))
    results["check_2_physical_plausibility"] = {
        "passed": check_2_pass,
        "m1_range": [m1_min, m1_max],
        "lst_range": [lst_min, lst_max],
    }
    assert check_2_pass, "Check 2 failed: Physical bounds violated"

    # Check 3: Planted Compact Paradox Decision Rule Recovery (IMBAT vs CALM)
    # IMBAT was planted with negative correlation in LCZ 2/3 -> MUST be SUPPORTED
    # CALM was planted with positive correlation in LCZ 2/3 -> MUST be NOT SUPPORTED
    df_imbat_merged = gdf_segs.merge(df_imbat_met, on="seg_id")
    df_calm_met = pd.read_parquet(met_paths["CALM"])
    df_calm_merged = gdf_segs.merge(df_calm_met, on="seg_id")

    df_imbat_merged["m1"] = df_imbat_merged["utci_b15"]
    df_imbat_merged["m3"] = df_imbat_merged["tmrt_b15"]
    df_calm_merged["m1"] = df_calm_merged["utci_b15"]
    df_calm_merged["m3"] = df_calm_merged["tmrt_b15"]

    rule_imbat = evaluate_compact_paradox_rule(df_imbat_merged, n_boot=200)
    rule_calm = evaluate_compact_paradox_rule(df_calm_merged, n_boot=200)

    imbat_supported = (rule_imbat["decision"] == "SUPPORTED") and (rule_imbat["rho"] < 0)
    calm_not_supported = (rule_calm["decision"] == "NOT SUPPORTED") and (rule_calm["rho"] > 0)
    check_3_pass = bool(imbat_supported and calm_not_supported)
    results["check_3_decision_rule_recovery"] = {
        "passed": check_3_pass,
        "imbat_decision": rule_imbat["decision"],
        "imbat_rho": rule_imbat["rho"],
        "imbat_ci": [rule_imbat["ci95_lower"], rule_imbat["ci95_upper"]],
        "calm_decision": rule_calm["decision"],
        "calm_rho": rule_calm["rho"],
        "calm_ci": [rule_calm["ci95_lower"], rule_calm["ci95_upper"]],
    }
    assert check_3_pass, f"Check 3 failed: IMBAT={rule_imbat['decision']}, CALM={rule_calm['decision']}"

    # Check 4: Decoupling Tercile Partition Exhaustiveness
    df_dec = identify_decoupled_segments(df_imbat_merged)
    terc_counts = df_dec["lst_tercile"].value_counts().to_dict()
    total_valid = int(np.isfinite(df_imbat_merged["lst_med"]).sum())
    sum_terc = sum(terc_counts.values())
    check_4_pass = bool(sum_terc == total_valid)
    results["check_4_tercile_exhaustiveness"] = {
        "passed": check_4_pass,
        "total_valid_segments": total_valid,
        "tercile_sum": sum_terc,
        "is_decoupled_count": int(df_dec["is_decoupled"].sum()),
    }
    assert check_4_pass, "Check 4 failed: Tercile partition is not exhaustive"

    # Check 5: Bivariate Moran on 250 m Grid
    gdf_grid = gpd.read_file(grid_path, layer="grid")
    gdf_grid_annotated = aggregate_m1_to_grid250(df_imbat_merged, gdf_grid)
    _, moran_summary = run_bivariate_moran_250m(gdf_grid_annotated, permutations=199)
    moran_pass = bool(np.isfinite(moran_summary["global_bivariate_moran_I"]))
    results["check_5_bivariate_moran"] = {
        "passed": moran_pass,
        "global_I": moran_summary["global_bivariate_moran_I"],
        "p_sim": moran_summary["global_bivariate_moran_p_sim"],
        "significant_clusters": moran_summary["significant_clusters_count"],
    }
    assert moran_pass, "Check 5 failed: Bivariate Moran calculation failed"

    print("All 5 synthetic validation checks PASSED successfully!")
    return results


if __name__ == "__main__":
    import json
    res = run_synthetic_validation_suite()
    print(json.dumps(res, indent=2))
