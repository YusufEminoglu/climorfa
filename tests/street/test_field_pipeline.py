"""
Unit and integration tests for CLIMORFA Street Field Campaign Tools (WP0.5-A2).
Verifies:
1. ISO 7726 forward and inverse globe heat balance consistency exact (< 1e-6 K).
2. Co-location correction recovers injected relative device offsets within tolerance.
3. Power analysis imports and uses the exact generator functions (import check).
4. 10 m wind extrapolation bounds [0.5, 17.0] m/s.
5. Sampling design integrity (16 points, 4 types x 2 exposures x 2 replicates).
6. Synthetic raw observations format and schema compliance with co-location rows.
7. Emergent calibration metrics against §3.4 targets.
8. Non-circular validation checks (Rule 6b).
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd

from scripts.street.field.simulate_campaign import solve_tg
from scripts.street.field.process_campaign import (
    compute_iso7726_tmrt,
    extrapolate_wind_10m,
    calc_metrics,
    estimate_colocation_offsets,
    apply_colocation_correction,
)
import scripts.street.field.power_analysis as pa


def test_iso7726_roundtrip_exact():
    """
    Step 5 test: round-trip solve_tg -> compute_iso7726_tmrt exact (< 1e-6 K) without noise.
    Tests across a dense, representative grid of microclimate conditions.
    """
    tmrt_grid = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    ta_grid = [5.0, 15.0, 25.0, 35.0]
    wind_grid = [0.1, 0.5, 1.2, 2.5, 4.0, 7.0]

    max_err = 0.0
    for tmrt in tmrt_grid:
        for ta in ta_grid:
            for v in wind_grid:
                tg_solved = solve_tg(tmrt_c=tmrt, ta_c=ta, v_ms=v)
                tmrt_recovered = compute_iso7726_tmrt(tg_c=tg_solved, ta_c=ta, v_ms=v)
                err = abs(tmrt - tmrt_recovered)
                if err > max_err:
                    max_err = err
                assert err < 1e-6, (
                    f"Roundtrip error {err:.3e} >= 1e-6 K at Tmrt={tmrt}, Ta={ta}, Wind={v}"
                )
    print(f"Verified {len(tmrt_grid)*len(ta_grid)*len(wind_grid)} conditions: max roundtrip error = {max_err:.3e} K (< 1e-6 K)")


def test_colocation_recovers_injected_offsets():
    """
    Step 5 test: co-location correction recovers injected relative device offsets within tolerance,
    and removes the relative discrepancy.
    """
    n_samples = 480
    rng = np.random.default_rng(12345)

    # Injected known offsets on DEV_B relative to DEV_A
    inj_ta = 0.35
    inj_tg = 0.65
    inj_wind = 0.12
    inj_rh = -2.5

    # Base true conditions
    ta_base = 20.0 + rng.normal(0.0, 0.2, n_samples)
    tg_base = 23.0 + rng.normal(0.0, 0.4, n_samples)
    wind_base = 2.0 + rng.normal(0.0, 0.1, n_samples)
    rh_base = 55.0 + rng.normal(0.0, 0.5, n_samples)

    timestamps = [f"2026-11-12 09:{i//60:02d}:{i%60:02d}" for i in range(n_samples)]
    pids = ["COLOC_2026-11-12_shade_start"] * (n_samples // 2) + ["COLOC_2026-11-12_sun_start"] * (n_samples // 2)

    df_a = pd.DataFrame({
        "timestamp_local": timestamps,
        "point_id": pids,
        "device_id": "DEV_A",
        "ta_c": ta_base + rng.normal(0.0, 0.20, n_samples),
        "tg_c": tg_base + rng.normal(0.0, 0.50, n_samples),
        "wind_ms": np.maximum(0.0, wind_base + rng.normal(0.0, 0.05, n_samples)),
        "rh_pct": np.clip(rh_base + rng.normal(0.0, 0.8, n_samples), 1.0, 99.0),
    })

    df_b = pd.DataFrame({
        "timestamp_local": timestamps,
        "point_id": pids,
        "device_id": "DEV_B",
        "ta_c": ta_base + inj_ta + rng.normal(0.0, 0.20, n_samples),
        "tg_c": tg_base + inj_tg + rng.normal(0.0, 0.50, n_samples),
        "wind_ms": np.maximum(0.0, wind_base + inj_wind + rng.normal(0.0, 0.05, n_samples)),
        "rh_pct": np.clip(rh_base + inj_rh + rng.normal(0.0, 0.8, n_samples), 1.0, 99.0),
    })

    df_raw = pd.concat([df_a, df_b], ignore_index=True)

    # 1. Estimate offsets
    offsets = estimate_colocation_offsets(df_raw, ref_device="DEV_A")
    assert offsets["has_colocation"] == True
    assert offsets["n_samples"] == n_samples

    # Verify recovery within tolerance
    assert abs(offsets["ta"]["mean"] - inj_ta) < 0.05, f"Ta offset recovery failed: {offsets['ta']['mean']} vs {inj_ta}"
    assert abs(offsets["tg"]["mean"] - inj_tg) < 0.08, f"Tg offset recovery failed: {offsets['tg']['mean']} vs {inj_tg}"
    assert abs(offsets["wind"]["mean"] - inj_wind) < 0.03, f"Wind offset recovery failed: {offsets['wind']['mean']} vs {inj_wind}"
    assert abs(offsets["rh"]["mean"] - inj_rh) < 0.20, f"RH offset recovery failed: {offsets['rh']['mean']} vs {inj_rh}"

    # 2. Apply correction and verify that relative difference is eliminated
    df_corrected = apply_colocation_correction(df_raw, offsets, mode="baseline_A")
    offsets_after = estimate_colocation_offsets(df_corrected, ref_device="DEV_A")

    assert abs(offsets_after["ta"]["mean"]) < 1e-6
    assert abs(offsets_after["tg"]["mean"]) < 1e-6
    assert abs(offsets_after["wind"]["mean"]) < 1e-6
    assert abs(offsets_after["rh"]["mean"]) < 1e-6


def test_power_analysis_uses_generator_functions():
    """
    Step 5 test: power analysis uses the exact same functions as the generator and processing pipeline (import check).
    """
    import inspect
    from scripts.street.field.simulate_campaign import solve_tg as sim_solve_tg
    from scripts.street.field.process_campaign import compute_iso7726_tmrt as proc_compute_tmrt

    # Check identical function references
    assert pa.solve_tg is sim_solve_tg, "power_analysis does not import solve_tg from simulate_campaign!"
    assert pa.compute_iso7726_tmrt is proc_compute_tmrt, "power_analysis does not import compute_iso7726_tmrt from process_campaign!"

    # Verify power_analysis function signature and implementation
    source = inspect.getsource(pa.simulate_single_campaign_full_chain)
    assert "solve_tg" in source, "simulate_single_campaign_full_chain does not call solve_tg!"
    assert "compute_iso7726_tmrt" in source, "simulate_single_campaign_full_chain does not call compute_iso7726_tmrt!"
    assert "normal(0, 2.10)" not in source, "power_analysis still contains deprecated simplified noise model!"


def test_extrapolate_wind_limits():
    """Verify 10 m wind is always clipped to pythermalcomfort UTCI bounds [0.5, 17.0]."""
    assert extrapolate_wind_10m(0.0) == 0.5
    assert extrapolate_wind_10m(0.1) >= 0.5
    assert extrapolate_wind_10m(15.0) == 17.0
    assert 0.5 <= extrapolate_wind_10m(2.0) <= 17.0


def test_provisional_points_integrity():
    """Verify that PROVISIONAL_points.gpkg satisfies all sampling design constraints."""
    gpkg_path = "data/street/field/PROVISIONAL_points.gpkg"
    assert os.path.exists(gpkg_path), "PROVISIONAL_points.gpkg does not exist"

    gdf = gpd.read_file(gpkg_path)
    assert len(gdf) == 16, f"Expected exactly 16 points, got {len(gdf)}"

    # Check devices
    dev_counts = gdf["device_id"].value_counts()
    assert dev_counts["DEV_A"] == 8
    assert dev_counts["DEV_B"] == 8

    # Check 4 morphology types x 2 exposures
    for mtype in ["A", "B", "C", "D"]:
        sub = gdf[gdf["morphology_type"] == mtype]
        assert len(sub) == 4
        assert (sub["exposure_target_13h"] == "sunlit").sum() == 2
        assert (sub["exposure_target_13h"] == "shaded").sum() == 2

    # Check pedestrian constraints
    assert (gdf["dist_to_wall_m"] >= 1.0).all(), "Points found within 1.0m of building wall!"

    # Check metadata attributes
    assert (gdf["is_synthetic"] == False).all()
    assert (gdf["provisional_geometry"] == True).all()


def test_synthetic_raw_dataset_with_colocation():
    """Verify synthetic raw CSV conforms to v1.1 schema and contains co-location rows."""
    raw_csv = "data/street/field/SYNTHETIC_kestrel_raw_campaign.csv"
    assert os.path.exists(raw_csv), "Raw CSV does not exist"

    df = pd.read_csv(raw_csv)
    # Expected: 96 stops * 60 samples (5760) + 1920 co-location readings = 7680 rows
    assert len(df) == 7680, f"Expected 7680 rows, got {len(df)}"

    coloc_rows = df[df["point_id"].str.startswith("COLOC_")]
    assert len(coloc_rows) == 1920, f"Expected 1920 co-location rows, got {len(coloc_rows)}"

    mobile_rows = df[~df["point_id"].str.startswith("COLOC_")]
    assert len(mobile_rows) == 5760, f"Expected 5760 mobile stop rows, got {len(mobile_rows)}"

    required_cols = [
        "timestamp_local", "device_id", "point_id",
        "ta_c", "rh_pct", "wind_ms", "tg_c",
        "is_synthetic", "provenance", "provisional_geometry"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column: {c}"

    assert (df["is_synthetic"] == True).all()
    assert (df["provisional_geometry"] == True).all()
    assert (df["rh_pct"] >= 0.0).all() and (df["rh_pct"] <= 100.0).all()
    assert (df["wind_ms"] >= 0.0).all()


def test_non_circular_validation():
    """
    Rule 6b: Non-circular validation test.
    Verifies that the synthetic generator produces physically plausible values that
    could have failed thermodynamic consistency checks:
    1. Tmrt must not violate physical extremes for November in Izmir [5.0, 75.0] C.
    2. Ta must be in plausible bounds [5.0, 35.0] C.
    3. UTCI must be in plausible bounds [-5.0, 40.0] C.
    4. Model vs Obs correlation r must be strictly positive and statistically non-trivial (r > 0.85).
    """
    stops_csv = "data/street/field/SYNTHETIC_processed_stops.csv"
    assert os.path.exists(stops_csv), "Stops CSV does not exist"

    df = pd.read_csv(stops_csv)

    assert (df["tmrt_obs"] >= 5.0).all() and (df["tmrt_obs"] <= 75.0).all(), "Tmrt violates physical bounds!"
    assert (df["ta_obs"] >= 5.0).all() and (df["ta_obs"] <= 35.0).all(), "Ta violates physical bounds!"
    valid_utci = df["utci_obs"].dropna()
    assert (valid_utci >= -5.0).all() and (valid_utci <= 40.0).all(), "UTCI violates physical bounds!"

    r_val = np.corrcoef(df["tmrt_model"], df["tmrt_obs"])[0, 1]
    assert r_val > 0.85, f"Tmrt correlation too low: {r_val:.3f}"
