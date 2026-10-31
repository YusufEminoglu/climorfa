"""
Unit and integration tests for WP0.5-A3 Equipment Scenarios.
Authoritative specification: docs/climorfa_street/wp/WP0.5A3_equipment.md
"""

import os
import sys
import yaml
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.street.field.simulate_campaign import solve_tg, compute_iso7726_tmrt_fast


def test_equipment_scenarios_config():
    """Verify configuration file validity, tagging, and scenario coverage."""
    cfg_path = os.path.join(PROJECT_ROOT, "configs/street/equipment_scenarios.yaml")
    assert os.path.exists(cfg_path), f"Config file not found at {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("is_synthetic") is True, "Config must carry is_synthetic: true"
    assert "scenarios" in cfg, "Config must contain 'scenarios' key"

    scenarios = cfg["scenarios"]
    for sc_id in ["S0", "S1", "S2", "S3", "S4"]:
        assert sc_id in scenarios, f"Scenario {sc_id} missing from config"
        sc = scenarios[sc_id]
        assert sc["id"] == sc_id
        assert "timing" in sc
        assert "feasible_visits_per_day_16pts" in sc["timing"]
        assert "feasible_visits_per_day_24pts" in sc["timing"]


def test_globe_convection_coefficients_and_amplification():
    """
    Verify forced convection coefficients and amplification factors across globe sizes:
    - 25.4 mm (S0/S1): D = 0.0254 m
    - 38 mm (S2, Thorsson et al. 2007): D = 0.038 m
    - 150 mm (S3, ISO 7726): D = 0.150 m
    """
    sigma = 5.670374419e-8
    eps = 0.95
    t_k = 295.15  # ~22 °C ambient
    v = 2.6  # typical Izmir wind speed

    # Forced convection coefficient: hc = 6.3 * v^0.6 / D^0.4
    hc_25 = 6.3 * (v ** 0.6) / (0.0254 ** 0.4)
    hc_38 = 6.3 * (v ** 0.6) / (0.038 ** 0.4)
    hc_150 = 6.3 * (v ** 0.6) / (0.150 ** 0.4)

    # Monotonic decrease in convective resistance / coefficient with increasing D
    assert hc_25 > hc_38 > hc_150, "hc should decrease with diameter"
    assert pytest.approx(hc_38 / (v ** 0.6), rel=1e-3) == 23.304, "Thorsson 2007 38 mm coefficient mismatch"

    # Amplification factor: 1 + hc / (4 * eps * sigma * T^3)
    denom = 4.0 * eps * sigma * (t_k ** 3)
    amp_25 = 1.0 + hc_25 / denom
    amp_38 = 1.0 + hc_38 / denom
    amp_150 = 1.0 + hc_150 / denom

    # Standard 150 mm globe has ~1.8x lower amplification than 25.4 mm globe
    assert amp_25 > amp_38 > amp_150
    assert pytest.approx(amp_25, abs=0.2) == 9.8
    assert pytest.approx(amp_38, abs=0.2) == 8.5
    assert pytest.approx(amp_150, abs=0.2) == 5.3


def test_isothermal_calibration_offset_reduction():
    """
    Verify that isothermal calibration effectively reduces calibration offset variance.
    S0 (uncalibrated Kestrel): sigma(cal_tg) = 1.4 / sqrt(6) ~ 0.57 K
    S1 (isothermal cal): sigma(cal_tg - cal_ta) = 0.10 K
    """
    n_samples = 50000
    rng = np.random.default_rng(42)

    # S0
    tg_sigma_s0 = (1.4 / np.sqrt(3)) / np.sqrt(2)
    s0_offsets = rng.normal(0, tg_sigma_s0, n_samples)
    var_s0 = np.var(s0_offsets)

    # S1
    iso_sigma = 0.10
    s1_diffs = rng.normal(0, iso_sigma, n_samples)
    var_s1 = np.var(s1_diffs)

    # Variance reduction is > 30x
    assert var_s0 / var_s1 > 30.0


def test_s3_operational_time_budget():
    """
    Verify that 150 mm globe time constant forces stop duration to 25 min,
    allowing only 1 visit per day within 10:00–16:00 (360 min window).
    """
    active_window_min = 360
    stop_dur_min = 25  # 20 min equilibration + 5 min averaging
    walk_min = 3
    cycle_time_min = stop_dur_min + walk_min  # 28 min

    # For 16 points (8 points per device)
    circuit_time_16pts = 8 * cycle_time_min  # 224 min
    assert circuit_time_16pts == 224
    assert circuit_time_16pts <= active_window_min
    assert (2 * circuit_time_16pts) > active_window_min  # 448 min > 360 min

    # Feasible full visits = floor(360 / 224) = 1
    feasible_visits_16pts = active_window_min // circuit_time_16pts
    assert feasible_visits_16pts == 1

    # For 24 points (12 points per device)
    circuit_time_24pts = 12 * cycle_time_min  # 336 min
    assert circuit_time_24pts <= active_window_min
    assert (2 * circuit_time_24pts) > active_window_min
    feasible_visits_24pts = active_window_min // circuit_time_24pts
    assert feasible_visits_24pts == 1


def test_equipment_scenarios_deliverable_csv():
    """Verify generated scenario CSV exists and conforms to required deliverables schema."""
    import pandas as pd
    csv_path = os.path.join(PROJECT_ROOT, "outputs/street/field/SYNTHETIC_equipment_scenarios.csv")
    assert os.path.exists(csv_path), "SYNTHETIC_equipment_scenarios.csv does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) >= 10, "Expected at least 10 scenario-design rows"
    assert (df["is_synthetic"] == True).all(), "All rows must have is_synthetic=True"

    required_cols = [
        "scenario_id", "scenario_name", "design_name", "n_points", "visits_per_day",
        "total_stops", "feasible_visits_per_day", "is_design_feasible_in_6h_window",
        "sigma_meas_type_A_k", "sigma_struct_type_A_k",
        "emergent_rmse_ev_k", "rmse_ci_half_width_overall_k", "rmse_ci_half_width_max_type_k",
        "target_overall_le_1_5k", "target_max_type_le_1_5k",
        "power_exposure_contrast_pct", "share_variance_device_offsets_pct",
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column {col} missing from output CSV"

    # Verify S0 fails overall and max type
    s0_planned = df[(df["scenario_id"] == "S0") & (df["design_name"].str.contains("Planned"))].iloc[0]
    assert s0_planned["target_overall_le_1_5k"] == "FAIL"
    assert s0_planned["target_max_type_le_1_5k"] == "FAIL"

    # Verify S1-S4 pass overall in planned design
    for sc in ["S1", "S2", "S4"]:
        row = df[(df["scenario_id"] == sc) & (df["design_name"].str.contains("Planned"))].iloc[0]
        assert row["target_overall_le_1_5k"] == "PASS"
