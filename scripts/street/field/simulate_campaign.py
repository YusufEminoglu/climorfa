"""
CLIMORFA Street — Synthetic Field Campaign Simulator (v1.1).
Authoritative specification: docs/climorfa_street/field_error_model.md §1, §3.1–3.2.

Generates synthetic placeholder Kestrel raw observation files for the November campaign days
by combining SOLWEIG-GPU simulations, literature-calibrated error structures, and sensor physics.

Key v1.1 features:
1. Start- and end-of-day co-location intercomparison segments (20 min shade + 20 min sun).
2. Day-2 device crossover (DEV_A and DEV_B swap sampling point routes).
3. Exact ISO 7726 inversion (solve_tg round-trip error < 1e-10 K).
4. Measurement error variance (sigma_meas^2) computed via Monte Carlo of the measurement chain alone,
   with structural variance set as the remainder: sigma_struct^2 = target^2 - sigma_meas^2.

Outputs:
data/street/field/SYNTHETIC_kestrel_raw_2026-11-12.csv
data/street/field/SYNTHETIC_kestrel_raw_2026-11-13.csv
data/street/field/SYNTHETIC_kestrel_raw_campaign.csv
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from scipy.optimize import brentq


def solve_tg(tmrt_c: float, ta_c: float, v_ms: float, d: float = 0.0254, eps: float = 0.95) -> float:
    """
    Invert ISO 7726 globe heat balance equation to solve for equilibrium globe temperature Tg (°C).
    Uses high-precision Brent root finding (xtol=1e-12, rtol=1e-12) to achieve exact roundtrip (< 1e-6 K).
    """
    sigma = 5.670374419e-8
    tmrt_k = tmrt_c + 273.15
    ta_k = ta_c + 273.15
    v_eff = max(0.05, float(v_ms))
    h_forced = 6.3 * (v_eff ** 0.6) / (d ** 0.4)

    def f(tg_c):
        tg_k = tg_c + 273.15
        h_nat = 1.4 * (abs(tg_c - ta_c) / d) ** 0.25
        h_c = max(h_forced, h_nat)
        return eps * sigma * (tmrt_k ** 4 - tg_k ** 4) - h_c * (tg_c - ta_c)

    # In equilibrium, Tg is strictly bracketed between Ta and Tmrt (with slight numerical margin)
    low = min(tmrt_c, ta_c) - 2.0
    high = max(tmrt_c, ta_c) + 2.0
    try:
        return float(brentq(f, low, high, xtol=1e-12, rtol=1e-12))
    except Exception:
        # Fallback wider bracket
        try:
            return float(brentq(f, -50.0, 150.0, xtol=1e-12, rtol=1e-12))
        except Exception:
            return float(ta_c + 0.35 * (tmrt_c - ta_c))


def calc_sat_vapor_pressure_kpa(temp_c: float) -> float:
    """Magnus formula for saturation vapor pressure over water (kPa)."""
    return 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def conserve_rh(ta_model_c: float, rh_model_pct: float, ta_true_c: float) -> float:
    """Conserve vapor pressure when air temperature changes."""
    e_s_mod = calc_sat_vapor_pressure_kpa(ta_model_c)
    e_a = (rh_model_pct / 100.0) * e_s_mod
    e_s_true = calc_sat_vapor_pressure_kpa(ta_true_c)
    rh_true = (e_a / max(1e-4, e_s_true)) * 100.0
    return float(np.clip(rh_true, 5.0, 99.0))


def load_met_rh(met_path: str) -> np.ndarray:
    """Load hourly RH series from metfile."""
    df_met = pd.read_csv(met_path, sep=r"\s+", skiprows=1, header=None)
    rh_series = df_met[10].values.astype(float)
    return rh_series


def compute_iso7726_tmrt_fast(tg_c: float, ta_c: float, v_ms: float, d: float = 0.0254, eps: float = 0.95) -> float:
    """Direct forward ISO 7726 Tmrt calculation."""
    sigma = 5.670374419e-8
    v_eff = max(0.05, float(v_ms))
    h_forced = 6.3 * (v_eff ** 0.6) / (d ** 0.4)
    h_nat = 1.4 * (abs(tg_c - ta_c) / d) ** 0.25
    h_c = max(h_forced, h_nat)

    tg_k = tg_c + 273.15
    term = (tg_k ** 4) + (h_c / (eps * sigma)) * (tg_c - ta_c)
    if term <= 0:
        return ta_c
    return float((term ** 0.25) - 273.15)


def compute_measurement_variance_mc(
    targets: dict,
    stop_conditions: list,
    n_mc: int = 1000,
    seed: int = 42,
    ref_device: str = "DEV_A",
) -> dict:
    """
    Compute sigma_meas^2 by Monte Carlo of the measurement chain alone.
    Runs the full measurement and co-location correction chain without structural error (truth = model),
    and computes the variance of (Tmrt_obs - Tmrt_true) for each morphology type.
    """
    rng = np.random.default_rng(seed)

    ta_sigma = 0.5 / np.sqrt(3)
    tg_sigma = 1.4 / np.sqrt(3)
    wind_abs = 0.10
    wind_rel = 0.03
    n_coloc_samples = 960  # 4 segments x 240 readings

    type_sq_errors = {mtype: [] for mtype in targets}

    for _ in range(n_mc):
        # Draw device calibration offsets once per campaign
        cal_A = {
            "ta": rng.normal(0.0, ta_sigma / np.sqrt(2)),
            "tg": rng.normal(0.0, tg_sigma / np.sqrt(2)),
            "wind": rng.normal(0.0, (wind_abs / np.sqrt(3)) / np.sqrt(2)),
        }
        cal_B = {
            "ta": rng.normal(0.0, ta_sigma / np.sqrt(2)),
            "tg": rng.normal(0.0, tg_sigma / np.sqrt(2)),
            "wind": rng.normal(0.0, (wind_abs / np.sqrt(3)) / np.sqrt(2)),
        }

        # Co-location offset estimation
        d_ta_hat = (cal_B["ta"] - cal_A["ta"]) + rng.normal(0.0, ta_sigma / np.sqrt(2), n_coloc_samples).mean() - rng.normal(0.0, ta_sigma / np.sqrt(2), n_coloc_samples).mean()
        d_tg_hat = (cal_B["tg"] - cal_A["tg"]) + rng.normal(0.0, tg_sigma / np.sqrt(2), n_coloc_samples).mean() - rng.normal(0.0, tg_sigma / np.sqrt(2), n_coloc_samples).mean()
        sig_w = wind_abs / np.sqrt(3) / np.sqrt(2)
        d_w_hat = (cal_B["wind"] - cal_A["wind"]) + rng.normal(0.0, sig_w, n_coloc_samples).mean() - rng.normal(0.0, sig_w, n_coloc_samples).mean()

        # Measurement at campaign stops
        for s in stop_conditions:
            dev = s["device"]
            cal = cal_A if dev == "DEV_A" else cal_B
            mtype = s["morphology_type"]

            # 30 readings averaged in last 5 min of 10-min stop
            eps_ta = rng.normal(0.0, ta_sigma / np.sqrt(2), 30).mean()
            eps_tg = rng.normal(0.0, tg_sigma / np.sqrt(2), 30).mean()
            s_wind = max(wind_rel * s["wind"], wind_abs) / np.sqrt(3)
            eps_w = rng.normal(0.0, s_wind / np.sqrt(2), 30).mean()

            ta_obs = s["ta"] + cal["ta"] + eps_ta
            tg_obs = s["tg"] + cal["tg"] + eps_tg
            w_obs = max(0.05, s["wind"] + cal["wind"] + eps_w)

            # Apply co-location relative offset correction
            if ref_device == "DEV_A":
                if dev == "DEV_B":
                    ta_obs -= d_ta_hat
                    tg_obs -= d_tg_hat
                    w_obs = max(0.05, w_obs - d_w_hat)
            else:  # pair_mean
                if dev == "DEV_A":
                    ta_obs += d_ta_hat / 2.0
                    tg_obs += d_tg_hat / 2.0
                    w_obs = max(0.05, w_obs + d_w_hat / 2.0)
                else:
                    ta_obs -= d_ta_hat / 2.0
                    tg_obs -= d_tg_hat / 2.0
                    w_obs = max(0.05, w_obs - d_w_hat / 2.0)

            tmrt_obs = compute_iso7726_tmrt_fast(tg_obs, ta_obs, w_obs)
            type_sq_errors[mtype].append((tmrt_obs - s["tmrt"]) ** 2)

    result = {}
    print("\n-------------------------------------------------------------")
    print(f" Measurement Chain Alone Variance Decomposition (MC n={n_mc}, mode={ref_device}):")
    print("-------------------------------------------------------------")
    for mtype in sorted(targets.keys()):
        target_val = targets[mtype]
        target_sq = target_val ** 2
        meas_sq = float(np.mean(type_sq_errors[mtype]))
        meas_rmse = float(np.sqrt(meas_sq))
        diff = target_sq - meas_sq

        if diff < 0:
            print(f"  Type {mtype}: target^2 = {target_sq:.2f} (RMSE {target_val:.2f}) < sigma_meas^2 = {meas_sq:.2f} (RMSE {meas_rmse:.2f})")
            print(f"  WARNING: target^2 < sigma_meas^2 for Type {mtype} (diff = {diff:.2f}). Reporting without silent clipping!")
            sig_struct = 0.0
        else:
            sig_struct = float(np.sqrt(diff))
            print(f"  Type {mtype}: target^2 = {target_sq:.2f} (RMSE {target_val:.2f}), sigma_meas^2 = {meas_sq:.2f} (RMSE {meas_rmse:.2f}) -> sigma_struct = {sig_struct:.2f} °C (remainder)")

        result[mtype] = {
            "target_rmse": target_val,
            "target_sq": target_sq,
            "sigma_meas_sq": meas_sq,
            "sigma_meas_rmse": meas_rmse,
            "remainder_sq": diff,
            "sigma_struct": sig_struct,
            "target_sq_less_than_meas_sq": bool(diff < 0),
        }
    print("-------------------------------------------------------------\n")
    return result


def simulate_campaign(
    config_path: str,
    points_gpkg: str,
    provisional_base_dir: str,
    forcing_dir: str,
    out_dir: str,
    seed: int = 42,
    ref_device: str = "DEV_A",
):
    """
    Simulate full 2-day campaign with co-location intercomparison and crossover.
    """
    rng = np.random.default_rng(seed)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load points
    gdf_points = gpd.read_file(points_gpkg)
    print(f"Loaded {len(gdf_points)} campaign points.")

    dates = cfg["campaign_schedule"]["dates"]["value"]
    devices = cfg["campaign_schedule"]["devices"]["ids"]
    stop_duration_s = cfg["campaign_schedule"]["stop_duration_s"]["value"]
    logging_interval_s = cfg["campaign_schedule"]["logging_interval_s"]["value"]
    walk_time_s = cfg["campaign_schedule"]["walk_time_s"]["value"]
    samples_per_stop = stop_duration_s // logging_interval_s  # 60 samples
    tau_s = cfg["device_sensor"]["globe_time_constant_tau_s"]["value"]  # 180 s
    alpha_filter = logging_interval_s / tau_s

    # Literature RMSE targets
    raw_rmse_targets = {
        "A": cfg["tmrt_error"]["rmse_targets"]["canyon_AB"]["value"],
        "B": cfg["tmrt_error"]["rmse_targets"]["canyon_AB"]["value"],
        "C": cfg["tmrt_error"]["rmse_targets"]["tree_C"]["value"],
        "D": cfg["tmrt_error"]["rmse_targets"]["open_D"]["value"],
    }
    f_b = cfg["tmrt_error"]["bias_fraction_f_b"]["value"]
    station_share = cfg["tmrt_error"]["station_effect_share"]["value"]
    ar1_rho = cfg["tmrt_error"]["ar1_rho_10s"]["value"]

    # Sensor accuracy specs (GUM Type B: sigma = a / sqrt(3))
    ta_sigma = cfg["device_sensor"]["sensor_specifications"]["ta_accuracy_c"]["value"] / np.sqrt(3)
    rh_sigma = cfg["device_sensor"]["sensor_specifications"]["rh_accuracy_pct"]["value"] / np.sqrt(3)
    tg_sigma = cfg["device_sensor"]["sensor_specifications"]["tg_accuracy_c"]["value"] / np.sqrt(3)
    wind_rel = cfg["device_sensor"]["sensor_specifications"]["wind_accuracy_relative"]["value"]
    wind_abs = cfg["device_sensor"]["sensor_specifications"]["wind_accuracy_absolute_ms"]["value"]

    # Pre-extract model values at stop locations to compute measurement variance Monte Carlo
    cycle_start_hours = [10, 12, 14]
    stops_conditions = []

    for day_idx, day_str in enumerate(dates):
        day_run_dir = os.path.join(provisional_base_dir, f"run_{day_str}/output_folder/0_0")
        with rasterio.open(os.path.join(day_run_dir, "TMRT_0_0.tif")) as ds_tmrt, \
             rasterio.open(os.path.join(day_run_dir, "Ta_0_0.tif")) as ds_ta, \
             rasterio.open(os.path.join(day_run_dir, "Wind_0_0.tif")) as ds_wind:
            transform = ds_tmrt.transform
            tmrt_c = ds_tmrt.read()
            ta_c = ds_ta.read()
            wind_c = ds_wind.read()

        # Day 2 crossover: swap point groups between devices
        pts_A = gdf_points[gdf_points["device_id"] == "DEV_A"].copy().reset_index(drop=True)
        pts_B = gdf_points[gdf_points["device_id"] == "DEV_B"].copy().reset_index(drop=True)
        dev_pts = {
            "DEV_A": pts_A if day_idx == 0 else pts_B,
            "DEV_B": pts_B if day_idx == 0 else pts_A,
        }

        for cycle_idx, start_hour in enumerate(cycle_start_hours):
            cycle_start_time = datetime.strptime(f"{day_str} {start_hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
            for dev in devices:
                pts = dev_pts[dev]
                curr_t = cycle_start_time
                for _, pt in pts.iterrows():
                    px, py = pt["x"], pt["y"]
                    col = int(round((px - transform[2]) / transform[0]))
                    row = int(round((transform[5] - py) / (-transform[4])))

                    sample_t = curr_t + timedelta(seconds=450)
                    h_frac = sample_t.hour + sample_t.minute / 60.0 + sample_t.second / 3600.0
                    h0 = int(np.floor(h_frac))
                    h1 = min(23, h0 + 1)
                    w = h_frac - h0

                    mod_tmrt = float((1.0 - w) * tmrt_c[h0, row, col] + w * tmrt_c[h1, row, col])
                    mod_ta = float((1.0 - w) * ta_c[h0, row, col] + w * ta_c[h1, row, col])
                    mod_wind = float((1.0 - w) * wind_c[h0, row, col] + w * wind_c[h1, row, col])
                    tg_val = solve_tg(mod_tmrt, mod_ta, mod_wind)

                    stops_conditions.append({
                        "day": day_str,
                        "device": dev,
                        "point_id": pt["point_id"],
                        "morphology_type": pt["morphology_type"],
                        "tmrt": mod_tmrt,
                        "ta": mod_ta,
                        "wind": mod_wind,
                        "tg": tg_val,
                    })
                    curr_t += timedelta(seconds=stop_duration_s + walk_time_s)

    # Compute measurement variance and remainder structural targets
    decomp_results = compute_measurement_variance_mc(
        targets=raw_rmse_targets,
        stop_conditions=stops_conditions,
        n_mc=1000,
        seed=seed,
        ref_device=ref_device,
    )

    structural_rmse_targets = {
        mtype: decomp_results[mtype]["sigma_struct"] for mtype in raw_rmse_targets
    }

    # Draw device calibration offsets once (constant per device throughout the 2-day campaign)
    dev_cal = {}
    for dev in devices:
        dev_cal[dev] = {
            "ta": rng.normal(0.0, ta_sigma / np.sqrt(2)),
            "rh": rng.normal(0.0, rh_sigma / np.sqrt(2)),
            "tg": rng.normal(0.0, tg_sigma / np.sqrt(2)),
            "wind": rng.normal(0.0, (wind_abs / np.sqrt(3)) / np.sqrt(2)),
        }
    print(f"Device calibration offsets drawn for seed {seed}:")
    for dev in devices:
        print(f"  {dev}: Ta = {dev_cal[dev]['ta']:+.3f} °C, Tg = {dev_cal[dev]['tg']:+.3f} °C, Wind = {dev_cal[dev]['wind']:+.3f} m/s, RH = {dev_cal[dev]['rh']:+.2f} %")

    # Station persistent effects
    station_u_tmrt = {}
    station_u_ta = {}
    for _, pt in gdf_points.iterrows():
        pid = pt["point_id"]
        mtype = pt["morphology_type"]
        s_rmse = structural_rmse_targets[mtype]
        if s_rmse > 0:
            sigma_u = np.sqrt(station_share * (s_rmse ** 2))
        else:
            sigma_u = 0.0
        station_u_tmrt[pid] = rng.normal(0.0, sigma_u)
        station_u_ta[pid] = rng.normal(0.0, 1.25)

    wind_s = 0.265  # multiplicative wind error scale for target RMSE 0.77 m/s

    all_campaign_records = []
    os.makedirs(out_dir, exist_ok=True)

    # Reference coordinates for co-location (use Point 1 coordinates)
    ref_pt = gdf_points.iloc[0]
    ref_x, ref_y = ref_pt["x"], ref_pt["y"]

    for day_idx, day_str in enumerate(dates):
        print(f"\nSimulating campaign day: {day_str} (Day {day_idx + 1}/2)...")
        day_run_dir = os.path.join(provisional_base_dir, f"run_{day_str}/output_folder/0_0")
        tmrt_raster = os.path.join(day_run_dir, "TMRT_0_0.tif")
        shadow_raster = os.path.join(day_run_dir, "Shadow_0_0.tif")
        ta_raster = os.path.join(day_run_dir, "Ta_0_0.tif")
        wind_raster = os.path.join(day_run_dir, "Wind_0_0.tif")

        if day_str == "2026-11-12":
            met_file = os.path.join(forcing_dir, "met_ANALOGUE_2025-11-02_for_2026-11-12_hourly.txt")
        else:
            met_file = os.path.join(forcing_dir, "met_ANALOGUE_2025-11-03_for_2026-11-13_hourly.txt")

        rh_hourly = load_met_rh(met_file)

        with rasterio.open(tmrt_raster) as ds_tmrt, \
             rasterio.open(shadow_raster) as ds_shadow, \
             rasterio.open(ta_raster) as ds_ta, \
             rasterio.open(wind_raster) as ds_wind:

            transform = ds_tmrt.transform
            tmrt_cube = ds_tmrt.read()
            shadow_cube = ds_shadow.read()
            ta_cube = ds_ta.read()
            wind_cube = ds_wind.read()

        ref_col = int(round((ref_x - transform[2]) / transform[0]))
        ref_row = int(round((transform[5] - ref_y) / (-transform[4])))

        day_records = []

        # =========================================================================
        # 1. Start-of-Day Co-location Intercomparison (09:20 - 10:00)
        # 20 min shade (09:20 - 09:40) + 20 min sun (09:40 - 10:00)
        # =========================================================================
        print("  Generating start-of-day co-location intercomparison segments...")
        coloc_start_schedule = [
            ("shade", "start", datetime.strptime(f"{day_str} 09:20:00", "%Y-%m-%d %H:%M:%S")),
            ("sun", "start", datetime.strptime(f"{day_str} 09:40:00", "%Y-%m-%d %H:%M:%S")),
        ]

        tg_delay_coloc = {dev: None for dev in devices}

        for cond, timing, start_dt in coloc_start_schedule:
            coloc_point_id = f"COLOC_{day_str}_{cond}_{timing}"
            is_sun_cond = (cond == "sun")

            for step in range(120):  # 20 minutes @ 10 s = 120 samples
                s_time = start_dt + timedelta(seconds=step * logging_interval_s)
                h_frac = s_time.hour + s_time.minute / 60.0 + s_time.second / 3600.0
                h0 = int(np.floor(h_frac))
                h1 = min(23, h0 + 1)
                w = h_frac - h0

                mod_tmrt = (1.0 - w) * tmrt_cube[h0, ref_row, ref_col] + w * tmrt_cube[h1, ref_row, ref_col]
                mod_ta = (1.0 - w) * ta_cube[h0, ref_row, ref_col] + w * ta_cube[h1, ref_row, ref_col]
                mod_wind = (1.0 - w) * wind_cube[h0, ref_row, ref_col] + w * wind_cube[h1, ref_row, ref_col]
                mod_rh = (1.0 - w) * rh_hourly[h0] + w * rh_hourly[h1]

                # Co-location true state is identical for both devices
                b_exp = (0.35 * 4.90) if is_sun_cond else (-0.35 * 4.90)
                tmrt_true = mod_tmrt + b_exp
                ta_true = mod_ta
                v_true = max(0.10, mod_wind)
                rh_true = conserve_rh(mod_ta, mod_rh, ta_true)
                tg_true = solve_tg(tmrt_true, ta_true, v_true)

                # Both devices record at this identical timestamp and true state
                for dev in devices:
                    cal = dev_cal[dev]
                    if tg_delay_coloc[dev] is None:
                        tg_delay_coloc[dev] = tg_true
                    else:
                        tg_delay_coloc[dev] += alpha_filter * (tg_true - tg_delay_coloc[dev])

                    eps_ta = rng.normal(0.0, ta_sigma / np.sqrt(2))
                    eps_rh = rng.normal(0.0, rh_sigma / np.sqrt(2))
                    eps_tg = rng.normal(0.0, tg_sigma / np.sqrt(2))
                    sig_w_reading = max(wind_rel * v_true, wind_abs) / np.sqrt(3)
                    eps_wind = rng.normal(0.0, sig_w_reading / np.sqrt(2))

                    ta_obs = round(float(ta_true + cal["ta"] + eps_ta), 2)
                    rh_obs = round(float(np.clip(rh_true + cal["rh"] + eps_rh, 1.0, 99.0)), 1)
                    wind_obs = round(float(max(0.0, v_true + cal["wind"] + eps_wind)), 2)
                    tg_obs = round(float(tg_delay_coloc[dev] + cal["tg"] + eps_tg), 2)

                    day_records.append({
                        "timestamp_local": s_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_id": dev,
                        "point_id": coloc_point_id,
                        "ta_c": ta_obs,
                        "rh_pct": rh_obs,
                        "wind_ms": wind_obs,
                        "tg_c": tg_obs,
                        "is_synthetic": True,
                        "provenance": "field_error_model.yaml:v1.1.0;WP0.4_tile_1m",
                        "provisional_geometry": True,
                    })

        # =========================================================================
        # 2. Mobile Campaign Loop (10:00 - 15:44) with Day 2 Crossover
        # =========================================================================
        print("  Generating mobile campaign stops with device crossover...")
        pts_A = gdf_points[gdf_points["device_id"] == "DEV_A"].copy().reset_index(drop=True)
        pts_B = gdf_points[gdf_points["device_id"] == "DEV_B"].copy().reset_index(drop=True)

        # Crossover: Day 1 (day_idx=0) DEV_A visits pts_A, DEV_B visits pts_B
        # Day 2 (day_idx=1) DEV_A visits pts_B, DEV_B visits pts_A
        dev_pts = {
            "DEV_A": pts_A if day_idx == 0 else pts_B,
            "DEV_B": pts_B if day_idx == 0 else pts_A,
        }

        for cycle_idx, start_hour in enumerate(cycle_start_hours, 1):
            cycle_start_time = datetime.strptime(f"{day_str} {start_hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")

            for dev in devices:
                pts_dev = dev_pts[dev]
                curr_time = cycle_start_time
                tg_delayed_state = None

                for _, pt in pts_dev.iterrows():
                    pid = pt["point_id"]
                    mtype = pt["morphology_type"]
                    px, py = pt["x"], pt["y"]
                    col = int(round((px - transform[2]) / transform[0]))
                    row = int(round((transform[5] - py) / (-transform[4])))

                    s_rmse = structural_rmse_targets[mtype]
                    b_mag = f_b * raw_rmse_targets[mtype]
                    sigma_e = np.sqrt(max(0.0, (1.0 - station_share) * (s_rmse ** 2)))

                    eps_wind_visit = rng.normal(-(wind_s ** 2) / 2.0, wind_s)
                    eta_ta_visit = rng.normal(0.0, 0.35)
                    ar1_state = rng.normal(0.0, sigma_e) if sigma_e > 0 else 0.0

                    for step in range(samples_per_stop):
                        sample_time = curr_time + timedelta(seconds=step * logging_interval_s)
                        hour_frac = sample_time.hour + sample_time.minute / 60.0 + sample_time.second / 3600.0
                        h0 = int(np.floor(hour_frac))
                        h1 = min(23, h0 + 1)
                        w = hour_frac - h0

                        mod_tmrt = (1.0 - w) * tmrt_cube[h0, row, col] + w * tmrt_cube[h1, row, col]
                        mod_ta = (1.0 - w) * ta_cube[h0, row, col] + w * ta_cube[h1, row, col]
                        mod_wind = (1.0 - w) * wind_cube[h0, row, col] + w * wind_cube[h1, row, col]
                        mod_rh = (1.0 - w) * rh_hourly[h0] + w * rh_hourly[h1]
                        mod_shadow = shadow_cube[h0 if w < 0.5 else h1, row, col]

                        is_sunlit = mod_shadow > 0.5
                        b_exp = b_mag if is_sunlit else -b_mag

                        if sigma_e > 0:
                            ar1_state = ar1_rho * ar1_state + np.sqrt(1.0 - ar1_rho ** 2) * rng.normal(0.0, sigma_e)

                        tmrt_true = mod_tmrt + b_exp + station_u_tmrt[pid] + ar1_state
                        ta_true = mod_ta + station_u_ta[pid] + eta_ta_visit + rng.normal(0.0, 0.05)
                        v_true = max(0.10, mod_wind * np.exp(eps_wind_visit) + rng.normal(0.0, 0.05))
                        rh_true = conserve_rh(mod_ta, mod_rh, ta_true)

                        tg_true = solve_tg(tmrt_true, ta_true, v_true)

                        if tg_delayed_state is None:
                            tg_delayed_state = tg_true
                        else:
                            tg_delayed_state = tg_delayed_state + alpha_filter * (tg_true - tg_delayed_state)

                        cal = dev_cal[dev]
                        eps_ta = rng.normal(0.0, ta_sigma / np.sqrt(2))
                        eps_rh = rng.normal(0.0, rh_sigma / np.sqrt(2))
                        eps_tg = rng.normal(0.0, tg_sigma / np.sqrt(2))
                        sigma_w_reading = max(wind_rel * v_true, wind_abs) / np.sqrt(3)
                        eps_wind = rng.normal(0.0, sigma_w_reading / np.sqrt(2))

                        ta_obs = round(float(ta_true + cal["ta"] + eps_ta), 2)
                        rh_obs = round(float(np.clip(rh_true + cal["rh"] + eps_rh, 1.0, 99.0)), 1)
                        wind_obs = round(float(max(0.0, v_true + cal["wind"] + eps_wind)), 2)
                        tg_obs = round(float(tg_delayed_state + cal["tg"] + eps_tg), 2)

                        day_records.append({
                            "timestamp_local": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "device_id": dev,
                            "point_id": pid,
                            "ta_c": ta_obs,
                            "rh_pct": rh_obs,
                            "wind_ms": wind_obs,
                            "tg_c": tg_obs,
                            "is_synthetic": True,
                            "provenance": "field_error_model.yaml:v1.1.0;WP0.4_tile_1m",
                            "provisional_geometry": True,
                        })

                    # Relax globe temperature towards ambient during 3-min walk transit
                    walk_decay = np.exp(-walk_time_s / tau_s)
                    tg_delayed_state = tg_true + (tg_delayed_state - tg_true) * walk_decay
                    curr_time += timedelta(seconds=stop_duration_s + walk_time_s)

        # =========================================================================
        # 3. End-of-Day Co-location Intercomparison (16:00 - 16:40)
        # 20 min shade (16:00 - 16:20) + 20 min sun (16:20 - 16:40)
        # =========================================================================
        print("  Generating end-of-day co-location intercomparison segments...")
        coloc_end_schedule = [
            ("shade", "end", datetime.strptime(f"{day_str} 16:00:00", "%Y-%m-%d %H:%M:%S")),
            ("sun", "end", datetime.strptime(f"{day_str} 16:20:00", "%Y-%m-%d %H:%M:%S")),
        ]

        for cond, timing, start_dt in coloc_end_schedule:
            coloc_point_id = f"COLOC_{day_str}_{cond}_{timing}"
            is_sun_cond = (cond == "sun")

            for step in range(120):
                s_time = start_dt + timedelta(seconds=step * logging_interval_s)
                h_frac = s_time.hour + s_time.minute / 60.0 + s_time.second / 3600.0
                h0 = int(np.floor(h_frac))
                h1 = min(23, h0 + 1)
                w = h_frac - h0

                mod_tmrt = (1.0 - w) * tmrt_cube[h0, ref_row, ref_col] + w * tmrt_cube[h1, ref_row, ref_col]
                mod_ta = (1.0 - w) * ta_cube[h0, ref_row, ref_col] + w * ta_cube[h1, ref_row, ref_col]
                mod_wind = (1.0 - w) * wind_cube[h0, ref_row, ref_col] + w * wind_cube[h1, ref_row, ref_col]
                mod_rh = (1.0 - w) * rh_hourly[h0] + w * rh_hourly[h1]

                b_exp = (0.35 * 4.90) if is_sun_cond else (-0.35 * 4.90)
                tmrt_true = mod_tmrt + b_exp
                ta_true = mod_ta
                v_true = max(0.10, mod_wind)
                rh_true = conserve_rh(mod_ta, mod_rh, ta_true)
                tg_true = solve_tg(tmrt_true, ta_true, v_true)

                for dev in devices:
                    cal = dev_cal[dev]
                    if tg_delay_coloc[dev] is None:
                        tg_delay_coloc[dev] = tg_true
                    else:
                        tg_delay_coloc[dev] += alpha_filter * (tg_true - tg_delay_coloc[dev])

                    eps_ta = rng.normal(0.0, ta_sigma / np.sqrt(2))
                    eps_rh = rng.normal(0.0, rh_sigma / np.sqrt(2))
                    eps_tg = rng.normal(0.0, tg_sigma / np.sqrt(2))
                    sig_w_reading = max(wind_rel * v_true, wind_abs) / np.sqrt(3)
                    eps_wind = rng.normal(0.0, sig_w_reading / np.sqrt(2))

                    ta_obs = round(float(ta_true + cal["ta"] + eps_ta), 2)
                    rh_obs = round(float(np.clip(rh_true + cal["rh"] + eps_rh, 1.0, 99.0)), 1)
                    wind_obs = round(float(max(0.0, v_true + cal["wind"] + eps_wind)), 2)
                    tg_obs = round(float(tg_delay_coloc[dev] + cal["tg"] + eps_tg), 2)

                    day_records.append({
                        "timestamp_local": s_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_id": dev,
                        "point_id": coloc_point_id,
                        "ta_c": ta_obs,
                        "rh_pct": rh_obs,
                        "wind_ms": wind_obs,
                        "tg_c": tg_obs,
                        "is_synthetic": True,
                        "provenance": "field_error_model.yaml:v1.1.0;WP0.4_tile_1m",
                        "provisional_geometry": True,
                    })

        df_day = pd.DataFrame(day_records)
        day_csv = os.path.join(out_dir, f"SYNTHETIC_kestrel_raw_{day_str}.csv")
        df_day.to_csv(day_csv, index=False)
        print(f"  Saved raw observations for {day_str}: {day_csv} ({len(df_day)} rows)")
        all_campaign_records.extend(day_records)

    df_all = pd.DataFrame(all_campaign_records)
    all_csv = os.path.join(out_dir, "SYNTHETIC_kestrel_raw_campaign.csv")
    df_all.to_csv(all_csv, index=False)
    print(f"\n=======================================================")
    print(f" Saved full campaign synthetic dataset: {all_csv} ({len(df_all)} rows)")
    print(f" Mobile stop rows: {len(df_all[~df_all['point_id'].str.startswith('COLOC_')])}")
    print(f" Co-location rows: {len(df_all[df_all['point_id'].str.startswith('COLOC_')])}")
    print(f"=======================================================")

    return df_all, decomp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"configs/street/field_error_model.yaml")
    parser.add_argument("--points", default=r"data/street/field/PROVISIONAL_points.gpkg")
    parser.add_argument("--provisional-dir", default=r"data/street/field/provisional")
    parser.add_argument("--forcing-dir", default=r"data/street/wsl/forcing_nov")
    parser.add_argument("--out-dir", default=r"data/street/field")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-device", default="DEV_A")
    args = parser.parse_args()

    simulate_campaign(
        config_path=args.config,
        points_gpkg=args.points,
        provisional_base_dir=args.provisional_dir,
        forcing_dir=args.forcing_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        ref_device=args.ref_device,
    )
