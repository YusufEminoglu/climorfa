"""
CLIMORFA Street — Calibration Acceptance Check (v1.1).
Authoritative specification: docs/climorfa_street/field_error_model.md §3.4 v1.1.

Evaluates the end-to-end synthetic campaign generator over >= 200 seeds:
1. Computes expected values and 5th–95th percentile ranges for:
   - Tmrt RMSE by morphology type (A, B, C, D) and overall
   - UTCI RMSE overall
   - Air temperature (Ta) RMSE
   - Wind speed RMSE
   - Exposure systematic bias signs (Sunlit < 0, Shaded > 0 for model - obs)
2. Reports PASS/FAIL honestly per criterion without reinterpretation.
3. Selects the placeholder seed whose composite RMSE is closest to the median.
4. Regenerates the authoritative synthetic raw observations and processed outputs for that chosen seed.

Outputs:
outputs/street/field/SYNTHETIC_calibration_check_200seeds.json
outputs/street/field/SYNTHETIC_calibration_check_200seeds.csv
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time
from datetime import datetime, timedelta
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pythermalcomfort.models as ptm
from scipy.optimize import brentq

from scripts.street.field.simulate_campaign import (
    solve_tg,
    conserve_rh,
    load_met_rh,
    compute_iso7726_tmrt_fast,
    simulate_campaign,
)
from scripts.street.field.process_campaign import (
    extrapolate_wind_10m,
    process_campaign,
)


def run_calibration_check(
    config_path: str = "configs/street/field_error_model.yaml",
    points_gpkg: str = "data/street/field/PROVISIONAL_points.gpkg",
    provisional_base_dir: str = "data/street/field/provisional",
    forcing_dir: str = "data/street/wsl/forcing_nov",
    n_seeds: int = 200,
    base_seed: int = 100,
    out_dir: str = "outputs/street/field",
    data_dir: str = "data/street/field",
    ref_device: str = "DEV_A",
):
    print("=======================================================")
    print(f" Starting Calibration Acceptance Check over {n_seeds} seeds")
    print(f" Specification: docs/climorfa_street/field_error_model.md §3.4 v1.1")
    print("=======================================================\n")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gdf_points = gpd.read_file(points_gpkg)
    dates = cfg["campaign_schedule"]["dates"]["value"]
    devices = cfg["campaign_schedule"]["devices"]["ids"]
    stop_duration_s = cfg["campaign_schedule"]["stop_duration_s"]["value"]
    logging_interval_s = cfg["campaign_schedule"]["logging_interval_s"]["value"]
    walk_time_s = cfg["campaign_schedule"]["walk_time_s"]["value"]
    samples_per_stop = stop_duration_s // logging_interval_s  # 60
    tau_s = cfg["device_sensor"]["globe_time_constant_tau_s"]["value"]  # 180 s
    alpha_filter = logging_interval_s / tau_s

    raw_rmse_targets = {
        "A": cfg["tmrt_error"]["rmse_targets"]["canyon_AB"]["value"],
        "B": cfg["tmrt_error"]["rmse_targets"]["canyon_AB"]["value"],
        "C": cfg["tmrt_error"]["rmse_targets"]["tree_C"]["value"],
        "D": cfg["tmrt_error"]["rmse_targets"]["open_D"]["value"],
    }
    f_b = cfg["tmrt_error"]["bias_fraction_f_b"]["value"]
    station_share = cfg["tmrt_error"]["station_effect_share"]["value"]
    ar1_rho = cfg["tmrt_error"]["ar1_rho_10s"]["value"]

    ta_sigma = cfg["device_sensor"]["sensor_specifications"]["ta_accuracy_c"]["value"] / np.sqrt(3)
    rh_sigma = cfg["device_sensor"]["sensor_specifications"]["rh_accuracy_pct"]["value"] / np.sqrt(3)
    tg_sigma = cfg["device_sensor"]["sensor_specifications"]["tg_accuracy_c"]["value"] / np.sqrt(3)
    wind_rel = cfg["device_sensor"]["sensor_specifications"]["wind_accuracy_relative"]["value"]
    wind_abs = cfg["device_sensor"]["sensor_specifications"]["wind_accuracy_absolute_ms"]["value"]

    # Pre-load rasters and coordinates into memory
    cached_day_data = {}
    for day_str in dates:
        day_run_dir = os.path.join(provisional_base_dir, f"run_{day_str}/output_folder/0_0")
        with rasterio.open(os.path.join(day_run_dir, "TMRT_0_0.tif")) as ds_t, \
             rasterio.open(os.path.join(day_run_dir, "Shadow_0_0.tif")) as ds_s, \
             rasterio.open(os.path.join(day_run_dir, "Ta_0_0.tif")) as ds_a, \
             rasterio.open(os.path.join(day_run_dir, "Wind_0_0.tif")) as ds_w, \
             rasterio.open(os.path.join(day_run_dir, "UTCI_0_0.tif")) as ds_u:
            transform = ds_t.transform
            tmrt_c = ds_t.read()
            shadow_c = ds_s.read()
            ta_c = ds_a.read()
            wind_c = ds_w.read()
            utci_c = ds_u.read()

        if day_str == "2026-11-12":
            met_file = os.path.join(forcing_dir, "met_ANALOGUE_2025-11-02_for_2026-11-12_hourly.txt")
        else:
            met_file = os.path.join(forcing_dir, "met_ANALOGUE_2025-11-03_for_2026-11-13_hourly.txt")
        rh_hourly = load_met_rh(met_file)

        cached_day_data[day_str] = {
            "transform": transform,
            "tmrt": tmrt_c,
            "shadow": shadow_c,
            "ta": ta_c,
            "wind": wind_c,
            "utci": utci_c,
            "rh": rh_hourly,
        }

    # Reference co-location point (point 1)
    ref_pt = gdf_points.iloc[0]
    ref_x, ref_y = ref_pt["x"], ref_pt["y"]

    # Pre-extract stop conditions for measurement variance computation
    cycle_start_hours = [10, 12, 14]
    stops_conditions = []
    for day_idx, day_str in enumerate(dates):
        cdata = cached_day_data[day_str]
        transform = cdata["transform"]
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

                    mod_tmrt = float((1.0 - w) * cdata["tmrt"][h0, row, col] + w * cdata["tmrt"][h1, row, col])
                    mod_ta = float((1.0 - w) * cdata["ta"][h0, row, col] + w * cdata["ta"][h1, row, col])
                    mod_wind = float((1.0 - w) * cdata["wind"][h0, row, col] + w * cdata["wind"][h1, row, col])
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

    # Compute measurement variance MC
    from scripts.street.field.simulate_campaign import compute_measurement_variance_mc
    decomp = compute_measurement_variance_mc(
        targets=raw_rmse_targets,
        stop_conditions=stops_conditions,
        n_mc=1000,
        seed=base_seed,
        ref_device=ref_device,
    )
    structural_rmse_targets = {mtype: decomp[mtype]["sigma_struct"] for mtype in raw_rmse_targets}

    t0 = time.time()
    seed_records = []

    print(f"Running simulation + processing across {n_seeds} seeds...")
    for s_idx in range(n_seeds):
        curr_seed = base_seed + s_idx
        rng = np.random.default_rng(curr_seed)

        # Draw device calibration offsets once per campaign
        dev_cal = {}
        for dev in devices:
            dev_cal[dev] = {
                "ta": rng.normal(0.0, ta_sigma / np.sqrt(2)),
                "rh": rng.normal(0.0, rh_sigma / np.sqrt(2)),
                "tg": rng.normal(0.0, tg_sigma / np.sqrt(2)),
                "wind": rng.normal(0.0, (wind_abs / np.sqrt(3)) / np.sqrt(2)),
            }

        # Station effects
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

        # Simulate co-location segments (4 segments: start and end of day 1 and day 2)
        # 4 segments x 20 min (120 samples each) = 480 timestamps x 2 conditions = 960 samples total
        coloc_diff_ta = []
        coloc_diff_tg = []
        coloc_diff_w = []
        coloc_diff_rh = []

        for day_str in dates:
            for s_hour in [9.33, 16.0]:
                for is_sun in [False, True]:
                    for step in range(120):
                        # True state (shared)
                        v_true = 2.0
                        ta_true = 20.0
                        tg_true = 22.0

                        # DEV_A reading
                        eps_ta_a = rng.normal(0.0, ta_sigma / np.sqrt(2))
                        eps_tg_a = rng.normal(0.0, tg_sigma / np.sqrt(2))
                        eps_w_a = rng.normal(0.0, max(wind_rel * v_true, wind_abs) / np.sqrt(3) / np.sqrt(2))
                        eps_rh_a = rng.normal(0.0, rh_sigma / np.sqrt(2))

                        ta_a = ta_true + dev_cal["DEV_A"]["ta"] + eps_ta_a
                        tg_a = tg_true + dev_cal["DEV_A"]["tg"] + eps_tg_a
                        w_a = v_true + dev_cal["DEV_A"]["wind"] + eps_w_a
                        rh_a = 60.0 + dev_cal["DEV_A"]["rh"] + eps_rh_a

                        # DEV_B reading
                        eps_ta_b = rng.normal(0.0, ta_sigma / np.sqrt(2))
                        eps_tg_b = rng.normal(0.0, tg_sigma / np.sqrt(2))
                        eps_w_b = rng.normal(0.0, max(wind_rel * v_true, wind_abs) / np.sqrt(3) / np.sqrt(2))
                        eps_rh_b = rng.normal(0.0, rh_sigma / np.sqrt(2))

                        ta_b = ta_true + dev_cal["DEV_B"]["ta"] + eps_ta_b
                        tg_b = tg_true + dev_cal["DEV_B"]["tg"] + eps_tg_b
                        w_b = v_true + dev_cal["DEV_B"]["wind"] + eps_w_b
                        rh_b = 60.0 + dev_cal["DEV_B"]["rh"] + eps_rh_b

                        coloc_diff_ta.append(ta_b - ta_a)
                        coloc_diff_tg.append(tg_b - tg_a)
                        coloc_diff_w.append(w_b - w_a)
                        coloc_diff_rh.append(rh_b - rh_a)

        # Estimated relative offsets (DEV_B - DEV_A)
        d_ta_hat = float(np.mean(coloc_diff_ta))
        d_tg_hat = float(np.mean(coloc_diff_tg))
        d_w_hat = float(np.mean(coloc_diff_w))

        # Simulate 96 mobile stops
        stop_results = []
        for day_idx, day_str in enumerate(dates):
            cdata = cached_day_data[day_str]
            transform = cdata["transform"]
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
                    curr_time = cycle_start_time
                    tg_delayed = None

                    for _, pt in pts.iterrows():
                        pid = pt["point_id"]
                        mtype = pt["morphology_type"]
                        px, py = pt["x"], pt["y"]
                        col = int(round((px - transform[2]) / transform[0]))
                        row = int(round((transform[5] - py) / (-transform[4])))

                        s_rmse = structural_rmse_targets[mtype]
                        b_mag = f_b * raw_rmse_targets[mtype]
                        sigma_e = np.sqrt(max(0.0, (1.0 - station_share) * (s_rmse ** 2)))

                        eps_w_visit = rng.normal(-(0.265 ** 2) / 2.0, 0.265)
                        eta_ta_visit = rng.normal(0.0, 0.35)
                        ar1 = rng.normal(0.0, sigma_e) if sigma_e > 0 else 0.0

                        ta_obs_samples = []
                        tg_obs_samples = []
                        w_obs_samples = []
                        rh_obs_samples = []
                        mod_tmrt_samples = []
                        mod_ta_samples = []
                        mod_wind_samples = []
                        mod_utci_samples = []
                        mod_shadow_samples = []

                        for step in range(samples_per_stop):
                            sample_time = curr_time + timedelta(seconds=step * logging_interval_s)
                            h_frac = sample_time.hour + sample_time.minute / 60.0 + sample_time.second / 3600.0
                            h0 = int(np.floor(h_frac))
                            h1 = min(23, h0 + 1)
                            w = h_frac - h0

                            mod_tmrt = (1.0 - w) * cdata["tmrt"][h0, row, col] + w * cdata["tmrt"][h1, row, col]
                            mod_ta = (1.0 - w) * cdata["ta"][h0, row, col] + w * cdata["ta"][h1, row, col]
                            mod_wind = (1.0 - w) * cdata["wind"][h0, row, col] + w * cdata["wind"][h1, row, col]
                            mod_utci = (1.0 - w) * cdata["utci"][h0, row, col] + w * cdata["utci"][h1, row, col]
                            mod_rh = (1.0 - w) * cdata["rh"][h0] + w * cdata["rh"][h1]
                            mod_shad = cdata["shadow"][h0 if w < 0.5 else h1, row, col]

                            is_sun = mod_shad > 0.5
                            b_exp = b_mag if is_sun else -b_mag

                            if sigma_e > 0:
                                ar1 = ar1_rho * ar1 + np.sqrt(1.0 - ar1_rho ** 2) * rng.normal(0.0, sigma_e)

                            tmrt_true = mod_tmrt + b_exp + station_u_tmrt[pid] + ar1
                            ta_true = mod_ta + station_u_ta[pid] + eta_ta_visit + rng.normal(0.0, 0.05)
                            v_true = max(0.10, mod_wind * np.exp(eps_w_visit) + rng.normal(0.0, 0.05))
                            rh_true = conserve_rh(mod_ta, mod_rh, ta_true)
                            tg_true = solve_tg(tmrt_true, ta_true, v_true)

                            if tg_delayed is None:
                                tg_delayed = tg_true
                            else:
                                tg_delayed += alpha_filter * (tg_true - tg_delayed)

                            # Record only last 30 samples (second 5 min)
                            if step >= 30:
                                cal = dev_cal[dev]
                                eps_ta = rng.normal(0.0, ta_sigma / np.sqrt(2))
                                eps_rh = rng.normal(0.0, rh_sigma / np.sqrt(2))
                                eps_tg = rng.normal(0.0, tg_sigma / np.sqrt(2))
                                s_w = max(wind_rel * v_true, wind_abs) / np.sqrt(3)
                                eps_w = rng.normal(0.0, s_w / np.sqrt(2))

                                ta_obs_samples.append(ta_true + cal["ta"] + eps_ta)
                                tg_obs_samples.append(tg_delayed + cal["tg"] + eps_tg)
                                w_obs_samples.append(max(0.0, v_true + cal["wind"] + eps_w))
                                rh_obs_samples.append(np.clip(rh_true + cal["rh"] + eps_rh, 1.0, 99.0))

                                mod_tmrt_samples.append(mod_tmrt)
                                mod_ta_samples.append(mod_ta)
                                mod_wind_samples.append(mod_wind)
                                mod_utci_samples.append(mod_utci)
                                mod_shadow_samples.append(mod_shad)

                        # Mean of second 5 minutes
                        ta_m = float(np.mean(ta_obs_samples))
                        tg_m = float(np.mean(tg_obs_samples))
                        w_m = float(np.mean(w_obs_samples))
                        rh_m = float(np.mean(rh_obs_samples))

                        # Apply co-location relative offset correction (DEV_B adjusted to DEV_A)
                        if ref_device == "DEV_A":
                            if dev == "DEV_B":
                                ta_m -= d_ta_hat
                                tg_m -= d_tg_hat
                                w_m = max(0.05, w_m - d_w_hat)
                        else:  # pair_mean
                            if dev == "DEV_A":
                                ta_m += d_ta_hat / 2.0
                                tg_m += d_tg_hat / 2.0
                                w_m = max(0.05, w_m + d_w_hat / 2.0)
                            else:
                                ta_m -= d_ta_hat / 2.0
                                tg_m -= d_tg_hat / 2.0
                                w_m = max(0.05, w_m - d_w_hat / 2.0)

                        tmrt_obs = compute_iso7726_tmrt_fast(tg_c=tg_m, ta_c=ta_m, v_ms=w_m)
                        w10_obs = extrapolate_wind_10m(w_m)
                        try:
                            res_u = ptm.utci(tdb=ta_m, tr=tmrt_obs, v=w10_obs, rh=rh_m)
                            utci_obs = float(res_u.utci)
                        except Exception:
                            utci_obs = np.nan

                        mod_tmrt_m = float(np.mean(mod_tmrt_samples))
                        mod_ta_m = float(np.mean(mod_ta_samples))
                        mod_wind_m = float(np.mean(mod_wind_samples))
                        mod_utci_m = float(np.mean(mod_utci_samples))
                        is_sunlit_actual = float(np.mean(mod_shadow_samples)) > 0.5

                        stop_results.append({
                            "seed": curr_seed,
                            "type": mtype,
                            "is_sunlit": is_sunlit_actual,
                            "tmrt_diff": mod_tmrt_m - tmrt_obs,
                            "ta_diff": mod_ta_m - ta_m,
                            "wind_diff": mod_wind_m - w_m,
                            "utci_diff": (mod_utci_m - utci_obs) if not np.isnan(utci_obs) else np.nan,
                        })

                        # Relax globe temperature towards ambient during 3-min walk transit
                        walk_decay = np.exp(-walk_time_s / tau_s)
                        tg_delayed = tg_true + (tg_delayed - tg_true) * walk_decay
                        curr_time += timedelta(seconds=stop_duration_s + walk_time_s)

        df_stops_s = pd.DataFrame(stop_results)

        # Calculate metrics for this seed
        rmse_overall = float(np.sqrt(np.mean(df_stops_s["tmrt_diff"] ** 2)))
        rmse_A = float(np.sqrt(np.mean(df_stops_s[df_stops_s["type"] == "A"]["tmrt_diff"] ** 2)))
        rmse_B = float(np.sqrt(np.mean(df_stops_s[df_stops_s["type"] == "B"]["tmrt_diff"] ** 2)))
        rmse_C = float(np.sqrt(np.mean(df_stops_s[df_stops_s["type"] == "C"]["tmrt_diff"] ** 2)))
        rmse_D = float(np.sqrt(np.mean(df_stops_s[df_stops_s["type"] == "D"]["tmrt_diff"] ** 2)))

        clean_utci = df_stops_s["utci_diff"].dropna()
        rmse_utci = float(np.sqrt(np.mean(clean_utci ** 2)))
        rmse_ta = float(np.sqrt(np.mean(df_stops_s["ta_diff"] ** 2)))
        rmse_wind = float(np.sqrt(np.mean(df_stops_s["wind_diff"] ** 2)))

        bias_sun = float(np.mean(df_stops_s[df_stops_s["is_sunlit"] == True]["tmrt_diff"]))
        bias_sha = float(np.mean(df_stops_s[df_stops_s["is_sunlit"] == False]["tmrt_diff"]))

        seed_records.append({
            "seed": curr_seed,
            "rmse_overall": rmse_overall,
            "rmse_A": rmse_A,
            "rmse_B": rmse_B,
            "rmse_C": rmse_C,
            "rmse_D": rmse_D,
            "rmse_utci": rmse_utci,
            "rmse_ta": rmse_ta,
            "rmse_wind": rmse_wind,
            "bias_sunlit": bias_sun,
            "bias_shaded": bias_sha,
        })

    t_elapsed = time.time() - t0
    print(f"Completed {n_seeds} calibration runs in {t_elapsed:.2f} s ({t_elapsed/n_seeds*1000:.1f} ms/run).")

    df_eval = pd.DataFrame(seed_records)

    # Acceptance criteria evaluation
    criteria_def = [
        {"metric": "Type A Tmrt RMSE", "col": "rmse_A", "target": 4.90, "range": [4.41, 5.39], "unit": "°C", "rule": "±10% [4.41, 5.39]"},
        {"metric": "Type B Tmrt RMSE", "col": "rmse_B", "target": 4.90, "range": [4.41, 5.39], "unit": "°C", "rule": "±10% [4.41, 5.39]"},
        {"metric": "Type C Tmrt RMSE", "col": "rmse_C", "target": 6.10, "range": [5.49, 6.71], "unit": "°C", "rule": "±10% [5.49, 6.71]"},
        {"metric": "Type D Tmrt RMSE", "col": "rmse_D", "target": 7.00, "range": [6.30, 7.70], "unit": "°C", "rule": "±10% [6.30, 7.70]"},
        {"metric": "Overall UTCI RMSE", "col": "rmse_utci", "target": 2.84, "range": [2.00, 4.50], "unit": "°C", "rule": "in [2.0, 4.5]"},
        {"metric": "Air Temp Ta RMSE", "col": "rmse_ta", "target": 1.30, "range": [1.10, 1.50], "unit": "°C", "rule": "≈ 1.30"},
        {"metric": "Wind Speed RMSE", "col": "rmse_wind", "target": 0.77, "range": [0.65, 0.90], "unit": "m/s", "rule": "≈ 0.77"},
        {"metric": "Exposure Bias (Sunlit)", "col": "bias_sunlit", "target": "< 0", "range": [-100.0, 0.0], "unit": "°C", "rule": "Model - Obs < 0 (L2)"},
        {"metric": "Exposure Bias (Shaded)", "col": "bias_shaded", "target": "> 0", "range": [0.0, 100.0], "unit": "°C", "rule": "Model - Obs > 0 (L2)"},
    ]

    report_rows = []
    print("\n==========================================================================================")
    print(f" Calibration Acceptance Check Results (Expected Values & 5–95% Ranges over {n_seeds} Seeds):")
    print("==========================================================================================")

    all_passed = True
    for c in criteria_def:
        vals = df_eval[c["col"]].values
        ev = float(np.mean(vals))
        p05 = float(np.percentile(vals, 5.0))
        p95 = float(np.percentile(vals, 95.0))

        if c["col"] == "bias_sunlit":
            status = "PASS" if ev < 0.0 else "FAIL"
        elif c["col"] == "bias_shaded":
            status = "PASS" if ev > 0.0 else "FAIL"
        else:
            status = "PASS" if (c["range"][0] <= ev <= c["range"][1]) else "FAIL"

        if status == "FAIL":
            all_passed = False

        row = {
            "Criterion": c["metric"],
            "Expected Value": round(ev, 3),
            "5% Percentile": round(p05, 3),
            "95% Percentile": round(p95, 3),
            "Literature Target": c["target"] if isinstance(c["target"], str) else f"{c['target']:.2f} {c['unit']}",
            "Acceptance Band": c["rule"],
            "Status": status,
        }
        report_rows.append(row)
        print(f" {row['Criterion']:24s} | EV: {row['Expected Value']:+6.3f} | 5-95%: [{row['5% Percentile']:+6.3f}, {row['95% Percentile']:+6.3f}] | Band: {row['Acceptance Band']:20s} | Status: {row['Status']}")

    print("==========================================================================================\n")

    # Select placeholder seed: closest to median overall RMSE
    median_rmse = float(df_eval["rmse_overall"].median())
    df_eval["dist_to_median"] = np.abs(df_eval["rmse_overall"] - median_rmse)
    selected_row = df_eval.sort_values("dist_to_median").iloc[0]
    selected_seed = int(selected_row["seed"])

    print(f"Overall Tmrt RMSE Distribution:")
    print(f"  Mean:   {df_eval['rmse_overall'].mean():.3f} °C")
    print(f"  Median: {median_rmse:.3f} °C")
    print(f"  5-95%:  [{np.percentile(df_eval['rmse_overall'], 5):.3f}, {np.percentile(df_eval['rmse_overall'], 95):.3f}] °C")
    print(f"\n--> Selected Placeholder Seed per §3.4 v1.1: seed = {selected_seed} (RMSE = {selected_row['rmse_overall']:.3f} °C, dist to median = {selected_row['dist_to_median']:.4f} °C)")

    # Save summary tables
    df_report = pd.DataFrame(report_rows)
    out_csv = os.path.join(out_dir, "SYNTHETIC_calibration_check_200seeds.csv")
    out_json = os.path.join(out_dir, "SYNTHETIC_calibration_check_200seeds.json")
    os.makedirs(out_dir, exist_ok=True)
    df_report.to_csv(out_csv, index=False)

    summary_payload = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "selected_seed": selected_seed,
        "median_rmse_overall": median_rmse,
        "selected_seed_metrics": {
            "rmse_overall": round(float(selected_row["rmse_overall"]), 3),
            "rmse_A": round(float(selected_row["rmse_A"]), 3),
            "rmse_B": round(float(selected_row["rmse_B"]), 3),
            "rmse_C": round(float(selected_row["rmse_C"]), 3),
            "rmse_D": round(float(selected_row["rmse_D"]), 3),
            "rmse_utci": round(float(selected_row["rmse_utci"]), 3),
            "rmse_ta": round(float(selected_row["rmse_ta"]), 3),
            "rmse_wind": round(float(selected_row["rmse_wind"]), 3),
            "bias_sunlit": round(float(selected_row["bias_sunlit"]), 3),
            "bias_shaded": round(float(selected_row["bias_shaded"]), 3),
        },
        "all_criteria_passed": all_passed,
        "criteria_table": report_rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"Saved calibration check summary CSV:  {out_csv}")
    print(f"Saved calibration check summary JSON: {out_json}")

    # Regenerate official placeholder files using selected seed
    print(f"\nRegenerating official placeholder files with chosen seed {selected_seed}...")
    simulate_campaign(
        config_path=config_path,
        points_gpkg=points_gpkg,
        provisional_base_dir=provisional_base_dir,
        forcing_dir=forcing_dir,
        out_dir=data_dir,
        seed=selected_seed,
        ref_device=ref_device,
    )

    process_campaign(
        raw_csv=os.path.join(data_dir, "SYNTHETIC_kestrel_raw_campaign.csv"),
        points_gpkg=points_gpkg,
        provisional_base_dir=provisional_base_dir,
        out_stops_csv=os.path.join(data_dir, "SYNTHETIC_processed_stops.csv"),
        out_metrics_csv=os.path.join(out_dir, "SYNTHETIC_model_vs_obs_metrics.csv"),
        out_metrics_json=os.path.join(out_dir, "SYNTHETIC_model_vs_obs_metrics.json"),
        out_offsets_json=os.path.join(out_dir, "SYNTHETIC_device_offsets.json"),
        ref_device=ref_device,
    )

    print("\nAuthoritative placeholder regeneration completed successfully.")
    return df_report, summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"configs/street/field_error_model.yaml")
    parser.add_argument("--points", default=r"data/street/field/PROVISIONAL_points.gpkg")
    parser.add_argument("--provisional-dir", default=r"data/street/field/provisional")
    parser.add_argument("--forcing-dir", default=r"data/street/wsl/forcing_nov")
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--out-dir", default=r"outputs/street/field")
    parser.add_argument("--data-dir", default=r"data/street/field")
    parser.add_argument("--ref-device", default="DEV_A")
    args = parser.parse_args()

    run_calibration_check(
        config_path=args.config,
        points_gpkg=args.points,
        provisional_base_dir=args.provisional_dir,
        forcing_dir=args.forcing_dir,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        ref_device=args.ref_device,
    )
