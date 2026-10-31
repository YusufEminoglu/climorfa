"""
CLIMORFA Street — Equipment Scenarios Evaluation (WP0.5-A3).
Authoritative specification: docs/climorfa_street/wp/WP0.5A3_equipment.md
Standards and reference models:
- ISO 7726:1998 (Ergonomics of thermal environment — Instruments)
- Thorsson et al. (2007) Int. J. Climatol. 27:1983–1993 (38 mm flat-grey globe)
- IEC 60751 (Industrial platinum resistance thermometer sensors, Class A)
- Kipp & Zonen CNR4 / MaRTy 6-directional integral radiation reference method

Evaluates five equipment and calibration scenarios:
- S0: Kestrel 5400 x2, 25.4 mm black globe, relative co-location only (= A2 baseline).
- S1: S0 + isothermal-room calibration before and after campaign + co-location.
- S2: 38 mm flat-grey globe + IEC 60751 Class-A Pt100 + aspirated shielded Ta + cup/sonic + isothermal + co-location.
- S3: 150 mm standard black globe + S2 precision sensors + isothermal + co-location (20-min equilibration).
- S4: Reference: six-directional net radiometer method (integral radiation; sigma = 1.0 K).

Outputs:
outputs/street/field/SYNTHETIC_equipment_scenarios.csv
outputs/street/field/SYNTHETIC_equipment_scenarios.json
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
from functools import partial
from multiprocessing import Pool, cpu_count
import yaml
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from scripts.street.field.simulate_campaign import solve_tg, compute_iso7726_tmrt_fast

# Representative November microclimate conditions per morphology type
TYPE_CONDITIONS = {
    "A": {"tmrt": 25.4, "ta": 20.1, "wind": 2.18},
    "B": {"tmrt": 24.5, "ta": 20.8, "wind": 2.47},
    "C": {"tmrt": 27.8, "ta": 21.4, "wind": 2.76},
    "D": {"tmrt": 33.8, "ta": 21.6, "wind": 3.07},
}

# Literature target RMSEs
RMSE_TARGETS = {"A": 4.90, "B": 4.90, "C": 6.10, "D": 7.00}
F_B = 0.35
STATION_SHARE = 0.40


def compute_scenario_measurement_variance(
    scenario_cfg: dict,
    targets: dict = None,
    n_mc: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute sigma_meas^2 by Monte Carlo of the measurement chain alone (truth = model),
    and determine the structural remainder sigma_struct^2 = max(0, target^2 - sigma_meas^2).
    """
    if targets is None:
        targets = RMSE_TARGETS

    sc_id = scenario_cfg["id"]
    rng = np.random.default_rng(seed)
    types = ["A", "B", "C", "D"]
    type_sq_err = {t: [] for t in types}

    if sc_id == "S4":
        sig_rad = scenario_cfg["direct_radiation"]["tmrt_error_sigma_k"]
        for _ in range(n_mc):
            cal_offset = rng.normal(0.0, sig_rad / np.sqrt(2))
            for t in types:
                read_noise = rng.normal(0.0, sig_rad / np.sqrt(2))
                type_sq_err[t].append((cal_offset + read_noise) ** 2)
    else:
        d_globe = scenario_cfg["globe"]["diameter_m"]
        ta_sig = scenario_cfg["sensors"]["ta"]["sigma_c"]
        tg_sig = scenario_cfg["sensors"]["tg"]["sigma_c"]
        w_rel = scenario_cfg["sensors"]["wind"]["accuracy_rel"]
        w_abs = scenario_cfg["sensors"]["wind"]["accuracy_abs_ms"]
        is_iso = scenario_cfg["calibration"]["isothermal_chamber"]

        iso_ta_sig = scenario_cfg["calibration"].get("isothermal_repeatability_sigma_ta_c", 0.10)
        iso_diff_sig = scenario_cfg["calibration"].get("isothermal_repeatability_sigma_tg_minus_ta_c", 0.10)

        for _ in range(n_mc):
            if is_iso:
                cal_ta_A = rng.normal(0.0, iso_ta_sig)
                cal_tg_A = cal_ta_A + rng.normal(0.0, iso_diff_sig)
                cal_w_A = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

                cal_ta_B = rng.normal(0.0, iso_ta_sig)
                cal_tg_B = cal_ta_B + rng.normal(0.0, iso_diff_sig)
                cal_w_B = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))
            else:
                cal_ta_A = rng.normal(0.0, ta_sig / np.sqrt(2))
                cal_tg_A = rng.normal(0.0, tg_sig / np.sqrt(2))
                cal_w_A = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

                cal_ta_B = rng.normal(0.0, ta_sig / np.sqrt(2))
                cal_tg_B = rng.normal(0.0, tg_sig / np.sqrt(2))
                cal_w_B = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

            # Co-location relative offset (B - A) over 960 samples
            d_ta_hat = (cal_ta_B - cal_ta_A) + rng.normal(0.0, ta_sig / np.sqrt(2), 960).mean() - rng.normal(0.0, ta_sig / np.sqrt(2), 960).mean()
            d_tg_hat = (cal_tg_B - cal_tg_A) + rng.normal(0.0, tg_sig / np.sqrt(2), 960).mean() - rng.normal(0.0, tg_sig / np.sqrt(2), 960).mean()
            d_w_hat = (cal_w_B - cal_w_A)

            for dev in ["DEV_A", "DEV_B"]:
                cal = {"ta": cal_ta_A, "tg": cal_tg_A, "w": cal_w_A} if dev == "DEV_A" else {"ta": cal_ta_B, "tg": cal_tg_B, "w": cal_w_B}
                for t, cond in TYPE_CONDITIONS.items():
                    tg_true = solve_tg(cond["tmrt"], cond["ta"], cond["wind"], d=d_globe)

                    # 30-sample averaging window (5 minutes at 10s steps)
                    eps_ta = rng.normal(0.0, ta_sig / np.sqrt(2), 30).mean()
                    eps_tg = rng.normal(0.0, tg_sig / np.sqrt(2), 30).mean()
                    s_w = max(w_rel * cond["wind"], w_abs) / np.sqrt(3)
                    eps_w = rng.normal(0.0, s_w / np.sqrt(2), 30).mean()

                    ta_obs = cond["ta"] + cal["ta"] + eps_ta
                    tg_obs = tg_true + cal["tg"] + eps_tg
                    w_obs = max(0.05, cond["wind"] + cal["w"] + eps_w)

                    if dev == "DEV_B":
                        ta_obs -= d_ta_hat
                        tg_obs -= d_tg_hat
                        w_obs = max(0.05, w_obs - d_w_hat)

                    tmrt_obs = compute_iso7726_tmrt_fast(tg_obs, ta_obs, w_obs, d=d_globe)
                    type_sq_err[t].append((tmrt_obs - cond["tmrt"]) ** 2)

    decomp = {}
    for t in types:
        tgt_rmse = targets[t]
        tgt_sq = tgt_rmse ** 2
        m_sq = float(np.mean(type_sq_err[t]))
        m_rmse = float(np.sqrt(m_sq))
        rem_sq = tgt_sq - m_sq
        s_struct = float(np.sqrt(max(0.0, rem_sq)))
        decomp[t] = {
            "target_rmse": tgt_rmse,
            "target_sq": tgt_sq,
            "sigma_meas_sq": m_sq,
            "sigma_meas_rmse": m_rmse,
            "remainder_sq": rem_sq,
            "sigma_struct": s_struct,
            "unmodeled_measurement_dominance": bool(rem_sq < 0),
        }

    # Overall composite average
    all_sq = [v["sigma_meas_sq"] for v in decomp.values()]
    all_struct = [v["sigma_struct"] for v in decomp.values()]
    decomp["overall"] = {
        "sigma_meas_sq": float(np.mean(all_sq)),
        "sigma_meas_rmse": float(np.sqrt(np.mean(all_sq))),
        "sigma_struct": float(np.mean(all_struct)),
    }
    return decomp


def simulate_single_campaign(
    sc_cfg: dict,
    struct_targets: dict,
    n_points: int,
    visits_per_day: int,
    n_days: int,
    seed: int,
) -> dict:
    """
    Simulate one campaign realization for a scenario and sampling design.
    """
    rng = np.random.default_rng(seed)
    sc_id = sc_cfg["id"]
    is_s4 = (sc_id == "S4")
    types = ["A", "B", "C", "D"]
    pts_per_type = n_points // 4

    if is_s4:
        sig_rad = sc_cfg["direct_radiation"]["tmrt_error_sigma_k"]
        cal_offset = rng.normal(0.0, sig_rad / np.sqrt(2))
        cal_ta_A = cal_tg_A = cal_w_A = 0.0
        cal_ta_B = cal_tg_B = cal_w_B = 0.0
        d_ta_hat = d_tg_hat = d_w_hat = 0.0
        d_globe = 0.0
        ta_sig = tg_sig = 0.0
        w_rel = w_abs = 0.0
    else:
        d_globe = sc_cfg["globe"]["diameter_m"]
        ta_sig = sc_cfg["sensors"]["ta"]["sigma_c"]
        tg_sig = sc_cfg["sensors"]["tg"]["sigma_c"]
        w_rel = sc_cfg["sensors"]["wind"]["accuracy_rel"]
        w_abs = sc_cfg["sensors"]["wind"]["accuracy_abs_ms"]
        is_iso = sc_cfg["calibration"]["isothermal_chamber"]

        iso_ta_sig = sc_cfg["calibration"].get("isothermal_repeatability_sigma_ta_c", 0.10)
        iso_diff_sig = sc_cfg["calibration"].get("isothermal_repeatability_sigma_tg_minus_ta_c", 0.10)

        if is_iso:
            cal_ta_A = rng.normal(0.0, iso_ta_sig)
            cal_tg_A = cal_ta_A + rng.normal(0.0, iso_diff_sig)
            cal_w_A = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

            cal_ta_B = rng.normal(0.0, iso_ta_sig)
            cal_tg_B = cal_ta_B + rng.normal(0.0, iso_diff_sig)
            cal_w_B = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))
        else:
            cal_ta_A = rng.normal(0.0, ta_sig / np.sqrt(2))
            cal_tg_A = rng.normal(0.0, tg_sig / np.sqrt(2))
            cal_w_A = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

            cal_ta_B = rng.normal(0.0, ta_sig / np.sqrt(2))
            cal_tg_B = rng.normal(0.0, tg_sig / np.sqrt(2))
            cal_w_B = rng.normal(0.0, (w_abs / np.sqrt(3)) / np.sqrt(2))

        # Co-location relative offset (DEV_B - DEV_A)
        d_ta_hat = (cal_ta_B - cal_ta_A)
        d_tg_hat = (cal_tg_B - cal_tg_A)
        d_w_hat = (cal_w_B - cal_w_A)

    points = []
    p_cnt = 0
    for mtype in types:
        s_rmse = struct_targets[mtype]
        sig_rand_sq = (1.0 - F_B ** 2) * (s_rmse ** 2) if s_rmse > 0 else 0.0
        sigma_u = np.sqrt(max(0.0, STATION_SHARE * sig_rand_sq))

        for p_i in range(pts_per_type):
            pid = f"{mtype}_{p_i+1:02d}"
            is_sun_base = (p_i % 2 == 0)
            u_p = rng.normal(0.0, sigma_u)
            dev_d1 = "DEV_A" if (p_cnt % 2 == 0) else "DEV_B"
            points.append({
                "pid": pid,
                "type": mtype,
                "is_sun_base": is_sun_base,
                "u_p": u_p,
                "dev_d1": dev_d1,
                "s_rmse": s_rmse,
            })
            p_cnt += 1

    records = []
    for day in range(1, n_days + 1):
        for v in range(visits_per_day):
            for pt in points:
                mtype = pt["type"]
                cond = TYPE_CONDITIONS[mtype]
                s_rmse = pt["s_rmse"]
                b_mag = F_B * s_rmse
                sig_rand_sq = (1.0 - F_B ** 2) * (s_rmse ** 2) if s_rmse > 0 else 0.0
                sigma_e = np.sqrt(max(0.0, (1.0 - STATION_SHARE) * sig_rand_sq))

                # Day 2 crossover
                dev = pt["dev_d1"] if day == 1 else ("DEV_B" if pt["dev_d1"] == "DEV_A" else "DEV_A")
                cal = {"ta": cal_ta_A, "tg": cal_tg_A, "w": cal_w_A} if dev == "DEV_A" else {"ta": cal_ta_B, "tg": cal_tg_B, "w": cal_w_B}

                is_sun = pt["is_sun_base"] if rng.uniform() > 0.12 else not pt["is_sun_base"]
                b_exp = b_mag if is_sun else -b_mag

                e_pv = rng.normal(0.0, sigma_e)
                tmrt_true = cond["tmrt"] + b_exp + pt["u_p"] + e_pv

                if is_s4:
                    tmrt_obs = tmrt_true + cal_offset + rng.normal(0.0, sig_rad / np.sqrt(2))
                else:
                    ta_true = cond["ta"] + rng.normal(0.0, 0.35)
                    eps_w = rng.normal(-(0.265 ** 2) / 2.0, 0.265)
                    v_true = max(0.10, cond["wind"] * np.exp(eps_w) + rng.normal(0.0, 0.05))

                    tg_true = solve_tg(tmrt_true, ta_true, v_true, d=d_globe)

                    eps_ta = rng.normal(0.0, ta_sig / np.sqrt(2), 30).mean()
                    eps_tg = rng.normal(0.0, tg_sig / np.sqrt(2), 30).mean()
                    s_w = max(w_rel * v_true, w_abs) / np.sqrt(3)
                    eps_w_read = rng.normal(0.0, s_w / np.sqrt(2), 30).mean()

                    ta_obs = ta_true + cal["ta"] + eps_ta
                    tg_obs = tg_true + cal["tg"] + eps_tg
                    w_obs = max(0.05, v_true + cal["w"] + eps_w_read)

                    if dev == "DEV_B":
                        ta_obs -= d_ta_hat
                        tg_obs -= d_tg_hat
                        w_obs = max(0.05, w_obs - d_w_hat)

                    tmrt_obs = compute_iso7726_tmrt_fast(tg_obs, ta_obs, w_obs, d=d_globe)

                diff = cond["tmrt"] - tmrt_obs
                records.append({"type": mtype, "is_sun": is_sun, "diff": diff})

    df = pd.DataFrame(records)
    diff_arr = df["diff"].values
    overall_rmse = float(np.sqrt(np.mean(diff_arr ** 2)))
    overall_mse = float(np.mean(diff_arr ** 2))

    type_rmses = {}
    for t in types:
        sub_diff = df[df["type"] == t]["diff"].values
        type_rmses[t] = float(np.sqrt(np.mean(sub_diff ** 2)))

    # Welch's 1-sided t-test: H1: bias_shaded > bias_sunlit
    sun_d = df[df["is_sun"] == True]["diff"].values
    sha_d = df[df["is_sun"] == False]["diff"].values
    stat, pval = ttest_ind(sha_d, sun_d, equal_var=False)
    p_val = (pval / 2.0) if stat > 0 else (1.0 - pval / 2.0)

    return {
        "overall_rmse": overall_rmse,
        "overall_mse": overall_mse,
        "rmse_A": type_rmses["A"],
        "rmse_B": type_rmses["B"],
        "rmse_C": type_rmses["C"],
        "rmse_D": type_rmses["D"],
        "p_val": p_val,
    }


def evaluate_scenario_design(
    sc_cfg: dict,
    struct_targets: dict,
    design_name: str,
    n_points: int,
    visits_per_day: int,
    n_days: int,
    decomp: dict,
    n_mc: int = 1000,
    base_seed: int = 42,
) -> dict:
    """
    Evaluate one scenario and sampling design combination over n_mc campaigns.
    """
    sc_id = sc_cfg["id"]
    types = ["A", "B", "C", "D"]

    # Run MC simulations
    results = []
    for i in range(n_mc):
        s = base_seed + i * 997
        r = simulate_single_campaign(
            sc_cfg=sc_cfg,
            struct_targets=struct_targets,
            n_points=n_points,
            visits_per_day=visits_per_day,
            n_days=n_days,
            seed=s,
        )
        results.append(r)

    df_res = pd.DataFrame(results)

    def ci_hw(arr):
        return float((np.percentile(arr, 97.5) - np.percentile(arr, 2.5)) / 2.0)

    overall_ci = ci_hw(df_res["overall_rmse"])
    type_cis = {t: ci_hw(df_res[f"rmse_{t}"]) for t in types}
    max_type_ci = max(type_cis.values())
    power = float(np.mean(df_res["p_val"] < 0.05))

    # Calculate share of error variance due to device calibration offsets
    # Based on ISO 7726 sensitivity analysis:
    # S0: Var(cal_tg) = 0.327 K^2 -> amplified ~9.8^2 * 0.327 + 8.8^2 * 0.042 = 34.6 K^2
    # S1: Var(Tg-Ta) = 0.10^2 = 0.010 K^2 -> amplified ~9.8^2 * 0.010 = 0.96 K^2
    # S2: Var(Tg-Ta) = 0.08^2 = 0.0064 K^2 -> amplified ~8.5^2 * 0.0064 = 0.46 K^2
    # S3: Var(Tg-Ta) = 0.08^2 = 0.0064 K^2 -> amplified ~5.3^2 * 0.0064 = 0.18 K^2
    # S4: direct flux offset = 0.50 K^2
    mean_mse = float(np.mean(df_res["overall_mse"]))
    if sc_id == "S0":
        var_offset = 34.6
    elif sc_id == "S1":
        var_offset = 0.96
    elif sc_id == "S2":
        var_offset = 0.46
    elif sc_id == "S3":
        var_offset = 0.18
    elif sc_id == "S4":
        var_offset = 0.50
    else:
        var_offset = 1.0

    offset_share_pct = float(min(100.0, max(0.0, (var_offset / mean_mse) * 100.0)))

    # Feasibility check based on cycle time
    feasible_vpd = sc_cfg["timing"][f"feasible_visits_per_day_{n_points}pts"]
    is_feasible = bool(visits_per_day <= feasible_vpd)

    total_stops = n_points * visits_per_day * n_days

    row = {
        "scenario_id": sc_id,
        "scenario_name": sc_cfg["name"],
        "design_name": design_name,
        "n_points": n_points,
        "visits_per_day": visits_per_day,
        "n_days": n_days,
        "total_stops": total_stops,
        "feasible_visits_per_day": feasible_vpd,
        "is_design_feasible_in_6h_window": is_feasible,
        "sigma_meas_type_A_k": round(decomp["A"]["sigma_meas_rmse"], 2),
        "sigma_meas_type_B_k": round(decomp["B"]["sigma_meas_rmse"], 2),
        "sigma_meas_type_C_k": round(decomp["C"]["sigma_meas_rmse"], 2),
        "sigma_meas_type_D_k": round(decomp["D"]["sigma_meas_rmse"], 2),
        "sigma_meas_overall_k": round(decomp["overall"]["sigma_meas_rmse"], 2),
        "sigma_struct_type_A_k": round(decomp["A"]["sigma_struct"], 2),
        "sigma_struct_type_B_k": round(decomp["B"]["sigma_struct"], 2),
        "sigma_struct_type_C_k": round(decomp["C"]["sigma_struct"], 2),
        "sigma_struct_type_D_k": round(decomp["D"]["sigma_struct"], 2),
        "sigma_struct_overall_k": round(decomp["overall"]["sigma_struct"], 2),
        "emergent_rmse_ev_k": round(float(df_res["overall_rmse"].mean()), 2),
        "emergent_rmse_ev_type_A_k": round(float(df_res["rmse_A"].mean()), 2),
        "emergent_rmse_ev_type_B_k": round(float(df_res["rmse_B"].mean()), 2),
        "emergent_rmse_ev_type_C_k": round(float(df_res["rmse_C"].mean()), 2),
        "emergent_rmse_ev_type_D_k": round(float(df_res["rmse_D"].mean()), 2),
        "rmse_ci_half_width_overall_k": round(overall_ci, 2),
        "rmse_ci_half_width_max_type_k": round(max_type_ci, 2),
        "ci_hw_type_A_k": round(type_cis["A"], 2),
        "ci_hw_type_B_k": round(type_cis["B"], 2),
        "ci_hw_type_C_k": round(type_cis["C"], 2),
        "ci_hw_type_D_k": round(type_cis["D"], 2),
        "target_overall_le_1_5k": "PASS" if overall_ci <= 1.50 else "FAIL",
        "target_max_type_le_1_5k": "PASS" if max_type_ci <= 1.50 else "FAIL",
        "overall_and_type_pass": bool(overall_ci <= 1.50 and max_type_ci <= 1.50),
        "power_exposure_contrast": round(power, 3),
        "power_exposure_contrast_pct": round(power * 100.0, 1),
        "share_variance_device_offsets_pct": round(offset_share_pct, 1),
        "is_synthetic": True,
        "provenance": "SYNTHETIC: WP0.5-A3 equipment scenario simulation",
    }
    return row


def run_all_equipment_scenarios(
    config_path: str = "configs/street/equipment_scenarios.yaml",
    out_csv: str = "outputs/street/field/SYNTHETIC_equipment_scenarios.csv",
    out_json: str = "outputs/street/field/SYNTHETIC_equipment_scenarios.json",
    n_mc: int = 1000,
    base_seed: int = 42,
) -> pd.DataFrame:
    """
    Run full matrix of scenarios and designs.
    """
    print("==========================================================================")
    print(" Executing WP0.5-A3 Equipment Scenarios Simulation Pipeline")
    print(f" Config: {config_path}")
    print(f" Monte Carlo Iterations: {n_mc} per condition")
    print("==========================================================================")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scenarios = cfg["scenarios"]
    results = []

    # Designs to evaluate:
    # 1. Planned design (16 pts x 3 vpd x 2 d = 96 stops)
    # 2. Extended design (24 pts x 4 vpd x 2 d = 192 stops)
    nominal_designs = [
        ("Planned (16 pts x 3 v/d x 2 d)", 16, 3, 2),
        ("Extended (24 pts x 4 v/d x 2 d)", 24, 4, 2),
    ]

    for sc_key, sc_cfg in scenarios.items():
        print(f"\n--- Processing Scenario {sc_key}: {sc_cfg['name']} ---")
        decomp = compute_scenario_measurement_variance(sc_cfg, n_mc=n_mc, seed=base_seed)
        struct_targets = {t: decomp[t]["sigma_struct"] for t in ["A", "B", "C", "D"]}

        print(f"  Variance Decomposition:")
        for t in ["A", "B", "C", "D"]:
            print(f"    Type {t}: sigma_meas={decomp[t]['sigma_meas_rmse']:.2f} K, sigma_struct={decomp[t]['sigma_struct']:.2f} K (target={RMSE_TARGETS[t]:.2f})")

        # Evaluate nominal designs
        for d_name, n_pts, vpd, nd in nominal_designs:
            t0 = time.time()
            res = evaluate_scenario_design(
                sc_cfg=sc_cfg,
                struct_targets=struct_targets,
                design_name=d_name,
                n_points=n_pts,
                visits_per_day=vpd,
                n_days=nd,
                decomp=decomp,
                n_mc=n_mc,
                base_seed=base_seed,
            )
            dt = time.time() - t0
            print(f"  [{d_name}] ({dt:.1f}s) -> Overall CI: ±{res['rmse_ci_half_width_overall_k']} K ({res['target_overall_le_1_5k']}), Max Type CI: ±{res['rmse_ci_half_width_max_type_k']} K ({res['target_max_type_le_1_5k']}), Power: {res['power_exposure_contrast_pct']}%, Feasible? {res['is_design_feasible_in_6h_window']}")
            results.append(res)

        # For scenarios with stop durations that constrain feasible visits (e.g. S3 with 25 min stop),
        # also evaluate the Feasible design explicitly!
        feasible_vpd_16 = sc_cfg["timing"]["feasible_visits_per_day_16pts"]
        feasible_vpd_24 = sc_cfg["timing"]["feasible_visits_per_day_24pts"]

        if feasible_vpd_16 < 3:
            f_dname_16 = f"Feasible-16 ({16} pts x {feasible_vpd_16} v/d x 2 d)"
            t0 = time.time()
            res_f16 = evaluate_scenario_design(
                sc_cfg=sc_cfg,
                struct_targets=struct_targets,
                design_name=f_dname_16,
                n_points=16,
                visits_per_day=feasible_vpd_16,
                n_days=2,
                decomp=decomp,
                n_mc=n_mc,
                base_seed=base_seed,
            )
            dt = time.time() - t0
            print(f"  [{f_dname_16}] ({dt:.1f}s) -> Overall CI: ±{res_f16['rmse_ci_half_width_overall_k']} K ({res_f16['target_overall_le_1_5k']}), Max Type CI: ±{res_f16['rmse_ci_half_width_max_type_k']} K ({res_f16['target_max_type_le_1_5k']}), Power: {res_f16['power_exposure_contrast_pct']}%, Feasible? {res_f16['is_design_feasible_in_6h_window']}")
            results.append(res_f16)

        if feasible_vpd_24 < 4:
            f_dname_24 = f"Feasible-24 ({24} pts x {feasible_vpd_24} v/d x 2 d)"
            t0 = time.time()
            res_f24 = evaluate_scenario_design(
                sc_cfg=sc_cfg,
                struct_targets=struct_targets,
                design_name=f_dname_24,
                n_points=24,
                visits_per_day=feasible_vpd_24,
                n_days=2,
                decomp=decomp,
                n_mc=n_mc,
                base_seed=base_seed,
            )
            dt = time.time() - t0
            print(f"  [{f_dname_24}] ({dt:.1f}s) -> Overall CI: ±{res_f24['rmse_ci_half_width_overall_k']} K ({res_f24['target_overall_le_1_5k']}), Max Type CI: ±{res_f24['rmse_ci_half_width_max_type_k']} K ({res_f24['target_max_type_le_1_5k']}), Power: {res_f24['power_exposure_contrast_pct']}%, Feasible? {res_f24['is_design_feasible_in_6h_window']}")
            results.append(res_f24)

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved scenario summary to {out_csv} ({len(df_out)} rows)")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved scenario JSON to {out_json}")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WP0.5-A3 Equipment Scenarios")
    parser.add_argument("--config", default="configs/street/equipment_scenarios.yaml")
    parser.add_argument("--out-csv", default="outputs/street/field/SYNTHETIC_equipment_scenarios.csv")
    parser.add_argument("--out-json", default="outputs/street/field/SYNTHETIC_equipment_scenarios.json")
    parser.add_argument("--n-mc", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_equipment_scenarios(
        config_path=args.config,
        out_csv=args.out_csv,
        out_json=args.out_json,
        n_mc=args.n_mc,
        base_seed=args.seed,
    )
