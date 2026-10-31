"""
CLIMORFA Street — Campaign Data Processing & Validation Pipeline (v1.1).
Authoritative specification: docs/climorfa_street/field_error_model.md §1, §3.3, §3.5.

Processes Kestrel raw mobile loop observations (works identically for real and synthetic data):
1. Detects co-location intercomparison segments (point_id starting with 'COLOC_').
2. Estimates relative device offsets (Ta, Tg, wind, RH) with 95% confidence intervals.
3. Removes relative device offsets before ISO 7726 Tmrt calculation.
4. Discards first 5 minutes of each mobile stop (stabilization), averages the last 5 minutes.
5. Derives Tmrt using the ISO 7726 forced convection equation (1-inch black globe).
6. Extrapolates 1.1 m wind to 10 m wind using the logarithmic wind profile.
7. Computes UTCI using pythermalcomfort.
8. Matches observations with SOLWEIG model outputs at the stop point and time.
9. Computes comparison metrics (Bias, MAE, RMSE, Pearson R, Willmott d) overall,
   by morphology type, and by exposure, with point-block bootstrap 95% confidence intervals.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import pythermalcomfort.models as ptm
from scipy.stats import pearsonr


def compute_iso7726_tmrt(tg_c: float, ta_c: float, v_ms: float, d: float = 0.0254, eps: float = 0.95) -> float:
    """
    Compute mean radiant temperature Tmrt (°C) from globe temperature Tg using ISO 7726.
    Exact forward calculation matching the inversion solve_tg.
    """
    sigma = 5.670374419e-8
    v_eff = max(0.05, float(v_ms))
    h_forced = 6.3 * (v_eff ** 0.6) / (d ** 0.4)
    h_nat = 1.4 * (abs(tg_c - ta_c) / d) ** 0.25
    h_c = max(h_forced, h_nat)

    tg_k = tg_c + 273.15
    term = (tg_k ** 4) + (h_c / (eps * sigma)) * (tg_c - ta_c)
    if term <= 0:
        return ta_c
    tmrt_k = term ** 0.25
    return float(tmrt_k - 273.15)


def extrapolate_wind_10m(v_1_1m: float, z_sensor: float = 1.1, z_ref: float = 10.0, z0: float = 0.05) -> float:
    """Logarithmic wind profile extrapolation from sensor height to 10 m reference height, clamped to UTCI limits."""
    ratio = np.log(max(1.1, z_ref) / z0) / np.log(max(0.6, z_sensor) / z0)
    v_10 = float(v_1_1m * ratio)
    return float(np.clip(v_10, 0.5, 17.0))


def willmott_d(obs: np.ndarray, model: np.ndarray) -> float:
    """Willmott's Index of Agreement d (1981)."""
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    if len(obs) < 2:
        return np.nan
    obs_mean = np.mean(obs)
    num = np.sum((model - obs) ** 2)
    den = np.sum((np.abs(model - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    if den == 0.0:
        return 1.0
    return float(1.0 - (num / den))


def calc_metrics(obs: np.ndarray, model: np.ndarray) -> dict:
    """Compute summary comparison metrics."""
    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    diff = model - obs
    n = len(obs)
    if n == 0:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "r": np.nan, "d": np.nan}

    bias = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if n > 2 and np.std(obs) > 1e-4 and np.std(model) > 1e-4:
        r_val, _ = pearsonr(obs, model)
        r_val = float(r_val)
    else:
        r_val = np.nan

    d_val = willmott_d(obs, model)

    return {
        "n": int(n),
        "bias": round(bias, 3),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r": round(r_val, 3) if not np.isnan(r_val) else np.nan,
        "d": round(d_val, 3) if not np.isnan(d_val) else np.nan,
    }


def point_block_bootstrap(df: pd.DataFrame, var_obs: str, var_mod: str, n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Non-parametric point-block bootstrap: resamples unique points with replacement
    to preserve within-point visit correlations and spatial clustering.
    """
    rng = np.random.default_rng(seed)
    unique_points = df["point_id"].unique()
    n_pts = len(unique_points)
    if n_pts < 3:
        return {"bias_ci": [np.nan, np.nan], "rmse_ci": [np.nan, np.nan], "mae_ci": [np.nan, np.nan]}

    point_groups = {pid: grp for pid, grp in df.groupby("point_id")}

    biases, maes, rmses, rs, ds = [], [], [], [], []

    for _ in range(n_boot):
        sample_pts = rng.choice(unique_points, size=n_pts, replace=True)
        boot_df = pd.concat([point_groups[p] for p in sample_pts], ignore_index=True)
        m = calc_metrics(boot_df[var_obs].values, boot_df[var_mod].values)
        biases.append(m["bias"])
        maes.append(m["mae"])
        rmses.append(m["rmse"])
        if not np.isnan(m["r"]):
            rs.append(m["r"])
        if not np.isnan(m["d"]):
            ds.append(m["d"])

    def ci(arr):
        clean = [x for x in arr if not np.isnan(x)]
        if len(clean) == 0:
            return [np.nan, np.nan]
        return [round(float(np.percentile(clean, 2.5)), 3), round(float(np.percentile(clean, 97.5)), 3)]

    return {
        "bias_ci": ci(biases),
        "mae_ci": ci(maes),
        "rmse_ci": ci(rmses),
        "r_ci": ci(rs),
        "d_ci": ci(ds),
    }


def estimate_colocation_offsets(df_raw: pd.DataFrame, ref_device: str = "DEV_A") -> dict:
    """
    Estimate relative device offsets (Ta, Tg, wind, RH) from co-location intercomparison segments.
    Co-location rows are identified by point_id starting with 'COLOC_'.
    Returns estimated relative offsets (DEV_B - DEV_A) with standard errors and 95% confidence intervals.
    """
    coloc_mask = df_raw["point_id"].str.startswith("COLOC_")
    df_coloc = df_raw[coloc_mask].copy()

    if len(df_coloc) == 0:
        return {"has_colocation": False, "n_samples": 0}

    devs = df_coloc["device_id"].unique()
    if len(devs) < 2:
        return {"has_colocation": False, "n_samples": len(df_coloc), "note": "Only one device present in co-location"}

    df_a = df_coloc[df_coloc["device_id"] == ref_device].sort_values("timestamp_local").reset_index(drop=True)
    other_dev = [d for d in devs if d != ref_device][0]
    df_b = df_coloc[df_coloc["device_id"] == other_dev].sort_values("timestamp_local").reset_index(drop=True)

    # Merge on timestamp and condition/point_id
    paired = pd.merge(df_a, df_b, on=["timestamp_local", "point_id"], suffixes=("_ref", "_target"))

    if len(paired) == 0:
        # Fallback to index order within each point_id segment if timestamps had minor seconds drift
        paired_list = []
        for pid, grp in df_coloc.groupby("point_id"):
            sub_a = grp[grp["device_id"] == ref_device].sort_values("timestamp_local").reset_index(drop=True)
            sub_b = grp[grp["device_id"] == other_dev].sort_values("timestamp_local").reset_index(drop=True)
            min_l = min(len(sub_a), len(sub_b))
            if min_l > 0:
                m_sub = pd.DataFrame({
                    "timestamp_local": sub_a["timestamp_local"].iloc[:min_l],
                    "point_id": pid,
                    "ta_c_ref": sub_a["ta_c"].iloc[:min_l].values,
                    "ta_c_target": sub_b["ta_c"].iloc[:min_l].values,
                    "tg_c_ref": sub_a["tg_c"].iloc[:min_l].values,
                    "tg_c_target": sub_b["tg_c"].iloc[:min_l].values,
                    "wind_ms_ref": sub_a["wind_ms"].iloc[:min_l].values,
                    "wind_ms_target": sub_b["wind_ms"].iloc[:min_l].values,
                    "rh_pct_ref": sub_a["rh_pct"].iloc[:min_l].values,
                    "rh_pct_target": sub_b["rh_pct"].iloc[:min_l].values,
                })
                paired_list.append(m_sub)
        if len(paired_list) > 0:
            paired = pd.concat(paired_list, ignore_index=True)

    n_samples = len(paired)
    if n_samples == 0:
        return {"has_colocation": False, "n_samples": 0}

    # Differences: target (e.g. DEV_B) - ref (e.g. DEV_A)
    diff_ta = paired["ta_c_target"] - paired["ta_c_ref"]
    diff_tg = paired["tg_c_target"] - paired["tg_c_ref"]
    diff_wind = paired["wind_ms_target"] - paired["wind_ms_ref"]
    diff_rh = paired["rh_pct_target"] - paired["rh_pct_ref"]

    def stats(series):
        mean_val = float(series.mean())
        std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        se_val = float(std_val / np.sqrt(len(series))) if len(series) > 0 else 0.0
        ci_low = float(mean_val - 1.96 * se_val)
        ci_high = float(mean_val + 1.96 * se_val)
        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "se": round(se_val, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        }

    # Also compute breakdown by condition
    segment_stats = {}
    for pid, grp in paired.groupby("point_id"):
        segment_stats[pid] = {
            "n": len(grp),
            "ta_mean": round(float((grp["ta_c_target"] - grp["ta_c_ref"]).mean()), 4),
            "tg_mean": round(float((grp["tg_c_target"] - grp["tg_c_ref"]).mean()), 4),
            "wind_mean": round(float((grp["wind_ms_target"] - grp["wind_ms_ref"]).mean()), 4),
            "rh_mean": round(float((grp["rh_pct_target"] - grp["rh_pct_ref"]).mean()), 4),
        }

    return {
        "has_colocation": True,
        "reference_device": ref_device,
        "target_device": other_dev,
        "n_samples": int(n_samples),
        "ta": stats(diff_ta),
        "tg": stats(diff_tg),
        "wind": stats(diff_wind),
        "rh": stats(diff_rh),
        "segments": segment_stats,
    }


def apply_colocation_correction(df_raw: pd.DataFrame, offsets: dict, mode: str = "baseline_A") -> pd.DataFrame:
    """
    Remove relative device offsets from raw observation data before ISO 7726 processing.
    mode='baseline_A': reference device (DEV_A) is baseline, DEV_B is adjusted by -delta.
    mode='pair_mean': DEV_A adjusted by +delta/2, DEV_B adjusted by -delta/2.
    """
    if not offsets.get("has_colocation", False):
        return df_raw.copy()

    df_corr = df_raw.copy()
    ref_dev = offsets["reference_device"]
    tgt_dev = offsets["target_device"]

    delta_ta = offsets["ta"]["mean"]
    delta_tg = offsets["tg"]["mean"]
    delta_wind = offsets["wind"]["mean"]
    delta_rh = offsets["rh"]["mean"]

    mask_tgt = df_corr["device_id"] == tgt_dev
    mask_ref = df_corr["device_id"] == ref_dev

    if mode == "baseline_A":
        # Target device (DEV_B) is aligned to Reference device (DEV_A)
        df_corr.loc[mask_tgt, "ta_c"] -= delta_ta
        df_corr.loc[mask_tgt, "tg_c"] -= delta_tg
        df_corr.loc[mask_tgt, "wind_ms"] = np.maximum(0.0, df_corr.loc[mask_tgt, "wind_ms"] - delta_wind)
        df_corr.loc[mask_tgt, "rh_pct"] = np.clip(df_corr.loc[mask_tgt, "rh_pct"] - delta_rh, 1.0, 99.0)
    else:  # pair_mean
        df_corr.loc[mask_ref, "ta_c"] += delta_ta / 2.0
        df_corr.loc[mask_ref, "tg_c"] += delta_tg / 2.0
        df_corr.loc[mask_ref, "wind_ms"] = np.maximum(0.0, df_corr.loc[mask_ref, "wind_ms"] + delta_wind / 2.0)
        df_corr.loc[mask_ref, "rh_pct"] = np.clip(df_corr.loc[mask_ref, "rh_pct"] + delta_rh / 2.0, 1.0, 99.0)

        df_corr.loc[mask_tgt, "ta_c"] -= delta_ta / 2.0
        df_corr.loc[mask_tgt, "tg_c"] -= delta_tg / 2.0
        df_corr.loc[mask_tgt, "wind_ms"] = np.maximum(0.0, df_corr.loc[mask_tgt, "wind_ms"] - delta_wind / 2.0)
        df_corr.loc[mask_tgt, "rh_pct"] = np.clip(df_corr.loc[mask_tgt, "rh_pct"] - delta_rh / 2.0, 1.0, 99.0)

    return df_corr


def process_campaign(
    raw_csv: str,
    points_gpkg: str,
    provisional_base_dir: str,
    out_stops_csv: str,
    out_metrics_csv: str,
    out_metrics_json: str,
    out_offsets_json: str = r"outputs/street/field/SYNTHETIC_device_offsets.json",
    config_path: str = "configs/street/field_error_model.yaml",
    ref_device: str = "DEV_A",
    correction_mode: str = "baseline_A",
):
    print(f"Loading raw observations: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    df_raw["timestamp_dt"] = pd.to_datetime(df_raw["timestamp_local"])
    df_raw = df_raw.sort_values(by=["device_id", "point_id", "timestamp_dt"]).reset_index(drop=True)

    # 1. Co-location offset estimation and removal
    offsets = estimate_colocation_offsets(df_raw, ref_device=ref_device)
    if offsets.get("has_colocation", False):
        print("\n=======================================================")
        print(f" Co-location Intercomparison Relative Offset Estimates (DEV_B - DEV_A, n={offsets['n_samples']}):")
        print("=======================================================")
        for var in ["ta", "tg", "wind", "rh"]:
            stat = offsets[var]
            unit = "°C" if "t" in var else ("m/s" if var == "wind" else "%")
            print(f"  Delta {var.upper():4s}: {stat['mean']:+.4f} {unit} | SE: {stat['se']:.4f} | 95% CI: [{stat['ci_95'][0]:+.4f}, {stat['ci_95'][1]:+.4f}] {unit}")
        print("=======================================================\n")

        # Save offset estimates JSON
        os.makedirs(os.path.dirname(out_offsets_json), exist_ok=True)
        with open(out_offsets_json, "w", encoding="utf-8") as f:
            json.dump(offsets, f, indent=2)
        print(f"Saved device offset estimates JSON: {out_offsets_json}")

        # Apply correction
        df_proc = apply_colocation_correction(df_raw, offsets, mode=correction_mode)
    else:
        print("Notice: No co-location segments detected. Proceeding without device offset correction.")
        df_proc = df_raw.copy()

    # Discard co-location rows from campaign mobile stops analysis
    df_mobile = df_proc[~df_proc["point_id"].str.startswith("COLOC_")].copy()
    print(f"Extracted {len(df_mobile)} mobile campaign readings ({len(df_raw) - len(df_mobile)} co-location readings excluded from stops).")

    # Load points metadata
    gdf_points = gpd.read_file(points_gpkg)
    pt_meta = {}
    for _, pt in gdf_points.iterrows():
        pt_meta[pt["point_id"]] = {
            "x": pt["x"],
            "y": pt["y"],
            "morphology_type": pt["morphology_type"],
            "morphology_desc": pt["morphology_desc"],
            "street_name": pt["street_name"],
            "svf": pt["svf"],
        }

    # Group raw readings into visits / stops
    stops = []
    grouped = df_mobile.groupby(["device_id", "point_id"])

    for (dev, pid), group in grouped:
        group = group.sort_values("timestamp_dt").reset_index(drop=True)
        time_diffs = group["timestamp_dt"].diff().dt.total_seconds().fillna(0)
        visit_ids = (time_diffs > 120).cumsum()

        for vid, vgroup in group.groupby(visit_ids):
            vgroup = vgroup.sort_values("timestamp_dt").reset_index(drop=True)
            n_readings = len(vgroup)
            if n_readings < 10:
                continue

            # Drop first 5 minutes (first half of readings), average second 5 minutes
            split_idx = n_readings // 2
            avg_window = vgroup.iloc[split_idx:]

            t_mid = avg_window["timestamp_dt"].iloc[len(avg_window) // 2]
            ta_m = float(avg_window["ta_c"].mean())
            rh_m = float(avg_window["rh_pct"].mean())
            wind_m = float(avg_window["wind_ms"].mean())
            tg_m = float(avg_window["tg_c"].mean())

            # ISO 7726 Tmrt
            tmrt_obs = compute_iso7726_tmrt(tg_c=tg_m, ta_c=ta_m, v_ms=wind_m)

            # Extrapolate wind to 10 m
            v_10_obs = extrapolate_wind_10m(wind_m)

            # UTCI calculation
            try:
                res_utci = ptm.utci(tdb=ta_m, tr=tmrt_obs, v=v_10_obs, rh=rh_m)
                utci_obs = float(res_utci.utci)
            except Exception:
                utci_obs = np.nan

            is_synth = bool(vgroup["is_synthetic"].iloc[0]) if "is_synthetic" in vgroup.columns else False
            prov_geom = bool(vgroup["provisional_geometry"].iloc[0]) if "provisional_geometry" in vgroup.columns else False

            stops.append({
                "device_id": dev,
                "point_id": pid,
                "timestamp_mid": t_mid.strftime("%Y-%m-%d %H:%M:%S"),
                "date": t_mid.strftime("%Y-%m-%d"),
                "hour_frac": t_mid.hour + t_mid.minute / 60.0 + t_mid.second / 3600.0,
                "ta_obs": round(ta_m, 2),
                "rh_obs": round(rh_m, 1),
                "wind_obs": round(wind_m, 2),
                "wind_10m_obs": round(v_10_obs, 2),
                "tg_obs": round(tg_m, 2),
                "tmrt_obs": round(tmrt_obs, 2),
                "utci_obs": round(utci_obs, 2),
                "is_synthetic": is_synth,
                "provisional_geometry": prov_geom,
                **pt_meta.get(pid, {}),
            })

    df_stops = pd.DataFrame(stops)
    print(f"Processed {len(df_stops)} campaign stops across {df_stops['date'].nunique()} days.")

    # Match stops with SOLWEIG model raster values
    unique_dates = df_stops["date"].unique()
    matched_records = []

    for day_str in unique_dates:
        day_stops = df_stops[df_stops["date"] == day_str].copy()
        day_run_dir = os.path.join(provisional_base_dir, f"run_{day_str}/output_folder/0_0")
        tmrt_raster = os.path.join(day_run_dir, "TMRT_0_0.tif")
        shadow_raster = os.path.join(day_run_dir, "Shadow_0_0.tif")
        ta_raster = os.path.join(day_run_dir, "Ta_0_0.tif")
        wind_raster = os.path.join(day_run_dir, "Wind_0_0.tif")
        utci_raster = os.path.join(day_run_dir, "UTCI_0_0.tif")

        with rasterio.open(tmrt_raster) as ds_tmrt, \
             rasterio.open(shadow_raster) as ds_shadow, \
             rasterio.open(ta_raster) as ds_ta, \
             rasterio.open(wind_raster) as ds_wind, \
             rasterio.open(utci_raster) as ds_utci:

            transform = ds_tmrt.transform
            tmrt_c = ds_tmrt.read()
            shadow_c = ds_shadow.read()
            ta_c = ds_ta.read()
            wind_c = ds_wind.read()
            utci_c = ds_utci.read()

        for idx, row in day_stops.iterrows():
            px, py = row["x"], row["y"]
            col = int(round((px - transform[2]) / transform[0]))
            r_idx = int(round((transform[5] - py) / (-transform[4])))

            h_frac = row["hour_frac"]
            h0 = int(np.floor(h_frac))
            h1 = min(23, h0 + 1)
            w = h_frac - h0

            mod_tmrt = float((1.0 - w) * tmrt_c[h0, r_idx, col] + w * tmrt_c[h1, r_idx, col])
            mod_ta = float((1.0 - w) * ta_c[h0, r_idx, col] + w * ta_c[h1, r_idx, col])
            mod_wind = float((1.0 - w) * wind_cube[h0, r_idx, col] + w * wind_c[h1, r_idx, col]) if "wind_cube" in locals() else float((1.0 - w) * wind_c[h0, r_idx, col] + w * wind_c[h1, r_idx, col])
            mod_utci = float((1.0 - w) * utci_c[h0, r_idx, col] + w * utci_c[h1, r_idx, col])
            mod_shadow = float(shadow_c[h0 if w < 0.5 else h1, r_idx, col])

            exposure = "sunlit" if mod_shadow > 0.5 else "shaded"

            rec = dict(row)
            rec["tmrt_model"] = round(mod_tmrt, 2)
            rec["ta_model"] = round(mod_ta, 2)
            rec["wind_model"] = round(mod_wind, 2)
            rec["utci_model"] = round(mod_utci, 2)
            rec["shadow_model"] = round(mod_shadow, 2)
            rec["exposure_actual"] = exposure
            matched_records.append(rec)

    df_matched = pd.DataFrame(matched_records)
    os.makedirs(os.path.dirname(out_stops_csv), exist_ok=True)
    df_matched.to_csv(out_stops_csv, index=False)
    print(f"Saved processed stops table: {out_stops_csv}")

    # Compute comparison metrics overall, by morphology type, and by exposure
    subsets = {"Overall": df_matched}
    for mtype in sorted(df_matched["morphology_type"].unique()):
        subsets[f"Type_{mtype}"] = df_matched[df_matched["morphology_type"] == mtype]
    for exp in ["sunlit", "shaded"]:
        subsets[f"Exposure_{exp}"] = df_matched[df_matched["exposure_actual"] == exp]

    for mtype in sorted(df_matched["morphology_type"].unique()):
        for exp in ["sunlit", "shaded"]:
            sub = df_matched[(df_matched["morphology_type"] == mtype) & (df_matched["exposure_actual"] == exp)]
            if len(sub) > 0:
                subsets[f"Type_{mtype}_{exp}"] = sub

    metrics_rows = []
    metrics_dict = {}

    for name, sub in subsets.items():
        if len(sub) == 0:
            continue

        m_tmrt = calc_metrics(sub["tmrt_obs"].values, sub["tmrt_model"].values)
        ci_tmrt = point_block_bootstrap(sub, "tmrt_obs", "tmrt_model")

        m_utci = calc_metrics(sub["utci_obs"].values, sub["utci_model"].values)
        ci_utci = point_block_bootstrap(sub, "utci_obs", "utci_model")

        m_ta = calc_metrics(sub["ta_obs"].values, sub["ta_model"].values)
        m_wind = calc_metrics(sub["wind_obs"].values, sub["wind_model"].values)

        entry = {
            "subset": name,
            "n_stops": len(sub),
            "tmrt_bias": m_tmrt["bias"],
            "tmrt_bias_ci_95": ci_tmrt["bias_ci"],
            "tmrt_rmse": m_tmrt["rmse"],
            "tmrt_rmse_ci_95": ci_tmrt["rmse_ci"],
            "tmrt_mae": m_tmrt["mae"],
            "tmrt_r": m_tmrt["r"],
            "tmrt_d": m_tmrt["d"],
            "utci_bias": m_utci["bias"],
            "utci_bias_ci_95": ci_utci["bias_ci"],
            "utci_rmse": m_utci["rmse"],
            "utci_rmse_ci_95": ci_utci["rmse_ci"],
            "utci_mae": m_utci["mae"],
            "utci_r": m_utci["r"],
            "utci_d": m_utci["d"],
            "ta_rmse": m_ta["rmse"],
            "ta_bias": m_ta["bias"],
            "wind_rmse": m_wind["rmse"],
            "wind_bias": m_wind["bias"],
        }
        metrics_rows.append(entry)
        metrics_dict[name] = entry

    df_metrics = pd.DataFrame(metrics_rows)
    os.makedirs(os.path.dirname(out_metrics_csv), exist_ok=True)
    df_metrics.to_csv(out_metrics_csv, index=False)
    with open(out_metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"Saved model-vs-obs metrics CSV:  {out_metrics_csv}")
    print(f"Saved model-vs-obs metrics JSON: {out_metrics_json}")
    print("\n=======================================================")
    print(" Emergent Model vs Obs Metrics Table (with Co-location Offset Correction):")
    print("=======================================================")
    disp_cols = ["subset", "n_stops", "tmrt_bias", "tmrt_rmse", "utci_rmse", "ta_rmse", "wind_rmse", "tmrt_r", "tmrt_d"]
    print(df_metrics[disp_cols].to_string())

    return df_stops, df_metrics, offsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", default=r"data/street/field/SYNTHETIC_kestrel_raw_campaign.csv")
    parser.add_argument("--points", default=r"data/street/field/PROVISIONAL_points.gpkg")
    parser.add_argument("--provisional-dir", default=r"data/street/field/provisional")
    parser.add_argument("--out-stops", default=r"data/street/field/SYNTHETIC_processed_stops.csv")
    parser.add_argument("--out-metrics-csv", default=r"outputs/street/field/SYNTHETIC_model_vs_obs_metrics.csv")
    parser.add_argument("--out-metrics-json", default=r"outputs/street/field/SYNTHETIC_model_vs_obs_metrics.json")
    parser.add_argument("--out-offsets-json", default=r"outputs/street/field/SYNTHETIC_device_offsets.json")
    parser.add_argument("--ref-device", default="DEV_A")
    parser.add_argument("--correction-mode", default="baseline_A")
    args = parser.parse_args()

    process_campaign(
        raw_csv=args.raw_csv,
        points_gpkg=args.points,
        provisional_base_dir=args.provisional_dir,
        out_stops_csv=args.out_stops,
        out_metrics_csv=args.out_metrics_csv,
        out_metrics_json=args.out_metrics_json,
        out_offsets_json=args.out_offsets_json,
        ref_device=args.ref_device,
        correction_mode=args.correction_mode,
    )
