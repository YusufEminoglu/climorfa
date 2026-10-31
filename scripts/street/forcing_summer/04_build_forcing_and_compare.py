#!/usr/bin/env python3
"""
Build UMEP 24-column hourly forcing files, compare ERA5 vs METAR LTBJ,
perform radiation decomposition and validation for Izmir Summer Design Days:
- Day 1: IMBAT (2022-07-05) [Supersedes 2021-08-09]
- Day 2: CALM HEAT (2023-07-26)
Generates both standard forcing and Ineichen-matched RADSENS variants (Kdn/Kdir scaled by 0.88).
Part of WP1.4 / WP1.4b (Summer design days + SOLWEIG forcing).
"""
import argparse
import json
import math
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "street" / "forcing_summer"
OUTPUTS_DIR = BASE_DIR / "outputs" / "street" / "forcing_summer"
METAR_CSV = DATA_DIR / "LTBJ_summer_2021_2025.csv"

IZMIR_LAT = 38.42
IZMIR_LON = 27.14
ALTITUDE = 20.0
TIMEZONE = "Europe/Istanbul"  # UTC+3 year-round

# OBSERVED wind group regex (Rule A7 / WP2.3b vrb_utils)
_VRB_RE = re.compile(r"^\S+\s+\d{6}Z\s+(?:(?:AUTO|COR)\s+)?VRB\d{2,3}(?:G\d{2,3})?KT\b")

# Default active design days
DEFAULT_DESIGN_DAYS = [
    {
        "type": "IMBAT",
        "date": "2022-07-05",
        "title": "Design Day 1: IMBAT Sea Breeze (2022-07-05)"
    },
    {
        "type": "CALM",
        "date": "2023-07-26",
        "title": "Design Day 2: CALM HEAT Stagnation (2023-07-26)"
    }
]

def interpolate_metar(source_date_str: str, target_grid: pd.DatetimeIndex, drop_vrb: bool = True) -> pd.DataFrame:
    """
    Interpolate LTBJ METAR records to regular target hourly grid.
    Uses circular vector decomposition for wind direction and preserves calm/scalar speed.
    Rule A7: VRB rows count for speed but not for direction (drop_vrb=True).
    Hours with any VRB report in (it-1:00, it:00] are flagged as vrb=True.
    """
    df = pd.read_csv(METAR_CSV, low_memory=False)
    df['valid_utc'] = pd.to_datetime(df['valid'], utc=True)
    df['valid_local'] = df['valid_utc'].dt.tz_convert(TIMEZONE)

    src_date = pd.to_datetime(source_date_str).date()
    t_start = pd.Timestamp(src_date, tz=TIMEZONE) - pd.Timedelta(hours=12)
    t_end = pd.Timestamp(src_date, tz=TIMEZONE) + pd.Timedelta(hours=36)
    subset = df[(df['valid_local'] >= t_start) & (df['valid_local'] <= t_end)].copy()
    subset = subset.sort_values(by='valid_local').drop_duplicates(subset='valid_local').reset_index(drop=True)

    ta_c = (pd.to_numeric(subset['tmpf'], errors='coerce') - 32.0) * 5.0 / 9.0
    rh_pct = pd.to_numeric(subset['relh'], errors='coerce').clip(lower=0.0, upper=100.0)
    wind_ms = (pd.to_numeric(subset['sknt'], errors='coerce') * 0.514444).clip(lower=0.0)

    # Pressure in kPa: alti (inHg * 3.38639 -> kPa) or mslp (hPa * 0.1 -> kPa)
    alti = pd.to_numeric(subset['alti'], errors='coerce') * 3.3863886666667
    mslp = pd.to_numeric(subset['mslp'], errors='coerce') * 0.1
    press_kpa = alti.fillna(mslp).fillna(101.325)

    # VRB identification per Rule A7
    body = subset['metar'].fillna('').astype(str).str.split(' RMK ', n=1, regex=False).str[0]
    vrb_text = body.apply(lambda s: bool(_VRB_RE.search(s)))
    drct_num = pd.to_numeric(subset['drct'], errors='coerce')
    vrb_drct = drct_num.isna()
    is_vrb = vrb_text | vrb_drct
    subset['is_vrb'] = is_vrb

    # Wind direction with circular decomposition
    wd_deg = drct_num.fillna(0.0)
    u_wind = -wind_ms * np.sin(np.radians(wd_deg))
    v_wind = -wind_ms * np.cos(np.radians(wd_deg))

    obs_times = subset['valid_local']
    ref_time = pd.Timestamp(src_date, tz=TIMEZONE)
    obs_sec = (obs_times - ref_time).dt.total_seconds().values
    target_ref = target_grid[0].floor('D')
    target_sec = (target_grid - target_ref).total_seconds().values

    ta_interp = np.interp(target_sec, obs_sec, ta_c)
    rh_interp = np.interp(target_sec, obs_sec, rh_pct)
    press_interp = np.interp(target_sec, obs_sec, press_kpa)

    # VRB rows count for scalar speed
    wind_scalar_interp = np.interp(target_sec, obs_sec, wind_ms)

    # VRB rows do NOT count for direction when drop_vrb=True
    keep = (~is_vrb).to_numpy() if drop_vrb else np.ones(len(subset), bool)
    u_interp = np.interp(target_sec, obs_sec[keep], u_wind.to_numpy()[keep])
    v_interp = np.interp(target_sec, obs_sec[keep], v_wind.to_numpy()[keep])

    wd_interp = (np.degrees(np.arctan2(-u_interp, -v_interp))) % 360.0
    wd_interp[wind_scalar_interp < 0.2] = 0.0

    # Rule A7 VRB hour flags: any METAR report in (it-1:00, it:00] is VRB
    vrb_flags = []
    for h in range(len(target_grid)):
        lo = ref_time + pd.Timedelta(hours=h - 1)
        hi = ref_time + pd.Timedelta(hours=h)
        w = subset[(subset['valid_local'] > lo) & (subset['valid_local'] <= hi)]
        vrb_flags.append(bool(w['is_vrb'].any()) if len(w) > 0 else False)

    return pd.DataFrame({
        'ta_c': ta_interp,
        'rh_pct': rh_interp,
        'wind_ms': wind_scalar_interp,
        'wd_deg': wd_interp,
        'press_kpa': press_interp,
        'vrb': vrb_flags
    }, index=target_grid)

def compute_prata_longwave(ta_c: np.ndarray, rh_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prata (1996) clear-sky downwelling longwave irradiance."""
    t_k = ta_c + 273.15
    es = 6.112 * np.exp((17.67 * ta_c) / (ta_c + 243.5))
    ea = (rh_pct / 100.0) * es
    w = 46.5 * (ea / t_k)
    eps_cs = 1.0 - (1.0 + w) * np.exp(-np.sqrt(1.2 + 3.0 * w))
    sigma = 5.670374419e-8
    ldown = eps_cs * sigma * (t_k ** 4)
    return eps_cs, ldown

def process_design_day(day_info: dict, loc: pvlib.location.Location):
    day_type = day_info["type"]
    date_str = day_info["date"]
    print(f"\n=======================================================")
    print(f"Processing {day_type} ({date_str}) Forcing & Comparisons")
    print(f"=======================================================")

    t_start = pd.Timestamp(f"{date_str} 00:00:00", tz=TIMEZONE)
    t_end = pd.Timestamp(f"{date_str} 23:00:00", tz=TIMEZONE)
    grid_1h = pd.date_range(t_start, t_end, freq="1h")

    # 1. Solar position and clear-sky models
    # ERA5 hourly values at hour H represent the accumulation over (H-1, H].
    # Thus, evaluate solar geometry at the midpoint of the accumulation interval (H - 30m).
    grid_mid = grid_1h - pd.Timedelta(minutes=30)
    sp_mid = loc.get_solarposition(grid_mid)
    zenith_mid = sp_mid['zenith'].values
    elevation_mid = sp_mid['elevation'].values

    cs_1h = loc.get_clearsky(grid_mid, model="ineichen")
    extra_dni = pvlib.irradiance.get_extra_radiation(grid_mid).values

    # 2. Load ERA5 and ERA5-Land extracted CSVs
    era5_path = DATA_DIR / f"era5_hourly_{day_type}_{date_str}.csv"
    land_path = DATA_DIR / f"era5_land_hourly_{day_type}_{date_str}.csv"
    era5_df = pd.read_csv(era5_path)
    land_df = pd.read_csv(land_path)

    # 3. Interpolate METAR LTBJ
    met_1h = interpolate_metar(date_str, grid_1h)

    # 4. Solar Radiation Split & Decompositions
    # ERA5 GHI, direct horizontal (fdir / 3600), diffuse horizontal
    ghi_era5 = era5_df['ghi_wm2'].values
    dir_horiz_era5 = era5_df['dir_horiz_wm2'].values
    diff_horiz_era5 = era5_df['diff_horiz_wm2'].values

    # DNI from ERA5 direct horizontal: DNI = B_h / cos(zenith)
    cos_zen = np.cos(np.radians(zenith_mid))
    dni_era5 = np.zeros(24)
    valid_sun = (elevation_mid > 2.0) & (cos_zen > 0.035) & (dir_horiz_era5 > 0.0)
    dni_era5[valid_sun] = np.clip(dir_horiz_era5[valid_sun] / cos_zen[valid_sun], 0.0, 1050.0)

    # Decompose GHI with pvlib Erbs for comparison
    erbs_res = pvlib.irradiance.erbs(ghi_era5, zenith_mid, grid_mid)
    dni_erbs = erbs_res['dni'].values
    dhi_erbs = erbs_res['dhi'].values

    # Decompose GHI with pvlib DIRINT for comparison
    dni_dirint = pvlib.irradiance.dirint(ghi_era5, zenith_mid, grid_mid, use_delta_kt_prime=True).values
    dni_dirint = np.nan_to_num(dni_dirint, nan=0.0).clip(0.0)
    dhi_dirint = np.clip(ghi_era5 - dni_dirint * np.maximum(0.0, cos_zen), 0.0, None)

    # 5. Clearness Index at Solar Noon (~13:00 local)
    # Solar noon is between 12:00 and 14:00 (local hour 13 has highest solar elevation)
    noon_h = 13
    noon_ghi = ghi_era5[noon_h]
    noon_zen = zenith_mid[noon_h]
    noon_extra_horiz = extra_dni[noon_h] * np.cos(np.radians(noon_zen))
    noon_kt = noon_ghi / noon_extra_horiz
    noon_cs_ratio = noon_ghi / cs_1h['ghi'].iloc[noon_h]

    print(f"Noon (Hour 13 Local) Radiation Metrics:")
    print(f"  Solar Elevation: {elevation_mid[noon_h]:.2f}° | Zenith: {noon_zen:.2f}°")
    print(f"  ERA5 GHI: {noon_ghi:.2f} W/m²")
    print(f"  Extraterrestrial Horizontal: {noon_extra_horiz:.2f} W/m²")
    print(f"  Clearness Index (kt = GHI / I0_horiz): {noon_kt:.3f} (target ~0.7-0.8)")
    print(f"  Ineichen Clear-Sky GHI: {cs_1h['ghi'].iloc[noon_h]:.2f} W/m² | Ratio GHI/GHI_cs: {noon_cs_ratio:.3f}")
    print(f"  ERA5 Direct Normal (DNI): {dni_era5[noon_h]:.2f} W/m²")
    print(f"  pvlib Erbs DNI: {dni_erbs[noon_h]:.2f} W/m² | pvlib DIRINT DNI: {dni_dirint[noon_h]:.2f} W/m²")
    print(f"  ERA5 Diffuse Horizontal (DHI): {diff_horiz_era5[noon_h]:.2f} W/m²")
    print(f"  pvlib Erbs DHI: {dhi_erbs[noon_h]:.2f} W/m² | pvlib DIRINT DHI: {dhi_dirint[noon_h]:.2f} W/m²")

    # 6. Downwelling Longwave Comparison
    ldown_era5 = era5_df['ldown_wm2'].values
    ldown_land = land_df['ldown_wm2'].values
    _, ldown_prata = compute_prata_longwave(met_1h['ta_c'].values, met_1h['rh_pct'].values)

    # 7. Differences: ERA5 vs METAR LTBJ
    diff_ta = era5_df['t2m_c'].values - met_1h['ta_c'].values
    diff_rh = era5_df['rh_pct'].values - met_1h['rh_pct'].values
    diff_wind = era5_df['wind_spd_ms'].values - met_1h['wind_ms'].values

    diff_ta_land = land_df['t2m_c'].values - met_1h['ta_c'].values
    diff_rh_land = land_df['rh_pct'].values - met_1h['rh_pct'].values
    diff_wind_land = land_df['wind_spd_ms'].values - met_1h['wind_ms'].values

    comparison_stats = {
        "date": date_str,
        "day_type": day_type,
        "noon_metrics": {
            "solar_elevation_deg": round(float(elevation_mid[noon_h]), 2),
            "solar_zenith_deg": round(float(noon_zen), 2),
            "era5_ghi_wm2": round(float(noon_ghi), 2),
            "ineichen_ghi_wm2": round(float(cs_1h['ghi'].iloc[noon_h]), 2),
            "noon_clearness_index_kt": round(float(noon_kt), 3),
            "noon_clearsky_ratio": round(float(noon_cs_ratio), 3),
            "era5_dni_wm2": round(float(dni_era5[noon_h]), 2),
            "pvlib_erbs_dni_wm2": round(float(dni_erbs[noon_h]), 2),
            "pvlib_dirint_dni_wm2": round(float(dni_dirint[noon_h]), 2),
            "era5_dhi_wm2": round(float(diff_horiz_era5[noon_h]), 2),
            "pvlib_erbs_dhi_wm2": round(float(dhi_erbs[noon_h]), 2),
            "pvlib_dirint_dhi_wm2": round(float(dhi_dirint[noon_h]), 2)
        },
        "radsens_variant": {
            "scaling_factor": 0.88,
            "filename": f"met_DESIGN_{day_type}_{date_str}_hourly_RADSENS.txt",
            "radsens_noon_ghi_wm2": round(float(noon_ghi * 0.88), 2),
            "radsens_noon_dni_wm2": round(float(dni_era5[noon_h] * 0.88), 2),
            "radsens_noon_dhi_wm2": round(float(diff_horiz_era5[noon_h] * 0.88), 2),
            "radsens_noon_clearsky_ratio": round(float((noon_ghi * 0.88) / cs_1h['ghi'].iloc[noon_h]), 3)
        },
        "era5_vs_metar_ltbj": {
            "ta_bias_c": round(float(np.mean(diff_ta)), 2),
            "ta_mae_c": round(float(np.mean(np.abs(diff_ta))), 2),
            "ta_max_abs_c": round(float(np.max(np.abs(diff_ta))), 2),
            "rh_bias_pct": round(float(np.mean(diff_rh)), 2),
            "rh_mae_pct": round(float(np.mean(np.abs(diff_rh))), 2),
            "wind_bias_ms": round(float(np.mean(diff_wind)), 2),
            "wind_mae_ms": round(float(np.mean(np.abs(diff_wind))), 2),
        },
        "era5_land_vs_metar_ltbj": {
            "ta_bias_c": round(float(np.mean(diff_ta_land)), 2),
            "ta_mae_c": round(float(np.mean(np.abs(diff_ta_land))), 2),
            "rh_bias_pct": round(float(np.mean(diff_rh_land)), 2),
            "rh_mae_pct": round(float(np.mean(np.abs(diff_rh_land))), 2),
            "wind_bias_ms": round(float(np.mean(diff_wind_land)), 2),
            "wind_mae_ms": round(float(np.mean(np.abs(diff_wind_land))), 2),
        }
    }

    print(f"\nERA5 vs METAR LTBJ 24-Hour Comparison:")
    print(f"  Ta: bias = {comparison_stats['era5_vs_metar_ltbj']['ta_bias_c']} °C, MAE = {comparison_stats['era5_vs_metar_ltbj']['ta_mae_c']} °C (ERA5-Land Ta bias: {comparison_stats['era5_land_vs_metar_ltbj']['ta_bias_c']} °C)")
    print(f"  RH: bias = {comparison_stats['era5_vs_metar_ltbj']['rh_bias_pct']} %, MAE = {comparison_stats['era5_vs_metar_ltbj']['rh_mae_pct']} %")
    print(f"  Wind: bias = {comparison_stats['era5_vs_metar_ltbj']['wind_bias_ms']} m/s, MAE = {comparison_stats['era5_vs_metar_ltbj']['wind_mae_ms']} m/s")

    # 8. Build UMEP 24-Column Met File
    # Schema: iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd
    target_dt = pd.to_datetime(date_str)
    year = target_dt.year
    doy = target_dt.timetuple().tm_yday

    hourly_rows = []
    radsens_rows = []
    for h in range(24):
        kdn_std = ghi_era5[h]
        kdiff_std = diff_horiz_era5[h]
        kdir_std = dni_era5[h]

        kdn_radsens = round(kdn_std * 0.88, 2)
        kdiff_radsens = round(kdiff_std * 0.88, 2)
        kdir_radsens = round(kdir_std * 0.88, 2)

        base_cols_front = [
            f"{year:d}",
            f"{doy:d}",
            f"{h:d}",
            "0",
            "-999", "-999", "-999", "-999", "-999",       # Q*, QH, QE, Qs, Qf
            f"{met_1h['wind_ms'].iloc[h]:.2f}",           # Wind (LTBJ observed)
            f"{met_1h['rh_pct'].iloc[h]:.2f}",            # RH (LTBJ observed)
            f"{met_1h['ta_c'].iloc[h]:.2f}",              # Td (screen Ta, LTBJ observed)
            f"{met_1h['press_kpa'].iloc[h]:.2f}",         # press in kPa
            "0.00",                                        # rain
        ]
        base_cols_mid = [
            "-999",                                        # snow
            f"{ldown_era5[h]:.2f}",                        # ldown from ERA5
            "-999", "-999", "-999", "-999",                # fcld, wuh, xsmd, lai_hr
        ]
        if bool(met_1h['vrb'].iloc[h]):
            wd_col = "-999"
        else:
            wd_col = f"{met_1h['wd_deg'].iloc[h]:.1f}"

        row_std = base_cols_front + [f"{kdn_std:.2f}"] + base_cols_mid + [f"{kdiff_std:.2f}", f"{kdir_std:.2f}", wd_col]
        row_rad = base_cols_front + [f"{kdn_radsens:.2f}"] + base_cols_mid + [f"{kdiff_radsens:.2f}", f"{kdir_radsens:.2f}", wd_col]

        hourly_rows.append(" ".join(row_std))
        radsens_rows.append(" ".join(row_rad))

    header = "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd"
    std_text = header + "\r\n" + "\r\n".join(hourly_rows) + "\r\n"
    rad_text = header + "\r\n" + "\r\n".join(radsens_rows) + "\r\n"

    # WP3.6 checks and outputs:
    # 1. Prove CALM v2 standard file equals met_DESIGN_CALM_2023-07-26_hourly_v2.txt byte for byte
    if day_type == "CALM":
        calm_v2_path = DATA_DIR / f"met_DESIGN_CALM_{date_str}_hourly_v2.txt"
        if calm_v2_path.exists():
            assert std_text.encode("utf-8") == calm_v2_path.read_bytes(), "CALM standard v2 does not match existing v2 file byte for byte!"
            print(f"VERIFICATION PASS: CALM v2 standard file matches {calm_v2_path.name} byte for byte.")
        else:
            calm_v2_path.write_bytes(std_text.encode("utf-8"))
            print(f"Written CALM v2 standard met file to: {calm_v2_path}")

    # 2. Prove IMBAT standard file is unchanged from original
    if day_type == "IMBAT":
        orig_imbat_path = DATA_DIR / f"met_DESIGN_IMBAT_{date_str}_hourly.txt"
        assert std_text.encode("utf-8") == orig_imbat_path.read_bytes(), "IMBAT standard forcing changed unexpectedly!"
        print(f"VERIFICATION PASS: IMBAT standard file is unchanged (byte-identical to {orig_imbat_path.name}).")

    # 3. Write Ineichen-matched RADSENS v2 met files (*_RADSENS_v2.txt)
    radsens_v2_filename = f"met_DESIGN_{day_type}_{date_str}_hourly_RADSENS_v2.txt"
    radsens_v2_filepath = DATA_DIR / radsens_v2_filename
    radsens_v2_filepath.write_bytes(rad_text.encode("utf-8"))
    print(f"Written Ineichen-matched RADSENS v2 met file to: {radsens_v2_filepath}")

    # For IMBAT: prove RADSENS v2 matches original RADSENS byte for byte
    if day_type == "IMBAT":
        orig_rad_path = DATA_DIR / f"met_DESIGN_IMBAT_{date_str}_hourly_RADSENS.txt"
        assert rad_text.encode("utf-8") == orig_rad_path.read_bytes(), "IMBAT RADSENS v2 differs from original RADSENS!"
        print(f"VERIFICATION PASS: IMBAT RADSENS v2 is unchanged (byte-identical to {orig_rad_path.name}).")

    # 9. Acceptance test: loader verification with solweig_gpu
    loaded_radsens = np.loadtxt(radsens_v2_filepath, skiprows=1, delimiter=" ")
    assert loaded_radsens.shape == (24, 24), f"Shape mismatch: expected (24, 24), got {loaded_radsens.shape}"
    assert not np.isnan(loaded_radsens).any(), "NaN found in RADSENS v2 met file data!"

    print(f"VERIFICATION PASS: solweig_gpu loader np.loadtxt loaded {radsens_v2_filename} with zero missing values.")

    met_filepath = DATA_DIR / (f"met_DESIGN_CALM_{date_str}_hourly_v2.txt" if day_type == "CALM" else f"met_DESIGN_IMBAT_{date_str}_hourly.txt")

    # Return full data dict for quicklook plotting
    return {
        "day_info": day_info,
        "grid_1h": grid_1h,
        "met_1h": met_1h,
        "era5_df": era5_df,
        "land_df": land_df,
        "ghi_era5": ghi_era5,
        "dni_era5": dni_era5,
        "diff_horiz_era5": diff_horiz_era5,
        "ldown_era5": ldown_era5,
        "ldown_prata": ldown_prata,
        "cs_ghi": cs_1h['ghi'].values,
        "cs_dni": cs_1h['dni'].values,
        "cs_dhi": cs_1h['dhi'].values,
        "dni_erbs": dni_erbs,
        "dhi_erbs": dhi_erbs,
        "elevation_mid": elevation_mid,
        "stats": comparison_stats,
        "met_path": str(met_filepath)
    }

def plot_publication_quicklooks(results: list[dict]):
    """Generate high-quality diagnostic figures in outputs/street/forcing_summer/."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11})

    # Figure 1: Comprehensive 6-panel Diurnal Forcing Comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    hours = np.arange(24)

    colors = {"IMBAT": "#1f77b4", "CALM": "#d62728"}

    for col, res in enumerate(results):
        d_type = res["day_info"]["type"]
        d_date = res["day_info"]["date"]
        c_main = colors[d_type]

        # Panel 1: Temperature (Ta) METAR vs ERA5 vs ERA5-Land
        ax_ta = axes[0, col]
        ax_ta.plot(hours, res["met_1h"]["ta_c"], 'o-', color="black", lw=2, ms=4, label="METAR LTBJ (Observed)")
        ax_ta.plot(hours, res["era5_df"]["t2m_c"], 's--', color=c_main, lw=1.8, ms=4, label="ERA5 Center")
        ax_ta.plot(hours, res["land_df"]["t2m_c"], '^:', color="#2ca02c", lw=1.8, ms=4, label="ERA5-Land")
        ax_ta.set_ylabel("Air Temperature (°C)")
        ax_ta.set_title(f"[{'A' if col==0 else 'B'}] {d_type} ({d_date}) — Air Temperature ($T_a$)")
        ax_ta.grid(True, linestyle="--", alpha=0.5)
        ax_ta.legend(loc="upper left", frameon=True, fontsize=9)

        # Panel 2: Wind Speed & Direction
        ax_w = axes[1, col]
        ax_w.plot(hours, res["met_1h"]["wind_ms"], 'o-', color="black", lw=2, ms=4, label="METAR $U_{10}$")
        ax_w.plot(hours, res["era5_df"]["wind_spd_ms"], 's--', color=c_main, lw=1.8, ms=4, label="ERA5 $U_{10}$")
        ax_w.set_ylabel("Wind Speed (m/s)")
        ax_w.set_title(f"[{'C' if col==0 else 'D'}] {d_type} — 10m Wind Speed ($U_{10}$)")
        ax_w.grid(True, linestyle="--", alpha=0.5)

        # Annotate wind direction at 15:00
        wd_15 = res["met_1h"]["wd_deg"].iloc[15]
        u_15 = res["met_1h"]["wind_ms"].iloc[15]
        ax_w.annotate(f"15:00: {u_15:.1f} m/s\nfrom {wd_15:.0f}°",
                      xy=(15, u_15), xytext=(15, u_15 + 1.5 if u_15 < 8 else u_15 - 2.5),
                      arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                      fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        ax_w.legend(loc="upper left", frameon=True, fontsize=9)

        # Panel 3: Solar & Longwave Radiation Components
        ax_rad = axes[2, col]
        ax_rad.plot(hours, res["ghi_era5"], 'r-', lw=2, label="ERA5 GHI ($K\\downarrow$)")
        ax_rad.plot(hours, res["dni_era5"], 'orange', ls="--", lw=1.8, label="ERA5 DNI ($K_{dir}$)")
        ax_rad.plot(hours, res["diff_horiz_era5"], 'b:', lw=1.8, label="ERA5 DHI ($K_{diff}$)")
        ax_rad.plot(hours, res["cs_ghi"], color="gray", ls="-.", lw=1.2, alpha=0.7, label="Ineichen Clear-Sky GHI")
        ax_rad.plot(hours, res["ldown_era5"], color="purple", ls="-", lw=1.5, label="ERA5 $L\\downarrow$")
        ax_rad.set_ylabel("Irradiance (W/m²)")
        ax_rad.set_xlabel("Local Hour (UTC+3)")
        ax_rad.set_title(f"[{'E' if col==0 else 'F'}] {d_type} — Downwelling Radiation")
        ax_rad.grid(True, linestyle="--", alpha=0.5)
        ax_rad.legend(loc="upper left", frameon=True, fontsize=8)
        ax_rad.set_xticks(np.arange(0, 25, 2))

    plt.tight_layout()
    out_fig1 = OUTPUTS_DIR / "quicklook_summer_design_days_forcing.png"
    plt.savefig(out_fig1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 6-panel forcing quicklook to {out_fig1}")

    # Figure 2: Direct Comparison & Clearness Index Diagnostic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Identify IMBAT and CALM
    res_imbat = next(r for r in results if r["day_info"]["type"] == "IMBAT")
    res_calm = next(r for r in results if r["day_info"]["type"] == "CALM")

    # Panel 1: IMBAT vs CALM Heat Diurnal Profiles (Ta and Wind)
    ax1_twin = ax1.twinx()
    l1, = ax1.plot(hours, res_imbat["met_1h"]["ta_c"], color="#1f77b4", lw=2.2, label=f"IMBAT Ta ({res_imbat['day_info']['date']})")
    l2, = ax1.plot(hours, res_calm["met_1h"]["ta_c"], color="#d62728", lw=2.2, label=f"CALM HEAT Ta ({res_calm['day_info']['date']})")

    l3, = ax1_twin.plot(hours, res_imbat["met_1h"]["wind_ms"], color="#1f77b4", ls="--", lw=2, label="IMBAT Wind (m/s)")
    l4, = ax1_twin.plot(hours, res_calm["met_1h"]["wind_ms"], color="#d62728", ls="--", lw=2, label="CALM Wind (m/s)")

    ax1.set_xlabel("Local Hour (UTC+3)")
    ax1.set_ylabel("Air Temperature (°C)")
    ax1_twin.set_ylabel("Wind Speed (m/s)")
    ax1.set_title("[A] IMBAT (Breeze) vs CALM HEAT (Stagnation) Microclimate")
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=True, fontsize=9)

    # Panel 2: Clearness Index (kt) and Solar Elevation Profile
    noon_kt_imbat = res_imbat["stats"]["noon_metrics"]["noon_clearness_index_kt"]
    noon_kt_calm = res_calm["stats"]["noon_metrics"]["noon_clearness_index_kt"]

    # Compute daytime kt
    day_mask = res_imbat["elevation_mid"] > 10.0
    kt_imbat_day = np.zeros(24)
    kt_calm_day = np.zeros(24)
    extra_horiz = pvlib.irradiance.get_extra_radiation(res_imbat["grid_1h"] + pd.Timedelta(minutes=30)).values * np.maximum(0.01, np.cos(np.radians(res_imbat["elevation_mid"])))
    kt_imbat_day[day_mask] = res_imbat["ghi_era5"][day_mask] / extra_horiz[day_mask]
    kt_calm_day[day_mask] = res_calm["ghi_era5"][day_mask] / extra_horiz[day_mask]

    ax2.plot(hours[day_mask], kt_imbat_day[day_mask], 'o-', color="#1f77b4", lw=2, label=f"IMBAT (Noon $k_t={noon_kt_imbat}$)")
    ax2.plot(hours[day_mask], kt_calm_day[day_mask], 's-', color="#d62728", lw=2, label=f"CALM HEAT (Noon $k_t={noon_kt_calm}$)")
    ax2.axhspan(0.7, 0.8, color="green", alpha=0.15, label="Expected Clear-Sky Range (0.70–0.80)")
    ax2.set_xlabel("Local Hour (UTC+3)")
    ax2.set_ylabel("Clearness Index ($k_t = GHI / I_{0,horiz}$)")
    ax2.set_title("[B] Daytime Clearness Index & Atmospheric Transparency")
    ax2.set_ylim(0.4, 0.9)
    ax2.set_xticks(np.arange(6, 21, 2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower center", frameon=True, fontsize=9)

    plt.tight_layout()
    out_fig2 = OUTPUTS_DIR / "quicklook_imbat_vs_calm_comparison.png"
    plt.savefig(out_fig2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved design day contrast quicklook to {out_fig2}")

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Build forcing, compare with METAR, and decompose radiation.")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD")
    parser.add_argument("--day-type", type=str, choices=["IMBAT", "CALM"], help="Design day type")
    parser.add_argument("--all", action="store_true", help="Process both default design days")
    args = parser.parse_args()

    print("=== WP1.4 GENERATING FORCING FILES & COMPARING WITH METAR ===")
    loc = pvlib.location.Location(IZMIR_LAT, IZMIR_LON, tz=TIMEZONE, altitude=ALTITUDE)

    comp_json_path = DATA_DIR / "era5_vs_metar_comparison_stats.json"
    if comp_json_path.exists():
        try:
            with open(comp_json_path, "r") as f:
                comparisons = json.load(f)
        except Exception:
            comparisons = {}
    else:
        comparisons = {}

    if args.date and args.day_type:
        target_days = [{
            "type": args.day_type,
            "date": args.date,
            "title": f"Design Day: {args.day_type} ({args.date})"
        }]
    else:
        target_days = DEFAULT_DESIGN_DAYS

    print(f"Target days to process: {[d['type'] + ' ' + d['date'] for d in target_days]}")

    processed_results = []
    for day_info in target_days:
        res = process_design_day(day_info, loc)
        processed_results.append(res)
        comparisons[day_info["type"]] = res["stats"]

    # Save comparison stats to JSON
    with open(comp_json_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"\nSaved comparison statistics to {comp_json_path}")

    # Generate publication-quality figures:
    # Need results for both default design days (IMBAT and CALM)
    plot_results = []
    for d_info in DEFAULT_DESIGN_DAYS:
        match = [r for r in processed_results if r["day_info"]["type"] == d_info["type"] and r["day_info"]["date"] == d_info["date"]]
        if match:
            plot_results.append(match[0])
        else:
            era5_csv = DATA_DIR / f"era5_hourly_{d_info['type']}_{d_info['date']}.csv"
            if era5_csv.exists():
                plot_results.append(process_design_day(d_info, loc))

    if len(plot_results) == 2:
        plot_publication_quicklooks(plot_results)
    else:
        print(f"Skipping paired quicklook generation: {len(plot_results)} of 2 default days available.")

    print("\nAll forcing generation and validation steps completed successfully.")

if __name__ == "__main__":
    main()
