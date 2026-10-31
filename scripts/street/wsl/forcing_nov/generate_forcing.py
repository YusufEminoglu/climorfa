#!/usr/bin/env python3
"""
Generate UMEP 24-column hourly forcing and 10-minute CSV forcing for
the Izmir field campaign analogue days.
Work Package: WP0.5a.

Features:
- pvlib clear-sky (Ineichen, monthly Linke turbidity) for 38.42 N, 27.14 E, 20 m altitude, UTC+3
- Prata (1996) clear-sky downwelling longwave radiation from Ta and vapour pressure
- Interpolation of real METAR observations (LTBJ)
- 24-column UMEP met text file matching solweig_gpu reader
- 10-minute CSV for synthetic field generator
- Validation with np.loadtxt
- Quicklook PNG figure
"""
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

# Output directories
BASE_DIR = Path(__file__).resolve().parents[4]
METAR_DIR = BASE_DIR / "data" / "street" / "wsl" / "metar_nov"
OUT_DIR = BASE_DIR / "data" / "street" / "wsl" / "forcing_nov"

# Geographic location for Izmir (central campaign area)
LAT = 38.42
LON = 27.14
ALTITUDE = 20.0
TIMEZONE = "Europe/Istanbul"  # UTC+3 year-round

# Target placeholder dates
TARGET_DATES = {
    "day1": "2026-11-12",
    "day2": "2026-11-13"
}

# Selected analogue source dates:
# 1) Top consecutive pair (primary)
ANALOGUE_PAIRS = [
    {
        "type": "consecutive",
        "day1": {"source_date": "2025-11-02", "target_date": "2026-11-12"},
        "day2": {"source_date": "2025-11-03", "target_date": "2026-11-13"}
    },
    {
        "type": "top_individual",
        "day1": {"source_date": "2024-11-12", "target_date": "2026-11-12"},
        "day2": {"source_date": "2023-11-14", "target_date": "2026-11-13"}
    }
]

def compute_prata_longwave(ta_c: np.ndarray, rh_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute downwelling clear-sky longwave radiation using Prata (1996).
    Formula:
      T_k = ta_c + 273.15
      e_s = 6.112 * exp((17.67 * ta_c) / (ta_c + 243.5))   [hPa] (Magnus-Tetens)
      e_a = (rh_pct / 100.0) * e_s                          [hPa]
      w = 46.5 * (e_a / T_k)                                [dimensionless water vapor parameter]
      eps_cs = 1.0 - (1.0 + w) * exp(-sqrt(1.2 + 3.0 * w))  [clear-sky emissivity]
      sigma = 5.670374419e-8                                [W m^-2 K^-4]
      ldown = eps_cs * sigma * (T_k ** 4)                   [W m^-2]
    """
    t_k = ta_c + 273.15
    es = 6.112 * np.exp((17.67 * ta_c) / (ta_c + 243.5))
    ea = (rh_pct / 100.0) * es
    w = 46.5 * (ea / t_k)
    eps_cs = 1.0 - (1.0 + w) * np.exp(-np.sqrt(1.2 + 3.0 * w))
    sigma = 5.670374419e-8
    ldown = eps_cs * sigma * (t_k ** 4)
    return eps_cs, ldown

def interpolate_metar(source_date_str: str, target_grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolate LTBJ METAR observations onto target regular time grid.
    Uses circular vector decomposition for wind direction.
    """
    metar_path = METAR_DIR / "LTBJ_nov_2023_2025.csv"
    df = pd.read_csv(metar_path, low_memory=False)
    df['valid_utc'] = pd.to_datetime(df['valid'], utc=True)
    df['valid_local'] = df['valid_utc'].dt.tz_convert(TIMEZONE)

    # Filter +/- 24 hours around source date for seamless boundary interpolation
    src_date = pd.to_datetime(source_date_str).date()
    t_start = pd.Timestamp(src_date, tz=TIMEZONE) - pd.Timedelta(hours=12)
    t_end = pd.Timestamp(src_date, tz=TIMEZONE) + pd.Timedelta(hours=36)
    subset = df[(df['valid_local'] >= t_start) & (df['valid_local'] <= t_end)].copy()
    subset = subset.sort_values(by='valid_local').drop_duplicates(subset='valid_local').reset_index(drop=True)

    # Extract numerical variables
    ta_c = (pd.to_numeric(subset['tmpf'], errors='coerce') - 32.0) * 5.0 / 9.0
    rh_pct = pd.to_numeric(subset['relh'], errors='coerce').clip(lower=0.0, upper=100.0)
    wind_ms = (pd.to_numeric(subset['sknt'], errors='coerce') * 0.514444).clip(lower=0.0)

    # Pressure in kPa: use alti (inHg * 3.38639 -> kPa) or mslp (hPa * 0.1 -> kPa)
    alti = pd.to_numeric(subset['alti'], errors='coerce') * 3.3863886666667
    mslp = pd.to_numeric(subset['mslp'], errors='coerce') * 0.1
    press_kpa = alti.fillna(mslp).fillna(101.325)

    # Wind direction with vector decomposition
    wd_deg = pd.to_numeric(subset['drct'], errors='coerce').fillna(0.0)
    u_wind = -wind_ms * np.sin(np.radians(wd_deg))
    v_wind = -wind_ms * np.cos(np.radians(wd_deg))

    # Time coordinate in seconds from start of target day
    obs_times = subset['valid_local']
    ref_time = pd.Timestamp(src_date, tz=TIMEZONE)
    obs_sec = (obs_times - ref_time).dt.total_seconds().values

    # Target grid seconds
    target_ref = target_grid[0].floor('D')
    target_sec = (target_grid - target_ref).total_seconds().values

    # Linear interpolation
    ta_interp = np.interp(target_sec, obs_sec, ta_c)
    rh_interp = np.interp(target_sec, obs_sec, rh_pct)
    press_interp = np.interp(target_sec, obs_sec, press_kpa)
    u_interp = np.interp(target_sec, obs_sec, u_wind)
    v_interp = np.interp(target_sec, obs_sec, v_wind)

    # Reconstruct wind speed and direction
    wind_interp = np.sqrt(u_interp**2 + v_interp**2)
    # Direct interpolation of scalar wind speed preserves METAR wind magnitude better at low speeds
    wind_scalar_interp = np.interp(target_sec, obs_sec, wind_ms)

    # Recompute meteorological wind direction: angle from which wind is blowing (0=N, 90=E, 180=S, 270=W)
    wd_interp = (np.degrees(np.arctan2(-u_interp, -v_interp))) % 360.0
    # Where calm, set direction to 0
    wd_interp[wind_scalar_interp < 0.2] = 0.0

    return pd.DataFrame({
        'ta_c': ta_interp,
        'rh_pct': rh_interp,
        'wind_ms': wind_scalar_interp,
        'wd_deg': wd_interp,
        'press_kpa': press_interp
    }, index=target_grid)

def process_day(source_date_str: str, target_date_str: str):
    """
    Process one analogue day for a given target placeholder date.
    Generates:
    1) 24-column UMEP met text file (hourly)
    2) 10-minute CSV file
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loc = pvlib.location.Location(LAT, LON, tz=TIMEZONE, altitude=ALTITUDE)

    # Target date timestamps
    t_start = pd.Timestamp(f"{target_date_str} 00:00:00", tz=TIMEZONE)
    t_end_1h = pd.Timestamp(f"{target_date_str} 23:00:00", tz=TIMEZONE)
    t_end_10m = pd.Timestamp(f"{target_date_str} 23:50:00", tz=TIMEZONE)

    grid_1h = pd.date_range(t_start, t_end_1h, freq="1h")
    grid_10m = pd.date_range(t_start, t_end_10m, freq="10min")

    # 1. pvlib clear-sky radiation (Ineichen)
    cs_1h = loc.get_clearsky(grid_1h, model="ineichen")
    cs_10m = loc.get_clearsky(grid_10m, model="ineichen")

    # 2. Interpolate METAR observations
    met_1h = interpolate_metar(source_date_str, grid_1h)
    met_10m = interpolate_metar(source_date_str, grid_10m)

    # 3. Prata (1996) clear-sky downwelling longwave
    _, ldown_1h = compute_prata_longwave(met_1h['ta_c'].values, met_1h['rh_pct'].values)
    _, ldown_10m = compute_prata_longwave(met_10m['ta_c'].values, met_10m['rh_pct'].values)

    # 4. Prepare Hourly UMEP Format (24 columns)
    target_dt = pd.to_datetime(target_date_str)
    year = target_dt.year
    doy = target_dt.timetuple().tm_yday

    # 24-column schema:
    # iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd
    hourly_rows = []
    for h in range(24):
        row = [
            f"{year:d}",
            f"{doy:d}",
            f"{h:d}",
            "0",
            "-999", "-999", "-999", "-999", "-999",  # Q*, QH, QE, Qs, Qf
            f"{met_1h['wind_ms'].iloc[h]:.2f}",
            f"{met_1h['rh_pct'].iloc[h]:.2f}",
            f"{met_1h['ta_c'].iloc[h]:.2f}",
            f"{met_1h['press_kpa'].iloc[h]:.2f}",
            "0.00",                                   # rain
            f"{cs_1h['ghi'].iloc[h]:.2f}",            # Kdn (global horizontal)
            "-999",                                   # snow
            f"{ldown_1h[h]:.2f}",                     # ldown (Prata 1996)
            "-999", "-999", "-999", "-999",           # fcld, wuh, xsmd, lai_hr
            f"{cs_1h['dhi'].iloc[h]:.2f}",            # Kdiff (diffuse horizontal)
            f"{cs_1h['dni'].iloc[h]:.2f}",            # Kdir (direct normal)
            f"{met_1h['wd_deg'].iloc[h]:.1f}"         # Wd
        ]
        hourly_rows.append(" ".join(row))

    header = "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd"
    hourly_filename = f"met_ANALOGUE_{source_date_str}_for_{target_date_str}_hourly.txt"
    hourly_filepath = OUT_DIR / hourly_filename
    with open(hourly_filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(hourly_rows) + "\n")

    print(f"Saved hourly UMEP metfile ({len(hourly_rows)} hours): {hourly_filepath}")

    # 5. Prepare 10-minute CSV
    df_10m = pd.DataFrame({
        "timestamp_local": grid_10m.strftime("%Y-%m-%d %H:%M:%S"),
        "target_year": year,
        "target_doy": doy,
        "hour": grid_10m.hour,
        "minute": grid_10m.minute,
        "ta_c": met_10m['ta_c'].round(2),
        "rh_pct": met_10m['rh_pct'].round(2),
        "wind_ms": met_10m['wind_ms'].round(2),
        "wind_dir_deg": met_10m['wd_deg'].round(1),
        "press_kpa": met_10m['press_kpa'].round(2),
        "ghi_wm2": cs_10m['ghi'].round(2),
        "dni_wm2": cs_10m['dni'].round(2),
        "dhi_wm2": cs_10m['dhi'].round(2),
        "ldown_wm2": np.round(ldown_10m, 2),
        "is_analogue": True,
        "source_station": "LTBJ",
        "source_date": source_date_str,
        "target_date": target_date_str,
        "radiation_model": "pvlib_ineichen_linke",
        "longwave_model": "prata_1996"
    })
    csv_filename = f"met_ANALOGUE_{source_date_str}_for_{target_date_str}_10min.csv"
    csv_filepath = OUT_DIR / csv_filename
    df_10m.to_csv(csv_filepath, index=False)
    print(f"Saved 10-minute CSV forcing ({len(df_10m)} steps): {csv_filepath}")

    # 6. Validate with solweig_gpu's exact loader
    loaded_data = np.loadtxt(hourly_filepath, skiprows=1, delimiter=" ")
    assert loaded_data.shape == (24, 24), f"Expected shape (24, 24), got {loaded_data.shape}"
    print(f"Verification: np.loadtxt successfully loaded {hourly_filename} with shape {loaded_data.shape}")

    return {
        "hourly_path": hourly_filepath,
        "csv_path": csv_filepath,
        "met_1h": met_1h,
        "cs_1h": cs_1h,
        "ldown_1h": ldown_1h,
        "df_10m": df_10m
    }

def generate_quicklook(day1_res, day2_res, pair_name: str = "consecutive"):
    """
    Generate quicklook plot of Ta, RH, wind speed/dir, solar radiation (GHI/DNI/DHI), and Ldown.
    """
    df1 = day1_res["df_10m"]
    df2 = day2_res["df_10m"]
    comb = pd.concat([df1, df2], ignore_index=True)
    t = pd.to_datetime(comb["timestamp_local"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.patch.set_facecolor("white")

    # 1. Temperature & Humidity
    ax1 = axes[0]
    color_ta = "#d95f02"
    color_rh = "#1f78b4"
    ax1.plot(t, comb["ta_c"], color=color_ta, lw=2, label="Air Temp ($T_a$, °C)")
    ax1.set_ylabel("Air Temp (°C)", color=color_ta, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_ta)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_rh = ax1.twinx()
    ax1_rh.plot(t, comb["rh_pct"], color=color_rh, lw=1.8, linestyle="--", label="Rel Humidity (RH, %)")
    ax1_rh.set_ylabel("Rel. Humidity (%)", color=color_rh, fontsize=11, fontweight="bold")
    ax1_rh.tick_params(axis="y", labelcolor=color_rh)
    ax1_rh.set_ylim(0, 100)
    ax1.set_title(f"CLIMORFA WP0.5a November Campaign Forcing: {day1_res['df_10m']['source_date'].iloc[0]} & {day2_res['df_10m']['source_date'].iloc[0]} (for 2026-11-12/13)", fontsize=13, fontweight="bold")

    # 2. Wind Speed and Direction
    ax2 = axes[1]
    color_w = "#2ca02c"
    ax2.plot(t, comb["wind_ms"], color=color_w, lw=2, label="10-m Wind Speed ($U_{10}$, m/s)")
    ax2.axhline(6.0, color="red", linestyle=":", lw=1.5, label="Clear-sky threshold (6 m/s)")
    ax2.set_ylabel("Wind Speed (m/s)", color=color_w, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_w)
    ax2.set_ylim(0, 8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_dir = ax2.twinx()
    ax2_dir.scatter(t, comb["wind_dir_deg"], color="#7570b3", s=8, alpha=0.6, label="Wind Dir (°)")
    ax2_dir.set_ylabel("Wind Dir (°)", color="#7570b3", fontsize=11)
    ax2_dir.tick_params(axis="y", labelcolor="#7570b3")
    ax2_dir.set_ylim(0, 360)
    ax2_dir.set_yticks([0, 90, 180, 270, 360])
    ax2_dir.set_yticklabels(["N", "E", "S", "W", "N"])

    # 3. Shortwave Radiation (pvlib Ineichen)
    ax3 = axes[2]
    ax3.plot(t, comb["ghi_wm2"], color="#e7298a", lw=2, label=r"GHI ($K\downarrow$)")
    ax3.plot(t, comb["dni_wm2"], color="#e6ab02", lw=1.8, label=r"DNI ($K_{dir}$)")
    ax3.plot(t, comb["dhi_wm2"], color="#66a61e", lw=1.5, label=r"DHI ($K_{diff}$)")
    ax3.set_ylabel("Shortwave (W/m²)", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 750)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)

    # 4. Longwave Radiation (Prata 1996)
    ax4 = axes[3]
    ax4.plot(t, comb["ldown_wm2"], color="#7570b3", lw=2, label=r"Downwelling Longwave ($L\downarrow$, Prata 1996)")
    ax4.set_ylabel("Longwave (W/m²)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Local Time (UTC+3, Europe/Istanbul)", fontsize=11, fontweight="bold")
    ax4.set_ylim(250, 400)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", framealpha=0.9)

    # Day divider line
    midnight = pd.Timestamp(f"{day2_res['df_10m']['target_date'].iloc[0]} 00:00:00", tz=TIMEZONE)
    for ax in axes:
        ax.axvline(midnight, color="black", linestyle="-.", lw=1.5, alpha=0.7)

    plt.tight_layout()
    plot_filename = f"quicklook_forcing_nov_{pair_name}.png" if pair_name != "consecutive" else "quicklook_forcing_nov.png"
    plot_path = OUT_DIR / plot_filename
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Quicklook figure saved to: {plot_path}")
    return plot_path

def main():
    print("=== WP0.5a FORCING GENERATION ===")
    results_by_pair = {}

    for pair in ANALOGUE_PAIRS:
        p_type = pair["type"]
        d1 = pair["day1"]
        d2 = pair["day2"]
        print(f"\nProcessing {p_type.upper()} pair: {d1['source_date']} and {d2['source_date']}...")

        res1 = process_day(d1["source_date"], d1["target_date"])
        res2 = process_day(d2["source_date"], d2["target_date"])

        img_path = generate_quicklook(res1, res2, pair_name=p_type)
        results_by_pair[p_type] = {
            "day1": res1,
            "day2": res2,
            "img": img_path
        }

    print("\nAll forcing files and quicklook figures generated successfully!")

if __name__ == "__main__":
    main()
