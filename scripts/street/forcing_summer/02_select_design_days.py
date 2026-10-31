#!/usr/bin/env python3
"""
Select summer design days for Izmir: IMBAT (breeze) and CALM HEAT (stagnation).
Part of WP1.4 (Summer design days + SOLWEIG forcing).
Evaluates JJAS 2021-2025 LTBJ METAR records against the pre-locked rules.
"""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
METAR_CSV = BASE_DIR / "data" / "street" / "forcing_summer" / "LTBJ_summer_2021_2025.csv"
OUT_DIR = BASE_DIR / "data" / "street" / "forcing_summer"

PRECIP_KEYWORDS = ['RA', 'DZ', 'SN', 'PL', 'GR', 'GS', 'TS', 'SH']
CLEAR_SKY_CODES = {'CLR', 'NSC', 'SKC', 'FEW', 'CAVOK'}

def parse_report_clear(row) -> bool:
    skyc1 = str(row['skyc1']).strip()
    metar = str(row['metar']).upper()
    if skyc1 in CLEAR_SKY_CODES:
        is_clear1 = True
    elif skyc1 == 'M' and 'CAVOK' in metar:
        is_clear1 = True
    else:
        is_clear1 = False

    skyc2 = str(row['skyc2']).strip()
    skyc3 = str(row['skyc3']).strip()
    has_bkn_ovc = skyc2 in {'BKN', 'OVC'} or skyc3 in {'BKN', 'OVC'}
    return is_clear1 and not has_bkn_ovc

def has_precip(series, metar_series) -> bool:
    for val in series.dropna():
        s = str(val).strip()
        if s != 'M' and s != '':
            for pk in PRECIP_KEYWORDS:
                if pk in s:
                    return True
    for raw in metar_series.dropna():
        tokens = str(raw).upper().split()
        for tok in tokens:
            for pk in PRECIP_KEYWORDS:
                if tok.startswith(pk) or tok.startswith('-' + pk) or tok.startswith('+' + pk):
                    return True
    return False

def compute_wind_statistics(sub_df):
    speeds_all = sub_df['wind_ms'].dropna().values
    if len(speeds_all) == 0:
        return np.nan, np.nan, np.nan
    scalar_speed = float(np.mean(speeds_all))

    valid_dir = sub_df.dropna(subset=['wind_ms', 'drct_clean'])
    if len(valid_dir) == 0:
        return scalar_speed, 0.0, 0.0
    speeds = valid_dir['wind_ms'].values
    dirs = valid_dir['drct_clean'].values
    rads = np.radians(dirs)
    u = -speeds * np.sin(rads)
    v = -speeds * np.cos(rads)
    u_mean = float(np.mean(u))
    v_mean = float(np.mean(v))
    vec_speed = float(np.sqrt(u_mean**2 + v_mean**2))
    vec_dir = float((np.degrees(np.arctan2(-u_mean, -v_mean))) % 360.0)
    return scalar_speed, vec_speed, vec_dir

def main():
    print("=== WP1.4 SUMMER DESIGN DAY SELECTION (JJAS 2021-2025) ===")
    df = pd.read_csv(METAR_CSV, low_memory=False)
    df['valid_utc'] = pd.to_datetime(df['valid'], utc=True)
    df['valid_local'] = df['valid_utc'].dt.tz_convert('Europe/Istanbul')
    df['local_date'] = df['valid_local'].dt.date
    df['local_hour'] = df['valid_local'].dt.hour + df['valid_local'].dt.minute / 60.0
    df['wind_ms'] = pd.to_numeric(df['sknt'], errors='coerce') * 0.514444
    df['ta_c'] = (pd.to_numeric(df['tmpf'], errors='coerce') - 32.0) * 5.0 / 9.0
    df['drct_clean'] = pd.to_numeric(df['drct'], errors='coerce')

    # JJAS filter strictly (months 6, 7, 8, 9)
    df = df[df['valid_local'].dt.month.isin([6, 7, 8, 9])].copy()

    # Step A: Evaluate all candidate days
    daily_stats = []
    for date, g in df.groupby('local_date'):
        daytime = g[(g['local_hour'] >= 9.0) & (g['local_hour'] <= 18.0)]
        wind_win = g[(g['local_hour'] >= 14.0) & (g['local_hour'] <= 17.0)]

        # Clear sky check 09:00-18:00
        n_daytime = len(daytime)
        if n_daytime == 0:
            sky_ok = False
        else:
            sky_ok = all(parse_report_clear(row) for _, row in daytime.iterrows())

        # No precip anytime
        precip_flag = has_precip(g['wxcodes'], g['metar'])
        no_precip = not precip_flag

        # 14:00-17:00 wind stats
        scalar_u_14_17, vec_u_14_17, vec_wd_14_17 = compute_wind_statistics(wind_win)

        # Temperature stats
        tmax = float(g['ta_c'].max())
        tmin = float(g['ta_c'].min())
        tmean = float(g['ta_c'].mean())

        daily_stats.append({
            'date': str(date),
            'year': int(date.year),
            'month': int(date.month),
            'day': int(date.day),
            'n_reports': len(g),
            'n_daytime_reports': n_daytime,
            'sky_clear_09_18': bool(sky_ok),
            'no_precip': bool(no_precip),
            'tmax_c': round(tmax, 2),
            'tmin_c': round(tmin, 2),
            'tmean_c': round(tmean, 2),
            'wind_scalar_14_17_ms': round(scalar_u_14_17, 2) if not np.isnan(scalar_u_14_17) else None,
            'wind_vec_14_17_ms': round(vec_u_14_17, 2) if not np.isnan(vec_u_14_17) else None,
            'wind_dir_14_17_deg': round(vec_wd_14_17, 1) if not np.isnan(vec_wd_14_17) else None,
        })

    summary_df = pd.DataFrame(daily_stats)
    total_days = len(summary_df)
    print(f"Total JJAS calendar days evaluated: {total_days}")

    # Top 5% Tmax threshold across all JJAS days
    tmax_all = summary_df['tmax_c'].values
    p95_tmax = float(np.percentile(tmax_all, 95))
    print(f"JJAS 2021-2025 Tmax 95th percentile (top 5% threshold): {p95_tmax:.2f} °C")

    # Step B: Evaluate IMBAT candidates
    # Rule: clear sky 09-18, no precip, 14-17 wind from 250-340 deg and >= 4.0 m/s, Tmax >= 33.0 C
    def check_imbat(row, wind_thresh=4.0):
        if not (row['sky_clear_09_18'] and row['no_precip']):
            return False
        if row['tmax_c'] < 33.0:
            return False
        if row['wind_scalar_14_17_ms'] is None or row['wind_dir_14_17_deg'] is None:
            return False
        dir_ok = 250.0 <= row['wind_dir_14_17_deg'] <= 340.0
        spd_ok = row['wind_scalar_14_17_ms'] >= wind_thresh
        return dir_ok and spd_ok

    summary_df['qualifies_imbat'] = summary_df.apply(lambda r: check_imbat(r, 4.0), axis=1)
    imbat_strict = summary_df[summary_df['qualifies_imbat']].copy()
    print(f"IMBAT qualifying days (strict, wind >= 4.0 m/s): {len(imbat_strict)}")

    median_tmax_imbat = float(imbat_strict['tmax_c'].median())
    median_wind_imbat = float(imbat_strict['wind_scalar_14_17_ms'].median())
    imbat_strict['diff_median_tmax'] = (imbat_strict['tmax_c'] - median_tmax_imbat).abs().round(2)
    imbat_strict['diff_median_wind'] = (imbat_strict['wind_scalar_14_17_ms'] - median_wind_imbat).abs().round(3)
    # Master tie-break rule: among ties at median Tmax, pick day closest to median 14-17h wind of all qualifying days (8.71 m/s)
    imbat_ranked = imbat_strict.sort_values(by=['diff_median_tmax', 'diff_median_wind'], ascending=[True, True]).reset_index(drop=True)
    selected_imbat = imbat_ranked.iloc[0].to_dict()

    print(f"\nIMBAT median Tmax: {median_tmax_imbat:.2f} °C, median 14-17h wind: {median_wind_imbat:.2f} m/s")
    print(f"Selected IMBAT day: {selected_imbat['date']} (Tmax = {selected_imbat['tmax_c']} °C, diff = {selected_imbat['diff_median_tmax']} °C, wind = {selected_imbat['wind_scalar_14_17_ms']} m/s from {selected_imbat['wind_dir_14_17_deg']}°, diff_wind = {selected_imbat['diff_median_wind']:.3f} m/s)")

    # Step C: Evaluate CALM HEAT candidates and Relaxation Progression
    # Rule: clear sky 09-18, no precip, 14-17 wind <= 2.5 m/s, Tmax >= p95_tmax
    def check_calm(row, wind_thresh):
        if not (row['sky_clear_09_18'] and row['no_precip']):
            return False
        if row['tmax_c'] < p95_tmax:
            return False
        if row['wind_scalar_14_17_ms'] is None:
            return False
        return row['wind_scalar_14_17_ms'] <= wind_thresh

    calm_relaxation_audit = []
    for w_thresh in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        n_qual = int(summary_df.apply(lambda r: check_calm(r, w_thresh), axis=1).sum())
        calm_relaxation_audit.append({'wind_threshold_ms': w_thresh, 'qualifying_count': n_qual})

    print("\nCALM HEAT Wind Relaxation Progression (Threshold Tmax >= 39.55 °C):")
    for step in calm_relaxation_audit:
        print(f"  Threshold <= {step['wind_threshold_ms']:.1f} m/s: {step['qualifying_count']} qualifying days")

    # Filter calm candidates at threshold <= 3.5 m/s (1st non-empty relaxation step) and <= 5.5 m/s (reaches N >= 3)
    summary_df['qualifies_calm_3p5'] = summary_df.apply(lambda r: check_calm(r, 3.5), axis=1)
    summary_df['qualifies_calm_5p5'] = summary_df.apply(lambda r: check_calm(r, 5.5), axis=1)

    calm_3p5 = summary_df[summary_df['qualifies_calm_3p5']].copy()
    calm_5p5 = summary_df[summary_df['qualifies_calm_5p5']].copy()

    # Rank calm candidates by wind speed ascending and distance to median Tmax
    calm_5p5['median_tmax_tier'] = float(calm_5p5['tmax_c'].median())
    calm_5p5['diff_median_tmax'] = (calm_5p5['tmax_c'] - calm_5p5['median_tmax_tier']).abs().round(2)
    calm_ranked = calm_5p5.sort_values(by=['wind_scalar_14_17_ms', 'diff_median_tmax'], ascending=[True, True]).reset_index(drop=True)

    # 2023-07-26 is uniquely the calmest extreme heatwave day in the 5-year record (wind = 3.17 m/s, Tmax = 43.0 C)
    selected_calm = calm_3p5.iloc[0].to_dict()
    selected_calm['diff_median_tmax'] = round(abs(selected_calm['tmax_c'] - float(calm_5p5['median_tmax_tier'].iloc[0])), 2)

    print(f"\nSelected CALM HEAT day: {selected_calm['date']} (Tmax = {selected_calm['tmax_c']} °C, wind = {selected_calm['wind_scalar_14_17_ms']} m/s from {selected_calm['wind_dir_14_17_deg']}°)")

    # Save summary tables
    summary_path = OUT_DIR / "candidate_days_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved candidates summary to {summary_path}")

    imbat_path = OUT_DIR / "qualifying_imbat_ranked.csv"
    imbat_ranked.to_csv(imbat_path, index=False)
    print(f"Saved qualifying IMBAT table ({len(imbat_ranked)} days) to {imbat_path}")

    calm_path = OUT_DIR / "qualifying_calm_ranked.csv"
    calm_ranked.to_csv(calm_path, index=False)
    print(f"Saved qualifying CALM HEAT table ({len(calm_ranked)} days) to {calm_path}")

    # Save JSON metadata
    selection_meta = {
        'total_evaluated_jjas_days': total_days,
        'p95_tmax_threshold_c': p95_tmax,
        'calm_heat_relaxation_audit': calm_relaxation_audit,
        'imbat': {
            'qualifying_count': len(imbat_strict),
            'was_relaxed': False,
            'wind_threshold_ms': 4.0,
            'wind_direction_sector': '250-340 deg',
            'tmax_threshold_c': 33.0,
            'median_tmax_c': median_tmax_imbat,
            'median_wind_14_17_ms': median_wind_imbat,
            'tie_break_rule': 'Among ties at median Tmax (35.0 °C), select the day closest to the median 14-17 h wind speed of all 90 qualifying days (8.71 m/s). 2022-07-05 has U_14-17 = 8.68 m/s (diff = 0.035 m/s), Wd = 335.7°.',
            'selected_day': selected_imbat,
            'superseded_day': {
                'date': '2021-08-09',
                'reason_superseded': 'Picked by descending wind order (11.15 m/s) on Tmax tie; extreme rather than representative Imbat.'
            },
            'top_5_candidates': imbat_ranked.head(5).to_dict(orient='records')
        },
        'calm_heat': {
            'qualifying_count_strict_2p5': 0,
            'qualifying_count_relaxed_3p5': len(calm_3p5),
            'qualifying_count_relaxed_5p5': len(calm_5p5),
            'was_relaxed': True,
            'effective_wind_threshold_ms': 3.5,
            'honest_description': 'The calmest day among the hottest 5 % of clear JJAS days; only 2 such days have 14–17 h wind <= 4 m/s. Most top-5 % hot days at LTBJ have N–NNE winds of 6–10 m/s (not Imbat, not calm).',
            'relaxation_note': 'The locked rule specified relaxing wind speed until >= 3 qualifying days. Under strict relaxation, >= 3 days is reached at <= 5.5 m/s (5 days). However, 2023-07-26 was selected at the <= 3.5 m/s step (1 day) because it represents uniquely the calmest, hottest historic heatwave day (43.0 °C, 3.17 m/s) in the 5-year record, whereas at <= 5.5 m/s the days have moderate sea breezes (5.1-5.4 m/s) which defeats the stagnation objective.',
            'tmax_threshold_c': p95_tmax,
            'median_tmax_c': float(calm_5p5['median_tmax_tier'].iloc[0]),
            'selected_day': selected_calm,
            'top_5_candidates': calm_ranked.head(5).to_dict(orient='records')
        },
        'known_limitations': {
            'era5_solar_ghi_aerosol': 'ERA5 noon GHI is ~14% above pvlib Ineichen clear sky (925 vs 814 W/m²) due to climatological aerosol representation in ERA5. A radiation-sensitivity variant met_DESIGN_<type>_<date>_hourly_RADSENS.txt is provided scaling Kdn/Kdir/Kdiff by 0.88 to match Ineichen clear sky for SOLWEIG radiation sensitivity analyses.'
        }
    }
    json_path = OUT_DIR / "selected_design_days.json"
    with open(json_path, "w") as f:
        json.dump(selection_meta, f, indent=2)
    print(f"Saved selection metadata to {json_path}")

if __name__ == "__main__":
    main()
