#!/usr/bin/env python3
"""
Select analogue clear-sky November days for Izmir campaign forcing.
Work Package: WP0.5a.

LOCKED RULE:
1. Daytime window: 07:00-17:00 local time (Europe/Istanbul, UTC+3).
2. Cloud criteria: Every report between 07:00 and 17:00 local has skyc1 in
   {CLR, NSC, SKC, FEW, CAVOK} or no cloud group (and no BKN/OVC aloft).
3. Precipitation: No precipitation codes in present weather (wxcodes) anytime during the day.
4. Wind criteria: 10-m wind speed between 10:00 and 16:00 local <= 6.0 m/s.
5. Ranking: Among qualifying days, rank by daily Tmax closeness to the 2023-2025
   November clear-day median Tmax. Pick the top 2 consecutive days if possible, else the top 2.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "street" / "wsl"
METAR_CSV = DATA_DIR / "metar_nov" / "LTBJ_nov_2023_2025.csv"
OUT_DIR = DATA_DIR / "forcing_nov"

PRECIP_KEYWORDS = ['RA', 'DZ', 'SN', 'PL', 'GR', 'GS', 'TS', 'SH']
CLEAR_SKY_CODES = {'CLR', 'NSC', 'SKC', 'FEW', 'CAVOK', 'M'}

def evaluate_days():
    df = pd.read_csv(METAR_CSV, low_memory=False)
    df['valid_utc'] = pd.to_datetime(df['valid'], utc=True)
    df['valid_local'] = df['valid_utc'].dt.tz_convert('Europe/Istanbul')
    df['local_date'] = df['valid_local'].dt.date
    df['local_hour'] = df['valid_local'].dt.hour + df['valid_local'].dt.minute / 60.0
    df['wind_ms'] = pd.to_numeric(df['sknt'], errors='coerce') * 0.514444
    df['ta_c'] = (pd.to_numeric(df['tmpf'], errors='coerce') - 32.0) * 5.0 / 9.0

    # November only
    df = df[df['valid_local'].dt.month == 11]

    results = []
    for date, g in df.groupby('local_date'):
        daytime = g[(g['local_hour'] >= 7.0) & (g['local_hour'] <= 17.0)]
        wind_window = g[(g['local_hour'] >= 10.0) & (g['local_hour'] <= 16.0)]

        # 1. Cloud check
        skyc1_ok = daytime['skyc1'].isin(CLEAR_SKY_CODES).all()
        has_bkn_ovc = daytime['skyc2'].isin(['BKN', 'OVC']).any() or daytime['skyc3'].isin(['BKN', 'OVC']).any()

        # 2. Precip check
        def has_precip(series):
            for val in series.dropna():
                if str(val).strip() != 'M':
                    for pk in PRECIP_KEYWORDS:
                        if pk in str(val):
                            return True
            return False

        has_precip_full = has_precip(g['wxcodes'])

        # 3. Wind check (10:00 - 16:00)
        max_wind_10_16 = wind_window['wind_ms'].max()
        wind_ok = bool(max_wind_10_16 <= 6.0)

        tmax = float(g['ta_c'].max())
        tmin = float(g['ta_c'].min())
        tmean = float(g['ta_c'].mean())

        qualifies = bool(skyc1_ok and (not has_bkn_ovc) and (not has_precip_full) and wind_ok)

        results.append({
            'date': str(date),
            'year': int(date.year),
            'qualifies': qualifies,
            'skyc1_ok': bool(skyc1_ok),
            'has_bkn_ovc': bool(has_bkn_ovc),
            'no_precip': bool(not has_precip_full),
            'max_wind_10_16_ms': round(float(max_wind_10_16), 2),
            'wind_ok': bool(wind_ok),
            'tmax_c': round(tmax, 2),
            'tmin_c': round(tmin, 2),
            'tmean_c': round(tmean, 2),
            'n_reports': int(len(g))
        })

    res_df = pd.DataFrame(results)
    qualifying = res_df[res_df['qualifies']].copy()

    # Median clear-day Tmax
    median_tmax = float(qualifying['tmax_c'].median())
    qualifying['diff_median_c'] = (qualifying['tmax_c'] - median_tmax).abs().round(2)
    qualifying = qualifying.sort_values(by=['diff_median_c', 'date']).reset_index(drop=True)

    # Find consecutive pairs
    consecutive_pairs = []
    qualifying_dates = pd.to_datetime(qualifying['date']).dt.date.tolist()
    date_to_row = {row['date']: row for _, row in qualifying.iterrows()}

    for i, d1 in enumerate(qualifying_dates):
        d1_str = str(d1)
        for j, d2 in enumerate(qualifying_dates):
            d2_str = str(d2)
            if (d2 - d1).days == 1:
                r1 = date_to_row[d1_str]
                r2 = date_to_row[d2_str]
                mean_diff = round((r1['diff_median_c'] + r2['diff_median_c']) / 2.0, 2)
                consecutive_pairs.append({
                    'day1': d1_str,
                    'day2': d2_str,
                    'day1_tmax': r1['tmax_c'],
                    'day2_tmax': r2['tmax_c'],
                    'day1_diff': r1['diff_median_c'],
                    'day2_diff': r2['diff_median_c'],
                    'mean_diff': mean_diff,
                    'day1_max_wind': r1['max_wind_10_16_ms'],
                    'day2_max_wind': r2['max_wind_10_16_ms']
                })

    consecutive_df = pd.DataFrame(consecutive_pairs)
    if not consecutive_df.empty:
        consecutive_df = consecutive_df.sort_values(by=['mean_diff']).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "candidate_days_summary.csv"
    res_df.to_csv(summary_path, index=False)

    qual_path = OUT_DIR / "qualifying_days_ranked.csv"
    qualifying.to_csv(qual_path, index=False)

    print("=== WP0.5a ANALOGUE DAY SELECTION RESULTS ===")
    print(f"Total November days evaluated: {len(res_df)}")
    print(f"Qualifying clear-sky days: {len(qualifying)}")
    print(f"Clear-day median Tmax: {median_tmax:.2f} °C\n")

    print("Qualifying days ranked by closeness to median Tmax:")
    print(qualifying[['date', 'tmax_c', 'diff_median_c', 'max_wind_10_16_ms']].to_string())

    print("\nConsecutive qualifying pairs ranked by mean distance to median Tmax:")
    if not consecutive_df.empty:
        print(consecutive_df.to_string())
    else:
        print("No consecutive pairs found.")

    # Save selection metadata
    selection_meta = {
        'total_evaluated_days': len(res_df),
        'qualifying_days_count': len(qualifying),
        'clear_day_median_tmax_c': median_tmax,
        'consecutive_pairs': consecutive_pairs,
        'selected_consecutive_pair': consecutive_pairs[0] if consecutive_pairs else None,
        'top_individual_days': qualifying.head(2).to_dict(orient='records')
    }
    with open(OUT_DIR / "selected_analogue_days.json", "w") as f:
        json.dump(selection_meta, f, indent=2)

    return qualifying, consecutive_df

if __name__ == "__main__":
    evaluate_days()
