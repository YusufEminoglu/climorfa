#!/usr/bin/env python3
"""
Validation script for WP0.5a November campaign forcing outputs.
Checks 24-column UMEP met text files against solweig_gpu loader specification,
verifies 10-minute CSV files, and audits value ranges.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
FORCING_DIR = BASE_DIR / "data" / "street" / "wsl" / "forcing_nov"

EXPECTED_COLUMNS = [
    'iy', 'id', 'it', 'imin', 'Q*', 'QH', 'QE', 'Qs', 'Qf',
    'Wind', 'RH', 'Td', 'press', 'rain', 'Kdn', 'snow',
    'ldown', 'fcld', 'wuh', 'xsmd', 'lai_hr', 'Kdiff', 'Kdir', 'Wd'
]

FILES_TO_CHECK = [
    # Primary consecutive pair
    ("met_ANALOGUE_2025-11-02_for_2026-11-12_hourly.txt", "met_ANALOGUE_2025-11-02_for_2026-11-12_10min.csv", 2026, 316),
    ("met_ANALOGUE_2025-11-03_for_2026-11-13_hourly.txt", "met_ANALOGUE_2025-11-03_for_2026-11-13_10min.csv", 2026, 317),
    # Top individual pair
    ("met_ANALOGUE_2024-11-12_for_2026-11-12_hourly.txt", "met_ANALOGUE_2024-11-12_for_2026-11-12_10min.csv", 2026, 316),
    ("met_ANALOGUE_2023-11-14_for_2026-11-13_hourly.txt", "met_ANALOGUE_2023-11-14_for_2026-11-13_10min.csv", 2026, 317),
]

def validate_all():
    print("=== WP0.5a FORCING VALIDATION REPORT ===\n")
    all_ok = True

    for hourly_name, csv_name, expected_year, expected_doy in FILES_TO_CHECK:
        print(f"--- Checking: {hourly_name} ---")
        hourly_path = FORCING_DIR / hourly_name
        if not hourly_path.exists():
            print(f"[FAIL] Missing file: {hourly_path}")
            all_ok = False
            continue

        # 1. Check header
        with open(hourly_path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
        headers = header_line.split()
        if headers != EXPECTED_COLUMNS:
            print(f"[FAIL] Header mismatch!\nExpected: {EXPECTED_COLUMNS}\nGot:      {headers}")
            all_ok = False
        else:
            print(f"[PASS] 24 header columns match standard UMEP specification.")

        # 2. Check solweig_gpu loadtxt reader
        try:
            arr = np.loadtxt(hourly_path, skiprows=1, delimiter=' ')
        except Exception as e:
            print(f"[FAIL] np.loadtxt failed: {e}")
            all_ok = False
            continue

        if arr.shape != (24, 24):
            print(f"[FAIL] Incorrect shape: {arr.shape}, expected (24, 24)")
            all_ok = False
        else:
            print(f"[PASS] Loader shape verified: (24 rows, 24 columns).")

        # 3. Check values
        iy = arr[:, 0].astype(int)
        doy = arr[:, 1].astype(int)
        it = arr[:, 2].astype(int)
        imin = arr[:, 3].astype(int)
        wind = arr[:, 9]
        rh = arr[:, 10]
        td = arr[:, 11]
        press = arr[:, 12]
        rain = arr[:, 13]
        kdn = arr[:, 14]
        ldown = arr[:, 16]
        kdiff = arr[:, 21]
        kdir = arr[:, 22]
        wd = arr[:, 23]

        assert (iy == expected_year).all(), f"Year mismatch: expected {expected_year}"
        assert (doy == expected_doy).all(), f"DOY mismatch: expected {expected_doy}"
        assert (it == np.arange(24)).all(), f"Hours sequence mismatch"
        assert (imin == 0).all(), f"Minutes must be 0"
        assert (rain == 0.0).all(), f"Rain must be 0 for clear day"
        assert (wind >= 0.0).all() and (wind <= 10.0).all(), f"Wind out of bounds: {wind.min()}..{wind.max()}"
        assert (rh >= 0.0).all() and (rh <= 100.0).all(), f"RH out of bounds: {rh.min()}..{rh.max()}"
        assert (td >= -10.0).all() and (td <= 45.0).all(), f"Td out of bounds: {td.min()}..{td.max()}"
        assert (press >= 95.0).all() and (press <= 105.0).all(), f"Pressure out of bounds (kPa): {press.min()}..{press.max()}"
        assert (kdn >= 0.0).all(), f"Kdn negative: {kdn.min()}"
        assert (ldown >= 200.0).all() and (ldown <= 450.0).all(), f"Ldown out of bounds: {ldown.min()}..{ldown.max()}"
        assert (kdiff >= 0.0).all(), f"Kdiff negative: {kdiff.min()}"
        assert (kdir >= 0.0).all(), f"Kdir negative: {kdir.min()}"
        assert (wd >= 0.0).all() and (wd <= 360.0).all(), f"Wd out of bounds: {wd.min()}..{wd.max()}"

        print(f"[PASS] Meteorological ranges: Ta [{td.min():.1f}, {td.max():.1f} °C], RH [{rh.min():.1f}, {rh.max():.1f} %], Wind [{wind.min():.1f}, {wind.max():.1f} m/s], Press [{press.min():.1f}, {press.max():.1f} kPa]")
        print(f"[PASS] Radiation ranges: Kdn (GHI) max = {kdn.max():.1f} W/m², Kdir (DNI) max = {kdir.max():.1f} W/m², Ldown [{ldown.min():.1f}, {ldown.max():.1f} W/m²]")

        # 4. Check 10-minute CSV
        csv_path = FORCING_DIR / csv_name
        if not csv_path.exists():
            print(f"[FAIL] Missing 10-min CSV: {csv_path}")
            all_ok = False
            continue

        df_10m = pd.read_csv(csv_path)
        if len(df_10m) != 144:
            print(f"[FAIL] 10-min CSV has {len(df_10m)} rows, expected 144")
            all_ok = False
        else:
            print(f"[PASS] 10-min CSV verified: 144 time steps (00:00 to 23:50).")
        assert not df_10m.isnull().any().any(), "10-min CSV contains nulls"
        print(f"[PASS] No null/NaN values in 10-min CSV.\n")

    # Check quicklook figures
    png_path = FORCING_DIR / "quicklook_forcing_nov.png"
    if png_path.exists() and png_path.stat().st_size > 10000:
        print(f"[PASS] Primary quicklook PNG exists and valid ({png_path.stat().st_size / 1024:.1f} KB): {png_path}")
    else:
        print(f"[FAIL] Quicklook PNG missing or corrupted: {png_path}")
        all_ok = False

    png_top_ind = FORCING_DIR / "quicklook_forcing_nov_top_individual.png"
    if png_top_ind.exists():
        print(f"[PASS] Individual pair quicklook PNG exists ({png_top_ind.stat().st_size / 1024:.1f} KB): {png_top_ind}")

    print("\n" + "="*50)
    if all_ok:
        print("ALL ACCEPTANCE CHECKS PASSED SUCCESSFULLY (EXIT 0)")
        return 0
    else:
        print("VALIDATION FAILED (EXIT 1)")
        return 1

if __name__ == "__main__":
    sys.exit(validate_all())
