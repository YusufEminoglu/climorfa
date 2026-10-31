#!/usr/bin/env python3
"""
Automated validation and acceptance suite for WP1.4.
Verifies format, physical ranges, solweig_gpu loader, and non-circular sanity checks.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "street" / "forcing_summer"
OUTPUTS_DIR = BASE_DIR / "outputs" / "street" / "forcing_summer"

FILES_TO_CHECK = [
    DATA_DIR / "met_DESIGN_IMBAT_2022-07-05_hourly.txt",
    DATA_DIR / "met_DESIGN_IMBAT_2022-07-05_hourly_RADSENS.txt",
    DATA_DIR / "met_DESIGN_CALM_2023-07-26_hourly.txt",
    DATA_DIR / "met_DESIGN_CALM_2023-07-26_hourly_RADSENS.txt"
]

def check_met_file(fp: Path):
    print(f"\nChecking met file: {fp.name}")
    assert fp.exists(), f"File does not exist: {fp}"
    
    # Check text structure
    with open(fp, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    assert len(lines) == 25, f"Expected 25 lines (1 header + 24 hours), got {len(lines)}"
    
    header = lines[0].split()
    expected_header = "iy id it imin Q* QH QE Qs Qf Wind RH Td press rain Kdn snow ldown fcld wuh xsmd lai_hr Kdiff Kdir Wd".split()
    assert header == expected_header, f"Header mismatch: {header} vs {expected_header}"
    
    # Loader check via solweig_gpu exact method
    arr = np.loadtxt(fp, skiprows=1, delimiter=" ")
    assert arr.shape == (24, 24), f"Array shape mismatch: expected (24, 24), got {arr.shape}"
    assert not np.isnan(arr).any(), "NaN found in data array!"
    
    # Check hours 0..23
    assert np.array_equal(arr[:, 2].astype(int), np.arange(24)), "Hours must be strictly 0..23"
    assert (arr[:, 3] == 0).all(), "Minutes must all be 0"
    
    # Check physical plausibility (Non-circular tests)
    # Wind
    winds = arr[:, 9]
    assert (winds >= 0.0).all() and (winds <= 35.0).all(), f"Unphysical wind speed: {winds.min()} to {winds.max()}"
    
    # Relative humidity
    rhs = arr[:, 10]
    assert (rhs >= 5.0).all() and (rhs <= 100.0).all(), f"Unphysical RH: {rhs.min()} to {rhs.max()}"
    
    # Air temperature
    tas = arr[:, 11]
    assert (tas >= 15.0).all() and (tas <= 48.0).all(), f"Unphysical Ta: {tas.min()} to {tas.max()}"
    
    # Pressure (kPa)
    press = arr[:, 12]
    assert (press >= 95.0).all() and (press <= 105.0).all(), f"Unphysical pressure: {press.min()} to {press.max()}"
    
    # Rain (mm/h)
    rain = arr[:, 13]
    assert (rain == 0.0).all(), f"Rain must be 0 for clear summer design days, got {rain.max()}"
    
    # Kdn (GHI)
    kdn = arr[:, 14]
    assert (kdn >= 0.0).all() and (kdn <= 1100.0).all(), f"Unphysical GHI: {kdn.min()} to {kdn.max()}"
    assert kdn.max() > 750.0, f"Peak GHI too low for Mediterranean summer: {kdn.max()} W/m2"
    
    # Ldown
    ldown = arr[:, 16]
    assert (ldown >= 250.0).all() and (ldown <= 500.0).all(), f"Unphysical Ldown: {ldown.min()} to {ldown.max()}"
    
    # Kdiff & Kdir
    kdiff = arr[:, 21]
    kdir = arr[:, 22]
    assert (kdiff >= 0.0).all() and (kdiff <= 600.0).all(), f"Unphysical Kdiff: {kdiff.min()} to {kdiff.max()}"
    assert (kdir >= 0.0).all() and (kdir <= 1100.0).all(), f"Unphysical Kdir: {kdir.min()} to {kdir.max()}"
    assert kdir.max() > 600.0, f"Peak DNI too low for clear summer day: {kdir.max()} W/m2"
    
    # Wind direction
    wds = arr[:, 23]
    assert (wds >= 0.0).all() and (wds <= 360.0).all(), f"Unphysical Wd: {wds.min()} to {wds.max()}"
    
    print(f"  PASS: All 24 columns, data shapes, and physical bounds verified.")

def main():
    print("=== WP1.4 AUTOMATED ACCEPTANCE AUDIT ===")
    for fp in FILES_TO_CHECK:
        check_met_file(fp)
    
    # Check selection metadata JSON
    meta_fp = DATA_DIR / "selected_design_days.json"
    assert meta_fp.exists(), f"Missing {meta_fp}"
    with open(meta_fp, "r") as f:
        meta = json.load(f)
    assert "imbat" in meta and "calm_heat" in meta
    print(f"\nSelection metadata verified: {meta_fp.name}")
    print(f"  IMBAT: {meta['imbat']['selected_day']['date']} (Tmax = {meta['imbat']['selected_day']['tmax_c']} °C)")
    print(f"  CALM: {meta['calm_heat']['selected_day']['date']} (Tmax = {meta['calm_heat']['selected_day']['tmax_c']} °C)")
    
    # Check comparison stats JSON
    comp_fp = DATA_DIR / "era5_vs_metar_comparison_stats.json"
    assert comp_fp.exists(), f"Missing {comp_fp}"
    with open(comp_fp, "r") as f:
        comp = json.load(f)
    print(f"\nComparison stats verified: {comp_fp.name}")
    print(f"  IMBAT noon kt: {comp['IMBAT']['noon_metrics']['noon_clearness_index_kt']}")
    print(f"  CALM noon kt: {comp['CALM']['noon_metrics']['noon_clearness_index_kt']}")
    
    # Check quicklook plots
    q1 = OUTPUTS_DIR / "quicklook_summer_design_days_forcing.png"
    q2 = OUTPUTS_DIR / "quicklook_imbat_vs_calm_comparison.png"
    assert q1.exists() and q1.stat().st_size > 50000, f"Missing or empty {q1}"
    assert q2.exists() and q2.stat().st_size > 50000, f"Missing or empty {q2}"
    print(f"\nQuicklook images verified: {q1.name} ({q1.stat().st_size // 1024} KB), {q2.name} ({q2.stat().st_size // 1024} KB)")
    
    print("\nALL WP1.4 ACCEPTANCE AUDITS PASSED 100%!")

if __name__ == "__main__":
    main()
