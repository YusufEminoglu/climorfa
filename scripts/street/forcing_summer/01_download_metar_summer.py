#!/usr/bin/env python3
"""
Download IEM ASOS/METAR data for LTBJ (Izmir Adnan Menderes Airport)
for June-September (JJAS) across 2021-2025.
Part of WP1.4 (Summer design days + SOLWEIG forcing).
"""
import io
import os
import sys
import urllib.parse
import urllib.request
import pandas as pd
from pathlib import Path

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
STATION = "LTBJ"
YEARS = [2021, 2022, 2023, 2024, 2025]
PARAMS_DATA = [
    "tmpf", "dwpf", "relh", "drct", "sknt",
    "skyc1", "skyc2", "skyc3", "skyl1",
    "mslp", "alti", "wxcodes", "metar"
]

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "street" / "forcing_summer"

def download_data(station: str, year: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{station}_{year}_JJAS.csv"

    # Query from May 31 00:00 UTC to Oct 1 23:59 UTC to ensure complete coverage of Jun 1 - Sep 30 in UTC+3
    query_params = [
        ("station", station),
        ("year1", str(year)),
        ("month1", "5"),
        ("day1", "31"),
        ("year2", str(year)),
        ("month2", "10"),
        ("day2", "1"),
        ("hour2", "23"),
        ("minute2", "59"),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "yes"),
        ("report_type", "1"),
        ("report_type", "2"),
    ]
    for d in PARAMS_DATA:
        query_params.append(("data", d))

    url = f"{BASE_URL}?{urllib.parse.urlencode(query_params)}"
    print(f"Fetching {station} for {year} (JJAS) from IEM ASOS...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CLIMORFA-WP1.4/1.0"}
    )
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    with open(out_path, "wb") as f:
        f.write(content)

    df = pd.read_csv(io.BytesIO(content), low_memory=False)
    print(f"Station {station} ({year} JJAS): retrieved {len(df)} rows -> saved to {out_path}")
    return out_path

def main():
    print("=== WP1.4 METAR ACQUISITION FOR JJAS 2021-2025 (LTBJ) ===")
    dfs = []
    for yr in YEARS:
        out_file = download_data(STATION, yr)
        df = pd.read_csv(out_file, low_memory=False)
        dfs.append(df)
    comb_df = pd.concat(dfs, ignore_index=True)
    comb_path = OUT_DIR / f"{STATION}_summer_2021_2025.csv"
    comb_df.to_csv(comb_path, index=False)
    print(f"\nCombined {STATION} JJAS 2021-2025: {len(comb_df)} rows saved to {comb_path}")

if __name__ == "__main__":
    main()
