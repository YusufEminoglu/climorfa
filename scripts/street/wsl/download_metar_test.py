#!/usr/bin/env python3
"""
METAR access test: download Iowa Environmental Mesonet ASOS data
for stations LTBJ and LTBL for 2025-07-01 to 2025-07-03.
Work Package: WP0.2
"""
import os
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
STATIONS = ["LTBJ", "LTBL"]
PARAMS_DATA = ["tmpf", "dwpf", "drct", "sknt", "relh", "skyc1"]
START_DATE = (2025, 7, 1)
END_DATE = (2025, 7, 3)

OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "street" / "wsl" / "metar_test"

def download_station_data(station: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{station}.csv"

    # Query params:
    # Multiple 'data' params can be passed as a list of tuples to urllib.parse.urlencode
    query_params = [
        ("station", station),
        ("year1", str(START_DATE[0])),
        ("month1", str(START_DATE[1])),
        ("day1", str(START_DATE[2])),
        ("year2", str(END_DATE[0])),
        ("month2", str(END_DATE[1])),
        ("day2", str(END_DATE[2])),
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
    print(f"Fetching {station} from IEM ASOS...")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CLIMORFA-METAR-Test/1.0"}
    )
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())

    # Count rows
    with open(out_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0] if lines else ""
    row_count = max(0, len(lines) - 1)
    print(f"Station {station}: saved {len(lines)} lines ({row_count} data rows) to {out_path}")
    print(f"Header: {header}")
    if row_count > 0:
        print(f"First row: {lines[1]}")
        print(f"Last row:  {lines[-1]}")
    print()
    return out_path

def main():
    print("=== IEM ASOS METAR DOWNLOAD TEST ===")
    for st in STATIONS:
        download_station_data(st)

if __name__ == "__main__":
    main()
