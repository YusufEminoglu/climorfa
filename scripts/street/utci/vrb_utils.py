"""WP2.3b shared helpers for VRB (variable direction) METAR winds. Pre-analysis lock amendment A7 item 3.

Rule A7: an hour `it` (met-file row, local time UTC+3) is VRB-flagged if any METAR report with local time in
(it-1:00, it:00] is VRB (variable direction). For VRB hours the street-wind ratio is the mean of the 16 sector
ratios and Wd is not used; U10 is unchanged.

METAR vs the builder (scripts/street/forcing_summer/04_build_forcing_and_compare.py): the builder read the missing
`drct` of a VRB report as 0 deg (north). Here the VRB reports are identified from the raw METAR text
(wind group VRBddKT in the report body, i.e. before RMK) and from drct being non-numeric ('M').
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
FORC = ROOT / "data/street/forcing_summer"
METAR_CSV = FORC / "LTBJ_summer_2021_2025.csv"
TIMEZONE = "Europe/Istanbul"  # UTC+3 year-round (same as the builder)
DAY_DATES = {"IMBAT": "2022-07-05", "CALM": "2023-07-26"}
MET_ORIG = {d: FORC / f"met_DESIGN_{d}_{dt}_hourly.txt" for d, dt in DAY_DATES.items()}
MET_V2 = {d: FORC / f"met_DESIGN_{d}_{dt}_hourly_v2.txt" for d, dt in DAY_DATES.items()}
VRB_FLAGS_CSV = FORC / "vrb_flags.csv"

# OBSERVED wind group only: the first group after the time stamp (LTBJ ddhhmmZ [AUTO|COR] VRBddKT ...). A VRBddKT inside a
# BECMG/TEMPO trend group (e.g. "BECMG VRB02KT") is a forecast and must not flag the observation.
_VRB_RE = re.compile(r"^\S+\s+\d{6}Z\s+(?:(?:AUTO|COR)\s+)?VRB\d{2,3}(?:G\d{2,3})?KT\b")


def load_metar_local() -> pd.DataFrame:
    """All METAR reports with local time, numeric direction/speed and the VRB flag. Duplicated local times dropped
    (first kept), exactly as in the builder."""
    df = pd.read_csv(METAR_CSV, low_memory=False)
    df["valid_local"] = pd.to_datetime(df["valid"], utc=True).dt.tz_convert(TIMEZONE)
    df = df.sort_values("valid_local").drop_duplicates(subset="valid_local").reset_index(drop=True)
    body = df["metar"].fillna("").astype(str).str.split(" RMK ", n=1, regex=False).str[0]
    df["vrb_text"] = body.apply(lambda s: bool(_VRB_RE.search(s)))
    df["drct_num"] = pd.to_numeric(df["drct"], errors="coerce")
    df["vrb_drct"] = df["drct_num"].isna()
    df["vrb"] = df["vrb_text"] | df["vrb_drct"]
    df["sknt_num"] = pd.to_numeric(df["sknt"], errors="coerce")
    return df


def day_start(day: str) -> pd.Timestamp:
    return pd.Timestamp(DAY_DATES[day], tz=TIMEZONE)


def vrb_flag_table(metar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rule A7 flags for it = 0..23 of both days: columns day, it, vrb, n_reports, n_vrb."""
    metar = load_metar_local() if metar is None else metar
    rows = []
    for day in DAY_DATES:
        t0 = day_start(day)
        for it in range(24):
            lo, hi = t0 + pd.Timedelta(hours=it - 1), t0 + pd.Timedelta(hours=it)
            w = metar[(metar["valid_local"] > lo) & (metar["valid_local"] <= hi)]
            rows.append(dict(day=day, it=it, vrb=bool(w["vrb"].any()), n_reports=int(len(w)), n_vrb=int(w["vrb"].sum())))
    return pd.DataFrame(rows)


def read_vrb_flags(day: str) -> dict[int, bool]:
    """{it: bool} from data/street/forcing_summer/vrb_flags.csv; empty dict if the file is absent."""
    if not VRB_FLAGS_CSV.exists():
        return {}
    f = pd.read_csv(VRB_FLAGS_CSV)
    f = f[f["day"] == day]
    return {int(r.it): bool(r.vrb) for r in f.itertuples()}


def nonvrb_circular_mean(day: str, h_start: float = 14.0, h_end: float = 17.0, metar: pd.DataFrame | None = None) -> dict:
    """Circular mean of the direction of NON-VRB METAR reports with local time in (h_start:00, h_end:00]
    (hour-ending bands 15..17 = 14:00-17:00). Unit vectors, unweighted (same as the earlier mean over hourly Wd).
    Returns wd_mean_deg, R (resultant length of the unit vectors, 0..1), n, n_vrb_excluded, times."""
    metar = load_metar_local() if metar is None else metar
    t0 = day_start(day)
    lo, hi = t0 + pd.Timedelta(hours=h_start), t0 + pd.Timedelta(hours=h_end)
    w = metar[(metar["valid_local"] > lo) & (metar["valid_local"] <= hi)]
    ok = w[~w["vrb"] & w["drct_num"].notna()]
    ang = np.deg2rad(ok["drct_num"].to_numpy(float))
    n = int(len(ok))
    if n == 0:
        return dict(wd_mean_deg=float("nan"), R=float("nan"), n=0, n_vrb_excluded=int(w["vrb"].sum()), times=[])
    s, c = np.sin(ang).mean(), np.cos(ang).mean()
    return dict(wd_mean_deg=float(np.rad2deg(math.atan2(s, c)) % 360.0), R=float(math.hypot(s, c)), n=n,
                n_vrb_excluded=int(w["vrb"].sum()),
                times=[t.strftime("%H:%M") for t in ok["valid_local"]])
