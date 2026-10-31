"""WP2.3b: VRB flags (rule A7) + corrected-direction met files (v2). Original met files are never modified.

Outputs (new files only):
  data/street/forcing_summer/vrb_flags.csv                     day, it, vrb, n_reports, n_vrb  (both days, it 0..23)
  data/street/forcing_summer/met_DESIGN_<DAY>_<date>_hourly_v2.txt   only for days that have a VRB-flagged hour
Wd in v2: interpolated (builder method: speed-weighted u/v vector interpolation) from NON-VRB reports only; VRB-flagged
hours get -999. All other columns are byte-identical to the original (asserted).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
from scripts.street.utci.vrb_utils import (DAY_DATES, MET_ORIG, MET_V2, VRB_FLAGS_CSV, TIMEZONE, day_start,
                                           load_metar_local, vrb_flag_table)


def interp_wd(metar: pd.DataFrame, day: str, drop_vrb: bool) -> np.ndarray:
    """Builder method (04_build_forcing_and_compare.interpolate_metar) for Wd at the 24 hourly stamps.
    drop_vrb=False reproduces the builder exactly (missing drct -> 0 deg); drop_vrb=True skips VRB reports."""
    ref = day_start(day)
    sub = metar[(metar["valid_local"] >= ref - pd.Timedelta(hours=12)) & (metar["valid_local"] <= ref + pd.Timedelta(hours=36))]
    sub = sub.reset_index(drop=True)
    wind_ms = (sub["sknt_num"] * 0.514444).clip(lower=0.0)
    wd = sub["drct_num"].fillna(0.0)
    u = -wind_ms * np.sin(np.radians(wd))
    v = -wind_ms * np.cos(np.radians(wd))
    obs_sec = (sub["valid_local"] - ref).dt.total_seconds().values
    grid = pd.date_range(ref, ref + pd.Timedelta(hours=23), freq="1h")
    target_sec = (grid - grid[0].floor("D")).total_seconds().values
    keep = (~sub["vrb"]).to_numpy() if drop_vrb else np.ones(len(sub), bool)
    speed_i = np.interp(target_sec, obs_sec, wind_ms)           # scalar speed: all reports, as in the builder
    u_i = np.interp(target_sec, obs_sec[keep], u.to_numpy()[keep])
    v_i = np.interp(target_sec, obs_sec[keep], v.to_numpy()[keep])
    out = (np.degrees(np.arctan2(-u_i, -v_i))) % 360.0
    out[speed_i < 0.2] = 0.0
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    metar = load_metar_local()
    n_both = int((metar["vrb_text"] & metar["vrb_drct"]).sum())
    print(f"METAR rows {len(metar)}; VRB by text {int(metar['vrb_text'].sum())}, by non-numeric drct {int(metar['vrb_drct'].sum())}, both {n_both}")
    flags = vrb_flag_table(metar)
    flags.to_csv(VRB_FLAGS_CSV, index=False)
    print(f"wrote {VRB_FLAGS_CSV}")
    for day in DAY_DATES:
        f = flags[flags.day == day]
        print(day, "VRB-flagged it:", f[f.vrb]["it"].tolist(), "| n_reports 6..21:", f[(f.it >= 6) & (f.it <= 21)]["n_reports"].tolist())

    for day in DAY_DATES:
        orig_bytes = MET_ORIG[day].read_bytes()
        lines = orig_bytes.split(b"\n")
        # 0. fidelity of my re-implementation: drop_vrb=False must reproduce the original Wd column exactly
        wd_orig_txt = [ln.rstrip(b"\r").split(b" ")[-1].decode() for ln in lines[1:25]]
        wd_rebuilt = [f"{x:.1f}" for x in interp_wd(metar, day, drop_vrb=False)]
        assert wd_rebuilt == wd_orig_txt, (day, wd_rebuilt, wd_orig_txt)
        print(f"{day}: re-implementation reproduces the original Wd of all 24 hours exactly (drop_vrb=False)")

        f = flags[flags.day == day].set_index("it")
        if not f["vrb"].any():
            print(f"{day}: no VRB-flagged hour -> no v2 file written")
            continue
        wd_new = interp_wd(metar, day, drop_vrb=True)
        out_lines = [lines[0]]
        changed_unflagged = []
        for it in range(24):
            ln = lines[1 + it]
            cr = b"\r" if ln.endswith(b"\r") else b""
            body = ln.rstrip(b"\r")
            head, _, _old = body.rpartition(b" ")
            assert int(head.split(b" ")[2]) == it
            new_tok = b"-999" if bool(f.loc[it, "vrb"]) else f"{wd_new[it]:.1f}".encode()
            if not bool(f.loc[it, "vrb"]) and new_tok != _old:
                changed_unflagged.append((it, _old.decode(), new_tok.decode()))
            out_lines.append(head + b" " + new_tok + cr)
        out_lines += lines[25:]                               # trailing empty element (final newline)
        new_bytes = b"\n".join(out_lines)
        # every column except Wd byte-identical
        a = [ln.rstrip(b"\r").rpartition(b" ")[0] for ln in orig_bytes.split(b"\n") if ln.strip()]
        b = [ln.rstrip(b"\r").rpartition(b" ")[0] for ln in new_bytes.split(b"\n") if ln.strip()]
        assert a == b and len(a) == 25, "non-Wd columns differ"
        assert new_bytes.count(b"\n") == orig_bytes.count(b"\n") and new_bytes.count(b"\r\n") == orig_bytes.count(b"\r\n")
        assert not MET_V2[day].exists(), "refusing to overwrite an existing v2 file"
        MET_V2[day].write_bytes(new_bytes)
        print(f"{day}: wrote {MET_V2[day].name}; all non-Wd columns byte-identical (asserted for 24 rows + header)")
        print(f"{day}: Wd changed at UNFLAGGED hours (it, old, new): {changed_unflagged}")
        o = pd.read_csv(MET_ORIG[day], sep=r"\s+")
        n = pd.read_csv(MET_V2[day], sep=r"\s+")
        print(pd.DataFrame({"it": o["it"], "Wind": o["Wind"], "Wd_orig": o["Wd"], "Wd_v2": n["Wd"],
                            "vrb": [bool(f.loc[i, 'vrb']) for i in o["it"]]}).iloc[5:22].to_string(index=False))


if __name__ == "__main__":
    main()
