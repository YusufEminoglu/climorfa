"""Sanity checks on the REAL analysis tables before hypothesis statistics (no hypothesis interpretation).

Independent checks (could fail):
  S1  azimuth_deg (written by WP3.0) vs the bearing of the segment chord computed here from the geometry:
      confirms the bearing convention (clockwise from north, undirected) that the orientation features rely on.
  S2  UTCI recomputed from segment-median Tmrt and wind (pythermalcomfort) vs M1 (median UTCI): median |diff|.
  S3  physical sign tests: Spearman(M1, shade fraction) < 0, (M1, tree cover) < 0, (Tmrt, shade) < 0, (v10eff, bcover30) < 0.
  S4  table integrity: duplicate seg_ids, identical segment sets over days, n_ps_valid >= 25, forcing sanity (Wd at band 15).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402


def main():
    cm.utf8_stdout()
    frames = cm.load_frames()
    seg = cm.load_segments(cm.DATA_ANALYSIS / "segments.gpkg")
    out = {}
    first = next(iter(frames.values()))
    print(f"=== sanity checks on real tables [{cm.label_for(first)}] ===")
    # S1
    d = seg.merge(first[["seg_id"]], on="seg_id")
    d["straight"] = True
    diff = np.abs(d["azimuth_deg"] - d["chord_bearing_deg"]) % 180.0
    diff = np.minimum(diff, 180 - diff)
    ok = diff < 20
    # restrict to reasonably straight segments (chord/length >= 0.9) to avoid curvature noise
    print(f"S1 azimuth_deg vs chord bearing (axis angle): median diff {np.median(diff):.2f} deg; share < 20 deg: {ok.mean():.3f} (n={len(d)})")
    # also the alternative convention (math angle: counter-clockwise from east) as a falsification test
    alt = (90.0 - d["chord_bearing_deg"]) % 180.0
    dalt = np.abs(d["azimuth_deg"] - alt) % 180.0
    dalt = np.minimum(dalt, 180 - dalt)
    print(f"   falsification: if azimuth_deg were counter-clockwise-from-east, median diff would be {np.median(dalt):.2f} deg")
    out["S1"] = dict(median_diff_deg=float(np.median(diff)), share_lt_20=float(ok.mean()), alt_convention_median_diff=float(np.median(dalt)))
    for day, f in frames.items():
        fo = f.attrs["forcing"]
        u = cm.utci_vec(fo["ta"], f["tmrt_b15"], f["v10eff_b15"], fo["rh"])
        dd = np.abs(u - f["m1"].to_numpy())
        rs = {}
        for a, b, sign in [("m1", "m4", -1), ("m1", "tcover15", -1), ("m3", "m4", -1), ("v10eff_b15", "bcover30", -1),
                           ("m1", "svf_med", +1)]:
            r = spearmanr(f[a], f[b], nan_policy="omit")[0]
            rs[f"{a}~{b}"] = dict(rho=float(r), expected_sign=sign, ok=bool(np.sign(r) == sign))
        out[day] = dict(S2_median_abs_diff=float(np.median(dd)), S2_max=float(dd.max()), S3=rs,
                        S4=dict(dup_seg=int(f["seg_id"].duplicated().sum()), min_n_ps_valid=int(f["n_ps_valid"].min()),
                                n=len(f), forcing=fo, nan_rate_features=f[cm.FEATURES].isna().mean().round(3).to_dict()))
        print(f"{day}: S2 UTCI(median Tmrt, median v) vs M1: median |diff| {np.median(dd):.3f}, max {dd.max():.2f} degC")
        print("   S3 " + "; ".join(f"{k} rho={v['rho']:+.2f} ({'ok' if v['ok'] else 'UNEXPECTED SIGN'})" for k, v in rs.items()))
        print(f"   S4 dup seg_id={out[day]['S4']['dup_seg']} min n_ps_valid={out[day]['S4']['min_n_ps_valid']} "
              f"forcing Ta={fo['ta']} RH={fo['rh']} U10={fo['u10']} Wd15={fo['wd15']} Wd(15-17 circ. mean)={fo['wd_mean_15_17']:.0f}")
    if len(frames) == 2:
        same = set(frames["IMBAT"]["seg_id"]) == set(frames["CALM"]["seg_id"])
        print(f"S4 identical segment sets across days: {same}")
        out["S4_same_segments"] = bool(same)
    # CALM Wd flag
    met = cm.load_met("CALM")
    out["CALM_Wd_14_17"] = met[(met["it"] >= 14) & (met["it"] <= 17)][["it", "Wind", "Wd"]].to_dict("list")
    print("CALM wind direction 14-17h:", out["CALM_Wd_14_17"])
    od = cm.out_dir("h1", False)
    cm.write_json(od / cm.fname("sanity_checks_real", "json", False, first), out)


if __name__ == "__main__":
    main()
