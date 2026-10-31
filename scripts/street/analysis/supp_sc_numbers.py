"""Numbers for Supplementary Section S-C (street wind ratios and UTCI sensitivity to wind).

Read-only on the final tables; writes one JSON next to the other analysis outputs.
Run from the project root:
    C:\\Users\\YE\\miniforge3\\envs\\climorfa-street\\python.exe scripts/street/analysis/supp_sc_numbers.py
Output: outputs/street/analysis/sc/sc_numbers.json and a printed summary.

Classes follow scripts/street/stats: orientation = common.axis_angle / orientation_class against the day's wind reference,
H/W = terciles of non-NaN hw_ratio pooled over both days (hw_tercile_edges in h4_results.json), NaN = no enclosing
buildings; SVF quintiles over the pooled segment set, as in build_fig3_forcing_model.py.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pythermalcomfort.models import utci

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "street" / "stats"))
from common import axis_angle, orientation_class  # noqa: E402

DAYS = ["IMBAT", "CALM"]
FINAL = ROOT / "data/street/analysis/final_20261030"
H4 = json.loads((ROOT / "outputs/street/analysis/h4/results/h4_results.json").read_text())
EDGES = H4["hw_tercile_edges"]
OUT = ROOT / "outputs/street/analysis/sc"


def U(tdb, tr, v, rh):
    return np.asarray(utci(tdb=tdb, tr=tr, v=v, rh=rh, limit_inputs=False, round_output=False).utci, float)


def q(s, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = pd.Series(s).dropna()
    return {f"p{int(p * 100):02d}": float(s.quantile(p)) for p in ps}


def load():
    seg = gpd.read_file(FINAL / "segments.gpkg", ignore_geometry=True,
                        columns=["seg_id", "azimuth_deg", "hw_ratio"])
    frames = {}
    for day in DAYS:
        d = pd.read_parquet(FINAL / f"seg_metrics_{day}.parquet").merge(seg, on="seg_id", how="left", validate="1:1")
        wd = H4["results"][day]["forcing"]["wd_mean_15_17"]
        d["ang_wind"] = axis_angle(d["azimuth_deg"].to_numpy(float), wd)
        d["orient"] = orientation_class(d["ang_wind"])
        hw = d["hw_ratio"]
        d["hwc"] = pd.cut(hw, [-np.inf, EDGES[0], EDGES[1], np.inf], labels=["low", "mid", "high"]).astype(object)
        d.loc[hw.isna(), "hwc"] = "none"
        d["dM"] = d["utci_b15"] - d["utci_unif_b15"]
        frames[day] = d
    return frames


def cell(x):
    return f"{x['p50']:.2f} [{x['p25']:.2f}, {x['p75']:.2f}]"


def sgn(x, nd=2):
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    return f"{x:+.{nd}f}".replace("-", "$-$").replace("+", "$+$")


def pct(x):
    return sgn(x, 0 if abs(x) >= 2 else 1) + r"\%"


def spread(x):
    return f"{sgn(x['dM']['p50'])} [{sgn(x['dM']['p05'])}, {sgn(x['dM']['p95'])}]"


HEAD = "% Written by scripts/street/analysis/supp_sc_numbers.py; src: outputs/street/analysis/sc/sc_numbers.json"


def write_tables(out):
    """LaTeX tables for S-C, written from the JSON so that no number is typed by hand."""
    td = ROOT / "paper/manuscript_v2/src/tables/sc"
    td.mkdir(parents=True, exist_ok=True)
    i, c = out["days"]["IMBAT"], out["days"]["CALM"]

    names = [("low", r"Low ($\leq$0.354)"), ("mid", "Mid (0.354--0.633)"), ("high", r"High ($>$0.633)"),
             ("none", "No enclosing buildings")]
    L = [HEAD, r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Street-wind ratio $r$ at label 15 by $H/W$ class and orientation: median [25th, 75th percentile] over "
         r"segments of the segment-median ratio. Orientation is the angle between the street axis and the day's wind "
         r"reference: aligned $\leq$22.5\textdegree{}, perpendicular $\geq$67.5\textdegree{}, oblique between. The last column "
         r"compares the aligned and perpendicular medians. On CALM label 15 is a VRB hour, so $r$ is the mean of the 16 sector "
         r"ratios and carries no direction; the reference (193\textdegree{}) rests on $n=4$ non-VRB reports, $R=0.87$. "
         r"Modelled values, not measurements.}",
         r"\label{tab:sc_ratio}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"$H/W$ class & Aligned & Oblique & Perpendicular & Aligned vs perp. \\", r"\midrule"]
    for lab, d, ref in [("IMBAT", i, "335"), ("CALM", c, "193")]:
        L.append(r"\multicolumn{5}{@{}l}{\emph{" + lab + ", wind reference " + ref + r"\textdegree{}, $U_{10}=" + f"{d['u10']:.1f}"
                 + r"$~m\,s$^{-1}$}} \\")
        t = d["r_by_hw_orient"]
        for k, lb in names:
            L.append(f"{lb} & {cell(t[k + '|aligned'])} & {cell(t[k + '|oblique'])} & {cell(t[k + '|perpendicular'])} & "
                     f"{pct(d['aligned_vs_perp'][k]['pct'])} " + r"\\")
        t = d["r_by_orient"]
        L.append(f"All segments & {cell(t['aligned'])} & {cell(t['oblique'])} & {cell(t['perpendicular'])} & "
                 f"{pct(d['aligned_vs_perp']['all']['pct'])} " + r"\\")
        if lab == "IMBAT":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (td / "tab_sc_ratio.tex").write_text("\n".join(L), encoding="utf8")

    L = [HEAD, r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Street wind against uniform wind by sky view factor (SVF) quintile at label 15. $r$ is the median street-wind "
         r"ratio; M1$-$M5 is street-wind minus uniform-wind UTCI per segment (\textdegree{}C): median [5th, 95th percentile]. "
         r"Quintiles over the pooled segment set; Q1 is the most enclosed. Modelled values, not measurements.}",
         r"\label{tab:sc_svf}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{IMBAT ($U_{10}=9.6$~m\,s$^{-1}$)} & \multicolumn{2}{c}{CALM ($U_{10}=2.2$~m\,s$^{-1}$)} \\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"Quintile & Median SVF & $r$ & M1$-$M5 & $r$ & M1$-$M5 \\", r"\midrule"]
    for k in "12345":
        a, b = i["svf_quintile"][k], c["svf_quintile"][k]
        L.append(f"Q{k} & {a['svf_med']:.2f} & {a['r_med']:.2f} & {spread(a)} & {b['r_med']:.2f} & {spread(b)} " + r"\\")
    L += [r"\addlinespace", f"All & & {i['r_all']['p50']:.2f} & {spread(i)} & {c['r_all']['p50']:.2f} & {spread(c)} " + r"\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (td / "tab_sc_svf.tex").write_text("\n".join(L), encoding="utf8")

    rows = ["0.5", "1", "2", "3", "4", "5", "7", "8", "10", "12", "15", "17"]
    ri, rc = i["utci_response"]["reference"], c["utci_response"]["reference"]
    L = [HEAD, "% pythermalcomfort 4.7.0, utci(limit_inputs=False, round_output=False)", r"\begin{table}[!htbp]", r"\centering",
         r"\caption{UTCI (\textdegree{}C) against 10~m wind at the two reference states. IMBAT: $T_a$ "
         + f"{i['ta']:.2f}" + r"\textdegree{}C, RH " + f"{i['rh']:.2f}" + r"\%, $T_{mrt}$ " + f"{ri['tmrt']:.2f}"
         + r"\textdegree{}C. CALM: $T_a$ " + f"{c['ta']:.2f}" + r"\textdegree{}C, RH " + f"{c['rh']:.2f}" + r"\%, $T_{mrt}$ "
         + f"{rc['tmrt']:.2f}" + r"\textdegree{}C. $T_{mrt}$ is the median over pedestrian-space cells at label 15.}",
         r"\label{tab:sc_response}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}l" + "c" * len(rows) + r"@{}}", r"\toprule",
         r"Wind (m\,s$^{-1}$) & " + " & ".join(rows) + r" \\", r"\midrule",
         "IMBAT & " + " & ".join(f"{ri['at'][k]:.1f}" for k in rows) + r" \\",
         "CALM & " + " & ".join(f"{rc['at'][k]:.1f}" for k in rows) + r" \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (td / "tab_sc_response.tex").write_text("\n".join(L), encoding="utf8")


def main():
    fr = load()
    out = {"hw_edges": EDGES, "days": {}}
    pooled_svf = pd.concat([fr[d]["svf_med"] for d in DAYS])
    svf_edges = pooled_svf.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    out["svf_edges"] = svf_edges.tolist()

    for day in DAYS:
        d = fr[day].copy()
        f = H4["results"][day]["forcing"]
        ref = H4["results"][day]["reference"]
        r = d["ratio_med_b15"]
        res = {"n": int(len(d)), "n_r_nan": int(r.isna().sum()), "n_dM_nan": int(d["dM"].isna().sum()),
               "u10": f["u10"], "ta": f["ta"], "rh": f["rh"], "vrb15": f["vrb15"], "wd_ref": f["wd_mean_15_17"]}

        # --- wind ratio
        res["r_all"] = {**q(r, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)), "min": float(r.min()), "max": float(r.max()),
                        "mean": float(r.mean()), "share_lt_0_5": float((r < 0.5).mean()), "share_gt_1": float((r > 1).mean())}
        res["v10eff"] = {**q(d["v10eff_b15"]), "share_at_floor": float((d["v10eff_b15"] <= 0.5 + 1e-6).mean()),
                         "max_abs_diff_vs_u10_r": float(np.nanmax(np.abs(d["v10eff_b15"] - np.clip(f["u10"] * r, 0.5, 17))))}
        res["r_by_hw"] = {k: {"n": int(len(g)), **q(g["ratio_med_b15"])} for k, g in d.groupby("hwc")}
        res["r_by_orient"] = {k: {"n": int(len(g)), **q(g["ratio_med_b15"])} for k, g in d.groupby("orient")}
        res["r_by_hw_orient"] = {f"{h}|{o}": {"n": int(len(g)), **q(g["ratio_med_b15"])}
                                 for (h, o), g in d.groupby(["hwc", "orient"])}
        contrast = {}
        for h in ["low", "mid", "high", "none"]:
            a = res["r_by_hw_orient"][f"{h}|aligned"]["p50"]
            p = res["r_by_hw_orient"][f"{h}|perpendicular"]["p50"]
            contrast[h] = {"aligned": a, "perpendicular": p, "pct": 100 * (a / p - 1)}
        a, p = res["r_by_orient"]["aligned"]["p50"], res["r_by_orient"]["perpendicular"]["p50"]
        contrast["all"] = {"aligned": a, "perpendicular": p, "pct": 100 * (a / p - 1)}
        res["aligned_vs_perp"] = contrast

        # --- street vs uniform wind
        m1, m5, dm = d["utci_b15"], d["utci_unif_b15"], d["dM"]
        res["M1"] = {**q(m1), "var": float(m1.var())}
        res["M5"] = {**q(m5), "var": float(m5.var())}
        res["dM"] = {**q(dm, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)), "min": float(dm.min()), "max": float(dm.max()),
                     "share_pos": float((dm > 0).mean()), "share_abs_le_0_5": float((dm.abs() <= 0.5).mean()),
                     "share_abs_le_1": float((dm.abs() <= 1).mean())}
        res["rho_M1_M5"] = float(m1.corr(m5, method="spearman"))
        res["pearson_M1_M5"] = float(m1.corr(m5))
        res["rho_dM_r"] = float(dm.corr(r, method="spearman"))
        k = int(round(0.1 * m1.notna().sum()))
        top1 = set(d.nlargest(k, "utci_b15")["seg_id"])
        top5 = set(d.nlargest(k, "utci_unif_b15")["seg_id"])
        res["top_decile_overlap"] = len(top1 & top5) / k

        d["svfq"] = pd.cut(d["svf_med"], [-np.inf, *svf_edges, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
        tab = {}
        for sq, g in d.groupby("svfq"):
            tab[int(sq)] = {"n": int(len(g)), "svf_med": float(g["svf_med"].median()),
                            "r_med": float(g["ratio_med_b15"].median()), "tmrt_med": float(g["tmrt_b15"].median()),
                            "M1_med": float(g["utci_b15"].median()), "M5_med": float(g["utci_unif_b15"].median()),
                            "dM": q(g["dM"]), "share_pos": float((g["dM"] > 0).mean())}
        res["svf_quintile"] = tab
        res["enclosed_minus_open"] = {"M1": tab[1]["M1_med"] - tab[5]["M1_med"], "M5": tab[1]["M5_med"] - tab[5]["M5_med"]}

        # --- consistency of the stored columns with a direct UTCI evaluation (could fail; medians of UTCI vs UTCI of medians)
        cm5 = U(f["ta"], d["tmrt_b15"].to_numpy(), np.full(len(d), np.clip(f["u10"], 0.5, 17)), f["rh"])
        cm1 = U(f["ta"], d["tmrt_b15"].to_numpy(), d["v10eff_b15"].to_numpy(), f["rh"])
        res["check_utci_of_medians"] = {
            "M5_median_abs_diff": float(np.nanmedian(np.abs(cm5 - m5))), "M1_median_abs_diff": float(np.nanmedian(np.abs(cm1 - m1)))}

        # --- UTCI wind response at the reference states
        tmrt_low = float(d["tmrt_b15"].quantile(0.05))
        cool = d[d["utci_b15"] < d["utci_b15"].median() - 2]
        res["cool_cluster"] = {"n": int(len(cool)), "share": float(len(cool) / len(d)), "tmrt_med": float(cool["tmrt_b15"].median()),
                               "r_med": float(cool["ratio_med_b15"].median()), "dM": q(cool["dM"]),
                               "share_pos": float((cool["dM"] > 0).mean())}
        v = d["v10eff_b15"]
        res["v_range"] = {"min": float(v.min()), "max": float(v.max()), "share_gt_10_35": float((v > 10.35).mean())}
        states = {"reference": ref["tmrt_ref"], "tmrt_p05": tmrt_low, "cool_cluster": res["cool_cluster"]["tmrt_med"]}
        vv = np.round(np.arange(0.5, 17.0001, 0.01), 2)
        resp = {}
        for name, tm in states.items():
            u = U(f["ta"], np.full(vv.size, tm), vv, f["rh"])
            i = int(np.nanargmin(u))
            at = {str(v): float(u[np.argmin(np.abs(vv - v))]) for v in [0.5, 1, 2, 2.23, 3, 4, 5, 7, 7.32, 8, 9.6, 10, 12, 15, 17]}
            local = np.where((u[1:-1] < u[:-2]) & (u[1:-1] < u[2:]))[0] + 1
            localmax = np.where((u[1:-1] > u[:-2]) & (u[1:-1] > u[2:]))[0] + 1
            obs = (vv >= res["v_range"]["min"]) & (vv <= res["v_range"]["max"])
            u_at_u10 = float(u[np.argmin(np.abs(vv - f["u10"]))])
            below = vv[(vv < f["u10"]) & (u < u_at_u10)]
            resp[name] = {"tmrt": tm, "argmin_v": float(vv[i]), "min_utci": float(u[i]), "at": at, "n_nan": int(np.isnan(u).sum()),
                          "local_min_v": vv[local].tolist(), "local_max_v": vv[localmax].tolist(),
                          "utci_at_local_min": u[local].tolist(), "utci_at_local_max": u[localmax].tolist(),
                          "utci_at_u10": u_at_u10,
                          "smallest_v_below_u10_with_lower_utci": float(below.min()) if below.size else None,
                          "obs_utci_min": float(u[obs].min()), "obs_utci_max": float(u[obs].max())}
        res["utci_response"] = resp
        out["days"][day] = res

    a = fr["IMBAT"].set_index("seg_id")["ratio_med_b15"]
    b = fr["CALM"].set_index("seg_id")["ratio_med_b15"]
    out["rho_r_IMBAT_CALM"] = float(a.corr(b, method="spearman"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sc_numbers.json").write_text(json.dumps(out, indent=1))
    write_tables(json.loads(json.dumps(out)))
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
