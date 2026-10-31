"""H4 Shade vs breeze (pre_analysis_lock.md section 2, H4). Implemented as written.

Per segment and day, Shapley two-factor decomposition of M1 relative to the city reference
(Tmrt_ref, v_ref) = (median Tmrt, median v10_eff) over ALL PS cells at band 15; Ta, RH fixed at the forcing values:
  shade  = 1/2 [ U(T_i, v_ref) - U(T_ref, v_ref) + U(T_i, v_i) - U(T_ref, v_i) ]
  breeze = 1/2 [ U(T_ref, v_i) - U(T_ref, v_ref) + U(T_i, v_i) - U(T_i, v_ref) ]
with T_i = tmrt_b15 (segment median), v_i = v10eff_b15 (segment median), U = pythermalcomfort UTCI.
Reports Var(shade), Var(breeze) per day (block-bootstrap CI), IMBAT - CALM contrast, orientation class x H/W tercile
table. H4 "shade-dominated" iff Var(shade) > 4 x Var(breeze) on BOTH days.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402


def shapley_terms(ta, rh, t_i, v_i, t_ref, v_ref):
    """Two-factor Shapley terms (shade, breeze) and the UTCI of the cell inputs. Arrays in, arrays out."""
    u_iv = cm.utci_vec(ta, t_i, v_i, rh)
    u_ir = cm.utci_vec(ta, t_i, np.full_like(v_i, v_ref), rh)
    u_ri = cm.utci_vec(ta, np.full_like(t_i, t_ref), v_i, rh)
    u_rr = float(cm.utci_vec(ta, [t_ref], [v_ref], rh)[0])
    shade = 0.5 * ((u_ir - u_rr) + (u_iv - u_ri))
    breeze = 0.5 * ((u_ri - u_rr) + (u_iv - u_ir))
    return shade, breeze, u_iv, u_rr


def weighted_median(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w, float)[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def ref_from_segments(df):
    w = df["n_ps_valid"].to_numpy(float)
    return weighted_median(df["tmrt_b15"].to_numpy(float), w), weighted_median(df["v10eff_b15"].to_numpy(float), w)


def var_stats(shade, breeze, w=None):
    vs, vb = cm.wvar(shade, w), cm.wvar(breeze, w)
    return vs, vb, (vs / vb if vb > 0 else np.inf)


def analyse_day(day, df, ref_mode, args):
    fo = df.attrs["forcing"]
    ta, rh = fo["ta"], fo["rh"]
    d = df.dropna(subset=["tmrt_b15", "v10eff_b15", "m1"]).reset_index(drop=True)
    t_seg, v_seg = ref_from_segments(d)
    ref = dict(source="segments (n_ps_valid-weighted median of segment medians; approximation)", tmrt_ref=t_seg, v_ref=v_seg,
               segment_based=dict(tmrt_ref=t_seg, v_ref=v_seg))
    t_ref, v_ref = t_seg, v_seg
    if ref_mode == "rasters":
        import raster_ladder as rl
        rp = rl.raster_pass(day, d[["seg_id", "lcz_primary"]], sorted(d["tile_id"].unique()))
        if rp["n_tiles_used"] > 0:
            t_ref, v_ref = rp["tmrt_ref"], rp["v_ref"]
            ref.update(source=f"PS-cell median over {rp['n_tiles_used']} finished-tile rasters (n_ps={rp['n_ps_total']})",
                       tmrt_ref=t_ref, v_ref=v_ref)
    shade, breeze, u_iv, u_rr = shapley_terms(ta, rh, d["tmrt_b15"].to_numpy(float), d["v10eff_b15"].to_numpy(float), t_ref, v_ref)
    d["shade_term"], d["breeze_term"], d["u_inputs"] = shade, breeze, u_iv
    d["m1_minus_uref"] = d["m1"] - u_rr
    d["closure_resid"] = d["m1_minus_uref"] - (shade + breeze)
    ub, inv, counts = cm.boot_block_counts(d["block"].to_numpy(), cm.N_BOOT if args.n_boot is None else args.n_boot, cm.SEED + 4)
    W = [counts[r][inv] for r in range(counts.shape[0])]
    vs, vb, ratio = var_stats(shade, breeze)
    bs = np.array([var_stats(shade, breeze, w) for w in W])
    res = dict(day=day, label=cm.label_for(df), n_segments=len(d), forcing=fo, reference=ref, u_ref=u_rr,
               var_shade=vs, var_breeze=vb, ratio_var_shade_over_breeze=ratio,
               var_shade_ci95=cm.ci95(bs[:, 0]), var_breeze_ci95=cm.ci95(bs[:, 1]), ratio_ci95=cm.ci95(bs[:, 2]),
               ratio_gt_4=bool(vs > 4 * vb),
               cov_shade_breeze=float(np.cov(shade, breeze)[0, 1]),
               var_m1=float(np.var(d["m1"])), var_sum_terms=float(np.var(shade + breeze)),
               var_closure_resid=float(np.var(d["closure_resid"])),
               mean_closure_resid=float(d["closure_resid"].mean()),
               mean_shade=float(shade.mean()), mean_breeze=float(breeze.mean()),
               sd_tmrt=float(d["tmrt_b15"].std()), sd_v=float(d["v10eff_b15"].std()))
    # mechanism: UTCI sensitivities at the reference point (finite differences, pythermalcomfort)
    uT = float(cm.utci_vec(ta, [t_ref + 1.0], [v_ref], rh)[0] - u_rr)
    uV = float(cm.utci_vec(ta, [t_ref], [v_ref + 1.0], rh)[0] - u_rr)
    res["mechanism"] = dict(dUTCI_per_degC_Tmrt=uT, dUTCI_per_ms_wind=uV,
                            predicted_var_ratio_linearised=float((uT ** 2 * d["tmrt_b15"].var()) / max(uV ** 2 * d["v10eff_b15"].var(), 1e-12)),
                            note="UTCI wind response is weak near Ta ~ 35 C (lock caveat); linearised ratio = (dU/dT)^2 Var(Tmrt) / (dU/dv)^2 Var(v)")
    return d, res, W, ub, inv


NO_BLD = "no enclosing buildings"      # lock A8(2): hw_ratio is NaN when there are no enclosing buildings on either side
HW_CLASSES = ["low", "mid", "high", NO_BLD]


def hw_class(hw, tercile_edges) -> pd.Series:
    """H/W class per segment. Terciles use the edges computed on NON-NaN hw_ratio only (main()); segments with NaN
    hw_ratio form their own class NO_BLD (no imputation anywhere)."""
    hw = pd.Series(np.asarray(hw, float), index=getattr(hw, "index", None))
    cls = pd.cut(hw, [-np.inf, tercile_edges[0], tercile_edges[1], np.inf], labels=["low", "mid", "high"]).astype(object)
    cls[hw.isna()] = NO_BLD
    return cls


def orientation_table(d, tercile_edges, W, day):
    """orientation class x H/W class (terciles + 'no enclosing buildings'): n, mean terms with block-bootstrap CI, var terms.

    d must be the frame returned by analyse_day (index 0..n-1, aligned with the bootstrap weight vectors in W).
    """
    d = d.reset_index(drop=True)
    hwc = hw_class(d["hw_ratio"], tercile_edges)
    rows = []
    for oc in ["aligned", "oblique", "perpendicular"]:
        for t in HW_CLASSES:
            m = ((d["orient_class"] == oc) & (hwc == t)).to_numpy()
            if m.sum() < 5:
                rows.append(dict(day=day, orient=oc, hw_tercile=t, n=int(m.sum())))
                continue
            s, b = d.loc[m, "shade_term"].to_numpy(), d.loc[m, "breeze_term"].to_numpy()
            row = dict(day=day, orient=oc, hw_tercile=t, n=int(m.sum()), mean_shade=s.mean(), mean_breeze=b.mean(),
                       var_shade=s.var(), var_breeze=b.var())
            if len(W) and len(W[0]) == len(d):
                ms = np.array([np.sum(w[m] * s) / np.sum(w[m]) if w[m].sum() > 0 else np.nan for w in W])
                mb = np.array([np.sum(w[m] * b) / np.sum(w[m]) if w[m].sum() > 0 else np.nan for w in W])
                row.update(mean_shade_lo=cm.ci95(ms)[0], mean_shade_hi=cm.ci95(ms)[1],
                           mean_breeze_lo=cm.ci95(mb)[0], mean_breeze_hi=cm.ci95(mb)[1])
            rows.append(row)
    return pd.DataFrame(rows)


def did_effect(d, tercile_edges, W):
    """Interaction effect: [mean breeze(aligned) - mean breeze(perpendicular)] in high-H/W minus the same in low-H/W (and for shade).
    Uses the low / high terciles only; the 'no enclosing buildings' class does not enter (it has no H/W)."""
    d = d.reset_index(drop=True)
    hwc = hw_class(d["hw_ratio"], tercile_edges)
    ok = d["orient_class"].notna()
    out = {}
    for term in ("shade_term", "breeze_term"):
        def stat(w):
            def cellmean(oc, t):
                m = ((d["orient_class"] == oc) & (hwc == t) & ok).to_numpy()
                ww = w[m]
                return np.sum(ww * d.loc[m, term].to_numpy()) / np.sum(ww) if ww.sum() > 0 else np.nan
            return (cellmean("aligned", "high") - cellmean("perpendicular", "high")) -                    (cellmean("aligned", "low") - cellmean("perpendicular", "low"))
        pt = stat(np.ones(len(d)))
        bs = np.array([stat(w) for w in W])
        out[term] = dict(did=float(pt), ci95=cm.ci95(bs))
    return out


def main():
    cm.utf8_stdout()
    ap = argparse.ArgumentParser()
    cm.add_common_args(ap)
    ap.add_argument("--ref-source", choices=["auto", "segments", "rasters"], default="auto")
    args = ap.parse_args()
    frames = cm.frames_from_args(args)
    od = cm.out_dir("h4", args.synthetic, args.tag, args.out_suffix)
    ref_mode = args.ref_source
    if ref_mode == "auto":
        ref_mode = "segments" if args.synthetic else "rasters"
    results, tables, did, per = {}, [], {}, {}
    # shared H/W terciles over the pooled segment set (identical attribute across days)
    hw_all = pd.concat([f["hw_ratio"] for f in frames.values()]).dropna()
    edges = (float(hw_all.quantile(1 / 3)), float(hw_all.quantile(2 / 3)))
    cache = {}
    for day, f in frames.items():
        d, res, W, ub, inv = analyse_day(day, f, ref_mode, args)
        results[day] = res
        cache[day] = (d, W, inv)
        tables.append(orientation_table(d, edges, W, day))
        hwc = hw_class(d["hw_ratio"], edges)
        results[day]["hw_class_counts"] = {k: int((hwc == k).sum()) for k in HW_CLASSES}
        results[day]["hw_nan_share"] = float(d["hw_ratio"].isna().mean())
        did[day] = did_effect(d, edges, W)
        d.drop(columns=[c for c in ["geometry"] if c in d]).to_parquet(
            od / cm.fname(f"seg_h4_terms_{day}", "parquet", args.synthetic, f))
    # IMBAT - CALM contrast on common segments (paired block bootstrap)
    contrast = None
    if len(cache) == 2:
        a, b = cache["IMBAT"][0].set_index("seg_id"), cache["CALM"][0].set_index("seg_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        ub2, inv2, c2 = cm.boot_block_counts(a["block"].to_numpy(), args.n_boot, cm.SEED + 5)
        def st(w):
            va, vb_ = var_stats(a["shade_term"], a["breeze_term"], w), var_stats(b["shade_term"], b["breeze_term"], w)
            return (va[0] - vb_[0], va[1] - vb_[1], va[2] - vb_[2], va[2], vb_[2])
        pt = np.array(st(None)); bs = np.array([st(c2[r][inv2]) for r in range(c2.shape[0])])
        names = ["dVar_shade", "dVar_breeze", "dRatio", "ratio_IMBAT", "ratio_CALM"]
        contrast = dict(n_common=int(len(common)),
                        **{n: dict(value=float(pt[i]), ci95=cm.ci95(bs[:, i])) for i, n in enumerate(names[:3])},
                        note="IMBAT minus CALM on segments present in both days; paired 2 km block bootstrap")
    dec = {d: dict(ratio=r["ratio_var_shade_over_breeze"], gt4=r["ratio_gt_4"]) for d, r in results.items()}
    dec["H4"] = cm.verdict(all(r["ratio_gt_4"] for r in results.values()) and len(results) == 2)
    dec["note"] = "shade-dominated iff Var(shade) > 4 x Var(breeze) on both days"
    if len(results) < 2:
        dec["note"] += f"; only {sorted(results)} available -> verdict not final"
    first = next(iter(frames.values()))
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(od / cm.fname("h4_orientation_x_hw_table", "csv", args.synthetic, first), index=False)
    cm.write_json(od / cm.fname("h4_results", "json", args.synthetic, first),
                  dict(constants=cm.constants_dict(), hw_tercile_edges=edges,
                       hw_tercile_note="terciles of non-NaN hw_ratio (pooled over both days); NaN hw_ratio = class 'no enclosing buildings' (A8 item 2)",
                       results=results,
                       contrast_IMBAT_minus_CALM=contrast, orientation_hw_interaction_DiD=did, decision=dec,
                       ref_mode=ref_mode))
    print(f"\n=== H4 shade vs breeze  [{cm.label_for(first)}] ===")
    for day, r in results.items():
        print(f"{day}: n={r['n_segments']} ref: Tmrt={r['reference']['tmrt_ref']:.2f} v={r['reference']['v_ref']:.2f} "
              f"[{r['reference']['source']}]")
        if "segment_based" in r["reference"] and ref_mode == "rasters":
            sb = r["reference"]["segment_based"]
            print(f"   (segment-weighted approximation of ref: Tmrt={sb['tmrt_ref']:.2f} v={sb['v_ref']:.2f})")
        print("   Var(shade)  = " + cm.fmt_ci(r["var_shade"], *r["var_shade_ci95"]))
        print("   Var(breeze) = " + cm.fmt_ci(r["var_breeze"], *r["var_breeze_ci95"]))
        print("   ratio       = " + cm.fmt_ci(r["ratio_var_shade_over_breeze"], *r["ratio_ci95"], nd=2) + f"  -> ratio>4: {r['ratio_gt_4']}")
        m = r["mechanism"]
        print(f"   mechanism: dUTCI/dTmrt={m['dUTCI_per_degC_Tmrt']:.3f} C/C, dUTCI/dv={m['dUTCI_per_ms_wind']:.3f} C per m/s; "
              f"linearised var ratio={m['predicted_var_ratio_linearised']:.2f}; closure var={r['var_closure_resid']:.4f}")
    if contrast:
        print("IMBAT - CALM: " + "; ".join(f"{k} {v['value']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
                                          for k, v in contrast.items() if isinstance(v, dict)))
    for day in results:
        print(f"{day} interaction DiD (aligned-perp, high-low H/W): " + "; ".join(
            f"{k} {v['did']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for k, v in did[day].items()))
    for day, r in results.items():
        print(f"{day} H/W classes (A8-2): {r['hw_class_counts']}  NaN share={r['hw_nan_share']:.3f}")
    print(f"H4 decision: {dec['H4']} ({dec['note']})")


if __name__ == "__main__":
    main()
