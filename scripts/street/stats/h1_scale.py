"""H1 Scale mismatch (pre_analysis_lock.md section 2, H1). Implemented as written.

Per day, on segments:
  1. LMM  M1 ~ 1 + (1 | district / LCZ)   (statsmodels MixedLM, REML; nested = district intercept + LCZ variance
     component within district). ICC_district, ICC_LCZ|district, residual (within-LCZ) share.
     Non-convergence -> nested one-way ANOVA (method-of-moments) fallback, reported (lock names no fallback: open issue).
  2. 5-fold spatial-block CV (2 km blocks, seed 20261030): R2 of LCZ-only fixed-effect model vs LightGBM on the H2
     features minus LCZ.
  3. Scale ladder eta2 of M1 by LCZ at segment, 50, 100, 250, 500 m.
  H1 SUPPORTED if residual share > 0.50 AND CV R2 (LCZ-only) < 0.30, on BOTH days (most literal reading).
  95 % CIs: 2 km block bootstrap, 1000 resamples (LMM refit each resample; CV R2 resamples the fixed OOF predictions).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Variance components
# ----------------------------------------------------------------------------------------------
def anova_nested(y, d, l):
    """Unbalanced nested ANOVA (Searle) estimators of var(district), var(LCZ | district), var(resid). Truncated at 0."""
    df = pd.DataFrame({"y": y, "d": d, "l": l})
    N = len(df)
    cell = df.groupby(["d", "l"])["y"].agg(["size", "mean"]).reset_index()
    dist = df.groupby("d")["y"].agg(["size", "mean"])
    ybar = df["y"].mean()
    a, c = len(dist), len(cell)
    cell = cell.join(dist.rename(columns={"size": "n_i", "mean": "m_i"}), on="d")
    sse = float(((df["y"] - df.groupby(["d", "l"])["y"].transform("mean")) ** 2).sum())
    ssb = float((cell["size"] * (cell["mean"] - cell["m_i"]) ** 2).sum())
    ssa = float((dist["size"] * (dist["mean"] - ybar) ** 2).sum())
    mse = sse / (N - c)
    if c - a <= 0:                     # every district has a single LCZ cell: LCZ-in-district not separable
        return 0.0, 0.0, float(((df["y"] - ybar) ** 2).sum() / (N - 1))
    msb = ssb / (c - a)
    s1 = float((cell["size"] ** 2 / cell["n_i"]).sum())
    k1 = (N - s1) / (c - a)
    var_l = max((msb - mse) / k1, 0.0)
    if a < 2:                          # one district: var(district) not estimable (set to 0, flagged by caller)
        return 0.0, var_l, mse
    msa = ssa / (a - 1)
    s2 = float((cell["size"] ** 2).sum() / N)
    k2 = (s1 - s2) / (a - 1)
    k3 = (N - float((dist["size"] ** 2).sum()) / N) / (a - 1)
    var_d = max((msa - mse - k2 * var_l) / k3, 0.0)
    return var_d, var_l, mse


def lmm_components(y, d, l, allow_fallback=True) -> dict:
    """Nested LMM. Returns dict(var_d, var_l, var_e, converged, method, boundary)."""
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"y": np.asarray(y, float), "d": pd.Series(d).astype(str).to_numpy(),
                       "lcz_c": "L" + pd.Series(l).astype(str).to_numpy()})
    res = dict(method="mixedlm_reml", converged=False)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            md = smf.mixedlm("y ~ 1", df, groups="d", re_formula="1", vc_formula={"lcz": "0 + lcz_c"})
            r = md.fit(reml=True)                           # statsmodels default (bfgs, then its own lbfgs retry)
            if not r.converged:                             # documented 2nd attempt: derivative-free Powell
                r = md.fit(reml=True, method=["powell"])
                res["method"] = "mixedlm_reml_powell"
        vd, vl, ve = float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)
        # converged = statsmodels' final optimizer flag (after its own lbfgs retry). A boundary MLE (a variance
        # component ~ 0, "Hessian not positive definite") is a valid REML solution and is flagged, not treated as failure.
        res.update(var_d=vd, var_l=vl, var_e=ve, converged=bool(r.converged),
                   boundary=bool(min(vd, vl) < 1e-6 * (vd + vl + ve)), warnings=sorted({str(x.message)[:60] for x in w}))
        if not np.all(np.isfinite([vd, vl, ve])):
            res["converged"] = False
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:120]
    if (not res["converged"]) and allow_fallback:
        vd, vl, ve = anova_nested(df["y"].to_numpy(), df["d"].to_numpy(), df["lcz_c"].to_numpy())
        res.update(var_d=vd, var_l=vl, var_e=ve, method="anova_fallback", boundary=bool(min(vd, vl) < 1e-9))
    return res


def _boot_lmm(args):
    y, d, l, rows = args
    r = lmm_components(y[rows], d[rows], l[rows])
    tot = r["var_d"] + r["var_l"] + r["var_e"]
    return (r["var_d"] / tot, r["var_l"] / tot, r["var_e"] / tot, (r["var_d"] + r["var_l"]) / tot,
            1.0 if r["method"] == "anova_fallback" else 0.0)


# ----------------------------------------------------------------------------------------------
# Scale ladder
# ----------------------------------------------------------------------------------------------
def ladder_units_from_segments(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Fallback (and synthetic) ladder: segments aggregated to g<L>_id with n_ps_valid-weighted mean of segment M1 and
    n_ps-weighted majority LCZ. Approximation of the PS-cell median (segment medians only are in the contract)."""
    out = {}
    for L in (50, 100, 250, 500):
        col = f"g{L:03d}_id"
        w = df["n_ps_valid"].to_numpy(float)
        t = df.assign(_w=w, _wm=w * df["m1"].to_numpy(float))
        g = t.groupby(col)
        u = pd.DataFrame({"n_ps": g["_w"].sum(), "m1_med": g["_wm"].sum() / g["_w"].sum()})
        k = t[t["lcz_primary"] > 0].groupby([col, "lcz_primary"])["_w"].sum().reset_index().sort_values([col, "_w"], ascending=[True, False])
        u["lcz"] = k.drop_duplicates(col).set_index(col)["lcz_primary"]
        u["lcz"] = u["lcz"].fillna(0).astype(int)      # 0 = no valid LCZ in the cell -> dropped from eta2 (A6)
        xy = g[["x_mid", "y_mid"]].mean()
        u["bx"] = np.floor(xy["x_mid"] / cm.BLOCK_M).astype(int)
        u["by"] = np.floor(xy["y_mid"] / cm.BLOCK_M).astype(int)
        out[L] = u.reset_index().rename(columns={col: "cell_id"})
    return out


def ladder_eta(units: pd.DataFrame, ub_boot=None):
    """eta2 of m1 by pooled LCZ over units; block-bootstrap CI via block multiplicities."""
    u = units[(units["n_ps"] >= 25) & (units["lcz"] > 0)].copy()
    u["lcz_pool"] = cm.lcz_pool(u["lcz"]).astype(int)
    blocks = (u["bx"].astype(str) + "_" + u["by"].astype(str)).to_numpy()
    ub, inv, counts = cm.boot_block_counts(blocks, cm.N_BOOT, cm.SEED + 11)
    y, g = u["m1_med"].to_numpy(), u["lcz_pool"].to_numpy()
    point = cm.eta_squared(y, g)
    boots = np.array([cm.eta_squared(y, g, counts[r][inv]) for r in range(cm.N_BOOT)])
    return dict(n_units=int(len(u)), eta2=point, **{"ci95": cm.ci95(boots)})


# ----------------------------------------------------------------------------------------------
def run_day(day: str, df: pd.DataFrame, args, ladder_units: dict | None) -> dict:
    t0 = time.time()
    lab = cm.label_for(df)
    n = len(df)
    y = df["m1"].to_numpy(float)
    res = dict(day=day, label=lab, n_segments=n, n_blocks=int(df["block"].nunique()), n_folds_nonempty=int(df["fold"].nunique()),
               n_districts=int(df["district"].nunique()),
               lcz_counts=df.loc[df["lcz_valid"], "lcz"].value_counts().sort_index().to_dict(), forcing=df.attrs.get("forcing"))
    # --- LMM ----------------------------------------------------------------------------------
    # A6: lcz_valid == False (LCZ raster no-data) excluded from every LCZ-based statistic. A1: district "unknown" is kept
    # and forms its own level of the district random effect.
    v = df["lcz_valid"].to_numpy()
    dl = df[v].reset_index(drop=True)
    res["n_lcz_excluded_A6"] = int((~v).sum())
    res["lmm_n_segments"] = len(dl)
    res["lmm_n_districts"] = int(dl["district"].nunique())
    res["lmm_n_unknown_district_segments"] = int((dl["district"].str.lower() == "unknown").sum())
    yl = dl["m1"].to_numpy(float)
    pt = lmm_components(yl, dl["district"], dl["lcz"])
    tot = pt["var_d"] + pt["var_l"] + pt["var_e"]
    shares = dict(icc_district=pt["var_d"] / tot, icc_lcz_in_district=pt["var_l"] / tot,
                  icc_same_lcz_same_district=(pt["var_d"] + pt["var_l"]) / tot, resid_share=pt["var_e"] / tot)
    ub, inv, counts = cm.boot_block_counts(df["block"].to_numpy(), cm.N_BOOT, cm.SEED)
    ubl, invl, countsl = cm.boot_block_counts(dl["block"].to_numpy(), cm.N_BOOT, cm.SEED)
    jobs = [(yl, dl["district"].to_numpy(), dl["lcz"].to_numpy(), cm.rows_from_counts(invl, countsl[r]))
            for r in range(args.n_boot)]
    from joblib import Parallel, delayed
    bl = Parallel(n_jobs=min(args.jobs, cm.MAX_PROCS), backend="loky")(delayed(_boot_lmm)(j) for j in jobs)
    bl = np.array(bl)
    res["lmm"] = dict(point=pt, shares=shares, n_boot=args.n_boot,
                      ci95={k: cm.ci95(bl[:, i]) for i, k in enumerate(
                          ["icc_district", "icc_lcz_in_district", "resid_share", "icc_same_lcz_same_district"])},
                      bootstrap_fallback_share=float(bl[:, 4].mean()),
                      var_components=dict(district=pt["var_d"], lcz_in_district=pt["var_l"], resid=pt["var_e"]))
    # --- CV R2 --------------------------------------------------------------------------------
    p_lcz = cm.lcz_only_oof(df)
    feats = cm.FEATURES
    p_mor = cm.lgb_oof(df, feats, n_jobs=min(args.jobs, cm.MAX_PROCS))
    w = [counts[r][inv] for r in range(args.n_boot)]
    vf = v.astype(float)
    r2_lcz = cm.wr2(y, np.nan_to_num(p_lcz), vf)                 # LCZ-only: lcz_valid rows only (A6)
    r2_mor = cm.wr2(y, p_mor, vf)                                # morphology model on the SAME rows (like-for-like)
    r2_mor_all = cm.wr2(y, p_mor)                                # ... and on all rows (A6: non-LCZ analysis)
    b_lcz = np.array([cm.wr2(y, np.nan_to_num(p_lcz), wr * vf) for wr in w])
    b_mor = np.array([cm.wr2(y, p_mor, wr * vf) for wr in w])
    b_diff = b_mor - b_lcz
    res["cv"] = dict(
        r2_lcz_only=r2_lcz, r2_lcz_only_ci95=cm.ci95(b_lcz), r2_morphology_lgbm=r2_mor,
        r2_morphology_lgbm_ci95=cm.ci95(b_mor), r2_morphology_lgbm_all_rows=r2_mor_all, r2_diff=r2_mor - r2_lcz, r2_diff_ci95=cm.ci95(b_diff),
        per_fold_r2_lcz=[cm.wr2(y[(df["fold"] == k) & v], p_lcz[(df["fold"] == k) & v]) if ((df["fold"] == k) & v).sum() else None for k in range(cm.N_FOLDS)],
        per_fold_r2_morph=[cm.wr2(y[df["fold"] == k], p_mor[df["fold"] == k]) if (df["fold"] == k).sum() else None for k in range(cm.N_FOLDS)],
        ci_note="CIs resample 2 km blocks of the fixed out-of-fold predictions (models are not refit per resample)")
    # --- ladder -------------------------------------------------------------------------------
    seg_eta = dict(n_units=int(v.sum()), eta2=cm.eta_squared(y, df["lcz"], vf),
                   ci95=cm.ci95([cm.eta_squared(y, df["lcz"].to_numpy(), wr * vf) for wr in w]))
    lad = {"segment": seg_eta}
    for L, u in (ladder_units or {}).items():
        lad[str(L)] = ladder_eta(u)
    res["ladder_eta2"] = lad
    res["ladder_source"] = args.ladder_source_used
    # supplementary (NOT a decision quantity): 1 - eta2_segment as a naive "within-LCZ" share
    res["supplementary"] = dict(one_minus_eta2_segment=1 - seg_eta["eta2"])
    res["runtime_s"] = time.time() - t0
    return res


def main():
    cm.utf8_stdout()
    ap = argparse.ArgumentParser()
    cm.add_common_args(ap)
    ap.add_argument("--ladder-source", choices=["auto", "segments", "rasters"], default="auto")
    args = ap.parse_args()
    frames = cm.frames_from_args(args)
    od = cm.out_dir("h1", args.synthetic, args.tag, args.out_suffix)
    results = {}
    for day, df in frames.items():
        src = args.ladder_source
        if src == "auto":
            src = "segments" if args.synthetic else "rasters"
        if src == "rasters":
            import raster_ladder as rl
            seg = df[["seg_id", "lcz_primary"]].dropna()   # lcz_primary == 0 (A6) is ignored in the cell majority
            tiles = sorted(df["tile_id"].unique())
            rp = rl.raster_pass(day, seg, tiles)
            units = rp["ladder"] if rp["n_tiles_used"] > 0 else None
            if units is None:
                print(f"[{day}] raster ladder unavailable -> segment-aggregated fallback")
                src = "segments"
                units = ladder_units_from_segments(df)
            else:
                print(f"[{day}] raster ladder from {rp['n_tiles_used']} tiles")
        else:
            units = ladder_units_from_segments(df)
        args.ladder_source_used = ("PS-cell median over finished-tile rasters" if src == "rasters" else
                                   "segment-aggregated (n_ps-weighted mean of segment medians; approximation)")
        results[day] = run_day(day, df, args, units)
    # ---- decision rule -----------------------------------------------------------------------
    dec = {}
    ok_all = True
    for day, r in results.items():
        rs = r["lmm"]["shares"]["resid_share"]
        cv = r["cv"]["r2_lcz_only"]
        c1, c2 = rs > 0.5, cv < 0.30
        dec[day] = dict(resid_share=rs, resid_gt_0_5=c1, cv_r2_lcz_only=cv, cv_lt_0_30=c2, day_pass=c1 and c2)
        ok_all &= (c1 and c2)
    dec["H1"] = cm.verdict(ok_all and len(results) == 2)
    dec["note"] = ("Both conditions must hold on both days (literal reading). "
                   + ("" if len(results) == 2 else f"Only {sorted(results)} available -> verdict not final."))
    out = dict(constants=cm.constants_dict(), results=results, decision=dec,
               lmm_converged={d: results[d]["lmm"]["point"]["converged"] for d in results})
    first = next(iter(frames.values()))
    stem = cm.fname("h1_results", "json", args.synthetic, first)
    cm.write_json(od / stem, out)
    # printed summary
    print(f"\n=== H1 scale mismatch  [{cm.label_for(first)}] ===")
    for day, r in results.items():
        s = r["lmm"]["shares"]
        ci = r["lmm"]["ci95"]
        p = r["lmm"]["point"]
        print(f"{day}: n={r['n_segments']} blocks={r['n_blocks']} folds={r['n_folds_nonempty']} LMM n={r['lmm_n_segments']} "
              f"districts={r['lmm_n_districts']} (unknown-district segs kept as own level: {r['lmm_n_unknown_district_segments']}; "
              f"excluded for LCZ=0 (A6): {r['n_lcz_excluded_A6']}) method={p['method']} converged={p['converged']} boundary={p.get('boundary')} "
              f"fallback_in_boot={r['lmm']['bootstrap_fallback_share']:.3f}")
        print("   ICC_district        = " + cm.fmt_ci(s["icc_district"], *ci["icc_district"]))
        print("   ICC_LCZ|district    = " + cm.fmt_ci(s["icc_lcz_in_district"], *ci["icc_lcz_in_district"]))
        print("   residual share      = " + cm.fmt_ci(s["resid_share"], *ci["resid_share"]))
        c = r["cv"]
        print("   CV R2 LCZ-only      = " + cm.fmt_ci(c["r2_lcz_only"], *c["r2_lcz_only_ci95"]))
        print("   CV R2 morph LightGBM= " + cm.fmt_ci(c["r2_morphology_lgbm"], *c["r2_morphology_lgbm_ci95"]))
        print("   eta2 ladder: " + "; ".join(f"{k}: {v['eta2']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] (n={v['n_units']})"
                                              for k, v in r["ladder_eta2"].items()))
        print(f"   {day}: resid>0.5 {dec[day]['resid_gt_0_5']} ; CV R2 LCZ<0.30 {dec[day]['cv_lt_0_30']}")
    print(f"H1 decision: {dec['H1']}  ({dec['note']})")
    # draft figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for day, r in results.items():
            ks = list(r["ladder_eta2"].keys())
            ys = [r["ladder_eta2"][k]["eta2"] for k in ks]
            lo = [r["ladder_eta2"][k]["ci95"][0] for k in ks]
            hi = [r["ladder_eta2"][k]["ci95"][1] for k in ks]
            ax.errorbar(range(len(ks)), ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], marker="o", capsize=3, label=day)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_ylabel("eta$^2$ (LCZ -> M1)")
        ax.set_xlabel("unit (m)")
        ax.set_title(f"DRAFT scale ladder\n{cm.label_for(first)}", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(od / cm.fname("fig_h1_ladder_DRAFT", "png", args.synthetic, first), dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
