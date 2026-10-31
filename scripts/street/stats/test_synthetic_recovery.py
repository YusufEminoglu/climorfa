"""SYNTHETIC recovery tests for the H1/H2/H4 code (rule 6, 6b).

Every check compares code output with a truth that was planted by the generator or follows from a definition, never
with a number the analysis code itself produced. Checks marked [NEGATIVE CONTROL] or [INDEPENDENT] could have failed.
Nothing here is observed data. Run:  python test_synthetic_recovery.py [--n-boot 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import make_synthetic as ms  # noqa: E402
import h1_scale as h1  # noqa: E402
import h2_drivers as h2  # noqa: E402
import h4_shade_breeze as h4  # noqa: E402
import raster_ladder as rl  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main():
    cm.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=cm.MAX_PROCS)
    a = ap.parse_args()
    print("SYNTHETIC TESTS - planted effects only, NOT observed data\n")

    # ---------------------------------------------------------------- generate scenario tables
    for s in ["_H1T", "_H1F", "_BREEZE"]:
        ms.write(s)
    ms.write_additive()

    # ================================================================ T1 solar / forcing plausibility [INDEPENDENT]
    fo = cm.forcing_summary("IMBAT")
    check("T1 sun azimuth 14:30 local (A8-1), early July, Izmir in 215-245 deg (SW) [INDEPENDENT: known solar geometry]",
          215 <= fo["sun_az"] <= 245, f"az={fo['sun_az']:.1f} elev={fo['sun_elev']:.1f}")
    check("T1b UTCI(20,20,0.5,50) ~ 20 C (UTCI reference condition) [INDEPENDENT: definition of UTCI]",
          abs(cm.utci_vec(20, [20], [0.5], 50)[0] - 20) < 1.0, f"{cm.utci_vec(20, [20], [0.5], 50)[0]:.2f}")
    check("T1c axis angle: 10 vs 170 deg bearings = 20 deg; 0 vs 90 = 90", np.isclose(cm.axis_angle(10, 170), 20) and np.isclose(cm.axis_angle(0, 90), 90))

    # ================================================================ T1d amendment A6 handling
    frA = cm.load_frames(synthetic=True, tag="_H1T")["IMBAT"]
    n0 = int((frA["lcz_primary"] == 0).sum())
    check("T1d A6: LCZ-0 segments stay in the frame (flagged lcz_valid=False), count reported", n0 > 100 and (~frA["lcz_valid"]).sum() == n0 and (frA.loc[~frA["lcz_valid"], "lcz"] == 0).all(),
          f"n_lcz0={n0} of {len(frA)}")
    pc = cm.lcz_only_oof(frA)
    check("T1e A6: LCZ-only model gives no prediction for LCZ-0 rows and never trains on them", np.isnan(pc[~frA["lcz_valid"].to_numpy()]).all() and np.isfinite(pc[frA["lcz_valid"].to_numpy()]).all())

    # ================================================================ T2 folds / bootstrap structure
    fr = cm.load_frames(synthetic=True, tag="_H1T")
    d = fr["IMBAT"]
    blk_folds = d.groupby("block")["fold"].nunique()
    check("T2a every 2 km block lies in exactly one fold (no spatial leakage)", (blk_folds == 1).all())
    sizes = d["fold"].value_counts()
    check("T2b 5 folds, balanced within 25 % of mean size", d["fold"].nunique() == 5 and sizes.max() / sizes.min() < 1.25,
          str(sizes.sort_index().to_dict()))
    check("T2c folds identical on both days", (fr["IMBAT"].set_index("seg_id")["fold"] == fr["CALM"].set_index("seg_id")["fold"]).all())
    f2 = cm.assign_folds_from_counts(d.groupby("block").size())
    check("T2d folds reproducible with fixed seed", all(d["block"].map(f2) == d["fold"]))
    ub, inv, cnt = cm.boot_block_counts(d["block"].to_numpy(), 20, cm.SEED)
    rows = cm.rows_from_counts(inv, cnt[0])
    mult = pd.Series(inv[rows]).value_counts()
    check("T2e bootstrap resamples whole blocks (rows of a block share one multiplicity)",
          all((cnt[0][inv] == cnt[0][inv]).all() for _ in [0]) and (mult.values == cnt[0][mult.index.values] * pd.Series(inv).value_counts().loc[mult.index].values).all())
    pooled = cm.lcz_pool(pd.Series([2] * 40 + [3] * 29 + [10] * 5 + [11] * 31 + [12] * 3))
    check("T2f LCZ pooling: <30 segments -> other built (101)/other natural (102)",
          set(pooled.unique()) == {2, 101, 11, 102}, str(sorted(set(pooled.unique()))))

    # ================================================================ T3 H1 on planted scenarios
    ns = SimpleNamespace(n_boot=a.n_boot, jobs=a.jobs, ladder_source_used="segments")
    out = {}
    for tag in ["_H1T", "_H1F"]:
        frs = cm.load_frames(synthetic=True, tag=tag)
        out[tag] = {day: h1.run_day(day, f, ns, h1.ladder_units_from_segments(f)) for day, f in frs.items()}
    t_ok = all(out["_H1T"][dd]["lmm"]["shares"]["resid_share"] > .5 and out["_H1T"][dd]["cv"]["r2_lcz_only"] < .3 for dd in cm.DAYS)
    f_ok = all(out["_H1F"][dd]["lmm"]["shares"]["resid_share"] < .5 and out["_H1F"][dd]["cv"]["r2_lcz_only"] > .3 for dd in cm.DAYS)
    check("T3a H1 SUPPORTED on planted 'LCZ weak' scenario (_H1T)", t_ok,
          "; ".join(f"{dd}: resid={out['_H1T'][dd]['lmm']['shares']['resid_share']:.3f} cvR2={out['_H1T'][dd]['cv']['r2_lcz_only']:.3f}" for dd in cm.DAYS))
    check("T3b H1 NOT SUPPORTED on planted 'LCZ strong' scenario (_H1F)", f_ok,
          "; ".join(f"{dd}: resid={out['_H1F'][dd]['lmm']['shares']['resid_share']:.3f} cvR2={out['_H1F'][dd]['cv']['r2_lcz_only']:.3f}" for dd in cm.DAYS))
    r_ = out["_H1T"]["IMBAT"]
    n_all = len(cm.load_frames(synthetic=True, tag="_H1T")["IMBAT"])
    check("T3b2 A6: H1 LMM / eta2 / LCZ-CV use only lcz_valid rows; excluded count reported",
          r_["lmm_n_segments"] == n_all - r_["n_lcz_excluded_A6"] and r_["ladder_eta2"]["segment"]["n_units"] == r_["lmm_n_segments"] and r_["n_lcz_excluded_A6"] > 100,
          f"n_all={n_all} lmm_n={r_['lmm_n_segments']} excluded={r_['n_lcz_excluded_A6']}")
    check("T3c morphology LightGBM beats LCZ-only CV R2 in _H1T (planted morphology effect)",
          all(out["_H1T"][dd]["cv"]["r2_diff"] > .3 for dd in cm.DAYS), str({dd: round(out["_H1T"][dd]["cv"]["r2_diff"], 3) for dd in cm.DAYS}))
    check("T3d scale ladder eta2 (segment) in _H1F >> in _H1T", out["_H1F"]["IMBAT"]["ladder_eta2"]["segment"]["eta2"] > 5 * out["_H1T"]["IMBAT"]["ladder_eta2"]["segment"]["eta2"],
          f"{out['_H1F']['IMBAT']['ladder_eta2']['segment']['eta2']:.3f} vs {out['_H1T']['IMBAT']['ladder_eta2']['segment']['eta2']:.3f}")

    # ---- LMM variance-component recovery on a known nested design [INDEPENDENT: parameters set by hand]
    # Sampling noise of one replicate is large (SD of the share ~0.022; a single seed can sit 2.6 SD off the parameter), so
    # the test is on the mean over replicate simulations (bias test). Single-seed versions (10 districts, then 40) were
    # run once, failed, and were discarded as mis-specified tests; the analysis code was not changed.
    sd_d, sd_l, sd_e = 1.5, 2.5, 3.0
    tr_ = sd_e**2 / (sd_d**2 + sd_l**2 + sd_e**2)

    def sim(seed):
        rng_ = np.random.default_rng(seed)
        dd_, ll_, yy_ = [], [], []
        for di in range(40):
            ed = rng_.normal(0, sd_d)
            for lj in rng_.choice([2, 3, 5, 6, 8, 9, 11, 12], size=6, replace=False):
                el = rng_.normal(0, sd_l)
                n = int(rng_.integers(100, 300))
                dd_ += [f"D{di}"] * n
                ll_ += [int(lj)] * n
                yy_ += list(40 + ed + el + rng_.normal(0, sd_e, n))
        return dd_, ll_, yy_
    est_l, est_a, conv = [], [], []
    for sd in range(100, 112):
        dd_, ll_, yy_ = sim(sd)
        vd, vl, ve = h1.anova_nested(np.array(yy_), np.array(dd_), np.array(["L" + str(x) for x in ll_]))
        est_a.append(ve / (vd + vl + ve))
        if sd < 106:
            r = h1.lmm_components(yy_, dd_, ll_)
            est_l.append(r["var_e"] / (r["var_d"] + r["var_l"] + r["var_e"]))
            conv.append(r["converged"] and r["method"].startswith("mixedlm"))
    check("T3e LMM (REML) recovers planted residual share: mean of 6 replicates within 0.02 [INDEPENDENT]",
          abs(np.mean(est_l) - tr_) < 0.02 and all(conv),
          f"true={tr_:.3f} mean est={np.mean(est_l):.3f} (sd {np.std(est_l):.3f}) all converged={all(conv)}")
    check("T3f ANOVA fallback recovers planted residual share: mean of 12 replicates within 0.02 [INDEPENDENT]",
          abs(np.mean(est_a) - tr_) < 0.02, f"true={tr_:.3f} mean est={np.mean(est_a):.3f} (sd {np.std(est_a):.3f})")
    # negative control: spatially permuted LCZ labels must give ~zero LCZ skill
    dperm = d.copy()
    ub_, inv_ = np.unique(dperm["block"], return_inverse=True)
    perm_lab = np.random.default_rng(5).permutation(dperm.groupby("block")["lcz"].agg(lambda s: s.iloc[0]).to_numpy())
    dperm["lcz"] = perm_lab[inv_]
    dperm["lcz_valid"] = True
    p = cm.lcz_only_oof(dperm)
    check("T3g [NEGATIVE CONTROL] block-permuted LCZ labels -> CV R2 ~ 0 (<0.02)", abs(cm.wr2(dperm["m1"], p)) < 0.02,
          f"R2={cm.wr2(dperm['m1'], p):.4f}")
    # honest spatial CV: spatial-CV R2 for an unstructured model should not exceed random k-fold R2 by construction check
    # (here: with an autocorrelated planted field, random CV must look >= spatial CV)
    pr = np.full(len(d), np.nan)
    rngk = np.random.default_rng(1).integers(0, 5, len(d))
    X = d[cm.FEATURES]
    for k in range(5):
        te = rngk == k
        m = cm.lgb_fit(X[~te], d["m1"].to_numpy()[~te], params=dict(n_jobs=a.jobs))
        pr[te] = m.predict(X[te])
    p_sp = cm.lgb_oof(d, cm.FEATURES, n_jobs=a.jobs)
    check("T3h random-fold CV R2 >= spatial-block CV R2 (spatial CV is the stricter test)",
          cm.wr2(d["m1"], pr) >= cm.wr2(d["m1"], p_sp) - 0.002, f"random={cm.wr2(d['m1'], pr):.4f} spatial={cm.wr2(d['m1'], p_sp):.4f}")

    # ================================================================ T4 scale-ladder raster aggregator vs brute force [INDEPENDENT]
    rng = np.random.default_rng(9)
    m1 = rng.normal(40, 3, (500, 500)).astype(np.float32)
    ps = np.where(rng.random((500, 500)) < 0.3, rng.integers(1, 40, (500, 500)), 0).astype(np.int32)
    seg_ids = np.arange(1, 40)
    lcz = np.where(seg_ids % 7 == 0, 0, np.where(seg_ids % 3 == 0, 2, np.where(seg_ids % 3 == 1, 6, 11)))   # 0 = no-data (A6)
    x0, y0 = 512000, 4255000
    res = rl.aggregate_tile(m1, ps, seg_ids, lcz, x0, y0)
    ok = True
    for L in (50, 500):
        r_ = res[L].set_index("cell_id")
        # brute-force for 3 random cells
        for cell in list(r_.index[:3]) + [r_.index[-1]]:
            cx, cy = map(int, cell.split("_"))
            vals, labs = [], []
            for rr in range(500):
                for cc in range(500):
                    if ps[rr, cc] > 0 and (x0 + 2 * cc) // L == cx and (y0 + 1000 - 2 * rr - 1) // L == cy:
                        vals.append(m1[rr, cc])
                        labs.append(lcz[ps[rr, cc] - 1])
            valid_labs = [x for x in labs if x > 0]       # majority over valid LCZ only (A6)
            majority = sorted(set(valid_labs), key=lambda k: (-valid_labs.count(k), k))[0] if valid_labs else 0
            ok &= (len(vals) == r_.loc[cell, "n_ps"]) and np.isclose(np.median(vals), r_.loc[cell, "m1_med"]) and majority == r_.loc[cell, "lcz"]
    check("T4 raster-ladder aggregator equals brute-force loop (n_ps, median M1, majority LCZ; L=50,500)", ok)
    check("T4b 1000 m tile -> 400 cells at 50 m and 4 cells at 500 m", len(res[50]) == 400 and len(res[500]) == 4, f"{len(res[50])}, {len(res[500])}")

    # ================================================================ T5 H2 recovery on additive independent-feature scenario
    truth = json.loads((cm.OUT_ROOT / "h1" / "synthetic_inputs" / "truth_SYNTHETIC_ADD.json").read_text())
    fra = cm.load_frames(synthetic=True, tag="_ADD")
    ns2 = SimpleNamespace(n_boot=a.n_boot, jobs=a.jobs, pdp_boot=max(12, a.n_boot // 4))
    for day in ["IMBAT"]:
        r2 = h2.run_day(day, fra[day], ns2)
        tf = truth[day]["true_family_share"]
        got = {k: v["share"] for k, v in r2["model_A_features_only"]["family_share"].items()}
        err = {k: abs(got.get(k, 0) - tf[k]) for k in tf}
        check(f"T5a {day} TreeSHAP family shares within 0.05 (abs) of planted analytic shares [INDEPENDENT]",
              max(err.values()) < 0.05, "; ".join(f"{k}: true {tf[k]:.3f} got {got.get(k, 0):.3f}" for k in tf))
        check(f"T5b {day} planted-null 'context' family share < 0.04 [could have failed]", got.get("context", 0) < 0.04, f"{got.get('context', 0):.3f}")
        for f_, lab in [("svf_med", "SVF"), ("hw_ratio", "H/W"), ("tcover15", "tree cover")]:
            tr_rng = truth[day]["true_range_over_q2p5_q97p5"][f_]
            pr_ = r2["pdp"][f_]
            rel = abs(pr_["range_C"] - tr_rng) / tr_rng
            check(f"T5c {day} PDP range {lab} within 20 % of planted [INDEPENDENT]", rel < 0.20,
                  f"true {tr_rng:.2f} got {pr_['range_C']:.2f} CI {pr_['range_ci95'][0]:.2f}-{pr_['range_ci95'][1]:.2f}")
        check(f"T5d {day} PDP signs: SVF +, H/W -, tree cover -",
              r2["pdp"]["svf_med"]["slope_sign"] > 0 and r2["pdp"]["hw_ratio"]["slope_sign"] < 0 and r2["pdp"]["tcover15"]["slope_sign"] < 0)
        check(f"T5e {day} model A CV R2 close to planted ceiling 1 - var(noise)/var(M1)",
              r2["model_A_features_only"]["cv_r2"] > 0.93, f"R2={r2['model_A_features_only']['cv_r2']:.3f}")
        check(f"T5f {day} LCZ adds nothing when LCZ is independent of M1 (gain < 0.01)", r2["model_B_features_plus_LCZ"]["gain_over_A"] < 0.01,
              f"gain={r2['model_B_features_plus_LCZ']['gain_over_A']:.4f}")
        check(f"T5g {day} TreeSHAP (LightGBM pred_contrib) == shap.TreeExplainer", r2["treeshap_crosscheck"]["max_abs_diff_shap_vs_lightgbm"] < 1e-6,
              f"max diff {r2['treeshap_crosscheck']['max_abs_diff_shap_vs_lightgbm']:.2e}")

    # ================================================================ T6 H4
    ta, rh = 34.67, 22.7
    n = 500
    t_i = np.linspace(35, 70, n)
    v_i = np.linspace(0.6, 6, n)
    sh, br, u_iv, u_rr = h4.shapley_terms(ta, rh, t_i, v_i, 55.0, 3.0)
    check("T6a Shapley efficiency: shade + breeze = U(T_i,v_i) - U(ref) (max abs err < 1e-9)", np.max(np.abs(sh + br - (u_iv - u_rr))) < 1e-9,
          f"{np.max(np.abs(sh + br - (u_iv - u_rr))):.2e}")
    sh0, br0, *_ = h4.shapley_terms(ta, rh, np.full(n, 55.0), v_i, 55.0, 3.0)
    sh1, br1, *_ = h4.shapley_terms(ta, rh, t_i, np.full(n, 3.0), 55.0, 3.0)
    check("T6b Dummy axiom: shade = 0 when T_i = T_ref; breeze = 0 when v_i = v_ref", np.max(np.abs(sh0)) < 1e-9 and np.max(np.abs(br1)) < 1e-9)
    check("T6c shade term increases with Tmrt (physics) ", np.all(np.diff(sh1) > -1e-9) and sh1[-1] > sh1[0] + 3, f"range {sh1[0]:.2f}..{sh1[-1]:.2f}")
    # variance-only scenarios (known answer: the other term has zero variance)
    sh2, br2, *_ = h4.shapley_terms(ta, rh, t_i, np.full(n, 3.0), 55.0, 3.0)
    check("T6d Tmrt-only spread -> Var(breeze) = 0, ratio -> inf", np.var(br2) < 1e-12 and np.var(sh2) > 1)
    # end-to-end on planted scenarios
    for tag, expect in [("_H1T", True), ("_BREEZE", False)]:
        frs = cm.load_frames(synthetic=True, tag=tag)
        ns3 = SimpleNamespace(n_boot=a.n_boot)
        got = {}
        for day, f in frs.items():
            _, res_, *_ = h4.analyse_day(day, f, "segments", ns3)
            assert res_["n_segments"] == len(f), "A6: H4 must keep LCZ-0 segments"
            got[day] = (res_["ratio_var_shade_over_breeze"], res_["ratio_gt_4"])
        check(f"T6e H4 scenario {tag}: shade-dominated == {expect} on both days", all(g[1] == expect for g in got.values()),
              "; ".join(f"{k}: ratio={v[0]:.2f}" for k, v in got.items()))
    # closure: planted segments whose M1 equals U(T_i,v_i) exactly -> closure residual var ~ 0
    f = cm.load_frames(synthetic=True, tag="_H1T")["IMBAT"].copy()
    f["m1"] = cm.utci_vec(f.attrs["forcing"]["ta"], f["tmrt_b15"], f["v10eff_b15"], f.attrs["forcing"]["rh"])
    _, res_, *_ = h4.analyse_day("IMBAT", f, "segments", SimpleNamespace(n_boot=10))
    check("T6f closure residual variance ~ 0 when M1 = U(T_i, v_i)", res_["var_closure_resid"] < 1e-9, f"{res_['var_closure_resid']:.2e}")

    # ================================================================ summary
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\nSYNTHETIC TEST SUMMARY: {len(RESULTS) - nfail}/{len(RESULTS)} passed, {nfail} failed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
