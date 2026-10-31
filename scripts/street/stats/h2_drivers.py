"""H2 Mechanisms (pre_analysis_lock.md section 2, H2). Implemented as written.

LightGBM on segments, M1 per day, same 5-fold spatial-block folds as H1 (2 km blocks, seed 20261030).
  Model A: street features only (5 families). Model B (separate): features + LCZ as categorical.
  * out-of-fold CV R2 (+ 95 % block-bootstrap CI over the fixed out-of-fold predictions)
  * TreeSHAP family shares = share of total mean|SHAP| per family, out-of-fold SHAP (LightGBM pred_contrib), CI by
    2 km block bootstrap of the stored SHAP matrix
  * partial dependence for H/W, SVF, tree cover and orientation (azimuth, joint over az_sin/az_cos/ang_sun/ang_wind);
    95 % CI = percentile band from 1000 block-bootstrap REFITS of model A
  * within-LCZ residual explained: R2 (spatial CV) of a features-only model on M1 minus its (training-fold) LCZ mean
Lock gives no decision threshold for H2; results are reported only.
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

PDP_FEATS = {"hw_ratio": "H/W", "svf_med": "SVF", "tcover15": "tree cover (15 m)"}
N_GRID = 25
PDP_ROWS = 2000


def family_shares(shap_abs_mean: np.ndarray, feats: list[str], extra_family: dict | None = None) -> dict:
    fam = {}
    for j, f in enumerate(feats):
        k = (extra_family or {}).get(f) or cm.FAMILY_OF[f]
        fam[k] = fam.get(k, 0.0) + shap_abs_mean[j]
    tot = sum(fam.values())
    return {k: v / tot for k, v in fam.items()}


def oof_with_shap(df, feats, categorical=None, jobs=1):
    pred, sh = cm.lgb_oof(df, feats, categorical=categorical, contrib=True, n_jobs=jobs)
    return pred, sh[:, :-1]     # drop expected-value column


def residual_model_r2(df, feats, jobs=1):
    """R2 of features-only model on M1 minus training-fold LCZ mean, spatial CV. Returns (R2, oof_pred, target)."""
    y = df["m1"].to_numpy(float)
    tgt = np.full(len(df), np.nan)
    pred = np.full(len(df), np.nan)
    X = df[feats]
    for k in range(cm.N_FOLDS):
        te = (df["fold"] == k).to_numpy()
        tr = ~te
        gm = df.loc[tr].groupby("lcz")["m1"].mean()
        mu = df["lcz"].map(gm).fillna(df.loc[tr, "m1"].mean()).to_numpy()
        r = y - mu
        tgt[te] = r[te]
        m = cm.lgb_fit(X[tr], r[tr], params=dict(n_jobs=jobs))
        pred[te] = m.predict(X[te])
    return pred, tgt


def orientation_grid(az_grid, fo):
    sun, wnd = fo["sun_az"], fo["wd_mean_15_17"]
    return dict(az_sin=np.sin(np.deg2rad(az_grid)), az_cos=np.cos(np.deg2rad(az_grid)),
                ang_sun=cm.axis_angle(az_grid, sun), ang_wind=cm.axis_angle(az_grid, wnd))


def pdp_curves(model, Xeval: pd.DataFrame, grids: dict, fo) -> dict:
    """grids: {name: grid values}. Returns {name: mean prediction curve}. 'orientation' sets four columns jointly."""
    out = {}
    for name, g in grids.items():
        vals = []
        for v in g:
            Xt = Xeval.copy()
            if name == "orientation":
                for c, arr in orientation_grid(np.array([v]), fo).items():
                    Xt[c] = float(arr[0])
            else:
                Xt[name] = v
            vals.append(float(np.mean(model.predict(Xt))))
        out[name] = np.array(vals)
    return out


def _boot_pdp(args):
    Xn, y, cols, rows, Xeval, grids, fo = args
    X = pd.DataFrame(Xn[rows], columns=cols)
    m = cm.lgb_fit(X, y[rows], params=dict(n_jobs=1))
    cur = pdp_curves(m, Xeval, grids, fo)
    return {k: v for k, v in cur.items()}


def run_day(day, df, args):
    t0 = time.time()
    fo = df.attrs["forcing"]
    jobs = min(args.jobs, cm.MAX_PROCS)
    y = df["m1"].to_numpy(float)
    feats = cm.FEATURES
    res = dict(day=day, label=cm.label_for(df), n_segments=len(df), forcing=fo)
    ub, inv, counts = cm.boot_block_counts(df["block"].to_numpy(), cm.N_BOOT, cm.SEED + 2)
    W = np.array([counts[r][inv] for r in range(args.n_boot)], dtype=np.float32)    # (n_boot, n)

    # --- Model A (features) -------------------------------------------------------------------------
    pa, sa = oof_with_shap(df, feats, jobs=jobs)
    r2a = cm.wr2(y, pa)
    b = np.array([cm.wr2(y, pa, w) for w in W])
    res["model_A_features_only"] = dict(cv_r2=r2a, cv_r2_ci95=cm.ci95(b))
    absA = np.abs(sa)
    fam_pt = family_shares(absA.mean(axis=0), feats)
    fam_boot = {k: [] for k in fam_pt}
    for w in W:
        fam = family_shares((absA * w[:, None]).sum(axis=0) / w.sum(), feats)
        for k in fam_boot:
            fam_boot[k].append(fam[k])
    res["model_A_features_only"]["family_share"] = {k: dict(share=fam_pt[k], ci95=cm.ci95(fam_boot[k])) for k in fam_pt}
    res["model_A_features_only"]["feature_mean_abs_shap"] = dict(zip(feats, absA.mean(axis=0).tolist()))

    # --- LCZ-based models: only lcz_valid rows (Amendment A6: LCZ 0 = raster no-data is not a class) -------------------
    v = df["lcz_valid"].to_numpy()
    res["n_lcz_excluded_A6"] = int((~v).sum())
    res["n_segments_lcz_models"] = int(v.sum())
    dfv = df[v].reset_index(drop=True)
    Wv = W[:, v]
    yv = y[v]
    pav = cm.lgb_oof(dfv, feats, n_jobs=jobs)            # model A on the same rows, for a like-for-like gain
    r2av = cm.wr2(yv, pav)
    bav = np.array([cm.wr2(yv, pav, w) for w in Wv])
    res["model_A_features_only"]["cv_r2_on_lcz_valid_rows"] = r2av
    res["model_A_features_only"]["cv_r2_on_lcz_valid_rows_ci95"] = cm.ci95(bav)
    # --- Model B (+ LCZ categorical) ----------------------------------------------------------------
    featsB = feats + ["lcz"]
    pb, sb = oof_with_shap(dfv, featsB, categorical=["lcz"], jobs=jobs)
    r2b = cm.wr2(yv, pb)
    bb = np.array([cm.wr2(yv, pb, w) for w in Wv])
    absB = np.abs(sb)
    famB = family_shares(absB.mean(axis=0), featsB, extra_family={"lcz": "lcz"})
    famB_boot = {k: [] for k in famB}
    for w in Wv:
        fam = family_shares((absB * w[:, None]).sum(axis=0) / w.sum(), featsB, extra_family={"lcz": "lcz"})
        for k in famB_boot:
            famB_boot[k].append(fam[k])
    res["model_B_features_plus_LCZ"] = dict(
        cv_r2=r2b, cv_r2_ci95=cm.ci95(bb), gain_over_A=r2b - r2av, gain_over_A_ci95=cm.ci95(bb - bav),
        gain_note="vs model A refit on the same lcz_valid rows",
        family_share={k: dict(share=famB[k], ci95=cm.ci95(famB_boot[k])) for k in famB})
    # LCZ-only reference
    pl = cm.lcz_only_oof(dfv)
    res["lcz_only_reference"] = dict(cv_r2=cm.wr2(yv, pl), cv_r2_ci95=cm.ci95([cm.wr2(yv, pl, w) for w in Wv]))

    # --- within-LCZ residual explained -------------------------------------------------------------
    pr, tg = residual_model_r2(dfv, feats, jobs=jobs)
    rr = cm.wr2(tg, pr)
    res["within_lcz_residual"] = dict(
        r2=rr, r2_ci95=cm.ci95([cm.wr2(tg, pr, w) for w in Wv]),
        residual_target_variance_share_of_M1=float(np.var(tg) / np.var(yv)),
        note="features-only LightGBM on M1 minus training-fold LCZ mean; spatial-block CV; CI resamples OOF predictions")

    # --- independent check of TreeSHAP: shap.TreeExplainer on the final model, 1500 rows ---------------
    X = df[feats]
    m_all = cm.lgb_fit(X, y, params=dict(n_jobs=jobs))
    import shap
    rng = np.random.default_rng(cm.SEED)
    ix = rng.choice(len(df), min(1500, len(df)), replace=False)
    sv = shap.TreeExplainer(m_all).shap_values(X.iloc[ix])
    lg = m_all.predict(X.iloc[ix], pred_contrib=True)[:, :-1]
    res["treeshap_crosscheck"] = dict(max_abs_diff_shap_vs_lightgbm=float(np.max(np.abs(sv - lg))),
                                      corr=float(np.corrcoef(sv.ravel(), lg.ravel())[0, 1]))

    # --- partial dependence --------------------------------------------------------------------------
    grids = {}
    for f in PDP_FEATS:
        grids[f] = np.nanquantile(df[f].to_numpy(float), np.linspace(0.025, 0.975, N_GRID))
    grids["orientation"] = np.linspace(0, 170, 18)
    Xeval = X.iloc[rng.choice(len(df), min(PDP_ROWS, len(df)), replace=False)].reset_index(drop=True)
    pdp0 = pdp_curves(m_all, Xeval, grids, fo)
    from joblib import Parallel, delayed
    Xn = X.to_numpy(float)
    nb = args.pdp_boot
    jobsl = [(Xn, y, feats, cm.rows_from_counts(inv, counts[r]), Xeval, grids, fo) for r in range(nb)]
    bp = Parallel(n_jobs=jobs, backend="loky")(delayed(_boot_pdp)(j) for j in jobsl)
    pdp = {}
    for k in grids:
        arr = np.array([b_[k] for b_ in bp])
        lo, hi = np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
        rng_ = arr.max(axis=1) - arr.min(axis=1)
        pdp[k] = dict(grid=grids[k].tolist(), pdp=pdp0[k].tolist(), lo=lo.tolist(), hi=hi.tolist(),
                      range_C=float(pdp0[k].max() - pdp0[k].min()), range_ci95=cm.ci95(rng_),
                      range_ci95_basic=[float(2 * (pdp0[k].max() - pdp0[k].min()) - cm.ci95(rng_)[1]), float(2 * (pdp0[k].max() - pdp0[k].min()) - cm.ci95(rng_)[0])],
                      slope_sign=float(np.sign(np.polyfit(grids[k], pdp0[k], 1)[0])), n_boot=nb)
    res["pdp"] = pdp
    res["runtime_s"] = time.time() - t0
    return res


def main():
    cm.utf8_stdout()
    ap = argparse.ArgumentParser()
    cm.add_common_args(ap)
    ap.add_argument("--pdp-boot", type=int, default=cm.N_BOOT, help="bootstrap REFITS for PDP bands (lock: 1000)")
    args = ap.parse_args()
    frames = cm.frames_from_args(args)
    od = cm.out_dir("h2", args.synthetic, args.tag, args.out_suffix)
    results = {d: run_day(d, f, args) for d, f in frames.items()}
    first = next(iter(frames.values()))
    cm.write_json(od / cm.fname("h2_results", "json", args.synthetic, first),
                  dict(constants=cm.constants_dict(), results=results))
    print(f"\n=== H2 mechanisms  [{cm.label_for(first)}] (no decision rule in the lock; reported only) ===")
    for d, r in results.items():
        A, B = r["model_A_features_only"], r["model_B_features_plus_LCZ"]
        print(f"{d}: n={r['n_segments']}")
        print("   CV R2 A (features)   = " + cm.fmt_ci(A["cv_r2"], *A["cv_r2_ci95"]))
        print(f"   (LCZ models on {r['n_segments_lcz_models']} lcz_valid rows; {r['n_lcz_excluded_A6']} LCZ-0 segments excluded, A6)")
        print("   CV R2 B (+LCZ)       = " + cm.fmt_ci(B["cv_r2"], *B["cv_r2_ci95"]))
        print("   CV R2 LCZ-only       = " + cm.fmt_ci(r["lcz_only_reference"]["cv_r2"], *r["lcz_only_reference"]["cv_r2_ci95"]))
        print("   within-LCZ residual R2 = " + cm.fmt_ci(r["within_lcz_residual"]["r2"], *r["within_lcz_residual"]["r2_ci95"]))
        print("   TreeSHAP family shares (model A): " + "; ".join(
            f"{k} {v['share']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]" for k, v in
            sorted(A["family_share"].items(), key=lambda kv: -kv[1]["share"])))
        print("   TreeSHAP family shares (model B): " + "; ".join(
            f"{k} {v['share']:.3f}" for k, v in sorted(B["family_share"].items(), key=lambda kv: -kv[1]["share"])))
        print(f"   TreeSHAP crosscheck vs shap pkg: max|diff|={r['treeshap_crosscheck']['max_abs_diff_shap_vs_lightgbm']:.2e}")
        print("   PDP range (degC, 95% CI): " + "; ".join(
            f"{k} {v['range_C']:.2f} [{v['range_ci95'][0]:.2f},{v['range_ci95'][1]:.2f}] slope {'+' if v['slope_sign'] > 0 else '-'}"
            for k, v in r["pdp"].items()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(results), 4, figsize=(13, 3.2 * len(results)), squeeze=False)
        for i, (d, r) in enumerate(results.items()):
            for j, (k, v) in enumerate(r["pdp"].items()):
                ax = axs[i, j]
                ax.fill_between(v["grid"], v["lo"], v["hi"], alpha=.25)
                ax.plot(v["grid"], v["pdp"])
                ax.set_title(f"{d} {k}", fontsize=8)
                ax.set_ylabel("UTCI (degC)")
        fig.suptitle(f"DRAFT partial dependence, {cm.label_for(first)}", fontsize=9)
        fig.tight_layout()
        fig.savefig(od / cm.fname("fig_h2_pdp_DRAFT", "png", args.synthetic, first), dpi=130)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
