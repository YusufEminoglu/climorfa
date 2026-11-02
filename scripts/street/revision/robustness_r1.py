"""Revision R1: post-hoc, descriptive robustness quantities for H1, H3 and H4 (not part of the registered analyses).

Read-only on the frozen inputs and results. Reuses the registered helpers in scripts/street/stats (folds, block bootstrap,
LightGBM settings, UTCI call, Shapley terms, LMM) without modifying them.

Run from the project root:
    python -m scripts.street.revision.robustness_r1
Outputs: outputs/street/analysis/revision_r1/robustness_r1.json and short CSVs.

Sections
  0  reproduction check (LCZ-only and morphology CV R2, Var(S)/Var(B))
  A1 LCZ class mean M1 with 95 % block-bootstrap CI, largest and segment-weighted typical between-class gap
  A2 CV R2 of single continuous predictors (SVF, tree cover, building cover)
  A3 100 m support benchmark (length-weighted 100 m aggregates of the H2 street features; alone and with LCZ)
  A4 cool cluster (segments > 2 C below the day median): variance share, CV R2 with those segments excluded
  A5 tipping point for an LCZ-structured air-temperature contrast (assumed contrast vector)
  B  H3 extras: Spearman rho at 250 m cells; decoupled share with a 1 C minimum separation vs permutation null
  C  H4 extras: Cov(S,B) and Var(S+B) shares; wind-dispersion sensitivity; alternative reference winds
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "scripts" / "street" / "stats"
if str(STATS) not in sys.path:
    sys.path.insert(0, str(STATS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common as cm  # noqa: E402  (registered helpers, read-only)
import h1_scale as h1  # noqa: E402
import h4_shade_breeze as h4  # noqa: E402

FINAL = ROOT / "data/street/analysis/final_20261030"
OUT = ROOT / "outputs/street/analysis/revision_r1"
H1_JSON = ROOT / "outputs/street/analysis/h1/results/h1_results.json"
H4_JSON = ROOT / "outputs/street/analysis/h4/results/h4_results.json"
H3_DIR = ROOT / "outputs/street/analysis/h3"
JOBS = cm.MAX_PROCS

NATURAL = set(range(11, 18)) | {cm.POOL_NAT}
# A5 assumption: air-temperature contrast per LCZ class in K per unit k (built: compact > open > sparse; natural 0)
CONTRAST = {1: 1.0, 2: 1.0, 3: 1.0, 10: 1.0, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.25}
SINGLE = {"svf_med": "sky view factor", "tcover15": "tree cover (15 m)", "bcover30": "building cover (30 m)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def boot_weights(blocks, seed):
    ub, inv, counts = cm.boot_block_counts(np.asarray(blocks), cm.N_BOOT, seed)
    return inv, counts


def r2_ci(y, p, inv, counts, base_w):
    pt = cm.wr2(y, p, base_w)
    bs = np.array([cm.wr2(y, p, counts[r][inv] * base_w) for r in range(counts.shape[0])])
    return dict(r2=pt, ci95=cm.ci95(bs))


def lcz_cv(df, y=None):
    d = df if y is None else df.assign(m1=y)
    p = cm.lcz_only_oof(d)
    vf = d["lcz_valid"].to_numpy(float)
    return cm.wr2(d["m1"].to_numpy(float), np.nan_to_num(p), vf), p


def weighted_median(x, w):
    return h4.weighted_median(x, w)


# ----------------------------------------------------------------------------------------------
# 0  reproduction
# ----------------------------------------------------------------------------------------------
def reproduce(frames, h4ref):
    out = {}
    for day, df in frames.items():
        y = df["m1"].to_numpy(float)
        vf = df["lcz_valid"].to_numpy(float)
        r_lcz, p_lcz = lcz_cv(df)
        p_mor = cm.lgb_oof(df, cm.FEATURES, n_jobs=JOBS)
        r_mor = cm.wr2(y, p_mor, vf)
        df.attrs["p_mor"] = p_mor
        df.attrs["p_lcz"] = p_lcz
        fo = df.attrs["forcing"]
        d = df.dropna(subset=["tmrt_b15", "v10eff_b15", "m1"]).reset_index(drop=True)
        t_ref, v_ref = h4ref[day]["tmrt_ref"], h4ref[day]["v_ref"]
        s, b, _, _ = h4.shapley_terms(fo["ta"], fo["rh"], d["tmrt_b15"].to_numpy(float), d["v10eff_b15"].to_numpy(float),
                                      t_ref, v_ref)
        ratio = np.var(s) / np.var(b)
        out[day] = dict(n_segments=len(df), n_lcz_valid=int(vf.sum()), cv_r2_lcz_only=r_lcz, cv_r2_morphology=r_mor,
                        var_S=float(np.var(s)), var_B=float(np.var(b)), ratio_varS_varB=float(ratio),
                        h4_reference=dict(tmrt_ref=t_ref, v_ref=v_ref, ta=fo["ta"], rh=fo["rh"]))
        log(f"[repro] {day}: LCZ-only {r_lcz:.4f}  morph {r_mor:.4f}  ratio {ratio:.2f}")
    return out


# ----------------------------------------------------------------------------------------------
# A1 class means and gaps
# ----------------------------------------------------------------------------------------------
def class_means(df):
    v = df["lcz_valid"].to_numpy()
    d = df[v].reset_index(drop=True)
    cnt = d["lcz_primary"].value_counts()
    keep = sorted(int(k) for k in cnt[cnt >= cm.MIN_CLASS_N].index)
    d = d[d["lcz_primary"].isin(keep)].reset_index(drop=True)
    y = d["m1"].to_numpy(float)
    g = d["lcz_primary"].to_numpy(int)
    codes = np.array(keep)
    gi = np.searchsorted(codes, g)
    inv, counts = boot_weights(d["block"].to_numpy(), cm.SEED)

    def stats(w):
        sw = np.bincount(gi, weights=w, minlength=len(codes))
        sy = np.bincount(gi, weights=w * y, minlength=len(codes))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = sy / sw
        ok = sw > 0
        m = mu[ok]
        nn = sw[ok]
        largest = float(m.max() - m.min())
        D = np.abs(m[:, None] - m[None, :])
        W = nn[:, None] * nn[None, :]
        np.fill_diagonal(W, 0.0)
        typical = float((D * W).sum() / W.sum())         # segment-pair-weighted mean |gap| between different classes
        # weighted median of pairwise gaps
        iu = np.triu_indices(len(m), 1)
        typ_med = float(weighted_median(D[iu], W[iu]))
        sd_between = float(np.sqrt(np.sum(nn * (m - np.sum(nn * m) / nn.sum()) ** 2) / nn.sum()))
        return mu, largest, typical, typ_med, sd_between

    mu, largest, typical, typ_med, sdb = stats(np.ones(len(d)))
    B = [stats(counts[r][inv].astype(float)) for r in range(counts.shape[0])]
    mus = np.array([b[0] for b in B])
    rows = []
    for j, c in enumerate(codes):
        lo, hi = cm.ci95(mus[:, j])
        rows.append(dict(lcz=int(c), n_segments=int((g == c).sum()), mean_m1=float(mu[j]), ci95_lo=lo, ci95_hi=hi,
                         diff_from_city_mean=float(mu[j] - y.mean())))
    hottest, coolest = codes[np.argmax(mu)], codes[np.argmin(mu)]
    return dict(
        classes=rows, n_segments_used=len(d), city_mean_m1=float(y.mean()), sd_m1=float(y.std()),
        largest_gap=dict(value=largest, ci95=cm.ci95([b[1] for b in B]), hottest_class=int(hottest), coolest_class=int(coolest)),
        typical_gap_segment_weighted_mean=dict(value=typical, ci95=cm.ci95([b[2] for b in B])),
        typical_gap_segment_weighted_median=dict(value=typ_med, ci95=cm.ci95([b[3] for b in B])),
        between_class_sd_segment_weighted=dict(value=sdb, ci95=cm.ci95([b[4] for b in B])),
        note=("Unpooled lcz_primary classes with >= 30 segments (LCZ 0 excluded, A6). Typical gap = mean |mu_a - mu_b| over "
              "all pairs of segments in different classes (weights n_a n_b); also its weighted median. CI: 2 km block "
              "bootstrap, 1000 resamples, seed 20261030 (class means recomputed per resample)."))


# ----------------------------------------------------------------------------------------------
# A2 single predictors
# ----------------------------------------------------------------------------------------------
def single_predictors(df):
    y = df["m1"].to_numpy(float)
    vf = df["lcz_valid"].to_numpy(float)
    inv, counts = boot_weights(df["block"].to_numpy(), cm.SEED)
    out = {}
    for f, name in SINGLE.items():
        p = cm.lgb_oof(df, [f], n_jobs=JOBS)
        r = r2_ci(y, p, inv, counts, vf)
        r["r2_all_rows"] = cm.wr2(y, p)
        r["label"] = name
        rho = pd.Series(df[f]).corr(pd.Series(y), method="spearman")
        r["spearman_rho_with_m1"] = float(rho)
        out[f] = r
        log(f"[A2] {df.attrs['day']} {f}: {r['r2']:.4f}")
    return out


# ----------------------------------------------------------------------------------------------
# A3 100 m support benchmark
# ----------------------------------------------------------------------------------------------
def grid100_assignment():
    import geopandas as gpd
    seg = gpd.read_file(FINAL / "segments.gpkg", layer="segments", columns=["seg_id", "length_m"])
    grid = gpd.read_file(FINAL / "grid_100.gpkg", layer="grid", columns=["g100_id"])
    pts = gpd.GeoDataFrame(seg[["seg_id", "length_m"]], geometry=seg.geometry.centroid, crs=seg.crs)
    j = gpd.sjoin(pts, grid[["g100_id", "geometry"]], how="left", predicate="intersects")
    j = j.sort_values(["seg_id", "g100_id"]).drop_duplicates("seg_id")
    return pd.DataFrame(j[["seg_id", "length_m", "g100_id"]]).rename(columns={"g100_id": "cell100"})


def agg100(df, asg):
    d = df.drop(columns=[c for c in ("length_m",) if c in df]).merge(asg, on="seg_id", how="left")
    if "length_m" not in d:
        raise RuntimeError("length_m missing")
    feats = cm.FEATURES
    w = d["length_m"].to_numpy(float)
    agg = {}
    key = d["cell100"]
    for f in feats:
        x = d[f].to_numpy(float)
        ok = np.isfinite(x) & key.notna().to_numpy()
        t = pd.DataFrame({"k": key[ok].to_numpy(), "wx": (w * x)[ok], "w": w[ok]}).groupby("k").sum()
        agg[f] = t["wx"] / t["w"]
    A = pd.DataFrame(agg)
    A.columns = [f"c100_{f}" for f in feats]
    d = d.merge(A, left_on="cell100", right_index=True, how="left")
    return d


def support_benchmark(df, asg):
    d = agg100(df, asg)
    assert len(d) == len(df) and (d["seg_id"].to_numpy() == df["seg_id"].to_numpy()).all()
    cf = [f"c100_{f}" for f in cm.FEATURES]
    y = d["m1"].to_numpy(float)
    v = d["lcz_valid"].to_numpy()
    inv, counts = boot_weights(d["block"].to_numpy(), cm.SEED)
    pa = cm.lgb_oof(d, cf, n_jobs=JOBS)
    ra = r2_ci(y, pa, inv, counts, v.astype(float))
    ra["r2_all_rows"] = cm.wr2(y, pa)
    dv = d[v].reset_index(drop=True)
    dv["lcz"] = dv["lcz"].astype(int)
    pb = cm.lgb_oof(dv, cf + ["lcz"], categorical=["lcz"], n_jobs=JOBS)
    invv, countsv = boot_weights(dv["block"].to_numpy(), cm.SEED)
    rb = r2_ci(dv["m1"].to_numpy(float), pb, invv, countsv, np.ones(len(dv)))
    n_cells = int(d["cell100"].nunique())
    seg_per_cell = d.groupby("cell100").size()
    log(f"[A3] {df.attrs['day']}: 100m alone {ra['r2']:.4f}  LCZ+100m {rb['r2']:.4f}")
    return dict(r2_100m_aggregates_only=ra, r2_lcz_plus_100m_aggregates=rb, n_cells=n_cells,
                n_segments_without_cell=int(d["cell100"].isna().sum()),
                segments_per_cell_median=float(seg_per_cell.median()), segments_per_cell_mean=float(seg_per_cell.mean()),
                note=("Each segment receives the length-weighted mean of the 14 H2 street features over all segments whose "
                      "line centroid lies in the same 100 m cell (NaN-aware per feature). Same LightGBM settings and 2 km "
                      "folds; R2 on lcz_valid rows; CI resamples fixed out-of-fold predictions."))


# ----------------------------------------------------------------------------------------------
# A4 cool cluster
# ----------------------------------------------------------------------------------------------
def cool_cluster(df):
    y = df["m1"].to_numpy(float)
    med = float(np.median(y))
    cool = y < med - 2.0
    ss = (y - y.mean()) ** 2
    share_var = float(ss[cool].sum() / ss.sum())
    d = df[~cool].reset_index(drop=True)
    yd = d["m1"].to_numpy(float)
    vf = d["lcz_valid"].to_numpy(float)
    inv, counts = boot_weights(d["block"].to_numpy(), cm.SEED)
    _, pl = lcz_cv(d)
    rl = r2_ci(yd, np.nan_to_num(pl), inv, counts, vf)
    pm = cm.lgb_oof(d, cm.FEATURES, n_jobs=JOBS)
    rm = r2_ci(yd, pm, inv, counts, vf)
    lcz_mix = df.loc[cool & df["lcz_valid"].to_numpy(), "lcz_primary"].value_counts().sort_index().to_dict()
    log(f"[A4] {df.attrs['day']}: cool n={cool.sum()} var share {share_var:.3f}  LCZ {rl['r2']:.4f} morph {rm['r2']:.4f}")
    return dict(day_median_m1=med, threshold=med - 2.0, n_cool=int(cool.sum()), share_segments=float(cool.mean()),
                share_of_total_m1_variance=share_var, var_m1_all=float(np.var(y)), var_m1_excluding=float(np.var(yd)),
                mean_m1_cool=float(y[cool].mean()), cool_lcz_counts=lcz_mix,
                excluded=dict(n_segments=len(d), cv_r2_lcz_only=rl, cv_r2_morphology=rm),
                note=("Variance share = sum over cool segments of (M1 - mean)^2 / total sum of squares (deviations from the "
                      "full-sample mean). Models refit on the remaining segments with the same fold labels."))


# ----------------------------------------------------------------------------------------------
# A5 tipping point
# ----------------------------------------------------------------------------------------------
def dutci_dta(ta, t_ref, v_ref, rh, h=0.5):
    up = float(cm.utci_vec(ta + h, [t_ref], [v_ref], rh)[0])
    dn = float(cm.utci_vec(ta - h, [t_ref], [v_ref], rh)[0])
    return (up - dn) / (2 * h)


def tipping(df, ref):
    from scipy.optimize import brentq
    fo = df.attrs["forcing"]
    g = dutci_dta(fo["ta"], ref["tmrt_ref"], ref["v_ref"], fo["rh"])
    lab = df["lcz_primary"].to_numpy(int)
    c = np.array([CONTRAST.get(int(k), 0.0) for k in lab])
    c[~df["lcz_valid"].to_numpy()] = 0.0
    y0 = df["m1"].to_numpy(float)
    v = df["lcz_valid"].to_numpy()

    def cv_at(k):
        return lcz_cv(df, y0 + k * c * g)[0]

    dl = df[v].reset_index(drop=True)
    yl0, cl = y0[v], c[v]
    dist, lz = dl["district"].to_numpy(), dl["lcz"].to_numpy()
    cache = {}

    def resid_at(k):
        k = round(float(k), 4)
        if k not in cache:
            r = h1.lmm_components(yl0 + k * cl * g, dist, lz)
            tot = r["var_d"] + r["var_l"] + r["var_e"]
            cache[k] = (r["var_e"] / tot, r["method"], r["converged"])
        return cache[k][0]

    def solve(f, target, increasing, kmax=0.5):
        """Smallest bracket [lo, kmax] (doubling) with f crossing target, then Brent root."""
        lo = 0.0

        def not_reached(k):
            return f(k) < target if increasing else f(k) > target
        while not_reached(kmax):
            lo, kmax = kmax, kmax * 2
            if kmax > 64:
                return float("nan")
        return float(brentq(lambda k: f(k) - target, lo, kmax, xtol=1e-3))

    k_cv = solve(cv_at, 0.30, increasing=True)
    log(f"[A5] {df.attrs['day']}: dU/dTa={g:.3f}  k(CV=0.30)={k_cv:.3f}")
    k_res = solve(resid_at, 0.50, increasing=False)
    log(f"[A5] {df.attrs['day']}: k(resid=0.50)={k_res:.3f}")
    return _tip_out(df, fo, ref, g, cv_at, resid_at, cache, k_cv, k_res)


def _tip_out(df, fo, ref, g, cv_at, resid_at, cache, k_cv, k_res):
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    curve = [dict(k=k, cv_r2_lcz_only=cv_at(k), utci_offset_compact_C=k * g) for k in grid]
    return dict(dUTCI_dTa_at_reference=g, reference_state=dict(ta=fo["ta"], rh=fo["rh"], tmrt=ref["tmrt_ref"], v=ref["v_ref"]),
                k_at_cv_r2_lcz_0_30=k_cv, utci_offset_compact_at_cv_0_30=k_cv * g,
                k_at_within_district_lcz_share_0_50=k_res, utci_offset_compact_at_resid_0_50=k_res * g,
                resid_share_at_k0=resid_at(0.0), lmm_fits=[dict(k=k, resid_share=v_[0], method=v_[1], converged=v_[2])
                                                          for k, v_ in sorted(cache.items())],
                cv_curve=curve)


# ----------------------------------------------------------------------------------------------
# B  H3 extras
# ----------------------------------------------------------------------------------------------
def spearman_cells(day):
    from scipy.stats import spearmanr
    c = pd.read_parquet(H3_DIR / f"h3_moran_250m_cells_{day}.parquet")
    ok = np.isfinite(c["lst_med"]) & np.isfinite(c["m1_x"])
    c = c[ok].reset_index(drop=True)
    a = c["g250_id"].str.split("_", expand=True).astype(int)
    bx = np.floor((a[0] * 250 + 125) / cm.BLOCK_M).astype(int)
    by = np.floor((a[1] * 250 + 125) / cm.BLOCK_M).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str)).to_numpy()
    inv, counts = boot_weights(blocks, cm.SEED)
    x, y, ym = c["lst_med"].to_numpy(), c["m1_x"].to_numpy(), c["m1_median"].to_numpy()
    rho = float(spearmanr(x, y).statistic)
    rho_med = float(spearmanr(x, ym).statistic)
    bs = []
    for r in range(counts.shape[0]):
        rows = cm.rows_from_counts(inv, counts[r])
        bs.append(spearmanr(x[rows], y[rows]).statistic)
    return dict(n_cells=len(c), n_blocks=int(len(np.unique(blocks))), rho=rho, ci95=cm.ci95(bs), rho_with_cell_median_m1=rho_med,
                note=("250 m cells with both LST (lst_med) and M1 (m1_x = n_ps_valid-weighted mean of segment M1, the "
                      "column used in the registered Moran analysis); 2 km block bootstrap of cells, 1000, seed 20261030."))


def decoupled(seg, met, day):
    d = seg[["seg_id", "lst_med", "x_mid", "y_mid"]].merge(met[["seg_id", "utci_b15"]], on="seg_id")
    d = d[np.isfinite(d["lst_med"]) & np.isfinite(d["utci_b15"])].reset_index(drop=True)
    lst, m1 = d["lst_med"].to_numpy(), d["utci_b15"].to_numpy()
    q1, q2 = np.quantile(lst, 1 / 3), np.quantile(lst, 2 / 3)
    lt = np.where(lst <= q1, 1, np.where(lst <= q2, 2, 3))
    med = float(np.median(m1))
    mq1, mq2 = np.quantile(m1, 1 / 3), np.quantile(m1, 2 / 3)

    def shares(m):
        mt = np.where(m <= mq1, 1, np.where(m <= mq2, 2, 3))
        terc = ((lt == 3) & (mt == 1)) | ((lt == 1) & (mt == 3))
        hs = (lt == 3) & (m <= med - 1.0)
        cs = (lt == 1) & (m >= med + 1.0)
        return terc.mean(), (hs | cs).mean(), hs.mean(), cs.mean()

    obs = shares(m1)
    rng = np.random.default_rng(cm.SEED)
    perm = np.array([shares(rng.permutation(m1)) for _ in range(cm.N_BOOT)])
    blocks = (np.floor(d["x_mid"] / cm.BLOCK_M).astype(int).astype(str) + "_" +
              np.floor(d["y_mid"] / cm.BLOCK_M).astype(int).astype(str)).to_numpy()
    inv, counts = boot_weights(blocks, cm.SEED)
    mt = np.where(m1 <= mq1, 1, np.where(m1 <= mq2, 2, 3))
    flag = (((lt == 3) & (m1 <= med - 1.0)) | ((lt == 1) & (m1 >= med + 1.0))).astype(float)
    bs = [np.sum(counts[r][inv] * flag) / np.sum(counts[r][inv]) for r in range(counts.shape[0])]
    p_le = float((np.sum(perm[:, 1] <= obs[1]) + 1) / (len(perm) + 1))
    return dict(
        n_segments=len(d), m1_median=med, share_m1_le_median_minus_1=float((m1 <= med - 1).mean()),
        share_m1_ge_median_plus_1=float((m1 >= med + 1).mean()),
        tercile_definition=dict(observed=float(obs[0]), permutation_mean=float(perm[:, 0].mean()),
                                permutation_ci95=cm.ci95(perm[:, 0]), analytic_independence=2 / 9),
        min_separation_1C=dict(observed=float(obs[1]), observed_ci95_block_bootstrap=cm.ci95(bs),
                               hot_surface_cool_ped=float(obs[2]), cool_surface_hot_ped=float(obs[3]),
                               permutation_mean=float(perm[:, 1].mean()), permutation_ci95=cm.ci95(perm[:, 1]),
                               observed_over_expected=float(obs[1] / perm[:, 1].mean()),
                               perm_p_one_sided_le=p_le),
        note=("Segment level, citywide LST terciles. Minimum-separation rule: top LST tercile with M1 <= day median - 1 C, "
              "or bottom LST tercile with M1 >= day median + 1 C. Null: 1000 random permutations of M1 against LST "
              "(seed 20261030), same rule. Tercile rule (registered H3 definition) recomputed for reference."))


# ----------------------------------------------------------------------------------------------
# C  H4 extras
# ----------------------------------------------------------------------------------------------
def h4_extras(df, ref):
    fo = df.attrs["forcing"]
    ta, rh = fo["ta"], fo["rh"]
    d = df.dropna(subset=["tmrt_b15", "v10eff_b15", "m1"]).reset_index(drop=True)
    T, V = d["tmrt_b15"].to_numpy(float), d["v10eff_b15"].to_numpy(float)
    w_ps = d["n_ps_valid"].to_numpy(float)
    t_ref, v_ref = ref["tmrt_ref"], ref["v_ref"]
    inv, counts = boot_weights(d["block"].to_numpy(), cm.SEED + 4)
    s, b, _, _ = h4.shapley_terms(ta, rh, T, V, t_ref, v_ref)

    def decomp(w):
        vs, vb = cm.wvar(s, w), cm.wvar(b, w)
        ms, mb = np.sum(w * s) / np.sum(w), np.sum(w * b) / np.sum(w)
        cv = np.sum(w * (s - ms) * (b - mb)) / np.sum(w)
        tot = vs + vb + 2 * cv
        return np.array([cv, vs / tot, vb / tot, 2 * cv / tot, tot, np.corrcoef(s, b)[0, 1] if w is None else cv / np.sqrt(vs * vb)])

    pt = decomp(np.ones(len(d)))
    bs = np.array([decomp(counts[r][inv].astype(float)) for r in range(counts.shape[0])])
    names = ["cov_S_B", "share_S", "share_B", "share_2cov", "var_S_plus_B", "corr_S_B"]
    cov = {n: dict(value=float(pt[i]), ci95=cm.ci95(bs[:, i])) for i, n in enumerate(names)}
    cov["note"] = "population (ddof=0) moments; shares of Var(S+B) = Var S + Var B + 2 Cov; CI 2 km block bootstrap, seed 20261030+4"

    def ratio_ci(s_, b_):
        r_pt = np.var(s_) / np.var(b_)
        rb = [cm.wvar(s_, counts[r][inv]) / cm.wvar(b_, counts[r][inv]) for r in range(counts.shape[0])]
        return dict(var_S=float(np.var(s_)), var_B=float(np.var(b_)), ratio=float(r_pt), ratio_ci95=cm.ci95(rb))

    disp = {}
    for a in (0.5, 1.0, 1.5, 2.0):
        Va = v_ref * (V / v_ref) ** a
        s_, b_, _, _ = h4.shapley_terms(ta, rh, T, Va, t_ref, v_ref)
        r = ratio_ci(s_, b_)
        r.update(sd_v=float(np.std(Va)), v_min=float(Va.min()), v_max=float(Va.max()))
        disp[str(a)] = r
    alt = {}
    v_mean_ps = float(np.sum(w_ps * V) / np.sum(w_ps))
    for name, vr in (("registered_PS_median", v_ref), ("city_mean_v_ps_weighted", v_mean_ps),
                     ("city_mean_v_unweighted", float(V.mean())), ("fixed_7_ms", 7.0)):
        s_, b_, _, _ = h4.shapley_terms(ta, rh, T, V, t_ref, vr)
        r = ratio_ci(s_, b_)
        r["v_ref_used"] = vr
        alt[name] = r
    log(f"[C] {df.attrs['day']}: cov {pt[0]:.4f}; disp " + ", ".join(f"a={k}: {v_['ratio']:.2f}" for k, v_ in disp.items())
        + "; alt " + ", ".join(f"{k}: {v_['ratio']:.2f}" for k, v_ in alt.items()))
    return dict(n_segments=len(d), reference=dict(tmrt_ref=t_ref, v_ref=v_ref, ta=ta, rh=rh, sd_v=float(V.std())),
                covariance=cov, wind_dispersion=disp, alternative_reference_wind=alt,
                note=("Dispersion: v_i -> v_ref (v_i / v_ref)^a, Tmrt, Ta, RH and the Tmrt reference unchanged; S and B "
                      "recomputed with the registered Shapley formula and pythermalcomfort UTCI. Alternative references "
                      "change v_ref only (Tmrt_ref kept)."))


# ----------------------------------------------------------------------------------------------
def main():
    cm.utf8_stdout()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = cm.load_frames(seg_path=FINAL / "segments.gpkg",
                            met_paths={d: FINAL / f"seg_metrics_{d}.parquet" for d in cm.DAYS})
    for d, f in frames.items():
        f.attrs["day"] = d
    h4j = json.loads(H4_JSON.read_text(encoding="utf-8"))
    h1j = json.loads(H1_JSON.read_text(encoding="utf-8"))
    h4ref = {d: h4j["results"][d]["reference"] for d in cm.DAYS}

    res = dict(description="Revision R1 post-hoc descriptive robustness quantities (not registered analyses)",
               constants=cm.constants_dict(), inputs=dict(final_dir=str(FINAL.relative_to(ROOT)),
                                                          h4_reference_from=str(H4_JSON.relative_to(ROOT))))
    rep = reproduce(frames, h4ref)
    targets = {"IMBAT": dict(lcz=0.030, mor=0.71, ratio=5.15), "CALM": dict(lcz=0.011, mor=0.63, ratio=632)}
    ok = True
    for d in cm.DAYS:
        r, t = rep[d], targets[d]
        r["registered"] = dict(cv_r2_lcz_only=h1j["results"][d]["cv"]["r2_lcz_only"],
                               cv_r2_morphology=h1j["results"][d]["cv"]["r2_morphology_lgbm"],
                               ratio=h4j["results"][d]["ratio_var_shade_over_breeze"])
        r["match"] = bool(round(r["cv_r2_lcz_only"], 3) == t["lcz"] and round(r["cv_r2_morphology"], 2) == t["mor"]
                          and abs(r["ratio_varS_varB"] / t["ratio"] - 1) < 0.005)
        ok &= r["match"]
    res["reproduction"] = rep
    res["reproduction_ok"] = ok
    if not ok:
        cm.write_json(OUT / "robustness_r1.json", res)
        raise SystemExit("reproduction failed; stopping")

    asg = grid100_assignment()
    seg = cm.load_segments(FINAL / "segments.gpkg")
    A, B, C = {}, {}, {}
    for d in cm.DAYS:
        df = frames[d]
        A[d] = dict(class_means=class_means(df), single_predictors=single_predictors(df),
                    support_100m=support_benchmark(df, asg), cool_cluster=cool_cluster(df),
                    tipping_point=tipping(df, h4ref[d]))
        met = pd.read_parquet(FINAL / f"seg_metrics_{d}.parquet")
        B[d] = dict(spearman_250m_cells=spearman_cells(d), decoupled=decoupled(seg, met, d))
        C[d] = h4_extras(df, h4ref[d])
    res["A_H1_extras"] = A
    res["A5_assumption"] = dict(
        contrast_vector_K_per_unit_k=CONTRAST, natural_classes=0.0,
        text=("Assumed LCZ-structured air-temperature contrast, not an observation: offset_i = k * c(LCZ_i) * dUTCI/dTa, "
              "with c = 1 for LCZ 1, 2, 3, 10 (compact / heavy industry), 0.5 for LCZ 4-8 (open, large low-rise), "
              "0.25 for LCZ 9 (sparse), 0 for natural classes (11-17 and pooled natural 102) and LCZ 0. k is in K. "
              "dUTCI/dTa = central difference (+/- 0.5 K) of pythermalcomfort UTCI at the day's H4 reference state "
              "(Ta, RH from forcing, Tmrt_ref and v_ref = PS-cell medians), RH held fixed. The offset is added to M1; "
              "LCZ-only CV R2 uses the registered folds; the within district x LCZ share is the H1 nested LMM residual "
              "share (REML, same fallback rule). LCZ 1 and 7 do not occur in the sample."))
    res["B_H3_extras"] = B
    res["C_H4_extras"] = C
    res["runtime_s"] = time.time() - t0
    cm.write_json(OUT / "robustness_r1.json", res)

    # ---- short CSVs ----
    rows = []
    for d in cm.DAYS:
        for r in A[d]["class_means"]["classes"]:
            rows.append(dict(day=d, **r))
    pd.DataFrame(rows).to_csv(OUT / "a1_lcz_class_means.csv", index=False)
    rows = []
    for d in cm.DAYS:
        a = A[d]
        rows.append(dict(day=d, model="LCZ-only (repro)", cv_r2=rep[d]["cv_r2_lcz_only"]))
        rows.append(dict(day=d, model="morphology LightGBM (repro)", cv_r2=rep[d]["cv_r2_morphology"]))
        for f, r in a["single_predictors"].items():
            rows.append(dict(day=d, model=f"single: {f}", cv_r2=r["r2"], lo=r["ci95"][0], hi=r["ci95"][1]))
        s = a["support_100m"]
        for k in ("r2_100m_aggregates_only", "r2_lcz_plus_100m_aggregates"):
            rows.append(dict(day=d, model=k, cv_r2=s[k]["r2"], lo=s[k]["ci95"][0], hi=s[k]["ci95"][1]))
        e = a["cool_cluster"]["excluded"]
        for k in ("cv_r2_lcz_only", "cv_r2_morphology"):
            rows.append(dict(day=d, model=f"cool cluster excluded: {k}", cv_r2=e[k]["r2"], lo=e[k]["ci95"][0], hi=e[k]["ci95"][1]))
    pd.DataFrame(rows).to_csv(OUT / "a_cv_r2_benchmarks.csv", index=False)
    rows = []
    for d in cm.DAYS:
        for k in A[d]["tipping_point"]["cv_curve"]:
            rows.append(dict(day=d, **k))
    pd.DataFrame(rows).to_csv(OUT / "a5_tipping_cv_curve.csv", index=False)
    rows = []
    for d in cm.DAYS:
        for a_, r in C[d]["wind_dispersion"].items():
            rows.append(dict(day=d, variant=f"dispersion a={a_}", v_ref=C[d]["reference"]["v_ref"], var_S=r["var_S"],
                             var_B=r["var_B"], ratio=r["ratio"], lo=r["ratio_ci95"][0], hi=r["ratio_ci95"][1]))
        for k, r in C[d]["alternative_reference_wind"].items():
            rows.append(dict(day=d, variant=f"v_ref {k}", v_ref=r["v_ref_used"], var_S=r["var_S"], var_B=r["var_B"],
                             ratio=r["ratio"], lo=r["ratio_ci95"][0], hi=r["ratio_ci95"][1]))
    pd.DataFrame(rows).to_csv(OUT / "c_h4_wind_sensitivity.csv", index=False)
    log(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
