"""Post-hoc, descriptive robustness quantities for H1, H3 and H4 (not part of the registered analyses).

Read-only on the frozen inputs and results. Reuses the registered helpers in scripts/street/stats (folds, block bootstrap,
LightGBM settings, UTCI call, Shapley terms, LMM) without modifying them.

Run from the project root:
    python -m scripts.street.posthoc.robustness
Outputs: outputs/street/analysis/posthoc/robustness.json and short CSVs.

Sections
  0  reproduction check (LCZ-only and morphology CV R2, Var(S)/Var(B))
  A1 LCZ class mean M1 with 95 % block-bootstrap CI, largest and segment-weighted typical between-class gap
  A2 CV R2 of single continuous predictors (SVF, tree cover, building cover)
  A3 100 m support benchmark (length-weighted 100 m aggregates of the H2 street features; alone and with LCZ)
  A4 cool cluster (segments > 2 C below the day median): variance share, CV R2 with those segments excluded
  A5 tipping point for an LCZ-structured air-temperature contrast (assumed contrast vector)
  B  H3 extras: Spearman rho at 250 m cells; decoupled share with a 1 C minimum separation vs permutation null
  C  H4 extras: Cov(S,B) and Var(S+B) shares; wind-dispersion sensitivity; alternative reference winds
  D  second batch: D1 other hours (labels 11, 12, 17); D2 LST timing (labels 11, 12); D3 bias band (-0.6, +1.35 C);
     D4 stage-1 Kruskal-Wallis epsilon2 of LST by weak class (raw and elevation/coast residualised)
  E  third batch: E5 spatial dependence (blocks, kNN Moran's I of OOF residuals, correlogram, block size x fold seed);
     E6 H1 (cool-cluster AUROC, weighted LCZ-only CV R2, LMM residual diagnostics, within-share across variants);
     E7 H2 (with / without H/W, TreeSHAP variant, feature correlations); E8 H3 (partial Spearman, within-block
     permutation decoupling test); E9 H4 (dispersion exponent at ratio 4, terms by canyon class)
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
OUT = ROOT / "outputs/street/analysis/posthoc"
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
# D  second batch (D1 other hours, D2 LST timing, D3 bias band, D4 stage-1 LST by weak class)
# ----------------------------------------------------------------------------------------------
HOURS = (11, 12, 17)
BIAS_SHIFTS = {"none": 0.0, "RADSENS_-0.6": -0.6, "landcover_+1.35": 1.35}
V8_CSV = ROOT / "data/03_processed/grid_250m_model_features_v8.csv"
V8_COLS = dict(weak_label="lcz_weak_label", confidence="lcz_weak_confidence", lst="lst_c_median_mean",
               elevation="dsm_elevation_m_mean", coast="coast_min_distance_m", row="row_id", col="col_id")


def other_hours(df):
    out = {}
    inv, counts = boot_weights(df["block"].to_numpy(), cm.SEED)
    vf = df["lcz_valid"].to_numpy(float)
    for h in HOURS:
        col = f"utci_b{h:02d}"
        y = df[col].to_numpy(float)
        assert np.isfinite(y).all()
        d = df.assign(m1=y)
        _, pl = lcz_cv(d)
        rl = r2_ci(y, np.nan_to_num(pl), inv, counts, vf)
        pm = cm.lgb_oof(d, cm.FEATURES, n_jobs=JOBS)
        rm = r2_ci(y, pm, inv, counts, vf)
        g = df["lcz"].to_numpy()
        eta = cm.eta_squared(y, g, vf)
        eb = [cm.eta_squared(y, g, counts[r][inv] * vf) for r in range(counts.shape[0])]
        out[col] = dict(mean=float(y.mean()), sd=float(y.std()), cv_r2_lcz_only=rl, cv_r2_morphology=rm,
                        eta2_segment=dict(value=eta, ci95=cm.ci95(eb)))
        log(f"[D1] {df.attrs['day']} {col}: LCZ {rl['r2']:.4f} morph {rm['r2']:.4f} eta2 {eta:.4f}")
    return out


def _spearman_boot(x, y, blocks):
    from scipy.stats import spearmanr
    inv, counts = boot_weights(blocks, cm.SEED)
    rho = float(spearmanr(x, y).statistic)
    bs = [spearmanr(x[rows], y[rows]).statistic for rows in (cm.rows_from_counts(inv, counts[r]) for r in range(counts.shape[0]))]
    return dict(n=len(x), n_blocks=int(len(np.unique(blocks))), rho=rho, ci95=cm.ci95(bs))


def lst_timing(df, day):
    cells = pd.read_parquet(H3_DIR / f"h3_moran_250m_cells_{day}.parquet")[["g250_id", "lst_med", "m1_x"]]
    seg = df[np.isfinite(df["lst_med"])].reset_index(drop=True)
    out = {}
    w = seg["n_ps_valid"].fillna(1).clip(lower=1).to_numpy(float)     # registered moran.py weight
    for h in (11, 12, 15):
        col = f"utci_b{h:02d}"
        r = dict(segment=_spearman_boot(seg["lst_med"].to_numpy(float), seg[col].to_numpy(float), seg["block"].to_numpy()))
        t = pd.DataFrame({"g250_id": seg["g250_id"], "wx": w * seg[col].to_numpy(float), "w": w}).groupby("g250_id").sum()
        agg = (t["wx"] / t["w"]).rename("u").reset_index()
        c = cells.merge(agg, on="g250_id", how="inner")
        c = c[np.isfinite(c["lst_med"]) & np.isfinite(c["u"])].reset_index(drop=True)
        if h == 15:   # aggregation check against the registered cell table
            r["cell_aggregation_check_max_abs_diff_vs_m1_x"] = float(np.nanmax(np.abs(c["u"] - c["m1_x"])))
        a = c["g250_id"].str.split("_", expand=True).astype(int)
        blocks = (np.floor((a[0] * 250 + 125) / cm.BLOCK_M).astype(int).astype(str) + "_" +
                  np.floor((a[1] * 250 + 125) / cm.BLOCK_M).astype(int).astype(str)).to_numpy()
        r["cell_250m"] = _spearman_boot(c["lst_med"].to_numpy(float), c["u"].to_numpy(float), blocks)
        out[col] = r
        log(f"[D2] {day} {col}: seg rho {r['segment']['rho']:.3f}  cell rho {r['cell_250m']['rho']:.3f}")
    return out


def bias_band(df):
    y = df["m1"].to_numpy(float)
    return {k: dict(shift_C=s, share_ge_38=float(np.mean(y + s >= 38.0)), share_ge_46=float(np.mean(y + s >= 46.0)),
                    median_m1=float(np.median(y + s))) for k, s in BIAS_SHIFTS.items()}


def stage1_lst():
    from scipy.stats import kruskal
    c = V8_COLS
    df = pd.read_csv(V8_CSV, usecols=list(c.values()))
    conf = pd.to_numeric(df[c["confidence"]], errors="coerce")
    pop = df[df[c["weak_label"]].isin([3, 6, 8, 9]) & (conf >= 0.60)].copy()
    n_pop = len(pop)
    a = pop.dropna(subset=[c["lst"], c["elevation"], c["coast"]]).reset_index(drop=True)
    cls = a[c["weak_label"]].astype(int).to_numpy()
    lst = a[c["lst"]].to_numpy(float)
    X = np.column_stack([np.ones(len(a)), a[c["elevation"]].to_numpy(float), a[c["coast"]].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, lst, rcond=None)
    resid = lst - X @ beta
    blocks = ((a[c["row"]] // 5).astype(str) + "_" + (a[c["col"]] // 5).astype(str)).to_numpy()
    inv, counts = boot_weights(blocks, cm.SEED)

    def eps2(v, rows=None):
        if rows is not None:
            v, k_ = v[rows], cls[rows]
        else:
            k_ = cls
        groups = [v[k_ == g] for g in (3, 6, 8, 9) if (k_ == g).sum() > 0]
        H = kruskal(*groups).statistic
        n, k = sum(map(len, groups)), len(groups)
        return float(max(0.0, (H - k + 1) / (n - k))), float(H)

    res = {}
    for name, v in (("lst_raw", lst), ("lst_resid_elev_coast", resid)):
        e, H = eps2(v)
        bs = [eps2(v, cm.rows_from_counts(inv, counts[r]))[0] for r in range(counts.shape[0])]
        res[name] = dict(epsilon2=e, kruskal_H=H, ci95=cm.ci95(bs))
        log(f"[D4] {name}: eps2 {e:.3f}")
    res.update(n_population=n_pop, n_used=len(a), n_blocks=int(len(np.unique(blocks))),
               class_counts={int(k): int(v) for k, v in pd.Series(cls).value_counts().sort_index().items()},
               ols=dict(intercept=float(beta[0]), per_m_elevation=float(beta[1]), per_m_coast=float(beta[2]),
                        r2=float(1 - resid.var() / lst.var())),
               columns=V8_COLS, source=str(V8_CSV.relative_to(ROOT)),
               note=("Population: lcz_weak_label in {3,6,8,9} and lcz_weak_confidence >= 0.60 (5,339 cells; units with "
                     "missing LST dropped). epsilon2 = max(0, (H - k + 1) / (n - k)) as in generate_manuscript_evidence.py. "
                     "Blocks = 5 x 5 cells of the 250 m grid (row_id // 5, col_id // 5); 1000 resamples, seed 20261030. "
                     "Residualised LST = OLS residual of lst_c_median_mean on dsm_elevation_m_mean and coast_min_distance_m "
                     "(linear, fitted once on the full population). Elevation is the cell-mean surface (DSM) elevation, "
                     "the only elevation field in the v8 matrix."))
    return res


def batch2(frames):
    D = {}
    for d in cm.DAYS:
        df = frames[d]
        D[d] = dict(other_hours=other_hours(df), lst_timing=lst_timing(df, d), bias_band=bias_band(df))
    D["stage1_lst_by_weak_class"] = stage1_lst()
    return D


def batch2_csv(D):
    rows = []
    for d in cm.DAYS:
        for col, r in D[d]["other_hours"].items():
            for k in ("cv_r2_lcz_only", "cv_r2_morphology"):
                rows.append(dict(day=d, outcome=col, quantity=k, value=r[k]["r2"], lo=r[k]["ci95"][0], hi=r[k]["ci95"][1]))
            rows.append(dict(day=d, outcome=col, quantity="eta2_segment", value=r["eta2_segment"]["value"],
                             lo=r["eta2_segment"]["ci95"][0], hi=r["eta2_segment"]["ci95"][1]))
        for col, r in D[d]["lst_timing"].items():
            for k in ("segment", "cell_250m"):
                rows.append(dict(day=d, outcome=col, quantity=f"spearman_lst_{k}", value=r[k]["rho"],
                                 lo=r[k]["ci95"][0], hi=r[k]["ci95"][1]))
        for k, r in D[d]["bias_band"].items():
            for q in ("share_ge_38", "share_ge_46", "median_m1"):
                rows.append(dict(day=d, outcome=f"m1 shift {k}", quantity=q, value=r[q]))
    s = D["stage1_lst_by_weak_class"]
    for k in ("lst_raw", "lst_resid_elev_coast"):
        rows.append(dict(day="250m stage 1", outcome=k, quantity="epsilon2_kruskal", value=s[k]["epsilon2"],
                         lo=s[k]["ci95"][0], hi=s[k]["ci95"][1]))
    pd.DataFrame(rows).to_csv(OUT / "d_batch2.csv", index=False)


# ----------------------------------------------------------------------------------------------
# E  third batch (E5 spatial dependence, E6 H1, E7 H2, E8 H3, E9 H4)
# ----------------------------------------------------------------------------------------------
KNN_K = 8
BLOCK_SIZES_KM = (1, 2, 5, 10)
FOLD_SEEDS = tuple(cm.SEED + i for i in range(5))
CORR_EDGES = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
SHORT_R = 500.0

BOOTSTRAP_STATEMENT = {
    "H1": ("LMM variance shares: model REFIT on every block-bootstrap resample (rows repeated by block multiplicity; "
           "seed 20261030). CV R2 (LCZ-only and morphology): the out-of-fold predictions are computed once and only "
           "re-weighted by block multiplicities (no refit, folds fixed). Segment eta2 and ladder eta2: recomputed on "
           "re-weighted rows (ladder seed 20261030+11)."),
    "H2": ("CV R2 of models A/B, LCZ-only reference and within-LCZ residual R2: fixed out-of-fold predictions re-weighted "
           "(no refit; seed 20261030+2). TreeSHAP family shares: the stored out-of-fold SHAP matrix re-weighted (no refit). "
           "Partial-dependence bands: 1000 REFITS of model A on block-resampled rows (full sample, not cross-validated)."),
    "H3": ("Spearman rho: recomputed on each resample of 2 km blocks (own RNG stream, seed 20261030); no model involved."),
    "H4": ("Var(S), Var(B), ratio: the Shapley terms are computed once with the fixed reference (Tmrt_ref, v_ref) and only "
           "re-weighted by block multiplicities (seed 20261030+4); the reference is not re-estimated per resample."),
}


def _blocks_for(df, size_m):
    bx = np.floor(df["x_mid"] / size_m).astype(int)
    by = np.floor(df["y_mid"] / size_m).astype(int)
    return bx.astype(str) + "_" + by.astype(str)


def _knn_moran(xy, z, k=KNN_K, n_perm=199, seed=cm.SEED):
    from scipy.spatial import cKDTree
    z = np.asarray(z, float) - np.mean(z)
    _, nn = cKDTree(xy).query(xy, k=k + 1)
    nn = nn[:, 1:]
    den = np.sum(z ** 2)
    I = float(np.sum(z * z[nn].mean(axis=1)) / den)
    rng = np.random.default_rng(seed)
    perm = np.array([np.sum(zp * zp[nn].mean(axis=1)) / den for zp in (rng.permutation(z) for _ in range(n_perm))])
    return dict(I=I, perm_mean=float(perm.mean()), perm_sd=float(perm.std()), z=float((I - perm.mean()) / perm.std()),
                p_sim=float((np.sum(perm >= I) + 1) / (n_perm + 1)))


def _correlogram(xy, resids, seed=cm.SEED, n_sub=6000):
    """Moran-type correlogram for several residual vectors sharing the same points. Returns [(rows, below), ...]."""
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist
    edges = np.array(CORR_EDGES, float)
    pairs = cKDTree(xy).query_pairs(SHORT_R, output_type="ndarray")
    dd = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(xy), min(n_sub, len(xy)), replace=False)
    ds = pdist(xy[sub]).astype(np.float32)
    iu = np.triu_indices(len(sub), 1)
    short_masks = [(dd > lo) & (dd <= hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi <= SHORT_R]
    long_masks = [(ds > lo) & (ds <= hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > SHORT_R]
    out = []
    for r in resids:
        z = (r - r.mean()) / r.std()
        pp = z[pairs[:, 0]] * z[pairs[:, 1]]
        zs = z[sub]
        ps = (zs[iu[0]] * zs[iu[1]]).astype(np.float32)
        rows = []
        ms, ml = iter(short_masks), iter(long_masks)
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= SHORT_R:
                m = next(ms)
                v, n, src = (float(pp[m].mean()) if m.any() else float("nan")), int(m.sum()), "all pairs"
            else:
                m = next(ml)
                v, n, src = (float(ps[m].mean()) if m.any() else float("nan")), int(m.sum()), f"random subsample of {len(sub)}"
            rows.append(dict(d_lo=float(lo), d_hi=float(hi), corr=v, n_pairs=n, pairs=src))
        below = next((x["d_lo"] for x in rows if x["corr"] < 0.05), float("nan"))
        out.append((rows, below))
    return out


def _cv_with_blocks(frames, size_m, seed):
    """LCZ-only and morphology CV R2 with folds rebuilt from size_m blocks and seed (same balancing rule as registered)."""
    blk = {d: _blocks_for(f, size_m) for d, f in frames.items()}
    cnt = pd.concat([b.value_counts() for b in blk.values()], axis=1).fillna(0).max(axis=1)
    fm = cm.assign_folds_from_counts(cnt, seed=seed)
    out = {}
    for d, f in frames.items():
        g = f.copy()
        g["fold"] = blk[d].map(fm).astype(int).to_numpy()
        y = g["m1"].to_numpy(float)
        vf = g["lcz_valid"].to_numpy(float)
        rl, _ = lcz_cv(g)
        pm = cm.lgb_oof(g, cm.FEATURES, n_jobs=JOBS)
        out[d] = dict(lcz=rl, mor=cm.wr2(y, pm, vf), n_blocks=int(blk[d].nunique()))
    return out


def spatial_dependence(frames):
    res = {}
    for d, df in frames.items():
        tpb = df.groupby("block")["tile_id"].nunique()
        spb = df.groupby("block").size()
        v = df["lcz_valid"].to_numpy()
        y = df["m1"].to_numpy(float)
        xy = df[["x_mid", "y_mid"]].to_numpy(float)
        r_l = (y - np.nan_to_num(df.attrs["p_lcz"]))[v]
        r_m = (y - df.attrs["p_mor"])[v]
        mor = dict(m1_raw=_knn_moran(xy[v], y[v]), resid_lcz_only=_knn_moran(xy[v], r_l), resid_morphology=_knn_moran(xy[v], r_m))
        (cg_l, b_l), (cg_m, b_m), (cg_y, b_y) = _correlogram(xy[v], [r_l, r_m, y[v]])
        pd.DataFrame([dict(series=s_, **row) for s_, cg in (("resid_lcz_only", cg_l), ("resid_morphology", cg_m),
                                                             ("m1_raw", cg_y)) for row in cg]).to_csv(
            OUT / f"e5_correlogram_{d}.csv", index=False)
        res[d] = dict(n_blocks=int(df["block"].nunique()), n_tiles=int(df["tile_id"].nunique()),
                      tiles_per_block=dict(median=float(tpb.median()), mean=float(tpb.mean()), min=int(tpb.min()), max=int(tpb.max())),
                      segments_per_block=dict(median=float(spb.median()), min=int(spb.min()), max=int(spb.max())),
                      moran_knn=mor, correlogram_lcz_only=cg_l, correlogram_morphology=cg_m,
                      correlogram_m1_raw=cg_y, distance_corr_below_0_05_m1_raw_m=b_y,
                      distance_corr_below_0_05_lcz_only_m=b_l, distance_corr_below_0_05_morphology_m=b_m)
        log(f"[E5] {d}: blocks {res[d]['n_blocks']}  I lcz {mor['resid_lcz_only']['I']:.3f} mor {mor['resid_morphology']['I']:.3f}"
            f"  below0.05 at {b_l} / {b_m} m")
    grid = []
    for km in BLOCK_SIZES_KM:
        for s in FOLD_SEEDS:
            r = _cv_with_blocks(frames, km * 1000.0, s)
            for d in r:
                grid.append(dict(day=d, block_km=km, seed=s, n_blocks=r[d]["n_blocks"], cv_r2_lcz_only=r[d]["lcz"],
                                 cv_r2_morphology=r[d]["mor"]))
        log(f"[E5] block {km} km done")
    g = pd.DataFrame(grid)
    g.to_csv(OUT / "e5_cv_block_size_seeds.csv", index=False)
    summ = g.groupby(["day", "block_km"]).agg(lcz_mean=("cv_r2_lcz_only", "mean"), lcz_min=("cv_r2_lcz_only", "min"),
                                              lcz_max=("cv_r2_lcz_only", "max"), mor_mean=("cv_r2_morphology", "mean"),
                                              mor_min=("cv_r2_morphology", "min"), mor_max=("cv_r2_morphology", "max")).reset_index()
    res["cv_block_size_x_seed"] = dict(rows=grid, summary=summ.to_dict(orient="records"), seeds=list(FOLD_SEEDS))
    res["weights_note"] = (f"Moran's I with k-nearest-neighbour weights (k = {KNN_K}, row-standardised, segment midpoints, "
                           "lcz_valid rows), 199 random permutations. Correlogram: mean product of standardised residuals "
                           f"per distance band (Moran-type autocorrelation); all pairs up to {SHORT_R:.0f} m, a random "
                           "subsample of 6,000 segments beyond. 'Distance below 0.05' = lower edge of the first band whose "
                           "correlation is < 0.05.")
    res["registered_bootstrap"] = BOOTSTRAP_STATEMENT
    return res


# ---- E6 ---------------------------------------------------------------------------------------
def _weighted_lcz_oof(df, w):
    pred = np.full(len(df), np.nan)
    valid = df["lcz_valid"].to_numpy()
    y = df["m1"].to_numpy(float)
    for k in range(cm.N_FOLDS):
        te = (df["fold"] == k).to_numpy()
        tr = (~te) & valid & (w > 0)
        te = te & valid
        t = pd.DataFrame({"g": df.loc[tr, "lcz"].to_numpy(), "wy": w[tr] * y[tr], "w": w[tr]}).groupby("g").sum()
        gm = t["wy"] / t["w"]
        pred[te] = df.loc[te, "lcz"].map(gm).fillna(np.sum(w[tr] * y[tr]) / np.sum(w[tr])).to_numpy()
    return pred


def _auc_ci(lab, score, inv, counts, base):
    from sklearn.metrics import roc_auc_score
    m = base > 0
    pt = roc_auc_score(lab[m], score[m])
    bs = []
    for r in range(counts.shape[0]):
        w = counts[r][inv] * base
        k = w > 0
        bs.append(roc_auc_score(lab[k], score[k], sample_weight=w[k]))
    return dict(auroc=float(pt), ci95=cm.ci95(bs))


def _lmm_full(y, d, l):
    import statsmodels.formula.api as smf
    from scipy.stats import kurtosis, skew
    df = pd.DataFrame({"y": y, "d": pd.Series(d).astype(str).to_numpy(), "lcz_c": "L" + pd.Series(l).astype(str).to_numpy()})
    md = smf.mixedlm("y ~ 1", df, groups="d", re_formula="1", vc_formula={"lcz": "0 + lcz_c"})
    r = md.fit(reml=True)
    vd, vl, ve = float(r.cov_re.iloc[0, 0]), float(r.vcomp[0]), float(r.scale)
    res = np.asarray(r.resid, float)
    return dict(var_district=vd, var_lcz_in_district=vl, var_resid=ve, sum_components=vd + vl + ve,
                sample_var=float(np.var(y, ddof=1)), sum_over_sample_var=float((vd + vl + ve) / np.var(y, ddof=1)),
                resid_share=ve / (vd + vl + ve), converged=bool(r.converged), intercept=float(r.fe_params.iloc[0]),
                cond_resid_skew=float(skew(res)), cond_resid_excess_kurtosis=float(kurtosis(res)),
                y_skew=float(skew(y)), y_excess_kurtosis=float(kurtosis(y)),
                n_districts=int(df["d"].nunique()), n_district_lcz_cells=int(df.groupby(["d", "lcz_c"]).ngroups))


def h1_extras2(frames):
    seg_w = pd.read_parquet(ROOT / "outputs/street/analysis/h5/h5_seg_outcomes.parquet", columns=["seg_id", "ped_weight"])
    out = {}
    for d, df in frames.items():
        y = df["m1"].to_numpy(float)
        v = df["lcz_valid"].to_numpy()
        vf = v.astype(float)
        inv, counts = boot_weights(df["block"].to_numpy(), cm.SEED)
        cool = (y < np.median(y) - 2.0).astype(int)
        # AUROC: LCZ-only classifier = training-fold cool frequency per (pooled) LCZ class
        sc_l = cm.lcz_only_oof(df.assign(m1=cool.astype(float)))
        sc_m = cm.lgb_oof(df.assign(m1=cool.astype(float)), cm.FEATURES, n_jobs=JOBS)
        auc_l = _auc_ci(cool, np.nan_to_num(sc_l), inv, counts, vf)
        auc_m = _auc_ci(cool, sc_m, inv, counts, vf)
        # weighted LCZ-only CV R2
        L = df["length_m"].to_numpy(float)
        P = df["seg_id"].map(seg_w.set_index("seg_id")["ped_weight"]).fillna(0.0).to_numpy(float)
        wres = {}
        for name, w in (("length_m", L), ("h5_pedestrian_weight", P)):
            p = _weighted_lcz_oof(df, w)
            wres[name] = dict(r2_weighted_fit_weighted_score=r2_ci(y, np.nan_to_num(p), inv, counts, vf * w),
                              r2_unweighted_fit_weighted_score=cm.wr2(y, np.nan_to_num(df.attrs["p_lcz"]), vf * w),
                              share_of_rows_with_positive_weight=float(np.mean(w[v] > 0)))
        # LMM diagnostics
        dl = df[v].reset_index(drop=True)
        yl = dl["m1"].to_numpy(float)
        full = _lmm_full(yl, dl["district"], dl["lcz"])
        main = yl >= np.median(y) - 2.0
        mm = _lmm_full(yl[main], dl["district"][main], dl["lcz"][main])
        dist_means = dl.groupby("district")["m1"].agg(["mean", "size"])
        out[d] = dict(
            cool_cluster_auroc=dict(lcz_only=auc_l, morphology_lgbm=auc_m, prevalence=float(cool[v].mean())),
            lcz_only_weighted_cv_r2=wres,
            lmm_diagnostics=dict(registered_fit=full, main_mode_fit_excluding_cool_cluster=mm,
                                 n_main_mode=int(main.sum()),
                                 sd_of_district_means=float(dist_means["mean"].std()),
                                 n_districts=int(len(dist_means))))
        log(f"[E6] {d}: AUROC lcz {auc_l['auroc']:.3f} mor {auc_m['auroc']:.3f}; LMM sum/var {full['sum_over_sample_var']:.3f}"
            f" main {mm['sum_over_sample_var']:.3f}; skew {full['cond_resid_skew']:.2f} kurt {full['cond_resid_excess_kurtosis']:.1f}")
    # within-share across re-extraction variants (frozen outputs)
    var = []
    for p in sorted((ROOT / "outputs/street/analysis/h1").glob("results*/h1_results*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        for dd, r in j.get("results", {}).items():
            var.append(dict(variant=p.parent.name.replace("results_", "").replace("results", "registered"), file=p.name, day=dd,
                            n_segments=r.get("n_segments"), label=r.get("label"),
                            resid_share=r["lmm"]["shares"]["resid_share"], cv_r2_lcz_only=r["cv"]["r2_lcz_only"],
                            cv_r2_morphology=r["cv"]["r2_morphology_lgbm"]))
    pd.DataFrame(var).to_csv(OUT / "e6_within_share_variants.csv", index=False)
    out["within_share_variants"] = var
    out["within_share_variants_note"] = ("Read from outputs/street/analysis/h1/results*/h1_results*.json. *_PARTIAL files are "
                                         "the 40-tile GPU re-extraction variants (primary40 = same 40 tiles, the S-D "
                                         "reference); others are full-sample sensitivity runs; synthetic folders are not read.")
    out["auroc_note"] = ("Cool cluster = M1 < day median - 2 C. LCZ-only classifier score = training-fold share of cool "
                         "segments in the pooled LCZ class (registered folds); morphology score = LightGBM (registered "
                         "settings, regression objective on the 0/1 label) out-of-fold. AUROC on lcz_valid rows; CI = 2 km "
                         "block bootstrap of the fixed out-of-fold scores.")
    return out


# ---- E7 ---------------------------------------------------------------------------------------
def h2_extras(frames):
    out = {}
    h2j = json.loads((ROOT / "outputs/street/analysis/h2/results/h2_results.json").read_text(encoding="utf-8"))
    fam_rows = []
    for d, df in frames.items():
        y = df["m1"].to_numpy(float)
        vf = df["lcz_valid"].to_numpy(float)
        inv, counts = boot_weights(df["block"].to_numpy(), cm.SEED)
        f_no = [f for f in cm.FEATURES if f != "hw_ratio"]
        p_no = cm.lgb_oof(df, f_no, n_jobs=JOBS)
        with_ = r2_ci(y, df.attrs["p_mor"], inv, counts, vf)
        without = r2_ci(y, p_no, inv, counts, vf)
        diff = [cm.wr2(y, df.attrs["p_mor"], counts[r][inv] * vf) - cm.wr2(y, p_no, counts[r][inv] * vf)
                for r in range(counts.shape[0])]
        C = df[cm.FEATURES].corr(method="spearman")
        C.to_csv(OUT / f"e7_feature_spearman_{d}.csv")
        fams = list(cm.FAMILIES)
        M = pd.DataFrame(index=fams, columns=fams, dtype=float)
        for a in fams:
            for b in fams:
                sub = C.loc[cm.FAMILIES[a], cm.FAMILIES[b]].abs().to_numpy()
                if a == b:
                    sub = sub[~np.eye(len(sub), dtype=bool)]
                M.loc[a, b] = float(np.nanmean(sub)) if sub.size else float("nan")
        M.to_csv(OUT / f"e7_family_mean_abs_spearman_{d}.csv")
        feats_ = list(C.columns)
        pairs_ = [(feats_[i], feats_[j]) for i in range(len(feats_)) for j in range(i + 1, len(feats_))]
        pairs_.sort(key=lambda p: -abs(C.loc[p[0], p[1]]) if np.isfinite(C.loc[p[0], p[1]]) else 0)
        top = [dict(a=i, b=j, rho=float(C.loc[i, j])) for i, j in pairs_[:8]]
        out[d] = dict(cv_r2_with_hw=with_, cv_r2_without_hw=without, diff_with_minus_without=dict(
            value=with_["r2"] - without["r2"], ci95=cm.ci95(diff)), hw_nan_share=float(df["hw_ratio"].isna().mean()),
            family_mean_abs_spearman=M.to_dict(), strongest_pairs=top,
            treeshap_crosscheck_registered=h2j["results"][d]["treeshap_crosscheck"])
        log(f"[E7] {d}: with H/W {with_['r2']:.4f} without {without['r2']:.4f}")
    out["treeshap_variant"] = ("Path-dependent TreeSHAP: the registered H2 code uses LightGBM predict(pred_contrib=True), which "
                               "implements the tree_path_dependent algorithm (conditional expectations from the training "
                               "cover of each node; no background dataset). Its cross-check uses shap.TreeExplainer(model) "
                               "without data, whose default is also feature_perturbation='tree_path_dependent' (registered "
                               "max |diff| reported per day). Interventional SHAP was not used.")
    out["family_corr_note"] = ("Spearman correlations of the 14 H2 features (segment level, pairwise complete). Family matrix = "
                               "mean |rho| over feature pairs between two families (diagonal: within-family off-diagonal "
                               "pairs; ventilation has one feature so its diagonal is NaN).")
    return out


# ---- E8 ---------------------------------------------------------------------------------------
def _partial_spearman(x, y, Z):
    from scipy.stats import rankdata
    R = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z.T])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - R @ np.linalg.lstsq(R, rx, rcond=None)[0]
    ey = ry - R @ np.linalg.lstsq(R, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def h3_extras2(frames):
    out = {}
    for d, df in frames.items():
        s = df[np.isfinite(df["lst_med"]) & np.isfinite(df["svf_med"]) & np.isfinite(df["tcover15"])].reset_index(drop=True)
        x, y = s["lst_med"].to_numpy(float), s["m1"].to_numpy(float)
        Z = s[["svf_med", "tcover15"]].to_numpy(float)
        inv, counts = boot_weights(s["block"].to_numpy(), cm.SEED)
        pt = _partial_spearman(x, y, Z)
        bs = []
        for r in range(counts.shape[0]):
            rows = cm.rows_from_counts(inv, counts[r])
            bs.append(_partial_spearman(x[rows], y[rows], Z[rows]))
        from scipy.stats import spearmanr
        part = dict(n=len(s), partial_rho=pt, ci95=cm.ci95(bs), zero_order_rho=float(spearmanr(x, y).statistic),
                    partial_rho_svf_only=_partial_spearman(x, y, Z[:, :1]),
                    partial_rho_tcover_only=_partial_spearman(x, y, Z[:, 1:]))
        # decoupling with within-block permutation
        dd = df[np.isfinite(df["lst_med"])].reset_index(drop=True)
        lst, m1 = dd["lst_med"].to_numpy(float), dd["m1"].to_numpy(float)
        q1, q2 = np.quantile(lst, 1 / 3), np.quantile(lst, 2 / 3)
        lt = np.where(lst <= q1, 1, np.where(lst <= q2, 2, 3))
        med = float(np.median(m1))
        mq1, mq2 = np.quantile(m1, 1 / 3), np.quantile(m1, 2 / 3)

        def shares(m):
            mt = np.where(m <= mq1, 1, np.where(m <= mq2, 2, 3))
            return (np.mean(((lt == 3) & (mt == 1)) | ((lt == 1) & (mt == 3))),
                    np.mean(((lt == 3) & (m <= med - 1.0)) | ((lt == 1) & (m >= med + 1.0))))
        obs = shares(m1)
        codes, binv = np.unique(dd["block"].to_numpy(), return_inverse=True)
        order = np.argsort(binv, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(binv[order]) != 0, True])
        rng = np.random.default_rng(cm.SEED)
        perm = []
        for _ in range(cm.N_BOOT):
            mp = m1.copy()
            for a, b in zip(bounds[:-1], bounds[1:]):
                idx = order[a:b]
                mp[idx] = m1[rng.permutation(idx)]
            perm.append(shares(mp))
        perm = np.array(perm)
        dec = {}
        for i, k in enumerate(("tercile_rule", "min_separation_1C")):
            dec[k] = dict(observed=float(obs[i]), null_mean=float(perm[:, i].mean()), null_ci95=cm.ci95(perm[:, i]),
                          observed_over_null=float(obs[i] / perm[:, i].mean()),
                          p_one_sided_le=float((np.sum(perm[:, i] <= obs[i]) + 1) / (len(perm) + 1)))
        out[d] = dict(partial_spearman_lst_m1=part, decoupling_within_block_permutation=dec, n_blocks=int(len(codes)))
        log(f"[E8] {d}: partial rho {pt:.3f}; within-block null tercile {dec['tercile_rule']['null_mean']:.3f} "
            f"minsep {dec['min_separation_1C']['null_mean']:.4f}")
    out["note"] = ("Partial Spearman = Pearson correlation of the rank residuals of LST and M1 after OLS on the ranks of "
                   "svf_med and tcover15 (segment level); CI 2 km block bootstrap (ranks recomputed per resample). "
                   "Block permutation: M1 is shuffled WITHIN each 2 km block (LST fixed), 1000 permutations, seed "
                   "20261030; this keeps every block's M1 distribution (and so any block-scale LST-M1 coupling) and "
                   "breaks only the within-block pairing. Terciles and median are invariant to the shuffle.")
    return out


# ---- E9 ---------------------------------------------------------------------------------------
def h4_extras2(frames, h4ref, edges):
    out = {}
    df = frames["IMBAT"]
    fo = df.attrs["forcing"]
    d = df.dropna(subset=["tmrt_b15", "v10eff_b15", "m1"]).reset_index(drop=True)
    T, V = d["tmrt_b15"].to_numpy(float), d["v10eff_b15"].to_numpy(float)
    t_ref, v_ref = h4ref["IMBAT"]["tmrt_ref"], h4ref["IMBAT"]["v_ref"]
    grid = []
    for a in np.round(np.arange(1.0, 1.301, 0.01), 2):
        s, b, _, _ = h4.shapley_terms(fo["ta"], fo["rh"], T, v_ref * (V / v_ref) ** a, t_ref, v_ref)
        grid.append(dict(a=float(a), ratio=float(np.var(s) / np.var(b))))
    g = pd.DataFrame(grid)
    g.to_csv(OUT / "e9_dispersion_fine.csv", index=False)
    i = int(np.argmax(g["ratio"].to_numpy() < 4.0))
    a0, a1 = g["a"].iloc[i - 1], g["a"].iloc[i]
    r0, r1 = np.log(g["ratio"].iloc[i - 1]), np.log(g["ratio"].iloc[i])
    a_cross = float(a0 + (np.log(4.0) - r0) * (a1 - a0) / (r1 - r0))
    out["IMBAT_dispersion_crossing"] = dict(a_at_ratio_4=a_cross, grid=grid,
                                            note="linear interpolation of log ratio between the bracketing 0.01 steps")
    log(f"[E9] IMBAT ratio crosses 4 at a = {a_cross:.3f}")
    for day, f in frames.items():
        t = pd.read_parquet(ROOT / f"outputs/street/analysis/h4/results/seg_h4_terms_{day}.parquet",
                            columns=["seg_id", "shade_term", "breeze_term", "tmrt_b15", "v10eff_b15", "hw_ratio", "svf_med"])
        t["hw_class"] = h4.hw_class(t["hw_ratio"], edges).to_numpy()
        t["svf_quintile"] = pd.qcut(t["svf_med"], 5, labels=["Q1 (deepest)", "Q2", "Q3", "Q4", "Q5 (most open)"]).astype(str)
        rows = []
        for col in ("hw_class", "svf_quintile"):
            gg = t.groupby(col).agg(n=("seg_id", "size"), mean_S=("shade_term", "mean"), mean_B=("breeze_term", "mean"),
                                    mean_tmrt=("tmrt_b15", "mean"), mean_v=("v10eff_b15", "mean")).reset_index()
            for _, r in gg.iterrows():
                rows.append(dict(day=day, grouping=col, cls=r[col], **{k: r[k] for k in ("n", "mean_S", "mean_B", "mean_tmrt", "mean_v")}))
        pd.DataFrame(rows).to_csv(OUT / f"e9_terms_by_canyon_{day}.csv", index=False)
        ref = h4ref[day]
        fo_ = f.attrs["forcing"]
        dUdv = float(cm.utci_vec(fo_["ta"], [ref["tmrt_ref"]], [ref["v_ref"] + 0.5], fo_["rh"])[0]
                     - cm.utci_vec(fo_["ta"], [ref["tmrt_ref"]], [ref["v_ref"] - 0.5], fo_["rh"])[0])
        # covariance decomposition: main effects vs shared interaction; between vs within SVF quintiles
        T_, V_ = t["tmrt_b15"].to_numpy(float), t["v10eff_b15"].to_numpy(float)
        tr, vr = ref["tmrt_ref"], ref["v_ref"]
        uiv = cm.utci_vec(fo_["ta"], T_, V_, fo_["rh"])
        uir = cm.utci_vec(fo_["ta"], T_, np.full_like(V_, vr), fo_["rh"])
        uri = cm.utci_vec(fo_["ta"], np.full_like(T_, tr), V_, fo_["rh"])
        urr = float(cm.utci_vec(fo_["ta"], [tr], [vr], fo_["rh"])[0])
        Sm, Bm, I_ = uir - urr, uri - urr, uiv - uir - uri + urr
        S_, B_ = t["shade_term"].to_numpy(float), t["breeze_term"].to_numpy(float)

        def cv_(a, b):
            return float(np.mean((a - a.mean()) * (b - b.mean())))
        q = pd.qcut(t["svf_med"], 5, labels=False).to_numpy()
        Sb = pd.Series(S_).groupby(q).transform("mean").to_numpy()
        Bb = pd.Series(B_).groupby(q).transform("mean").to_numpy()
        cov_dec = dict(cov_S_B=cv_(S_, B_), main_effects_cov=cv_(Sm, Bm),
                       interaction_cross_terms=cv_(Sm, I_ / 2) + cv_(I_ / 2, Bm) + float(np.var(I_)) / 4,
                       between_svf_quintiles=cv_(Sb, Bb), within_svf_quintiles=cv_(S_ - Sb, B_ - Bb),
                       note=("S = Sm + I/2, B = Bm + I/2 with Sm = U(Ti,vref)-U(ref), Bm = U(Tref,vi)-U(ref), "
                             "I = interaction; Cov(S,B) = Cov(Sm,Bm) + [Cov(Sm,I/2) + Cov(I/2,Bm) + Var(I)/4]. "
                             "Between/within = law of total covariance over SVF quintiles."))
        out[day] = dict(by_class=rows, cov_decomposition=cov_dec,
                        corr_tmrt_v=float(np.corrcoef(t["tmrt_b15"], t["v10eff_b15"])[0, 1]),
                        spearman_tmrt_v=float(t["tmrt_b15"].corr(t["v10eff_b15"], method="spearman")),
                        corr_S_B=float(np.corrcoef(t["shade_term"], t["breeze_term"])[0, 1]),
                        dUTCI_dv_at_reference=dUdv)
        log(f"[E9] {day}: corr(Tmrt,v) {out[day]['corr_tmrt_v']:.3f}; dU/dv {dUdv:.3f}")
    out["note"] = ("Mean Shapley terms by H/W class (registered tercile edges, NaN = no enclosing buildings) and by SVF "
                   "quintile (pooled per day). Sign of Cov(S,B) = sign of corr(Tmrt, v) x sign(dUTCI/dTmrt > 0) x "
                   "sign(dUTCI/dv) at the reference.")
    return out


def batch3(frames, h4ref):
    edges = tuple(json.loads(H4_JSON.read_text(encoding="utf-8"))["hw_tercile_edges"])
    return dict(E5_spatial_dependence=spatial_dependence(frames), E6_H1=h1_extras2(frames), E7_H2=h2_extras(frames),
                E8_H3=h3_extras2(frames), E9_H4=h4_extras2(frames, h4ref, edges))


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

    res = dict(description="Post-hoc descriptive robustness quantities (not registered analyses)",
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
        cm.write_json(OUT / "robustness.json", res)
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
    res["D_batch2"] = batch2(frames)
    batch2_csv(res["D_batch2"])
    cm.write_json(OUT / "robustness.json", res)          # checkpoint before the long third batch
    res["E_batch3"] = batch3(frames, h4ref)
    res["runtime_s"] = time.time() - t0
    cm.write_json(OUT / "robustness.json", res)

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
