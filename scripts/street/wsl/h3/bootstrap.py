"""Spatial block bootstrap and Spearman correlation analysis for H3.

Pre-analysis lock (§2 H3):
- 2 km spatial blocks
- 1000 resamples, 95% CI
- Fixed seed 20261030
- Spearman rho(LST, M3) and rho(LST, M1) overall and by LCZ
- Compact paradox rule: rho(LST, M1) < 0 within LCZ 2/3 with 95% CI excluding 0.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats

SEED = 20261030
BLOCK_M = 2000.0
N_BOOT = 1000
MIN_CLASS_N = 30
POOL_BUILT = 101
POOL_NAT = 102

LCZ_NAMES = {
    1: "LCZ 1: Compact highrise",
    2: "LCZ 2: Compact midrise",
    3: "LCZ 3: Compact lowrise",
    4: "LCZ 4: Open highrise",
    5: "LCZ 5: Open midrise",
    6: "LCZ 6: Open lowrise",
    7: "LCZ 7: Lightweight lowrise",
    8: "LCZ 8: Large lowrise",
    9: "LCZ 9: Sparsely built",
    10: "LCZ 10: Heavy industry",
    101: "LCZ Other built (<30 segs)",
    102: "LCZ Other natural (<30 segs)",
    "A": "LCZ A: Dense trees",
    "B": "LCZ B: Scattered trees",
    "C": "LCZ C: Bush, scrub",
    "D": "LCZ D: Low plants",
    "E": "LCZ E: Bare rock/paved",
    "F": "LCZ F: Bare soil/sand",
    "G": "LCZ G: Water",
}


def lcz_valid_mask(lcz_series: pd.Series) -> pd.Series:
    """True where the LCZ label is a real class. Amendment A6: 0 is the LCZ raster no-data value (sea, outside the
    map), not a class; missing values (NaN / <NA>) are not classes either."""
    s = lcz_series
    return s.notna() & ~s.astype(str).str.strip().isin(["0", "0.0"])


def mask_lcz_nodata(lcz_series: pd.Series) -> pd.Series:
    """Object Series with NaN wherever lcz_valid_mask is False (integral floats are converted to int)."""
    valid = lcz_valid_mask(lcz_series)
    out = lcz_series.astype(object).where(valid, np.nan)

    def _fix(v):
        if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
            return int(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    return out.map(_fix)


def lcz_nodata_report(df: pd.DataFrame, cols=("lcz_primary", "lcz_weak250")) -> dict:
    """Count of segments excluded from every by-LCZ statistic (A6), per label column. They stay in the overall Spearman."""
    rep = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        n_zero = int(s.astype(str).str.strip().isin(["0", "0.0"]).sum())
        n_missing = int(s.isna().sum())
        rep[c] = dict(n_total=int(len(s)), n_zero_nodata=n_zero, n_missing=n_missing,
                      n_excluded_from_by_lcz=int((~lcz_valid_mask(s)).sum()), n_valid=int(lcz_valid_mask(s).sum()))
    return rep


def pool_lcz_series(lcz_series: pd.Series, min_n: int = MIN_CLASS_N) -> pd.Series:
    """Pool classes with < min_n segments into 'other built' (1-10) or 'other natural' (>=11 or letters).

    Amendment A6: LCZ 0 (raster no-data) and missing labels are NOT classes. They become NaN here (so every by-LCZ
    statistic drops them) and do not take part in the class counts used for pooling.
    """
    s = mask_lcz_nodata(lcz_series)
    counts = s.value_counts(dropna=True)
    small_classes = set(counts[counts < min_n].index)

    def _remap(val):
        if pd.isna(val):
            return val
        if val in small_classes:
            try:
                ival = int(val)
                if 1 <= ival <= 10:
                    return POOL_BUILT
                else:
                    return POOL_NAT
            except (ValueError, TypeError):
                # String labels like 'A', 'B', etc.
                return POOL_NAT
        return val

    return s.apply(_remap)


def assign_2km_blocks(df: pd.DataFrame, x_col: str = "x_mid", y_col: str = "y_mid") -> pd.DataFrame:
    """Assign 2 km spatial block IDs based on segment midpoint coordinates in EPSG:5253."""
    bx = np.floor(df[x_col].to_numpy(float) / BLOCK_M).astype(int)
    by = np.floor(df[y_col].to_numpy(float) / BLOCK_M).astype(int)
    df["bx_2k"] = bx
    df["by_2k"] = by
    df["block_2k"] = bx.astype(str) + "_" + by.astype(str)
    return df


def block_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float, int, int]:
    """Compute Spearman rho with 95% spatial block bootstrap confidence interval.

    Returns:
        (rho_point, ci_lower, ci_upper, n_valid, n_blocks)
    """
    ok = np.isfinite(x) & np.isfinite(y) & (blocks != "") & pd.notna(blocks)
    x_valid = x[ok]
    y_valid = y[ok]
    b_valid = blocks[ok]

    n_valid = len(x_valid)
    if n_valid < 3:
        return float("nan"), float("nan"), float("nan"), n_valid, 0

    # Point estimate
    res = stats.spearmanr(x_valid, y_valid)
    rho_point = float(res.statistic if hasattr(res, "statistic") else res[0])

    unique_blocks = np.unique(b_valid)
    n_blocks = len(unique_blocks)

    if n_blocks < 2:
        # Cannot resample blocks with replacement meaningfully
        return rho_point, float("nan"), float("nan"), n_valid, n_blocks

    # Pre-index rows for each block
    block_indices = [np.where(b_valid == b)[0] for b in unique_blocks]

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(n_boot):
            # Sample blocks with replacement
            b_draw = rng.integers(0, n_blocks, size=n_blocks)
            counts = np.bincount(b_draw, minlength=n_blocks)
            mult_blocks = np.repeat(np.arange(n_blocks), counts)
            sampled_row_indices = np.concatenate([block_indices[i] for i in mult_blocks])

            xs = x_valid[sampled_row_indices]
            ys = y_valid[sampled_row_indices]

            # Calculate Spearman rho on resample
            r_val = stats.spearmanr(xs, ys)
            boot_rhos[r] = float(r_val.statistic if hasattr(r_val, "statistic") else r_val[0])

    finite_boots = boot_rhos[np.isfinite(boot_rhos)]
    if len(finite_boots) == 0:
        return rho_point, float("nan"), float("nan"), n_valid, n_blocks

    ci_lower = float(np.percentile(finite_boots, 2.5))
    ci_upper = float(np.percentile(finite_boots, 97.5))

    return rho_point, ci_lower, ci_upper, n_valid, n_blocks


def run_spearman_analysis(
    df: pd.DataFrame,
    lcz_col: str = "lcz_primary",
    pool: bool = True,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute Spearman rho(LST, M3) and rho(LST, M1) overall and across LCZ classes."""
    work_df = df.copy()
    # A6: LCZ 0 (raster no-data) / missing is excluded from every by-LCZ statistic but kept in the overall Spearman.
    nodata = lcz_nodata_report(work_df, (lcz_col,))
    if pool:
        work_df["lcz_eval"] = pool_lcz_series(work_df[lcz_col])
    else:
        work_df["lcz_eval"] = mask_lcz_nodata(work_df[lcz_col])

    records = []

    # 1. Overall
    for metric_name, col_ped in [("M1 (UTCI_b15)", "m1"), ("M3 (Tmrt_b15)", "m3")]:
        rho, lo, hi, n_obs, n_blk = block_bootstrap_spearman(
            x=work_df["lst_med"].to_numpy(float),
            y=work_df[col_ped].to_numpy(float),
            blocks=work_df["block_2k"].to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )
        records.append({
            "scope": "overall",
            "lcz_label": "ALL",
            "metric_ped": metric_name,
            "rho": rho,
            "ci95_lower": lo,
            "ci95_upper": hi,
            "ci95_excludes_zero": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)) if np.isfinite(lo) and np.isfinite(hi) else False,
            "n_segments": n_obs,
            "n_blocks": n_blk,
        })

    # 2. By LCZ
    classes = sorted(work_df["lcz_eval"].dropna().unique(), key=lambda v: str(v))
    for c in classes:
        sub = work_df[work_df["lcz_eval"] == c]
        c_label = LCZ_NAMES.get(c, f"LCZ {c}")
        for metric_name, col_ped in [("M1 (UTCI_b15)", "m1"), ("M3 (Tmrt_b15)", "m3")]:
            rho, lo, hi, n_obs, n_blk = block_bootstrap_spearman(
                x=sub["lst_med"].to_numpy(float),
                y=sub[col_ped].to_numpy(float),
                blocks=sub["block_2k"].to_numpy(),
                n_boot=n_boot,
                seed=seed,
            )
            records.append({
                "scope": f"by_lcz ({lcz_col})",
                "lcz_label": c_label,
                "metric_ped": metric_name,
                "rho": rho,
                "ci95_lower": lo,
                "ci95_upper": hi,
                "ci95_excludes_zero": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)) if np.isfinite(lo) and np.isfinite(hi) else False,
                "n_segments": n_obs,
                "n_blocks": n_blk,
            })

    out = pd.DataFrame(records)
    out.attrs["lcz_nodata_excluded"] = nodata
    return out


def evaluate_compact_paradox_rule(
    df: pd.DataFrame,
    lcz_col: str = "lcz_primary",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Pre-analysis lock decision rule for H3:

    'The "compact paradox" counts as observed only if rho(LST, M1) < 0 within LCZ 2/3
    with a 95 % CI that excludes 0.'
    """
    # Subset to LCZ 2 and 3 (A6: LCZ 0 = raster no-data is never in this subset)
    # Note: handle both int and str representations of LCZ
    is_lcz23 = df[lcz_col].isin([2, 3, "2", "3", "2_compact_midrise", "3_compact_lowrise"])
    sub_pooled = df[is_lcz23]

    rho, lo, hi, n_obs, n_blk = block_bootstrap_spearman(
        x=sub_pooled["lst_med"].to_numpy(float),
        y=sub_pooled["m1"].to_numpy(float),
        blocks=sub_pooled["block_2k"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )

    # Decision rule: rho < 0 and 95% CI excludes 0 (i.e. upper CI < 0)
    observed = bool(np.isfinite(hi) and (hi < 0.0) and (rho < 0.0))

    result = {
        "rule": "The compact paradox counts as observed only if rho(LST, M1) < 0 within LCZ 2/3 with a 95% CI that excludes 0.",
        "target_class": "LCZ 2/3 (compact midrise & compact lowrise)",
        "lcz_col": lcz_col,
        "rho": rho,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "ci95_excludes_zero": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)) if np.isfinite(lo) and np.isfinite(hi) else False,
        "n_segments": n_obs,
        "n_blocks": n_blk,
        "compact_paradox_observed": observed,
        "decision": "SUPPORTED" if observed else "NOT SUPPORTED",
        "lcz_nodata_excluded_A6": lcz_nodata_report(df, (lcz_col,)),
    }

    # Also compute individually for LCZ 2 and LCZ 3 as informative diagnostics
    for target_val, target_name in [(2, "LCZ 2 (compact midrise)"), (3, "LCZ 3 (compact lowrise)")]:
        sub_single = df[df[lcz_col].isin([target_val, str(target_val)])]
        rho_s, lo_s, hi_s, n_s, n_blk_s = block_bootstrap_spearman(
            x=sub_single["lst_med"].to_numpy(float),
            y=sub_single["m1"].to_numpy(float),
            blocks=sub_single["block_2k"].to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )
        result[f"diagnostic_{target_val}"] = {
            "name": target_name,
            "rho": rho_s,
            "ci95_lower": lo_s,
            "ci95_upper": hi_s,
            "ci95_excludes_zero": bool((lo_s > 0 and hi_s > 0) or (lo_s < 0 and hi_s < 0)) if np.isfinite(lo_s) and np.isfinite(hi_s) else False,
            "n_segments": n_s,
            "n_blocks": n_blk_s,
        }

    return result
