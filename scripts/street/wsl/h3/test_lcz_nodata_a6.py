"""Unit test for amendment A6 (WP3.7): LCZ 0 (raster no-data) is excluded from every by-LCZ H3 statistic and counted,
but stays in the overall Spearman. In-memory only, nothing is written (rule 6). Run from the project root:

    python -m pytest scripts/street/wsl/h3 -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.street.wsl.h3.bootstrap import (
    block_bootstrap_spearman,
    lcz_nodata_report,
    pool_lcz_series,
    run_spearman_analysis,
)
from scripts.street.wsl.h3.decoupling import compute_decoupled_shares_by_lcz, identify_decoupled_segments


def _toy(n_zero_primary=40, n_zero_weak=25, n_nan_weak=10, seed=7):
    """SYNTHETIC toy table (is_synthetic=True): 120 LCZ-2, 120 LCZ-3 segments, plus LCZ-0 segments with an extreme,
    deliberately different LST-M1 relation so that leaking them into a class would visibly change that class."""
    rng = np.random.default_rng(seed)
    rows = []
    for lcz, k in ((2, 120), (3, 120)):
        lst = rng.normal(40, 3, k)
        rows.append(pd.DataFrame(dict(lcz_primary=lcz, lcz_weak250=lcz, lst_med=lst, m1=30 + 0.3 * lst + rng.normal(0, 1, k))))
    lst0 = rng.normal(40, 3, n_zero_primary)
    rows.append(pd.DataFrame(dict(lcz_primary=0, lcz_weak250=0, lst_med=lst0, m1=60 - 0.8 * lst0 + rng.normal(0, 0.1, n_zero_primary))))
    df = pd.concat(rows, ignore_index=True)
    df["m3"] = df["m1"] * 1.5 + rng.normal(0, 1, len(df))
    # weak250: some LCZ-2 rows become 0, some become NaN
    idx2 = df.index[df["lcz_primary"] == 2]
    df.loc[idx2[:n_zero_weak], "lcz_weak250"] = 0
    df["lcz_weak250"] = df["lcz_weak250"].astype(float)
    df.loc[idx2[n_zero_weak:n_zero_weak + n_nan_weak], "lcz_weak250"] = np.nan
    df["block_2k"] = [f"{i % 12}_0" for i in range(len(df))]
    df["is_synthetic"] = True
    return df


def test_report_counts_zero_and_missing():
    df = _toy()
    rep = lcz_nodata_report(df, ("lcz_primary", "lcz_weak250"))
    assert rep["lcz_primary"]["n_zero_nodata"] == 40 and rep["lcz_primary"]["n_excluded_from_by_lcz"] == 40
    assert rep["lcz_weak250"]["n_zero_nodata"] == 25 + 40       # 25 recoded + the 40 LCZ-0-primary rows
    assert rep["lcz_weak250"]["n_missing"] == 10
    assert rep["lcz_weak250"]["n_excluded_from_by_lcz"] == 65 + 10
    assert rep["lcz_primary"]["n_valid"] + rep["lcz_primary"]["n_excluded_from_by_lcz"] == len(df)


def test_pooling_never_creates_a_class_from_zero():
    # LCZ 0 with < 30 members must NOT be pooled into 'other natural' (the pre-A6 behaviour) but dropped.
    s = pd.Series([2] * 50 + [0] * 5 + [7] * 3)
    out = pool_lcz_series(s)
    assert out.isna().sum() == 5
    assert (out == 102).sum() == 0
    assert (out == 101).sum() == 3          # LCZ 7 (<30) -> other built
    assert 0 not in set(out.dropna())


def test_by_lcz_rows_exclude_zero_overall_keeps_it():
    df = _toy()
    res = run_spearman_analysis(df, lcz_col="lcz_primary", pool=True, n_boot=40)
    overall = res[(res["scope"] == "overall") & res["metric_ped"].str.contains("M1")].iloc[0]
    assert overall["n_segments"] == len(df)                       # overall Spearman keeps LCZ-0 segments
    by = res[res["scope"] != "overall"]
    assert by["n_segments"].sum() == 2 * (240)                    # 2 metrics x 240 valid segments; no LCZ-0 anywhere
    assert not any("LCZ 0" in str(l) for l in by["lcz_label"])
    assert res.attrs["lcz_nodata_excluded"]["lcz_primary"]["n_excluded_from_by_lcz"] == 40
    # independent check, could fail: LCZ-2 rho from the function equals rho computed directly on LCZ-2 rows only
    sub = df[df["lcz_primary"] == 2]
    direct = block_bootstrap_spearman(sub["lst_med"].to_numpy(float), sub["m1"].to_numpy(float),
                                      sub["block_2k"].to_numpy(), n_boot=40)[0]
    row = by[(by["lcz_label"].str.startswith("LCZ 2")) & by["metric_ped"].str.contains("M1")].iloc[0]
    assert abs(row["rho"] - direct) < 1e-12
    assert row["rho"] > 0.5                                       # the planted positive relation; LCZ-0 rows (negative) not mixed in


def test_overall_differs_from_valid_only():
    # The LCZ-0 rows carry a planted opposite relation, so keeping them in 'overall' must change the overall rho.
    df = _toy()
    full = block_bootstrap_spearman(df["lst_med"].to_numpy(float), df["m1"].to_numpy(float), df["block_2k"].to_numpy(), n_boot=20)[0]
    v = df[df["lcz_primary"] > 0]
    valid_only = block_bootstrap_spearman(v["lst_med"].to_numpy(float), v["m1"].to_numpy(float), v["block_2k"].to_numpy(), n_boot=20)[0]
    res = run_spearman_analysis(df, n_boot=20)
    ov = res[(res["scope"] == "overall") & res["metric_ped"].str.contains("M1")].iloc[0]["rho"]
    assert abs(ov - full) < 1e-12 and abs(ov - valid_only) > 1e-3


def test_weak250_zero_and_nan_excluded():
    df = _toy()
    res = run_spearman_analysis(df, lcz_col="lcz_weak250", pool=True, n_boot=20)
    by = res[res["scope"] != "overall"]
    n_valid = int(((df["lcz_weak250"] > 0)).sum())
    assert by["n_segments"].sum() == 2 * n_valid
    assert res.attrs["lcz_nodata_excluded"]["lcz_weak250"]["n_missing"] == 10


def test_decoupled_shares_by_lcz_exclude_zero():
    df = _toy()
    dec = identify_decoupled_segments(df)
    sh = compute_decoupled_shares_by_lcz(dec, lcz_col="lcz_primary", pool=True)
    all_row = sh[sh["group"] == "ALL"].iloc[0]
    assert all_row["n_segments"] == len(df) and all_row["n_lcz_nodata_excluded"] == 40
    assert sh[sh["group"] != "ALL"]["n_segments"].sum() == 240
    assert "0" not in set(sh["group"])
