"""WP3.7 tests (pytest). In-memory data only; nothing is written. Anything built here is SYNTHETIC (is_synthetic=True).

Run from the project root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m pytest scripts/street/stats/test_wp37.py -q

  A8(1) sun azimuth at 14:30 local, checked against an independent low-precision solar-position formula (Meeus / NOAA)
  A8(2) H4 'no enclosing buildings' class, terciles on non-NaN only, no imputation of NaN features
  A8(3) nomotorway sensitivity tables (run on the current PARTIAL tables)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
import h4_shade_breeze as h4  # noqa: E402


# ------------------------------------------------------------------------------------------------ A8(1)
def _meeus_azimuth(date: str, hh_utc: float, lat=38.42, lon=27.14) -> float:
    """Independent solar azimuth (deg from north, clockwise): low-precision Astronomical Almanac / Meeus formulas."""
    y, m, d = map(int, date.split("-"))
    a = (14 - m) // 12
    yy, mm = y + 4800 - a, m + 12 * a - 3
    jdn = d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    jd = jdn - 0.5 + hh_utc / 24.0
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    eps = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    H = math.radians((gmst * 15 + lon) % 360) - ra
    la = math.radians(lat)
    az = math.degrees(math.atan2(math.sin(H), math.cos(H) * math.sin(la) - math.tan(dec) * math.cos(la))) + 180.0
    return az % 360


@pytest.mark.parametrize("day", ["IMBAT", "CALM"])
def test_sun_azimuth_is_1430_local(day):
    fo = cm.forcing_summary(day)
    az_ref = _meeus_azimuth(cm.DAY_DATES[day], 11.5)           # 14:30 UTC+3 = 11:30 UTC
    az_old_ref = _meeus_azimuth(cm.DAY_DATES[day], 12.0)       # 15:00 UTC+3 = 12:00 UTC
    assert abs(fo["sun_az"] - az_ref) < 0.5, (fo["sun_az"], az_ref)            # independent agreement
    assert abs(fo["sun_az_1500_legacy"] - az_old_ref) < 0.5
    assert abs(fo["sun_az"] - fo["sun_az_1500_legacy"]) > 5.0                  # the change is real
    assert fo["sun_az"] < fo["sun_az_1500_legacy"]                             # afternoon: azimuth increases with time


def _met_cols(met: pd.DataFrame) -> pd.DataFrame:
    """Fill the metric columns build_frame reads (SYNTHETIC constants; only utci_b15 is used by the tests)."""
    for c in ("m2_strong", "m2_vstrong", "tmrt_b15", "shade_frac_10_18", "utci_unif_b15"):
        met[c] = 1.0
    return met


def test_ang_sun_uses_1430_azimuth():
    seg = pd.DataFrame(dict(seg_id=[1, 2], azimuth_deg=[0.0, 90.0], x_mid=[1.0, 2.0], y_mid=[1.0, 2.0],
                            district=["a", "b"], lcz_primary=[2, 3]))
    met = _met_cols(pd.DataFrame(dict(seg_id=[1, 2], utci_b15=[30.0, 31.0])))
    df = cm.build_frame(seg, met, "IMBAT")
    az14 = df.attrs["forcing"]["sun_az"]
    np.testing.assert_allclose(df["ang_sun"], cm.axis_angle(np.array([0.0, 90.0]), az14))
    assert not np.allclose(df["ang_sun"], cm.axis_angle(np.array([0.0, 90.0]), df.attrs["forcing"]["sun_az_1500_legacy"]))


# ------------------------------------------------------------------------------------------------ A8(2)
def _h4_toy(n=400, seed=3):
    rng = np.random.default_rng(seed)
    hw = rng.uniform(0.1, 2.0, n)
    hw[rng.random(n) < 0.24] = np.nan                            # ~24 % without enclosing buildings
    return pd.DataFrame(dict(hw_ratio=hw, orient_class=rng.choice(["aligned", "oblique", "perpendicular"], n),
                             shade_term=rng.normal(0, 1, n), breeze_term=rng.normal(0, 0.3, n), is_synthetic=True))


def test_h4_hw_nan_forms_own_class_and_terciles_exclude_nan():
    d = _h4_toy()
    nn = d["hw_ratio"].dropna()
    edges = (float(nn.quantile(1 / 3)), float(nn.quantile(2 / 3)))
    cls = h4.hw_class(d["hw_ratio"], edges)
    assert (cls == h4.NO_BLD).sum() == d["hw_ratio"].isna().sum()
    assert (cls[d["hw_ratio"].isna()] == h4.NO_BLD).all() and (cls[d["hw_ratio"].notna()] != h4.NO_BLD).all()
    # terciles computed on non-NaN only: the three classes hold ~1/3 of the NON-NaN segments each
    for t in ("low", "mid", "high"):
        assert abs((cls == t).sum() / len(nn) - 1 / 3) < 0.02
    # reference that could fail: terciles would shift if NaN were imputed (e.g. as 0)
    edges_imputed0 = (float(d["hw_ratio"].fillna(0).quantile(1 / 3)), float(d["hw_ratio"].fillna(0).quantile(2 / 3)))
    assert edges != edges_imputed0
    tab = h4.orientation_table(d, edges, [], "TEST")
    assert tab["n"].sum() == len(d)                              # every segment lands in exactly one cell
    nb = tab[tab["hw_tercile"] == h4.NO_BLD]
    assert nb["n"].sum() == d["hw_ratio"].isna().sum() and len(nb) == 3
    row = nb[nb["orient"] == "aligned"].iloc[0]
    m = (d["orient_class"] == "aligned") & d["hw_ratio"].isna()
    assert abs(row["mean_shade"] - d.loc[m, "shade_term"].mean()) < 1e-12


def test_h4_bootstrap_weights_aligned_with_nan_class():
    d = _h4_toy()
    nn = d["hw_ratio"].dropna()
    edges = (float(nn.quantile(1 / 3)), float(nn.quantile(2 / 3)))
    blocks = np.arange(len(d)) // 20
    ub, inv, counts = cm.boot_block_counts(blocks, 30, 11)
    W = [counts[r][inv] for r in range(counts.shape[0])]
    tab = h4.orientation_table(d, edges, W, "TEST")
    # CIs exist for the NaN class too (the earlier code dropped the CIs whenever NaN-hw rows were removed)
    assert "mean_shade_lo" in tab.columns and tab[tab["hw_tercile"] == h4.NO_BLD]["mean_shade_lo"].notna().any()
    ok = tab.dropna(subset=["mean_shade_lo"])
    assert (ok["mean_shade_lo"] <= ok["mean_shade_hi"]).all()
    did = h4.did_effect(d, edges, W)
    assert set(did) == {"shade_term", "breeze_term"}


def test_no_imputation_nan_features_reach_lightgbm():
    """build_frame keeps NaN hw_ratio; LightGBM learns from the missingness natively (planted: y depends on NaN-ness)."""
    rng = np.random.default_rng(5)
    n = 1500
    hw = rng.uniform(0.2, 2.0, n)
    miss = rng.random(n) < 0.24
    seg = pd.DataFrame(dict(seg_id=np.arange(n), azimuth_deg=rng.uniform(0, 180, n), x_mid=rng.uniform(0, 20000, n),
                            y_mid=rng.uniform(0, 20000, n), district=rng.choice(list("abc"), n),
                            lcz_primary=rng.choice([2, 3, 5], n)))
    seg["hw_ratio"] = np.where(miss, np.nan, hw)
    for f in cm.FEATURES:
        if f not in seg.columns and f not in ("az_sin", "az_cos", "ang_sun", "ang_wind"):
            seg[f] = rng.normal(0, 1, n)
    y = 30 + 5.0 * miss + 0.2 * rng.normal(size=n)               # planted: NaN-hw segments are 5 C hotter
    met = _met_cols(pd.DataFrame(dict(seg_id=np.arange(n), utci_b15=y)))
    df = cm.build_frame(seg, met, "IMBAT")
    df["is_synthetic"] = True
    cm.assign_shared_folds({"IMBAT": df})
    assert int(df["hw_ratio"].isna().sum()) == int(miss.sum())                  # build_frame did not fill anything
    assert int(df[cm.FEATURES].isna().sum().sum()) == int(miss.sum())            # no other feature became NaN or filled
    pred = cm.lgb_oof(df, cm.FEATURES, n_jobs=1)
    assert cm.wr2(df["m1"].to_numpy(), pred) > 0.9                               # only possible if NaN is used natively
    mk = df["hw_ratio"].isna().to_numpy()
    assert abs(pred[mk].mean() - pred[~mk].mean() - 5.0) < 0.5


def test_no_imputation_calls_in_stats_code():
    """Static scan: no imputer anywhere; every fillna in scripts/street/stats is on a label, weight, count or reference mean."""
    root = Path(__file__).resolve().parent
    banned = ("SimpleImputer", "IterativeImputer", "KNNImputer", "impute")
    allowed_fillna_ctx = ("lcz", 'm1"].mean', "n_ps", "unstack", ".fillna(0).max(axis=1)")
    for p in root.glob("*.py"):
        if p.name == Path(__file__).name:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            assert not any(b in line for b in banned), f"{p.name}:{i}: {line.strip()}"
            if "fillna(" in line:
                assert any(a in line for a in allowed_fillna_ctx), f"{p.name}:{i}: unexpected fillna: {line.strip()}"


# ------------------------------------------------------------------------------------------------ A8(3)
SENS_TEST_ROOT = cm.OUT_ROOT / "sens_PARTIAL_test" / "nomotorway"


@pytest.mark.skipif(not (SENS_TEST_ROOT / "segments_nomotorway.gpkg").exists(),
                    reason="run extract_sensitivity --variant nomotorway --out-root outputs/street/analysis/sens_PARTIAL_test first")
def test_nomotorway_tables_drop_exactly_otoyol():
    import geopandas as gpd
    prim = gpd.read_file(cm.DATA_ANALYSIS / "segments.gpkg", layer="segments")
    nm = gpd.read_file(SENS_TEST_ROOT / "segments_nomotorway.gpkg", layer="segments")
    n_mw = int((prim["road_class"] == "OTOYOL").sum())
    assert n_mw > 0 and len(nm) == len(prim) - n_mw
    assert (nm["road_class"] != "OTOYOL").all()
    assert set(nm["seg_id"]) == set(prim.loc[prim["road_class"] != "OTOYOL", "seg_id"])
    mw_ids = set(prim.loc[prim["road_class"] == "OTOYOL", "seg_id"])
    for d in cm.DAYS:
        m = pd.read_parquet(SENS_TEST_ROOT / f"seg_metrics_{d}.parquet")
        assert not set(m["seg_id"]) & mw_ids
        assert bool(m["partial_run"].all())                                      # PARTIAL flag inherited
