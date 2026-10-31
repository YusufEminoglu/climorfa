"""WP3.1/3.2/3.4 shared helpers: data loading, features, spatial-block folds, block bootstrap, LightGBM, UTCI.

Implements pre_analysis_lock.md section 1-3 exactly. Nothing here is tuned after seeing data.
All fixed choices are in the CONSTANTS block and are echoed into every results JSON.

Python: C:/Users/YE/miniforge3/envs/climorfa-street/python.exe
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ANALYSIS = PROJECT_ROOT / "data" / "street" / "analysis"
OUT_ROOT = PROJECT_ROOT / "outputs" / "street" / "analysis"
MET_FILES = {
    "IMBAT": PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt",
    "CALM": PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt",
}
# WP2.3b / lock A7 item 3: use the corrected met file (Wd = -999 in VRB-flagged hours) when it exists.
for _d, _p in list(MET_FILES.items()):
    _v2 = _p.with_name(_p.stem + "_v2.txt")
    if _v2.exists():
        MET_FILES[_d] = _v2
DAYS = ("IMBAT", "CALM")
DAY_DATES = {"IMBAT": "2022-07-05", "CALM": "2023-07-26"}

# ----------------------------------------------------------------------------------------------
# CONSTANTS (pre-analysis lock; do not edit after data have been seen)
# ----------------------------------------------------------------------------------------------
SEED = 20261030            # lock: fixed seed
BLOCK_M = 2000.0           # lock: 2 km spatial blocks
N_FOLDS = 5                # lock: 5-fold spatial-block CV
N_BOOT = 1000              # lock: 1000 block-bootstrap resamples, 95 % CI
MIN_CLASS_N = 30           # lock: LCZ classes with < 30 segments are pooled
POOL_BUILT, POOL_NAT = 101, 102   # "other built" (LCZ 1-10) / "other natural" (LCZ 11-17, A-G)
MAX_PROCS = 4              # task constraint: at most 4 CPU processes (GPU job running; WP3.7)
UTCI_STRONG, UTCI_VSTRONG = 32.0, 38.0
BAND_MAIN = 15
BANDS_M2 = list(range(10, 19))
# LightGBM: the lock fixes no hyper-parameters; these are fixed a priori and never tuned (open issue in report).
LGB_PARAMS = dict(
    objective="regression", n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
    n_jobs=1, verbose=-1,
)
LAT, LON = 38.42, 27.14   # Izmir centre, for solar azimuth only
SUN_TIME_UTC = "11:30"    # lock A8(1): 14:30 local (UTC+3) = midpoint of the label-15 interval (14:00-15:00)

FAMILIES = {
    "shade_geometry": ["svf_med", "hw_ratio", "bh_mean30", "bcover30"],
    "vegetation": ["tcover15", "tht15"],
    "orientation": ["az_sin", "az_cos", "ang_sun", "ang_wind"],
    "ventilation": ["ratio_med_b15"],
    "context": ["dist_coast_m", "elev_m", "slope_deg"],
}
FEATURES = [f for fam in FAMILIES.values() for f in fam]
FAMILY_OF = {f: fam for fam, fs in FAMILIES.items() for f in fs}


def constants_dict() -> dict:
    return dict(SEED=SEED, BLOCK_M=BLOCK_M, N_FOLDS=N_FOLDS, N_BOOT=N_BOOT, MIN_CLASS_N=MIN_CLASS_N,
                LGB_PARAMS={k: v for k, v in LGB_PARAMS.items()}, FAMILIES=FAMILIES, LAT=LAT, LON=LON, SUN_TIME_UTC=SUN_TIME_UTC)


def utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------
# Forcing and geometry of the design days
# ----------------------------------------------------------------------------------------------
def load_met(day: str) -> pd.DataFrame:
    df = pd.read_csv(MET_FILES[day], sep=r"\s+")
    return df[(df["it"] >= 6) & (df["it"] <= 21)].reset_index(drop=True)


def forcing_summary(day: str) -> dict:
    """Ta, RH, U10, Wd at band 15 (met row it=15, exactly as the UTCI extractor uses it) + wind/sun angles.

    Wind direction for the orientation features (WP2.3b, lock A7 item 3) = circular mean of the direction of the NON-VRB
    METAR reports with local time in (14:00, 17:00] (the span of bands 15-17, band = hour-ending); unit vectors,
    unweighted. R = resultant length of those unit vectors (0..1; R < 0.5 -> CALM orientation results descriptive only,
    lock A7). wd15 is NaN and vrb15 True when the band-15 hour is VRB-flagged (the met-file Wd is then -999).
    Sun azimuth = solar azimuth at 14:30 local (UTC+3) = 11:30 UTC, the midpoint of the label-15 interval (hour-ending
    14:00-15:00), matching SOLWEIG-GPU's midpoint sun position (lock amendment A8 item 1). The former 15:00 stamp value
    is kept as sun_az_1500_legacy for the record only; it is not used by any feature.
    """
    from scripts.street.utci.vrb_utils import nonvrb_circular_mean, read_vrb_flags
    met = load_met(day)
    row = met[met["it"] == BAND_MAIN].iloc[0]
    vrb15 = bool(read_vrb_flags(day).get(BAND_MAIN, False))
    wd15 = float("nan") if vrb15 else float(row["Wd"])
    ref = nonvrb_circular_mean(day, 14.0, 17.0)
    wd_mean = ref["wd_mean_deg"]
    import pvlib
    sp = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([pd.Timestamp(f"{DAY_DATES[day]} {SUN_TIME_UTC}", tz="UTC")]), LAT, LON)
    sp_legacy = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([pd.Timestamp(f"{DAY_DATES[day]} 12:00", tz="UTC")]), LAT, LON)
    return dict(day=day, ta=float(row["Td"]), rh=float(row["RH"]), u10=float(row["Wind"]), wd15=wd15, vrb15=vrb15,
                wd_mean_15_17=wd_mean, wd_ref_R=ref["R"], wd_ref_n_reports=ref["n"],
                wd_ref_n_vrb_excluded=ref["n_vrb_excluded"], wd_ref_report_times=ref["times"], sun_az=float(sp["azimuth"].iloc[0]), sun_elev=float(sp["elevation"].iloc[0]),
                sun_time_local="14:30 (UTC+3)", sun_az_1500_legacy=float(sp_legacy["azimuth"].iloc[0]))


def axis_angle(az_deg, ref_deg):
    """Smallest angle (0-90 deg) between two undirected axes given as bearings."""
    d = np.abs((np.asarray(az_deg, float) - float(ref_deg)) % 180.0)
    return np.minimum(d, 180.0 - d)


def orientation_class(ang_wind):
    a = np.asarray(ang_wind, float)
    out = np.where(a <= 22.5, "aligned", np.where(a >= 67.5, "perpendicular", "oblique")).astype(object)
    out[np.isnan(a)] = None
    return out


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def load_segments(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df
    import geopandas as gpd
    g = gpd.read_file(path, layer="segments")
    mid = g.geometry.interpolate(0.5, normalized=True)
    df = pd.DataFrame(g.drop(columns="geometry"))
    df["x_mid"] = mid.x.to_numpy()
    df["y_mid"] = mid.y.to_numpy()
    # geometric bearing of the chord (independent check of azimuth_deg, see sanity_checks)
    xs0 = g.geometry.apply(lambda l: l.coords[0][0]).to_numpy()
    ys0 = g.geometry.apply(lambda l: l.coords[0][1]).to_numpy()
    xs1 = g.geometry.apply(lambda l: l.coords[-1][0]).to_numpy()
    ys1 = g.geometry.apply(lambda l: l.coords[-1][1]).to_numpy()
    df["chord_bearing_deg"] = np.degrees(np.arctan2(xs1 - xs0, ys1 - ys0)) % 180.0
    return df


def default_paths(synthetic: bool, tag: str = "") -> tuple[Path, dict]:
    if synthetic:
        base = OUT_ROOT / "h1" / "synthetic_inputs"
        seg = base / f"segments_SYNTHETIC{tag}.parquet"
        met = {d: base / f"seg_metrics_{d}_SYNTHETIC{tag}.parquet" for d in DAYS}
    else:
        seg = DATA_ANALYSIS / "segments.gpkg"
        met = {d: DATA_ANALYSIS / f"seg_metrics_{d}.parquet" for d in DAYS}
    return seg, met


def lcz_pool(lcz: pd.Series, min_n: int = MIN_CLASS_N) -> pd.Series:
    """Classes with < min_n units are pooled into 'other built' (1-10) or 'other natural' (11-17)."""
    s = lcz.astype("Int64")
    counts = s.value_counts()
    small = set(counts[counts < min_n].index.tolist())
    out = s.copy()
    for k in small:
        out[s == k] = POOL_BUILT if (pd.notna(k) and 1 <= k <= 10) else POOL_NAT
    return out


def build_frame(seg: pd.DataFrame, met: pd.DataFrame, day: str, lcz_col: str = "lcz_primary") -> pd.DataFrame:
    """Merge segment attributes and day metrics; derive metrics M1-M5, features, block id."""
    df = seg.merge(met, on="seg_id", how="inner", suffixes=("", "_m"))
    fo = forcing_summary(day)
    df["m1"] = df["utci_b15"]
    df["m2"] = df["m2_strong"]
    df["m2v"] = df["m2_vstrong"]
    df["m3"] = df["tmrt_b15"]
    df["m4"] = df["shade_frac_10_18"]
    df["m5"] = df["utci_unif_b15"]
    az = df["azimuth_deg"].to_numpy(float)
    df["az_sin"] = np.sin(np.deg2rad(az))
    df["az_cos"] = np.cos(np.deg2rad(az))
    df["ang_sun"] = axis_angle(az, fo["sun_az"])
    df["ang_wind"] = axis_angle(az, fo["wd_mean_15_17"])
    df["orient_class"] = orientation_class(df["ang_wind"])
    df["bx"] = np.floor(df["x_mid"] / BLOCK_M).astype(int)
    df["by"] = np.floor(df["y_mid"] / BLOCK_M).astype(int)
    df["block"] = df["bx"].astype(str) + "_" + df["by"].astype(str)
    df["district"] = df["district"].astype(str)
    df = df[np.isfinite(df["m1"])].reset_index(drop=True)
    # Amendment A6: LCZ value 0 (or missing) is the raster no-data value, not a class. Such segments stay in the table
    # (they are used by non-LCZ analyses: H2 morphology model, H4) but are flagged lcz_valid=False and excluded by every
    # LCZ-based statistic (H1 LMM and eta2, LCZ-only CV, H2 LCZ model / within-LCZ residual).
    lab = df[lcz_col].fillna(0).astype(int)
    df["lcz_primary"] = lab            # downstream code uses lcz_primary / lcz; lcz_col selects the label source
    df["lcz_valid"] = lab > 0
    df["lcz"] = 0
    df.loc[df["lcz_valid"], "lcz"] = lcz_pool(lab[df["lcz_valid"]]).astype(int)
    df.attrs["n_lcz_excluded"] = int((~df["lcz_valid"]).sum())
    df.attrs["forcing"] = fo
    df.attrs["day"] = day
    return df


def load_frames(synthetic: bool = False, tag: str = "", seg_path=None, met_paths=None,
                lcz_col: str = "lcz_primary") -> dict[str, pd.DataFrame]:
    sp, mp = default_paths(synthetic, tag)
    if seg_path:
        sp = Path(seg_path)
    if met_paths:
        mp = {d: Path(p) for d, p in met_paths.items()}
    seg = load_segments(sp)
    if synthetic:
        assert "SYNTHETIC" in str(sp).upper() and seg["is_synthetic"].astype(bool).all(), "synthetic guard (rule 6)"
    else:
        # rule 6: a real run must never read synthetic inputs
        from scripts.street.integrity.synthetic import assert_real  # noqa
        assert_real(sp)
    frames = {}
    for d in DAYS:
        if not Path(mp[d]).exists():
            continue
        met = pd.read_parquet(mp[d])
        if synthetic:
            assert "SYNTHETIC" in str(mp[d]).upper() and met["is_synthetic"].astype(bool).all()
        else:
            assert_real(mp[d])
            assert "is_synthetic" not in met.columns or not met["is_synthetic"].astype(bool).any()
        f = build_frame(seg, met, d, lcz_col)
        f.attrs["partial_run"] = bool(met["partial_run"].any()) if "partial_run" in met else None
        f.attrs["n_tiles"] = int(met["n_tiles_at_extraction"].max()) if "n_tiles_at_extraction" in met else None
        f.attrs["is_synthetic"] = bool(synthetic)
        frames[d] = f
    if not frames:
        raise FileNotFoundError("no seg_metrics tables found")
    assign_shared_folds(frames)
    return frames


def label_for(frame: pd.DataFrame) -> str:
    """Result label required by the lock/schema: PARTIAL (n tiles) or FINAL, or SYNTHETIC."""
    if frame.attrs.get("is_synthetic"):
        return "SYNTHETIC (planted-effect test, not observed data)"
    if frame.attrs.get("partial_run"):
        return f"PARTIAL ({frame.attrs.get('n_tiles')} tiles)"
    return f"FINAL ({frame.attrs.get('n_tiles')} tiles)"


# ----------------------------------------------------------------------------------------------
# Spatial-block folds and block bootstrap
# ----------------------------------------------------------------------------------------------
def assign_folds_from_counts(block_counts: pd.Series, k: int = N_FOLDS, seed: int = SEED) -> dict:
    """Random block order (fixed seed), each block to the currently lightest fold (balances segment counts)."""
    blocks = sorted(block_counts.index.tolist())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    load = np.zeros(k)
    out = {}
    for i in order:
        b = blocks[i]
        f = int(np.argmin(load))
        out[b] = f
        load[f] += float(block_counts[b])
    return out


def assign_shared_folds(frames: dict[str, pd.DataFrame]) -> None:
    """Folds are defined on the union of blocks over both days so IMBAT and CALM share identical folds."""
    cnt = pd.concat([f.groupby("block").size() for f in frames.values()], axis=1).fillna(0).max(axis=1)
    fm = assign_folds_from_counts(cnt)
    for f in frames.values():
        f["fold"] = f["block"].map(fm).astype(int)


def boot_block_counts(blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Multiplicity matrix (n_boot, n_blocks): block resampling with replacement, one RNG per resample (parallel-safe).

    Returns (unique_blocks, row_to_block_index, counts). Row weight of resample r = counts[r, row_to_block_index].
    """
    ub, inv = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    counts = np.zeros((n_boot, nb), dtype=np.int32)
    for r in range(n_boot):
        rng = np.random.default_rng([seed, r])
        draw = rng.integers(0, nb, size=nb)
        counts[r] = np.bincount(draw, minlength=nb)
    return ub, inv, counts


def rows_from_counts(inv: np.ndarray, counts_r: np.ndarray) -> np.ndarray:
    """Row indices of one bootstrap sample (rows repeated by their block multiplicity)."""
    w = counts_r[inv]
    return np.repeat(np.arange(len(inv)), w)


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def pct_ci(x) -> dict:
    lo, hi = ci95(x)
    return dict(lo=lo, hi=hi, n_valid=int(np.isfinite(np.asarray(x, float)).sum()))


# ----------------------------------------------------------------------------------------------
# Weighted statistics (used on bootstrap weights)
# ----------------------------------------------------------------------------------------------
def wr2(y, yhat, w=None) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    ybar = np.sum(w * y) / np.sum(w)
    sst = np.sum(w * (y - ybar) ** 2)
    sse = np.sum(w * (y - yhat) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def wvar(x, w=None) -> float:
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = np.sum(w * x) / np.sum(w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def eta_squared(y, g, w=None) -> float:
    """Share of variance of y explained by categorical g (SS_between / SS_total), optional frequency weights."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    g = np.asarray(g)
    ok = (w > 0) & np.isfinite(y)
    y, g, w = y[ok], g[ok], w[ok]
    ybar = np.sum(w * y) / np.sum(w)
    sst = np.sum(w * (y - ybar) ** 2)
    if sst <= 0:
        return float("nan")
    codes, inv = np.unique(g, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y)
    gm = sy / sw
    ssb = np.sum(sw * (gm - ybar) ** 2)
    return float(ssb / sst)


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def lcz_only_oof(df: pd.DataFrame, y: str = "m1", lcz_col: str = "lcz") -> np.ndarray:
    """LCZ-only fixed-effect model (one mean per LCZ class), 5-fold spatial-block CV. Unseen class -> training mean."""
    pred = np.full(len(df), np.nan)       # NaN for lcz_valid == False rows (A6: not part of any LCZ statistic)
    valid = df["lcz_valid"].to_numpy() if "lcz_valid" in df else np.ones(len(df), bool)
    for k in range(N_FOLDS):
        te = (df["fold"] == k).to_numpy()
        tr = (~te) & valid
        te = te & valid
        if te.sum() == 0 or tr.sum() == 0:
            continue
        gm = df.loc[tr].groupby(lcz_col)[y].mean()
        pred[te] = df.loc[te, lcz_col].map(gm).fillna(df.loc[tr, y].mean()).to_numpy()
    return pred


def lgb_fit(X, y, params=None, categorical=None):
    import lightgbm as lgb
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    m = lgb.LGBMRegressor(**p)
    if categorical:
        m.fit(X, y, categorical_feature=categorical)
    else:
        m.fit(X, y)
    return m


def lgb_oof(df: pd.DataFrame, feats: list[str], y: np.ndarray | None = None, categorical=None, contrib=False,
            y_col: str = "m1", n_jobs: int = 1):
    """Spatial-block CV out-of-fold predictions (and optional out-of-fold TreeSHAP contributions).

    Out-of-fold SHAP: each fold's model explains only its held-out segments (pred_contrib=True is LightGBM's
    native TreeSHAP; cross-checked against shap.TreeExplainer in the synthetic test).
    """
    yv = df[y_col].to_numpy(float) if y is None else np.asarray(y, float)
    X = df[feats].copy()
    if categorical:
        for c in categorical:
            X[c] = X[c].astype("category")
    pred = np.full(len(df), np.nan)
    shap_vals = np.full((len(df), len(feats) + 1), np.nan) if contrib else None
    for k in range(N_FOLDS):
        te = (df["fold"] == k).to_numpy()
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        m = lgb_fit(X[tr], yv[tr], params=dict(n_jobs=n_jobs), categorical=categorical)
        pred[te] = m.predict(X[te])
        if contrib:
            shap_vals[te] = m.predict(X[te], pred_contrib=True)
    return (pred, shap_vals) if contrib else pred


# ----------------------------------------------------------------------------------------------
# UTCI
# ----------------------------------------------------------------------------------------------
def utci_vec(ta, tr, v, rh) -> np.ndarray:
    """UTCI via pythermalcomfort (same call as scripts/street/utci/compute_street_utci.py)."""
    from pythermalcomfort.models import utci
    tr = np.atleast_1d(np.asarray(tr, float))
    v = np.atleast_1d(np.asarray(v, float))
    tr, v = np.broadcast_arrays(tr, v)
    res = utci(tdb=float(ta), tr=tr, v=v, rh=float(rh), limit_inputs=False, round_output=False)
    return np.asarray(res.utci, float)


# ----------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------
def out_dir(h: str, synthetic: bool, tag: str = "", suffix: str = "") -> Path:
    d = OUT_ROOT / h / (f"synthetic{tag}" if synthetic else f"results{suffix}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fname(stem: str, ext: str, synthetic: bool, label_frame: pd.DataFrame | None = None) -> str:
    if synthetic:
        return f"{stem}_SYNTHETIC.{ext}"
    if label_frame is not None and label_frame.attrs.get("partial_run"):
        return f"{stem}_PARTIAL.{ext}"
    return f"{stem}.{ext}"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def add_common_args(ap) -> None:
    ap.add_argument("--synthetic", action="store_true", help="use SYNTHETIC test tables (planted effects)")
    ap.add_argument("--tag", default="", help="synthetic scenario tag, e.g. _H1T")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--jobs", type=int, default=MAX_PROCS)
    ap.add_argument("--lcz-col", default="lcz_primary", choices=["lcz_primary", "lcz_weak250"],
                    help="LCZ label source (lock sensitivity: lcz_weak250)")
    ap.add_argument("--seg-path", default=None, help="override segments table (sensitivity runs)")
    ap.add_argument("--imbat-path", default=None)
    ap.add_argument("--calm-path", default=None)
    ap.add_argument("--out-suffix", default="", help="suffix for results folder in sensitivity runs, e.g. _weak250")


def frames_from_args(args) -> dict:
    mp = None
    if args.imbat_path or args.calm_path:
        _, base = default_paths(args.synthetic, args.tag)
        mp = dict(base)
        if args.imbat_path:
            mp["IMBAT"] = args.imbat_path
        if args.calm_path:
            mp["CALM"] = args.calm_path
    return load_frames(synthetic=args.synthetic, tag=args.tag, seg_path=args.seg_path, met_paths=mp,
                       lcz_col=args.lcz_col)


def verdict(flag: bool) -> str:
    return "SUPPORTED" if flag else "NOT SUPPORTED"


def fmt_ci(x: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# make `scripts.street...` importable when run as script
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
