"""WP3.6: Extraction sensitivity machinery (pre_analysis_lock.md section 3 & A7.6).

Variants:
  1. ps10: PS buffer 10 m (rebuild ps_index under data/street/analysis/sens/ps10/, segments_ps10.gpkg)
  2. ps20: PS buffer 20 m (rebuild ps_index under data/street/analysis/sens/ps20/, segments_ps20.gpkg)
  3. p90:  p90 instead of median for M1 (utci_b15) and M3 (tmrt_b15) (same ps_index, primary segments.gpkg)
  4. lcz_filter: lcz_primary from LCZ band 2 (LCZ_Filter per A2) (same ps_index, segments_lcz_filter.gpkg)
  5. nomotorway: drops segments with road_class == 'OTOYOL' (YOLTIP motorway class; lock section 1 and A8 item 3) from
     segments and from seg_metrics before the statistics (same ps_index and rasters, so the remaining rows are
     identical to the primary ones)

Outputs:
  data/street/analysis/sens/<variant>/seg_metrics_<DAY>.parquet
  data/street/analysis/sens/<variant>/segments_<variant>.gpkg (for ps10, ps20, lcz_filter)
  data/street/analysis/sens/segments_<variant>.gpkg (convenience copy)

Usage:
  # Test on currently available completed tiles (PARTIAL):
  python scripts/street/analysis/extract_sensitivity.py --variant all --workers 2

  # Final run on all 427 tiles once the full citywide run finishes:
  python scripts/street/analysis/extract_sensitivity.py --variant all --all-tiles --workers 4

  # Motorway sensitivity only (cheap: filters the primary tables), optionally into another root:
  python scripts/street/analysis/extract_sensitivity.py --variant nomotorway [--out-root outputs/street/analysis/sens_PARTIAL_test]
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure local imports
ANALYSIS_DIR_PARENT = Path(__file__).resolve().parent
if str(ANALYSIS_DIR_PARENT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR_PARENT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

from common import (
    ANALYSIS_DIR,
    BUF_PX,
    DAYS,
    INNER_PX,
    LCZ_TIF,
    LST_TIF,
    MET_FILES,
    MIN_PS_CELLS,
    OUT_DIR,
    PRE_PS_GPKG,
    PS_DIR,
    RATIO16_DIR,
    RES,
    ROOT,
    SEGMENTS_GPKG,
    SHORELINE_GPKG,
    SOLWEIG_DIR,
    TILE_DIR,
    TILE_M,
    TILES_CORE,
    TILES_DONE_CSV,
    UTCI_DIR,
    WEAK_GPKG,
    lower_priority,
    tile_xy,
)
from extract_seg_metrics import (
    HOURS,
    N_TILES_TOTAL,
    NODATA,
    SCHEMA_COLS,
    UTCI_COLS,
    atomic_parquet,
    met_b15,
    tile_paths,
    tile_ready,
)

SENS_DATA_DIR = ANALYSIS_DIR / "sens"
VARIANTS = ("ps10", "ps20", "p90", "lcz_filter", "nomotorway")
MOTORWAY_CLASS = "OTOYOL"     # exact road_class value (YOLTIP) of motorways in segments.gpkg: 630 of 74178 segments

# Global worker state for building PS index
G_WORKER: dict = {}


def mode_int(v: np.ndarray):
    if v.size == 0:
        return None
    u, c = np.unique(v, return_counts=True)
    return u[np.argmax(c)]


def densify_lines(lines: np.ndarray, step: float):
    L = shapely.length(lines)
    n = np.ceil(L / step).astype(np.int64) + 1
    idx = np.repeat(np.arange(len(lines)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    d = L[idx] * k / np.repeat(n - 1, n)
    pts = shapely.line_interpolate_point(lines[idx], d)
    return shapely.get_coordinates(pts), idx


def load_weak_lookup():
    w = gpd.read_file(WEAK_GPKG)
    b = w.geometry.bounds
    gx = np.round(b.minx.values / 250.0).astype(np.int64)
    gy = np.round(b.miny.values / 250.0).astype(np.int64)
    lab = w["lcz_weak_label"].values
    dist = w["district"].fillna("unknown").values
    return {(int(a), int(c)): (str(d), (None if np.isnan(l) else int(l))) for a, c, d, l in zip(gx, gy, dist, lab)}


def init_ps_worker(pre: gpd.GeoDataFrame, lcz_band: int = 1):
    lower_priority()
    geoms = pre.geometry.values
    pxy, pseg = densify_lines(geoms, 2.0)
    order = np.argsort(pxy[:, 0], kind="stable")
    G_WORKER["px"] = pxy[order, 0]
    G_WORKER["py"] = pxy[order, 1]
    G_WORKER["pseg"] = pseg[order]
    G_WORKER["geoms"] = geoms
    G_WORKER["seg_id"] = pre["seg_id"].values.astype(np.int64)
    G_WORKER["seg_tile"] = pre["tile_id"].values
    G_WORKER["tr"] = Transformer.from_crs(5253, 4326, always_xy=True)
    with rasterio.open(LST_TIF) as s:
        a = s.read(1).astype(np.float64)
        a[(a == -9999.0) | ~np.isfinite(a)] = np.nan
        G_WORKER["lst"] = a
        G_WORKER["lst_tf"] = s.transform
    with rasterio.open(LCZ_TIF) as s:
        G_WORKER["lcz"] = s.read(lcz_band)
        G_WORKER["lcz_tf"] = s.transform
    G_WORKER["weak"] = load_weak_lookup()


def read_tile_layers(tid: str):
    arrs = {}
    for lyr in ("dem", "building_dsm", "landmask"):
        with rasterio.open(TILE_DIR / lyr / f"{tid}.tif") as s:
            arrs[lyr] = s.read(1).astype(np.float64)
    return arrs


def write_ps_raster(tid: str, ps: np.ndarray, x0: int, y0: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = dict(
        driver="GTiff",
        height=INNER_PX,
        width=INNER_PX,
        count=1,
        dtype="int32",
        crs="EPSG:5253",
        transform=from_origin(x0, y0 + TILE_M, RES, RES),
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(out_dir / f"{tid}.tif", "w", **prof) as dst:
        dst.write(ps.reshape(INNER_PX, INNER_PX), 1)


def build_ps_tile(args):
    """Build a ps_index raster and collect cell attributes for a tile at a specified buffer."""
    tid, buffer_m, out_ps_dir = args
    x0, y0 = tile_xy(tid)
    A = read_tile_layers(tid)
    d, b, lm = A["dem"], A["building_dsm"], A["landmask"]
    ok = (b != -9999.0) & (d != -9999.0)
    bh = np.where(ok, b - d, 0.0)
    bld = ok & (bh > 0.5)
    land = lm == 1
    I = (slice(BUF_PX, BUF_PX + INNER_PX), slice(BUF_PX, BUF_PX + INNER_PX))
    land_in, bld_in, ok_in = land[I], bld[I], ok[I]
    cand = land_in & ~bld_in & ok_in
    xs = x0 + RES / 2 + RES * np.arange(INNER_PX)
    ys = y0 + TILE_M - RES / 2 - RES * np.arange(INNER_PX)
    XX, YY = np.meshgrid(xs, ys)

    # Search window with margin
    margin = buffer_m + 5.0
    lo = np.searchsorted(G_WORKER["px"], x0 - margin, side="left")
    hi = np.searchsorted(G_WORKER["px"], x0 + TILE_M + margin, side="right")
    py = G_WORKER["py"][lo:hi]
    m = (py >= y0 - margin) & (py <= y0 + TILE_M + margin)
    sel_x, sel_y, sel_seg = G_WORKER["px"][lo:hi][m], py[m], G_WORKER["pseg"][lo:hi][m]

    ps = np.zeros(INNER_PX * INNER_PX, dtype=np.int32)
    rows = []
    if len(sel_x) == 0 or not cand.any():
        write_ps_raster(tid, ps, x0, y0, out_ps_dir)
        return tid, rows

    tree_kd = cKDTree(np.column_stack([sel_x, sel_y]))
    cflat = np.flatnonzero(cand.ravel())
    q = np.column_stack([XX.ravel()[cflat], YY.ravel()[cflat]])
    dist, ix = tree_kd.query(q, k=1, distance_upper_bound=buffer_m)
    within = np.isfinite(dist)
    cflat, seg_g = cflat[within], sel_seg[ix[within]]

    # Drop foreign tile cells (Rule A3)
    own = G_WORKER["seg_tile"][seg_g] == tid
    cflat, seg_g = cflat[own], seg_g[own]

    # Drop segments with < 25 cells
    u, cnt = np.unique(seg_g, return_counts=True)
    small = u[cnt < MIN_PS_CELLS]
    big_mask = ~np.isin(seg_g, small)
    cflat, seg_g = cflat[big_mask], seg_g[big_mask]
    ps[cflat] = G_WORKER["seg_id"][seg_g]
    write_ps_raster(tid, ps, x0, y0, out_ps_dir)

    if len(cflat) == 0:
        return tid, rows

    # Compute static attributes for the surviving segments over these PS cells
    rr, cc = np.divmod(cflat, INNER_PX)
    X = xs[cc]
    Y = ys[rr]

    # LST (30 m nearest)
    tf = G_WORKER["lst_tf"]
    lc = np.floor((X - tf.c) / tf.a).astype(int)
    lr = np.floor((tf.f - Y) / -tf.e).astype(int)
    lst = G_WORKER["lst"]
    okl = (lc >= 0) & (lc < lst.shape[1]) & (lr >= 0) & (lr < lst.shape[0])
    lstv = np.full(len(X), np.nan)
    lstv[okl] = lst[lr[okl], lc[okl]]

    # LCZ (100 m nearest)
    lon, lat = G_WORKER["tr"].transform(X, Y)
    tf2 = G_WORKER["lcz_tf"]
    zc = np.floor((lon - tf2.c) / tf2.a).astype(int)
    zr = np.floor((lat - tf2.f) / tf2.e).astype(int)
    lcz = G_WORKER["lcz"]
    okz = (zc >= 0) & (zc < lcz.shape[1]) & (zr >= 0) & (zr < lcz.shape[0])
    lczv = np.zeros(len(X), dtype=np.int64)
    lczv[okz] = lcz[zr[okz], zc[okz]]
    lczv[(lczv < 1) | (lczv > 17)] = 0

    # DEM / Slope
    gy_, gx_ = np.gradient(np.where(d == -9999.0, np.nan, d), RES)
    slope = np.degrees(np.arctan(np.hypot(gx_, gy_)))[I].ravel()[cflat]
    elev = d[I].ravel()[cflat]
    ids = {L: (np.floor(X / L).astype(np.int64) * 1_000_000 + np.floor(Y / L).astype(np.int64)) for L in (50, 100, 250, 500)}

    order = np.argsort(seg_g, kind="stable")
    sg = seg_g[order]
    bounds = np.flatnonzero(np.diff(sg)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(sg)]])
    for s0, s1 in zip(starts, ends):
        gi = int(sg[s0])
        sel = order[s0:s1]
        out_row = {"seg_idx": gi, "seg_id": int(G_WORKER["seg_id"][gi]), "tile_id": tid, "n_ps": int(len(sel))}
        v = lstv[sel]
        out_row["lst_med"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        zv = lczv[sel]
        zv = zv[zv > 0]
        out_row["lcz_primary"] = int(mode_int(zv)) if zv.size else 0
        out_row["elev_m"] = float(np.nanmedian(elev[sel]))
        sv = slope[sel]
        out_row["slope_deg"] = float(np.nanmedian(sv)) if np.isfinite(sv).any() else np.nan
        for L in (50, 100, 250, 500):
            k = int(mode_int(ids[L][sel]))
            kx, ky = divmod(k, 1_000_000)
            out_row[f"g{L:03d}_id"] = f"{kx}_{ky}"
            if L == 250:
                dd = G_WORKER["weak"].get((kx, ky))
                out_row["district"] = dd[0] if dd else "unknown"
                out_row["lcz_weak250"] = dd[1] if (dd and dd[1] is not None) else np.nan
        rows.append(out_row)

    return tid, rows


def extract_tile_day_sensitivity(tid: str, day: str, ps_dir: Path, is_p90: bool = False):
    """Extract segment metrics from a tile using specified ps_index and aggregation statistic."""
    from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction

    u10, wd, vrb15 = met_b15(day)
    with rasterio.open(ps_dir / f"{tid}.tif") as s:
        ps = s.read(1).ravel()
    m = ps > 0
    if not m.any():
        return tid, day, pd.DataFrame(columns=["seg_id"] + UTCI_COLS)
    seg = ps[m].astype(np.int64)

    with rasterio.open(UTCI_DIR / day / "utci_street" / f"{tid}.tif") as s:
        U = s.read().reshape(16, -1)[:, m].astype(np.float64)
    U[U == NODATA] = np.nan
    df = pd.DataFrame(U.T, columns=UTCI_COLS)
    df["seg_id"] = seg

    with rasterio.open(UTCI_DIR / day / "utci_uniform" / f"{tid}.tif") as s:
        v = s.read(10).ravel()[m].astype(np.float64)
    v[v == NODATA] = np.nan
    df["utci_unif_b15"] = v

    with rasterio.open(SOLWEIG_DIR / day / "tmrt" / f"{tid}.tif") as s:
        v = s.read(10).ravel()[m].astype(np.float64)
    v[(v == NODATA) | ~np.isfinite(v)] = np.nan
    df["tmrt_b15"] = v

    with rasterio.open(SOLWEIG_DIR / day / "svf" / f"{tid}.tif") as s:
        v = s.read(1).ravel()[m].astype(np.float64)
    v[(v == NODATA) | ~np.isfinite(v)] = np.nan
    df["svf_med"] = v

    with rasterio.open(SOLWEIG_DIR / day / "shadow" / f"{tid}.tif") as s:
        sh = s.read(indexes=list(range(10 - 5, 18 - 5 + 1))).reshape(9, -1)[:, m].astype(np.float64)
    sh[sh == 255.0] = np.nan
    df["shade_frac_10_18"] = 1.0 - np.nanmean(sh / 100.0, axis=0)

    with rasterio.open(RATIO16_DIR / f"{tid}.tif") as s:
        r16 = s.read().astype(np.float32)

    ratio_full = r16.mean(axis=0) if vrb15 else interpolate_ratio_at_direction(r16, wd)
    ratio = ratio_full.ravel()[m].astype(np.float64)
    df["ratio_med_b15"] = ratio
    df["v10eff_b15"] = np.clip(u10 * ratio, 0.5, 17.0)

    gb = df.groupby("seg_id")
    med = gb.median()
    out = pd.DataFrame(index=med.index)
    out["n_ps_valid"] = gb["utci_b15"].count().astype(np.int64)

    for c in UTCI_COLS + ["utci_unif_b15", "tmrt_b15", "v10eff_b15", "ratio_med_b15", "svf_med"]:
        out[c] = med[c]

    # Sensitivity: p90 instead of median for M1 (utci_b15) and M3 (tmrt_b15)
    if is_p90:
        out["utci_b15"] = gb["utci_b15"].quantile(0.9)
        out["tmrt_b15"] = gb["tmrt_b15"].quantile(0.9)

    out["tmrt_b15_p90"] = gb["tmrt_b15"].quantile(0.9)
    out["shade_frac_10_18"] = gb["shade_frac_10_18"].mean()
    h1018 = [f"utci_b{h:02d}" for h in range(10, 19)]
    ok = med[h1018].notna().all(axis=1)
    out["m2_strong"] = (med[h1018] >= 32.0).sum(axis=1).where(ok).astype("float64")
    out["m2_vstrong"] = (med[h1018] >= 38.0).sum(axis=1).where(ok).astype("float64")
    out = out.reset_index()
    return tid, day, out


def _safe_extract(job):
    tid, day, ps_dir, is_p90 = job
    try:
        return extract_tile_day_sensitivity(tid, day, ps_dir, is_p90)
    except Exception as e:
        print(f"ERROR extracting ({tid}, {day}): {e!r}", flush=True)
        return tid, day, None


def run_ps_buffer_variant(var_name: str, buffer_m: float, tiles: list[str], workers: int, pre_df: gpd.GeoDataFrame, primary_seg: gpd.GeoDataFrame):
    """Run PS buffer sensitivity (ps10 or ps20): rebuild ps_index, write segments_<var>.gpkg and seg_metrics_<DAY>.parquet."""
    t0 = time.time()
    print(f"\n=======================================================")
    print(f"Running PS Buffer Sensitivity: {var_name} ({buffer_m:.0f} m buffer) on {len(tiles)} tiles")
    print(f"=======================================================")

    out_var_dir = SENS_DATA_DIR / var_name
    out_var_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rebuild ps_index GeoTIFFs
    jobs = [(t, buffer_m, out_var_dir) for t in tiles]

    print(f"Building {var_name} ps_index rasters in {out_var_dir}...")
    tile_rows = []
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=init_ps_worker, initargs=(pre_df, 1)) as pool:
            for tid, rows in pool.imap_unordered(build_ps_tile, jobs, chunksize=2):
                tile_rows.extend(rows)
    else:
        init_ps_worker(pre_df, 1)
        for j in jobs:
            tid, rows = build_ps_tile(j)
            tile_rows.extend(rows)

    print(f"Built ps_index rasters for {len(tiles)} tiles ({time.time() - t0:.1f} s). Surviving segment instances: {len(tile_rows)}")

    # 2. Build segments_<var>.gpkg
    if tile_rows:
        df_new = pd.DataFrame(tile_rows)
        # Corridor and geometry attributes from primary segments (or pre_ps)
        base_cols = ["seg_id", "length_m", "azimuth_deg", "road_class", "dist_coast_m", "hw_ratio", "bh_mean30", "bcover30", "tcover15", "tht15", "hw_w_mean_m", "hw_valid_share", "geometry"]
        ref_lookup = primary_seg[base_cols].drop_duplicates("seg_id").set_index("seg_id")

        merged = df_new.join(ref_lookup, on="seg_id", how="inner")
        cols = [
            "seg_id", "tile_id", "district", "lcz_primary", "lcz_weak250", "length_m", "n_ps",
            "azimuth_deg", "hw_ratio", "bh_mean30", "bcover30", "tcover15", "tht15", "dist_coast_m",
            "elev_m", "slope_deg", "lst_med", "g050_id", "g100_id", "g250_id", "g500_id",
            "road_class", "hw_w_mean_m", "hw_valid_share",
        ]
        out_gpkg = gpd.GeoDataFrame(merged[cols].copy(), geometry=merged["geometry"].values, crs="EPSG:5253")
        out_gpkg["lcz_weak250"] = out_gpkg["lcz_weak250"].astype("Int64")
        out_gpkg["lcz_primary"] = out_gpkg["lcz_primary"].astype("int64")
        out_gpkg = out_gpkg.sort_values("seg_id").reset_index(drop=True)

        gpkg_path1 = out_var_dir / f"segments_{var_name}.gpkg"
        gpkg_path2 = SENS_DATA_DIR / f"segments_{var_name}.gpkg"
        for p in (gpkg_path1, gpkg_path2):
            if p.exists():
                p.unlink()
            out_gpkg.to_file(p, layer="segments", driver="GPKG")
        print(f"Wrote {gpkg_path1} and {gpkg_path2} ({len(out_gpkg)} segments).")

    # 3. Extract dynamic metrics (seg_metrics_<DAY>.parquet)
    print(f"Extracting segment metrics for {var_name}...")
    extract_jobs = [(t, d, out_var_dir, False) for t in tiles for d in DAYS]
    res_dict = {d: [] for d in DAYS}
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=lower_priority) as pool:
            for tid, day, df in pool.imap_unordered(_safe_extract, extract_jobs, chunksize=2):
                if df is not None:
                    res_dict[day].append(df)
    else:
        for j in extract_jobs:
            tid, day, df = _safe_extract(j)
            if df is not None:
                res_dict[day].append(df)

    n_tiles = len(tiles)
    for d in DAYS:
        if res_dict[d]:
            full_df = pd.concat(res_dict[d], ignore_index=True).sort_values("seg_id").reset_index(drop=True)
            full_df["partial_run"] = bool(n_tiles < N_TILES_TOTAL)
            full_df["n_tiles_at_extraction"] = np.int64(n_tiles)
            full_df["seg_id"] = full_df["seg_id"].astype(np.int64)
            full_df["n_ps_valid"] = full_df["n_ps_valid"].astype(np.int64)
            out_p = out_var_dir / f"seg_metrics_{d}.parquet"
            atomic_parquet(full_df[SCHEMA_COLS], out_p)
            print(f"Wrote {out_p} ({len(full_df)} rows, partial_run={bool(n_tiles < N_TILES_TOTAL)}, n_tiles={n_tiles}).")


def run_p90_variant(tiles: list[str], workers: int):
    """Run p90 metric sensitivity: p90 instead of median for M1/M3 on primary ps_index."""
    t0 = time.time()
    print(f"\n=======================================================")
    print(f"Running Metric Sensitivity: p90 (M1/M3 90th percentile) on {len(tiles)} tiles")
    print(f"=======================================================")

    out_var_dir = SENS_DATA_DIR / "p90"
    out_var_dir.mkdir(parents=True, exist_ok=True)

    extract_jobs = [(t, d, PS_DIR, True) for t in tiles for d in DAYS]
    res_dict = {d: [] for d in DAYS}
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=lower_priority) as pool:
            for tid, day, df in pool.imap_unordered(_safe_extract, extract_jobs, chunksize=2):
                if df is not None:
                    res_dict[day].append(df)
    else:
        for j in extract_jobs:
            tid, day, df = _safe_extract(j)
            if df is not None:
                res_dict[day].append(df)

    n_tiles = len(tiles)
    for d in DAYS:
        if res_dict[d]:
            full_df = pd.concat(res_dict[d], ignore_index=True).sort_values("seg_id").reset_index(drop=True)
            full_df["partial_run"] = bool(n_tiles < N_TILES_TOTAL)
            full_df["n_tiles_at_extraction"] = np.int64(n_tiles)
            full_df["seg_id"] = full_df["seg_id"].astype(np.int64)
            full_df["n_ps_valid"] = full_df["n_ps_valid"].astype(np.int64)
            out_p = out_var_dir / f"seg_metrics_{d}.parquet"
            atomic_parquet(full_df[SCHEMA_COLS], out_p)
            print(f"Wrote {out_p} ({len(full_df)} rows, partial_run={bool(n_tiles < N_TILES_TOTAL)}, n_tiles={n_tiles}, elapsed {time.time()-t0:.1f} s).")


def run_lcz_filter_variant(tiles: list[str], primary_seg: gpd.GeoDataFrame):
    """Run LCZ sensitivity: lcz_primary from LCZ band 2 (LCZ_Filter, amendment A2)."""
    t0 = time.time()
    print(f"\n=======================================================")
    print(f"Running LCZ Sensitivity: lcz_filter (LCZ band 2) across tiles")
    print(f"=======================================================")

    out_var_dir = SENS_DATA_DIR / "lcz_filter"
    out_var_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample LCZ band 2 over PS cells
    with rasterio.open(LCZ_TIF) as s:
        lcz2 = s.read(2)
        tf2 = s.transform
    tr = Transformer.from_crs(5253, 4326, always_xy=True)

    seg_modes = {}
    for tid in tiles:
        ps_p = PS_DIR / f"{tid}.tif"
        if not ps_p.exists():
            continue
        x0, y0 = tile_xy(tid)
        with rasterio.open(ps_p) as s:
            ps = s.read(1)
        m = ps > 0
        if not m.any():
            continue
        r, c = np.where(m)
        seg_ids = ps[m]
        xs = x0 + RES / 2 + RES * c
        ys = y0 + TILE_M - RES / 2 - RES * r
        lons, lats = tr.transform(xs, ys)
        zc = np.floor((lons - tf2.c) / tf2.a).astype(int)
        zr = np.floor((lats - tf2.f) / tf2.e).astype(int)
        okz = (zc >= 0) & (zc < lcz2.shape[1]) & (zr >= 0) & (zr < lcz2.shape[0])
        val = np.zeros(len(xs), dtype=int)
        val[okz] = lcz2[zr[okz], zc[okz]]
        val[(val < 1) | (val > 17)] = 0
        df = pd.DataFrame({"seg_id": seg_ids, "lcz": val})
        df = df[df["lcz"] > 0]
        if len(df):
            for sid, grp in df.groupby("seg_id"):
                seg_modes[sid] = int(grp["lcz"].value_counts().index[0])

    print(f"Extracted LCZ band 2 for {len(seg_modes)} segments across {len(tiles)} tiles ({time.time()-t0:.1f} s).")

    # 2. Update segments table with lcz_primary from band 2
    lcz_seg = primary_seg.copy()
    new_lcz = [seg_modes.get(sid, lcz_seg["lcz_primary"].iloc[i]) for i, sid in enumerate(lcz_seg["seg_id"])]
    lcz_seg["lcz_primary"] = np.int64(new_lcz)

    gpkg_path1 = out_var_dir / "segments_lcz_filter.gpkg"
    gpkg_path2 = SENS_DATA_DIR / "segments_lcz_filter.gpkg"
    for p in (gpkg_path1, gpkg_path2):
        if p.exists():
            p.unlink()
        lcz_seg.to_file(p, layer="segments", driver="GPKG")
    print(f"Wrote {gpkg_path1} and {gpkg_path2} ({len(lcz_seg)} segments).")

    # 3. Dynamic metrics: identical to primary seg_metrics because PS cells and climate rasters are identical
    for d in DAYS:
        prim_parquet = ANALYSIS_DIR / f"seg_metrics_{d}.parquet"
        if prim_parquet.exists():
            dst = out_var_dir / f"seg_metrics_{d}.parquet"
            shutil.copy2(prim_parquet, dst)
            print(f"Copied {prim_parquet.name} -> {dst}")


def run_nomotorway_variant(primary_seg: gpd.GeoDataFrame, out_root: Path):
    """Motorway sensitivity (lock section 1, A8 item 3): drop segments with road_class == 'OTOYOL' before the statistics.

    PS cells, rasters and per-segment metrics are unchanged, so seg_metrics_<DAY>.parquet is the primary table
    restricted to the kept seg_ids (partial_run / n_tiles_at_extraction flags are inherited)."""
    t0 = time.time()
    print("\n=======================================================")
    print(f"Running Motorway Sensitivity: nomotorway (drop road_class == '{MOTORWAY_CLASS}')")
    print("=======================================================")
    out_var_dir = out_root / "nomotorway"
    out_var_dir.mkdir(parents=True, exist_ok=True)

    if "road_class" not in primary_seg.columns:
        raise RuntimeError("segments table has no road_class column; cannot run nomotorway")
    classes = primary_seg["road_class"].astype(str).value_counts().to_dict()
    if MOTORWAY_CLASS not in classes:
        raise RuntimeError(f"road_class value '{MOTORWAY_CLASS}' not found; values present: {classes}")
    is_mw = primary_seg["road_class"].astype(str) == MOTORWAY_CLASS
    kept = primary_seg.loc[~is_mw].copy().reset_index(drop=True)
    print(f"road_class values: {classes}")
    print(f"Dropped {int(is_mw.sum())} of {len(primary_seg)} segments ({MOTORWAY_CLASS}); kept {len(kept)}.")

    gpkg_path1 = out_var_dir / "segments_nomotorway.gpkg"
    gpkg_path2 = out_root / "segments_nomotorway.gpkg"
    for p in (gpkg_path1, gpkg_path2):
        if p.exists():
            p.unlink()
        kept.to_file(p, layer="segments", driver="GPKG")
    print(f"Wrote {gpkg_path1} and {gpkg_path2}.")

    drop_ids = set(primary_seg.loc[is_mw, "seg_id"].astype(np.int64).tolist())
    for d in DAYS:
        prim = ANALYSIS_DIR / f"seg_metrics_{d}.parquet"
        if not prim.exists():
            print(f"(skip {d}: {prim} not found)")
            continue
        m = pd.read_parquet(prim)
        n0 = len(m)
        m = m[~m["seg_id"].astype(np.int64).isin(drop_ids)].reset_index(drop=True)
        out_p = out_var_dir / f"seg_metrics_{d}.parquet"
        atomic_parquet(m, out_p)
        print(f"Wrote {out_p} ({len(m)} of {n0} rows kept; {n0 - len(m)} motorway rows removed; "
              f"partial_run={bool(m['partial_run'].any()) if 'partial_run' in m else None}).")
    print(f"nomotorway done in {time.time() - t0:.1f} s")


def main():
    global SENS_DATA_DIR
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    mp.freeze_support()

    ap = argparse.ArgumentParser(description="WP3.6 Extraction Sensitivity Machinery")
    ap.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all", help="Sensitivity variant to run")
    ap.add_argument("--tiles", nargs="*", default=None, help="Specific tile IDs to process")
    ap.add_argument("--all-tiles", action="store_true", help="Process all 427 core tiles (final run command)")
    ap.add_argument("--workers", type=int, default=2, help="Number of CPU worker processes (max 4 per rules)")
    ap.add_argument("--out-root", default=None,
                    help="override the output root (default data/street/analysis/sens); use e.g. outputs/street/analysis/... for tests")
    args = ap.parse_args()

    t_start = time.time()
    lower_priority()
    if args.out_root:
        SENS_DATA_DIR = Path(args.out_root)
        if not SENS_DATA_DIR.is_absolute():
            SENS_DATA_DIR = ROOT / SENS_DATA_DIR
    SENS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine tiles to process
    if args.tiles:
        tiles = sorted(args.tiles)
    elif args.all_tiles:
        tiles = sorted(gpd.read_file(TILES_CORE)["tile_id"].tolist())
    else:
        # Default: tiles available now in WP3.0_tiles_done.csv
        if TILES_DONE_CSV.exists():
            done = pd.read_csv(TILES_DONE_CSV)
            done_both = done.groupby("tile_id")["day"].nunique().loc[lambda s: s >= 2].index.tolist()
            tiles = sorted(done_both)
        else:
            tiles = ["T_0484_4242", "T_0484_4243", "T_0487_4241", "T_0487_4242"]

    print(f"CLIMORFA WP3.6 Extraction Sensitivity: processing {len(tiles)} tiles (all_tiles={args.all_tiles}, workers={args.workers})")
    print(f"Selected variants: {args.variant}")

    # Load base inputs
    variants_to_run = list(VARIANTS) if args.variant == "all" else [args.variant]
    needs_pre = any(v in variants_to_run for v in ("ps10", "ps20"))
    print("Loading pre_ps and primary segments...")
    pre_df = gpd.read_file(PRE_PS_GPKG) if needs_pre else None
    primary_seg = gpd.read_file(SEGMENTS_GPKG)

    if "ps10" in variants_to_run:
        run_ps_buffer_variant("ps10", 10.0, tiles, args.workers, pre_df, primary_seg)

    if "ps20" in variants_to_run:
        run_ps_buffer_variant("ps20", 20.0, tiles, args.workers, pre_df, primary_seg)

    if "p90" in variants_to_run:
        run_p90_variant(tiles, args.workers)

    if "lcz_filter" in variants_to_run:
        run_lcz_filter_variant(tiles, primary_seg)

    if "nomotorway" in variants_to_run:
        run_nomotorway_variant(primary_seg, SENS_DATA_DIR)

    print(f"\nAll sensitivity extraction tasks completed successfully in {time.time() - t_start:.1f} s.")


if __name__ == "__main__":
    main()
