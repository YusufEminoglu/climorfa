"""WP3.0 steps 2-3: PS mask + nearest-segment index (ps_index rasters for all tiles) and static segment attributes.

Run after build_network.py:   python build_street_units.py [--workers 4] [--tiles T_.. T_..] [--limit N]

Per tile (inner 500 x 500 px of the 2 m grid, same transform as the SOLWEIG outputs):
  PS candidate = landmask == 1  AND  building_dsm - dem <= 0.5 m  AND  within 15 m of a road centreline
  nearest centreline = nearest densified (2 m) vertex of ANY pre-PS segment (including segments in the buffer zone or in
  neighbouring tiles). A cell whose nearest segment belongs to a different tile is set to 0 (that road is owned by the other
  tile); segments with < 25 PS cells in their own tile are dropped (cells -> 0).
Static attributes: see docstrings of seg_attrs().
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

from common import *  # noqa

import geopandas as gpd
import rasterio
import shapely
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- globals filled by init_worker
G: dict = {}


def densify(lines: np.ndarray, step: float):
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
    assert np.allclose(b.minx.values, gx * 250.0) and np.allclose(b.miny.values, gy * 250.0)
    lab = w["lcz_weak_label"].values
    dist = w["district"].fillna("unknown").values
    return {(int(a), int(c)): (str(d), (None if np.isnan(l) else int(l))) for a, c, d, l in zip(gx, gy, dist, lab)}


def init_worker(pre: gpd.GeoDataFrame):
    lower_priority()
    from pyproj import Transformer
    geoms = pre.geometry.values
    pxy, pseg = densify(geoms, 2.0)
    order = np.argsort(pxy[:, 0], kind="stable")
    G["px"] = pxy[order, 0]
    G["py"] = pxy[order, 1]
    G["pseg"] = pseg[order]
    G["geoms"] = geoms
    G["seg_id"] = pre["seg_id"].values.astype(np.int64)
    tile_ids = pre["tile_id"].values
    G["seg_tile"] = tile_ids
    G["tr"] = Transformer.from_crs(5253, 4326, always_xy=True)
    with rasterio.open(LST_TIF) as s:
        a = s.read(1).astype(np.float64)
        a[(a == -9999.0) | ~np.isfinite(a)] = np.nan
        G["lst"] = a
        G["lst_tf"] = s.transform
    with rasterio.open(LCZ_TIF) as s:
        G["lcz"] = s.read(1)  # band 1 = "LCZ" (bands: LCZ, LCZ_Filter, LCZ_Probability per gee/scripts/download_lcz_weak_labels.py)
        G["lcz_tf"] = s.transform
    G["weak"] = load_weak_lookup()
    sh = shapely.to_wkb  # noqa
    # shoreline is handled in the parent


def read_tile_layers(tid: str):
    arrs = {}
    for lyr in ("dem", "building_dsm", "tree_dsm", "landmask"):
        with rasterio.open(TILE_DIR / lyr / f"{tid}.tif") as s:
            arrs[lyr] = s.read(1).astype(np.float64)
    return arrs


def mode_int(v: np.ndarray):
    if v.size == 0:
        return None
    u, c = np.unique(v, return_counts=True)
    return u[np.argmax(c)]


def hw_stations(line, bld_full, bx0, by1, step_station=5.0, max_r=50.0):
    """Front-to-front street width along the segment normal.

    Stations every ~5 m (centred); at each station a ray is marched along +/- normal in 1 m steps up to 50 m; the distance to
    the first building cell (building_dsm - dem > 0.5 m, cell nearest to the sample point) is the front distance on that side.
    A station is valid if both sides hit a building and the station itself is not on a building cell.
    Returns (W_mean_m or nan, valid_share, n_valid)."""
    L = line.length
    ns = max(1, int(L // step_station))
    sp = (np.arange(ns) + 0.5) * L / ns
    p = shapely.get_coordinates(shapely.line_interpolate_point(line, sp))
    pm = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(sp - 2.0, 0, L)))
    pp = shapely.get_coordinates(shapely.line_interpolate_point(line, np.clip(sp + 2.0, 0, L)))
    t = pp - pm
    tn = np.hypot(t[:, 0], t[:, 1])
    good = tn > 0
    t = t / np.where(tn > 0, tn, 1.0)[:, None]
    nx, ny = -t[:, 1], t[:, 0]
    steps = np.arange(1.0, max_r + 1.0)
    H, W = bld_full.shape
    res = []
    for side in (1.0, -1.0):
        X = p[:, 0][:, None] + side * steps[None, :] * nx[:, None]
        Y = p[:, 1][:, None] + side * steps[None, :] * ny[:, None]
        col = np.floor((X - bx0) / RES).astype(int)
        row = np.floor((by1 - Y) / RES).astype(int)
        inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        hit = np.zeros(X.shape, dtype=bool)
        hit[inb] = bld_full[row[inb], col[inb]]
        has = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        res.append((has, steps[first]))
    c0 = np.floor((p[:, 0] - bx0) / RES).astype(int)
    r0 = np.floor((by1 - p[:, 1]) / RES).astype(int)
    inb0 = (c0 >= 0) & (c0 < W) & (r0 >= 0) & (r0 < H)
    on_b = np.zeros(len(p), dtype=bool)
    on_b[inb0] = bld_full[r0[inb0], c0[inb0]]
    valid = res[0][0] & res[1][0] & ~on_b & good
    if valid.sum() == 0:
        return np.nan, 0.0, 0
    Wd = res[0][1][valid] + res[1][1][valid]
    return float(Wd.mean()), float(valid.mean()), int(valid.sum())


def process_tile(tid: str):
    t0 = time.time()
    x0, y0 = tile_xy(tid)
    bx0, by1 = x0 - 100.0, y0 + 1100.0
    A = read_tile_layers(tid)
    d, b, tr, lm = A["dem"], A["building_dsm"], A["tree_dsm"], A["landmask"]
    ok = (b != -9999.0) & (d != -9999.0)
    bh = np.where(ok, b - d, 0.0)
    bld = ok & (bh > 0.5)
    land = lm == 1
    tree = (tr >= 2.0) & ~bld
    I = (slice(BUF_PX, BUF_PX + INNER_PX), slice(BUF_PX, BUF_PX + INNER_PX))
    land_in, bld_in, ok_in = land[I], bld[I], ok[I]
    cand = land_in & ~bld_in & ok_in
    xs = x0 + RES / 2 + RES * np.arange(INNER_PX)
    ys = y0 + TILE_M - RES / 2 - RES * np.arange(INNER_PX)
    XX, YY = np.meshgrid(xs, ys)

    # candidate road vertices (segments from any tile, incl. buffer zone)
    lo = np.searchsorted(G["px"], x0 - 20.0, side="left")
    hi = np.searchsorted(G["px"], x0 + TILE_M + 20.0, side="right")
    py = G["py"][lo:hi]
    m = (py >= y0 - 20.0) & (py <= y0 + TILE_M + 20.0)
    sel_x, sel_y, sel_seg = G["px"][lo:hi][m], py[m], G["pseg"][lo:hi][m]

    stats = {"tile_id": tid, "land_cells": int(land_in.sum()), "cand_cells": int(cand.sum())}
    ps = np.zeros(INNER_PX * INNER_PX, dtype=np.int32)
    rows = []
    if len(sel_x) == 0 or not cand.any():
        stats.update(ps_cells_15m=0, ps_cells=0, foreign_dropped=0, small_seg_dropped_cells=0, n_segments=0, n_small_segments=0)
        write_ps(tid, ps, x0, y0)
        stats["seconds"] = round(time.time() - t0, 2)
        return stats, rows
    tree_kd = cKDTree(np.column_stack([sel_x, sel_y]))
    cflat = np.flatnonzero(cand.ravel())
    q = np.column_stack([XX.ravel()[cflat], YY.ravel()[cflat]])
    dist, ix = tree_kd.query(q, k=1, distance_upper_bound=PS_BUFFER_M)
    within = np.isfinite(dist)
    cflat, seg_g = cflat[within], sel_seg[ix[within]]
    stats["ps_cells_15m"] = int(len(cflat))
    own = G["seg_tile"][seg_g] == tid
    stats["foreign_dropped"] = int((~own).sum())
    cflat, seg_g = cflat[own], seg_g[own]
    u, cnt = np.unique(seg_g, return_counts=True)
    small = u[cnt < MIN_PS_CELLS]
    big_mask = ~np.isin(seg_g, small)
    stats["n_small_segments"] = int(len(small))
    stats["small_seg_dropped_cells"] = int((~big_mask).sum())
    cflat, seg_g = cflat[big_mask], seg_g[big_mask]
    stats["ps_cells"] = int(len(cflat))
    ps[cflat] = G["seg_id"][seg_g]
    write_ps(tid, ps, x0, y0)

    # ---------- static attributes
    if len(cflat):
        rr, cc = np.divmod(cflat, INNER_PX)
        X = xs[cc]
        Y = ys[rr]
        # LST (nearest, 30 m)
        tf = G["lst_tf"]
        lc = np.floor((X - tf.c) / tf.a).astype(int)
        lr = np.floor((tf.f - Y) / -tf.e).astype(int)
        lst = G["lst"]
        okl = (lc >= 0) & (lc < lst.shape[1]) & (lr >= 0) & (lr < lst.shape[0])
        lstv = np.full(len(X), np.nan)
        lstv[okl] = lst[lr[okl], lc[okl]]
        # LCZ (nearest, 100 m, EPSG:4326 raster)
        lon, lat = G["tr"].transform(X, Y)
        tf2 = G["lcz_tf"]
        zc = np.floor((lon - tf2.c) / tf2.a).astype(int)
        zr = np.floor((lat - tf2.f) / tf2.e).astype(int)
        lcz = G["lcz"]
        okz = (zc >= 0) & (zc < lcz.shape[1]) & (zr >= 0) & (zr < lcz.shape[0])
        lczv = np.zeros(len(X), dtype=np.int64)
        lczv[okz] = lcz[zr[okz], zc[okz]]
        lczv[(lczv < 1) | (lczv > 17)] = 0
        # DEM / slope
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
            rows.append(seg_attrs(gi, sel, lstv, lczv, slope, elev, ids, tid, bld, bh, land, tree, tr, bx0, by1))
    stats["n_segments"] = len(rows)
    stats["seconds"] = round(time.time() - t0, 2)
    return stats, rows


def seg_attrs(gi, sel, lstv, lczv, slope, elev, ids, tid, bld, bh, land, tree, tr, bx0, by1):
    """Static attributes of one segment.

    n_ps; lst_med = median of 30 m Landsat LST (nearest) over PS cells; lcz_primary = majority of the global LCZ band 'LCZ'
    over PS cells (classes 1-17; 0/nodata ignored; 0 returned if no valid class); g###_id / lcz_weak250 / district = the
    grid cell holding the majority of the segment's PS cells; elev_m / slope_deg = median over PS cells (slope from
    np.gradient of the 2 m DEM); corridor stats below use ALL land cells in the geometric corridor of the segment line."""
    line = G["geoms"][gi]
    out = {"seg_idx": gi, "seg_id": int(G["seg_id"][gi]), "tile_id": tid, "n_ps": int(len(sel))}
    v = lstv[sel]
    out["lst_med"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    zv = lczv[sel]
    zv = zv[zv > 0]
    out["lcz_primary"] = int(mode_int(zv)) if zv.size else 0
    out["elev_m"] = float(np.nanmedian(elev[sel]))
    sv = slope[sel]
    out["slope_deg"] = float(np.nanmedian(sv)) if np.isfinite(sv).any() else np.nan
    for L in (50, 100, 250, 500):
        k = int(mode_int(ids[L][sel]))
        kx, ky = divmod(k, 1_000_000)
        out[f"g{L:03d}_id"] = f"{kx}_{ky}"
        if L == 250:
            dd = G["weak"].get((kx, ky))
            out["district"] = dd[0] if dd else "unknown"
            out["lcz_weak250"] = dd[1] if (dd and dd[1] is not None) else np.nan
    # ---- corridor stats
    xmin, ymin, xmax, ymax = line.bounds
    H, W = bld.shape
    c0 = max(0, int(np.floor((xmin - 31.0 - bx0) / RES)))
    c1 = min(W, int(np.ceil((xmax + 31.0 - bx0) / RES)) + 1)
    r0 = max(0, int(np.floor((by1 - ymax - 31.0) / RES)))
    r1 = min(H, int(np.ceil((by1 - ymin + 31.0) / RES)) + 1)
    cx = bx0 + RES / 2 + RES * np.arange(c0, c1)
    cy = by1 - RES / 2 - RES * np.arange(r0, r1)
    CX, CY = np.meshgrid(cx, cy)
    _la = np.empty(1, dtype=object); _la[0] = line
    p1, _ = densify(_la, 1.0)
    kd = cKDTree(p1)
    dd, _ = kd.query(np.column_stack([CX.ravel(), CY.ravel()]), k=1, distance_upper_bound=30.0)
    dd = dd.reshape(CX.shape)
    w30 = np.isfinite(dd)
    w15 = np.isfinite(dd) & (dd <= 15.0)
    sl = (slice(r0, r1), slice(c0, c1))
    land_w, bld_w, tree_w, bh_w, tr_w = land[sl], bld[sl], tree[sl], bh[sl], tr[sl]
    den30 = (land_w & w30).sum()
    nb = (bld_w & land_w & w30)
    out["bcover30"] = float(nb.sum() / den30) if den30 else np.nan
    out["bh_mean30"] = float(bh_w[nb].mean()) if nb.any() else np.nan
    den15 = (land_w & w15).sum()
    nt = tree_w & land_w & w15
    out["tcover15"] = float(nt.sum() / den15) if den15 else np.nan
    out["tht15"] = float(tr_w[nt].mean()) if nt.any() else np.nan
    Wm, vs, nv = hw_stations(line, bld, bx0, by1)
    out["hw_w_mean_m"] = Wm
    out["hw_valid_share"] = vs
    ns = max(1, int(line.length // 5.0))
    if nv >= 2 and vs >= 0.2 and np.isfinite(out["bh_mean30"]):
        out["hw_ratio"] = out["bh_mean30"] / Wm
    else:
        out["hw_ratio"] = np.nan
    return out


def write_ps(tid, ps, x0, y0):
    PS_DIR.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=INNER_PX, width=INNER_PX, count=1, dtype="int32", crs="EPSG:5253",
                transform=from_origin(x0, y0 + TILE_M, RES, RES), nodata=0, compress="deflate", predictor=2,
                tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(PS_DIR / f"{tid}.tif", "w", **prof) as dst:
        dst.write(ps.reshape(INNER_PX, INNER_PX), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-assemble", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    lower_priority()
    pre = gpd.read_file(PRE_PS_GPKG, layer="segments_pre_ps")
    tiles = sorted(gpd.read_file(TILES_CORE)["tile_id"].tolist())
    if a.tiles:
        tiles = [t for t in tiles if t in set(a.tiles)]
    if a.limit:
        tiles = tiles[:a.limit]
    print(f"{len(pre)} pre-PS segments; {len(tiles)} tiles; workers={a.workers}", flush=True)
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=init_worker, initargs=(pre,)) as pool:
        for i, (st, rows) in enumerate(pool.imap_unordered(process_tile, tiles, chunksize=1), 1):
            results.append((st, rows))
            if i % 20 == 0 or i == len(tiles):
                print(f"[{time.time()-t0:6.1f}s] {i}/{len(tiles)} tiles", flush=True)
    stats = pd.DataFrame([r[0] for r in results]).sort_values("tile_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats["ps_share_of_land"] = stats["ps_cells"] / stats["land_cells"].where(stats["land_cells"] > 0)
    stats.to_csv(OUT_DIR / ("tile_ps_stats.csv" if not (a.tiles or a.limit) else "tile_ps_stats_subset.csv"), index=False)
    if a.no_assemble:
        return
    allrows = [r for _, rows in results for r in rows]
    df = pd.DataFrame(allrows)
    base = pre.set_index(pre.index)
    df["length_m"] = base.loc[df["seg_idx"], "length_m"].values
    df["azimuth_deg"] = base.loc[df["seg_idx"], "azimuth_deg"].values
    df["road_class"] = base.loc[df["seg_idx"], "road_class"].values
    geom = base.geometry.values[df["seg_idx"].values]
    # distance to coast from the segment midpoint
    shore = gpd.read_file(SHORELINE_GPKG)
    st_tree = shapely.STRtree(shore.geometry.values)
    mid = shapely.line_interpolate_point(geom, 0.5, normalized=True)
    qi, ti = st_tree.query_nearest(mid, all_matches=False)
    dc = np.full(len(df), np.nan)
    dc[qi] = shapely.distance(mid[qi], shore.geometry.values[ti])
    df["dist_coast_m"] = dc
    cols = ["seg_id", "tile_id", "district", "lcz_primary", "lcz_weak250", "length_m", "n_ps", "azimuth_deg", "hw_ratio",
            "bh_mean30", "bcover30", "tcover15", "tht15", "dist_coast_m", "elev_m", "slope_deg", "lst_med",
            "g050_id", "g100_id", "g250_id", "g500_id", "road_class", "hw_w_mean_m", "hw_valid_share"]
    out = gpd.GeoDataFrame(df[cols].copy(), geometry=geom, crs="EPSG:5253")
    out["lcz_weak250"] = out["lcz_weak250"].astype("Int64")
    out["lcz_primary"] = out["lcz_primary"].astype("int64")
    out = out.sort_values("seg_id").reset_index(drop=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    fn = SEGMENTS_GPKG if not (a.tiles or a.limit) else OUT_DIR / "segments_subset_test.gpkg"
    if fn.exists():
        fn.unlink()
    out.to_file(fn, layer="segments", driver="GPKG")
    print(f"wrote {fn} rows={len(out)} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mp.freeze_support()
    main()
