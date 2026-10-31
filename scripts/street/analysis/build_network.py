"""WP3.0 step 1: street segments from roads_fua (before the PS filter).

Procedure (all coordinates snapped to 1 cm integers so that vertex matching is exact):
 1. read roads_fua, drop null geometries, explode multi-lines, drop exact duplicates; keep lines that intersect the union of
    the 427 tile extents + 200 m margin.
 2. vertex degree = (#line endpoints at the vertex) + 2 x (#lines having it as an interior vertex). Nodes = degree >= 3.
    Every line is split at nodes. Pieces are then chained through degree-2 vertices (GEOS line_merge), so a segment is the
    maximal run of road between two nodes / dead ends.
 3. split at the 1000 m tile borders (global grid, which contains the inner squares of tiles_core).
 4. merge segments shorter than 20 m into the longest same-tile neighbour that shares an end vertex
    (merging across a tile border is never done, so a segment always lies in exactly one tile).
 5. seg_id = 1..N (sorted by tile id, then midpoint x, y), tile_id from the midpoint, azimuth (length-weighted circular
    mean of the doubled bearing, 0-180), road_class (YOLTIP of the nearest original line, majority over sample points).
Output: outputs/street/analysis/wp30/segments_pre_ps.gpkg (layer segments_pre_ps) + network_stats.json
"""
from __future__ import annotations

import heapq
import json
import sys
import time

import numpy as np
import pandas as pd

from common import *  # noqa  (also fixes sys.path / env)

import geopandas as gpd
import shapely
from shapely import STRtree


def snap_lines(parts: np.ndarray):
    """Round to 1 cm, drop repeated vertices. Returns list of (n,2) int64 coordinate arrays (cm) and kept original idx."""
    coords = shapely.get_coordinates(parts)
    n = shapely.get_num_coordinates(parts)
    li = np.repeat(np.arange(len(parts)), n)
    ci = np.round(coords * 100.0).astype(np.int64)
    same = np.zeros(len(ci), dtype=bool)
    same[1:] = (ci[1:] == ci[:-1]).all(axis=1) & (li[1:] == li[:-1])
    keep = ~same
    ci, li = ci[keep], li[keep]
    counts = np.bincount(li, minlength=len(parts))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out, idx = [], []
    for i in range(len(parts)):
        if counts[i] >= 2:
            out.append(ci[starts[i]:starts[i] + counts[i]])
            idx.append(i)
    return out, np.array(idx)


def key_of(ci: np.ndarray) -> np.ndarray:
    return ci[:, 0] * 1_000_000_000 + ci[:, 1]


def split_at_nodes(lines: list[np.ndarray]):
    allc = np.concatenate(lines)
    allk = key_of(allc)
    n = np.array([len(l) for l in lines])
    starts = np.cumsum(n) - n
    is_end = np.zeros(len(allk), dtype=bool)
    is_end[starts] = True
    is_end[starts + n - 1] = True
    deg = pd.Series(np.where(is_end, 1, 2)).groupby(allk).sum()
    node_keys = deg.index[deg.values >= 3].values
    is_node = np.isin(allk, node_keys)
    pieces = []
    n_split = 0
    for i, l in enumerate(lines):
        s = starts[i]
        flags = is_node[s + 1:s + len(l) - 1]
        if flags.any():
            cuts = np.flatnonzero(flags) + 1
            b = [0] + cuts.tolist() + [len(l) - 1]
            for a, c in zip(b[:-1], b[1:]):
                pieces.append(l[a:c + 1])
            n_split += 1
        else:
            pieces.append(l)
    return pieces, int(len(node_keys)), n_split


def chain_merge(pieces: list[np.ndarray]) -> list[np.ndarray]:
    arr = np.empty(len(pieces), dtype=object)
    for i, p in enumerate(pieces):
        arr[i] = shapely.linestrings(p / 100.0)
    mls = shapely.multilinestrings(arr)
    merged = shapely.line_merge(mls)
    parts = shapely.get_parts(merged)
    out = []
    for g in parts:
        out.append(np.round(shapely.get_coordinates(g) * 100.0).astype(np.int64))
    return out


def split_at_tile_borders(lines_cm: list[np.ndarray]):
    out = []
    n_cross = 0
    for l in lines_cm:
        xy = l / 100.0
        gx = np.floor(xy[:, 0] / TILE_M)
        gy = np.floor(xy[:, 1] / TILE_M)
        if gx.min() == gx.max() and gy.min() == gy.max():
            out.append(l)
            continue
        n_cross += 1
        pts = [xy[0]]
        cut = [False]
        for i in range(len(xy) - 1):
            p, q = xy[i], xy[i + 1]
            ts = []
            for ax, g0, g1 in ((0, gx[i], gx[i + 1]), (1, gy[i], gy[i + 1])):
                if g0 != g1:
                    lo, hi = (g0, g1) if g1 > g0 else (g1, g0)
                    d = q[ax] - p[ax]
                    for k in range(int(lo) + 1, int(hi) + 1):
                        t = (k * TILE_M - p[ax]) / d
                        ts.append(min(max(t, 0.0), 1.0))
            ts = sorted(ts)
            pending = False
            last = -1.0
            for t in ts:
                if t - last <= 1e-9:
                    continue
                last = t
                if t <= 1e-12:
                    cut[-1] = True
                elif t >= 1 - 1e-12:
                    pending = True
                else:
                    pts.append(p + t * (q - p))
                    cut.append(True)
            pts.append(q.copy())
            cut.append(pending)
        pts = np.round(np.array(pts) * 100.0).astype(np.int64)
        cut_arr = np.array(cut)
        keep = np.ones(len(pts), dtype=bool)
        for j in range(1, len(pts)):
            if (pts[j] == pts[j - 1]).all():
                keep[j] = False
                cut_arr[j - 1] = cut_arr[j - 1] or cut_arr[j]
        pts = pts[keep]
        cut_arr = cut_arr[keep]
        cut_arr[0] = False
        cut_arr[-1] = False
        idx = [0] + np.flatnonzero(cut_arr).tolist() + [len(pts) - 1]
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if b_ > a_:
                out.append(pts[a_:b_ + 1])
    return out, n_cross


def line_len(c: np.ndarray) -> float:
    d = np.diff(c / 100.0, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def mid_tile(c: np.ndarray) -> tuple[int, int]:
    g = shapely.linestrings(c / 100.0)
    m = shapely.line_interpolate_point(g, 0.5, normalized=True)
    x, y = shapely.get_coordinates(m)[0]
    return int(np.floor(x / TILE_M)), int(np.floor(y / TILE_M))


def merge_short(lines: list[np.ndarray]):
    """Merge segments < MIN_SEG_LEN_M into the longest same-tile neighbour sharing an end vertex."""
    segs = {i: l for i, l in enumerate(lines)}
    tile = {i: mid_tile(l) for i, l in segs.items()}
    length = {i: line_len(l) for i, l in segs.items()}
    ends: dict[int, set] = {}

    def add_ends(i):
        l = segs[i]
        for k in {int(key_of(l[:1])[0]), int(key_of(l[-1:])[0])}:
            ends.setdefault(k, set()).add(i)

    def del_ends(i):
        l = segs[i]
        for k in {int(key_of(l[:1])[0]), int(key_of(l[-1:])[0])}:
            ends[k].discard(i)

    for i in segs:
        add_ends(i)
    heap = [(length[i], i) for i in segs if length[i] < MIN_SEG_LEN_M]
    heapq.heapify(heap)
    next_id = len(lines)
    n_merged = 0
    n_unmergeable = 0
    unmergeable = set()
    while heap:
        L, s = heapq.heappop(heap)
        if s not in segs or abs(length[s] - L) > 1e-9:
            continue
        ks = [int(key_of(segs[s][:1])[0]), int(key_of(segs[s][-1:])[0])]
        cands = set()
        for k in ks:
            cands |= {j for j in ends.get(k, ()) if j != s and tile[j] == tile[s]}
        if not cands:
            unmergeable.add(s)
            continue
        n = max(cands, key=lambda j: (length[j], -j))
        ls, ln = segs[s], segs[n]
        kn = {int(key_of(ln[:1])[0]), int(key_of(ln[-1:])[0])}
        shared = [k for k in ks if k in kn]
        k = shared[0]
        # orient s so it ends at k, n so it starts at k
        if int(key_of(ls[-1:])[0]) != k:
            ls = ls[::-1]
        if int(key_of(ln[:1])[0]) != k:
            ln = ln[::-1]
        new = np.concatenate([ls, ln[1:]])
        del_ends(s)
        del_ends(n)
        t = tile[s]
        del segs[s], segs[n]
        m = next_id
        next_id += 1
        segs[m] = new
        tile[m] = t
        length[m] = line_len(new)
        add_ends(m)
        n_merged += 1
        if length[m] < MIN_SEG_LEN_M:
            heapq.heappush(heap, (length[m], m))
    return list(segs.values()), n_merged, len(unmergeable & set(segs.keys()))


def azimuths(lines_cm: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(len(lines_cm))
    for i, l in enumerate(lines_cm):
        d = np.diff(l / 100.0, axis=0)
        w = np.hypot(d[:, 0], d[:, 1])
        th = np.arctan2(d[:, 0], d[:, 1])  # compass bearing from north, clockwise
        c = (w * np.cos(2 * th)).sum()
        s = (w * np.sin(2 * th)).sum()
        out[i] = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
    return out


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    tiles = gpd.read_file(TILES_CORE)
    area = shapely.union_all(tiles.geometry.values).buffer(200.0)

    roads = gpd.read_file(ROADS_GPKG, layer="roads_fua")
    stats["roads_features"] = int(len(roads))
    stats["roads_null_geometry"] = int(roads.geometry.isna().sum())
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads["YOLTIP"] = roads["YOLTIP"].fillna("UNKNOWN")
    parts, pidx = shapely.get_parts(roads.geometry.values, return_index=True)
    cls = roads["YOLTIP"].values[pidx]
    stats["roads_parts_after_explode"] = int(len(parts))
    stats["roads_total_length_km_all"] = float(shapely.length(parts).sum() / 1000)
    # restrict to the area of interest
    tr = STRtree(parts)
    sel = np.unique(tr.query(area, predicate="intersects"))
    parts, cls = parts[sel], cls[sel]
    stats["roads_parts_in_aoi"] = int(len(parts))
    # snap, drop repeated vertices
    lines, kept = snap_lines(parts)
    parts, cls = parts[kept], cls[kept]
    # exact duplicates (direction-agnostic)
    snapped = np.empty(len(lines), dtype=object)
    for i, l in enumerate(lines):
        snapped[i] = shapely.linestrings(l / 100.0)
    norm = shapely.to_wkb(shapely.normalize(snapped))
    dup = pd.Series(norm).duplicated().values
    stats["duplicate_lines_removed"] = int(dup.sum())
    lines = [l for l, d in zip(lines, dup) if not d]
    snapped = snapped[~dup]
    cls = cls[~dup]
    stats["input_lines_for_network"] = len(lines)
    stats["input_length_km"] = float(shapely.length(snapped).sum() / 1000)
    orig_tree = STRtree(snapped)
    print(f"[{time.time()-t0:5.1f}s] lines in AOI after dedup: {len(lines)}", flush=True)

    pieces, n_nodes, n_split = split_at_nodes(lines)
    stats["nodes_deg_ge3"] = n_nodes
    stats["lines_split_at_nodes"] = n_split
    stats["pieces_after_node_split"] = len(pieces)
    print(f"[{time.time()-t0:5.1f}s] nodes {n_nodes}, pieces {len(pieces)}", flush=True)

    chains = chain_merge(pieces)
    stats["segments_after_chain_merge"] = len(chains)
    print(f"[{time.time()-t0:5.1f}s] chains {len(chains)}", flush=True)

    tiled, n_cross = split_at_tile_borders(chains)
    stats["lines_crossing_tile_border"] = n_cross
    stats["segments_after_tile_split"] = len(tiled)
    print(f"[{time.time()-t0:5.1f}s] after tile split {len(tiled)} (crossing {n_cross})", flush=True)
    lens_pre = np.array([line_len(l) for l in tiled])
    stats["n_lt20m_before_merge"] = int((lens_pre < MIN_SEG_LEN_M).sum())

    final, n_merged, n_unmerge = merge_short(tiled)
    stats["short_merges_done"] = n_merged
    stats["segments_final_pre_ps"] = len(final)
    lens = np.array([line_len(l) for l in final])
    stats["n_lt20m_after_merge"] = int((lens < MIN_SEG_LEN_M).sum())
    stats["short_unmergeable_no_same_tile_neighbour"] = n_unmerge
    print(f"[{time.time()-t0:5.1f}s] merged {n_merged}; final {len(final)}; <20 m left {stats['n_lt20m_after_merge']}", flush=True)

    geoms = np.empty(len(final), dtype=object)
    for i, l in enumerate(final):
        geoms[i] = shapely.linestrings(l / 100.0)
    mids = shapely.get_coordinates(shapely.line_interpolate_point(geoms, 0.5, normalized=True))
    kx = np.floor(mids[:, 0] / TILE_M).astype(int)
    ky = np.floor(mids[:, 1] / TILE_M).astype(int)
    tile_ids = np.array([f"T_{a:04d}_{b:04d}" for a, b in zip(kx, ky)])
    order = np.lexsort((mids[:, 1], mids[:, 0], tile_ids))
    geoms = geoms[order]
    final = [final[i] for i in order]
    tile_ids = tile_ids[order]
    mids = mids[order]
    seg_id = np.arange(1, len(geoms) + 1, dtype=np.int64)
    az = azimuths(final)
    length_m = shapely.length(geoms)

    # road class: nearest original line, majority over sample points (~every 15 m)
    ns = np.maximum(1, np.floor(length_m / 15.0).astype(int))
    rep = np.repeat(np.arange(len(geoms)), ns)
    frac = (np.arange(ns.sum()) - np.repeat(np.cumsum(ns) - ns, ns) + 0.5) / np.repeat(ns, ns)
    spts = shapely.line_interpolate_point(geoms[rep], frac, normalized=True)
    qi, ti = orig_tree.query_nearest(spts, all_matches=False)
    cdf = pd.DataFrame({"seg": rep[qi], "cls": cls[ti]})
    road_class = cdf.groupby("seg")["cls"].agg(lambda s: s.value_counts().index[0]).reindex(range(len(geoms))).fillna("UNKNOWN").values

    gdf = gpd.GeoDataFrame({
        "seg_id": seg_id, "tile_id": tile_ids, "length_m": length_m, "azimuth_deg": az,
        "road_class": road_class,
    }, geometry=geoms, crs="EPSG:5253")
    gdf.to_file(PRE_PS_GPKG, layer="segments_pre_ps", driver="GPKG")
    stats["length_km_final_pre_ps"] = float(length_m.sum() / 1000)
    stats["road_class_counts_pre_ps"] = {k: int(v) for k, v in pd.Series(road_class).value_counts().items()}
    stats["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "network_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    lower_priority()
    main()
