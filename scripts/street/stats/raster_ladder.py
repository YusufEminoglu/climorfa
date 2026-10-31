"""Raster pass over finished tiles (read-only on data/street/*).

Why this exists: the data contract has no PS-cell-level M1 for the 50/100/250/500 m ladder and no city-wide PS-cell
reference for H4. This module reads, for each finished tile, the band-15 UTCI/Tmrt rasters and the ps_index raster
and aggregates them exactly as the lock states (median over PS cells):

  * grid-ladder cells: n_ps, median M1 over PS cells in the cell, majority LCZ over PS cells (LCZ of the owning
    segment, i.e. lcz_primary), cell block id
  * H4 reference: median Tmrt and median v10_eff (clip(U10 x ratio, 0.5, 17)) over ALL PS cells, at band 15

Band 15 = 10th band (index 9) of the 16-band rasters (06:00..21:00 LST).
The pure function `aggregate_tile` is unit-tested against a brute-force loop on random arrays (independent check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

LEVELS = (50, 100, 250, 500)
MIN_PS_CELLS = 25          # same minimum as the lock's segment rule; applied to grid units (assumption, reported)
NODATA = -9999.0
B15_IDX = 9


def tile_origin(tile_id: str) -> tuple[int, int]:
    _, xs, ys = tile_id.split("_")
    return int(xs) * 1000, int(ys) * 1000


def aggregate_tile(m1: np.ndarray, ps: np.ndarray, seg_ids_sorted: np.ndarray, seg_lcz_sorted: np.ndarray,
                   x0: int, y0: int, levels=LEVELS) -> dict[int, pd.DataFrame]:
    """m1, ps: (500, 500). Returns {L: DataFrame[cell_id, n_ps, m1_med, lcz, bx, by]} for each level L (m)."""
    valid = (ps > 0) & np.isfinite(m1) & (m1 > NODATA + 1)
    rr, cc = np.nonzero(valid)
    if rr.size == 0:
        return {L: pd.DataFrame(columns=["cell_id", "n_ps", "m1_med", "lcz", "bx", "by"]) for L in levels}
    vals = m1[rr, cc].astype(float)
    sid = ps[rr, cc]
    pos = np.searchsorted(seg_ids_sorted, sid)
    pos = np.clip(pos, 0, len(seg_ids_sorted) - 1)
    ok = seg_ids_sorted[pos] == sid
    lcz = np.where(ok, seg_lcz_sorted[pos], -1)
    xs = 2 * cc                   # pixel offset (m) from tile x0 (west edge)
    ys = 1000 - 2 * rr - 1        # pixel centre offset (m) from tile y0 (south edge)
    out = {}
    for L in levels:
        cx = (x0 + xs) // L
        cy = (y0 + ys) // L
        df = pd.DataFrame({"cx": cx, "cy": cy, "v": vals, "lcz": lcz})
        g = df.groupby(["cx", "cy"])
        res = g["v"].agg(n_ps="size", m1_med="median").reset_index()
        # majority LCZ over PS cells with a VALID LCZ only (A6: 0 / unknown segment = no-data, not a class)
        cnt = df[df["lcz"] > 0].groupby(["cx", "cy", "lcz"]).size().reset_index(name="k")
        cnt = cnt.sort_values(["cx", "cy", "k", "lcz"], ascending=[True, True, False, True])
        maj = cnt.drop_duplicates(["cx", "cy"])[["cx", "cy", "lcz"]]
        res = res.merge(maj, on=["cx", "cy"], how="left")
        res["lcz"] = res["lcz"].fillna(0).astype(int)
        res["cell_id"] = res["cx"].astype(str) + "_" + res["cy"].astype(str)
        res["bx"] = np.floor((res["cx"] * L + L / 2) / cm.BLOCK_M).astype(int)
        res["by"] = np.floor((res["cy"] * L + L / 2) / cm.BLOCK_M).astype(int)
        out[L] = res[["cell_id", "n_ps", "m1_med", "lcz", "bx", "by"]]
    return out


def read_band(path: Path, idx: int = B15_IDX) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as s:
        return s.read(idx + 1)


def raster_pass(day: str, seg: pd.DataFrame, tiles: list[str]) -> dict:
    """Loop over finished tiles. Returns dict(ladder={L: DataFrame}, tmrt_ref, v_ref, n_ps_total, n_tiles_used)."""
    from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction
    import rasterio
    root = cm.PROJECT_ROOT / "data" / "street"
    fo = cm.forcing_summary(day)
    seg_sorted = seg.sort_values("seg_id")
    sid_s = seg_sorted["seg_id"].to_numpy()
    lcz_s = seg_sorted["lcz_primary"].to_numpy().astype(int)
    acc = {L: [] for L in LEVELS}
    tm_all, v_all = [], []
    used = 0
    for t in tiles:
        p_ps = cm.DATA_ANALYSIS / "ps_index" / f"{t}.tif"
        p_u = root / "utci" / day / "utci_street" / f"{t}.tif"
        p_t = root / "solweig" / day / "tmrt" / f"{t}.tif"
        p_r = root / "wind" / "ratio16" / f"{t}.tif"
        if not (p_ps.exists() and p_u.exists() and p_t.exists() and p_r.exists()):
            continue
        with rasterio.open(p_ps) as s:
            ps = s.read(1)
        m1 = read_band(p_u)
        tm = read_band(p_t)
        with rasterio.open(p_r) as s:
            r16 = s.read()
            # WP2.3b / lock A7: VRB hour -> mean of the 16 sector ratios (as in compute_street_utci / extract_seg_metrics)
            ratio = r16.mean(axis=0) if fo.get("vrb15") else interpolate_ratio_at_direction(r16, fo["wd15"])
        x0, y0 = tile_origin(t)
        res = aggregate_tile(m1, ps, sid_s, lcz_s, x0, y0)
        for L in LEVELS:
            r = res[L].copy()
            r["tile_id"] = t
            acc[L].append(r)
        vm = (ps > 0) & (tm > NODATA + 1) & np.isfinite(tm) & np.isfinite(ratio)
        tm_all.append(tm[vm].astype(np.float32))
        v_all.append(np.clip(fo["u10"] * ratio[vm], 0.5, 17.0).astype(np.float32))
        used += 1
    ladder = {L: (pd.concat(acc[L], ignore_index=True) if acc[L] else pd.DataFrame()) for L in LEVELS}
    tm_cat = np.concatenate(tm_all) if tm_all else np.array([])
    v_cat = np.concatenate(v_all) if v_all else np.array([])
    return dict(ladder=ladder, tmrt_ref=float(np.median(tm_cat)) if tm_cat.size else float("nan"),
                v_ref=float(np.median(v_cat)) if v_cat.size else float("nan"), n_ps_total=int(tm_cat.size),
                n_tiles_used=used)
