"""WP3.0 step 4: dynamic segment metrics (analysis_schema.md A.3), resumable.

For every tile that has BOTH design days complete in data/street/utci/<DAY>/utci_{street,uniform}/ and in
data/street/solweig/<DAY>/{tmrt,shadow,svf} (files present, readable, older than --min-age-s so a half-written file is never
read), summarise the PS cells of every segment (ps_index) and append the rows to data/street/analysis/seg_metrics_<DAY>.parquet.

Hour labels: schema column suffix NN is the local hour label of data/street/solweig/band_hour_table.csv / the met-file
`it` column (NN = 06..21). The 16-band rasters hold these hours in order, so hour NN is raster band (NN - 5), 1-based
(band_hour_table: band 10 = local_time 15:00 = schema label 15).
  utci_bNN       median over PS cells of utci_street band (NN-5)
  utci_unif_b15  median of utci_uniform at label 15
  tmrt_b15(_p90) median / 90th percentile of Tmrt at label 15
  v10eff_b15     median clip(U10 x ratio, 0.5, 17), ratio = interpolate_ratio_at_direction(ratio16, Wd) with U10 and Wd of
                 the met-file row it == 15 of that day;  ratio_med_b15 = median ratio
  svf_med        median SVF (day-invariant)
  shade_frac_10_18  mean over PS cells and labels 10..18 of (1 - sunlit/100); shadow raster = sunlit % (255 nodata)
  m2_strong / m2_vstrong  number of labels 10..18 whose segment-median utci_street >= 32 / >= 38 (NaN if any median is NaN)
  n_ps_valid     PS cells with a valid utci_street at label 15
partial_run = (tiles done < 427); n_tiles_at_extraction = tiles done (both days) after this run.
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

from common import *  # noqa

import geopandas as gpd
import rasterio

N_TILES_TOTAL = 427
HOURS = list(range(6, 22))
UTCI_COLS = [f"utci_b{h:02d}" for h in HOURS]
SCHEMA_COLS = (["seg_id", "n_ps_valid"] + UTCI_COLS + ["utci_unif_b15", "tmrt_b15", "tmrt_b15_p90", "v10eff_b15",
               "ratio_med_b15", "svf_med", "shade_frac_10_18", "m2_strong", "m2_vstrong", "partial_run",
               "n_tiles_at_extraction"])
NODATA = -9999.0


def met_b15(day: str):
    """U10, Wd and VRB flag of met row it == 15. WP2.3b / lock A7 item 3: when the day has a corrected v2 met file, it and
    data/street/forcing_summer/vrb_flags.csv are used; in a VRB hour Wd is undefined (-999 in v2, returned as NaN) and the
    ratio is the mean of the 16 sector ratios (see extract_tile_day). Days without a v2 file (IMBAT) behave as before."""
    from scripts.street.utci.vrb_utils import MET_V2, read_vrb_flags
    v2 = MET_V2[day]
    if v2.exists():
        df = pd.read_csv(v2, sep=r"\s+")
        vrb = bool(read_vrb_flags(day).get(15, False))
    else:
        df = pd.read_csv(MET_FILES[day], sep=r"\s+")
        vrb = False
        assert not read_vrb_flags(day).get(15, False), f"{day}: VRB at it=15 but no v2 met file"
    r = df[df["it"] == 15].iloc[0]
    wd = float(r["Wd"])
    if vrb:
        assert wd == -999, f"{day}: VRB hour must carry Wd=-999 in the v2 file, got {wd}"
        wd = float("nan")
    return float(r["Wind"]), wd, vrb


def tile_paths(tid: str, day: str):
    return [SOLWEIG_DIR / day / "tmrt" / f"{tid}.tif", SOLWEIG_DIR / day / "shadow" / f"{tid}.tif",
            SOLWEIG_DIR / day / "svf" / f"{tid}.tif", UTCI_DIR / day / "utci_street" / f"{tid}.tif",
            UTCI_DIR / day / "utci_uniform" / f"{tid}.tif"]


def tile_ready(tid: str, min_age_s: float) -> bool:
    if not (RATIO16_DIR / f"{tid}.tif").exists() or not (PS_DIR / f"{tid}.tif").exists():
        return False
    now = time.time()
    for day in DAYS:
        for p in tile_paths(tid, day):
            if not p.exists() or now - p.stat().st_mtime < min_age_s:
                return False
    return True


def extract_tile_day(args):
    tid, day = args
    from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction
    u10, wd, vrb15 = met_b15(day)
    with rasterio.open(PS_DIR / f"{tid}.tif") as s:
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
    # VRB hour (rule A7): direction-averaged sheltering = mean of the 16 sector ratios (same as compute_street_utci.py)
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
    out["tmrt_b15_p90"] = gb["tmrt_b15"].quantile(0.9)
    out["shade_frac_10_18"] = gb["shade_frac_10_18"].mean()
    h1018 = [f"utci_b{h:02d}" for h in range(10, 19)]
    ok = med[h1018].notna().all(axis=1)
    out["m2_strong"] = (med[h1018] >= 32.0).sum(axis=1).where(ok).astype("float64")
    out["m2_vstrong"] = (med[h1018] >= 38.0).sum(axis=1).where(ok).astype("float64")
    out = out.reset_index()
    return tid, day, out


def atomic_parquet(df: pd.DataFrame, path):
    tmp = str(path) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-age-s", type=float, default=120.0)
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force-day", choices=list(DAYS), default=None,
                    help="WP2.3b: re-extract this day for every tile already extracted (both days done) and replace its rows in "
                         "seg_metrics_<DAY>.parquet; other day, tiles-done file, partial_run and n_tiles are left as they are")
    a = ap.parse_args()
    lower_priority()
    t0 = time.time()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tiles = sorted(gpd.read_file(TILES_CORE)["tile_id"].tolist())
    assert len(tiles) == N_TILES_TOTAL
    done = pd.read_csv(TILES_DONE_CSV) if TILES_DONE_CSV.exists() else pd.DataFrame(columns=["tile_id", "day", "extracted_at"])
    done_both = set(done.groupby("tile_id")["day"].nunique().loc[lambda s: s >= 2].index)
    if a.force_day:
        force_reextract(a, sorted(t for t in done_both if (not a.tiles or t in a.tiles)))
        return
    todo = [t for t in tiles if t not in done_both and (not a.tiles or t in a.tiles)]
    ready = [t for t in todo if tile_ready(t, a.min_age_s)]
    if a.limit:
        ready = ready[:a.limit]
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} tiles done={len(done_both)} pending={len(todo)} "
          f"ready now={len(ready)}", flush=True)
    new = {d: [] for d in DAYS}
    new_done = []
    failed = []
    if ready:
        jobs = [(t, d) for t in ready for d in DAYS]
        res = {}
        if a.workers > 1:
            with mp.get_context("spawn").Pool(a.workers, initializer=lower_priority) as pool:
                for tid, day, df in pool.imap_unordered(_safe, jobs, chunksize=1):
                    res[(tid, day)] = df
        else:
            for j in jobs:
                tid, day, df = _safe(j)
                res[(tid, day)] = df
        for t in ready:
            if all(isinstance(res.get((t, d)), pd.DataFrame) for d in DAYS):
                for d in DAYS:
                    new[d].append(res[(t, d)])
                stamp = datetime.datetime.now().isoformat(timespec="seconds")
                new_done += [{"tile_id": t, "day": d, "extracted_at": stamp} for d in DAYS]
            else:
                failed.append(t)
    if new_done:
        n_tiles = len(done_both) + len(new_done) // 2
        for d in DAYS:
            p = ANALYSIS_DIR / f"seg_metrics_{d}.parquet"
            old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=SCHEMA_COLS)
            add = pd.concat(new[d], ignore_index=True) if new[d] else pd.DataFrame()
            if len(add):
                old = old[~old["seg_id"].isin(add["seg_id"])]
                old = pd.concat([old, add], ignore_index=True) if len(old) else add
            old = old.sort_values("seg_id").reset_index(drop=True)
            old["partial_run"] = bool(n_tiles < N_TILES_TOTAL)
            old["n_tiles_at_extraction"] = np.int64(n_tiles)
            old["seg_id"] = old["seg_id"].astype(np.int64)
            old["n_ps_valid"] = old["n_ps_valid"].astype(np.int64)
            atomic_parquet(old[SCHEMA_COLS], p)
        done = pd.concat([done, pd.DataFrame(new_done)], ignore_index=True)
        tmp = str(TILES_DONE_CSV) + ".tmp"
        done.to_csv(tmp, index=False)
        os.replace(tmp, TILES_DONE_CSV)
    n_done = done.groupby("tile_id")["day"].nunique().loc[lambda s: s >= 2].shape[0] if len(done) else 0
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} extracted {len(new_done)//2} tiles "
          f"({time.time()-t0:.1f}s); failed={failed}; total done={n_done}/{N_TILES_TOTAL}"
          f"{'  COMPLETE' if n_done >= N_TILES_TOTAL else '  PARTIAL (%d tiles)' % n_done}", flush=True)
    # exit code 0 always; completeness is in the tiles_done file


def force_reextract(a, tiles_done: list[str]):
    """Re-extract one day for tiles already extracted. All-or-nothing: if any tile fails or its UTCI raster is older than the
    corrected met file (stale, not yet recomputed), nothing is written."""
    from scripts.street.utci.vrb_utils import MET_V2
    day = a.force_day
    t0 = time.time()
    v2m = MET_V2[day].stat().st_mtime if MET_V2[day].exists() else 0.0
    stale = [t for t in tiles_done if (UTCI_DIR / day / "utci_street" / f"{t}.tif").stat().st_mtime < v2m]
    if stale:
        print(f"ABORT: {len(stale)} tiles have a {day} utci_street raster older than the v2 met file: {stale[:5]}", flush=True)
        return
    jobs = [(t, day) for t in tiles_done]
    res = {}
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers, initializer=lower_priority) as pool:
            for tid, d, df in pool.imap_unordered(_safe, jobs, chunksize=1):
                res[tid] = df
    else:
        for j in jobs:
            tid, d, df = _safe(j)
            res[tid] = df
    failed = [t for t in tiles_done if not isinstance(res.get(t), pd.DataFrame)]
    if failed:
        print(f"ABORT: failed tiles {failed}; nothing written", flush=True)
        return
    p = ANALYSIS_DIR / f"seg_metrics_{day}.parquet"
    old = pd.read_parquet(p)
    add = pd.concat([res[t] for t in tiles_done], ignore_index=True)
    keep_partial, keep_n = old["partial_run"].iloc[0], old["n_tiles_at_extraction"].iloc[0]
    old = old[~old["seg_id"].isin(add["seg_id"])]
    add["partial_run"] = bool(keep_partial)
    add["n_tiles_at_extraction"] = np.int64(keep_n)
    new = pd.concat([old, add], ignore_index=True).sort_values("seg_id").reset_index(drop=True)
    new["seg_id"] = new["seg_id"].astype(np.int64)
    new["n_ps_valid"] = new["n_ps_valid"].astype(np.int64)
    new["partial_run"] = bool(keep_partial)
    new["n_tiles_at_extraction"] = np.int64(keep_n)
    atomic_parquet(new[SCHEMA_COLS], p)
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} force re-extracted {day} for {len(tiles_done)} tiles "
          f"({len(add)} segments replaced, {len(old)} kept, {time.time()-t0:.1f}s); partial_run={bool(keep_partial)} "
          f"n_tiles={int(keep_n)}", flush=True)


def _safe(job):
    try:
        return extract_tile_day(job)
    except Exception as e:  # unreadable / half written file: retry next hour
        print(f"ERROR {job}: {e!r}", flush=True)
        return job[0], job[1], None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mp.freeze_support()
    main()
