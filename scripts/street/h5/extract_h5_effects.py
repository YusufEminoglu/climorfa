"""CLIMORFA Street — H5 per-segment scenario effects (A10 ruling 4; validation input for ruling 5).

For every tile of a scenario and both design days, the segment metrics of the BASELINE and of the SCENARIO are extracted
by calling scripts/street/analysis/extract_seg_metrics.extract_tile_day itself (the WP3.0 function, not a copy): same
ps_index, same per-band segment median over PS cells, same bands 10-18, same >= 32 / >= 38 degC counts, same VRB/met
handling. Only the raster roots the function reads are switched (module globals SOLWEIG_DIR / UTCI_DIR, and PS_DIR /
RATIO16_DIR for synthetic tests). The scenario runner writes <scen>/solweig/<DAY>/{tmrt,shadow,svf} and
<scen>/utci/<DAY>/{utci_street,utci_uniform} in exactly the main-run layout.

Output per scenario and day: data/street/h5/<scen>/seg_effects_<DAY>.parquet (SYNTHETIC_ prefix in synthetic runs)
  seg_id, tile_id, day, n_ps (segments.gpkg), n_ps_valid_base, n_ps_valid_scen,
  m2_38_base, m2_38_scen, m2_32_base, m2_32_scen, dm2_38 = base - scen, dm2_32, m1_base, m1_scen, dm1 = m1_base - m1_scen,
  tmrt_b15_base, tmrt_b15_scen, scenario, partial_run, n_tiles_extracted, n_tiles_expected, is_synthetic
  A11 (descriptive companions, post-processing of the same UTCI rasters): cell_h38_base / _scen = mean over the segment's PS
  cells of the hours 10-18 with cell UTCI_street >= 38 degC, dcell_h38 = base - scen; mean_utci_b15_base / _scen = mean
  UTCI_street over the PS cells at band 15, dmean_b15 = base - scen.
(positive dm2 / dm1 = fewer heat-stress hours / cooler with trees). A tile is extracted only when all scenario rasters
validate and are newer than the scenario tree DSM; partial_run = (tiles extracted < tiles expected).
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# The WP3.0 extractor does `from common import *` from its own folder. Import it with that folder on the path, then
# remove the folder and the cached `common` again: other packages (scripts/street/stats, used by the figures) have
# their own module named `common`, which must not be shadowed. esm keeps its own references to the names it imported.
_AN = str(PROJECT_ROOT / "scripts/street/analysis")
_had_common = sys.modules.get("common")
sys.path.insert(0, _AN)
try:
    import extract_seg_metrics as esm  # noqa: E402
finally:
    sys.path.remove(_AN)
    if _had_common is None:
        sys.modules.pop("common", None)
    else:
        sys.modules["common"] = _had_common

H5_BASE = PROJECT_ROOT / "data/street/h5"
DAYS = ("IMBAT", "CALM")
BASE_DIRS = {"solweig": esm.SOLWEIG_DIR, "utci": esm.UTCI_DIR, "ps": esm.PS_DIR, "ratio16": esm.RATIO16_DIR}
TILE_KEYS = [("solweig", "tmrt", 16), ("solweig", "shadow", 16), ("solweig", "svf", 1), ("utci", "utci_street", 16),
             ("utci", "utci_uniform", 16)]


def _validate(path: Path, bands: int) -> bool:
    from scripts.street.h5.run_scenario_solweig import validate_geotiff
    return validate_geotiff(path, bands)


def raster_root_ok(dirs: dict, tid: str, days=DAYS, ref: Path | None = None) -> bool:
    for d in days:
        for root, sub, nb in TILE_KEYS:
            p = Path(dirs[root]) / d / sub / f"{tid}.tif"
            if not _validate(p, nb):
                return False
            if ref is not None and ref.exists() and p.stat().st_mtime < ref.stat().st_mtime:
                return False
    return (Path(dirs["ps"]) / f"{tid}.tif").exists() and (Path(dirs["ratio16"]) / f"{tid}.tif").exists()


def _init_worker():
    esm.lower_priority()          # below-normal priority, as the WP3.0 extractor workers


def _job(args):
    """One (tile, day, which) extraction with esm.extract_tile_day; module globals switched to the given roots."""
    tid, day, which, dirs = args
    try:
        esm.SOLWEIG_DIR = Path(dirs["solweig"])
        esm.UTCI_DIR = Path(dirs["utci"])
        esm.PS_DIR = Path(dirs["ps"])
        esm.RATIO16_DIR = Path(dirs["ratio16"])
        _, _, df = esm.extract_tile_day((tid, day))
        return tid, day, which, df
    except Exception as e:  # reported and treated as not extracted
        print(f"ERROR {tid} {day} {which}: {e!r}", flush=True)
        return tid, day, which, None


def cell_metrics(tid: str, day: str, dirs: dict) -> pd.DataFrame:
    """A11: per segment, the mean over PS cells of the hours 10-18 with cell UTCI_street >= 38 degC, and the mean UTCI_street
    over PS cells at band 15. A PS cell counts only if all nine bands are valid (band 15 mean: if band 15 is valid)."""
    with rasterio.open(Path(dirs["ps"]) / f"{tid}.tif") as s:
        ps = s.read(1).ravel()
    m = ps > 0
    if not m.any():
        return pd.DataFrame(columns=["seg_id", "cell_h38", "mean_utci_b15"])
    with rasterio.open(Path(dirs["utci"]) / day / "utci_street" / f"{tid}.tif") as s:
        u = s.read(list(range(5, 14))).reshape(9, -1)[:, m].astype(np.float64)   # bands 5..13 = labels 10..18
    bad = (u == esm.NODATA) | ~np.isfinite(u)
    ok9 = ~bad.any(axis=0)
    hours = (u >= 38.0).sum(axis=0).astype(np.float64)
    b15 = u[5].copy()                                                            # label 15 = 6th of labels 10..18
    b15[bad[5]] = np.nan
    df = pd.DataFrame({"seg_id": ps[m].astype(np.int64), "h": np.where(ok9, hours, np.nan), "b15": b15})
    g = df.groupby("seg_id")
    return pd.DataFrame({"cell_h38": g["h"].mean(), "mean_utci_b15": g["b15"].mean()}).reset_index()


def _full_job(args):
    """_job (the WP3.0 extractor) plus the A11 cell metrics of the same rasters."""
    tid, day, which, dirs = args
    out = _job(args)
    if out[3] is None:
        return out
    try:
        return tid, day, which, out[3].merge(cell_metrics(tid, day, dirs), on="seg_id", how="left")
    except Exception as e:
        print(f"ERROR {tid} {day} {which} (A11 cell metrics): {e!r}", flush=True)
        return tid, day, which, None


def combine(base: pd.DataFrame, scen: pd.DataFrame, tid: str, day: str) -> pd.DataFrame:
    if len(base) == 0 and len(scen) == 0:          # tile without PS cells
        return pd.DataFrame()
    b = base.set_index("seg_id")
    s = scen.set_index("seg_id")
    if set(b.index) != set(s.index):
        raise RuntimeError(f"{tid} {day}: baseline and scenario segment sets differ ({len(b)} vs {len(s)})")
    s = s.loc[b.index]
    out = pd.DataFrame(index=b.index)
    out["tile_id"] = tid
    out["day"] = day
    out["n_ps_valid_base"] = b["n_ps_valid"].astype(np.int64)
    out["n_ps_valid_scen"] = s["n_ps_valid"].astype(np.int64)
    out["m2_38_base"], out["m2_38_scen"] = b["m2_vstrong"], s["m2_vstrong"]
    out["m2_32_base"], out["m2_32_scen"] = b["m2_strong"], s["m2_strong"]
    out["dm2_38"] = out["m2_38_base"] - out["m2_38_scen"]
    out["dm2_32"] = out["m2_32_base"] - out["m2_32_scen"]
    out["m1_base"], out["m1_scen"] = b["utci_b15"], s["utci_b15"]
    out["dm1"] = out["m1_base"] - out["m1_scen"]
    out["tmrt_b15_base"], out["tmrt_b15_scen"] = b["tmrt_b15"], s["tmrt_b15"]
    out["cell_h38_base"], out["cell_h38_scen"] = b["cell_h38"], s["cell_h38"]
    out["dcell_h38"] = out["cell_h38_base"] - out["cell_h38_scen"]
    out["mean_utci_b15_base"], out["mean_utci_b15_scen"] = b["mean_utci_b15"], s["mean_utci_b15"]
    out["dmean_b15"] = out["mean_utci_b15_base"] - out["mean_utci_b15_scen"]
    return out.reset_index()


def extract_effects(scenario: str, expected_tiles: list[str], days=DAYS, scen_root: Path | None = None,
                    base_dirs: dict | None = None, out_dir: Path | None = None, segments_path: Path | None = None,
                    workers: int = 2, is_synthetic: bool | None = None, check_baseline_parquet: bool = True) -> dict:
    """Extract baseline and scenario segment metrics for `expected_tiles` and write seg_effects_<DAY>.parquet."""
    scen_root = Path(scen_root) if scen_root else H5_BASE / scenario
    out_dir = Path(out_dir) if out_dir else scen_root
    base_dirs = dict(BASE_DIRS, **(base_dirs or {}))
    scen_dirs = {"solweig": scen_root / "solweig", "utci": scen_root / "utci", "ps": base_dirs["ps"],
                 "ratio16": base_dirs["ratio16"]}
    syn = ("SYNTHETIC" in str(out_dir).upper()) if is_synthetic is None else bool(is_synthetic)
    if syn and "SYNTHETIC" not in str(out_dir).upper():
        raise RuntimeError(f"rule 6: synthetic effects must be written under a SYNTHETIC folder ({out_dir})")
    prefix = "SYNTHETIC_" if syn else ""
    tree_dir = scen_root / "tree_dsm"
    ready, not_ready = [], []
    for t in expected_tiles:
        ref = tree_dir / f"{t}.tif"
        if raster_root_ok(scen_dirs, t, days, ref if ref.exists() else None) and raster_root_ok(base_dirs, t, days):
            ready.append(t)
        else:
            not_ready.append(t)
    print(f"[{scenario}] expected {len(expected_tiles)} tiles, ready {len(ready)}, not ready {len(not_ready)} "
          f"{not_ready[:5]}", flush=True)
    jobs = [(t, d, w, base_dirs if w == "base" else scen_dirs) for t in ready for d in days for w in ("base", "scen")]
    res = {}
    t0 = time.time()
    if workers > 1 and len(jobs) > 2:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            for tid, day, w, df in pool.imap_unordered(_full_job, jobs, chunksize=1):
                res[(tid, day, w)] = df
    else:
        for j in jobs:
            tid, day, w, df = _full_job(j)
            res[(tid, day, w)] = df
    print(f"[{scenario}] extracted {len(jobs)} tile-day-runs in {time.time() - t0:.0f} s", flush=True)
    failed = sorted({t for (t, d, w), df in res.items() if df is None})
    done = [t for t in ready if t not in set(failed)]
    seg_n_ps = None
    sp = Path(segments_path) if segments_path else esm.SEGMENTS_GPKG
    if sp.exists():
        import geopandas as gpd
        g = gpd.read_file(sp, layer="segments", columns=["seg_id", "n_ps"], ignore_geometry=True)
        seg_n_ps = dict(zip(g["seg_id"].astype(np.int64), g["n_ps"]))
    partial = len(done) < len(expected_tiles)
    manifest = {"scenario": scenario, "extracted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "n_tiles_expected": len(expected_tiles), "n_tiles_extracted": len(done), "partial_run": partial,
                "not_ready": not_ready, "failed": failed, "is_synthetic": syn, "scen_root": str(scen_root),
                "base_dirs": {k: str(v) for k, v in base_dirs.items()}, "extractor": "extract_seg_metrics.extract_tile_day",
                "files": {}, "baseline_parquet_check": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in days:
        parts = [combine(res[(t, d, "base")], res[(t, d, "scen")], t, d) for t in done]
        eff = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["seg_id"])
        if len(eff):
            eff["seg_id"] = eff["seg_id"].astype(np.int64)
            eff.insert(3, "n_ps", eff["seg_id"].map(seg_n_ps) if seg_n_ps else np.nan)
        eff["scenario"] = scenario
        eff["partial_run"] = partial
        eff["n_tiles_extracted"] = np.int64(len(done))
        eff["n_tiles_expected"] = np.int64(len(expected_tiles))
        eff["is_synthetic"] = syn
        eff = eff.sort_values("seg_id").reset_index(drop=True) if len(eff) else eff
        fp = out_dir / f"{prefix}seg_effects_{d}.parquet"
        tmp = str(fp) + ".tmp"
        eff.to_parquet(tmp, index=False)
        os.replace(tmp, fp)
        manifest["files"][d] = str(fp)
        print(f"[{scenario}] {d}: {len(eff)} segments -> {fp}  partial_run={partial}", flush=True)
        bp = esm.ANALYSIS_DIR / f"seg_metrics_{d}.parquet"
        if check_baseline_parquet and not syn and bp.exists() and len(eff):
            ref = pd.read_parquet(bp, columns=["seg_id", "m2_strong", "m2_vstrong"]).set_index("seg_id")
            j = eff.set_index("seg_id").join(ref, how="inner")
            d38 = (j["m2_38_base"] - j["m2_vstrong"]).abs()
            d32 = (j["m2_32_base"] - j["m2_strong"]).abs()
            manifest["baseline_parquet_check"][d] = {"n_compared": int(len(j)), "max_abs_diff_m2_38": float(d38.max()),
                                                     "max_abs_diff_m2_32": float(d32.max()),
                                                     "n_mismatch": int(((d38 > 0) | (d32 > 0)).sum())}
            print(f"[{scenario}] {d}: re-extracted baseline vs {bp.name}: {manifest['baseline_parquet_check'][d]}")
    (out_dir / f"{prefix}seg_effects_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def default_expected_tiles(scenario: str) -> list[str]:
    from scripts.street.h5.run_scenario_solweig import default_tiles
    return default_tiles(scenario)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mp.freeze_support()
    ap = argparse.ArgumentParser(description="Per-segment H5 scenario effects (A10).")
    ap.add_argument("--scenario", required=True, help="P or S_exact")
    ap.add_argument("--tiles", default="default", help="'default' (P: 427; S_exact: validation tiles) or list/file")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="report how many tiles are ready; extract nothing")
    a = ap.parse_args()
    from scripts.street.h5.run_scenario_solweig import read_tiles_arg
    tiles = read_tiles_arg(a.tiles, a.scenario)
    if a.dry_run:
        sr = H5_BASE / a.scenario
        sd = {"solweig": sr / "solweig", "utci": sr / "utci", "ps": BASE_DIRS["ps"], "ratio16": BASE_DIRS["ratio16"]}
        n = sum(raster_root_ok(sd, t) for t in tiles)
        print(f"[DRY-RUN {a.scenario}] {n}/{len(tiles)} tiles have complete scenario rasters; nothing extracted.")
        sys.exit(0)
    extract_effects(a.scenario, tiles, workers=a.workers)
