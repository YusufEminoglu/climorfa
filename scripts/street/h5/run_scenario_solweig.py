"""CLIMORFA Street — H5 scenario SOLWEIG + street UTCI runner (A10 rulings 4 and 5).

  --scenario P                    all 427 tiles (tree template on every segment)
  --scenario S_exact --tiles ...  the 40 validation tiles (default: selection/s_exact_validation_tiles.csv)

Per tile, identical to the main run (scripts/street/solweig/run_solweig_citywide.py, WP2.1):
  1. stage Building_DSM, DEM and the SCENARIO tree DSM (data/street/h5/<scen>/tree_dsm/<tile>.tif; a missing scenario
     tree DSM is an error, there is no silent fallback to the baseline canopy);
  2. walls/aspect and SVF (153 patches) are RECOMPUTED with the scenario canopy (solweig_gpu.run_walls_aspect,
     solweig_gpu.calculate_svf(patch_option=2));
  3. SOLWEIG for both design days with the same met files and dates as the main run (DESIGN_DAYS of WP2.1; SOLWEIG
     Tmrt does not use the wind direction, lock A7 item 3), cropped to the inner 500 x 500 px by the WP2.1 functions;
     SVF is also copied into <day>/svf like the main run so the WP3.0 extractor can read the scenario unchanged.
Street UTCI (CPU), identical to scripts/street/utci/compute_street_utci.py (WP2.3 / WP2.3b):
  met rows via compute_street_utci.load_day_met(day) -> the v2 met file + vrb_flags.csv when a v2 file exists (CALM,
  lock A7); VRB hour -> ratio = mean of the 16 sector ratios, else interpolate_ratio_at_direction(ratio16, Wd);
  v10_eff = clip(U10 x ratio, 0.5, 17); valid cells = not roof / sea (load_tile_masks) & Tmrt valid & ratio16 valid;
  pythermalcomfort utci(limit_inputs=False, round_output=False); Ta = met column "Td" (the air temperature column of
  these UMEP-format files, as in the main run).

Resumable: a tile is skipped when all of its outputs validate (readable, 16 bands, 500 x 500; SVF 1 band) AND are newer
than the scenario tree DSM; SOLWEIG and UTCI are resumed separately. Outputs are written atomically (tmp + rename) for
UTCI. Progress: data/street/h5/<scen>/run_progress.csv.  --validate-only reports completeness (exit 1 if incomplete);
--dry-run touches no GPU and prints the GPU-hour estimate from the WP2.1 progress log.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("GDAL_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TILES_BASE = PROJECT_ROOT / "data/street/tiles/2m"
H5_BASE = PROJECT_ROOT / "data/street/h5"
RATIO16_DIR = PROJECT_ROOT / "data/street/wind/ratio16"
WP21_PROGRESS = PROJECT_ROOT / "logs/wp/WP2.1_progress.csv"
DAYS = ("IMBAT", "CALM")
N_TILES_TOTAL = 427


# ---------------------------------------------------------------------------------------------------------------------
def validate_geotiff(path, bands: int, shape=(500, 500)) -> bool:
    """Same test as run_solweig_citywide.validate_geotiff (exists, readable, band count, shape, first/last band)."""
    path = str(path)
    if not os.path.isfile(path):
        return False
    try:
        with rasterio.open(path) as s:
            if s.count != bands or s.shape != tuple(shape):
                return False
            s.read(1)
            if bands > 1:
                s.read(bands)
        return True
    except Exception:
        return False


def scen_paths(scenario: str, tile_id: str, h5_base: Path = H5_BASE) -> dict:
    b = Path(h5_base) / scenario
    p = {"tree": b / "tree_dsm" / f"{tile_id}.tif", "svf": b / "solweig" / "svf" / f"{tile_id}.tif"}
    for d in DAYS:
        for k in ("tmrt", "utci", "shadow", "svf"):
            p[f"{d}_{k}"] = b / "solweig" / d / k / f"{tile_id}.tif"
        for k in ("utci_street", "utci_uniform"):
            p[f"{d}_{k}"] = b / "utci" / d / k / f"{tile_id}.tif"
    return p


def _fresh(path: Path, ref: Path) -> bool:
    return path.exists() and ref.exists() and path.stat().st_mtime >= ref.stat().st_mtime


def solweig_done(scenario: str, tile_id: str, days=DAYS, h5_base: Path = H5_BASE) -> bool:
    p = scen_paths(scenario, tile_id, h5_base)
    keys = ["svf"] + [f"{d}_{k}" for d in days for k in ("tmrt", "utci", "shadow")]
    ok = all(validate_geotiff(p[k], 1 if k.endswith("svf") else 16) and _fresh(p[k], p["tree"]) for k in keys)
    return ok and all(validate_geotiff(p[f"{d}_svf"], 1) for d in days)


def utci_done(scenario: str, tile_id: str, days=DAYS, h5_base: Path = H5_BASE) -> bool:
    p = scen_paths(scenario, tile_id, h5_base)
    keys = [f"{d}_{k}" for d in days for k in ("utci_street", "utci_uniform")]
    return all(validate_geotiff(p[k], 16) and _fresh(p[k], p[f"{k.split('_')[0]}_tmrt"]) for k in keys)


def tile_complete(scenario: str, tile_id: str, days=DAYS, h5_base: Path = H5_BASE) -> bool:
    return solweig_done(scenario, tile_id, days, h5_base) and utci_done(scenario, tile_id, days, h5_base)


def default_tiles(scenario: str) -> list[str]:
    if scenario == "P":
        import geopandas as gpd
        return sorted(gpd.read_file(PROJECT_ROOT / "data/street/grid/tiles_core.gpkg")["tile_id"].tolist())
    if scenario == "S_exact":
        vt = H5_BASE / "selection" / "s_exact_validation_tiles.csv"
        if not vt.exists():
            raise FileNotFoundError(f"{vt} missing: run select_scenarios.py first, or pass --tiles")
        return pd.read_csv(vt)["tile_id"].astype(str).tolist()
    d = H5_BASE / scenario / "tree_dsm"
    return sorted(f.stem for f in d.glob("*.tif"))


def read_tiles_arg(spec: str | None, scenario: str) -> list[str]:
    if not spec or spec == "default":
        return default_tiles(scenario)
    if spec == "all":
        return default_tiles("P")
    p = Path(spec)
    if p.exists():
        if p.suffix == ".csv":
            return pd.read_csv(p)["tile_id"].astype(str).tolist()
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [t.strip() for t in spec.split(",") if t.strip()]


# ---------------------------------------------------------------------------------------------------------------------
def empirical_seconds() -> dict:
    """Per-tile timings of the main run (WP2.1 progress log): median seconds per tile-day, and the median gap between
    tiles (= staging + walls/aspect + SVF + cropping) derived from consecutive tile start/end stamps."""
    out = {"day_s": 47.0, "overhead_s": 70.0, "tile_wall_s": 161.0, "n_tiles_basis": 0, "source": "fallback"}
    if not WP21_PROGRESS.exists():
        return out
    d = pd.read_csv(WP21_PROGRESS)
    d = d[d["status"] == "OK"].copy()
    if d.empty:
        return out
    d["start"] = pd.to_datetime(d["start"], format="mixed")
    d["end"] = pd.to_datetime(d["end"], format="mixed")
    t = d.groupby("tile").agg(s=("start", "min"), e=("end", "max")).sort_values("s")
    wall = (t["s"].shift(-1) - t["s"]).dt.total_seconds().dropna()
    wall = wall[wall < 3600]                         # drop restarts / pauses
    day_s = float(d["seconds"].median())
    tw = float(wall.median()) if len(wall) else out["tile_wall_s"]
    return {"day_s": day_s, "overhead_s": max(0.0, tw - 2 * day_s), "tile_wall_s": tw, "n_tiles_basis": int(len(t)),
            "source": str(WP21_PROGRESS)}


def estimate_gpu_hours(n_tiles: int, n_days: int = 2, recompute_svf: bool = True) -> dict:
    e = empirical_seconds()
    per_tile = n_days * e["day_s"] + (e["overhead_s"] if recompute_svf else 0.15 * e["overhead_s"])
    return {"n_tiles": n_tiles, "seconds_per_tile": round(per_tile, 1), "gpu_hours": round(n_tiles * per_tile / 3600, 2),
            "utci_cpu_hours": round(n_tiles * 2 * 3.6 / 3600, 2), "basis": e}


# ---------------------------------------------------------------------------------------------------------------------
def compute_street_utci_arrays(tmrt_data: np.ndarray, invalid_mask: np.ndarray, ratio16_data: np.ndarray,
                               df_met: pd.DataFrame):
    """Street and uniform UTCI (16, 500, 500), line-for-line the per-hour block of compute_street_utci.process_tile."""
    from scripts.street.utci.compute_street_utci import NODATA_VALUE, interpolate_ratio_at_direction, utci
    utci_street = np.full((16, 500, 500), NODATA_VALUE, dtype=np.float32)
    utci_uniform = np.full((16, 500, 500), NODATA_VALUE, dtype=np.float32)
    for h_idx in range(16):
        row = df_met.iloc[h_idx]
        ta, rh, u10, wd = float(row["Td"]), float(row["RH"]), float(row["Wind"]), float(row["Wd"])
        tmrt_h = tmrt_data[h_idx]
        cell_valid = (~invalid_mask) & (tmrt_h != -9999.0) & (~np.isnan(tmrt_h)) & (~np.isnan(ratio16_data[0]))
        if not np.any(cell_valid):
            continue
        ratio_h = ratio16_data.mean(axis=0) if bool(row["vrb"]) else interpolate_ratio_at_direction(ratio16_data, wd)
        v10_eff = np.clip(u10 * ratio_h, 0.5, 17.0)
        v10_unif = float(np.clip(u10 * 1.0, 0.5, 17.0))
        tr_valid = tmrt_h[cell_valid]
        res_s = utci(tdb=ta, tr=tr_valid, v=v10_eff[cell_valid], rh=rh, limit_inputs=False, round_output=False)
        utci_street[h_idx, cell_valid] = res_s.utci.astype(np.float32)
        res_u = utci(tdb=ta, tr=tr_valid, v=np.full_like(tr_valid, v10_unif), rh=rh, limit_inputs=False,
                     round_output=False)
        utci_uniform[h_idx, cell_valid] = res_u.utci.astype(np.float32)
    return utci_street, utci_uniform


def run_scenario_utci_tile(tile_id: str, scenario: str, days=DAYS, h5_base: Path = H5_BASE, force: bool = False,
                           met_cache: dict | None = None) -> dict:
    from rasterio.transform import from_origin
    from scripts.street.utci.compute_street_utci import load_day_met, load_tile_masks, write_geotiff_16band
    t0 = time.time()
    p = scen_paths(scenario, tile_id, h5_base)
    ratio_p = RATIO16_DIR / f"{tile_id}.tif"
    if not ratio_p.exists():
        raise FileNotFoundError(f"missing wind ratio16: {ratio_p}")
    _, a, b = tile_id.split("_")
    transform = from_origin(float(a) * 1000.0, float(b) * 1000.0 + 1000.0, 2.0, 2.0)
    invalid_mask = load_tile_masks(tile_id)
    with rasterio.open(ratio_p) as fr:
        ratio16 = fr.read()
    names = [f"hour_{it:02d}_{it:02d}00LST" for it in range(6, 22)]
    met_cache = {} if met_cache is None else met_cache
    for day in days:
        if not force and all(validate_geotiff(p[f"{day}_{k}"], 16) and _fresh(p[f"{day}_{k}"], p[f"{day}_tmrt"])
                             for k in ("utci_street", "utci_uniform")):
            continue
        if day not in met_cache:
            met_cache[day] = load_day_met(day)
        with rasterio.open(p[f"{day}_tmrt"]) as s:
            tmrt = s.read()
        us, uu = compute_street_utci_arrays(tmrt, invalid_mask, ratio16, met_cache[day])
        for key, arr in ((f"{day}_utci_street", us), (f"{day}_utci_uniform", uu)):
            tmp = p[key].with_name(p[key].stem + ".tmp.tif")
            write_geotiff_16band(tmp, arr, transform, "EPSG:5253", names)
            os.replace(tmp, p[key])
    return {"tile_id": tile_id, "status": "OK", "seconds": round(time.time() - t0, 2)}


def run_scenario_solweig_tile(tile_id: str, scenario: str, days=DAYS, h5_base: Path = H5_BASE,
                              force: bool = False) -> dict:
    import torch
    import solweig_gpu
    from osgeo import gdal
    from scripts.street.solweig.run_solweig_citywide import (DESIGN_DAYS, crop_and_save_comfort_layers,
                                                             crop_and_save_svf, get_canonical_epsg5253_wkt)
    gdal.UseExceptions()
    torch.set_num_threads(4)
    p = scen_paths(scenario, tile_id, h5_base)
    if not p["tree"].exists():
        raise FileNotFoundError(f"missing scenario tree DSM {p['tree']} (run plant_trees.py --scenario {scenario})")
    out_base = Path(h5_base) / scenario / "solweig"
    scratch = Path(h5_base) / scenario / "scratch" / tile_id
    prep = scratch / "prep"
    wkt = get_canonical_epsg5253_wkt()
    t_all = time.time()
    if scratch.exists():
        shutil.rmtree(scratch)
    (prep / "metfiles").mkdir(parents=True, exist_ok=True)
    try:
        for src, name in ((TILES_BASE / "building_dsm" / f"{tile_id}.tif", "Building_DSM"),
                          (TILES_BASE / "dem" / f"{tile_id}.tif", "DEM"), (p["tree"], "Trees")):
            if not src.exists():
                raise FileNotFoundError(f"missing tile raster {src}")
            (prep / name).mkdir(parents=True, exist_ok=True)
            sds = gdal.Open(str(src))
            dds = gdal.GetDriverByName("GTiff").CreateCopy(str(prep / name / f"{name}_0_0.tif"), sds)
            dds.SetProjection(wkt)
            dds.FlushCache()
            dds = sds = None
        t0 = time.time()
        solweig_gpu.run_walls_aspect(str(prep))
        solweig_gpu.calculate_svf(str(prep), patch_option=2, overwrite=True)
        t_svf = round(time.time() - t0, 1)
        crop_and_save_svf(str(prep / "SVF/SkyViewFactor_0_0.tif"), str(p["svf"]), wkt)
        for d in days:
            p[f"{d}_svf"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p["svf"], p[f"{d}_svf"])
        day_s = {}
        for d in days:
            cfg = DESIGN_DAYS[d]
            met_src = PROJECT_ROOT / cfg["met_file"]
            work = scratch / f"work_{d}"
            work.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(met_src), str(prep / "metfiles/metfile_0_0.txt"))
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            solweig_gpu.run_utci_tiles(base_path=str(work), preprocess_dir=str(prep), selected_date_str=cfg["date"],
                                       save_tmrt=True, save_svf=False, save_shadow=True, save_ta=False,
                                       save_wind=False)
            day_s[d] = round(time.time() - t0, 1)
            crop_and_save_comfort_layers(work_output_dir=str(work / "output_folder"), tile_id=tile_id, day_name=d,
                                         output_base_dir=str(out_base), met_file_path=str(met_src))
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return {"tile_id": tile_id, "status": "OK", "seconds": round(time.time() - t_all, 1), "svf_s": t_svf,
            "day_seconds": day_s}


def _log_progress(scenario: str, rec: dict, h5_base: Path):
    f = Path(h5_base) / scenario / "run_progress.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([rec]).to_csv(f, mode="a", header=not f.exists(), index=False)


def validate_scenario(scenario: str, tiles: list[str], days=DAYS, h5_base: Path = H5_BASE) -> dict:
    done = [t for t in tiles if tile_complete(scenario, t, days, h5_base)]
    missing = [t for t in tiles if t not in set(done)]
    return {"scenario": scenario, "n_expected": len(tiles), "n_complete": len(done), "missing": missing,
            "complete": not missing}


def run_scenario_pipeline(scenario: str, tiles: list[str], days=DAYS, h5_base: Path = H5_BASE, force: bool = False,
                          dry_run: bool = False) -> dict:
    todo_sol = [t for t in tiles if force or not solweig_done(scenario, t, days, h5_base)]
    todo_utci = [t for t in tiles if force or not tile_complete(scenario, t, days, h5_base)]
    no_tree = [t for t in tiles if not scen_paths(scenario, t, h5_base)["tree"].exists()]
    no_ratio = [t for t in tiles if not (RATIO16_DIR / f"{t}.tif").exists()]
    est = estimate_gpu_hours(len(todo_sol), len(days))
    print(f"[{scenario}] tiles {len(tiles)}: SOLWEIG to run {len(todo_sol)}, UTCI to run {len(todo_utci)}, "
          f"tree DSM missing {len(no_tree)}, ratio16 missing {len(no_ratio)}")
    print(f"[{scenario}] GPU estimate: {len(todo_sol)} tiles x {est['seconds_per_tile']:.0f} s = "
          f"{est['gpu_hours']:.2f} GPU-h (basis: WP2.1 log, {est['basis']['n_tiles_basis']} tiles, median "
          f"{est['basis']['day_s']:.1f} s/tile-day + {est['basis']['overhead_s']:.0f} s staging/walls/SVF); "
          f"street UTCI ~{est['utci_cpu_hours']:.2f} CPU-h")
    if dry_run:
        print(f"[DRY-RUN {scenario}] no GPU used, nothing written.")
        return {"scenario": scenario, "dry_run": True, "estimate": est, "todo_solweig": len(todo_sol),
                "tree_missing": len(no_tree)}
    if no_tree:
        raise FileNotFoundError(f"[{scenario}] {len(no_tree)} tiles have no scenario tree DSM, e.g. {no_tree[:5]}")
    if no_ratio:
        raise FileNotFoundError(f"[{scenario}] {len(no_ratio)} tiles have no ratio16, e.g. {no_ratio[:5]}")
    met_cache = {}
    for i, tid in enumerate(tiles, 1):
        start = datetime.datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        try:
            rs = {"status": "SKIPPED"}
            if force or not solweig_done(scenario, tid, days, h5_base):
                rs = run_scenario_solweig_tile(tid, scenario, days, h5_base, force)
                if not solweig_done(scenario, tid, days, h5_base):
                    raise RuntimeError(f"SOLWEIG outputs for {tid} do not validate after the run")
            ru = run_scenario_utci_tile(tid, scenario, days, h5_base, force, met_cache)
            if not tile_complete(scenario, tid, days, h5_base):
                raise RuntimeError(f"outputs for {tid} do not validate after the run")
            _log_progress(scenario, {"tile_id": tid, "start": start, "seconds": round(time.time() - t0, 1),
                                     "solweig": rs["status"], "solweig_seconds": rs.get("seconds", 0.0),
                                     "utci_seconds": ru["seconds"], "status": "OK"}, h5_base)
            print(f"[{scenario}] {i}/{len(tiles)} {tid} OK ({time.time() - t0:.0f} s, SOLWEIG {rs['status']})",
                  flush=True)
        except Exception as e:
            _log_progress(scenario, {"tile_id": tid, "start": start, "seconds": round(time.time() - t0, 1),
                                     "status": f"ERROR: {e!r}"}, h5_base)
            print(f"[{scenario}] {tid} FAILED: {e!r}", flush=True)
            raise
    v = validate_scenario(scenario, tiles, days, h5_base)
    print(f"[{scenario}] validation: {v['n_complete']}/{v['n_expected']} tiles complete")
    if not v["complete"]:
        raise RuntimeError(f"[{scenario}] incomplete after the run: {v['missing'][:10]}")
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="H5 scenario SOLWEIG + street UTCI (A10).")
    ap.add_argument("--scenario", required=True, help="P or S_exact (or another planted scenario)")
    ap.add_argument("--tiles", default="default", help="'default' (P: 427 core tiles; S_exact: validation tiles), "
                                                       "'all', a comma list, or a .csv/.txt file")
    ap.add_argument("--days", default="IMBAT,CALM")
    ap.add_argument("--h5-base", default=str(H5_BASE))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no GPU: count work, check inputs, print estimate")
    ap.add_argument("--validate-only", action="store_true", help="report completeness; exit 1 if incomplete")
    ap.add_argument("--wait-pid", type=int, default=None, help="wait for this PID to exit before using the GPU")
    a = ap.parse_args()
    days = tuple(d.strip() for d in a.days.split(","))
    try:
        tiles = read_tiles_arg(a.tiles, a.scenario)
    except FileNotFoundError as e:
        if a.dry_run and a.scenario == "S_exact":
            print(f"[DRY-RUN S_exact] {e}; estimating for 40 validation tiles")
            est = estimate_gpu_hours(40, len(days))
            print(json.dumps(est, indent=1))
            sys.exit(0)
        raise
    if a.validate_only:
        v = validate_scenario(a.scenario, tiles, days, Path(a.h5_base))
        print(json.dumps({k: v[k] for k in ("scenario", "n_expected", "n_complete", "complete")}),
              "missing:", v["missing"][:20])
        sys.exit(0 if v["complete"] else 1)
    if a.wait_pid and not a.dry_run:
        import psutil
        while psutil.pid_exists(a.wait_pid):
            print(f"{datetime.datetime.now():%H:%M:%S} waiting for PID {a.wait_pid}", flush=True)
            time.sleep(60)
    run_scenario_pipeline(a.scenario, tiles, days, Path(a.h5_base), a.force, a.dry_run)
