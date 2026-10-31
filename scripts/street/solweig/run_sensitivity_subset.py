"""
CLIMORFA Street — WP3.6 Step 3: GPU sensitivity runner (stratified subset of 40 tiles; lock §3, A7 item 6, A9)
==============================================================================================================
Variants (both design days, IMBAT 2022-07-05 and CALM 2023-07-26):
  radsens    RADSENS v2 forcing (Kdn, Kdiff, Kdir differ from the primary met files, nothing else), primary buildings
  h26 / h34  storey height 2.6 / 3.4 m, primary forcing
  landcover  SOLWEIG land-cover raster, primary forcing and geometry

Every variant follows the MAIN-RUN path, by calling the main-run functions rather than copies of them:
  SOLWEIG    staging, run_walls_aspect, calculate_svf(patch_option=2), run_utci_tiles, then
             run_solweig_citywide.crop_and_save_svf / crop_and_save_comfort_layers. Walls and the full SVF set are
             computed for EVERY variant: the main run keeps only the cropped SVF, which SOLWEIG-GPU cannot use as a
             cache, so "reusing" it saves nothing (SOLWEIG-GPU would recompute the SVF inside each day run).
  met files  h26, h34, landcover: the files of the main run (DESIGN_DAYS; SOLWEIG ignores Wd, lock A7 item 3).
             radsens: the _RADSENS_v2 files (A7 follow-up).
  UTCI       compute_street_utci.load_day_met / load_met_forcing + load_tile_masks, then
             run_scenario_solweig.compute_street_utci_arrays, which is line-for-line compute_street_utci.process_tile
             and reproduces the stored baseline utci_street exactly (see test_sensitivity_subset.py).
  metrics    extract_seg_metrics.extract_tile_day itself, with its raster roots switched to the variant.

Differences from the main run that remain on purpose:
  * radsens: _RADSENS_v2 met files (the sensitivity itself).
  * h26 / h34: rasterised on the fly from buildings_heights.gpkg (no stored tiles); with roof_z_primary this method
    reproduces the stored primary building_dsm (checked on all 40 tiles in test_sensitivity_subset.py).
  * landcover: Landcover raster staged (the sensitivity itself).
  * roof / sea masks and PS cells stay those of the primary building DSM (same footprints; PS definition unchanged).
  * tables carry partial_run = True, n_tiles_at_extraction = 40: they are a subset, never the 427-tile run.

Resumable: a tile/day is skipped when its outputs validate (--force recomputes). Safety lock: waits for the main run
and for the H5 GPU queue (PID files) unless --no-wait. --dry-run and --estimate never import torch or solweig_gpu.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import re
import shutil
import sys
import time
from pathlib import Path

# UTF-8 stdout configuration
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

DAYS = ("IMBAT", "CALM")
DATES = {"IMBAT": "2022-07-05", "CALM": "2023-07-26"}
FORC = PROJECT_ROOT / "data/street/forcing_summer"
SENS_DIR = PROJECT_ROOT / "data/street/sens"
ANALYSIS_SENS = PROJECT_ROOT / "data/street/analysis/sens"
SCRATCH_DIR = PROJECT_ROOT / "outputs/street/sens/_scratch"
TILES_2M = PROJECT_ROOT / "data/street/tiles/2m"
RATIO16_DIR = PROJECT_ROOT / "data/street/wind/ratio16"
PID_FILES = ("logs/wp/WP2.1_full_run.pid", "logs/wp/H5_gpu_queue.pid")
NODATA = -9999.0

# 40 Stratified Tiles across 11 districts and 5 dominant LCZ classes (Seed: 20261030)
SUBSET_40_TILES = [
    "T_0505_4250", "T_0504_4251", "T_0510_4259", "T_0514_4260", "T_0511_4261",
    "T_0517_4254", "T_0517_4261", "T_0518_4262", "T_0519_4262", "T_0527_4264",
    "T_0519_4255", "T_0520_4254", "T_0522_4255", "T_0523_4256", "T_0512_4247",
    "T_0516_4250", "T_0516_4251", "T_0519_4247", "T_0516_4248", "T_0522_4245",
    "T_0511_4243", "T_0509_4245", "T_0511_4244", "T_0511_4246", "T_0487_4242",
    "T_0489_4249", "T_0506_4250", "T_0509_4249", "T_0509_4246", "T_0508_4260",
    "T_0510_4257", "T_0513_4253", "T_0515_4253", "T_0513_4255", "T_0502_4250",
    "T_0507_4262", "T_0502_4260", "T_0502_4262", "T_0503_4262", "T_0501_4263",
]

# SOLWEIG met files. PRIMARY = exactly the files of the main run (run_solweig_citywide.DESIGN_DAYS).
PRIMARY_FORCING = {d: FORC / f"met_DESIGN_{d}_{DATES[d]}_hourly.txt" for d in DAYS}
RADSENS_FORCING = {d: FORC / f"met_DESIGN_{d}_{DATES[d]}_hourly_RADSENS_v2.txt" for d in DAYS}

GPU_VARIANTS = {
    "radsens": {"name": "Radiation Sensitivity v2", "forcing": RADSENS_FORCING, "building_col": None,
                "use_landcover": False, "desc": "RADSENS v2 forcing with primary building heights"},
    "h26": {"name": "Storey Height 2.6 m", "forcing": PRIMARY_FORCING, "building_col": "roof_z_h26",
            "use_landcover": False, "desc": "Lower-bound storey height with primary forcing"},
    "h34": {"name": "Storey Height 3.4 m", "forcing": PRIMARY_FORCING, "building_col": "roof_z_h34",
            "use_landcover": False, "desc": "Upper-bound storey height with primary forcing"},
    "landcover": {"name": "Land-Cover Sensitivity", "forcing": PRIMARY_FORCING, "building_col": None,
                  "use_landcover": True, "desc": "Land-cover raster with primary forcing and geometry"},
}


def read_pid_file(p: Path) -> int | None:
    """PID from a file in any of the encodings used here (ASCII, UTF-8 BOM, or UTF-16 from PowerShell Out-File)."""
    p = Path(p)
    if not p.exists():
        return None
    b = p.read_bytes()
    txt = b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8-sig", errors="ignore")
    m = re.search(r"\d+", txt)
    return int(m.group(0)) if m else None


def check_gpu_owners() -> list[tuple[str, int]]:
    """Active GPU jobs that must finish first: (pid file, pid), for python processes that are running."""
    active = []
    for f in PID_FILES:
        pid = read_pid_file(PROJECT_ROOT / f)
        try:
            if pid and psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
                active.append((f, pid))
        except psutil.Error:
            pass
    return active


def wait_for_gpu(poll_interval_s: int = 60) -> None:
    active = check_gpu_owners()
    if not active:
        print("[PID Check] No main run / H5 queue active. Proceeding.")
        return
    print(f"[SAFETY LOCK] GPU in use by {active}; waiting (Ctrl+C to cancel).", flush=True)
    while check_gpu_owners():
        time.sleep(poll_interval_s)
    print("[SAFETY LOCK] GPU is free.", flush=True)


def print_estimate():
    """GPU estimate from the WP2.1 progress log. Every variant runs staging + walls + SVF + 2 days per tile."""
    from scripts.street.h5.run_scenario_solweig import estimate_gpu_hours
    e = estimate_gpu_hours(len(SUBSET_40_TILES))
    b = e["basis"]
    print(f"Basis: {b['source']} ({b['n_tiles_basis']} tiles): median {b['day_s']:.1f} s per tile-day, "
          f"{b['overhead_s']:.1f} s per tile for staging + walls/aspect + SVF + cropping.")
    for v in GPU_VARIANTS:
        print(f"  {v:10s} {len(SUBSET_40_TILES)} tiles x {e['seconds_per_tile']:.0f} s = {e['gpu_hours']:.2f} GPU-h")
    print(f"  TOTAL      {e['gpu_hours'] * len(GPU_VARIANTS):.2f} GPU-h")


def rasterize_building_dsm(tile_id: str, col: str) -> np.ndarray:
    """Building DSM from buildings_heights.gpkg: roof elevation on footprint cells, DEM elsewhere.

    Two rules make this reproduce the stored primary building_dsm (test_sensitivity_subset.py, 40/40 tiles):
      * the background stays the DEM. Burning into a 0 background and taking max(dem, raster) lifts ground below 0 m to 0
        and invents building cells on low coastal tiles (T_0503_4262: 751 false cells, up to 1 m);
      * no building on sea-masked cells (landmask = 0). The stored DSM has none (T_0489_4249, T_0513_4255: 2,612 cells,
        including a 79.85 m tower footprint, would otherwise cast shadows the primary run does not have)."""
    _, a, b = tile_id.split("_")
    x0, y0 = int(a) * 1000, int(b) * 1000
    transform = from_origin(x0 - 100, y0 + 1100, 2.0, 2.0)
    with rasterio.open(TILES_2M / "dem" / f"{tile_id}.tif") as s:
        dem = s.read(1)
    with rasterio.open(TILES_2M / "landmask" / f"{tile_id}.tif") as s:
        land = s.read(1) >= 0.5
    bldg = gpd.read_file(PROJECT_ROOT / "data/street/buildings/buildings_heights.gpkg",
                         bbox=box(x0 - 100, y0 - 100, x0 + 1100, y0 + 1100), columns=[col])
    shapes = [(g, float(v)) for g, v in zip(bldg.geometry, bldg[col]) if np.isfinite(v)]
    r = rasterize(shapes, out_shape=(600, 600), transform=transform, fill=NODATA, dtype="float32", all_touched=False)
    roof = (r != NODATA) & (dem != NODATA) & land
    dsm = dem.copy()
    dsm[roof] = np.maximum(dem[roof], r[roof])
    return dsm


def ensure_building_dsm(tile_id: str, building_col: str | None, scratch_dir: Path) -> Path:
    """Stored primary DSM, a stored variant tile if one exists, else rasterised into scratch_dir."""
    if building_col is None:
        p = TILES_2M / "building_dsm" / f"{tile_id}.tif"
        if not p.exists():
            raise FileNotFoundError(f"Primary building DSM missing: {p}")
        return p
    disk = TILES_2M / f"building_dsm_{building_col.replace('roof_z_', '')}" / f"{tile_id}.tif"
    if disk.exists():
        return disk
    target = scratch_dir / f"building_dsm_{building_col}_{tile_id}.tif"
    with rasterio.open(TILES_2M / "dem" / f"{tile_id}.tif") as s:
        profile = s.profile.copy()
    profile.update(dtype="float32", nodata=NODATA, compress="deflate")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(rasterize_building_dsm(tile_id, building_col).astype(np.float32), 1)
    return target


def stage_tile_inputs(tile_id: str, bld_path: Path, prep_dir: Path, wkt_5253: str, lc_path: Path | None = None):
    """Stage rasters with canonical EPSG:5253 (run_solweig_citywide.stage_tile_inputs plus optional Landcover)."""
    from osgeo import gdal
    gdal.UseExceptions()
    layers = [(bld_path, "Building_DSM"), (TILES_2M / "dem" / f"{tile_id}.tif", "DEM"),
              (TILES_2M / "tree_dsm" / f"{tile_id}.tif", "Trees")]
    if lc_path is not None:
        layers.append((lc_path, "Landcover"))
    for src_path, name in layers:
        (prep_dir / name).mkdir(parents=True, exist_ok=True)
        src_ds = gdal.Open(str(src_path))
        dst_ds = gdal.GetDriverByName("GTiff").CreateCopy(str(prep_dir / name / f"{name}_0_0.tif"), src_ds)
        dst_ds.SetProjection(wkt_5253)
        dst_ds.FlushCache()
        dst_ds = src_ds = None


# ---------------------------------------------------------------------------------------------------------------------
def tile_paths(variant_dir: Path, tid: str, day: str) -> dict:
    return {"tmrt": variant_dir / "solweig" / day / "tmrt" / f"{tid}.tif",
            "utci": variant_dir / "solweig" / day / "utci" / f"{tid}.tif",
            "shadow": variant_dir / "solweig" / day / "shadow" / f"{tid}.tif",
            "svf": variant_dir / "solweig" / day / "svf" / f"{tid}.tif",
            "utci_street": variant_dir / "utci" / day / "utci_street" / f"{tid}.tif",
            "utci_uniform": variant_dir / "utci" / day / "utci_uniform" / f"{tid}.tif"}


def solweig_day_done(variant_dir: Path, tid: str, day: str) -> bool:
    from scripts.street.h5.run_scenario_solweig import validate_geotiff
    p = tile_paths(variant_dir, tid, day)
    return all(validate_geotiff(p[k], 16) for k in ("tmrt", "utci", "shadow")) and validate_geotiff(p["svf"], 1)


def utci_day_done(variant_dir: Path, tid: str, day: str) -> bool:
    from scripts.street.h5.run_scenario_solweig import validate_geotiff
    p = tile_paths(variant_dir, tid, day)
    if not p["tmrt"].exists():
        return False
    return all(validate_geotiff(p[k], 16) and p[k].stat().st_mtime >= p["tmrt"].stat().st_mtime
               for k in ("utci_street", "utci_uniform"))


def run_tile_solweig(variant: str, tid: str, days_to_run: list[str], variant_dir: Path, scratch_root: Path) -> dict:
    """Walls + SVF + the requested days for one tile, as run_solweig_citywide.process_single_tile does."""
    import torch
    import solweig_gpu
    from scripts.street.solweig.run_solweig_citywide import (crop_and_save_comfort_layers, crop_and_save_svf,
                                                             get_canonical_epsg5253_wkt)
    vcfg = GPU_VARIANTS[variant]
    wkt = get_canonical_epsg5253_wkt()
    scratch = scratch_root / tid
    prep = scratch / "prep"
    if scratch.exists():
        shutil.rmtree(scratch)
    (prep / "metfiles").mkdir(parents=True, exist_ok=True)
    out_base = variant_dir / "solweig"
    t_all = time.time()
    try:
        bld = ensure_building_dsm(tid, vcfg["building_col"], scratch)
        lc = None
        if vcfg["use_landcover"]:
            lc = SENS_DIR / "landcover" / "inputs" / f"{tid}.tif"
            if not lc.exists():
                raise FileNotFoundError(f"Missing landcover raster: {lc}")
        stage_tile_inputs(tid, bld, prep, wkt, lc)
        solweig_gpu.run_walls_aspect(str(prep))
        solweig_gpu.calculate_svf(str(prep), patch_option=2, overwrite=True)
        svf_out = out_base / "svf" / f"{tid}.tif"
        crop_and_save_svf(str(prep / "SVF" / "SkyViewFactor_0_0.tif"), str(svf_out), wkt)
        for d in DAYS:
            dp = tile_paths(variant_dir, tid, d)["svf"]
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(svf_out, dp)
        secs = {}
        for d in days_to_run:
            met = vcfg["forcing"][d]
            work = scratch / f"work_{d}"
            work.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(met), str(prep / "metfiles" / "metfile_0_0.txt"))
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            solweig_gpu.run_utci_tiles(base_path=str(work), preprocess_dir=str(prep), selected_date_str=DATES[d],
                                       save_tmrt=True, save_svf=False, save_shadow=True, save_ta=False, save_wind=False)
            secs[d] = round(time.time() - t0, 1)
            crop_and_save_comfort_layers(work_output_dir=str(work / "output_folder"), tile_id=tid, day_name=d,
                                         output_base_dir=str(out_base), met_file_path=str(met))
    finally:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()
    return {"seconds": round(time.time() - t_all, 1), "day_seconds": secs}


def variant_met(variant: str, day: str) -> pd.DataFrame:
    """Met rows it = 6..21 with vrb flags for the UTCI step: the main-run reader; radsens = the same with its own file."""
    from scripts.street.utci.compute_street_utci import load_day_met, load_met_forcing
    from scripts.street.utci.vrb_utils import read_vrb_flags
    ref = load_day_met(day)
    if GPU_VARIANTS[variant]["forcing"] is not RADSENS_FORCING:
        return ref
    df = load_met_forcing(RADSENS_FORCING[day], read_vrb_flags(day))
    for c in ("it", "Td", "RH", "Wind", "Wd", "vrb"):                  # radiation is the only permitted difference
        if not df[c].equals(ref[c]):
            raise RuntimeError(f"{RADSENS_FORCING[day].name}: column {c} differs from the primary met data")
    return df


def tile_utci_arrays(tid: str, tmrt_path: Path, df_met: pd.DataFrame):
    """Street and uniform UTCI of one tile/day from a Tmrt raster (nothing written; used by the parity test too)."""
    from scripts.street.h5.run_scenario_solweig import compute_street_utci_arrays
    from scripts.street.utci.compute_street_utci import load_tile_masks
    with rasterio.open(RATIO16_DIR / f"{tid}.tif") as s:
        ratio16 = s.read()
    with rasterio.open(tmrt_path) as s:
        tmrt = s.read()
    return compute_street_utci_arrays(tmrt, load_tile_masks(tid), ratio16, df_met)


def write_tile_utci(variant: str, tid: str, variant_dir: Path, met_cache: dict, days=DAYS, force: bool = False):
    from scripts.street.utci.compute_street_utci import write_geotiff_16band
    _, a, b = tid.split("_")
    transform = from_origin(float(a) * 1000.0, float(b) * 1000.0 + 1000.0, 2.0, 2.0)
    names = [f"hour_{it:02d}_{it:02d}00LST" for it in range(6, 22)]
    for d in days:
        if not force and utci_day_done(variant_dir, tid, d):
            continue
        if d not in met_cache:
            met_cache[d] = variant_met(variant, d)
        p = tile_paths(variant_dir, tid, d)
        us, uu = tile_utci_arrays(tid, p["tmrt"], met_cache[d])
        for key, arr in (("utci_street", us), ("utci_uniform", uu)):
            tmp = p[key].with_name(p[key].stem + ".tmp.tif")
            write_geotiff_16band(tmp, arr, transform, "EPSG:5253", names)
            os.replace(tmp, p[key])


def extract_variant_metrics(variant: str, tiles: list[str], variant_dir: Path, analysis_dir: Path | None) -> dict:
    """seg_metrics_<DAY>.parquet with extract_seg_metrics.extract_tile_day itself (raster roots switched)."""
    from scripts.street.h5.extract_h5_effects import BASE_DIRS, _job, esm
    dirs = {"solweig": variant_dir / "solweig", "utci": variant_dir / "utci", "ps": BASE_DIRS["ps"],
            "ratio16": BASE_DIRS["ratio16"]}
    syn = "SYNTHETIC" in str(variant_dir).upper()
    prefix = "SYNTHETIC_" if syn else ""
    n = {}
    for d in DAYS:
        parts = []
        for t in tiles:
            _, _, _, df = _job((t, d, "scen", dirs))
            if df is None:
                raise RuntimeError(f"[{variant}] metric extraction failed for {t} {d}")
            if len(df):
                parts.append(df)
        out = pd.concat(parts, ignore_index=True).sort_values("seg_id").reset_index(drop=True)
        out["partial_run"] = True                     # a 40-tile subset, never the full run
        out["n_tiles_at_extraction"] = np.int64(len(tiles))
        out["seg_id"] = out["seg_id"].astype(np.int64)
        out["n_ps_valid"] = out["n_ps_valid"].astype(np.int64)
        out = out[esm.SCHEMA_COLS]
        if syn:
            out = out.assign(is_synthetic=True)
        targets = [variant_dir / f"{prefix}seg_metrics_{d}.parquet"]
        if analysis_dir is not None:
            targets.append(analysis_dir / f"seg_metrics_{d}.parquet")
        for fp in targets:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(fp) + ".tmp"
            out.to_parquet(tmp, index=False)
            os.replace(tmp, fp)
        n[d] = int(len(out))
        print(f"[{variant}] {d}: {len(out)} segments", flush=True)
    return n


def validate_variant(variant: str, tiles: list[str], variant_dir: Path | None = None) -> dict:
    variant_dir = Path(variant_dir) if variant_dir else SENS_DIR / variant
    bad = [(t, d) for t in tiles for d in DAYS
           if not (solweig_day_done(variant_dir, t, d) and utci_day_done(variant_dir, t, d))]
    prefix = "SYNTHETIC_" if "SYNTHETIC" in str(variant_dir).upper() else ""
    tables = {d: (variant_dir / f"{prefix}seg_metrics_{d}.parquet").exists() for d in DAYS}
    return {"variant": variant, "n_expected": len(tiles) * 2, "n_complete": len(tiles) * 2 - len(bad),
            "tables": tables, "missing": bad[:10], "complete": not bad and all(tables.values())}


def run_variant(variant: str, tiles: list[str], force: bool = False, dry_run: bool = False,
                out_root: Path | None = None, scratch_root: Path | None = None):
    """Full pipeline of one variant; resumable per tile and day. out_root redirects outputs (tests)."""
    if variant not in GPU_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(GPU_VARIANTS)}")
    vcfg = GPU_VARIANTS[variant]
    variant_dir = Path(out_root or SENS_DIR) / variant
    analysis_dir = None if out_root else ANALYSIS_SENS / variant
    scratch_root = Path(scratch_root or SCRATCH_DIR) / variant
    print(f"\n{'=' * 80}\n  VARIANT '{variant}' ({vcfg['name']}): {vcfg['desc']}\n  Tiles: {len(tiles)}\n{'=' * 80}",
          flush=True)
    todo = {t: [d for d in DAYS if force or not solweig_day_done(variant_dir, t, d)] for t in tiles}
    n_sol = sum(1 for v in todo.values() if v)
    if dry_run:
        bad = []
        for t in tiles:
            need = [TILES_2M / k / f"{t}.tif" for k in ("dem", "tree_dsm", "building_dsm", "landmask")]
            need += [RATIO16_DIR / f"{t}.tif", PROJECT_ROOT / f"data/street/analysis/ps_index/{t}.tif"]
            if vcfg["use_landcover"]:
                need.append(SENS_DIR / "landcover" / "inputs" / f"{t}.tif")
            bad += [str(p) for p in need if not p.exists()]
        bad += [str(p) for p in vcfg["forcing"].values() if not p.exists()]
        if vcfg["building_col"]:
            import pyogrio
            flds = pyogrio.read_info(PROJECT_ROOT / "data/street/buildings/buildings_heights.gpkg")["fields"]
            if vcfg["building_col"] not in flds:
                bad.append(f"buildings_heights.gpkg lacks {vcfg['building_col']}")
        for d in DAYS:
            variant_met(variant, d)
        print(f"[DRY-RUN {variant}] tiles needing SOLWEIG: {n_sol}/{len(tiles)}; missing inputs: {len(bad)} {bad[:3]}; "
              f"met files {[p.name for p in vcfg['forcing'].values()]}. No GPU used, nothing written.", flush=True)
        if bad:
            raise FileNotFoundError(f"[{variant}] {len(bad)} missing inputs, e.g. {bad[:3]}")
        return

    log = variant_dir / "run_progress.csv"
    met_cache: dict = {}
    for i, tid in enumerate(tiles, 1):
        t0 = time.time()
        days = todo[tid]
        status = "OK"
        res: dict = {}
        try:
            if days:
                res = run_tile_solweig(variant, tid, days, variant_dir, scratch_root)
                bad = [d for d in DAYS if not solweig_day_done(variant_dir, tid, d)]
                if bad:
                    raise RuntimeError(f"SOLWEIG outputs of {tid} do not validate after the run: {bad}")
            write_tile_utci(variant, tid, variant_dir, met_cache, force=force)
        except Exception as e:
            status = f"ERROR: {e!r}"
            raise
        finally:
            log.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"tile": tid, "end": datetime.datetime.now().isoformat(timespec="seconds"),
                           "seconds": round(time.time() - t0, 1), "solweig_days": ",".join(days),
                           "status": status}]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"[{variant}] {i}/{len(tiles)} {tid} {'SOLWEIG ' + ','.join(days) if days else 'already done'} "
              f"({time.time() - t0:.0f} s)", flush=True)
    extract_variant_metrics(variant, tiles, variant_dir, analysis_dir)
    v = validate_variant(variant, tiles, variant_dir)
    if not v["complete"]:
        raise RuntimeError(f"[{variant}] incomplete after the run: {v}")
    print(f"[OK] Variant '{variant}' complete: {v['n_complete']}/{v['n_expected']} tile-days.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CLIMORFA Street — GPU Sensitivity Runner (40 Tiles)")
    parser.add_argument("--variant", choices=list(GPU_VARIANTS) + ["all"], default="all")
    parser.add_argument("--estimate", action="store_true", help="print the GPU-hour estimate and exit")
    parser.add_argument("--dry-run", action="store_true", help="check inputs and met data; no GPU, nothing written")
    parser.add_argument("--validate-only", action="store_true", help="report completeness; exit 1 if incomplete")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for the main run / H5 queue")
    parser.add_argument("--force", action="store_true", help="recompute completed tiles")
    parser.add_argument("--tiles", default=None, help="comma-separated tile IDs or a CSV with column tile_id")
    args = parser.parse_args()
    if args.estimate:
        print_estimate()
        return
    if args.tiles:
        tiles = (pd.read_csv(args.tiles)["tile_id"].tolist() if os.path.isfile(args.tiles)
                 else [t.strip() for t in args.tiles.split(",")])
    else:
        tiles = SUBSET_40_TILES
    variants = list(GPU_VARIANTS) if args.variant == "all" else [args.variant]
    if args.validate_only:
        vs = [validate_variant(v, tiles) for v in variants]
        for v in vs:
            print(v)
        sys.exit(0 if all(v["complete"] for v in vs) else 1)
    active = check_gpu_owners()
    print(f"[PID Check] active GPU jobs: {active}")
    if active and not args.dry_run and not args.no_wait:
        wait_for_gpu()
    for v in variants:
        run_variant(v, tiles, force=args.force, dry_run=args.dry_run)
    print("\nAll requested sensitivity variants finished.")


if __name__ == "__main__":
    main()
