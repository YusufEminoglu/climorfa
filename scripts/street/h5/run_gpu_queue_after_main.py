"""CLIMORFA Street — GPU queue after the main SOLWEIG run (H5 A10 + GPU sensitivities). NOT started automatically.

Steps (stops on the first failure, every step logged with timestamps to logs/wp/H5_gpu_queue.log):
  0. preflight: inputs present; static check of run_sensitivity_subset.py for known crash defects (a blocker stops the
     queue BEFORE waiting, unless --skip-sens);
  1. wait for the main run PID (default 31796) to exit;
  2. verify 427/427 tiles have valid SOLWEIG tmrt + utci (16 bands) for IMBAT and CALM;
  3. GPU sensitivities: run_sensitivity_subset.py --variant radsens, h26, h34, landcover (each --no-wait), then check
     their 40-tile outputs;
  4. scenario P: plant_trees.py --scenario P (all 427 tiles), run_scenario_solweig.py --scenario P, --validate-only;
  5. wait until data/street/analysis/seg_metrics_{IMBAT,CALM}.parquet show n_tiles_at_extraction == 427 and
     partial_run False (if the WP3.0 re-runner is dead, extract_seg_metrics.py is invoked once per poll; timeout
     --extraction-timeout-h), then select_scenarios.py --require-complete (final selection), then plant + run S_exact on
     its 40 validation tiles and validate.
The queue does NOT extract H5 effects or evaluate H5; those are CPU steps the master runs after review:
  python -m scripts.street.h5.extract_h5_effects --scenario P ; ... --scenario S_exact ; python -m scripts.street.h5.evaluate_h5

--dry-run: runs every step's own dry-run / estimate (no GPU, no waiting, nothing written except the log) and prints
the GPU-hour estimate per step (empirical, from logs/wp/WP2.1_progress.csv).
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = sys.executable
LOG = PROJECT_ROOT / "logs/wp/H5_gpu_queue.log"
SENS_SCRIPT = PROJECT_ROOT / "scripts/street/solweig/run_sensitivity_subset.py"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts/street/analysis/extract_seg_metrics.py"
RERUNNER_PID = PROJECT_ROOT / "logs/wp/WP3.0_rerunner.pid"
SENS_VARIANTS = ("radsens", "h26", "h34", "landcover")
DAYS = ("IMBAT", "CALM")
N_TILES = 427
MAIN_PID = 31796


class QueueError(RuntimeError):
    pass


def log(msg: str):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], step: str) -> int:
    """Run a command from the project root, streaming its output (timestamped) into the log."""
    log(f"[{step}] $ {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        log(f"[{step}]   {line.rstrip()}")
    rc = p.wait()
    log(f"[{step}] exit code {rc} after {(time.time() - t0) / 60:.1f} min")
    return rc


def must(cmd: list[str], step: str):
    rc = run_cmd(cmd, step)
    if rc != 0:
        raise QueueError(f"step {step} failed: command exited with code {rc}: {' '.join(cmd)}")


def pid_alive(pid: int) -> bool:
    import psutil
    if not psutil.pid_exists(pid):
        return False
    try:
        return "python" in psutil.Process(pid).name().lower()      # guards against PID reuse by another program
    except Exception:
        return False


def pid_from_file(p: Path) -> int | None:
    if not p.exists():
        return None
    m = re.search(r"\d+", p.read_text(encoding="utf-8-sig", errors="ignore"))
    return int(m.group(0)) if m else None


# ---------------------------------------------------------------------------------------------------------------------
def preflight_sensitivity() -> tuple[list[str], list[str]]:
    """Static + functional checks of run_sensitivity_subset.py. Returns (blockers, warnings)."""
    import pandas as pd
    src = SENS_SCRIPT.read_text(encoding="utf-8")
    blockers, warnings = [], []
    # D1: column 'Ta' read from the met files, which have no 'Ta' column (air temperature is 'Td')
    met_cols = set(pd.read_csv(PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly_RADSENS_v2.txt",
                               sep=r"\s+", nrows=1).columns)
    if re.search(r"""r_met\[\s*["']Ta["']\s*\]""", src) and "Ta" not in met_cols:
        blockers.append("D1 compute_subset_utci_and_extract reads r_met['Ta'] but the met files have no 'Ta' column "
                        "(air temperature is 'Td'): KeyError in the UTCI step of every variant, after its SOLWEIG runs")
    # D2: pythermalcomfort utci returns a dataclass; assigning it into a float array raises
    from pythermalcomfort.models import utci
    import numpy as np
    returns_obj = hasattr(utci(tdb=30.0, tr=np.array([40.0]), v=np.array([1.0]), rh=40.0), "utci")
    if returns_obj and re.search(r"\]\s*=\s*calc_utci\(", src):
        blockers.append("D2 'utci_st[b, valid_mask] = calc_utci(...)' assigns the pythermalcomfort UTCI object (not "
                        ".utci) into a float array: ValueError in the UTCI step")
    # D3: radsens / landcover (reuse_primary_svf) never run walls/aspect -> solweig_gpu.run_utci_tiles lists prep/walls
    p_if = src.find('if vcfg["reuse_primary_svf"]')
    p_else = src.find("else:", p_if) if p_if >= 0 else -1
    walls = [m.start() for m in re.finditer(r"run_walls_aspect\(", src)]
    if p_if >= 0 and not any(w < p_else for w in walls):
        blockers.append("D3 for radsens and landcover (reuse_primary_svf) run_walls_aspect is never called, so "
                        "solweig_gpu.run_utci_tiles fails on the missing prep/walls folder (os.listdir) at the first "
                        "tile; and only the cropped SVF is copied, so SOLWEIG-GPU would recompute the full SVF inside "
                        "every day run anyway (no time saved)")
    if "validate_geotiff" not in src and "ALREADY" not in src:
        warnings.append("W1 run_variant is not resumable: a restart recomputes all 40 tiles of the variant")
    if "limit_inputs=False" not in src:
        warnings.append("W2 sensitivity UTCI calls omit limit_inputs=False / round_output=False (the main run uses "
                        "both): values outside the UTCI validity range become NaN and outputs are rounded to 0.1 degC")
    if "RADSENS_v2" not in src:
        blockers.append("radsens does not reference the _RADSENS_v2 met files (A7 follow-up)")
    for d, dt in (("IMBAT", "2022-07-05"), ("CALM", "2023-07-26")):
        f = PROJECT_ROOT / f"data/street/forcing_summer/met_DESIGN_{d}_{dt}_hourly_RADSENS_v2.txt"
        if not f.exists():
            blockers.append(f"missing {f.name}")
    from scripts.street.solweig.run_sensitivity_subset import SUBSET_40_TILES
    miss_lc = [t for t in SUBSET_40_TILES if not (PROJECT_ROOT / f"data/street/sens/landcover/inputs/{t}.tif").exists()]
    if miss_lc:
        blockers.append(f"landcover inputs missing for {len(miss_lc)} tiles")
    import pyogrio
    flds = set(pyogrio.read_info(PROJECT_ROOT / "data/street/buildings/buildings_heights.gpkg")["fields"])
    if not {"roof_z_h26", "roof_z_h34"} <= flds:
        blockers.append("buildings_heights.gpkg lacks roof_z_h26 / roof_z_h34")
    return blockers, warnings


def preflight(skip_sens: bool) -> list[str]:
    log("[0 preflight] checking inputs")
    blockers = []
    for p in (PROJECT_ROOT / "data/street/analysis/segments.gpkg", PROJECT_ROOT / "data/street/grid/tiles_core.gpkg",
              PROJECT_ROOT / "data/street/analysis/grid_250.gpkg", SENS_SCRIPT, EXTRACT_SCRIPT):
        if not p.exists():
            blockers.append(f"missing {p}")
    import geopandas as gpd
    tiles = sorted(gpd.read_file(PROJECT_ROOT / "data/street/grid/tiles_core.gpkg")["tile_id"])
    if len(tiles) != N_TILES:
        blockers.append(f"tiles_core has {len(tiles)} tiles, expected {N_TILES}")
    for sub in ("tree_dsm", "building_dsm", "dem", "landmask"):
        miss = [t for t in tiles if not (PROJECT_ROOT / f"data/street/tiles/2m/{sub}/{t}.tif").exists()]
        if miss:
            blockers.append(f"{len(miss)} tiles lack data/street/tiles/2m/{sub}")
    miss = [t for t in tiles if not (PROJECT_ROOT / f"data/street/wind/ratio16/{t}.tif").exists()]
    if miss:
        blockers.append(f"{len(miss)} tiles lack ratio16")
    miss = [t for t in tiles if not (PROJECT_ROOT / f"data/street/analysis/ps_index/{t}.tif").exists()]
    if miss:
        blockers.append(f"{len(miss)} tiles lack ps_index")
    sb, sw = preflight_sensitivity()
    for w in sw:
        log(f"[0 preflight] WARNING (sensitivity script): {w}")
    for b in sb:
        log(f"[0 preflight] {'SKIPPED (--skip-sens)' if skip_sens else 'BLOCKER'} (sensitivity script): {b}")
    if not skip_sens:
        blockers += [f"run_sensitivity_subset.py: {b}" for b in sb]
    for b in blockers:
        log(f"[0 preflight] BLOCKER: {b}")
    log(f"[0 preflight] {len(blockers)} blocker(s)")
    return blockers


def step1_wait(pid: int, poll_s: int, dry: bool):
    alive = pid_alive(pid)
    log(f"[1 wait] main run PID {pid} {'is running' if alive else 'is not running'}")
    if dry:
        return
    last = 0.0
    while pid_alive(pid):
        if time.time() - last > 1800:
            log(f"[1 wait] PID {pid} still running; polling every {poll_s} s")
            last = time.time()
        time.sleep(poll_s)
    log(f"[1 wait] PID {pid} has exited")


def step2_verify(dry: bool) -> dict:
    import geopandas as gpd
    from scripts.street.h5.run_scenario_solweig import validate_geotiff
    tiles = sorted(gpd.read_file(PROJECT_ROOT / "data/street/grid/tiles_core.gpkg")["tile_id"])
    res = {}
    for d in DAYS:
        for k in ("tmrt", "utci"):
            ok = [t for t in tiles if validate_geotiff(PROJECT_ROOT / f"data/street/solweig/{d}/{k}/{t}.tif", 16)]
            res[f"{d}/{k}"] = len(ok)
    log(f"[2 verify] valid main-run rasters: {res} (need {N_TILES} each)")
    if not dry and any(v != N_TILES for v in res.values()):
        raise QueueError(f"step 2 failed: main run incomplete {res}")
    return res


def sens_outputs_ok(v: str) -> tuple[bool, str]:
    import pandas as pd  # noqa: F401
    from scripts.street.solweig.run_sensitivity_subset import SUBSET_40_TILES
    from scripts.street.h5.run_scenario_solweig import validate_geotiff
    base = PROJECT_ROOT / f"data/street/sens/{v}"
    n = {d: sum(validate_geotiff(base / f"solweig/{d}/tmrt/{t}.tif", 16) for t in SUBSET_40_TILES) for d in DAYS}
    pq = {d: (base / f"seg_metrics_{d}.parquet").exists() for d in DAYS}
    good = all(x == len(SUBSET_40_TILES) for x in n.values()) and all(pq.values())
    return good, f"tmrt valid {n}, seg_metrics present {pq}"


def step3_sens(dry: bool, variants=SENS_VARIANTS):
    if dry:
        must([PY, str(SENS_SCRIPT), "--estimate"], "3 sens estimate")
        for v in variants:
            must([PY, str(SENS_SCRIPT), "--variant", v, "--dry-run"], f"3 sens {v} dry-run")
        return
    for v in variants:
        must([PY, str(SENS_SCRIPT), "--variant", v, "--no-wait"], f"3 sens {v}")
        good, msg = sens_outputs_ok(v)
        log(f"[3 sens {v}] outputs: {msg}")
        if not good:
            raise QueueError(f"step 3 failed: variant {v} outputs incomplete ({msg})")


def step4_P(dry: bool):
    if dry:
        must([PY, "-m", "scripts.street.h5.plant_trees", "--scenario", "P", "--dry-run"], "4 P plant dry-run")
        must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "P", "--dry-run"], "4 P run dry-run")
        return
    must([PY, "-m", "scripts.street.h5.plant_trees", "--scenario", "P"], "4 P plant")
    must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "P"], "4 P run")
    must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "P", "--validate-only"], "4 P validate")


def extraction_status() -> dict:
    import pyarrow.parquet as pq
    st = {}
    for d in DAYS:
        p = PROJECT_ROOT / f"data/street/analysis/seg_metrics_{d}.parquet"
        if not p.exists():
            st[d] = (0, True)
            continue
        t = pq.read_table(p, columns=["n_tiles_at_extraction", "partial_run"]).to_pandas()
        st[d] = (int(t["n_tiles_at_extraction"].max()), bool(t["partial_run"].any()))
    return st


def step5_wait_extraction(dry: bool, poll_s: int, timeout_h: float):
    st = extraction_status()
    log(f"[5 extraction] status {st} (need ({N_TILES}, False) on both days)")
    if dry:
        return
    t0 = time.time()
    while not all(n == N_TILES and not part for n, part in st.values()):
        if time.time() - t0 > timeout_h * 3600:
            raise QueueError(f"step 5 failed: segment extraction not complete after {timeout_h} h: {st}")
        rp = pid_from_file(RERUNNER_PID)
        if rp is None or not pid_alive(rp):
            log("[5 extraction] WP3.0 re-runner not alive: invoking extract_seg_metrics.py once")
            run_cmd([PY, str(EXTRACT_SCRIPT), "--workers", "2"], "5 extract")
        time.sleep(poll_s)
        st = extraction_status()
        log(f"[5 extraction] status {st}")
    log("[5 extraction] complete on both days")


def step5_select_and_S(dry: bool):
    if dry:
        must([PY, "-m", "scripts.street.h5.select_scenarios", "--dry-run"], "5 select dry-run")
        must([PY, "-m", "scripts.street.h5.plant_trees", "--scenario", "S_exact", "--dry-run"], "5 S plant dry-run")
        must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "S_exact", "--dry-run"],
             "5 S run dry-run")
        return
    must([PY, "-m", "scripts.street.h5.select_scenarios", "--require-complete"], "5 select")
    must([PY, "-m", "scripts.street.h5.plant_trees", "--scenario", "S_exact"], "5 S plant")
    must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "S_exact"], "5 S run")
    must([PY, "-m", "scripts.street.h5.run_scenario_solweig", "--scenario", "S_exact", "--validate-only"],
         "5 S validate")


def estimates(skip_sens: bool) -> list[tuple[str, float, str]]:
    from scripts.street.h5.run_scenario_solweig import empirical_seconds
    e = empirical_seconds()
    full = 2 * e["day_s"] + e["overhead_s"]
    reuse = 2 * (e["day_s"] + e["overhead_s"])  # no SVF cache in prep -> SVF recomputed inside each day run
    rows = []
    if not skip_sens:
        rows += [("3 sens radsens (40 tiles)", 40 * reuse / 3600, "SVF recomputed inside each day run (only the "
                  "cropped SVF is copied); assumes defect D3 is fixed"),
                 ("3 sens h26 (40 tiles)", 40 * full / 3600, "walls + SVF + 2 days"),
                 ("3 sens h34 (40 tiles)", 40 * full / 3600, "walls + SVF + 2 days"),
                 ("3 sens landcover (40 tiles)", 40 * reuse / 3600, "as radsens")]
    rows += [("4 P (427 tiles)", N_TILES * full / 3600, "walls + SVF recomputed with the P canopy + 2 days"),
             ("5 S_exact (40 tiles)", 40 * full / 3600, "walls + SVF + 2 days")]
    log(f"[estimate] basis: {e['source']} ({e['n_tiles_basis']} tiles): median {e['day_s']:.1f} s per tile-day, "
        f"{e['overhead_s']:.1f} s per tile staging + walls/aspect + SVF + cropping (tile wall {e['tile_wall_s']:.1f} s)")
    for name, h, why in rows:
        log(f"[estimate] {name:30s} {h:6.2f} GPU-h   ({why})")
    log(f"[estimate] TOTAL {'(without sensitivities) ' if skip_sens else ''}{sum(r[1] for r in rows):.2f} GPU-h; "
        f"CPU extras (street UTCI ~3.6 s/tile-day; planting P ~10-20 min) not included")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pid", type=int, default=MAIN_PID, help="main SOLWEIG run PID to wait for (default 31796)")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--extraction-poll-s", type=int, default=600)
    ap.add_argument("--extraction-timeout-h", type=float, default=48.0)
    ap.add_argument("--skip-sens", action="store_true", help="skip step 3 (and its preflight blockers)")
    a = ap.parse_args()
    log("=" * 100)
    log(f"H5 GPU queue {'DRY-RUN' if a.dry_run else 'START'} (python {PY}; pid {a.pid}; skip_sens {a.skip_sens})")
    try:
        blockers = preflight(a.skip_sens)
        if blockers and not a.dry_run:
            raise QueueError(f"preflight: {len(blockers)} blocker(s); nothing started. First: {blockers[0]}")
        estimates(a.skip_sens)
        step1_wait(a.pid, a.poll_s, a.dry_run)
        step2_verify(a.dry_run)
        if not a.skip_sens:
            step3_sens(a.dry_run)
        else:
            log("[3 sens] skipped (--skip-sens)")
        step4_P(a.dry_run)
        step5_wait_extraction(a.dry_run, a.extraction_poll_s, a.extraction_timeout_h)
        step5_select_and_S(a.dry_run)
        log(f"H5 GPU queue {'DRY-RUN finished' if a.dry_run else 'FINISHED: all steps OK'}"
            + ("" if a.dry_run else ". Next (CPU, master): extract_h5_effects --scenario P / S_exact, evaluate_h5"))
        if a.dry_run and blockers:
            log(f"DRY-RUN found {len(blockers)} blocker(s) that would stop a real launch at preflight")
    except QueueError as e:
        log(f"QUEUE STOPPED: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"QUEUE STOPPED (unexpected error): {e!r}")
        raise


if __name__ == "__main__":
    main()
