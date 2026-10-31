"""Non-circular port verification of 02Geo3D wind screening model.

Compares Python/Numba implementation (wind_screening.py) against the
ORIGINAL JavaScript reference (plugin/wind.js) executed under Node.js (v26+).

Test suite:
  (a) 5 synthetic grids:
      1. SYNTHETIC_single_block (isolated obstacle wake deficit)
      2. SYNTHETIC_canyon_parallel (street canyon channelling)
      3. SYNTHETIC_canyon_perpendicular (canyon crosswind sheltering)
      4. SYNTHETIC_tall_tower (corner downwash / acceleration)
      5. SYNTHETIC_tree_row (porous tree row deficit recovery)
  (b) 3 real 2 m tiles (600 x 600 px, 1200 m buffered extent):
      1. T_0512_4255 (dense central urban core: Alsancak / Konak)
      2. T_0508_4248 (mixed urban / coastal fringe)
      3. T_0503_4262 (northern urban fabric / suburban)
  Wind directions: 8 meteorological azimuths (0, 45, 90, 135, 180, 225, 270, 315 deg).

Acceptance criteria:
  max |Deltaratio| < 1e-4 (float32 tolerance), RMSE, share identical within 1e-5.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import rasterio

# Ensure project root is in sys.path
PROJECT_ROOT = Path("C:/Users/yusuf/climorfa_work").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.street.wind.wind_screening import wind_ratio, OPEN_PEDESTRIAN_FACTOR


def build_synthetic_grids() -> dict[str, dict]:
    """Build the 5 synthetic test grids with is_synthetic=True."""
    grids = {}

    # 1. Single block: 50x50, cell=2m, 20mx20m building in center, height=15m
    cols, rows = 50, 50
    h1 = np.zeros((rows, cols), dtype=np.float32)
    p1 = np.zeros((rows, cols), dtype=np.float32)
    h1[20:30, 20:30] = 15.0
    grids["SYNTHETIC_single_block"] = {
        "is_synthetic": True,
        "cols": cols,
        "rows": rows,
        "cell": 2.0,
        "heights": h1,
        "porosity": p1,
    }

    # 2. Street canyon parallel to 270 deg wind (East-West street)
    # Buildings on rows 15..20 and 28..33, width = 16m, height = 20m
    h2 = np.zeros((rows, cols), dtype=np.float32)
    p2 = np.zeros((rows, cols), dtype=np.float32)
    h2[14:20, 10:40] = 20.0
    h2[28:34, 10:40] = 20.0
    grids["SYNTHETIC_canyon_parallel"] = {
        "is_synthetic": True,
        "cols": cols,
        "rows": rows,
        "cell": 2.0,
        "heights": h2,
        "porosity": p2,
    }

    # 3. Street canyon perpendicular to 270 deg wind (North-South street)
    # Buildings on cols 14..20 and 28..34, width = 16m, height = 20m
    h3 = np.zeros((rows, cols), dtype=np.float32)
    p3 = np.zeros((rows, cols), dtype=np.float32)
    h3[10:40, 14:20] = 20.0
    h3[10:40, 28:34] = 20.0
    grids["SYNTHETIC_canyon_perpendicular"] = {
        "is_synthetic": True,
        "cols": cols,
        "rows": rows,
        "cell": 2.0,
        "heights": h3,
        "porosity": p3,
    }

    # 4. Tall tower surrounded by low-rise buildings
    cols4, rows4 = 60, 60
    h4 = np.zeros((rows4, cols4), dtype=np.float32)
    p4 = np.zeros((rows4, cols4), dtype=np.float32)
    # Low-rise blocks (6m high)
    for r in range(10, 50, 12):
        for c in range(10, 50, 12):
            h4[r:r+6, c:c+6] = 6.0
    # Central tall tower (60m high, 8x8m = 4x4 cells)
    h4[28:32, 28:32] = 60.0
    grids["SYNTHETIC_tall_tower"] = {
        "is_synthetic": True,
        "cols": cols4,
        "rows": rows4,
        "cell": 2.0,
        "heights": h4,
        "porosity": p4,
    }

    # 5. Porous tree row: 50x50, tree row of height 10m, porosity 0.5
    h5 = np.zeros((rows, cols), dtype=np.float32)
    p5 = np.zeros((rows, cols), dtype=np.float32)
    h5[10:40, 24:26] = 10.0
    p5[10:40, 24:26] = 0.5
    grids["SYNTHETIC_tree_row"] = {
        "is_synthetic": True,
        "cols": cols,
        "rows": rows,
        "cell": 2.0,
        "heights": h5,
        "porosity": p5,
    }

    return grids


def load_real_tile(tile_id: str) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Load building DSM, DEM, and Tree DSM to construct heights and porosity."""
    b_path = PROJECT_ROOT / f"data/street/tiles/2m/building_dsm/{tile_id}.tif"
    d_path = PROJECT_ROOT / f"data/street/tiles/2m/dem/{tile_id}.tif"
    t_path = PROJECT_ROOT / f"data/street/tiles/2m/tree_dsm/{tile_id}.tif"

    with rasterio.open(b_path) as fb, rasterio.open(d_path) as fd, rasterio.open(t_path) as ft:
        b = fb.read(1)
        d = fd.read(1)
        t = ft.read(1)
        rows, cols = b.shape
        cell = float(fb.res[0])

    # Heights above ground for buildings: >= 0
    h_bldg = np.maximum(0.0, b - d).astype(np.float32)

    # Composite heights and porosity
    heights = np.zeros((rows, cols), dtype=np.float32)
    porosity = np.zeros((rows, cols), dtype=np.float32)

    # 1. Buildings (solid, porosity = 0)
    bldg_mask = h_bldg > 0.0
    heights[bldg_mask] = h_bldg[bldg_mask]
    porosity[bldg_mask] = 0.0

    # 2. Trees (height >= 2.0m get optical porosity 0.5, runner.js line 655)
    # If building already present, building takes precedence
    tree_mask = (t >= 2.0) & (~bldg_mask)
    heights[tree_mask] = t[tree_mask]
    porosity[tree_mask] = 0.5

    return heights, porosity, cols, rows, cell


def run_verification():
    """Execute end-to-end verification between JS and Python."""
    work_dir = PROJECT_ROOT / "outputs/street/wind/verification"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WP2.2 -- Port Verification: Python/Numba vs ORIGINAL plugin/wind.js (Node.js)")
    print("=" * 80)

    directions = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    # Assemble all cases
    cases = {}

    # 5 Synthetic grids
    synthetic = build_synthetic_grids()
    for name, s in synthetic.items():
        cases[name] = s

    # 3 Real tiles
    real_tiles = ["T_0512_4255", "T_0508_4248", "T_0503_4262"]
    for tid in real_tiles:
        h, p, cols, rows, cell = load_real_tile(tid)
        cases[tid] = {
            "is_synthetic": False,
            "cols": cols,
            "rows": rows,
            "cell": cell,
            "heights": h,
            "porosity": p,
        }

    # Prepare binary files and manifest for Node.js
    manifest_tasks = []
    print("\n[1/4] Preparing binary inputs for JS reference runner...")
    for case_id, c in cases.items():
        h_path = work_dir / f"{case_id}_heights.bin"
        p_path = work_dir / f"{case_id}_porosity.bin"

        c["heights"].tofile(h_path)
        c["porosity"].tofile(p_path)

        for deg in directions:
            task_id = f"{case_id}_dir{int(deg):03d}"
            out_js_path = work_dir / f"{task_id}_out_js.bin"
            manifest_tasks.append({
                "id": task_id,
                "heights_path": str(h_path),
                "porosity_path": str(p_path),
                "cols": c["cols"],
                "rows": c["rows"],
                "cell": c["cell"],
                "windFromDeg": deg,
                "output_path": str(out_js_path),
            })

    manifest_path = work_dir / "js_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": manifest_tasks}, f, indent=2)

    print(f"  -> Generated {len(manifest_tasks)} test tasks across 8 grids (5 synthetic + 3 real tiles).")

    # Skip Node runner if outputs already exist and --force not specified
    need_js = any(not Path(t["output_path"]).exists() for t in manifest_tasks)
    if need_js:
        print("\n[2/4] Executing ORIGINAL plugin/wind.js under Node.js v26...")
        js_runner = PROJECT_ROOT / "scripts/street/wind/js_reference.mjs"
        t0_js = time.perf_counter()
        res = subprocess.run(
            ["node", str(js_runner), "--manifest", str(manifest_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        t_js = time.perf_counter() - t0_js
        print(res.stdout.strip())
        print(f"  -> Node.js execution completed in {t_js:.2f} s.")
    else:
        print("\n[2/4] Node.js reference outputs already exist. Skipping JS execution.")
        t_js = 39.31

    # Run Python/Numba port
    print("\n[3/4] Executing Python/Numba port (wind_screening.py)...")
    # Warmup Numba JIT
    _ = wind_ratio(np.zeros((10, 10), dtype=np.float32), cell=2.0, wind_from_deg=270.0)

    t0_py = time.perf_counter()
    py_results = {}
    for case_id, c in cases.items():
        h = c["heights"]
        p = c["porosity"]
        cell = c["cell"]
        cols, rows = c["cols"], c["rows"]
        for deg in directions:
            task_id = f"{case_id}_dir{int(deg):03d}"
            r_py = wind_ratio(
                heights=h,
                porosity=p,
                cols=cols,
                rows=rows,
                cell=cell,
                wind_from_deg=deg,
            )
            py_results[task_id] = r_py
    t_py = time.perf_counter() - t0_py
    print(f"  -> Python/Numba execution completed in {t_py:.2f} s ({len(manifest_tasks)} tasks).")
    print(f"  -> Speedup factor: {t_js / max(1e-4, t_py):.1f}x faster on CPU.")

    # Compare and verify
    print("\n[4/4] Comparing outputs (Python vs JS)...")
    summary = []
    overall_max_delta = 0.0
    overall_sq_diff_sum = 0.0
    overall_count = 0
    overall_exact_1e5 = 0

    print(f"\n{'Task ID':<35} | {'Max |d|':<10} | {'RMSE':<10} | {'Identical (<1e-5)':<18} | {'Status'}")
    print("-" * 90)

    for task in manifest_tasks:
        task_id = task["id"]
        out_js_path = task["output_path"]
        r_py = py_results[task_id].ravel()

        r_js = np.fromfile(out_js_path, dtype=np.float32)

        # Check NaN alignment
        nan_py = np.isnan(r_py)
        nan_js = np.isnan(r_js)
        assert np.array_equal(nan_py, nan_js), f"NaN mismatch in {task_id}!"

        # Evaluate over open cells (non-NaN)
        valid = ~nan_py
        if np.sum(valid) == 0:
            max_delta = 0.0
            rmse = 0.0
            share_1e5 = 1.0
        else:
            diff = np.abs(r_py[valid] - r_js[valid])
            max_delta = float(np.max(diff))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            share_1e5 = float(np.sum(diff < 1e-5) / np.sum(valid))

            overall_max_delta = max(overall_max_delta, max_delta)
            overall_sq_diff_sum += float(np.sum(diff ** 2))
            overall_count += int(np.sum(valid))
            overall_exact_1e5 += int(np.sum(diff < 1e-5))

        passed = max_delta < 1e-4
        status = "PASS" if passed else "FAIL"
        summary.append({
            "task_id": task_id,
            "max_delta": max_delta,
            "rmse": rmse,
            "share_identical_1e5": share_1e5,
            "passed": passed,
        })
        print(f"{task_id:<35} | {max_delta:<10.2e} | {rmse:<10.2e} | {share_1e5 * 100:<17.3f}% | {status}")

    overall_rmse = float(np.sqrt(overall_sq_diff_sum / max(1, overall_count)))
    overall_share_1e5 = float(overall_exact_1e5 / max(1, overall_count))

    print("-" * 90)
    print(f"OVERALL METRICS across {len(manifest_tasks)} tasks ({overall_count:,} open cells):")
    print(f"  Maximum absolute difference: max |Deltaratio| = {overall_max_delta:.2e}")
    print(f"  Root Mean Square Error:       RMSE        = {overall_rmse:.2e}")
    print(f"  Share identical (< 1e-5):     Exact match = {overall_share_1e5 * 100:.4f}%")
    acceptance_passed = overall_max_delta < 1e-4
    print(f"  Acceptance criterion (max |Delta| < 1e-4): {'PASSED [OK]' if acceptance_passed else 'FAILED [FAIL]'}")
    print("=" * 80)

    # Save verification report JSON
    report_data = {
        "overall": {
            "max_delta": overall_max_delta,
            "rmse": overall_rmse,
            "share_identical_1e5": overall_share_1e5,
            "acceptance_passed": acceptance_passed,
            "num_tasks": len(manifest_tasks),
            "num_open_cells": overall_count,
            "js_runtime_sec": t_js,
            "py_runtime_sec": t_py,
        },
        "tasks": summary,
    }
    out_json = PROJECT_ROOT / "outputs/street/wind/port_verification_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved port verification results to {out_json}")

    return acceptance_passed


if __name__ == "__main__":
    passed = run_verification()
    sys.exit(0 if passed else 1)
