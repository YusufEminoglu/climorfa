"""
CLIMORFA Street — Run Provisional SOLWEIG-GPU Simulations.
Executes SOLWEIG-GPU 2.0.0 on the existing WP0.4 1 m tile rasters
for the two November analogue days:
- 2026-11-12 (forcing: met_ANALOGUE_2025-11-02_for_2026-11-12_hourly.txt)
- 2026-11-13 (forcing: met_ANALOGUE_2025-11-03_for_2026-11-13_hourly.txt)

Writes runs under data/street/field/provisional/.
"""

import os
import sys
import time
import shutil
import gc
import torch
import solweig_gpu

PROJECT_ROOT = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data/street/benchmark")
FORCING_DIR = os.path.join(PROJECT_ROOT, "data/street/wsl/forcing_nov")
OUT_BASE = os.path.join(PROJECT_ROOT, "data/street/field/provisional")

SIMULATIONS = [
    {
        "date_str": "2026-11-12",
        "met_file": os.path.join(FORCING_DIR, "met_ANALOGUE_2025-11-02_for_2026-11-12_hourly.txt"),
        "base_path": os.path.join(OUT_BASE, "run_2026-11-12"),
    },
    {
        "date_str": "2026-11-13",
        "met_file": os.path.join(FORCING_DIR, "met_ANALOGUE_2025-11-03_for_2026-11-13_hourly.txt"),
        "base_path": os.path.join(OUT_BASE, "run_2026-11-13"),
    },
]


def run_provisional_simulation(sim_cfg, shared_prep_from=None):
    date_str = sim_cfg["date_str"]
    met_file = os.path.abspath(sim_cfg["met_file"])
    base_path = os.path.abspath(sim_cfg["base_path"])
    os.makedirs(base_path, exist_ok=True)

    bld_path = os.path.abspath(os.path.join(BENCHMARK_DIR, "Building_DSM_1m.tif"))
    dem_path = os.path.abspath(os.path.join(BENCHMARK_DIR, "DEM_1m.tif"))
    trees_path = os.path.abspath(os.path.join(BENCHMARK_DIR, "Trees_1m.tif"))

    print(f"\n=======================================================")
    print(f" Running Provisional SOLWEIG-GPU: {date_str}")
    print(f" Output: {base_path}")
    print(f" Metfile: {met_file}")
    print(f"=======================================================")

    t0_all = time.time()

    # Preprocessing
    print("\n--- [1/4] Preprocessing & Metfile Tiling ---")
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    prep_dir = solweig_gpu.preprocess(
        base_path=base_path,
        selected_date_str=date_str,
        building_dsm_filename=bld_path,
        dem_filename=dem_path,
        trees_filename=trees_path,
        tile_size=1000,
        overlap=0,
        use_own_met=True,
        own_met_file=met_file,
    )
    print(f"Preprocess completed in {time.time() - t0:.2f} s. Prep dir: {prep_dir}")

    # If shared SVF / walls / aspect can be reused from Day 1:
    if shared_prep_from and os.path.exists(shared_prep_from):
        print(f"\nReusing walls, aspect, and SVF from {shared_prep_from}...")
        for sub in ["walls", "aspect", "SVF"]:
            src_sub = os.path.join(shared_prep_from, sub)
            dst_sub = os.path.join(prep_dir, sub)
            if os.path.exists(src_sub):
                if os.path.exists(dst_sub):
                    shutil.rmtree(dst_sub)
                shutil.copytree(src_sub, dst_sub)
        print("Reused geometry precomputations successfully.")
    else:
        # Walls & Aspect
        print("\n--- [2/4] Walls & Aspect ---")
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        solweig_gpu.run_walls_aspect(prep_dir)
        print(f"Walls/Aspect completed in {time.time() - t0:.2f} s.")

        # SVF
        print("\n--- [3/4] Sky View Factor (SVF) ---")
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        solweig_gpu.calculate_svf(prep_dir, patch_option=2, overwrite=True)
        print(f"SVF completed in {time.time() - t0:.2f} s.")

    # Radiation / Tmrt / UTCI
    print("\n--- [4/4] Radiation, Tmrt & UTCI (24 hourly steps) ---")
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    solweig_gpu.run_utci_tiles(
        base_path=base_path,
        preprocess_dir=prep_dir,
        selected_date_str=date_str,
        save_tmrt=True,
        save_svf=True,
        save_shadow=True,
        save_ta=True,
        save_wind=True,
    )
    print(f"Radiation/Tmrt/UTCI completed in {time.time() - t0:.2f} s.")
    print(f"Total time for {date_str}: {time.time() - t0_all:.2f} s.")

    return prep_dir


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    day1_prep = None
    for i, sim in enumerate(SIMULATIONS):
        if i == 0:
            day1_prep = run_provisional_simulation(sim, shared_prep_from=None)
        else:
            run_provisional_simulation(sim, shared_prep_from=day1_prep)

    print("\n>>> All provisional SOLWEIG simulations completed successfully!")


if __name__ == "__main__":
    main()
