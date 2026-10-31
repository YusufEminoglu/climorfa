"""
CLIMORFA Street — WP0.4 SOLWEIG-GPU Benchmark Runner
Runs SOLWEIG-GPU 2.0.0 thermal comfort simulations for central Alsancak benchmark tile:
1. 2 m resolution (350 x 350 pixels, 24 hourly steps)
2. 1 m resolution (700 x 700 pixels, 24 hourly steps)
3. Tile size vs GPU memory exploration on NVIDIA T1000 8GB

Records wall time per stage, peak GPU memory, tile sizes, and OOM boundaries.
"""

import os
import sys
import time
import json
import gc
import subprocess
import torch
import solweig_gpu

def get_nvidia_smi_memory():
    """Query current GPU memory usage via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True
        )
        parts = res.stdout.strip().split(",")
        return {"used_mb": float(parts[0]), "total_mb": float(parts[1])}
    except Exception:
        return {"used_mb": None, "total_mb": None}

def run_single_simulation(res_str, raster_res, tile_size):
    project_root = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
    data_dir = os.path.join(project_root, "data/street/benchmark")
    base_path = os.path.join(project_root, f"outputs/street/benchmark/run_{res_str}")
    os.makedirs(base_path, exist_ok=True)

    bld_path = os.path.abspath(os.path.join(data_dir, f"Building_DSM_{res_str}.tif"))
    dem_path = os.path.abspath(os.path.join(data_dir, f"DEM_{res_str}.tif"))
    trees_path = os.path.abspath(os.path.join(data_dir, f"Trees_{res_str}.tif"))
    met_path = os.path.abspath(os.path.join(data_dir, "met_SYNTHETIC_benchmark_2025-07-15.txt"))

    date_str = "2025-07-15"

    timings = {}
    memory_stats = {}

    print(f"\n=======================================================")
    print(f" Starting SOLWEIG-GPU Benchmark: {res_str} (tile_size={tile_size})")
    print(f"=======================================================")

    t_total_start = time.time()

    # Stage 1: Preprocess
    print("\n--- [Stage 1/4] Preprocessing & Tiling ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    prep_dir = solweig_gpu.preprocess(
        base_path=base_path,
        selected_date_str=date_str,
        building_dsm_filename=bld_path,
        dem_filename=dem_path,
        trees_filename=trees_path,
        tile_size=tile_size,
        overlap=0,
        use_own_met=True,
        own_met_file=met_path,
    )
    t_prep = time.time() - t0
    timings["preprocess_s"] = round(t_prep, 3)
    memory_stats["preprocess_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    memory_stats["preprocess_res_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
    print(f"Preprocess completed in {t_prep:.2f} s. Peak alloc: {memory_stats['preprocess_alloc_mb']} MB")

    # Stage 2: Walls and Aspect
    print("\n--- [Stage 2/4] Walls & Aspect Calculation ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    solweig_gpu.run_walls_aspect(prep_dir)
    t_walls = time.time() - t0
    timings["walls_aspect_s"] = round(t_walls, 3)
    memory_stats["walls_aspect_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    memory_stats["walls_aspect_res_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
    print(f"Walls/Aspect completed in {t_walls:.2f} s. Peak alloc: {memory_stats['walls_aspect_alloc_mb']} MB")

    # Stage 3: SVF Calculation
    print("\n--- [Stage 3/4] Sky View Factor (SVF) Calculation ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    solweig_gpu.calculate_svf(prep_dir, patch_option=2, overwrite=True)
    t_svf = time.time() - t0
    timings["svf_s"] = round(t_svf, 3)
    memory_stats["svf_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    memory_stats["svf_res_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
    smi_svf = get_nvidia_smi_memory()
    memory_stats["svf_smi_used_mb"] = smi_svf["used_mb"]
    print(f"SVF completed in {t_svf:.2f} s. Peak alloc: {memory_stats['svf_alloc_mb']} MB, Reserved: {memory_stats['svf_res_mb']} MB")

    # Stage 4: Hourly Radiation / Tmrt / UTCI
    print("\n--- [Stage 4/4] Hourly Radiation, Tmrt & UTCI (24 steps) ---")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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
    t_utci = time.time() - t0
    timings["radiation_tmrt_utci_s"] = round(t_utci, 3)
    memory_stats["utci_alloc_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    memory_stats["utci_res_mb"] = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2)
    smi_utci = get_nvidia_smi_memory()
    memory_stats["utci_smi_used_mb"] = smi_utci["used_mb"]
    print(f"UTCI/Tmrt completed in {t_utci:.2f} s. Peak alloc: {memory_stats['utci_alloc_mb']} MB, Reserved: {memory_stats['utci_res_mb']} MB")

    t_total = time.time() - t_total_start
    timings["total_wall_time_s"] = round(t_total, 3)
    timings["time_per_hourly_step_s"] = round(t_utci / 24.0, 4)

    print(f"\n>>> Total Wall Time ({res_str}): {t_total:.2f} s ({t_total/60.0:.2f} min)")

    return {
        "resolution": res_str,
        "pixel_size_m": raster_res,
        "raster_shape": [int(700 / raster_res), int(700 / raster_res)],
        "tile_size": tile_size,
        "timings": timings,
        "memory_stats": memory_stats,
    }

def explore_tile_size_memory():
    """Explore peak VRAM scaling with tile size on T1000 8GB."""
    print("\n=======================================================")
    print(" Exploring Tile Size vs VRAM Limits (NVIDIA T1000 8GB)")
    print("=======================================================")

    device = torch.device("cuda")
    sizes = [500, 700, 1000, 1200, 1400, 1600, 1800, 1900, 2000, 2100, 2200]
    results = []

    for sz in sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            # SVF 3D shadow matrices: 3 tensors of shape (sz, sz, 153)
            shmat = torch.zeros((sz, sz, 153), dtype=torch.float32, device=device)
            vegshmat = torch.zeros((sz, sz, 153), dtype=torch.float32, device=device)
            vbshvegshmat = torch.zeros((sz, sz, 153), dtype=torch.float32, device=device)
            # Intermediate 2D raytracing arrays (approx 12 tensors)
            scratch = [torch.zeros((sz, sz), dtype=torch.float32, device=device) for _ in range(12)]

            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
            smi = get_nvidia_smi_memory()

            fits_dedicated = peak_alloc <= 7600.0  # Safe dedicated VRAM boundary

            results.append({
                "tile_size": sz,
                "pixels": sz * sz,
                "status": "SUCCESS",
                "peak_alloc_mb": round(peak_alloc, 1),
                "peak_reserved_mb": round(peak_res, 1),
                "fits_dedicated_8gb": fits_dedicated,
                "note": "Fits dedicated VRAM" if fits_dedicated else "Approaching/using host RAM paging"
            })
            print(f"tile_size={sz:4d}x{sz:4d}: Peak Alloc={peak_alloc:6.1f} MB | Reserved={peak_res:6.1f} MB | Dedicated: {fits_dedicated}")
            del shmat, vegshmat, vbshvegshmat, scratch
        except RuntimeError as e:
            results.append({
                "tile_size": sz,
                "pixels": sz * sz,
                "status": "OOM",
                "error": str(e),
                "fits_dedicated_8gb": False
            })
            print(f"tile_size={sz:4d}x{sz:4d}: OOM ({e})")
            break

    return results

def main():
    project_root = r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
    out_dir = os.path.join(project_root, "outputs/street/benchmark")
    os.makedirs(out_dir, exist_ok=True)

    gpu_info = {
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "solweig_gpu_version": solweig_gpu.__version__,
    }
    print("Environment:", gpu_info)

    # Run 1: 2 m resolution (350 x 350)
    metrics_2m = run_single_simulation(res_str="2m", raster_res=2.0, tile_size=500)

    # Run 2: 1 m resolution (700 x 700)
    metrics_1m = run_single_simulation(res_str="1m", raster_res=1.0, tile_size=1000)

    # Exploration: Tile size vs VRAM limits
    tile_scaling = explore_tile_size_memory()

    all_benchmark_data = {
        "hardware": gpu_info,
        "run_2m": metrics_2m,
        "run_1m": metrics_1m,
        "tile_scaling_exploration": tile_scaling,
    }

    out_json = os.path.join(out_dir, "benchmark_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_data, f, indent=2)
    print(f"\n>>> Saved all benchmark metrics to: {out_json}")

if __name__ == "__main__":
    main()
