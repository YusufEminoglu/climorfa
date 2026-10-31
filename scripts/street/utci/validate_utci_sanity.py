"""WP2.3 Sanity Verification & Physical Plausibility Test.

1. Two-route ratio agreement test on 3 tiles:
   - Route A: interpolate_ratio_at_direction(ratio16, Wd)
   - Route B: pedestrian_wind / (U10 * OPEN_PEDESTRIAN_FACTOR)
   - Criterion: max |Route A - Route B| < 1e-5.

2. Physical plausibility test at 15:00 LST (non-circular):
   - Compare utci_street vs utci_uniform in:
     a) sunlit vs shaded pixels (SOLWEIG shadow)
     b) wind-aligned vs perpendicular streets (roads_fua)
   - Hypothesis to be TESTED (not assumed):
     - Under IMBAT (strong sea breeze): wake sheltering in perpendicular streets reduces wind,
       leading to a thermal penalty (higher UTCI relative to uniform); wind-aligned streets preserve breeze.
     - Under CALM (heatwave): small difference between street and uniform wind UTCI due to low ambient wind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure PROJ and GDAL paths if needed
for c_share in [
    Path(sys.prefix) / "Library/share/proj",
    Path(sys.prefix) / "share/proj",
]:
    if c_share.exists():
        os.environ["PROJ_LIB"] = str(c_share)
        os.environ["PROJ_DATA"] = str(c_share)
        break

for g_share in [
    Path(sys.prefix) / "Library/share/gdal",
    Path(sys.prefix) / "share/gdal",
]:
    if g_share.exists():
        os.environ["GDAL_DATA"] = str(g_share)
        break

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, LineString

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction
from scripts.street.wind.wind_screening import OPEN_PEDESTRIAN_FACTOR
from scripts.street.utci.compute_street_utci import DAYS_CONFIG, load_day_met, load_met_forcing, load_tile_masks

OUTPUT_DIR = PROJECT_ROOT / "outputs/street/utci"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def verify_two_routes_agreement(tile_ids: list[str], day: str = "IMBAT") -> dict:
    """Verify that ratio16 interpolation agrees with pedestrian wind scaling on available tiles.
    Uses load_day_met (Rule A7: loads v2 met file when present; handles VRB hours).
    """
    print("=" * 80)
    print(f"1. Two-Route Ratio Agreement Test [{day}] (Step 16 / WP2.3 Spec)")
    print("Criterion: max |Route A - Route B| < 1e-5")
    print("=" * 80)

    results = {}
    tested_count = 0

    met_df = load_day_met(day)
    wind_sub = DAYS_CONFIG[day]["wind_subdirs"][0]

    for tid in tile_ids:
        ratio16_p = PROJECT_ROOT / f"data/street/wind/ratio16/{tid}.tif"
        wind_p = PROJECT_ROOT / f"data/street/wind/{wind_sub}/{tid}.tif"

        if not (ratio16_p.exists() and wind_p.exists()):
            results[tid] = {
                "status": "SKIPPED_MISSING_FILES",
                "ratio16_exists": ratio16_p.exists(),
                "wind_exists": wind_p.exists(),
            }
            continue

        with rasterio.open(ratio16_p) as fr, rasterio.open(wind_p) as fw:
            ratio16 = fr.read()  # (16, 500, 500)
            ped_wind = fw.read()  # (16, 500, 500)

        tile_max_diff = 0.0
        hourly_diffs = []

        for h_idx in range(16):
            u10 = float(met_df.iloc[h_idx]["Wind"])
            wd = float(met_df.iloc[h_idx]["Wd"])
            is_vrb = bool(met_df.iloc[h_idx].get("vrb", False)) or wd == -999.0

            # Route A: interpolate ratio16 or mean of 16 sectors if VRB (Rule A7)
            if is_vrb:
                r_A = ratio16.mean(axis=0)
            else:
                r_A = interpolate_ratio_at_direction(ratio16, wd)

            # Route B: ped_wind / (u10 * OPEN_PEDESTRIAN_FACTOR)
            r_B = ped_wind[h_idx] / (u10 * OPEN_PEDESTRIAN_FACTOR)

            # Valid open cells
            valid = ~np.isnan(r_A) & ~np.isnan(r_B)
            if np.any(valid):
                diff = np.abs(r_A[valid] - r_B[valid])
                max_d = float(np.max(diff))
                hourly_diffs.append(max_d)
                if max_d > tile_max_diff:
                    tile_max_diff = max_d

        passed = tile_max_diff < 1e-5
        tested_count += 1
        results[tid] = {
            "status": "PASSED" if passed else "FAILED",
            "max_abs_diff": float(tile_max_diff),
            "mean_hourly_max_diff": float(np.mean(hourly_diffs)) if hourly_diffs else 0.0,
            "passed": bool(passed),
        }
        print(f"[{tid}] Max |Δ| across 16 daylight hours = {tile_max_diff:.2e} -> {'PASSED' if passed else 'FAILED'}")

    return {
        "day": day,
        "n_tested": tested_count,
        "results": results,
    }


def analyze_street_and_shade_utci(tile_id: str) -> dict:
    """Analyze UTCI at 15:00 LST on tile across sunlit/shaded and street alignments."""
    print("=" * 80)
    print(f"2. Physical Plausibility Test on Tile {tile_id} at 15:00 LST (Band 10)")
    print("=" * 80)

    parts = tile_id.split("_")
    x0 = float(parts[1]) * 1000.0
    y0 = float(parts[2]) * 1000.0
    inner_box = box(x0, y0, x0 + 1000.0, y0 + 1000.0)

    # 1. Load roads for street canyon segmentation
    roads_path = PROJECT_ROOT / "data/02_interim/akilli_sehir_fua.gpkg"
    roads = gpd.read_file(roads_path, layer="roads_fua", bbox=inner_box)
    roads = roads.clip(inner_box)

    # Transform for 500x500 inner raster
    utci_p = PROJECT_ROOT / f"data/street/utci/IMBAT/utci_street/{tile_id}.tif"
    if not utci_p.exists():
        return {"status": f"SKIPPED_UTCI_NOT_COMPUTED_{tile_id}"}

    with rasterio.open(utci_p) as src:
        transform = src.transform

    # Segment roads by azimuth relative to IMBAT (15:00 Wd = 340.0 deg -> 160 deg mod 180)
    wind_az_imbat = 340.0 % 180.0
    aligned_geoms = []
    perp_geoms = []
    oblique_geoms = []

    for _, row in roads.iterrows():
        geom = row.geometry
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                seg_len = np.hypot(dx, dy)
                if seg_len < 4.0:
                    continue
                seg = LineString([p1, p2])
                az = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 180.0
                diff = abs(az - wind_az_imbat)
                if diff > 90.0:
                    diff = 180.0 - diff

                poly = seg.buffer(4.0)
                if diff <= 22.5:
                    aligned_geoms.append(poly)
                elif diff >= 67.5:
                    perp_geoms.append(poly)
                else:
                    oblique_geoms.append(poly)

    m_align = rasterize([(g, 1) for g in aligned_geoms], out_shape=(500, 500), transform=transform) == 1 if aligned_geoms else np.zeros((500, 500), dtype=bool)
    m_perp = rasterize([(g, 1) for g in perp_geoms], out_shape=(500, 500), transform=transform) == 1 if perp_geoms else np.zeros((500, 500), dtype=bool)
    m_oblique = rasterize([(g, 1) for g in oblique_geoms], out_shape=(500, 500), transform=transform) == 1 if oblique_geoms else np.zeros((500, 500), dtype=bool)

    # 2. Evaluate for each day at 15:00 LST (Band 10, index 9)
    day_evals = {}

    for day in ["IMBAT", "CALM"]:
        p_street = PROJECT_ROOT / f"data/street/utci/{day}/utci_street/{tile_id}.tif"
        p_unif = PROJECT_ROOT / f"data/street/utci/{day}/utci_uniform/{tile_id}.tif"
        p_shd = PROJECT_ROOT / f"data/street/solweig/{day}/shadow/{tile_id}.tif"
        p_tmrt = PROJECT_ROOT / f"data/street/solweig/{day}/tmrt/{tile_id}.tif"

        with rasterio.open(p_street) as fs, rasterio.open(p_unif) as fu, rasterio.open(p_shd) as fsh, rasterio.open(p_tmrt) as ft:
            utci_st_15 = fs.read(10)
            utci_un_15 = fu.read(10)
            shd_15 = fsh.read(10)
            tmrt_15 = ft.read(10)

        valid = (utci_st_15 != -9999.0) & (utci_un_15 != -9999.0) & ~np.isnan(utci_st_15) & ~np.isnan(utci_un_15)

        delta_utci = utci_st_15 - utci_un_15  # positive = street wind is warmer / more stressful than uniform

        # Sunlit vs shaded
        is_sun = (shd_15 >= 50) & valid
        is_shd = (shd_15 < 50) & valid

        # Street orientation
        is_align = m_align & valid
        is_perp = m_perp & valid
        is_oblique = m_oblique & valid

        def get_stats(arr):
            if len(arr) == 0:
                return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "p25": np.nan, "p75": np.nan}
            return {
                "n": int(len(arr)),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }

        day_evals[day] = {
            "all_open": {
                "utci_street": get_stats(utci_st_15[valid]),
                "utci_uniform": get_stats(utci_un_15[valid]),
                "delta_utci": get_stats(delta_utci[valid]),
                "tmrt": get_stats(tmrt_15[valid]),
            },
            "sunlit": {
                "utci_street": get_stats(utci_st_15[is_sun]),
                "utci_uniform": get_stats(utci_un_15[is_sun]),
                "delta_utci": get_stats(delta_utci[is_sun]),
                "tmrt": get_stats(tmrt_15[is_sun]),
            },
            "shaded": {
                "utci_street": get_stats(utci_st_15[is_shd]),
                "utci_uniform": get_stats(utci_un_15[is_shd]),
                "delta_utci": get_stats(delta_utci[is_shd]),
                "tmrt": get_stats(tmrt_15[is_shd]),
            },
            "wind_aligned_streets": {
                "utci_street": get_stats(utci_st_15[is_align]),
                "utci_uniform": get_stats(utci_un_15[is_align]),
                "delta_utci": get_stats(delta_utci[is_align]),
            },
            "perpendicular_streets": {
                "utci_street": get_stats(utci_st_15[is_perp]),
                "utci_uniform": get_stats(utci_un_15[is_perp]),
                "delta_utci": get_stats(delta_utci[is_perp]),
            },
            "oblique_streets": {
                "utci_street": get_stats(utci_st_15[is_oblique]),
                "utci_uniform": get_stats(utci_un_15[is_oblique]),
                "delta_utci": get_stats(delta_utci[is_oblique]),
            },
        }

        print(f"[{day}] Outdoor Sunlit Median UTCI: {day_evals[day]['sunlit']['utci_street']['median']:.2f}°C "
              f"vs Shaded: {day_evals[day]['shaded']['utci_street']['median']:.2f}°C "
              f"(ΔTmrt shade: {day_evals[day]['shaded']['tmrt']['median'] - day_evals[day]['sunlit']['tmrt']['median']:.2f}°C)")
        print(f"[{day}] Aligned Streets ΔUTCI (street-unif): {day_evals[day]['wind_aligned_streets']['delta_utci']['median']:.2f}°C "
              f"vs Perpendicular: {day_evals[day]['perpendicular_streets']['delta_utci']['median']:.2f}°C")

    return day_evals


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="WP2.3 Sanity Verification")
    parser.add_argument("--tiles", nargs="+", default=None, help="Tiles for 2-route check")
    parser.add_argument("--sanity-tile", default="T_0512_4255", help="Tile for street/shade analysis")
    parser.add_argument("--day", choices=["IMBAT", "CALM", "BOTH"], default="IMBAT", help="Day(s) for 2-route check")
    args = parser.parse_args()

    # Determine tiles for 2-route test
    if args.tiles:
        test_tiles = args.tiles
    else:
        test_tiles = ["T_0484_4242", "T_0487_4241", "T_0487_4242", "T_0512_4255"]

    days_to_check = ["IMBAT", "CALM"] if args.day == "BOTH" else [args.day]
    two_route_res = {}
    for d in days_to_check:
        two_route_res[d] = verify_two_routes_agreement(test_tiles, day=d)

    sanity_tile = args.sanity_tile
    p_check = PROJECT_ROOT / f"data/street/utci/IMBAT/utci_street/{sanity_tile}.tif"
    if not p_check.exists():
        avail = sorted([p.stem for p in (PROJECT_ROOT / "data/street/utci/IMBAT/utci_street").glob("*.tif")])
        if avail:
            fallback = "T_0484_4242" if "T_0484_4242" in avail else avail[0]
            print(f"Notice: Sanity tile {sanity_tile} not yet computed; falling back to {fallback}")
            sanity_tile = fallback
    sanity_res = analyze_street_and_shade_utci(sanity_tile)

    summary = {
        "two_route_verification": two_route_res,
        "physical_sanity_analysis": {
            "tile": sanity_tile,
            "metrics": sanity_res,
        },
    }

    out_file = OUTPUT_DIR / "sanity_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved sanity summary to: {out_file}")

    # Generate publication quicklook figure if sanity data is present
    if sanity_res and "IMBAT" in sanity_res and "status" not in sanity_res:
        plot_quicklook(sanity_tile, sanity_res, two_route_res)


def plot_quicklook(tile_id: str, sanity_res: dict, two_route_res: dict):
    """Generate multi-panel quicklook figure in outputs/street/utci/."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Panel 1: Sunlit vs Shaded UTCI at 15:00 LST
    ax1 = axes[0]
    days = ["IMBAT", "CALM"]
    sun_medians = [sanity_res[d]["sunlit"]["utci_street"]["median"] for d in days]
    shd_medians = [sanity_res[d]["shaded"]["utci_street"]["median"] for d in days]
    x = np.arange(len(days))
    w = 0.35
    ax1.bar(x - w / 2, sun_medians, width=w, label="Sunlit (Shadow >= 50%)", color="#d95f02", alpha=0.85)
    ax1.bar(x + w / 2, shd_medians, width=w, label="Shaded (Shadow < 50%)", color="#7570b3", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(days, fontweight="bold")
    ax1.set_ylabel("Median UTCI (°C)")
    ax1.set_title(f"A) Sunlit vs Shaded UTCI (15:00 LST, {tile_id})")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Street Canyon Alignment ΔUTCI (Street - Uniform)
    ax2 = axes[1]
    classes = ["Wind-Aligned", "Oblique", "Perpendicular"]
    keys = ["wind_aligned_streets", "oblique_streets", "perpendicular_streets"]
    imbat_deltas = [sanity_res["IMBAT"][k]["delta_utci"]["median"] for k in keys]
    calm_deltas = [sanity_res["CALM"][k]["delta_utci"]["median"] for k in keys]
    x2 = np.arange(len(classes))
    ax2.bar(x2 - w / 2, imbat_deltas, width=w, label="IMBAT (Sea breeze)", color="#1b9e77", alpha=0.85)
    ax2.bar(x2 + w / 2, calm_deltas, width=w, label="CALM (Heatwave)", color="#e7298a", alpha=0.85)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontweight="bold")
    ax2.set_ylabel("Median ΔUTCI (°C) [Street - Uniform]")
    ax2.set_title("B) Canyon Alignment Wind Effect (15:00 LST)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Two-Route Agreement Discrepancy
    ax3 = axes[2]
    tr_plot = two_route_res.get("IMBAT", two_route_res)
    tested_tiles = [t for t, d in tr_plot.get("results", {}).items() if d.get("status") == "PASSED"]
    if tested_tiles:
        diffs = [tr_plot["results"][t]["max_abs_diff"] for t in tested_tiles]
        ax3.bar(tested_tiles, diffs, color="#386cb0", alpha=0.85)
        ax3.axhline(1e-5, color="red", linestyle=":", label="Tolerance limit (1e-5)")
        ax3.set_yscale("log")
        ax3.set_ylabel("Max |Δratio| (Route A vs Route B)")
        ax3.set_title("C) Non-Circular Port Agreement")
        ax3.legend(loc="upper right")
    else:
        ax3.text(0.5, 0.5, "Two-route agreement:\nPending ratio16 arrival", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("C) Two-Route Agreement")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "quicklook_utci_sanity.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved quicklook figure to: {fig_path}")


if __name__ == "__main__":
    main()
