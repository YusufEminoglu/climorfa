"""WP2.3 — Street-level UTCI from SOLWEIG Tmrt + 02Geo3D street wind.

Agent: agy (Windows desktop, CPU only)
Conda env: climorfa-street

Method:
  Follows the 02Geo3D plugin exactly (web/src/analysis/runners.js ~line 748):
  UTCI(tile, hour) = pythermalcomfort utci(tdb = Ta(h), tr = Tmrt(tile, h), v = v10_eff, rh = RH(h))
  with:
    v10_eff = clip(U10(h) * ratio(tile, h), 0.5, 17.0)
    (the plugin passes the ratio-scaled 10 m-equivalent wind to UTCI)
  and for uniform wind reference (no-ventilation-structure):
    v10_uniform = clip(U10(h) * 1.0, 0.5, 17.0)

Inputs:
  - Tmrt: data/street/solweig/<DAY>/tmrt/<tile>.tif (16 bands, 06:00-21:00 LST)
  - ratio16: data/street/wind/ratio16/<tile>.tif (16 sectors, 22.5 deg intervals)
  - Forcing met files:
      data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt
      data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt
  - Building DSM & DEM: data/street/tiles/2m/building_dsm, dem (roofs: building_dsm - dem > 0.5 m)
  - Landmask: data/street/tiles/2m/landmask (sea cells = 0)

Outputs:
  data/street/utci/<DAY>/{utci_street,utci_uniform}/<tile>.tif (16 bands, float32, DEFLATE pred 3)
  VRTs for each layer.
  logs/wp/WP2.3_progress.csv
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import sys
import time
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

os.environ["GDAL_NUM_THREADS"] = "4"

from osgeo import gdal
gdal.UseExceptions()

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pythermalcomfort.models import utci
from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction
from scripts.street.wind.wind_screening import OPEN_PEDESTRIAN_FACTOR
from scripts.street.utci.vrb_utils import MET_V2, VRB_FLAGS_CSV, read_vrb_flags

# WP2.3b / pre-analysis lock A7 item 3 (VRB winds). For a day that has a corrected met file
# (met_DESIGN_<DAY>_<date>_hourly_v2.txt, written by build_vrb_flags_and_met_v2.py) the v2 file and
# data/street/forcing_summer/vrb_flags.csv are used. In a VRB-flagged hour the wind direction is undefined (Wd = -999 in
# v2) and the street-wind ratio is the MEAN OF THE 16 SECTOR RATIOS; U10 is unchanged. Hours that are not VRB-flagged and
# days without a v2 file (IMBAT: no VRB report) follow exactly the original code path (bit-identical output).

DAYS_CONFIG = {
    "IMBAT": {
        "met_file": PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt",
        "wind_subdirs": [
            "DESIGN_IMBAT_2022-07-05",
            "IMBAT",
        ],
    },
    "CALM": {
        "met_file": PROJECT_ROOT / "data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt",
        "wind_subdirs": [
            "DESIGN_CALM_2023-07-26",
            "CALM",
        ],
    },
}

NODATA_VALUE: float = -9999.0


def load_met_forcing(met_file_path: Path, vrb: dict[int, bool] | None = None) -> pd.DataFrame:
    """Load and filter met file for hours 06:00 to 21:00 LST (16 hours); adds boolean column `vrb` (rule A7)."""
    df = pd.read_csv(met_file_path, sep=r"\s+")
    df_sliced = df[(df["it"] >= 6) & (df["it"] <= 21)].copy().reset_index(drop=True)
    if len(df_sliced) != 16:
        raise ValueError(f"Expected 16 daylight hours (06-21) in {met_file_path}, got {len(df_sliced)}")
    df_sliced["vrb"] = [bool((vrb or {}).get(int(it), False)) for it in df_sliced["it"]]
    return df_sliced


def load_day_met(day: str) -> pd.DataFrame:
    """Met rows (it 6..21) + vrb flags for a design day. Uses the v2 met file when present; a VRB-flagged hour must have
    Wd = -999 there (guards against ever feeding the builder's spurious 0 deg into a VRB hour)."""
    v2 = MET_V2[day]
    if v2.exists():
        flags = read_vrb_flags(day)
        if not flags:
            raise RuntimeError(f"{v2.name} exists but {VRB_FLAGS_CSV.name} is missing/empty for {day}")
        df = load_met_forcing(v2, flags)
        bad = df[df["vrb"] & (df["Wd"] != -999)]
        bad2 = df[~df["vrb"] & (df["Wd"] < 0)]
        if len(bad) or len(bad2):
            raise RuntimeError(f"{v2.name}: Wd inconsistent with vrb flags for it={bad['it'].tolist()+bad2['it'].tolist()}")
        return df
    flags = read_vrb_flags(day)
    if any(flags.get(it, False) for it in range(6, 22)):
        raise RuntimeError(f"{day} has VRB-flagged hours in {VRB_FLAGS_CSV.name} but no v2 met file")
    return load_met_forcing(DAYS_CONFIG[day]["met_file"], flags)


def compute_vapour_pressure_kpa(tdb: float, rh: float) -> float:
    """Compute water vapour pressure in kPa from air temperature (C) and RH (%)."""
    g = [
        -2836.5744,
        -6028.076559,
        19.54263612,
        -0.02737830188,
        0.000016261698,
        7.0229056e-10,
        -1.8680009e-13,
    ]
    tk = tdb + 273.15
    es = 2.7150305 * np.log(tk)
    for count, coeff in enumerate(g):
        es = es + coeff * np.power(tk, count - 2)
    es = np.exp(es) * 0.01  # Pa to hPa
    eh_pa = es * (rh / 100.0)
    pa_kpa = eh_pa / 10.0  # hPa to kPa
    return float(pa_kpa)


def check_utci_validity(
    tdb: float,
    tr: np.ndarray,
    v: np.ndarray,
    rh: float,
) -> tuple[int, int, float]:
    """Check pythermalcomfort UTCI validity range and return violation counts.

    Valid ranges:
      -50 <= tdb <= 50 C
      -30 <= tr - tdb <= 70 C
      0.5 <= v <= 17.0 m/s
      0 <= rh <= 100 %
      0 <= pa <= 5.0 kPa

    Returns (total_cells, violated_cells, violation_share).
    """
    total = int(tr.size)
    if total == 0:
        return 0, 0, 0.0

    pa = compute_vapour_pressure_kpa(tdb, rh)

    cond_tdb = (-50.0 <= tdb <= 50.0)
    cond_rh = (0.0 <= rh <= 100.0)
    cond_pa = (0.0 <= pa <= 5.0)

    # Pixel-level conditions
    diff = tr - tdb
    cond_diff = (diff >= -30.0) & (diff <= 70.0)
    cond_v = (v >= 0.5) & (v <= 17.0)

    if not (cond_tdb and cond_rh and cond_pa):
        # Global met forcing violates validity
        return total, total, 1.0

    valid_mask = cond_diff & cond_v
    viol_count = int(np.sum(~valid_mask))
    viol_share = float(viol_count / total)
    return total, viol_count, viol_share


def load_tile_masks(tile_id: str) -> np.ndarray:
    """Load building DSM, DEM, and landmask for tile to compute invalid mask.

    Returns boolean mask (500, 500) where True = invalid (roof or sea or nodata).
    """
    b_path = PROJECT_ROOT / f"data/street/tiles/2m/building_dsm/{tile_id}.tif"
    d_path = PROJECT_ROOT / f"data/street/tiles/2m/dem/{tile_id}.tif"
    l_path = PROJECT_ROOT / f"data/street/tiles/2m/landmask/{tile_id}.tif"

    with rasterio.open(b_path) as fb, rasterio.open(d_path) as fd, rasterio.open(l_path) as fl:
        # Read inner 500x500 pixels (indices 50:550, 50:550)
        b = fb.read(1)[50:550, 50:550]
        d = fd.read(1)[50:550, 50:550]
        lm = fl.read(1)[50:550, 50:550]

    # 1. Roof mask: building_dsm - dem > 0.5 m
    h_bldg = b - d
    roof_mask = (h_bldg > 0.5) | (b == -9999.0) | (d == -9999.0)

    # 2. Sea mask: landmask == 0 or nodata (-9999.0)
    sea_mask = (lm < 0.5) | (lm == -9999.0)

    invalid_mask = roof_mask | sea_mask
    return invalid_mask


def write_geotiff_16band(
    out_path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: str = "EPSG:5253",
    band_names: list[str] | None = None,
):
    """Write 16-band float32 GeoTIFF with DEFLATE + Predictor 3, tiled 256x256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count, rows, cols = data.shape

    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA_VALUE,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
        if band_names:
            for b, name in enumerate(band_names, 1):
                dst.set_band_description(b, name)


def write_band_hour_table(out_dir: Path):
    """Write standardized band-hour table for UTCI layers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b, it in enumerate(range(6, 22), 1):
        utc = (it - 3) % 24
        rows.append({
            "band": b,
            "local_time_lst": f"{it:02d}:00",
            "utc_time": f"{utc:02d}:00",
            "utci_variable": "Universal Thermal Climate Index (UTCI)",
            "utci_units": "°C",
            "utci_dtype": "float32",
            "utci_street_model": "pythermalcomfort.models.utci with 02Geo3D screened wind (clip U10*ratio, 0.5, 17)",
            "utci_uniform_model": "pythermalcomfort.models.utci with uniform wind (clip U10, 0.5, 17)",
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "band_hour_table.csv", index=False)
    # Also write to outputs/street/utci
    out_outputs = PROJECT_ROOT / "outputs/street/utci"
    out_outputs.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_outputs / "band_hour_table.csv", index=False)


def build_vrts(utci_base_dir: Path, completed_tiles: list[str]):
    """Build GDAL VRT mosaics for each day and variant."""
    for day in ["IMBAT", "CALM"]:
        for variant in ["utci_street", "utci_uniform"]:
            var_dir = utci_base_dir / day / variant
            tifs = [str(var_dir / f"{tid}.tif") for tid in completed_tiles if (var_dir / f"{tid}.tif").exists()]
            vrt_path = utci_base_dir / day / f"{variant}.vrt"
            if tifs:
                vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
                gdal.BuildVRT(str(vrt_path), tifs, options=vrt_options)


def process_tile(
    tile_id: str,
    met_data: dict[str, pd.DataFrame],
    force: bool = False,
    force_day: str | None = None,
    out_base: Path | None = None,
) -> dict[str, any]:
    """Process UTCI (street and uniform) for both design days on a single tile.

    force_day (WP2.3b): recompute ONLY that day, and only for a tile that already has that day's outputs; the other
    day's files are not touched. out_base redirects the output root (default data/street/utci; used for tests).
    """
    t_start = time.perf_counter()
    iso_start = datetime.datetime.now().isoformat()

    utci_base = PROJECT_ROOT / "data/street/utci"
    days = (force_day,) if force_day else ("IMBAT", "CALM")
    out_root = Path(out_base) if out_base else utci_base
    parts = tile_id.split("_")
    x0 = float(parts[1]) * 1000.0
    y0 = float(parts[2]) * 1000.0
    transform = from_origin(x0, y0 + 1000.0, 2.0, 2.0)

    # 1. Verify existence of inputs
    ratio16_path = PROJECT_ROOT / f"data/street/wind/ratio16/{tile_id}.tif"
    if not ratio16_path.exists():
        return {
            "tile": tile_id,
            "status": "SKIPPED_MISSING_RATIO16",
            "seconds": 0.0,
        }

    for day in days:
        tmrt_path = PROJECT_ROOT / f"data/street/solweig/{day}/tmrt/{tile_id}.tif"
        if not tmrt_path.exists():
            return {
                "tile": tile_id,
                "status": f"SKIPPED_MISSING_SOLWEIG_{day}",
                "seconds": 0.0,
            }

    # 2. Check if already done
    all_done = True
    for day in days:
        p_street = utci_base / f"{day}/utci_street/{tile_id}.tif"
        p_uniform = utci_base / f"{day}/utci_uniform/{tile_id}.tif"
        if not (p_street.exists() and p_uniform.exists()):
            all_done = False
            break
    if force_day:
        if not all_done:
            return {"tile": tile_id, "status": f"SKIPPED_NO_EXISTING_{force_day}_OUTPUT", "seconds": 0.0}
    elif all_done and not force:
        return {
            "tile": tile_id,
            "status": "ALREADY_DONE",
            "seconds": 0.0,
        }

    # 3. Load masks and ratio16
    invalid_mask = load_tile_masks(tile_id)  # (500, 500)

    with rasterio.open(ratio16_path) as fr:
        ratio16_data = fr.read()  # (16, 500, 500)

    hourly_band_names = [f"hour_{it:02d}_{it:02d}00LST" for it in range(6, 22)]

    day_stats = {}

    for day in days:
        df_met = met_data[day]
        tmrt_path = PROJECT_ROOT / f"data/street/solweig/{day}/tmrt/{tile_id}.tif"
        with rasterio.open(tmrt_path) as ftmrt:
            tmrt_data = ftmrt.read()  # (16, 500, 500)

        utci_street = np.full((16, 500, 500), NODATA_VALUE, dtype=np.float32)
        utci_uniform = np.full((16, 500, 500), NODATA_VALUE, dtype=np.float32)

        day_total_valid_evals = 0
        day_viol_evals = 0
        street_valid_vals = []
        uniform_valid_vals = []

        for h_idx in range(16):
            row = df_met.iloc[h_idx]
            ta = float(row["Td"])
            rh = float(row["RH"])
            u10 = float(row["Wind"])
            wd = float(row["Wd"])

            tmrt_h = tmrt_data[h_idx]

            # Valid open ground cells: not roof/sea, tmrt is not nodata/nan, ratio is not nan
            cell_valid = (
                (~invalid_mask)
                & (tmrt_h != -9999.0)
                & (~np.isnan(tmrt_h))
                & (~np.isnan(ratio16_data[0]))
            )

            if not np.any(cell_valid):
                continue

            # Wind ratio for this hour. VRB hour (rule A7): direction undefined -> mean of the 16 sector ratios.
            if bool(row["vrb"]):
                ratio_h = ratio16_data.mean(axis=0)
            else:
                ratio_h = interpolate_ratio_at_direction(ratio16_data, wd)

            # Effective 10 m-equivalent wind for UTCI
            v10_eff = np.clip(u10 * ratio_h, 0.5, 17.0)
            v10_unif = float(np.clip(u10 * 1.0, 0.5, 17.0))

            tr_valid = tmrt_h[cell_valid]
            v_street_valid = v10_eff[cell_valid]

            # Check validity limits on outdoor cells
            tot_cells, viol_cells, _ = check_utci_validity(
                tdb=ta,
                tr=tr_valid,
                v=v_street_valid,
                rh=rh,
            )
            day_total_valid_evals += tot_cells
            day_viol_evals += viol_cells

            # Compute UTCI using pythermalcomfort
            res_street = utci(
                tdb=ta,
                tr=tr_valid,
                v=v_street_valid,
                rh=rh,
                limit_inputs=False,
                round_output=False,
            )
            utci_street[h_idx, cell_valid] = res_street.utci.astype(np.float32)
            street_valid_vals.append(res_street.utci)

            # Uniform wind reference
            res_uniform = utci(
                tdb=ta,
                tr=tr_valid,
                v=np.full_like(tr_valid, v10_unif),
                rh=rh,
                limit_inputs=False,
                round_output=False,
            )
            utci_uniform[h_idx, cell_valid] = res_uniform.utci.astype(np.float32)
            uniform_valid_vals.append(res_uniform.utci)

        # Write output GeoTIFFs
        p_street = out_root / f"{day}/utci_street/{tile_id}.tif"
        p_uniform = out_root / f"{day}/utci_uniform/{tile_id}.tif"

        write_geotiff_16band(
            out_path=p_street,
            data=utci_street,
            transform=transform,
            band_names=hourly_band_names,
        )
        write_geotiff_16band(
            out_path=p_uniform,
            data=utci_uniform,
            transform=transform,
            band_names=hourly_band_names,
        )

        mean_street = float(np.mean(np.concatenate(street_valid_vals))) if street_valid_vals else float("nan")
        mean_unif = float(np.mean(np.concatenate(uniform_valid_vals))) if uniform_valid_vals else float("nan")
        viol_share = float(day_viol_evals / day_total_valid_evals) if day_total_valid_evals > 0 else 0.0

        day_stats[day] = {
            "mean_street": mean_street,
            "mean_uniform": mean_unif,
            "viol_share": viol_share,
        }

    t_elapsed = round(time.perf_counter() - t_start, 2)
    iso_end = datetime.datetime.now().isoformat()

    return {
        "tile": tile_id,
        "start": iso_start,
        "end": iso_end,
        "seconds": t_elapsed,
        "status": "OK",
        "day_stats": day_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="WP2.3 Street-level UTCI computation")
    parser.add_argument("--tiles", nargs="+", default=None, help="Specific tile IDs to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tiles to process")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if done")
    parser.add_argument("--force-day", choices=["IMBAT", "CALM"], default=None,
                        help="WP2.3b: recompute only this day, only for tiles that already have its outputs "
                             "(other day untouched; no progress-CSV rows, no VRT rebuild). Log: outputs/street/utci/wp23b/")
    parser.add_argument("--out-dir", default=None,
                        help="redirect output root (default data/street/utci); for tests, e.g. outputs/street/utci/wp23b/...")
    args = parser.parse_args()

    print("=" * 80)
    print("WP2.3 — Street-level UTCI (SOLWEIG Tmrt + 02Geo3D Wind)")
    print("Environment: climorfa-street | CPU only | Resumable")
    print("=" * 80)

    # 1. Load met forcing for both days
    met_data = {}
    for day, cfg in DAYS_CONFIG.items():
        met_data[day] = load_day_met(day)
        src = MET_V2[day].name if MET_V2[day].exists() else cfg["met_file"].name
        vrb_it = met_data[day].loc[met_data[day]["vrb"], "it"].tolist()
        print(f"Loaded {len(met_data[day])} daylight hours for {day} ({src}); VRB-flagged hours: {vrb_it}")

    if args.out_dir and not args.force_day:
        parser.error("--out-dir is only allowed together with --force-day")

    # 2. Write band-hour documentation table (not in --force-day mode: nothing but the forced day's rasters is written)
    utci_base = PROJECT_ROOT / "data/street/utci"
    if not args.force_day:
        write_band_hour_table(utci_base)

    # 3. Determine target tiles
    tiles_gdf = gpd.read_file(PROJECT_ROOT / "data/street/grid/tiles_core.gpkg")
    all_tile_ids = sorted(tiles_gdf["tile_id"].tolist())

    if args.tiles:
        target_tiles = [t for t in args.tiles if t in set(all_tile_ids)]
    else:
        target_tiles = all_tile_ids

    if args.limit:
        target_tiles = target_tiles[:args.limit]

    print(f"Targeting {len(target_tiles)} tiles from core grid ({len(all_tile_ids)} total).")

    # 3b. WP2.3b force-recompute of one day (existing tiles only; other day, progress CSV and VRT list untouched)
    if args.force_day:
        out_base = Path(args.out_dir).resolve() if args.out_dir else None
        log_dir = PROJECT_ROOT / "outputs/street/utci/wp23b"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_rows = []
        for i, tid in enumerate(target_tiles, 1):
            res = process_tile(tid, met_data, force=True, force_day=args.force_day, out_base=out_base)
            msg = res["status"]
            if res["status"] == "OK":
                ds = res["day_stats"][args.force_day]
                msg += f" in {res['seconds']}s mean street {ds['mean_street']:.3f} unif {ds['mean_uniform']:.3f}"
                log_rows.append(dict(tile=tid, day=args.force_day, end=res["end"], seconds=res["seconds"],
                                     mean_utci_street=ds["mean_street"], mean_utci_uniform=ds["mean_uniform"],
                                     out_dir=str(out_base or utci_base)))
            if res["status"] == "OK" or args.tiles:
                print(f"[{i}/{len(target_tiles)}] {tid} {args.force_day}: {msg}", flush=True)
        if log_rows:
            tag = "default" if not out_base else out_base.name
            pd.DataFrame(log_rows).to_csv(log_dir / f"force_{args.force_day}_{tag}_log.csv", index=False)
        print(f"force-day {args.force_day}: recomputed {len(log_rows)} tiles")
        return

    # 4. Progress log path
    progress_csv = PROJECT_ROOT / "logs/wp/WP2.3_progress.csv"
    progress_rows = []
    if progress_csv.exists():
        df_prog = pd.read_csv(progress_csv)
        completed_set = set(df_prog[df_prog["status"] == "OK"]["tile"].unique())
    else:
        completed_set = set()

    # 5. Process tiles
    n_processed = 0
    n_skipped_missing = 0
    n_already_done = 0

    done_tiles_list = list(completed_set)

    for i, tid in enumerate(target_tiles, 1):
        res = process_tile(tid, met_data, force=args.force)
        st = res["status"]

        if st == "OK":
            n_processed += 1
            done_tiles_list.append(tid)
            print(f"[{i}/{len(target_tiles)}] {tid} SUCCESS in {res['seconds']}s | "
                  f"IMBAT street: {res['day_stats']['IMBAT']['mean_street']:.2f}°C, unif: {res['day_stats']['IMBAT']['mean_uniform']:.2f}°C | "
                  f"CALM street: {res['day_stats']['CALM']['mean_street']:.2f}°C, unif: {res['day_stats']['CALM']['mean_uniform']:.2f}°C")

            for day in ["IMBAT", "CALM"]:
                d_stat = res["day_stats"][day]
                record = {
                    "tile": tid,
                    "day": day,
                    "start": res["start"],
                    "end": res["end"],
                    "seconds": res["seconds"],
                    "mean_utci_street": round(d_stat["mean_street"], 2),
                    "mean_utci_uniform": round(d_stat["mean_uniform"], 2),
                    "viol_validity_share": round(d_stat["viol_share"], 6),
                    "status": "OK",
                }
                progress_rows.append(record)

                # Append to progress CSV
                pd.DataFrame([record]).to_csv(
                    progress_csv,
                    mode="a",
                    header=not progress_csv.exists() or os.path.getsize(progress_csv) == 0,
                    index=False,
                )

        elif st == "ALREADY_DONE":
            n_already_done += 1
            done_tiles_list.append(tid)
        else:
            n_skipped_missing += 1
            if args.tiles:
                print(f"[{i}/{len(target_tiles)}] {tid}: {st}")

    print("\n" + "=" * 80)
    print(f"Batch Summary: {n_processed} processed, {n_already_done} already done, {n_skipped_missing} skipped (missing inputs).")

    # Update VRTs for completed tiles
    unique_done = sorted(list(set(done_tiles_list)))
    if unique_done:
        print(f"Updating VRT virtual mosaics for {len(unique_done)} completed tiles...")
        build_vrts(utci_base, unique_done)
        print("VRTs updated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
