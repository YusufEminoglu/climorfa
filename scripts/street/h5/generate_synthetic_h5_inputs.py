"""CLIMORFA Street — WP3.5 Synthetic Data Generator for H5 Testing.

Generates schema-compliant SYNTHETIC segments, seg_metrics_IMBAT, and grid_250
when WP3.0/WP3.0c tables are not yet finalized, per Rule 6 and pre_analysis_lock.md.

Rule 6:
  - is_synthetic = True in DataFrame attributes and columns.
  - SYNTHETIC in all output filenames.
  - Explicitly marked as non-observed placeholder data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, LineString, MultiLineString

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
OUTPUT_DIR = PROJECT_ROOT / "data/street/h5"


def generate_synthetic_inputs(
    target_tiles: list[str] | None = None,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 20261030,
) -> dict[str, Path]:
    """Generate synthetic segment, metric, and grid tables for testing H5 machinery."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target_tiles is None:
        target_tiles = ["T_0512_4255", "T_0484_4242", "T_0487_4241"]

    # Load core tiles
    tiles_gpkg = PROJECT_ROOT / "data/street/grid/tiles_core.gpkg"
    gdf_tiles = gpd.read_file(tiles_gpkg)
    gdf_sel_tiles = gdf_tiles[gdf_tiles["tile_id"].isin(target_tiles)].copy()

    # Load roads_fua
    roads_gpkg = PROJECT_ROOT / "data/02_interim/akilli_sehir_fua.gpkg"
    gdf_roads = gpd.read_file(roads_gpkg, layer="roads_fua")

    # Load reach features for realistic reach values
    reach_csv = PROJECT_ROOT / "data/03_processed/grid_250m_road_network_reach_features.csv"
    df_reach = pd.read_csv(reach_csv)
    reach_col = "network_reach_400m_road_length_m"

    # Intersect roads with target tiles
    segments_list = []
    seg_id_counter = 100001

    for _, tile_row in gdf_sel_tiles.iterrows():
        tid = tile_row["tile_id"]
        tile_geom = tile_row["geometry"]  # inner 1000m polygon
        roads_in_tile = gdf_roads[gdf_roads.intersects(tile_geom)].copy()

        for _, road_row in roads_in_tile.iterrows():
            clipped = road_row["geometry"].intersection(tile_geom)
            if clipped.is_empty:
                continue

            lines = []
            if isinstance(clipped, LineString):
                lines = [clipped]
            elif isinstance(clipped, MultiLineString):
                lines = list(clipped.geoms)

            for line in lines:
                length_m = float(line.length)
                if length_m < 20.0:
                    continue

                # Azimuth
                coords = list(line.coords)
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                azimuth = (math.degrees(math.atan2(dx, dy)) + 360.0) % 180.0

                # Grid cell IDs per schema: g<L>_id = f"{int(x0//L)}_{int(y0//L)}"
                cx, cy = coords[0]
                g050_id = f"{int(cx // 50)}_{int(cy // 50)}"
                g100_id = f"{int(cx // 100)}_{int(cy // 100)}"
                g250_id = f"{int(cx // 250)}_{int(cy // 250)}"
                g500_id = f"{int(cx // 500)}_{int(cy // 500)}"

                # Assign realistic LCZ class (2: Compact mid-rise, 3: Compact low-rise, 5: Open mid-rise, 8: Large low-rise)
                lcz_class = int(rng.choice([2, 3, 5, 8], p=[0.4, 0.3, 0.2, 0.1]))

                n_ps = max(30, int(length_m * rng.uniform(1.5, 3.5)))

                segments_list.append({
                    "seg_id": seg_id_counter,
                    "tile_id": tid,
                    "district": "Konak" if tid == "T_0512_4255" else "Core",
                    "lcz_primary": lcz_class,
                    "lcz_weak250": lcz_class,
                    "length_m": round(length_m, 2),
                    "n_ps": n_ps,
                    "azimuth_deg": round(azimuth, 1),
                    "hw_ratio": round(float(rng.uniform(0.5, 2.8)), 2),
                    "bh_mean30": round(float(rng.uniform(6.0, 24.0)), 1),
                    "bcover30": round(float(rng.uniform(0.2, 0.65)), 3),
                    "tcover15": round(float(rng.uniform(0.02, 0.30)), 3),
                    "tht15": round(float(rng.uniform(3.0, 9.0)), 1),
                    "dist_coast_m": round(float(rng.uniform(100.0, 3500.0)), 1),
                    "elev_m": round(float(rng.uniform(2.0, 65.0)), 1),
                    "slope_deg": round(float(rng.uniform(0.5, 12.0)), 1),
                    "lst_med": round(float(rng.uniform(36.0, 44.0)), 2),
                    "g050_id": g050_id,
                    "g100_id": g100_id,
                    "g250_id": g250_id,
                    "g500_id": g500_id,
                    "is_synthetic": True,
                    "geometry": line,
                })
                seg_id_counter += 1

    gdf_segments = gpd.GeoDataFrame(segments_list, crs="EPSG:5253")

    # Generate matching seg_metrics_IMBAT
    metrics_list = []
    for _, seg in gdf_segments.iterrows():
        sid = seg["seg_id"]
        lcz = seg["lcz_primary"]

        # LCZ 2 and 3 have higher thermal stress on average
        base_stress = 6 if lcz in [2, 3] else 4
        m2_strong = int(np.clip(rng.poisson(base_stress), 0, 9))
        m2_vstrong = int(np.clip(rng.poisson(max(0, m2_strong - 3)), 0, 9))

        utci_med = round(float(32.0 + m2_strong * 1.2 + rng.normal(0, 0.5)), 2)

        row_metric = {
            "seg_id": sid,
            "n_ps_valid": seg["n_ps"],
            "utci_b15": utci_med,
            "utci_unif_b15": round(utci_med + float(rng.uniform(-0.5, 1.2)), 2),
            "tmrt_b15": round(float(52.0 + m2_strong * 1.8 + rng.normal(0, 1.0)), 2),
            "tmrt_b15_p90": round(float(58.0 + m2_strong * 2.0), 2),
            "v10eff_b15": round(float(rng.uniform(0.8, 2.5)), 2),
            "ratio_med_b15": round(float(rng.uniform(0.2, 0.7)), 2),
            "svf_med": round(float(rng.uniform(0.25, 0.75)), 3),
            "shade_frac_10_18": round(float(rng.uniform(0.15, 0.55)), 3),
            "m2_strong": m2_strong,
            "m2_vstrong": m2_vstrong,
            "partial_run": True,
            "n_tiles_at_extraction": len(target_tiles),
            "is_synthetic": True,
        }
        for b in range(6, 22):
            if f"utci_b{b:02d}" not in row_metric:
                row_metric[f"utci_b{b:02d}"] = round(float(26.0 + (b - 6) * 1.0 + rng.normal(0, 0.4)), 2)
        metrics_list.append(row_metric)

    df_metrics = pd.DataFrame(metrics_list)

    # Generate matching grid_250.gpkg
    unique_g250 = gdf_segments["g250_id"].unique()
    grid_list = []
    for g250 in unique_g250:
        parts = g250.split("_")
        gx = int(parts[0]) * 250
        gy = int(parts[1]) * 250
        cell_geom = box(gx, gy, gx + 250, gy + 250)

        # Match reach from reach csv or sample
        reach_val = float(df_reach[reach_col].sample(1, random_state=int(parts[0]) % 1000).values[0])
        pop_val = float(rng.uniform(80.0, 750.0))

        grid_list.append({
            "g250_id": g250,
            "tile_id": gdf_segments[gdf_segments["g250_id"] == g250]["tile_id"].iloc[0],
            "lcz_primary_major": int(gdf_segments[gdf_segments["g250_id"] == g250]["lcz_primary"].mode().iloc[0]),
            "lcz_weak250": int(gdf_segments[gdf_segments["g250_id"] == g250]["lcz_weak250"].mode().iloc[0]),
            "lst_med": round(float(rng.uniform(37.0, 43.0)), 2),
            "pop": round(pop_val, 1),
            "reach": round(reach_val, 2),
            "district": gdf_segments[gdf_segments["g250_id"] == g250]["district"].iloc[0],
            "is_synthetic": True,
            "geometry": cell_geom,
        })

    gdf_grid = gpd.GeoDataFrame(grid_list, crs="EPSG:5253")

    # Save files with SYNTHETIC in filenames
    out_segments_path = output_dir / "SYNTHETIC_segments.gpkg"
    out_metrics_path = output_dir / "SYNTHETIC_seg_metrics_IMBAT.parquet"
    out_grid_path = output_dir / "SYNTHETIC_grid_250.gpkg"

    gdf_segments.to_file(out_segments_path, layer="segments", driver="GPKG")
    df_metrics.to_parquet(out_metrics_path, index=False)
    gdf_grid.to_file(out_grid_path, layer="grid", driver="GPKG")

    print(f"Generated synthetic test data:")
    print(f"  Segments: {len(gdf_segments)} rows -> {out_segments_path}")
    print(f"  Metrics : {len(df_metrics)} rows -> {out_metrics_path}")
    print(f"  Grid 250: {len(gdf_grid)} cells -> {out_grid_path}")

    return {
        "segments": out_segments_path,
        "metrics": out_metrics_path,
        "grid_250": out_grid_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic H5 inputs for machinery unit tests.")
    parser.add_argument("--outdir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    generate_synthetic_inputs(output_dir=Path(args.outdir))
