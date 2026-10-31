"""
CLIMORFA Street — Campaign Point Selection (Provisional for Part A).
Selects 16 sampling points on the Alsancak benchmark tile:
- 4 morphology types:
  (A) compact canyon E-W
  (B) compact canyon N-S
  (C) tree-lined street
  (D) open space (Kordon promenade / square)
- x {sunlit, shaded at 13:00 on first analogue day}
- x 2 replicates
Total = 16 points.

Criteria:
- Road azimuth from data/02_interim/akilli_sehir_fua.gpkg roads_fua
- H/W from building heights
- Tree cover from tree DSM
- SVF and shadow from SOLWEIG
- Public pedestrian space (>= 1m from building walls, not on carriageway center)

Outputs:
data/street/field/PROVISIONAL_points.gpkg (+ CSV)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import rasterio
from scipy.ndimage import distance_transform_edt, uniform_filter


def select_points(
    svf_tif: str,
    shadow_tif: str,
    bld_dsm_tif: str,
    dem_tif: str,
    tree_dsm_tif: str,
    roads_gpkg: str,
    out_gpkg: str,
    out_csv: str,
    provisional: bool = True,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    print("Loading rasters...")
    with rasterio.open(bld_dsm_tif) as src_bld:
        bld_arr = src_bld.read(1)
        transform = src_bld.transform
        crs = src_bld.crs
        bounds = src_bld.bounds
        res_x = transform[0]
        res_y = -transform[4]

    with rasterio.open(dem_tif) as src_dem:
        dem_arr = src_dem.read(1)

    with rasterio.open(tree_dsm_tif) as src_tree:
        tree_arr = src_tree.read(1)

    with rasterio.open(svf_tif) as src_svf:
        svf_arr = src_svf.read(1)

    # Shadow at 13:00 (band 14)
    with rasterio.open(shadow_tif) as src_shadow:
        shadow_arr = src_shadow.read(14)

    # Derived geometric fields
    bld_height = np.maximum(0.0, bld_arr - dem_arr)
    tree_height = np.maximum(0.0, tree_arr - dem_arr)

    bld_mask = bld_height > 2.0
    dist_to_wall = distance_transform_edt(~bld_mask) * res_x

    # Mean neighborhood building height for H/W calculation (approx 25m radius)
    mean_bld_h = uniform_filter(bld_height, size=25)
    tree_nearby = uniform_filter(tree_height, size=7)

    print(f"Raster loaded: shape {bld_arr.shape}, bounds {bounds}")

    # Load roads in tile
    print("Loading road network...")
    bbox_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    roads = gpd.read_file(roads_gpkg, layer="roads_fua", bbox=bbox_tuple)
    roads = roads.to_crs(crs)
    print(f"Loaded {len(roads)} road features.")

    # Candidate points generation
    candidates = []

    # Inner window to avoid 100m edge buffer
    x_min_inner = bounds.left + 50.0
    x_max_inner = bounds.right - 50.0
    y_min_inner = bounds.bottom + 50.0
    y_max_inner = bounds.top - 50.0

    for idx, row in roads.iterrows():
        geom = row.geometry
        street_name = str(row.get("ADINUMARAS", "Nameless"))

        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)

        for line in lines:
            length = line.length
            if length < 10.0:
                continue

            # Sample along line every 5 meters
            num_samples = int(length // 5)
            if num_samples < 1:
                continue

            distances = np.linspace(2.5, length - 2.5, num_samples)
            for d in distances:
                pt_center = line.interpolate(d)
                pt_prev = line.interpolate(max(0.0, d - 2.0))
                pt_next = line.interpolate(min(length, d + 2.0))

                dx = pt_next.x - pt_prev.x
                dy = pt_next.y - pt_prev.y
                seg_len = np.hypot(dx, dy)
                if seg_len < 1e-3:
                    continue

                azimuth = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
                # Unit normal (perpendicular) vectors for sidewalks on both sides
                nx = -dy / seg_len
                ny = dx / seg_len

                # Test sidewalk offsets
                for offset in [-5.0, -3.0, -2.0, 2.0, 3.0, 5.0]:
                    px = pt_center.x + nx * offset
                    py = pt_center.y + ny * offset

                    if not (x_min_inner <= px <= x_max_inner and y_min_inner <= py <= y_max_inner):
                        continue

                    # Raster pixel index
                    col = int(round((px - transform[2]) / res_x))
                    row_idx = int(round((transform[5] - py) / res_y))

                    if not (0 <= row_idx < bld_arr.shape[0] and 0 <= col < bld_arr.shape[1]):
                        continue

                    # Check building wall distance (>= 1.2 m from walls, not inside building)
                    d_wall = dist_to_wall[row_idx, col]
                    if d_wall < 1.2 or d_wall > 25.0:
                        continue

                    if bld_mask[row_idx, col]:
                        continue

                    svf_val = float(svf_arr[row_idx, col])
                    shadow_val = float(shadow_arr[row_idx, col])
                    tree_h_val = float(tree_height[row_idx, col])
                    tree_near_val = float(tree_nearby[row_idx, col])
                    bld_h_val = float(mean_bld_h[row_idx, col])

                    # Local canyon width approx 2 * d_wall + 3.0
                    canyon_w = max(6.0, 2.0 * d_wall)
                    hw_ratio = bld_h_val / canyon_w

                    is_sunlit_13h = shadow_val > 0.5

                    # Road orientation
                    az_mod180 = azimuth % 180
                    is_ew = 65 <= az_mod180 <= 115
                    is_ns = (0 <= az_mod180 <= 25) or (155 <= az_mod180 <= 180)

                    # Determine candidate category
                    # Type A: Compact canyon E-W
                    if is_ew and hw_ratio >= 0.70 and tree_near_val < 1.2 and d_wall <= 7.0:
                        candidates.append({
                            "type": "A",
                            "x": px, "y": py, "street_name": street_name,
                            "azimuth": azimuth, "hw_ratio": hw_ratio,
                            "tree_height": tree_h_val, "svf": svf_val,
                            "dist_wall": d_wall, "is_sunlit": is_sunlit_13h,
                        })

                    # Type B: Compact canyon N-S
                    if is_ns and hw_ratio >= 0.70 and tree_near_val < 1.2 and d_wall <= 7.0:
                        candidates.append({
                            "type": "B",
                            "x": px, "y": py, "street_name": street_name,
                            "azimuth": azimuth, "hw_ratio": hw_ratio,
                            "tree_height": tree_h_val, "svf": svf_val,
                            "dist_wall": d_wall, "is_sunlit": is_sunlit_13h,
                        })

                    # Type C: Tree-lined street (tree overhead or nearby)
                    if (tree_h_val >= 1.5 or tree_near_val >= 0.8) and d_wall <= 14.0:
                        candidates.append({
                            "type": "C",
                            "x": px, "y": py, "street_name": street_name,
                            "azimuth": azimuth, "hw_ratio": hw_ratio,
                            "tree_height": max(tree_h_val, tree_near_val), "svf": svf_val,
                            "dist_wall": d_wall, "is_sunlit": is_sunlit_13h,
                        })

                    # Type D: Open space (Kordon / square / wide waterfront)
                    if svf_val >= 0.58 and d_wall >= 8.0:
                        candidates.append({
                            "type": "D",
                            "x": px, "y": py, "street_name": street_name,
                            "azimuth": azimuth, "hw_ratio": hw_ratio,
                            "tree_height": tree_h_val, "svf": svf_val,
                            "dist_wall": d_wall, "is_sunlit": is_sunlit_13h,
                        })

    df_cand = pd.DataFrame(candidates)
    print(f"Total candidate points generated: {len(df_cand)}")
    for t in ["A", "B", "C", "D"]:
        sub = df_cand[df_cand["type"] == t]
        sun = (sub["is_sunlit"] == True).sum()
        sha = (sub["is_sunlit"] == False).sum()
        print(f"  Type {t}: {len(sub)} candidates (sunlit: {sun}, shaded: {sha})")

    # Select exactly 2 replicates per (Type, Exposure) combination
    selected_records = []
    min_dist_between_replicates = 30.0  # meters

    for t, desc in [
        ("A", "compact canyon E-W"),
        ("B", "compact canyon N-S"),
        ("C", "tree-lined street"),
        ("D", "open space (Kordon/square)"),
    ]:
        for exp_label, exp_val in [("sunlit", True), ("shaded", False)]:
            pool = df_cand[(df_cand["type"] == t) & (df_cand["is_sunlit"] == exp_val)].copy()
            if len(pool) < 2:
                raise RuntimeError(f"Insufficient candidates for Type {t} {exp_label}: found {len(pool)}")

            # Shuffle candidates with fixed seed
            pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

            picked = []
            for _, cand in pool.iterrows():
                # Check distance against already picked in this group
                too_close = False
                for p in picked:
                    dist = np.hypot(cand["x"] - p["x"], cand["y"] - p["y"])
                    if dist < min_dist_between_replicates:
                        too_close = True
                        break
                if not too_close:
                    picked.append(cand)
                    if len(picked) == 2:
                        break

            if len(picked) < 2:
                # Fallback: take top 2 candidates
                print(f"Selecting top 2 candidates for Type {t} {exp_label}...")
                picked = [pool.iloc[0], pool.iloc[1]]

            for rep_idx, p in enumerate(picked, 1):
                selected_records.append({
                    "morphology_type": t,
                    "morphology_desc": desc,
                    "exposure_target_13h": exp_label,
                    "replicate_id": rep_idx,
                    "x": round(float(p["x"]), 2),
                    "y": round(float(p["y"]), 2),
                    "street_name": p["street_name"],
                    "street_azimuth_deg": round(float(p["azimuth"]), 1),
                    "canyon_hw_ratio": round(float(p["hw_ratio"]), 2),
                    "tree_height_m": round(float(p["tree_height"]), 2),
                    "svf": round(float(p["svf"]), 3),
                    "dist_to_wall_m": round(float(p["dist_wall"]), 2),
                    "shadow_13h": 1.0 if exp_val else 0.0,
                    "is_synthetic": False,
                    "provisional_geometry": provisional,
                })

    df_sel = pd.DataFrame(selected_records)
    assert len(df_sel) == 16, f"Expected exactly 16 points, got {len(df_sel)}"

    # Sort and assign to devices: DEV_A gets replicate 1, DEV_B gets replicate 2
    # Each device gets exactly 8 points, 1 of each (type, exposure) combination!
    df_sel = df_sel.sort_values(by=["morphology_type", "exposure_target_13h", "replicate_id"]).reset_index(drop=True)
    df_sel["device_id"] = np.where(df_sel["replicate_id"] == 1, "DEV_A", "DEV_B")
    df_sel["point_id"] = [f"P{i+1:02d}" for i in range(len(df_sel))]

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_sel["x"], df_sel["y"])]
    gdf = gpd.GeoDataFrame(df_sel, geometry=geometry, crs=crs)

    # Save outputs
    os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)
    gdf.to_file(out_gpkg, driver="GPKG")
    df_sel.to_csv(out_csv, index=False)

    print(f"\n=======================================================")
    print(f" Successfully selected exactly {len(gdf)} points!")
    print(f" Saved GPKG: {out_gpkg}")
    print(f" Saved CSV:  {out_csv}")
    print(f"=======================================================")
    print("\nSelected points summary table:")
    cols = ["point_id", "device_id", "morphology_type", "exposure_target_13h", "street_name", "svf", "dist_to_wall_m"]
    print(gdf[cols].to_string())

    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svf-tif", default=r"data/street/field/provisional/run_2026-11-12/processed_inputs/SVF/SkyViewFactor_0_0.tif")
    parser.add_argument("--shadow-tif", default=r"data/street/field/provisional/run_2026-11-12/output_folder/0_0/Shadow_0_0.tif")
    parser.add_argument("--bld-dsm-tif", default=r"data/street/benchmark/Building_DSM_1m.tif")
    parser.add_argument("--dem-tif", default=r"data/street/benchmark/DEM_1m.tif")
    parser.add_argument("--tree-dsm-tif", default=r"data/street/benchmark/Trees_1m.tif")
    parser.add_argument("--roads-gpkg", default=r"data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--out-gpkg", default=r"data/street/field/PROVISIONAL_points.gpkg")
    parser.add_argument("--out-csv", default=r"data/street/field/PROVISIONAL_points.csv")
    parser.add_argument("--provisional", action="store_true", default=True)
    args = parser.parse_args()

    select_points(
        svf_tif=args.svf_tif,
        shadow_tif=args.shadow_tif,
        bld_dsm_tif=args.bld_dsm_tif,
        dem_tif=args.dem_tif,
        tree_dsm_tif=args.tree_dsm_tif,
        roads_gpkg=args.roads_gpkg,
        out_gpkg=args.out_gpkg,
        out_csv=args.out_csv,
        provisional=args.provisional,
    )
