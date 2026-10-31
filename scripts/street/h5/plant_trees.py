"""CLIMORFA Street — H5 tree-planting scenario engine (lock §2 H5; A10 rulings 4 and 5).

Locked tree template (unchanged by A10):
  * crown (tree-top) height 8.0 m, crown diameter 6.0 m (radius 3.0 m), trunk zone 2.0 m
  * centres in PS cells 2.5-4.0 m from the segment axis, on the side with more PS cells (fallback: other side)
  * 8.0 m spacing along the segment, centred; n_trees = floor(length_m / 8) (A10 ruling 3 capacity rule)
SOLWEIG-GPU reads the Trees raster as canopy height above ground and sets the trunk zone to 0.25 x canopy height
(solweig_gpu/utci_process.py: vegdem2 = temp1 * 0.25 + dem), so an 8 m crown gets exactly the locked 2 m trunk zone.

Scenarios:
  P        the template on EVERY segment of segments.gpkg with PS cells (A10 ruling 4); tree_dsm written for all 427
           tiles to data/street/h5/P/tree_dsm/<tile>.tif.
  S_exact  the template on the S segments only (A10 ruling 5), tree_dsm written for a tile list (default: the 40
           validation tiles in data/street/h5/selection/s_exact_validation_tiles.csv).
  <other>  any scenario table with seg_id, tile_id, geometry and n_trees (legacy behaviour).

Tree positions are computed ONCE per segment in its home tile (the 1.2 km tile raster always contains the whole
segment) and stored as <scen>/tree_positions.parquet. Every tile raster (inner 1 km + 100 m buffer) then receives every
crown that overlaps it, including crowns of segments of neighbouring tiles, so shade from across a tile border is
represented. S_exact reuses the P positions of its segments when P's positions exist, so S_exact plants exactly the
trees that P planted on those segments (the exact-vs-additive validation then differs only by interactions).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
TILES_BASE = PROJECT_ROOT / "data/street/tiles/2m"
H5_BASE = PROJECT_ROOT / "data/street/h5"
SEGMENTS_GPKG = PROJECT_ROOT / "data/street/analysis/segments.gpkg"
TILES_CORE = PROJECT_ROOT / "data/street/grid/tiles_core.gpkg"
SELECTION_DIR = H5_BASE / "selection"
NODATA = -9999.0
TEMPLATE = {"crown_height_m": 8.0, "crown_diameter_m": 6.0, "trunk_zone_m": 2.0, "lateral_offset_m": [2.5, 4.0],
            "spacing_m": 8.0, "n_trees_rule": "floor(length_m / 8)", "side_rule": "side with more PS cells"}


# ---------------------------------------------------------------------------------------------------------------------
# geometry of the template
# ---------------------------------------------------------------------------------------------------------------------
def _is_ps(r, c, bld_arr, dem_arr, lmask_arr):
    b = bld_arr[r, c]
    d = dem_arr[r, c]
    lm = lmask_arr[r, c] if lmask_arr is not None else 1.0
    return (b - d <= 0.5) and (lm >= 0.5) and (b != NODATA) and (d != NODATA)


def compute_side_ps_count(line: LineString, bld_arr, dem_arr, lmask_arr, transform: Affine,
                          sample_spacing: float = 4.0, test_offset: float = 3.25) -> tuple[int, int]:
    """Count PS-eligible sample points (land, not building) left vs right of the axis at the nominal 3.25 m offset."""
    total_len = line.length
    if total_len < 2.0:
        return 0, 0
    s_coords = np.arange(1.0, total_len, sample_spacing)
    if len(s_coords) == 0:
        s_coords = [total_len / 2.0]
    rows, cols = bld_arr.shape
    left_ps = right_ps = 0
    inv_t = ~transform
    for s in s_coords:
        p1 = line.interpolate(max(0.0, s - 0.5))
        p2 = line.interpolate(min(total_len, s + 0.5))
        dx, dy = p2.x - p1.x, p2.y - p1.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        tx, ty = dx / norm, dy / norm
        for sign in (1, -1):                       # +1 = left normal (-ty, tx), -1 = right
            x = p1.x + test_offset * (-ty) * sign
            y = p1.y + test_offset * tx * sign
            c, r = inv_t * (x, y)
            c, r = int(math.floor(c)), int(math.floor(r))
            if 0 <= r < rows and 0 <= c < cols and _is_ps(r, c, bld_arr, dem_arr, lmask_arr):
                if sign == 1:
                    left_ps += 1
                else:
                    right_ps += 1
    return left_ps, right_ps


def place_trees_on_segment(line: LineString, n_trees: int, bld_arr, dem_arr, lmask_arr, transform: Affine,
                           tree_spacing_m: float = 8.0, crown_diameter_m: float = 6.0):
    """Tree centres (x, y, signed lateral offset, distance along line). 8 m spacing, centred along the segment;
    offset searched in [2.5, 4.0] m on the favoured side, then the other side, else nominal 3.25 m favoured side."""
    total_len = line.length
    if n_trees <= 0 or total_len < 2.0:
        return []
    left_ps, right_ps = compute_side_ps_count(line, bld_arr, dem_arr, lmask_arr, transform)
    favored_left = left_ps >= right_ps
    if n_trees == 1:
        s_coords = [total_len / 2.0]
    else:
        span = (n_trees - 1) * tree_spacing_m
        start = max(1.0, (total_len - span) / 2.0)
        s_coords = [start + i * tree_spacing_m for i in range(n_trees)]
    rows, cols = bld_arr.shape
    inv_t = ~transform
    offsets_to_try = [3.25, 3.0, 3.5, 2.75, 3.75, 2.5, 4.0]
    placed = []
    for s in s_coords:
        p1 = line.interpolate(max(0.0, s - 0.5))
        p2 = line.interpolate(min(total_len, s + 0.5))
        dx, dy = p2.x - p1.x, p2.y - p1.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        tx, ty = dx / norm, dy / norm
        left, right = (-ty, tx), (ty, -tx)
        prim, sec = (left, right) if favored_left else (right, left)
        found = None
        for (nx, ny), sgn in ((prim, 1.0), (sec, -1.0)):
            for off in offsets_to_try:
                xc, yc = p1.x + off * nx, p1.y + off * ny
                c, r = inv_t * (xc, yc)
                c, r = int(math.floor(c)), int(math.floor(r))
                if 0 <= r < rows and 0 <= c < cols and _is_ps(r, c, bld_arr, dem_arr, lmask_arr):
                    found = (xc, yc, sgn * off, s)
                    break
            if found:
                break
        if found is None:
            found = (p1.x + 3.25 * prim[0], p1.y + 3.25 * prim[1], 3.25, s)
        placed.append(found)
    return placed


def rasterize_trees_onto_dsm(tree_dsm, bld_arr, dem_arr, placed_trees, transform: Affine,
                             crown_height: float = 8.0, crown_diameter: float = 6.0):
    """Burn crowns into a canopy-height raster (vectorised). A pixel is crown if its centre is within the crown radius
    of a tree centre; building cells (building_dsm - dem > 0.5 m) and no-data cells are never planted; an existing
    taller canopy is kept (max). Returns (modified, n_pixels_modified, added_area_m2)."""
    mod = tree_dsm.copy()
    rows, cols = tree_dsm.shape
    if len(placed_trees) == 0:
        return mod, 0, 0.0
    pts = np.asarray([(p[0], p[1]) for p in placed_trees], dtype=float)
    radius = crown_diameter / 2.0
    resx, resy = abs(transform.a), abs(transform.e)
    inv_t = ~transform
    colf, rowf = inv_t * (pts[:, 0], pts[:, 1])
    c0, r0 = np.floor(colf).astype(int), np.floor(rowf).astype(int)
    k = int(math.ceil(radius / min(resx, resy))) + 1
    blocked = np.zeros((rows, cols), bool)
    if bld_arr is not None and dem_arr is not None:
        blocked |= (bld_arr - dem_arr) > 0.5
        blocked |= (dem_arr == NODATA) | (bld_arr == NODATA)
    blocked |= tree_dsm == NODATA
    hit = np.zeros((rows, cols), bool)
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            rr, cc = r0 + dr, c0 + dc
            ok = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
            if not ok.any():
                continue
            xp, yp = transform * (cc[ok] + 0.5, rr[ok] + 0.5)
            inside = np.hypot(xp - pts[ok, 0], yp - pts[ok, 1]) <= radius
            hit[rr[ok][inside], cc[ok][inside]] = True
    newly = hit & ~blocked & (mod < crown_height)
    mod[newly] = crown_height
    n = int(newly.sum())
    return mod, n, float(n * resx * resy)


def _rasterize_trees_loop_reference(tree_dsm, bld_arr, dem_arr, placed_trees, transform, crown_height=8.0,
                                    crown_diameter=6.0):
    """Original WP3.5 pixel loop (kept as an independent reference for the vectorised version; no no-data guard)."""
    mod = tree_dsm.copy()
    radius = crown_diameter / 2.0
    rows, cols = tree_dsm.shape
    inv_t = ~transform
    mask = np.zeros((rows, cols), bool)
    for xc, yc, _, _ in placed_trees:
        c_min, r_min = inv_t * (xc - radius, yc + radius)
        c_max, r_max = inv_t * (xc + radius, yc - radius)
        for r in range(max(0, int(math.floor(min(r_min, r_max)))), min(rows, int(math.ceil(max(r_min, r_max))) + 1)):
            for c in range(max(0, int(math.floor(min(c_min, c_max)))), min(cols, int(math.ceil(max(c_min, c_max))) + 1)):
                x_p, y_p = transform * (c + 0.5, r + 0.5)
                if math.hypot(x_p - xc, y_p - yc) <= radius:
                    if bld_arr is not None and dem_arr is not None and (bld_arr[r, c] - dem_arr[r, c]) > 0.5:
                        continue
                    if mod[r, c] < crown_height:
                        mod[r, c] = crown_height
                        mask[r, c] = True
    return mod, int(mask.sum()), float(mask.sum() * abs(transform.a) * abs(transform.e))


# ---------------------------------------------------------------------------------------------------------------------
# tiles
# ---------------------------------------------------------------------------------------------------------------------
def tile_raster_bounds(tile_id: str) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of the 1.2 km tile raster (inner 1 km + 100 m buffer)."""
    _, a, b = tile_id.split("_")
    x0, y0 = int(a) * 1000, int(b) * 1000
    return x0 - 100.0, y0 - 100.0, x0 + 1100.0, y0 + 1100.0


def load_tile_rasters(tile_id: str, tiles_base: Path = TILES_BASE) -> dict:
    p = {k: Path(tiles_base) / k / f"{tile_id}.tif" for k in ("tree_dsm", "building_dsm", "dem", "landmask")}
    for k in ("tree_dsm", "building_dsm", "dem"):
        if not p[k].exists():
            raise FileNotFoundError(f"missing tile raster {p[k]}")
    with rasterio.open(p["tree_dsm"]) as s:
        out = {"tree": s.read(1), "transform": s.transform, "profile": s.profile.copy()}
    with rasterio.open(p["building_dsm"]) as s:
        out["bld"] = s.read(1)
    with rasterio.open(p["dem"]) as s:
        out["dem"] = s.read(1)
    if p["landmask"].exists():
        with rasterio.open(p["landmask"]) as s:
            out["lmask"] = s.read(1)
    else:
        out["lmask"] = None
    return out


def all_core_tiles() -> list[str]:
    return sorted(gpd.read_file(TILES_CORE)["tile_id"].tolist())


def read_tile_list(spec: str | None) -> list[str] | None:
    """Comma list, or a .csv (column tile_id) / .txt (one per line) file; None passes through."""
    if not spec:
        return None
    p = Path(spec)
    if p.exists():
        if p.suffix == ".csv":
            return [str(t) for t in pd.read_csv(p)["tile_id"].tolist()]
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [t.strip() for t in spec.split(",") if t.strip()]


# ---------------------------------------------------------------------------------------------------------------------
# scenario segments and positions
# ---------------------------------------------------------------------------------------------------------------------
def load_scenario_segments(scenario: str, segments: str | Path | None, all_segments: bool,
                           spacing: float = 8.0) -> gpd.GeoDataFrame:
    """P / --all-segments: every segment of segments.gpkg with PS cells, n_trees = floor(L/8). Otherwise a scenario
    table (parquet / gpkg) with seg_id, tile_id, n_trees; geometry is taken from segments.gpkg when absent."""
    if all_segments:
        src = Path(segments) if segments else SEGMENTS_GPKG
        g = gpd.read_file(src, layer="segments")
        g = g[g["n_ps"] > 0].copy()
        g["n_trees"] = np.floor(g["length_m"] / spacing).astype(np.int64)
        return g[["seg_id", "tile_id", "length_m", "n_ps", "n_trees", "geometry"]]
    if segments is None:
        segments = SELECTION_DIR / "scenario_S_segments.parquet"
    p = Path(segments)
    if p.suffix == ".parquet":
        try:
            g = gpd.read_parquet(p)
        except Exception:
            g = pd.read_parquet(p)
    else:
        g = gpd.read_file(p)
    if "geometry" not in g.columns or not isinstance(g, gpd.GeoDataFrame):
        geo = gpd.read_file(SEGMENTS_GPKG, layer="segments", columns=["seg_id"])
        g = geo.merge(pd.DataFrame(g).drop(columns=["geometry"], errors="ignore"), on="seg_id", how="inner")
    if "n_trees" not in g.columns:
        g["n_trees"] = np.floor(g["length_m"] / spacing).astype(np.int64)
    return g


def compute_tree_positions(gdf: gpd.GeoDataFrame, tiles_base: Path = TILES_BASE, spacing: float = 8.0,
                           crown_diameter: float = 6.0, log_every: int = 25) -> pd.DataFrame:
    """Tree centres for every segment, computed in its home tile raster. Columns seg_id, tile_id, tree_idx, x, y,
    offset_m, s_m."""
    rows = []
    tiles = sorted(gdf["tile_id"].unique())
    t0 = time.time()
    for i, tid in enumerate(tiles, 1):
        r = load_tile_rasters(tid, tiles_base)
        sub = gdf[gdf["tile_id"] == tid]
        for seg_id, geom, n in zip(sub["seg_id"], sub.geometry, sub["n_trees"]):
            pts = place_trees_on_segment(geom, int(n), r["bld"], r["dem"], r["lmask"], r["transform"], spacing,
                                         crown_diameter)
            for k, (x, y, off, s) in enumerate(pts):
                rows.append((int(seg_id), tid, k, x, y, off, s))
        if log_every and (i % log_every == 0 or i == len(tiles)):
            print(f"  positions: {i}/{len(tiles)} tiles, {len(rows):,} trees ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows, columns=["seg_id", "tile_id", "tree_idx", "x", "y", "offset_m", "s_m"])


def positions_hash(pos: pd.DataFrame) -> str:
    a = pos[["seg_id", "x", "y"]].sort_values(["seg_id", "x", "y"]).to_numpy()
    return hashlib.sha1(np.ascontiguousarray(np.round(a, 3)).tobytes()).hexdigest()[:16]


def write_tile(tile_id: str, pos: pd.DataFrame, out_path: Path, tiles_base: Path, crown_height: float,
               crown_diameter: float, scenario: str) -> dict:
    xmin, ymin, xmax, ymax = tile_raster_bounds(tile_id)
    rad = crown_diameter / 2.0
    m = (pos["x"].to_numpy() > xmin - rad) & (pos["x"].to_numpy() < xmax + rad) & \
        (pos["y"].to_numpy() > ymin - rad) & (pos["y"].to_numpy() < ymax + rad)
    sub = pos[m]
    r = load_tile_rasters(tile_id, tiles_base)
    trees = list(zip(sub["x"], sub["y"], sub["offset_m"], sub["s_m"]))
    mod, npx, area = rasterize_trees_onto_dsm(r["tree"], r["bld"], r["dem"], trees, r["transform"], crown_height,
                                              crown_diameter)
    prof = r["profile"]
    prof.update(driver="GTiff", dtype="float32", nodata=NODATA, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(mod.astype(np.float32), 1)
        dst.set_band_description(1, f"Canopy height above ground (Scenario {scenario})")
    tmp.replace(out_path)
    home = int((sub["tile_id"] == tile_id).sum())
    return {"tile_id": tile_id, "trees_overlapping": int(len(sub)), "trees_home_tile": home,
            "trees_from_neighbours": int(len(sub) - home), "pixels_modified": npx, "canopy_area_added_m2": area,
            "status": "OK"}


def plant_scenario(scenario: str, segments=None, all_segments: bool = False, tiles: list[str] | None = None,
                   out_base: Path = H5_BASE, tiles_base: Path = TILES_BASE, positions_from: str | None = "P",
                   crown_height: float = 8.0, crown_diameter: float = 6.0, spacing: float = 8.0,
                   force: bool = False, dry_run: bool = False) -> dict:
    """Compute tree positions and write <out_base>/<scenario>/tree_dsm/<tile>.tif for every tile in `tiles`."""
    out_base, tiles_base = Path(out_base), Path(tiles_base)
    scen_dir = out_base / scenario
    tree_dir = scen_dir / "tree_dsm"
    if all_segments is None:
        all_segments = scenario == "P"
    if dry_run and not all_segments and not Path(segments or SELECTION_DIR / "scenario_S_segments.parquet").exists():
        print(f"[DRY-RUN {scenario}] scenario table {segments or SELECTION_DIR / 'scenario_S_segments.parquet'} does "
              f"not exist yet (written by select_scenarios.py in queue step 5); nothing to count. GPU: none.")
        return {"scenario": scenario, "dry_run": True, "table_missing": True}
    gdf = load_scenario_segments(scenario, segments, all_segments, spacing)
    syn = "SYNTHETIC" in str(out_base).upper() or ("is_synthetic" in gdf.columns and bool(gdf["is_synthetic"].any()))
    if syn and "SYNTHETIC" not in str(out_base).upper():
        raise RuntimeError(f"rule 6: synthetic scenario table must be planted under a SYNTHETIC folder ({out_base})")
    if tiles is None:
        if scenario == "S_exact":
            vt = SELECTION_DIR / "s_exact_validation_tiles.csv"
            if not vt.exists() and not dry_run:
                raise FileNotFoundError(f"{vt} missing: run select_scenarios.py first (or pass --tiles)")
            tiles = read_tile_list(str(vt)) if vt.exists() else []
        elif all_segments and tiles_base == TILES_BASE:
            tiles = all_core_tiles()
        else:
            tiles = sorted(gdf["tile_id"].unique().tolist())
    n_trees = int(gdf["n_trees"].sum())
    print(f"[{scenario}] segments: {len(gdf):,}, trees: {n_trees:,}, home tiles: {gdf['tile_id'].nunique()}, "
          f"tiles to write: {len(tiles)}")
    missing_inputs = [t for t in tiles if not (tiles_base / "tree_dsm" / f"{t}.tif").exists()]
    existing = [t for t in tiles if (tree_dir / f"{t}.tif").exists()]
    if dry_run:
        est_s = 0.0005 * n_trees + 1.5 * len(tiles)
        print(f"[DRY-RUN {scenario}] tile rasters missing: {len(missing_inputs)} {missing_inputs[:5]}; "
              f"tree_dsm already written: {len(existing)}/{len(tiles)}; CPU estimate ~{est_s / 60:.0f} min; "
              f"GPU: none. Nothing written.")
        return {"scenario": scenario, "dry_run": True, "n_segments": int(len(gdf)), "n_trees": n_trees,
                "n_tiles": len(tiles), "missing_inputs": missing_inputs, "existing": len(existing)}
    if missing_inputs:
        raise FileNotFoundError(f"missing base tile rasters for {len(missing_inputs)} tiles: {missing_inputs[:5]}")

    pos_path = scen_dir / "tree_positions.parquet"
    pos = None
    p_pos = out_base / str(positions_from) / "tree_positions.parquet" if positions_from else None
    if scenario != positions_from and p_pos is not None and p_pos.exists():
        pp = pd.read_parquet(p_pos)
        sel = pp[pp["seg_id"].isin(gdf["seg_id"])]
        cnt = sel.groupby("seg_id").size().reindex(gdf["seg_id"]).fillna(0).astype(int).to_numpy()
        if np.array_equal(cnt, gdf["n_trees"].to_numpy().astype(int)):
            pos = sel.reset_index(drop=True)
            print(f"[{scenario}] reusing {len(pos):,} tree positions of scenario {positions_from} ({p_pos})")
        else:
            print(f"[{scenario}] {positions_from} positions do not match n_trees; recomputing")
    if pos is None:
        print(f"[{scenario}] computing tree positions in home tiles ...")
        pos = compute_tree_positions(gdf, tiles_base, spacing, crown_diameter)
    if "is_synthetic" in gdf.columns or syn:
        pos["is_synthetic"] = syn
    scen_dir.mkdir(parents=True, exist_ok=True)
    old_hash = None
    man_path = scen_dir / "planting_manifest.json"
    if man_path.exists():
        try:
            old_hash = json.loads(man_path.read_text(encoding="utf-8")).get("positions_hash")
        except Exception:
            old_hash = None
    h = positions_hash(pos)
    if old_hash and old_hash != h and existing and not force:
        raise RuntimeError(f"[{scenario}] tree positions changed since the existing tree_dsm tiles were written "
                           f"({old_hash} -> {h}); rerun with --force")
    pos.to_parquet(pos_path, index=False)

    results = []
    t0 = time.time()
    for i, tid in enumerate(tiles, 1):
        outp = tree_dir / f"{tid}.tif"
        if outp.exists() and not force:
            results.append({"tile_id": tid, "status": "EXISTS"})
            continue
        res = write_tile(tid, pos, outp, tiles_base, crown_height, crown_diameter, scenario)
        results.append(res)
        if i % 25 == 0 or i == len(tiles):
            print(f"  [{scenario}] {i}/{len(tiles)} tiles written ({time.time() - t0:.0f} s)", flush=True)
    manifest = {"scenario": scenario, "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "template": TEMPLATE, "segments_source": str(segments or (SEGMENTS_GPKG if all_segments else "")),
                "all_segments": bool(all_segments), "n_segments": int(len(gdf)), "n_trees_planned": n_trees,
                "n_trees_placed": int(len(pos)), "positions_hash": h, "positions_path": str(pos_path),
                "tiles": tiles, "n_tiles": len(tiles), "is_synthetic": bool(syn), "tile_results": results}
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{scenario}] done: {len(pos):,} trees, {len(tiles)} tile rasters; manifest {man_path}")
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
def run_synthetic_unit_tests():
    """Template tests on synthetic geometry (no file I/O)."""
    print("\n=== plant_trees unit tests (synthetic geometry) ===")
    transform = Affine(2.0, 0.0, 500000.0, 0.0, -2.0, 4250000.0)
    rows, cols = 100, 100
    line = LineString([(500010.0, 4249900.0), (500090.0, 4249900.0)])          # 80 m east-west
    bld = np.zeros((rows, cols), np.float32)
    dem = np.zeros((rows, cols), np.float32)
    lm = np.ones((rows, cols), np.float32)
    bld[:50, :] = 10.0                                                          # buildings north (left) of the axis
    left, right = compute_side_ps_count(line, bld, dem, lm, transform)
    assert left == 0 and right > 0, (left, right)
    print(f"  [PASS] favoured side: left={left}, right={right} -> right (south)")
    n = int(math.floor(line.length / 8.0))
    placed = place_trees_on_segment(line, n, bld, dem, lm, transform, 8.0)
    assert len(placed) == n == 10
    sp = [math.hypot(placed[i + 1][0] - placed[i][0], placed[i + 1][1] - placed[i][1]) for i in range(n - 1)]
    assert abs(np.mean(sp) - 8.0) < 0.1
    offs = [abs(p[1] - 4249900.0) for p in placed]
    assert all(2.5 <= o <= 4.0 for o in offs), offs
    assert all(p[1] < 4249900.0 for p in placed), "trees must be on the south (PS) side"
    print(f"  [PASS] n = floor(80/8) = {n}, spacing {np.mean(sp):.2f} m, offsets {min(offs):.2f}-{max(offs):.2f} m")
    base = np.zeros((rows, cols), np.float32)
    mod, npx, area = rasterize_trees_onto_dsm(base, bld, dem, placed, transform)
    assert mod.max() == 8.0 and np.all(mod[:50, :] == 0.0)
    print(f"  [PASS] crown height 8.0 m, no crown pixel inside buildings ({npx} px)")
    rng = np.random.default_rng(7)
    rnd = [(500000 + rng.uniform(0, 200), 4250000 - rng.uniform(0, 200), 3.25, 0.0) for _ in range(300)]
    pre = rng.uniform(0, 10, (rows, cols)).astype(np.float32)
    a, na, _ = rasterize_trees_onto_dsm(pre, bld, dem, rnd, transform)
    b, nb, _ = _rasterize_trees_loop_reference(pre, bld, dem, rnd, transform)
    assert np.array_equal(a, b) and na == nb, "vectorised rasteriser differs from the reference loop"
    print(f"  [PASS] vectorised rasteriser == reference pixel loop on 300 random crowns ({na} px)")
    _, px1, a1 = rasterize_trees_onto_dsm(np.zeros((rows, cols), np.float32), np.zeros_like(bld), dem,
                                          [(500051.0, 4249849.0, 3.25, 40.0)], transform)
    assert px1 == 9 and a1 == 36.0, (px1, a1)
    print(f"  [PASS] pixel-centred crown: {px1} px = {a1:.0f} m2 (disc pi*3^2 = {math.pi * 9:.1f} m2)")
    print("=== plant_trees unit tests passed ===\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plant H5 scenario trees into tree_dsm rasters.")
    ap.add_argument("--scenario", default="P", help="P, S_exact, or another scenario name")
    ap.add_argument("--segments", default=None, help="scenario table (S_exact default: selection/scenario_S_segments"
                                                     ".parquet; P default: data/street/analysis/segments.gpkg)")
    ap.add_argument("--all-segments", action="store_true", help="plant every segment (implied by --scenario P)")
    ap.add_argument("--tiles", default=None, help="comma list or file of tiles to write (P default: all 427; "
                                                  "S_exact default: selection/s_exact_validation_tiles.csv)")
    ap.add_argument("--out-base", default=str(H5_BASE))
    ap.add_argument("--tiles-base", default=str(TILES_BASE))
    ap.add_argument("--positions-from", default="P", help="reuse this scenario's tree positions when they match")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="count segments/trees/tiles and check inputs; write nothing")
    ap.add_argument("--test", action="store_true", help="run the synthetic unit tests only")
    a = ap.parse_args()
    if a.test:
        run_synthetic_unit_tests()
        sys.exit(0)
    plant_scenario(a.scenario, a.segments, a.all_segments or a.scenario == "P", read_tile_list(a.tiles),
                   Path(a.out_base), Path(a.tiles_base), a.positions_from, force=a.force, dry_run=a.dry_run)
