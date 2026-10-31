"""CLIMORFA Street — H5 machinery tests after amendment A10. SYNTHETIC data only (rule 6); no GPU; no real H5 outcome.

Everything is written under data/street/h5/SYNTHETIC_A10/ (deleted and rebuilt on each run). Every table carries
is_synthetic = True and SYNTHETIC in its file name.

Chain tested end to end:
  synthetic world (8 tiles, 800 segments, ps_index, ratio16, baseline + scenario-P 16-band rasters with a planted
  per-segment cooling) -> WP3.0 extractor on the baseline (selection metrics) -> select_scenarios (both variants)
  -> S_exact rasters on the validation tiles (two worlds: additive, and with a planted non-additive interaction)
  -> extract_h5_effects (P and S_exact) -> evaluate_h5.

Independent checks (rule 6b), i.e. checks the code could fail:
  * truth is computed from the GENERATOR's segment-level band values, never from the extractor: the per-segment M2_38 /
    M2_32 of baseline and P must match exactly (non-PS cells hold 55 degC and two PS cells per segment are no-data, so
    an extractor that leaks non-PS cells or mishandles no-data changes M2);
  * the additive S and per-draw Z outcomes must equal the truth sums (prorated last Z segment included);
  * the Z target class must be LCZ 2 although LCZ 0 (no-data, A6) and LCZ 10 (capacity < N) have higher mean M2_38;
  * draws: N absorbed exactly, no replacement, length-proportional first pick (statistical, could fail);
  * validation: additive world -> rel diff 0, status 'test'; interaction world -> rel diff equals the truth value
    (> 0.15) and status 'descriptive only'; the 15 % boundary is tested on numbers;
  * evaluation stops when P effects are partial or S_exact effects are missing;
  * the scenario street-UTCI function reproduces pythermalcomfort with the VRB (mean of 16 ratios) and directional
    branches on a synthetic tile, and (optional, --parity-real-baseline TILE) reproduces the stored BASELINE
    utci_street raster of a real tile from its baseline Tmrt (no scenario involved, nothing written);
  * A11: 30 % of the segments get a crown-only effect (30 % of their cells cool by 6 degC, the median does not move), so
    dM2_seg = 0 while the companions respond; per-segment companions must equal truth computed from the generator's arrays;
    a tampered companion must leave ratio, verdict and decision unchanged;
  * planting: vectorised rasteriser == reference loop; crowns of a segment cross into the neighbour tile raster;
    S_exact reuses P's tree positions exactly.
Circular (display) checks are labelled as such in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from scripts.street.h5 import select_scenarios as SEL  # noqa: E402
from scripts.street.h5 import plant_trees as PT  # noqa: E402
from scripts.street.h5 import extract_h5_effects as EX  # noqa: E402
from scripts.street.h5 import evaluate_h5 as EV  # noqa: E402
from scripts.street.h5 import run_scenario_solweig as RS  # noqa: E402

SYN = PROJECT_ROOT / "data/street/h5/SYNTHETIC_A10"
DAYS = ("IMBAT", "CALM")
HOURS = list(range(6, 22))
M2_HOURS = list(range(10, 19))
AMP = {6: -6, 7: -5, 8: -4, 9: -2.5, 10: -1.5, 11: -0.8, 12: -0.3, 13: 0.0, 14: 0.2, 15: 0.3, 16: 0.0, 17: -0.5,
       18: -1.2, 19: -2.5, 20: -4, 21: -5}
NONPS_UTCI = 55.0
NOISE = np.linspace(-0.2, 0.2, 900)
PASSED = []


def ok(msg, independent=True):
    PASSED.append(msg)
    print(f"  [PASS{'' if independent else ', circular/display'}] {msg}")


# =====================================================================================================================
# synthetic world
# =====================================================================================================================
def nudge(v: np.ndarray, gap: float = 0.3) -> np.ndarray:
    v = v.copy()
    for thr in (32.0, 38.0):
        near = np.abs(v - thr) < gap
        v[near] = thr + np.where(v[near] >= thr, gap, -gap)
    return v


def m2(vals: np.ndarray, thr: float) -> np.ndarray:
    """vals: (n_seg, 16) band values for hours 6..21 -> count over hours 10..18 >= thr (truth, generator side)."""
    idx = [HOURS.index(h) for h in M2_HOURS]
    return (vals[:, idx] >= thr).sum(axis=1).astype(float)


def scen_values(base: np.ndarray, cool: np.ndarray) -> np.ndarray:
    """base - cool, with cool adjusted (>= 0) so that no scenario value lies within 0.3 of 32 / 38."""
    s = nudge(base - cool)
    return np.minimum(s, base)  # cooling never warms


class World:
    tiles = [f"T_{900 + i:04d}_{4900 + j:04d}" for j in range(2) for i in range(4)]

    def __init__(self, seed=20261030):
        rng = np.random.default_rng(seed)
        plan = [(0, 120, (200, 300), 40.0), (10, 40, (20, 30), 40.5), (2, 200, (40, 230), 38.6),
                (3, 140, (40, 230), 37.0), (5, 150, (40, 230), 35.5), (6, 150, (40, 230), 34.0)]
        lcz = np.concatenate([[c] * n for c, n, _, _ in plan])
        lens = np.concatenate([rng.uniform(lo, hi, n) for _, n, (lo, hi), _ in plan])
        lev = np.concatenate([np.full(n, b) for _, n, _, b in plan]) + rng.normal(0, 1.5, len(lcz))
        perm = rng.permutation(len(lcz))
        lcz, lens, lev = lcz[perm], lens[perm], lev[perm]
        rows = []
        for k in range(len(lcz)):
            ti, blk = divmod(k, 100)
            tid = self.tiles[ti]
            br, bc = divmod(blk, 10)
            x0, y0 = int(tid.split("_")[1]) * 1000, int(tid.split("_")[2]) * 1000
            cx, cy = x0 + (bc * 50 + 25) * 2.0, y0 + 1000 - (br * 50 + 25) * 2.0
            rows.append(dict(seg_id=1_000_000 + ti * 1000 + blk, tile_id=tid, br=br, bc=bc, cx=cx, cy=cy,
                             lcz_primary=int(lcz[k]), length_m=round(float(lens[k]), 2), level=float(lev[k])))
        self.seg = pd.DataFrame(rows)
        self.seg["g250_id"] = [f"{int(x // 250)}_{int(y // 250)}" for x, y in zip(self.seg.cx, self.seg.cy)]
        self.base = {}
        self.cool = {}
        for d, dl in (("IMBAT", 0.0), ("CALM", 0.8)):
            v = self.seg["level"].to_numpy()[:, None] + dl + np.array([AMP[h] for h in HOURS])[None, :]
            self.base[d] = nudge(v)
            a = rng.uniform(0.0, 4.0, len(self.seg))
            shape = np.array([1.0 if 10 <= h <= 18 else 0.5 for h in HOURS])
            self.cool[d] = a[:, None] * shape[None, :]
        self.crown_seg = rng.random(len(self.seg)) < 0.30          # A11: cells cool, the segment median does not move
        for d in DAYS:
            self.cool[d][self.crown_seg] = 0.0
        self.cell_rec = {"baseline": {d: {} for d in DAYS}, "P": {d: {} for d in DAYS}}
        self.P = {d: scen_values(self.base[d], self.cool[d]) for d in DAYS}
        cells = sorted(set(self.seg["g250_id"]) | {f"{(int(t.split('_')[1]) * 1000) // 250 + i}_"
                                                    f"{(int(t.split('_')[2]) * 1000) // 250 + j}"
                                                    for t in self.tiles for i in range(4) for j in range(4)})
        self.grid = pd.DataFrame({"g250_id": cells, "pop_res": rng.lognormal(5, 1, len(cells)),
                                  "pop": rng.lognormal(5, 1, len(cells)), "reach": rng.uniform(2000, 9000, len(cells))})

    # ---- rasters ----
    def ps_array(self, tid):
        a = np.zeros((500, 500), np.int32)
        for r in self.seg[self.seg.tile_id == tid].itertuples():
            a[r.br * 50 + 10:r.br * 50 + 40, r.bc * 50 + 10:r.bc * 50 + 40] = r.seg_id
        return a

    def utci_stack(self, tid, vals: np.ndarray, rng, crown=False, rec=None) -> np.ndarray:
        """16-band raster: PS cells = segment value + symmetric noise (two extreme cells no-data), non-PS = 55.
        crown: on crown_seg segments 270 of the 900 cells cool by 6 degC. rec: {seg_id: (cell hours, mean band 15)}."""
        out = np.full((16, 500, 500), NONPS_UTCI, np.float32)
        out[:, :5, :5] = -9999.0
        sub = self.seg[self.seg.tile_id == tid]
        for i, r in zip(sub.index, sub.itertuples()):
            nz = rng.permutation(NOISE).reshape(30, 30)
            blk = vals[i][:, None, None] + nz[None, :, :]
            ext = (nz == NOISE[0]) | (nz == NOISE[-1])
            if crown and self.crown_seg[i]:
                blk[:, rng.permutation(900).reshape(30, 30) < 270] -= 6.0
            blk[:, ext] = -9999.0
            out[:, r.br * 50 + 10:r.br * 50 + 40, r.bc * 50 + 10:r.bc * 50 + 40] = blk
            if rec is not None:
                w_ = out[:, r.br * 50 + 10:r.br * 50 + 40, r.bc * 50 + 10:r.bc * 50 + 40][:, ~ext].astype(np.float64)
                rec[r.seg_id] = (float((w_[[HOURS.index(h) for h in M2_HOURS]] >= 38.0).sum(axis=0).mean()),
                                 float(w_[HOURS.index(15)].mean()))
        return out

    @staticmethod
    def write(path: Path, arr: np.ndarray, tid: str, dtype="float32", nodata=-9999.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        x0, y0 = int(tid.split("_")[1]) * 1000, int(tid.split("_")[2]) * 1000
        arr = arr if arr.ndim == 3 else arr[None]
        with rasterio.open(path, "w", driver="GTiff", width=500, height=500, count=arr.shape[0], dtype=dtype,
                           crs="EPSG:5253", transform=from_origin(x0, y0 + 1000, 2, 2), nodata=nodata,
                           compress="deflate") as d:
            d.write(arr.astype(dtype))

    def write_root(self, root: Path, vals_by_day: dict, tiles, seed, crown=False, rec=None):
        rng = np.random.default_rng(seed)
        for tid in tiles:
            for d in DAYS:
                u = self.utci_stack(tid, vals_by_day[d], rng, crown, None if rec is None else rec[d])
                self.write(root / "utci" / d / "utci_street" / f"{tid}.tif", u, tid)
                self.write(root / "utci" / d / "utci_uniform" / f"{tid}.tif", u, tid)
                self.write(root / "solweig" / d / "tmrt" / f"{tid}.tif", np.full((16, 500, 500), 60, np.float32), tid)
                self.write(root / "solweig" / d / "shadow" / f"{tid}.tif", np.full((16, 500, 500), 50, np.uint8), tid,
                           "uint8", 255)
                self.write(root / "solweig" / d / "svf" / f"{tid}.tif", np.full((500, 500), 0.5, np.float32), tid)

    def build(self):
        if SYN.exists():
            shutil.rmtree(SYN)
        SYN.mkdir(parents=True)
        for tid in self.tiles:
            self.write(SYN / "ps_index" / f"{tid}.tif", self.ps_array(tid), tid, "int32", 0)
            self.write(SYN / "ratio16" / f"{tid}.tif", np.full((16, 500, 500), 0.5, np.float32), tid)
        self.write_root(SYN / "baseline", self.base, self.tiles, 1, rec=self.cell_rec["baseline"])
        self.write_root(SYN / "P", self.P, self.tiles, 2, crown=True, rec=self.cell_rec["P"])
        g = gpd.GeoDataFrame(
            self.seg.assign(district="SYN", lcz_weak250=self.seg.lcz_primary, n_ps=900, is_synthetic=True)
            [["seg_id", "tile_id", "district", "lcz_primary", "lcz_weak250", "length_m", "n_ps", "g250_id",
              "is_synthetic"]],
            geometry=[LineString([(x - 40, y), (x + 40, y)]) for x, y in zip(self.seg.cx, self.seg.cy)],
            crs="EPSG:5253")
        g.to_file(SYN / "SYNTHETIC_segments.gpkg", layer="segments", driver="GPKG")
        gg = gpd.GeoDataFrame(self.grid.assign(is_synthetic=True), crs="EPSG:5253",
                              geometry=[box(int(c.split("_")[0]) * 250, int(c.split("_")[1]) * 250,
                                            int(c.split("_")[0]) * 250 + 250, int(c.split("_")[1]) * 250 + 250)
                                        for c in self.grid.g250_id])
        gg.to_file(SYN / "SYNTHETIC_grid_250.gpkg", layer="grid", driver="GPKG")

    def dirs(self, root: Path) -> dict:
        return {"solweig": root / "solweig", "utci": root / "utci", "ps": SYN / "ps_index", "ratio16": SYN / "ratio16"}


# =====================================================================================================================
def test_pure_selection_rules():
    print("\n--- selection rules (pure functions) ---")
    df = pd.DataFrame({"seg_id": [5, 3, 4, 1, 2], "m2_vstrong": [6, 6, 6, 9, 6], "utci_b15": [40.0, 41.0, 41.0, 39.0,
                                                                                               np.nan]})
    r = SEL.rank_for_budget(df, "m2_vstrong")
    assert r["seg_id"].tolist() == [1, 3, 4, 5, 2], r["seg_id"].tolist()
    ok("tie-break: metric desc, then M1 desc, then seg_id asc (NaN M1 last)")
    assert SEL.capacity_trees([20, 23.9, 24, 7.9]).tolist() == [2, 2, 3, 0]
    ok("capacity = floor(length / 8)")
    for a, e, exp in ((1.149, 1.0, "test"), (1.151, 1.0, "descriptive only"), (0.851, 1.0, "test"),
                      (0.849, 1.0, "descriptive only"), (0.0, 0.0, "test"), (1.0, 0.0, "descriptive only")):
        rel, st = EV.validation_status(a, e, 0.15)
        assert st == exp, (a, e, rel, st)
    ok("15 % rule boundary: rel 0.149 -> test, 0.151 -> descriptive only (both directions); exact 0 handled")
    c = pd.DataFrame({"seg_id": range(10), "length_m": np.r_[np.full(9, 10.0), 910.0], "capacity_trees": 100})
    first = np.array([SEL.draw_zone(c, 5, np.random.default_rng(i))["seg_id"].iloc[0] for i in range(2000)])
    share = float((first == 9).mean())
    assert 0.86 < share < 0.96, share
    ok(f"A5 length-proportional sampling: long segment (91 % of length) drawn first in {share:.3f} of 2000 draws")


def test_utci_function_synthetic():
    print("\n--- scenario street-UTCI function (synthetic tile) ---")
    from pythermalcomfort.models import utci
    from scripts.street.wind.compute_street_wind import interpolate_ratio_at_direction
    rng = np.random.default_rng(3)
    tmrt = rng.uniform(35, 70, (16, 500, 500)).astype(np.float32)
    inv = np.ones((500, 500), bool)
    inv[100:110, 200:210] = False
    r16 = rng.uniform(0.2, 0.9, (16, 500, 500)).astype(np.float32)
    met = pd.DataFrame({"it": HOURS, "Td": 34.0, "RH": 40.0, "Wind": 3.0, "Wd": 300.0, "vrb": False})
    met.loc[met.it == 15, ["Wd", "vrb"]] = [-999.0, True]
    us, uu = RS.compute_street_utci_arrays(tmrt, inv, r16, met)
    m = ~inv
    for h, vrb in ((15, True), (14, False)):
        b = HOURS.index(h)
        rat = r16.mean(axis=0) if vrb else interpolate_ratio_at_direction(r16, 300.0)
        ref = utci(tdb=34.0, tr=tmrt[b][m], v=np.clip(3.0 * rat[m], 0.5, 17), rh=40.0, limit_inputs=False,
                   round_output=False).utci
        assert np.allclose(us[b][m], ref, atol=1e-4), (h, np.abs(us[b][m] - ref).max())
    assert np.all(us[:, inv] == -9999.0)
    ok("VRB hour uses the mean of the 16 sector ratios, other hours the interpolated direction; invalid cells -9999")


def test_utci_parity_real_baseline(tile: str):
    print(f"\n--- OPTIONAL parity on real BASELINE tile {tile} (reads baseline rasters only, writes nothing) ---")
    from scripts.street.utci.compute_street_utci import load_day_met, load_tile_masks
    inv = load_tile_masks(tile)
    with rasterio.open(PROJECT_ROOT / f"data/street/wind/ratio16/{tile}.tif") as s:
        r16 = s.read()
    for d in DAYS:
        with rasterio.open(PROJECT_ROOT / f"data/street/solweig/{d}/tmrt/{tile}.tif") as s:
            tmrt = s.read()
        with rasterio.open(PROJECT_ROOT / f"data/street/utci/{d}/utci_street/{tile}.tif") as s:
            ref = s.read()
        us, _ = RS.compute_street_utci_arrays(tmrt, inv, r16, load_day_met(d))
        diff = float(np.abs(us.astype(np.float64) - ref.astype(np.float64)).max())
        same_mask = bool(np.array_equal(us == -9999.0, ref == -9999.0))
        print(f"    {d}: max |scenario-function - stored baseline utci_street| = {diff:.2e}, nodata masks equal: "
              f"{same_mask}")
        assert diff < 1e-3 and same_mask
    ok(f"scenario UTCI code reproduces the stored baseline utci_street of {tile} on both days (incl. CALM VRB hours)")


def test_planting_cross_tile():
    print("\n--- planting: positions, cross-tile crowns, S_exact reuse (synthetic 2-tile rasters) ---")
    PT.run_synthetic_unit_tests()
    base = SYN / "plant_world"
    tb = base / "tiles2m"
    tA, tB = "T_0950_4950", "T_0951_4950"
    for t in (tA, tB):
        x0, y0 = int(t.split("_")[1]) * 1000, int(t.split("_")[2]) * 1000
        prof = dict(driver="GTiff", width=600, height=600, count=1, dtype="float32", crs="EPSG:5253",
                    transform=from_origin(x0 - 100, y0 + 1100, 2, 2), nodata=-9999.0)
        for k, v in (("tree_dsm", 0.0), ("building_dsm", 0.0), ("dem", 0.0), ("landmask", 1.0)):
            (tb / k).mkdir(parents=True, exist_ok=True)
            with rasterio.open(tb / k / f"{t}.tif", "w", **prof) as d:
                d.write(np.full((1, 600, 600), v, np.float32))
    segs = gpd.GeoDataFrame({"seg_id": [1, 2, 3], "tile_id": [tA, tA, tB], "length_m": [200.0, 120.0, 81.0],
                             "n_ps": [900, 900, 900], "lcz_primary": 2, "is_synthetic": True},
                            geometry=[LineString([(950997, 4950400), (950997, 4950200)]),
                                      LineString([(950300, 4950500), (950420, 4950500)]),
                                      LineString([(951500, 4950700), (951581, 4950700)])], crs="EPSG:5253")
    sp = base / "SYNTHETIC_plant_segments.gpkg"
    segs.to_file(sp, layer="segments", driver="GPKG")
    man = PT.plant_scenario("P", sp, True, [tA, tB], out_base=base, tiles_base=tb)
    assert man["n_trees_placed"] == 25 + 15 + 10, man["n_trees_placed"]
    ok("P: n_trees = floor(L/8) per segment (25 + 15 + 10)")
    with rasterio.open(base / "P/tree_dsm" / f"{tB}.tif") as s:
        b_arr = s.read(1)
        r, c = s.index(951000.5, 4950300)
    assert b_arr[r, c] == 8.0, "crown of tile-A segment missing in tile-B raster"
    with rasterio.open(base / "P/tree_dsm" / f"{tA}.tif") as s:
        a_arr = s.read(1)
        r2, c2 = s.index(951000.5, 4950300)
    assert a_arr[r2, c2] == 8.0
    ok("a crown 0.25 m east of the tile border appears in both tile rasters (neighbour-tile shade represented)")
    s_tab = segs[segs.seg_id.isin([1, 3])].assign(n_trees=lambda x: np.floor(x.length_m / 8).astype(int))
    s_tab.to_parquet(base / "SYNTHETIC_S_table.parquet")
    man_s = PT.plant_scenario("S_exact", base / "SYNTHETIC_S_table.parquet", False, [tA, tB], out_base=base,
                              tiles_base=tb, positions_from="P")
    pp = pd.read_parquet(base / "P/tree_positions.parquet")
    ps_ = pd.read_parquet(base / "S_exact/tree_positions.parquet")
    exp = pp[pp.seg_id.isin([1, 3])].sort_values(["seg_id", "tree_idx"])[["x", "y"]].to_numpy()
    assert np.allclose(ps_.sort_values(["seg_id", "tree_idx"])[["x", "y"]].to_numpy(), exp)
    with rasterio.open(base / "S_exact/tree_dsm" / f"{tA}.tif") as s:
        sa = s.read(1)
        r3, c3 = s.index(950360, 4950503.25)
    assert sa[r3, c3] == 0.0 and a_arr[r3, c3] == 8.0, "S_exact must not plant segment 2"
    ok(f"S_exact reuses P positions exactly ({man_s['n_trees_placed']} trees) and plants only S segments")


# =====================================================================================================================
def run_chain(parity_tile: str | None = None):
    print("=======================================================")
    print("  H5 A10 machinery test (SYNTHETIC, data/street/h5/SYNTHETIC_A10)")
    print("=======================================================")
    test_pure_selection_rules()
    test_utci_function_synthetic()
    w = World()
    w.build()
    test_planting_cross_tile()

    # ---- baseline metrics via the WP3.0 extractor -> selection input ----
    print("\n--- baseline extraction (WP3.0 extract_tile_day on synthetic rasters) ---")
    parts = []
    for t in w.tiles:
        _, _, _, df = EX._job((t, "IMBAT", "base", w.dirs(SYN / "baseline")))
        parts.append(df)
    met = pd.concat(parts, ignore_index=True)
    met["partial_run"] = False
    met["n_tiles_at_extraction"] = len(w.tiles)
    met["is_synthetic"] = True
    met.to_parquet(SYN / "SYNTHETIC_seg_metrics_IMBAT.parquet", index=False)
    truth = w.seg.copy()
    for d in DAYS:
        truth[f"m2_38_base_{d}"], truth[f"m2_32_base_{d}"] = m2(w.base[d], 38), m2(w.base[d], 32)
        truth[f"m2_38_P_{d}"], truth[f"m2_32_P_{d}"] = m2(w.P[d], 38), m2(w.P[d], 32)
        truth[f"m1_{d}"] = w.base[d][:, HOURS.index(15)]
    j = met.merge(truth, on="seg_id")
    assert len(j) == 800 and np.array_equal(j["m2_vstrong"], j["m2_38_base_IMBAT"]) and \
        np.array_equal(j["m2_strong"], j["m2_32_base_IMBAT"]) and (j["n_ps_valid"] == 898).all()
    ok("extracted baseline M2_38 / M2_32 equal the generator truth for 800/800 segments; n_ps_valid 898 (no-data)")

    # ---- selection ----
    print("\n--- select_scenarios (both variants) ---")
    seldir = SYN / "selection"
    res = SEL.select_scenarios(SYN / "SYNTHETIC_segments.gpkg", SYN / "SYNTHETIC_seg_metrics_IMBAT.parquet", seldir,
                               n_validation_tiles=4)
    p = res["primary"]
    tt = truth.rename(columns={"m2_38_base_IMBAT": "m2_vstrong", "m1_IMBAT": "utci_b15"})
    tt = tt.sort_values(["m2_vstrong", "utci_b15", "seg_id"], ascending=[False, False, True])
    cum = tt["length_m"].cumsum().to_numpy()
    k = int(np.searchsorted(cum, 0.10 * tt["length_m"].sum()))
    n_exp = int(np.floor(tt["length_m"].iloc[:k + 1] / 8).sum())
    assert p["tree_budget_N"] == n_exp, (p["tree_budget_N"], n_exp)
    ok(f"budget N = {n_exp} (independent re-derivation from the truth table)")
    cls = truth.groupby("lcz_primary").agg(mean=("m2_38_base_IMBAT", "mean"),
                                          cap=("length_m", lambda x: int(np.floor(x / 8).sum())))
    assert cls.loc[0, "mean"] > cls.loc[2, "mean"] and cls.loc[0, "cap"] >= n_exp      # only A6 can exclude LCZ 0
    assert cls.loc[10, "mean"] > cls.loc[2, "mean"] and cls.loc[10, "cap"] < n_exp     # only capacity excludes LCZ 10
    assert cls.drop(index=[0, 10])["mean"].idxmax() == 2
    assert p["scenario_Z"]["target_lcz"] == 2, p["scenario_Z"]["target_lcz"]
    ok(f"Z target = LCZ 2: LCZ 0 (mean {cls.loc[0, 'mean']:.2f}, capacity >= N) excluded by A6, LCZ 10 (mean "
       f"{cls.loc[10, 'mean']:.2f}) excluded by capacity {cls.loc[10, 'cap']} < N; LCZ 2 mean {cls.loc[2, 'mean']:.2f}")
    z = pd.read_parquet(seldir / "SYNTHETIC_scenario_Z_draws.parquet")
    s = pd.read_parquet(seldir / "SYNTHETIC_scenario_S_segments.parquet")
    assert int(s["n_trees"].sum()) == n_exp and s["g250_id"].notna().all() and z["g250_id"].notna().all()
    for dr, g in z.groupby("draw_id"):
        assert int(g["n_trees"].sum()) == n_exp and g["seg_id"].is_unique and (g["lcz_primary"] == 2).all()
        assert np.allclose(g["prorate_frac"], g["n_trees"] / g["capacity_trees"])
        assert (g["prorate_frac"].iloc[:-1] == 1).all() and g["prorate_frac"].iloc[-1] <= 1
    assert z.groupby("draw_id")["seg_id"].apply(frozenset).nunique() == 20
    ok("20 draws: each absorbs N exactly, no replacement, only LCZ 2, last segment prorated, 20 distinct sets; g250_id")
    res2 = SEL.select_scenarios(SYN / "SYNTHETIC_segments.gpkg", SYN / "SYNTHETIC_seg_metrics_IMBAT.parquet",
                                SYN / "selection_repeat", n_validation_tiles=4)
    z2 = pd.read_parquet(SYN / "selection_repeat/SYNTHETIC_scenario_Z_draws.parquet")
    assert z.equals(z2) and res2["primary"]["s_exact_validation"]["tiles"] == p["s_exact_validation"]["tiles"]
    ok("selection reproducible (seed 20261030)", independent=False)
    vt = p["s_exact_validation"]["tiles"]
    assert vt == SEL.draw_validation_tiles(p["scenario_S"]["tiles"], 4, 20261030) and set(vt) <= set(
        p["scenario_S"]["tiles"])
    ok(f"validation tiles = seeded draw of 4 from the {len(p['scenario_S']['tiles'])} S tiles: {vt}")
    lk = res["locked"]
    assert lk["metric_col"] == "m2_strong" and (seldir / "SYNTHETIC_scenario_S_segments_M2_32.parquet").exists()
    ok(f"locked variant written (M2_32): N = {lk['tree_budget_N']}, target LCZ {lk['scenario_Z']['target_lcz']}",
       independent=False)

    # ---- S_exact worlds on the validation tiles ----
    s_ids = set(s["seg_id"])
    in_s = w.seg["seg_id"].isin(s_ids).to_numpy()
    worlds = {}
    for name, factor in (("additive", 1.0), ("interact", 0.5)):
        vals = {d: np.where(in_s[:, None], scen_values(w.base[d], factor * w.cool[d]), w.base[d]) for d in DAYS}
        root = SYN / f"S_exact_{name}"
        w.write_root(root, vals, vt, 3)
        worlds[name] = (root, vals)

    # ---- effects ----
    print("\n--- extract_h5_effects ---")
    bd = {"solweig": SYN / "baseline/solweig", "utci": SYN / "baseline/utci", "ps": SYN / "ps_index",
          "ratio16": SYN / "ratio16"}
    sp = SYN / "SYNTHETIC_segments.gpkg"
    EX.extract_effects("P", w.tiles, scen_root=SYN / "P", base_dirs=bd, segments_path=sp, workers=2)
    eP = pd.read_parquet(SYN / "P/SYNTHETIC_seg_effects_IMBAT.parquet").merge(truth, on="seg_id")
    for d in DAYS:
        e = pd.read_parquet(SYN / f"P/SYNTHETIC_seg_effects_{d}.parquet").merge(truth, on="seg_id")
        assert len(e) == 800
        assert np.array_equal(e["dm2_38"], e[f"m2_38_base_{d}"] - e[f"m2_38_P_{d}"])
        assert np.array_equal(e["dm2_32"], e[f"m2_32_base_{d}"] - e[f"m2_32_P_{d}"])
        assert (e["dm2_38"] >= 0).all() and e["is_synthetic"].all() and not e["partial_run"].any()
    ok(f"P effects: per-segment dM2_38 / dM2_32 equal the planted truth on both days (IMBAT sum dM2_38 = "
       f"{eP['dm2_38'].sum():.0f}, segments with dM2_38 > 0: {(eP['dm2_38'] > 0).sum()})")
    print("\n--- A11: per-segment companions against the generator ---")
    for d in DAYS:
        e = pd.read_parquet(SYN / f"P/SYNTHETIC_seg_effects_{d}.parquet").set_index("seg_id")
        for col, k in (("cell_h38", 0), ("mean_utci_b15", 1)):
            for side, key in (("base", "baseline"), ("scen", "P")):
                t_ = np.array([w.cell_rec[key][d][i][k] for i in e.index])
                assert np.allclose(e[f"{col}_{side}"], t_, rtol=1e-9, atol=1e-6), (d, col, side)
    flagged = e.index.isin(w.seg.loc[w.crown_seg, "seg_id"])
    assert (e.loc[flagged, "dm2_38"] == 0).all() and (e.loc[flagged, "dm2_32"] == 0).all()
    n_resp = int((e.loc[flagged, "dcell_h38"] > 0.01).sum())
    assert n_resp > 50 and (e.loc[flagged, "dmean_b15"] > 1.0).all()
    ok(f"A11 companions equal the generator truth on 800 segments, both days; on {int(flagged.sum())} crown-only segments "
       f"dM2 = 0 but dmean_b15 > 1 degC everywhere and dcell_h38 > 0 on {n_resp}")
    for name, (root, _) in worlds.items():
        EX.extract_effects("S_exact", vt, scen_root=root, base_dirs=bd, segments_path=sp, workers=1)

    # ---- evaluation ----
    print("\n--- evaluate_h5 (additive world) ---")
    wmap = dict(zip(w.grid.g250_id, w.grid.pop_res * w.grid.reach))
    rep = EV.evaluate_h5(seldir, SYN / "P", worlds["additive"][0], SYN / "SYNTHETIC_grid_250.gpkg", sp,
                         SYN / "results_additive")
    tw = truth.set_index("seg_id")
    dm_t = tw["m2_38_base_IMBAT"] - tw["m2_38_P_IMBAT"]
    S_t = float(sum(dm_t[i] * wmap[tw.loc[i, "g250_id"]] for i in s["seg_id"]))
    per = pd.read_csv(SYN / "results_additive/SYNTHETIC_h5_evaluation_summary.csv")
    Z_t = z.assign(c=lambda x: [dm_t[i] * wmap[g] * f for i, g, f in zip(x.seg_id, x.g250_id, x.prorate_frac)]) \
        .groupby("draw_id")["c"].sum()
    assert math.isclose(rep["scenario_S"]["delta_weighted_m2"], round(S_t, 2), rel_tol=1e-9, abs_tol=0.01)
    assert np.allclose(per.set_index("draw_id")["delta_weighted_m2_z"], Z_t, rtol=1e-9)
    r_t = np.median(S_t / Z_t.to_numpy())
    assert math.isclose(rep["targeting_ratio_S_vs_Z"]["median"], round(r_t, 3), abs_tol=0.0006)
    ok(f"additive outcome recovers the planted effect: S = {S_t:,.1f}, Z median = {np.median(Z_t):,.1f}, "
       f"S/Z median = {r_t:.3f} (truth sums from the generator)")
    assert rep["verdict_status"] == "test" and rep["validation"]["rel_diff"] == 0.0
    ok("validation, additive world: additive == exact (rel diff 0.0) -> verdict_status 'test'")
    assert rep["decision"]["status"].startswith("PARTIAL") and rep["decision"]["scientific_finding"] is None
    ok("synthetic results are labelled machinery test only (no finding)", independent=False)
    tiles_out = pd.read_csv(SYN / "results_additive/SYNTHETIC_h5_tile_outcomes.csv")
    assert math.isclose(tiles_out["S_additive_w_dm2_38_IMBAT"].sum(), S_t, rel_tol=1e-9)
    so = pd.read_parquet(SYN / "results_additive/SYNTHETIC_h5_seg_outcomes.parquet")
    assert len(so) == 800 and so["in_S"].sum() == len(s)
    ok("per-segment (800 rows) and per-tile outcome tables sum to the S outcome")

    print("\n--- A11: evaluation items ---")
    x = rep["a11"]["days"]["IMBAT"]
    b_ = tw["m2_38_base_IMBAT"]
    assert math.isclose(x["baseline_mean_m2_38"]["S"], round(float(b_[s["seg_id"]].mean()), 4), abs_tol=1e-4)
    assert math.isclose(x["share_m2_38_zero"]["S"], round(float((b_[s["seg_id"]] == 0).mean()), 4), abs_tol=1e-4)
    zm = z.groupby("draw_id")["seg_id"].apply(lambda v: b_[v].mean())
    z0 = z.groupby("draw_id")["seg_id"].apply(lambda v: (b_[v] == 0).mean())
    assert math.isclose(x["baseline_mean_m2_38"]["Z_per_draw"]["median"], round(float(zm.median()), 3), abs_tol=1e-3)
    assert math.isclose(x["share_m2_38_zero"]["Z_per_draw"]["median"], round(float(z0.median()), 3), abs_tol=1e-3)
    assert math.isclose(x["share_dm2_38_zero_in_P"]["all_segments"], round(float((dm_t == 0).mean()), 4), abs_tol=1e-4)
    for name, k in (("cell_h38", 0), ("mean_utci_b15", 1)):
        d_t = pd.Series({i: w.cell_rec["baseline"]["IMBAT"][i][k] - w.cell_rec["P"]["IMBAT"][i][k] for i in tw.index})
        S_c = float(sum(d_t[i] * wmap[tw.loc[i, "g250_id"]] for i in s["seg_id"]))
        Z_c = z.assign(c=lambda v: [d_t[i] * wmap[g] * f for i, g, f in zip(v.seg_id, v.g250_id, v.prorate_frac)]) \
            .groupby("draw_id")["c"].sum()
        r_c = (S_c / Z_c).to_numpy()
        got = x["companions"][name]["ratio_S_vs_Z"]
        assert math.isclose(got["median"], round(float(np.median(r_c)), 3), abs_tol=6e-4), (name, got, np.median(r_c))
        assert math.isclose(got["ci_95"][0], round(float(np.percentile(r_c, 2.5)), 3), abs_tol=6e-4)
        assert math.isclose(got["ci_95"][1], round(float(np.percentile(r_c, 97.5)), 3), abs_tol=6e-4)
        assert np.allclose(per.set_index("draw_id")[f"delta_weighted_{name}_z"], Z_c, rtol=1e-6)
        print(f"    {name}: S/Z median {got['median']} {got['ci_95']}")
    ok(f"A11 report items equal truth: mean M2_38 S {x['baseline_mean_m2_38']['S']} / Z "
       f"{x['baseline_mean_m2_38']['Z_per_draw']['median']}; M2_38 = 0 S {x['share_m2_38_zero']['S']} / Z "
       f"{x['share_m2_38_zero']['Z_per_draw']['median']}; dM2 = 0 in P {x['share_dm2_38_zero_in_P']['all_segments']}; "
       f"both companion ratios (median, 95 % range)")
    tam = SYN / "P_tampered_companions"
    tam.mkdir(exist_ok=True)
    for d in DAYS:
        e = pd.read_parquet(SYN / f"P/SYNTHETIC_seg_effects_{d}.parquet")
        e["dcell_h38"] = -5.0 * e["dcell_h38"] + 1.0
        e["dmean_b15"] = -5.0 * e["dmean_b15"] + 1.0
        e.to_parquet(tam / f"SYNTHETIC_seg_effects_{d}.parquet", index=False)
    rt = EV.evaluate_h5(seldir, tam, worlds["additive"][0], SYN / "SYNTHETIC_grid_250.gpkg", sp, SYN / "results_tampered")
    assert rt["targeting_ratio_S_vs_Z"] == rep["targeting_ratio_S_vs_Z"] and rt["verdict"] == rep["verdict"]
    assert rt["decision"] == rep["decision"] and rt["verdict_status"] == rep["verdict_status"]
    assert rt["validation"] == rep["validation"] and rt["variants"] == rep["variants"]
    assert rt["a11"]["days"]["IMBAT"]["companions"] != rep["a11"]["days"]["IMBAT"]["companions"]
    ok("tampering with both companions changes the A11 numbers and leaves ratio, verdict, decision and validation unchanged")
    old = SYN / "P_without_a11"
    old.mkdir(exist_ok=True)
    for d in DAYS:
        pd.read_parquet(SYN / f"P/SYNTHETIC_seg_effects_{d}.parquet").drop(columns=["dcell_h38"]).to_parquet(
            old / f"SYNTHETIC_seg_effects_{d}.parquet", index=False)
    try:
        EV.evaluate_h5(seldir, old, worlds["additive"][0], SYN / "SYNTHETIC_grid_250.gpkg", sp, SYN / "results_fail")
        raise AssertionError("effects without A11 columns were accepted")
    except RuntimeError as ex:
        print(f"    stopped as required: {str(ex)[:90]}")
    ok("evaluation stops on effects tables without the A11 columns")

    from scripts.street.figures import build_fig8_h5_targeting as F8
    shutil.copy(sp, SYN / "results_additive" / sp.name)          # Z geometry for the figure loader (synthetic ids)
    rs = F8.H5Results(SYN / "results_additive", False)
    fd = F8.load(rs)
    rule, chk = F8.crosscheck(rs, fd)
    assert rule == rep["decision"]["threshold_condition_met"] and fd["n_draws"] == 20
    ok(f"Figure 8 loader + crosscheck accept the new outputs (ratio {chk['median']:.3f}, Jaccard "
       f"{chk['jaccard_recomputed']:.4f}, N, S value, per-draw table)")

    print("\n--- evaluate_h5 (non-additive interaction world) ---")
    rep2 = EV.evaluate_h5(seldir, SYN / "P", worlds["interact"][0], SYN / "SYNTHETIC_grid_250.gpkg", sp,
                          SYN / "results_interact")
    vals_i = worlds["interact"][1]["IMBAT"]
    on_v = w.seg["tile_id"].isin(vt).to_numpy()
    dm_exact = m2(w.base["IMBAT"], 38) - m2(vals_i, 38)
    wv = w.seg["g250_id"].map(wmap).to_numpy()
    exact_t = float((dm_exact * wv)[on_v].sum())
    add_t = float((dm_t.reindex(w.seg["seg_id"]).to_numpy() * wv)[on_v & in_s].sum())
    rel_t = abs(add_t - exact_t) / abs(exact_t)
    assert rel_t > 0.15, rel_t
    assert math.isclose(rep2["validation"]["rel_diff"], round(rel_t, 4), abs_tol=1e-4)
    assert rep2["verdict_status"] == "descriptive only"
    ok(f"validation, interaction world (S trees alone give half the cooling): rel diff {rel_t:.4f} (truth) > 0.15 "
       f"-> verdict_status 'descriptive only'")

    print("\n--- failure modes ---")
    bad = SYN / "P_partial"
    bad.mkdir(exist_ok=True)
    for d in DAYS:
        e = pd.read_parquet(SYN / f"P/SYNTHETIC_seg_effects_{d}.parquet")
        e = e[e.tile_id != w.tiles[0]].assign(partial_run=True, n_tiles_extracted=len(w.tiles) - 1)
        e.to_parquet(bad / f"SYNTHETIC_seg_effects_{d}.parquet", index=False)
    for pdir, sdir, what in ((bad, worlds["additive"][0], "partial P"), (SYN / "P", SYN / "S_exact_missing",
                                                                         "missing S_exact")):
        try:
            EV.evaluate_h5(seldir, pdir, sdir, SYN / "SYNTHETIC_grid_250.gpkg", sp, SYN / "results_fail")
            raise AssertionError(f"evaluate_h5 did not stop on {what}")
        except RuntimeError as ex:
            print(f"    stopped as required ({what}): {ex}")
    ok("evaluation stops on partial P effects and on missing S_exact effects")
    try:
        SEL.select_scenarios(SYN / "SYNTHETIC_segments.gpkg", SYN / "SYNTHETIC_seg_metrics_IMBAT.parquet",
                             PROJECT_ROOT / "data/street/h5/selection_rule6_probe")
        raise AssertionError("synthetic selection was allowed into a non-SYNTHETIC folder")
    except RuntimeError:
        pass
    assert not (PROJECT_ROOT / "data/street/h5/selection_rule6_probe").exists()
    ok("rule 6: synthetic selection refused outside a SYNTHETIC folder (nothing written)")

    if parity_tile:
        test_utci_parity_real_baseline(parity_tile)

    (SYN / "SYNTHETIC_test_summary.json").write_text(json.dumps(
        {"is_synthetic": True, "passed": PASSED, "S_truth": S_t, "ratio_truth_median": r_t,
         "validation_rel_diff_interaction_truth": rel_t}, indent=2), encoding="utf-8")
    print(f"\n=======================================================\n  ALL {len(PASSED)} CHECKS PASSED\n"
          f"=======================================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-real-baseline", default=None, metavar="TILE",
                    help="also check the scenario UTCI code against the stored BASELINE utci_street of a real tile")
    a = ap.parse_args()
    run_chain(a.parity_real_baseline)
