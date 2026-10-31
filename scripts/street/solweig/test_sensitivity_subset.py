"""Checks for run_sensitivity_subset.py. No GPU: torch and solweig_gpu are replaced by stand-ins. Read-only on real data.

  1. met files: the variants differ from the main-run files only where they should.
  2. h26 / h34 DSM method: with roof_z_primary it must reproduce the stored building_dsm on all 40 tiles; h26 <= primary
     <= h34 on footprints (independent plausibility).
  3. UTCI parity: the variant UTCI code reproduces the stored baseline utci_street exactly on real tiles, both days.
  4. Mock end-to-end run of each variant on one real tile, written under outputs/street/SYNTHETIC/sens_mock/:
     the stand-in solweig_gpu behaves like the real one where the old defects were (os.listdir on prep/walls, missing
     SVF), and "simulates" by returning the stored BASELINE Tmrt/shadow/SVF. So the real staging, cropping, UTCI and
     extraction code must rebuild the stored baseline utci_street and the seg_metrics rows. This tests the plumbing,
     not physics: no sensitivity result exists or is computed here.
  5. Resume: a second run makes no SOLWEIG call; removing one output reruns only that tile; --force reruns everything.

The stand-in outputs are a MOCK (rule 6): SYNTHETIC folder, SYNTHETIC_ table names, is_synthetic = True.
"""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.street.solweig import run_sensitivity_subset as RS  # noqa: E402

MOCK = PROJECT_ROOT / "outputs/street/SYNTHETIC/sens_mock"
SOLWEIG = PROJECT_ROOT / "data/street/solweig"
UTCI = PROJECT_ROOT / "data/street/utci"
CALLS: list[tuple] = []
PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  [PASS] {msg}", flush=True)


def install_stand_ins():
    """torch without a GPU, and a solweig_gpu that fails the way the real one does and returns baseline rasters."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
                                       max_memory_allocated=lambda: 0)
    sys.modules["torch"] = torch
    sg = types.ModuleType("solweig_gpu")

    def run_walls_aspect(prep):
        CALLS.append(("walls", prep))
        for d, n in (("walls", "walls_0_0.tif"), ("aspect", "aspect_0_0.tif")):
            (Path(prep) / d).mkdir(exist_ok=True)
            (Path(prep) / d / n).write_bytes(b"")

    def calculate_svf(prep, patch_option=2, overwrite=True):
        assert patch_option == 2
        CALLS.append(("svf", prep))
        tile = sys.modules["solweig_gpu"].current_tile
        with rasterio.open(SOLWEIG / "svf" / f"{tile}.tif") as s:
            inner, prof = s.read(1), s.profile.copy()
        arr = np.full((600, 600), 0.5, np.float32)
        arr[50:550, 50:550] = inner
        t = s.transform
        prof.update(width=600, height=600, transform=rasterio.Affine(t.a, 0, t.c - 100, 0, t.e, t.f + 100))
        (Path(prep) / "SVF").mkdir(exist_ok=True)
        with rasterio.open(Path(prep) / "SVF" / "SkyViewFactor_0_0.tif", "w", **prof) as d:
            d.write(arr, 1)

    def run_utci_tiles(base_path, preprocess_dir, selected_date_str, **kw):
        prep = Path(preprocess_dir)
        import os
        for d in ("walls", "aspect"):
            os.listdir(prep / d)                         # the real function raises here when walls were not computed
        assert (prep / "SVF" / "SkyViewFactor_0_0.tif").exists(), "SVF was not computed before the day run"
        day = {"2022-07-05": "IMBAT", "2023-07-26": "CALM"}[selected_date_str]
        tile = sys.modules["solweig_gpu"].current_tile
        CALLS.append(("day", day, (prep / "metfiles/metfile_0_0.txt").read_bytes(),
                      (prep / "Landcover").is_dir(), str(prep)))
        out = Path(base_path) / "output_folder" / "0_0"
        out.mkdir(parents=True, exist_ok=True)
        for name, sub, div in (("TMRT", "tmrt", 1.0), ("UTCI", "utci", 1.0), ("Shadow", "shadow", 100.0)):
            with rasterio.open(SOLWEIG / day / sub / f"{tile}.tif") as s:
                b, t = s.read().astype(np.float32) / div, s.transform
            raw = np.zeros((24, 600, 600), np.float32)
            raw[6:22, 50:550, 50:550] = b
            with rasterio.open(out / f"{name}_0_0.tif", "w", driver="GTiff", width=600, height=600, count=24,
                               dtype="float32", crs="EPSG:5253", transform=rasterio.Affine(t.a, 0, t.c - 100, 0, t.e,
                                                                                            t.f + 100)) as d:
                d.write(raw)

    sg.run_walls_aspect, sg.calculate_svf, sg.run_utci_tiles = run_walls_aspect, calculate_svf, run_utci_tiles
    sg.current_tile = None
    sys.modules["solweig_gpu"] = sg


def test_met():
    print("\n--- 1. met files ---")
    from scripts.street.solweig.run_solweig_citywide import DESIGN_DAYS
    for d in RS.DAYS:
        assert RS.PRIMARY_FORCING[d] == PROJECT_ROOT / DESIGN_DAYS[d]["met_file"], d
        assert RS.PRIMARY_FORCING[d].read_bytes() == (PROJECT_ROOT / DESIGN_DAYS[d]["met_file"]).read_bytes()
        a = pd.read_csv(RS.PRIMARY_FORCING[d], sep=r"\s+")
        b = pd.read_csv(RS.RADSENS_FORCING[d], sep=r"\s+")
        diff = [c for c in a.columns if not a[c].equals(b[c])]
        assert set(diff) <= {"Kdn", "Kdiff", "Kdir", "Wd"}, diff      # Wd: the v2 correction of CALM (A7)
        for v in RS.GPU_VARIANTS:
            assert list(RS.variant_met(v, d)["Td"]) == list(RS.variant_met("h26", d)["Td"])
    ok("h26/h34/landcover use the main-run met files byte for byte; RADSENS_v2 differs only in radiation (and CALM Wd)")


def test_dsm():
    print("\n--- 2. building DSM method (all 40 tiles) ---")
    worst, ncell, lost_lo, gain_hi, nfoot = 0.0, 0, 0, 0, 0
    for t in RS.SUBSET_40_TILES:
        with rasterio.open(RS.TILES_2M / "building_dsm" / f"{t}.tif") as s:
            st = s.read(1)
        with rasterio.open(RS.TILES_2M / "dem" / f"{t}.tif") as s:
            dem = s.read(1)
        d = RS.rasterize_building_dsm(t, "roof_z_primary")
        m = dem != RS.NODATA
        dd = np.abs(d - st)[m]
        worst = max(worst, float(dd.max()))
        ncell += int((dd > 0.01).sum())
        assert np.array_equal(st == RS.NODATA, d == RS.NODATA), t
        low = RS.rasterize_building_dsm(t, "roof_z_h26")
        high = RS.rasterize_building_dsm(t, "roof_z_h34")
        foot = (d - dem > 0.5)
        flo, fhi = low - dem > 0.5, high - dem > 0.5
        assert not (flo & ~foot).any() and not (foot & ~fhi).any(), t        # lower roofs cannot create building cells
        lost_lo += int((foot & ~flo).sum())
        gain_hi += int((fhi & ~foot).sum())
        nfoot += int(foot.sum())
        assert np.all(low[foot] <= d[foot] + 1e-3) and np.all(high[foot] >= d[foot] - 1e-3), t
    print(f"    max |method - stored| = {worst:.4f} m; cells differing by > 1 cm: {ncell}")
    print(f"    footprint cells {nfoot}: h26 loses {lost_lo} ({lost_lo / nfoot * 100:.3f} %), h34 gains {gain_hi} "
          f"({gain_hi / nfoot * 100:.3f} %): roofs crossing the hillside DEM, physical")
    assert worst < 0.01 and ncell == 0
    ok("with roof_z_primary the on-the-fly DSM equals the stored building_dsm on 40/40 tiles; h26 <= primary <= h34")


def test_utci_parity(tiles):
    print("\n--- 3. UTCI parity with the stored baseline ---")
    for t in tiles:
        for d in RS.DAYS:
            us, uu = RS.tile_utci_arrays(t, SOLWEIG / d / "tmrt" / f"{t}.tif", RS.variant_met("h26", d))
            with rasterio.open(UTCI / d / "utci_street" / f"{t}.tif") as s:
                ref = s.read()
            with rasterio.open(UTCI / d / "utci_uniform" / f"{t}.tif") as s:
                ref_u = s.read()
            a, b = float(np.abs(us.astype(float) - ref.astype(float)).max()), \
                float(np.abs(uu.astype(float) - ref_u.astype(float)).max())
            print(f"    {t} {d}: max |diff| street {a:.2e}, uniform {b:.2e}")
            assert a == 0.0 and b == 0.0, (t, d, a, b)
    ok(f"UTCI code reproduces stored utci_street and utci_uniform exactly on {len(tiles)} real tiles, both days")


def seg_ref(day, ids):
    r = pd.read_parquet(PROJECT_ROOT / f"data/street/analysis/seg_metrics_{day}.parquet")
    return r[r["seg_id"].isin(ids)].set_index("seg_id").sort_index()


def test_mock_runs(tile):
    print(f"\n--- 4. mock end-to-end, tile {tile} ---")
    if MOCK.exists():
        shutil.rmtree(MOCK)
    sys.modules["solweig_gpu"].current_tile = tile
    root, scratch = MOCK / "SYNTHETIC_mock", MOCK / "SYNTHETIC_scratch"
    for v, cfg in RS.GPU_VARIANTS.items():
        CALLS.clear()
        RS.run_variant(v, [tile], out_root=root, scratch_root=scratch)
        order = [c[0] for c in CALLS]
        assert order == ["walls", "svf", "day", "day"], order
        for c, d in zip(CALLS[2:], RS.DAYS):
            assert c[1] == d and c[2] == cfg["forcing"][d].read_bytes(), (v, d)
            assert c[3] == cfg["use_landcover"], v
        vd = root / v
        for d in RS.DAYS:
            with rasterio.open(vd / "utci" / d / "utci_street" / f"{tile}.tif") as s, \
                    rasterio.open(UTCI / d / "utci_street" / f"{tile}.tif") as r:
                assert np.array_equal(s.read(), r.read()), (v, d)
                assert s.transform == r.transform and s.crs == r.crs
            t = pd.read_parquet(vd / f"SYNTHETIC_seg_metrics_{d}.parquet").set_index("seg_id").sort_index()
            assert t["is_synthetic"].all() and t["partial_run"].all() and (t["n_tiles_at_extraction"] == 1).all()
            ref = seg_ref(d, t.index)
            assert len(ref) == len(t) and len(t) > 0
            num = [c for c in ref.columns if c not in ("partial_run", "n_tiles_at_extraction")]
            for c in num:
                assert np.allclose(t[c].astype(float), ref[c].astype(float), equal_nan=True, atol=1e-6), (v, d, c)
        ok(f"{v}: walls -> SVF -> 2 days, right met file and Landcover staging; UTCI identical to baseline; "
           f"{len(t)} segment rows equal seg_metrics on all {len(num)} columns (both days)")
    assert not (RS.SENS_DIR / "radsens").exists() and not (RS.ANALYSIS_SENS / "radsens").exists()
    ok("mock runs wrote nothing into data/street/sens or data/street/analysis/sens")
    return root, scratch


def test_resume(root, scratch, tile):
    print("\n--- 5. resume ---")
    CALLS.clear()
    RS.run_variant("h26", [tile], out_root=root, scratch_root=scratch)
    assert not CALLS, CALLS
    ok("second run makes no SOLWEIG call")
    (root / "h26/solweig/CALM/tmrt" / f"{tile}.tif").unlink()
    v = RS.validate_variant("h26", [tile], root / "h26")
    assert not v["complete"], v
    ok("validate reports the removed output as incomplete")
    CALLS.clear()
    RS.run_variant("h26", [tile], out_root=root, scratch_root=scratch)
    assert [c[0] for c in CALLS] == ["walls", "svf", "day"] and CALLS[2][1] == "CALM", [c[:2] for c in CALLS]
    assert RS.validate_variant("h26", [tile], root / "h26")["complete"]
    ok("removing one output reruns only that day (walls + SVF + CALM), and validation passes again")
    CALLS.clear()
    RS.run_variant("h26", [tile], force=True, out_root=root, scratch_root=scratch)
    assert [c[0] for c in CALLS] == ["walls", "svf", "day", "day"]
    ok("--force recomputes both days")


def test_pid_reader():
    print("\n--- 6. PID file encodings ---")
    MOCK.mkdir(parents=True, exist_ok=True)
    for enc, raw in (("ascii", b"31644\r\n"), ("utf8bom", b"\xef\xbb\xbf31796\r\n"),
                     ("utf16", "31644\r\n".encode("utf-16"))):
        p = MOCK / f"SYNTHETIC_{enc}.pid"
        p.write_bytes(raw)
        assert RS.read_pid_file(p) in (31644, 31796), enc
    ok("PID reader handles ASCII, UTF-8 BOM and UTF-16 (PowerShell Out-File)")


if __name__ == "__main__":
    install_stand_ins()
    test_met()
    test_dsm()
    test_utci_parity(["T_0505_4250", "T_0503_4262", "T_0512_4247", "T_0489_4249"])
    r, sc = test_mock_runs("T_0505_4250")
    test_resume(r, sc, "T_0505_4250")
    test_pid_reader()
    print(f"\nALL {len(PASSED)} CHECKS PASSED")
