"""SYNTHETIC test of run_all.py --gpu-sens (A7 item 6, A9): the GPU variants are compared with the primary on the same tiles.

Uses the existing planted H1 test tables (outputs/street/analysis/h1/synthetic_inputs/*_H1T). Writes under
outputs/street/SYNTHETIC/sens_stats_test/ and outputs/street/analysis/{h1,h2}/synthetic_SENS_<variant>/. No real table is read
as an input to a statistic, and no real result is computed.

Planted: 6 tiles form the "subset"; each variant = the primary on those tiles with a known per-segment change. Checks that
could fail: primary40 holds exactly those tiles; a primary-vs-all-tiles comparison would give a different reference; the paired
table recovers the planted deltas; a missing tile, a differing seg_id set and a missing landcover table each stop the run;
the landcover folder with inputs only no longer falls back to the primary tables; H1 runs with --ladder-source segments.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_all as RA  # noqa: E402

ROOT = HERE.parents[2]
IN = ROOT / "outputs/street/analysis/h1/synthetic_inputs"
BASE = ROOT / "outputs/street/SYNTHETIC/sens_stats_test"
PLANT = {"radsens": 1.0, "h26": -0.5, "h34": 0.25, "landcover": 0.75}
PASSED = []


def ok(m):
    PASSED.append(m)
    print(f"  [PASS] {m}", flush=True)


def stops(fn, what):
    try:
        fn()
    except (RuntimeError, FileNotFoundError) as e:
        print(f"    stopped as required ({what}): {str(e)[:110]}")
        return
    raise AssertionError(f"did not stop: {what}")


def main():
    if BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True)
    seg_path = IN / "segments_SYNTHETIC_H1T.parquet"
    seg = pd.read_parquet(seg_path)
    tiles = sorted(seg["tile_id"].value_counts().head(60).index)[:6]
    (BASE / "subset").mkdir()
    csv = BASE / "subset" / "SYNTHETIC_subset.csv"
    pd.DataFrame({"tile_id": tiles}).to_csv(csv, index=False)
    prim = {d: IN / f"seg_metrics_{d}_SYNTHETIC_H1T.parquet" for d in RA.DAYS}
    sens = BASE / "sens"
    ids = set(seg.loc[seg["tile_id"].isin(tiles), "seg_id"])

    n = RA.build_primary40(sens, prim, seg_path, tiles)
    for d in RA.DAYS:
        t = pd.read_parquet(sens / "primary40" / f"seg_metrics_{d}.parquet")
        assert set(t["seg_id"]) == ids and t["partial_run"].all() and (t["n_tiles_at_extraction"] == 6).all()
    assert len(ids) < len(seg) / 10
    ok(f"primary40 = exactly the {len(ids)} segments of the 6 subset tiles ({len(seg)} in the primary), both days: {n}")
    stops(lambda: RA.build_primary40(BASE / "x", prim, seg_path, tiles + ["T_9999_9999"]), "subset tile not in the primary")

    for v, dx in PLANT.items():
        (sens / v).mkdir()
        for d in RA.DAYS:
            t = pd.read_parquet(sens / "primary40" / f"seg_metrics_{d}.parquet")
            for c in ("utci_b15", "tmrt_b15"):
                t[c] = t[c] + dx * (2.0 if c == "tmrt_b15" else 1.0)
            t.to_parquet(sens / v / f"seg_metrics_{d}.parquet", index=False)

    # a variant with other tiles must stop the comparison
    p = sens / "h34" / "seg_metrics_CALM.parquet"
    good = pd.read_parquet(p)
    good.iloc[1:].to_parquet(p, index=False)
    stops(lambda: RA.check_same_segments(sens, RA.GPU_VARIANTS), "variant with a different seg_id set")
    good.to_parquet(p, index=False)
    assert RA.check_same_segments(sens, RA.GPU_VARIANTS)["IMBAT"] == len(ids)
    ok("a variant whose seg_ids differ from primary40 stops the comparison")

    # landcover: the folder exists with inputs only, as data/street/sens/landcover does today
    orig = RA.HERE
    try:
        RA.HERE = BASE / "a" / "b" / "c"           # resolve_variant_args looks under HERE.parents[2] / data/street/sens
        (BASE / "data/street/sens/landcover/inputs").mkdir(parents=True)   # HERE.parents[2] = BASE
        try:
            RA.resolve_variant_args("landcover", str(BASE / "no_tables"))
            raise AssertionError("landcover with inputs only did not stop")
        except FileNotFoundError as e:
            assert "no seg_metrics_*.parquet in" in str(e), e       # the new check, not the "neither ... nor" branch
            print(f"    stopped as required (landcover folder without tables): {str(e)[:110]}")
    finally:
        RA.HERE = orig
    args = RA.resolve_variant_args("landcover", str(sens))
    assert str(sens / "landcover" / "seg_metrics_IMBAT.parquet") in args and "_landcover" in args
    ok("landcover resolves to its own tables; a folder with inputs only is an error, not a silent run on the primary tables")
    steps = RA.gpu_step_args("radsens", sens, seg_path, False, True)
    assert steps[0][-4:-2] == ["--ladder-source", "segments"] and "--ladder-source" not in steps[1]
    ok("H1 runs with --ladder-source segments on every member of the comparison")

    rc = RA.run_gpu_sens(sens, csv, prim, seg_path, BASE / "out", smoke=False, synthetic=True)
    assert rc == 0, rc
    pr = pd.read_csv(BASE / "out" / "gpu_sens_paired_m1_m3_SYNTHETIC.csv")
    assert len(pr) == 4 * 2 * 3 and (pr["n_segments"] == len(ids)).all()
    for v, dx in PLANT.items():
        q = pr[pr["variant"] == v]
        assert np.allclose(q.loc[q.metric == "M1", ["mean_delta", "median_delta", "p05", "p95"]], dx, atol=1e-5), v
        assert np.allclose(q.loc[q.metric == "M3", "mean_delta"], 2 * dx, atol=1e-5), v
        assert np.allclose(q.loc[q.metric == "M2_38", "mean_delta"], 0.0)
    ok("paired table recovers the planted M1 and M3 changes of all four variants, both days, on identical segments")
    hl = pd.read_csv(BASE / "out" / "gpu_sens_headlines_SYNTHETIC.csv")
    assert set(hl["variant"]) == set(PLANT) and set(hl["day"]) == set(RA.DAYS) and hl["delta"].notna().all()
    ref = json.loads(RA.result_json("h1", "primary40", False, True).read_text(encoding="utf-8"))
    assert ref["results"]["IMBAT"]["n_segments"] == int((pd.read_parquet(prim["IMBAT"])["seg_id"].isin(ids)).sum())
    ok(f"H1/H2 ran on primary40 and all four variants with {ref['results']['IMBAT']['n_segments']} segments; headline table has "
       f"{len(hl)} rows; the reference is the 6-tile primary, not the {len(seg)}-segment run")
    print(f"\nALL {len(PASSED)} CHECKS PASSED")


if __name__ == "__main__":
    main()
