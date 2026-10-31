"""One command to (re)run everything on the REAL analysis tables.

    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe scripts/street/stats/run_all.py

Runs sanity checks, H1, H2, H4 (lock settings: 1000 block-bootstrap resamples), then the LCZ-label sensitivity
(lcz_weak250) for H1 and H2. Steps run sequentially; each uses at most 4 processes. Results carry PARTIAL (n tiles) /
_PARTIAL file suffix automatically while the tables have partial_run = True.
Add --synthetic-tests to run the planted-effect tests first.  Add --skip-sensitivity to skip the weak-label runs.

GPU sensitivities (lock 3, A7 item 6, A9), after run_sensitivity_subset.py has finished:

    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe scripts/street/stats/run_all.py --gpu-sens

radsens, h26, h34 and landcover exist on 40 tiles only, so they are compared with the primary run restricted to the SAME 40
tiles, never with the 427-tile result. --gpu-sens writes data/street/analysis/sens/primary40/ (primary tables cut to the
tiles of subset_40_tiles.csv), stops unless every variant holds exactly the same seg_ids on both days, runs H1 and H2 on
primary40 and on each variant, and writes outputs/street/analysis/sens_gpu/gpu_sens_{headlines,paired_m1_m3}.csv and
gpu_sens_comparison.json. All five H1 runs use --ladder-source segments: raster_ladder reads the PRIMARY rasters, which
would give a variant the ladder of the primary run. The ladder of these runs therefore differs slightly in method from the
427-tile ladder, and is comparable only within this set.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ROOT = HERE.parents[2]
GPU_VARIANTS = ("radsens", "h26", "h34", "landcover")
PRIMARY40 = "primary40"
DAYS = ("IMBAT", "CALM")
OUT_ROOT = ROOT / "outputs" / "street" / "analysis"


def run(args):
    t = time.time()
    print(f"\n>>> {' '.join(args)}", flush=True)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(k, "4")                      # WP3.7: never more than 4 CPU threads/processes per step
    r = subprocess.run([PY, *args], cwd=str(HERE.parents[2]), env=env)
    print(f"<<< exit {r.returncode} in {time.time() - t:.0f}s", flush=True)
    return r.returncode


def resolve_variant_args(variant: str, sens_root: str | None = None) -> list[str]:
    """Resolve CLI arguments for H1/H2 sensitivity reruns (lock section 3, WP3.6; 'nomotorway' = A8 item 3).

    Variants with extraction products (ps10, ps20, p90, lcz_filter, nomotorway, radsens, h26, h34, landcover, primary40)
    need their data/street/analysis/sens/<variant>/ (or --sens-root) tables; a missing table is an ERROR, never a silent
    fallback to the primary tables.
    """
    root = HERE.parents[2]
    sens_analysis_dir = Path(sens_root) if sens_root else root / "data" / "street" / "analysis" / "sens"
    if not sens_analysis_dir.is_absolute():
        sens_analysis_dir = root / sens_analysis_dir
    sens_gpu_dir = root / "data" / "street" / "sens"

    if variant in ("weak250", "lcz_weak250"):
        return ["--lcz-col", "lcz_weak250", "--out-suffix", "_weak250"]

    args = ["--out-suffix", f"_{variant}"]

    # 1. Check data/street/analysis/sens/<variant>/
    var_dir = sens_analysis_dir / variant
    if var_dir.exists():
        imbat = var_dir / "seg_metrics_IMBAT.parquet"
        calm = var_dir / "seg_metrics_CALM.parquet"
        if imbat.exists():
            args += ["--imbat-path", str(imbat)]
        if calm.exists():
            args += ["--calm-path", str(calm)]
        for seg_cand in [
            var_dir / f"segments_{variant}.gpkg",
            sens_analysis_dir / f"segments_{variant}.gpkg",
        ]:
            if seg_cand.exists():
                args += ["--seg-path", str(seg_cand)]
                break
        if not any(a in args for a in ("--imbat-path", "--calm-path")):
            raise FileNotFoundError(f"variant '{variant}': no seg_metrics_*.parquet in {var_dir}")
        return args

    # 2. Check data/street/sens/<variant>/ (GPU sensitivity variants: radsens, h26, h34, landcover)
    gpu_var_dir = sens_gpu_dir / variant
    if gpu_var_dir.exists():
        imbat = gpu_var_dir / "seg_metrics_IMBAT.parquet"
        calm = gpu_var_dir / "seg_metrics_CALM.parquet"
        if imbat.exists():
            args += ["--imbat-path", str(imbat)]
        if calm.exists():
            args += ["--calm-path", str(calm)]
        for seg_cand in [
            gpu_var_dir / f"segments_{variant}.gpkg",
            sens_analysis_dir / f"segments_{variant}.gpkg",
        ]:
            if seg_cand.exists():
                args += ["--seg-path", str(seg_cand)]
                break
        # the folder alone proves nothing (data/street/sens/landcover/ holds the input rasters before any run)
        if not any(a in args for a in ("--imbat-path", "--calm-path")):
            raise FileNotFoundError(f"variant '{variant}': no seg_metrics_*.parquet in {gpu_var_dir}")
        return args

    raise FileNotFoundError(f"variant '{variant}': neither {var_dir} nor {gpu_var_dir} exists; run the extraction first")


def _tile_of(seg_path: Path) -> dict:
    import pandas as pd
    p = Path(seg_path)
    if p.suffix == ".parquet":
        t = pd.read_parquet(p, columns=["seg_id", "tile_id"])
    else:
        import geopandas as gpd
        t = gpd.read_file(p, layer="segments", columns=["seg_id", "tile_id"], ignore_geometry=True)
    return dict(zip(t["seg_id"], t["tile_id"]))


def build_primary40(sens_dir: Path, primary_paths: dict, seg_path: Path, tiles: list[str]) -> dict:
    """Primary seg_metrics cut to the sensitivity tiles (<sens_dir>/primary40/). Stops if a tile is not in the primary run."""
    import pandas as pd
    tile_of = _tile_of(seg_path)
    out_dir = Path(sens_dir) / PRIMARY40
    out_dir.mkdir(parents=True, exist_ok=True)
    n = {}
    for d in DAYS:
        m = pd.read_parquet(primary_paths[d])
        t = m["seg_id"].map(tile_of)
        miss = sorted(set(tiles) - set(t.dropna()))
        if miss:
            raise RuntimeError(f"primary {d} has no segments for {len(miss)} of the {len(tiles)} tiles, e.g. {miss[:5]}: "
                               "wait for the complete run")
        sub = m[t.isin(tiles)].sort_values("seg_id").reset_index(drop=True)
        sub["partial_run"] = True                     # a subset of the primary run, labelled like the variants
        sub["n_tiles_at_extraction"] = len(tiles)
        tmp = out_dir / f"seg_metrics_{d}.parquet.tmp"
        sub.to_parquet(tmp, index=False)
        os.replace(tmp, out_dir / f"seg_metrics_{d}.parquet")
        n[d] = int(len(sub))
    return n


def check_same_segments(sens_dir: Path, variants) -> dict:
    """Every variant must hold exactly the seg_ids of primary40, per day (same tiles, same PS cells)."""
    import pandas as pd
    out = {}
    for d in DAYS:
        ref = set(pd.read_parquet(Path(sens_dir) / PRIMARY40 / f"seg_metrics_{d}.parquet", columns=["seg_id"])["seg_id"])
        for v in variants:
            p = Path(sens_dir) / v / f"seg_metrics_{d}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"variant '{v}': missing {p}; run run_sensitivity_subset.py first")
            got = set(pd.read_parquet(p, columns=["seg_id"])["seg_id"])
            if got != ref:
                raise RuntimeError(f"variant '{v}' {d}: seg_ids differ from primary40 ({len(got - ref)} extra, "
                                   f"{len(ref - got)} missing); the comparison would not use the same tiles")
        out[d] = len(ref)
    return out


def gpu_step_args(v: str, sens_dir: Path, seg_path, smoke: bool, synthetic: bool) -> list[list[str]]:
    """H1 and H2 argument lists for one member of the GPU comparison. Synthetic runs use --tag to keep their folders apart."""
    h1x, h2x = [], []
    if smoke or synthetic:
        h1x, h2x = ["--n-boot", "40"], ["--n-boot", "40", "--pdp-boot", "20"]
    if synthetic:
        var = ["--synthetic", "--tag", f"_SENS_{v}", "--seg-path", str(seg_path),
               "--imbat-path", str(Path(sens_dir) / v / "seg_metrics_IMBAT.parquet"),
               "--calm-path", str(Path(sens_dir) / v / "seg_metrics_CALM.parquet")]
    else:
        var = resolve_variant_args(v, str(sens_dir))
        if smoke:
            var[var.index("--out-suffix") + 1] += "_smoke"
    return [[str(HERE / "h1_scale.py"), *var, "--ladder-source", "segments", *h1x],
            [str(HERE / "h2_drivers.py"), *var, *h2x]]


def result_json(h: str, v: str, smoke: bool, synthetic: bool) -> Path:
    if synthetic:
        return OUT_ROOT / h / f"synthetic_SENS_{v}" / f"{h}_results_SYNTHETIC.json"
    return OUT_ROOT / h / f"results_{v}{'_smoke' if smoke else ''}" / f"{h}_results_PARTIAL.json"


def headlines(h1: dict, h2: dict, day: str) -> dict:
    """The H1 / H2 headline numbers of the lock (section 3)."""
    r1, r2 = h1["results"][day], h2["results"][day]
    a = r2["model_A_features_only"]
    out = {"h1_resid_share": r1["lmm"]["shares"]["resid_share"], "h1_cv_r2_lcz_only": r1["cv"]["r2_lcz_only"],
           "h1_cv_r2_morphology": r1["cv"]["r2_morphology_lgbm"], "h1_eta2_segment": r1["ladder_eta2"]["segment"]["eta2"],
           "h2_cv_r2_features": a["cv_r2"], "h2_cv_r2_features_plus_lcz": r2["model_B_features_plus_LCZ"]["cv_r2"]}
    out.update({f"h2_share_{k}": v["share"] for k, v in a["family_share"].items()})
    return out


def compare_gpu(sens_dir: Path, variants, out_dir: Path, smoke: bool, synthetic: bool) -> dict:
    """Variant minus primary40: H1 / H2 headlines, and paired per-segment M1, M3, M2_38 on identical seg_ids."""
    import json
    import numpy as np
    import pandas as pd
    tag = "_SYNTHETIC" if synthetic else ""
    load = lambda h, v: json.loads(result_json(h, v, smoke, synthetic).read_text(encoding="utf-8"))  # noqa: E731
    h1p, h2p = load("h1", PRIMARY40), load("h2", PRIMARY40)
    rows, paired, verdicts = [], [], {PRIMARY40: h1p["decision"]["H1"]}
    for v in variants:
        h1v, h2v = load("h1", v), load("h2", v)
        verdicts[v] = h1v["decision"]["H1"]
        for d in DAYS:
            a, b = headlines(h1p, h2p, d), headlines(h1v, h2v, d)
            rows += [dict(variant=v, day=d, metric=k, primary40=a[k], value=b.get(k), delta=None if b.get(k) is None else
                          b[k] - a[k]) for k in a]
            x = pd.read_parquet(Path(sens_dir) / PRIMARY40 / f"seg_metrics_{d}.parquet").set_index("seg_id")
            y = pd.read_parquet(Path(sens_dir) / v / f"seg_metrics_{d}.parquet").set_index("seg_id").loc[x.index]
            for name, col in (("M1", "utci_b15"), ("M3", "tmrt_b15"), ("M2_38", "m2_vstrong")):
                df = (y[col] - x[col]).dropna().to_numpy(float)
                paired.append(dict(variant=v, day=d, metric=name, n_segments=int(df.size), mean_delta=float(df.mean()),
                                   median_delta=float(np.median(df)), p05=float(np.percentile(df, 5)),
                                   p95=float(np.percentile(df, 95)), primary40_mean=float(x[col].mean())))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"gpu_sens_headlines{tag}.csv", index=False)
    pd.DataFrame(paired).to_csv(Path(out_dir) / f"gpu_sens_paired_m1_m3{tag}.csv", index=False)
    res = {"is_synthetic": synthetic, "reference": "primary run restricted to the same 40 tiles (primary40)",
           "ladder_source": "segments, all five runs", "h1_verdict_on_40_tiles": verdicts, "n_boot_smoke": bool(smoke)}
    (Path(out_dir) / f"gpu_sens_comparison{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def run_gpu_sens(sens_dir: Path, subset_csv: Path, primary_paths: dict, seg_path: Path, out_dir: Path, variants=GPU_VARIANTS,
                 smoke: bool = False, synthetic: bool = False) -> int:
    import pandas as pd
    tiles = pd.read_csv(subset_csv)["tile_id"].astype(str).tolist()
    print("primary40 segments:", build_primary40(sens_dir, primary_paths, seg_path, tiles))
    print("same seg_ids confirmed:", check_same_segments(sens_dir, variants))
    rc = []
    for v in (PRIMARY40, *variants):
        rc += [run(x) for x in gpu_step_args(v, sens_dir, seg_path, smoke, synthetic)]
    print("\nexit codes:", rc)
    if max(rc):
        return max(rc)
    print(compare_gpu(sens_dir, variants, out_dir, smoke, synthetic))
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-tests", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--variant", default=None,
                    help="Run H1/H2 headline numbers on a sensitivity variant (e.g. ps10, ps20, p90, lcz_filter, nomotorway, "
                         "weak250). For radsens, h26, h34, landcover use --gpu-sens, which compares on the same 40 tiles")
    ap.add_argument("--smoke", action="store_true",
                    help="with --variant: 40 bootstrap resamples / 20 PDP refits and results folder suffix _smoke "
                         "(end-to-end test of the hook only; NOT lock settings)")
    ap.add_argument("--gpu-sens", action="store_true",
                    help="H1/H2 on primary40, radsens, h26, h34, landcover, on the SAME 40 tiles, plus the comparison tables")
    ap.add_argument("--sens-root", default=None,
                    help="root folder holding <variant>/seg_metrics_<DAY>.parquet and segments_<variant>.gpkg "
                         "(default data/street/analysis/sens); used for tests")
    a = ap.parse_args()
    s = lambda n: str(HERE / n)  # noqa: E731

    if a.gpu_sens:
        sens_dir = Path(a.sens_root) if a.sens_root else ROOT / "data" / "street" / "analysis" / "sens"
        an = ROOT / "data" / "street" / "analysis"
        sys.exit(run_gpu_sens(sens_dir, sens_dir / "subset_40_tiles.csv",
                              {d: an / f"seg_metrics_{d}.parquet" for d in DAYS}, an / "segments.gpkg",
                              OUT_ROOT / ("sens_gpu_smoke" if a.smoke else "sens_gpu"), smoke=a.smoke))

    if a.variant:
        var_args = resolve_variant_args(a.variant, a.sens_root)
        print(f"Running H1 and H2 on sensitivity variant '{a.variant}' with arguments: {var_args}")
        h1x, h2x = [], []
        if a.variant in (*GPU_VARIANTS, PRIMARY40):   # raster_ladder would read the primary rasters for a GPU variant
            h1x = ["--ladder-source", "segments"]
        if a.smoke:
            i = var_args.index("--out-suffix")
            var_args[i + 1] += "_smoke"
            h1x, h2x = [*h1x, "--n-boot", "40"], ["--n-boot", "40", "--pdp-boot", "20"]
        steps = [
            [s("h1_scale.py"), *var_args, *h1x],
            [s("h2_drivers.py"), *var_args, *h2x],
        ]
        rc = [run(x) for x in steps]
        print("\nexit codes:", rc)
        sys.exit(max(rc))

    steps = []
    if a.synthetic_tests:
        steps.append([s("test_synthetic_recovery.py"), "--n-boot", "40"])
    steps += [[s("sanity_checks_real.py")], [s("h1_scale.py")], [s("h4_shade_breeze.py")], [s("h2_drivers.py")]]
    if not a.skip_sensitivity:
        steps += [[s("h1_scale.py"), "--lcz-col", "lcz_weak250", "--out-suffix", "_weak250"],
                  [s("h2_drivers.py"), "--lcz-col", "lcz_weak250", "--out-suffix", "_weak250"]]
    rc = [run(x) for x in steps]
    print("\nexit codes:", rc)
    sys.exit(max(rc))


if __name__ == "__main__":
    main()
