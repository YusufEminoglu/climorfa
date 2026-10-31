"""CLIMORFA Street — H5 scenario selection (pre-analysis lock §2 H5, amendments A5, A6, A10).

Authoritative: docs/climorfa_street/pre_analysis_lock.md, analysis_schema.md, wp/_RULES.md.

A10 rulings implemented here (rulings 1-3, 5):
  * Budget ranking metric: primary M2_38 = column ``m2_vstrong`` on IMBAT (ruling 1). The locked M2 (``m2_strong``,
    >= 32 degC) is produced with the same machinery as the reported "locked" variant.
  * Ties in the ranking are broken by M1 (``utci_b15``) descending, then ``seg_id`` ascending (ruling 2).
  * Tree budget N = sum of floor(length_m / 8 m) over the hottest 10 % of segment length (the segment that crosses the
    10 % line is included). The capacity rule floor(length / 8 m) is the one written in ruling 3 and is used for S, Z and
    P alike, so a planted segment always carries the same trees in every scenario.
  * Z target class = the LCZ class with the highest mean metric among the classes whose planting capacity
    sum(floor(length_m / 8)) >= N (ruling 3). LCZ 0 is the raster no-data value and is never a class (A6). Classes with
    < 30 segments would be pooled (lock §1) and are not candidates either (cannot bind when capacity >= N in practice).
    The full class table is logged and written.
  * Z draws (A5): per draw, segments of the target class are sampled without replacement with probability proportional
    to length until N trees are absorbed; the last segment is prorated (n_trees < capacity, prorate_frac < 1).
    20 draws, draw k seeded with 20261030 + k (the WP3.5b convention).
  * g250_id is carried into the S and Z tables (WP4.6 open issue 2).
  * S_exact validation tiles (ruling 5): 40 tiles drawn without replacement from the sorted S tile set with
    numpy.random.default_rng(20261030) (primary variant only).

Outputs (default data/street/h5/selection/; a synthetic run must write into a folder whose path contains SYNTHETIC and
prefixes every file with SYNTHETIC_):
  scenario_S_segments[_M2_32].parquet/.gpkg, scenario_Z_draws[_M2_32].parquet, lcz_class_table[_M2_32].csv,
  scenarios_summary[_M2_32].json, s_exact_validation_tiles.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
DEFAULT_SEGMENTS = PROJECT_ROOT / "data/street/analysis/segments.gpkg"
DEFAULT_METRICS = PROJECT_ROOT / "data/street/analysis/seg_metrics_IMBAT.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data/street/h5/selection"
N_TILES_TOTAL = 427
SEED = 20261030
N_DRAWS = 20
N_VALIDATION_TILES = 40
M1_COL = "utci_b15"

# variant key -> (metric column, file suffix, human label)
VARIANTS = {
    "primary": ("m2_vstrong", "", "M2_38 (bands 10-18 with segment-median UTCI_street >= 38 degC), A10 ruling 1"),
    "locked": ("m2_strong", "_M2_32", "M2_32 (locked M2, >= 32 degC), reported variant"),
}


# ---------------------------------------------------------------------------------------------------------------------
# pure functions (unit-tested in test_h5_machinery.py)
# ---------------------------------------------------------------------------------------------------------------------
def capacity_trees(length_m, spacing_m: float = 8.0):
    """Planting capacity floor(length / spacing) (A10 ruling 3)."""
    return np.floor(np.asarray(length_m, dtype=float) / spacing_m).astype(np.int64)


def rank_for_budget(valid: pd.DataFrame, metric_col: str, m1_col: str = M1_COL) -> pd.DataFrame:
    """Sort by metric desc, then M1 desc, then seg_id asc (A10 ruling 2). NaN M1 sorts last within a metric tie."""
    out = valid.sort_values(by=[metric_col, m1_col, "seg_id"], ascending=[False, False, True],
                            na_position="last", kind="mergesort").copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_hottest_share(ranked: pd.DataFrame, target_pct: float, spacing_m: float):
    """Hottest `target_pct` of total length; the segment that crosses the line is included. Returns (S, N, total, target)."""
    total = float(ranked["length_m"].sum())
    target = total * target_pct
    cum = ranked["length_m"].cumsum().to_numpy()
    cut = int(np.searchsorted(cum, target, side="left"))
    s = ranked.iloc[: cut + 1].copy()
    s["cum_length_m"] = cum[: cut + 1]
    s["capacity_trees"] = capacity_trees(s["length_m"], spacing_m)
    s["n_trees"] = s["capacity_trees"]
    s["prorate_frac"] = 1.0
    s["planted_length_m"] = s["length_m"]
    return s, int(s["n_trees"].sum()), total, target


def lcz_class_table(valid: pd.DataFrame, metric_col: str, budget_n: int, spacing_m: float,
                    min_segments: int = 30, nodata_classes=(0,)) -> pd.DataFrame:
    """One row per lcz_primary value with capacity and eligibility (A10 ruling 3, A6)."""
    v = valid.copy()
    v["_cap"] = capacity_trees(v["length_m"], spacing_m)
    g = v.groupby("lcz_primary")
    tab = pd.DataFrame({
        "n_segments": g.size(),
        "length_m": g["length_m"].sum(),
        "capacity_trees": g["_cap"].sum(),
        f"mean_{metric_col}": g[metric_col].mean(),
        f"mean_{M1_COL}": g[M1_COL].mean() if M1_COL in v.columns else np.nan,
    }).reset_index()
    tab["is_nodata_class_A6"] = tab["lcz_primary"].isin(nodata_classes)
    tab["pooled_lt_min_segments"] = tab["n_segments"] < min_segments
    tab["capacity_ge_N"] = tab["capacity_trees"] >= budget_n
    tab["eligible"] = (~tab["is_nodata_class_A6"]) & (~tab["pooled_lt_min_segments"]) & tab["capacity_ge_N"]
    tab = tab.sort_values([f"mean_{metric_col}", "lcz_primary"], ascending=[False, True]).reset_index(drop=True)
    elig = tab[tab["eligible"]]
    tab["is_target"] = False
    if len(elig):
        tab.loc[elig.index[0], "is_target"] = True
    tab["budget_N"] = budget_n
    return tab


def draw_zone(cands: pd.DataFrame, budget_n: int, rng: np.random.Generator) -> pd.DataFrame:
    """A5: sample without replacement, p proportional to length, until N trees are absorbed; last segment prorated.

    `cands` needs seg_id, length_m, capacity_trees. Returns the drawn rows in draw order with n_trees, prorate_frac."""
    n = len(cands)
    p = cands["length_m"].to_numpy(float)
    p = p / p.sum()
    order = rng.choice(n, size=n, replace=False, p=p)
    rows, cur = [], 0
    for k, idx in enumerate(order):
        r = cands.iloc[idx]
        cap = int(r["capacity_trees"])
        take = min(cap, budget_n - cur)
        if take <= 0:
            break
        cur += take
        rows.append(dict(r.to_dict(), order_in_draw=k, n_trees=int(take), prorate_frac=take / cap,
                         planted_length_m=float(r["length_m"]) * take / cap, is_partial=bool(take < cap)))
        if cur >= budget_n:
            break
    out = pd.DataFrame(rows)
    if int(out["n_trees"].sum()) != budget_n:
        raise ValueError(f"draw absorbed {int(out['n_trees'].sum())} trees, expected N={budget_n}")
    return out


def draw_validation_tiles(s_tiles, n_tiles: int = N_VALIDATION_TILES, seed: int = SEED) -> list[str]:
    """A10 ruling 5: `n_tiles` tiles without replacement from the sorted S tile set, default_rng(seed)."""
    pool = sorted(set(s_tiles))
    k = min(n_tiles, len(pool))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]


def jaccard(a: set, b: set, lengths: dict | None = None) -> float:
    u = a | b
    if not u:
        return 0.0
    if lengths is None:
        return len(a & b) / len(u)
    return sum(lengths[s] for s in a & b) / sum(lengths[s] for s in u)


def _stats(x) -> dict:
    x = np.asarray(x, float)
    return {"median": round(float(np.median(x)), 4), "mean": round(float(np.mean(x)), 4),
            "ci_95": [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)],
            "min": round(float(np.min(x)), 4), "max": round(float(np.max(x)), 4), "std": round(float(np.std(x)), 4)}


# ---------------------------------------------------------------------------------------------------------------------
def load_inputs(segments_path: Path, metrics_path: Path):
    if not segments_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"missing input: {segments_path} / {metrics_path}")
    seg = gpd.read_file(segments_path, layer="segments")
    met = pd.read_parquet(metrics_path)
    syn = ("SYNTHETIC" in str(segments_path).upper() or "SYNTHETIC" in str(metrics_path).upper()
           or ("is_synthetic" in seg.columns and bool(seg["is_synthetic"].astype(bool).any()))
           or ("is_synthetic" in met.columns and bool(met["is_synthetic"].astype(bool).any())))
    need_seg = {"seg_id", "tile_id", "lcz_primary", "length_m", "n_ps", "g250_id"}
    need_met = {"seg_id", "m2_strong", "m2_vstrong", M1_COL}
    if need_seg - set(seg.columns) or need_met - set(met.columns):
        raise KeyError(f"missing columns: segments {need_seg - set(seg.columns)}, metrics {need_met - set(met.columns)}")
    partial = bool(met["partial_run"].astype(bool).any()) if "partial_run" in met.columns else True
    n_tiles = int(met["n_tiles_at_extraction"].max()) if "n_tiles_at_extraction" in met.columns else \
        int(seg["tile_id"].nunique())
    return seg, met, syn, partial, n_tiles


def run_variant(merged: gpd.GeoDataFrame, metric_col: str, suffix: str, label: str, prefix: str, out_dir: Path,
                meta: dict, target_pct: float, spacing_m: float, seed_base: int, n_draws: int,
                min_lcz_segments: int, n_validation_tiles: int, write_validation: bool, write: bool = True) -> dict:
    print(f"\n######## Variant {suffix or 'primary'}: {label} ########")
    n_nan = int(merged[metric_col].isna().sum())
    valid = merged[merged[metric_col].notna() & (merged["length_m"] >= 20.0) & (merged["n_ps"] >= 25)].copy()
    print(f"segments with metric: {len(valid)}  (excluded: {n_nan} with NaN {metric_col})")

    ranked = rank_for_budget(valid, metric_col)
    s, budget_n, total_len, target_len = select_hottest_share(ranked, target_pct, spacing_m)
    s["scenario"] = "S"
    s["metric_col"] = metric_col
    s["is_synthetic"] = meta["is_synthetic"]
    s_tiles = sorted(s["tile_id"].unique().tolist())
    tie_at_cut = int((ranked[metric_col] == s[metric_col].iloc[-1]).sum())
    print(f"--- S: {len(s)} segments, {s['length_m'].sum():,.1f} m of {total_len:,.1f} m "
          f"({s['length_m'].sum() / total_len * 100:.2f} %), N = {budget_n} trees, {len(s_tiles)} tiles; "
          f"{metric_col} at the cut = {s[metric_col].iloc[-1]} ({tie_at_cut} segments share it, M1 breaks the tie)")

    tab = lcz_class_table(valid, metric_col, budget_n, spacing_m, min_lcz_segments)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"--- LCZ class table (N = {budget_n}):\n{tab.to_string(index=False)}")
    if not tab["is_target"].any():
        raise ValueError(f"no eligible LCZ class: none has capacity >= N = {budget_n} (A10 ruling 3); see table above")
    trow = tab[tab["is_target"]].iloc[0]
    target_lcz = int(trow["lcz_primary"])
    n_lcz0 = int((valid["lcz_primary"] == 0).sum())
    print(f"--- Z target class: LCZ {target_lcz}, mean {metric_col} = {trow[f'mean_{metric_col}']:.4f}, "
          f"capacity {int(trow['capacity_trees'])} >= N {budget_n}; LCZ 0 segments excluded (A6): {n_lcz0}")

    cands = valid[valid["lcz_primary"] == target_lcz].copy().reset_index(drop=True)
    cands["capacity_trees"] = capacity_trees(cands["length_m"], spacing_m)
    keep = ["seg_id", "tile_id", "district", "lcz_primary", "g250_id", "length_m", "n_ps", metric_col, M1_COL,
            "capacity_trees"]
    cands = cands[[c for c in keep if c in cands.columns]]
    s_ids = set(s["seg_id"])
    lens = dict(zip(valid["seg_id"], valid["length_m"]))
    draws, jc, jl, ntile = [], [], [], []
    for d in range(n_draws):
        rng = np.random.default_rng(seed_base + d)
        z = draw_zone(cands, budget_n, rng)
        z["draw_id"] = d
        z["draw_seed"] = seed_base + d
        z["scenario"] = f"Z_draw_{d:02d}"
        z["metric_col"] = metric_col
        z["is_synthetic"] = meta["is_synthetic"]
        draws.append(z)
        zi = set(z["seg_id"])
        jc.append(jaccard(s_ids, zi))
        jl.append(jaccard(s_ids, zi, lens))
        ntile.append(z["tile_id"].nunique())
    zdf = pd.concat(draws, ignore_index=True)
    n_unique_sets = len({tuple(sorted(g["seg_id"])) for _, g in zdf.groupby("draw_id")})
    if n_draws > 1 and n_unique_sets == 1:
        raise ValueError("A5 violation: all Z draws selected the same segment set")
    z_tiles = sorted(zdf["tile_id"].unique().tolist())
    print(f"--- Z: {n_draws} draws, {len(zdf) / n_draws:.1f} segments/draw, {np.mean(ntile):.1f} tiles/draw, "
          f"{n_unique_sets} distinct segment sets; Jaccard S-Z median {np.median(jc):.4f}")

    vtiles = draw_validation_tiles(s_tiles, n_validation_tiles, seed_base) if write_validation else []
    if write_validation:
        print(f"--- S_exact validation tiles ({len(vtiles)} of {len(s_tiles)} S tiles, seed {seed_base}): {vtiles}")

    summary = {
        **meta,
        "variant": "primary" if not suffix else "locked",
        "variant_label": label,
        "metric_col": metric_col,
        "ranking": f"{metric_col} desc, {M1_COL} desc, seg_id asc (A10 ruling 2)",
        "target_pct": target_pct,
        "tree_spacing_m": spacing_m,
        "capacity_rule": f"floor(length_m / {spacing_m:g})",
        "seed_base": seed_base,
        "draw_seed_rule": "draw k uses default_rng(seed_base + k)",
        "n_draws": n_draws,
        "tree_budget_N": budget_n,
        "n_segments_with_metric": int(len(valid)),
        "n_segments_nan_metric": n_nan,
        "n_segments_lcz0_excluded_A6": n_lcz0,
        "total_segment_length_m": round(total_len, 2),
        "target_top10_length_m": round(target_len, 2),
        "scenario_S": {"n_segments": int(len(s)), "planted_length_m": round(float(s["length_m"].sum()), 2),
                       "n_trees": budget_n, "n_tiles": len(s_tiles), "tiles": s_tiles,
                       "metric_at_cut": float(s[metric_col].iloc[-1]), "n_segments_sharing_cut_metric": tie_at_cut},
        "scenario_Z": {"target_lcz": target_lcz, "target_lcz_mean_metric": round(float(trow[f"mean_{metric_col}"]), 4),
                       "target_lcz_mean_m2": round(float(trow[f"mean_{metric_col}"]), 4),
                       "target_lcz_capacity_trees": int(trow["capacity_trees"]),
                       "target_lcz_n_segments": int(trow["n_segments"]),
                       "eligible_classes": tab.loc[tab["eligible"], "lcz_primary"].astype(int).tolist(),
                       "n_draws": n_draws, "mean_segments_per_draw": round(len(zdf) / n_draws, 1),
                       "mean_tiles_per_draw": round(float(np.mean(ntile)), 1), "union_tiles_count": len(z_tiles),
                       "n_distinct_segment_sets": n_unique_sets},
        "jaccard_overlap": {"jaccard_segment_count": _stats(jc), "jaccard_length_weighted": _stats(jl)},
        "lcz_class_table": json.loads(tab.to_json(orient="records")),
    }
    if write_validation:
        summary["s_exact_validation"] = {"n_tiles": len(vtiles), "n_s_tiles": len(s_tiles), "seed": seed_base,
                                         "rule": "default_rng(seed).choice(sorted S tiles, size=min(40, n), "
                                                 "replace=False) (A10 ruling 5)", "tiles": vtiles}

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        cols_s = ["seg_id", "tile_id", "district", "lcz_primary", "g250_id", "length_m", "n_ps", "m2_strong",
                  "m2_vstrong", M1_COL, "rank", "cum_length_m", "capacity_trees", "n_trees", "prorate_frac",
                  "planted_length_m", "scenario", "metric_col", "is_synthetic", "geometry"]
        gs = gpd.GeoDataFrame(s[[c for c in cols_s if c in s.columns]], geometry="geometry", crs=merged.crs)
        gs.to_parquet(out_dir / f"{prefix}scenario_S_segments{suffix}.parquet", index=False)
        gs.to_file(out_dir / f"{prefix}scenario_S_segments{suffix}.gpkg", layer="scenario_S", driver="GPKG")
        zdf.to_parquet(out_dir / f"{prefix}scenario_Z_draws{suffix}.parquet", index=False)
        tab.to_csv(out_dir / f"{prefix}lcz_class_table{suffix}.csv", index=False)
        if write_validation:
            pd.DataFrame({"tile_id": vtiles, "draw_order": range(len(vtiles)), "seed": seed_base,
                          "is_synthetic": meta["is_synthetic"]}).to_csv(
                out_dir / f"{prefix}s_exact_validation_tiles.csv", index=False)
        with open(out_dir / f"{prefix}scenarios_summary{suffix}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"--- written to {out_dir} (suffix '{suffix}')")
    return summary


def select_scenarios(segments_path: Path | None = None, metrics_path: Path | None = None, out_dir: Path | None = None,
                     variants=("primary", "locked"), target_pct: float = 0.10, tree_spacing_m: float = 8.0,
                     seed_base: int = SEED, n_draws: int = N_DRAWS, min_lcz_segments: int = 30,
                     n_validation_tiles: int = N_VALIDATION_TILES, dry_run: bool = False,
                     require_complete: bool = False) -> dict:
    """Run the A10 selection for the requested variants. Returns {variant: summary}."""
    segments_path = Path(segments_path) if segments_path else DEFAULT_SEGMENTS
    metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    seg, met, syn, partial, n_tiles = load_inputs(segments_path, metrics_path)
    if syn and "SYNTHETIC" not in str(out_dir).upper():
        raise RuntimeError(f"rule 6: synthetic inputs must be written under a SYNTHETIC folder, not {out_dir}")
    if not syn and "SYNTHETIC" in str(out_dir).upper():
        raise RuntimeError(f"real inputs must not be written into a SYNTHETIC folder ({out_dir})")
    prefix = "SYNTHETIC_" if syn else ""
    complete = (not partial) and n_tiles >= N_TILES_TOTAL
    run_label = "COMPLETE (427 tiles)" if complete else f"PARTIAL ({n_tiles} tiles)"
    if syn:
        run_label = f"SYNTHETIC_{run_label}"
    print(f"segments: {segments_path} ({len(seg)})\nmetrics : {metrics_path} ({len(met)})\nstatus  : {run_label}")
    if require_complete and not complete:
        raise RuntimeError(f"--require-complete: IMBAT metrics are {run_label}; nothing written")
    if dry_run:
        print("[DRY-RUN] inputs and columns OK; nothing computed or written.")
        return {"dry_run": True, "run_status": run_label, "n_tiles_at_extraction": n_tiles, "partial_run": partial}

    mcols = ["seg_id", "m2_strong", "m2_vstrong", M1_COL]
    merged = seg.merge(met[mcols], on="seg_id", how="inner", validate="one_to_one")
    meta = {"run_status": run_label, "is_synthetic": syn, "partial_run": bool(not complete),
            "n_tiles_at_extraction": n_tiles, "segments_path": str(segments_path), "metrics_path": str(metrics_path),
            "n_segments_joined": int(len(merged)), "amendments": ["A5", "A6", "A10"]}
    out = {}
    for v in variants:
        metric_col, suffix, label = VARIANTS[v]
        out[v] = run_variant(merged, metric_col, suffix, label, prefix, out_dir, meta, target_pct, tree_spacing_m,
                             seed_base, n_draws, min_lcz_segments, n_validation_tiles, write_validation=(v == "primary"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="H5 scenario selection (A10).")
    ap.add_argument("--segments", default=None, help="segments.gpkg (default data/street/analysis/segments.gpkg)")
    ap.add_argument("--metrics", default=None, help="seg_metrics_IMBAT.parquet (default data/street/analysis/...)")
    ap.add_argument("--outdir", default=None, help="output folder (default data/street/h5/selection)")
    ap.add_argument("--variants", default="primary,locked", help="comma list of primary, locked")
    ap.add_argument("--spacing", type=float, default=8.0)
    ap.add_argument("--target-pct", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-validation-tiles", type=int, default=N_VALIDATION_TILES)
    ap.add_argument("--min-lcz-segments", type=int, default=30)
    ap.add_argument("--require-complete", action="store_true",
                    help="stop with an error unless the IMBAT metrics are complete (427 tiles, partial_run False)")
    ap.add_argument("--dry-run", action="store_true", help="check inputs only; compute and write nothing")
    a = ap.parse_args()
    select_scenarios(a.segments, a.metrics, a.outdir, tuple(a.variants.split(",")), a.target_pct, a.spacing,
                     a.seed, a.n_draws, a.min_lcz_segments, a.n_validation_tiles, a.dry_run,
                     require_complete=a.require_complete and not a.dry_run)
