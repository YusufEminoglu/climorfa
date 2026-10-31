"""CLIMORFA Street — H5 evaluation (lock §2 H5; amendments A4, A5, A6, A10). No placeholder effect.

Outcome (A10 ruling 4): a scenario's weighted reduction of heat-stress hours is
    sum over its segments of  dM2_seg x w(g250 cell of the segment) x prorate
with dM2_seg = M2_seg(baseline) - M2_seg(P) from the city-wide scenario P (data/street/h5/P/seg_effects_<DAY>.parquet),
prorate = 1 for S segments and for full Z segments, n_trees / capacity for the last (partial) segment of a Z draw.
w = pop_res x network_reach_400m_road_length_m per 250 m cell (A4), via load_grid_weights.
Variants: primary M2_38 (dm2_38, selection by m2_vstrong) and locked M2_32 (dm2_32, selection by m2_strong).
Ratio S / Z per draw; H5 supported if the median of the 20 per-draw ratios >= 1.5 (95 % CI = 2.5-97.5 % of the draws).
The verdict is taken on IMBAT (the day of the budget ranking and of Figure 8); CALM is computed and reported.

Validation (A10 ruling 5): on the S_exact validation tiles (data/street/h5/S_exact/seg_effects_<DAY>.parquet)
    additive = sum over S segments on those tiles of dM2_38(P) x w
    exact    = sum over ALL segments on those tiles of dM2_38(S_exact) x w   (the lock's "summed over all segments")
    rel_diff = |additive - exact| / |exact|;  rel_diff > 0.15 -> verdict_status = 'descriptive only', else 'test'.
The S-segments-only exact sum is reported as well.

A11 (descriptive, reported in every case; they NEVER enter the verdict, on the primary M2_38 selection, both days):
  baseline mean M2_38 of the S and Z segments and their share with M2_38 = 0; share of segments with dM2_seg = 0 in P;
  cell-based companion sum w x prorate x dcell_h38 and continuous companion sum w x prorate x dmean_b15, with S/Z per draw.

Stops with an error when the P or S_exact effects are missing or partial, or do not cover the scenario segments.
Outputs (outputs/street/analysis/h5/ by default; SYNTHETIC_ prefix + SYNTHETIC folder for synthetic inputs):
  h5_evaluation_report.json, h5_evaluation_summary.csv (primary, IMBAT, one row per draw; read by Figure 8),
  h5_evaluation_draws_all.csv, h5_seg_outcomes.parquet, h5_tile_outcomes.csv, and copies of the primary selection
  tables (scenario_S_segments.parquet, scenario_Z_draws.parquet, scenarios_summary.json) as the evaluated snapshot.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
CHOSEN_REACH_FIELD = "network_reach_400m_road_length_m"
H5_BASE = PROJECT_ROOT / "data/street/h5"
DEFAULT_OUT = PROJECT_ROOT / "outputs/street/analysis/h5"
N_TILES_TOTAL = 427
THRESHOLD = 1.5
REL_TOL = 0.15
PRIMARY_DAY = "IMBAT"
DAYS = ("IMBAT", "CALM")
VARIANTS = {"M2_38": ("", "dm2_38", "m2_vstrong"), "M2_32": ("_M2_32", "dm2_32", "m2_strong")}
A11_COLS = ("cell_h38_base", "cell_h38_scen", "dcell_h38", "mean_utci_b15_base", "mean_utci_b15_scen", "dmean_b15")
A11_COMPANIONS = {"cell_h38": "dcell_h38", "mean_utci_b15": "dmean_b15"}


def load_grid_weights(grid_path: Path | None = None, reach_field: str = CHOSEN_REACH_FIELD,
                      pop_col: str | None = None) -> tuple[dict[str, float], str]:
    """Pedestrian weight = reach x population per 250 m cell (A4: pop_res primary, pop = all-building sensitivity).
    The grid's `reach` column is network_reach_400m_road_length_m (chosen in WP3.0c); otherwise the field is joined from
    data/03_processed/grid_250m_road_network_reach_features.csv. A missing population column is an error."""
    if grid_path is None or not Path(grid_path).exists():
        real_grid = PROJECT_ROOT / "data/street/analysis/grid_250.gpkg"
        synth_grid = PROJECT_ROOT / "data/street/h5/SYNTHETIC_grid_250.gpkg"
        grid_path = real_grid if real_grid.exists() else synth_grid
    grid_path = Path(grid_path)
    print(f"Loading grid weights from: {grid_path}")
    g = gpd.read_file(grid_path, layer="grid")
    reach_col = "reach" if "reach" in g.columns else reach_field
    if reach_col not in g.columns:
        df_r = pd.read_csv(PROJECT_ROOT / "data/03_processed/grid_250m_road_network_reach_features.csv")
        g = g.merge(df_r[["grid_id", reach_field]], left_on="g250_id", right_on="grid_id", how="left")
        reach_col = reach_field
    if pop_col is None:
        pop_col = "pop_res" if "pop_res" in g.columns else "pop"
        if pop_col == "pop":
            print("  WARNING: 'pop_res' absent; using 'pop' (all-building dasymetry = the A4 sensitivity).")
    if pop_col not in g.columns:
        raise KeyError(f"population column {pop_col!r} not in {grid_path}")
    print(f"  Pedestrian weight: {reach_col} x {pop_col}")
    g["pedestrian_weight"] = g[reach_col].fillna(0.0) * g[pop_col].fillna(0.0)
    return dict(zip(g["g250_id"], g["pedestrian_weight"])), pop_col


# ---------------------------------------------------------------------------------------------------------------------
def ratio(s: float, z: float) -> float:
    if z > 0:
        return s / z
    return float("inf") if s > 0 else float("nan")


def validation_status(additive: float, exact: float, tol: float = REL_TOL) -> tuple[float, str]:
    """A10 ruling 5: relative difference of the additive vs exact weighted dM2_38 and the resulting verdict status."""
    rel = abs(additive - exact) / abs(exact) if exact != 0 else (0.0 if additive == 0 else float("inf"))
    return rel, ("descriptive only" if (not np.isfinite(rel) or rel > tol) else "test")


def _num(x, nd=4):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _dist(x) -> dict:
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return {"median": _num(np.median(x), 3), "mean": _num(np.mean(x), 3),
                "ci_95": [_num(np.percentile(x, 2.5), 3), _num(np.percentile(x, 97.5), 3)],
                "min": _num(np.min(x), 3), "max": _num(np.max(x), 3), "threshold": THRESHOLD}


def load_effects(scen_dir: Path, day: str, prefix: str, expected_tiles: list[str] | None, require_427: bool,
                 label: str) -> pd.DataFrame:
    fp = Path(scen_dir) / f"{prefix}seg_effects_{day}.parquet"
    if not fp.exists():
        raise RuntimeError(f"{label} effects missing: {fp} (run extract_h5_effects.py --scenario {label})")
    e = pd.read_parquet(fp)
    if e.empty:
        raise RuntimeError(f"{label} effects empty: {fp}")
    n_ex, n_exp = int(e["n_tiles_extracted"].iloc[0]), int(e["n_tiles_expected"].iloc[0])
    if bool(e["partial_run"].astype(bool).any()) or n_ex != n_exp:
        raise RuntimeError(f"{label} effects are partial: {n_ex}/{n_exp} tiles ({fp})")
    if require_427 and n_exp != N_TILES_TOTAL:
        raise RuntimeError(f"{label} effects cover {n_exp} expected tiles, not {N_TILES_TOTAL} ({fp})")
    if expected_tiles is not None and n_exp != len(set(expected_tiles)):
        # (a listed tile may hold no PS segment, so tile_id coverage is not tested row-wise; the count must agree)
        raise RuntimeError(f"{label} effects expected {n_exp} tiles, the selection lists {len(set(expected_tiles))}")
    miss = [c for c in A11_COLS if c not in e.columns]
    if miss:
        raise RuntimeError(f"{label} effects lack the A11 columns {miss}: rerun extract_h5_effects.py ({fp})")
    e["seg_id"] = e["seg_id"].astype(np.int64)
    return e


def scenario_outcomes(s_tab: pd.DataFrame, z_tab: pd.DataFrame, eff: pd.DataFrame, weights: dict,
                      dcol: str) -> dict:
    """Additive outcomes of S and of each Z draw for one effect column (dm2_38 / dm2_32) and one day."""
    dm = eff.set_index("seg_id")[dcol]
    need = set(s_tab["seg_id"]) | set(z_tab["seg_id"])
    missing = sorted(need - set(dm.index))
    if missing:
        raise RuntimeError(f"{len(missing)} scenario segments have no P effect (e.g. {missing[:5]}): P effects do not "
                           f"cover the selection")
    n_nan = int(dm.reindex(list(need)).isna().sum())
    dm = dm.fillna(0.0)

    def contrib(t):
        w = t["g250_id"].map(weights)
        return dm.reindex(t["seg_id"]).to_numpy() * w.fillna(0.0).to_numpy() * t["prorate_frac"].to_numpy(), \
            int(w.isna().sum())

    s_c, s_miss_w = contrib(s_tab)
    s_out = float(s_c.sum())
    per = []
    z_miss_w = 0
    for d, g in z_tab.groupby("draw_id"):
        c, mw = contrib(g)
        z_miss_w += mw
        zo = float(c.sum())
        per.append({"draw_id": int(d), "n_trees": int(g["n_trees"].sum()), "n_segments": int(len(g)),
                    "delta_weighted_m2_z": zo, "delta_weighted_m2_s": s_out, "targeting_ratio_S_vs_Z": ratio(s_out, zo)})
    per = pd.DataFrame(per)
    return {"S": s_out, "per_draw": per, "n_nan_effect": n_nan, "n_missing_weight_S": s_miss_w,
            "n_missing_weight_Z_rows": z_miss_w,
            "S_sum_dm2_unweighted": float(dm.reindex(s_tab["seg_id"]).sum())}


def a11_items(s_tab: pd.DataFrame, z_tab: pd.DataFrame, eff: pd.DataFrame, weights: dict) -> tuple[dict, pd.DataFrame]:
    """A11 descriptive items for one day. Returns (report dict, per-draw table with the companion columns)."""
    base = eff.set_index("seg_id")["m2_38_base"]
    dm = eff.set_index("seg_id")["dm2_38"]
    s_m2 = base.reindex(s_tab["seg_id"]).to_numpy(float)
    z_mean, z_zero = [], []
    for _, g in z_tab.groupby("draw_id"):
        m = base.reindex(g["seg_id"]).to_numpy(float)
        z_mean.append(float(np.nanmean(m)))
        z_zero.append(float(np.nanmean(m == 0)))
    fin = dm.notna()
    hot = fin & (base.reindex(dm.index) > 0)
    in_s = dm.index.isin(set(s_tab["seg_id"]))
    in_z = dm.index.isin(set(z_tab["seg_id"]))
    out = {"baseline_mean_m2_38": {"S": _num(np.nanmean(s_m2)), "Z_per_draw": _dist(z_mean)},
           "share_m2_38_zero": {"S": _num(np.nanmean(s_m2 == 0)), "Z_per_draw": _dist(z_zero)},
           "share_dm2_38_zero_in_P": {"all_segments": _num((dm[fin] == 0).mean()),
                                      "segments_with_baseline_m2_38_gt_0": _num((dm[hot] == 0).mean()),
                                      "S_segments": _num((dm[fin & in_s] == 0).mean()),
                                      "Z_segments_any_draw": _num((dm[fin & in_z] == 0).mean()),
                                      "n_segments": int(fin.sum())},
           "companions": {}}
    per = None
    for name, dcol in A11_COMPANIONS.items():
        o = scenario_outcomes(s_tab, z_tab, eff, weights, dcol)
        p = o["per_draw"][["draw_id", "delta_weighted_m2_z", "delta_weighted_m2_s", "targeting_ratio_S_vs_Z"]].rename(
            columns={"delta_weighted_m2_z": f"delta_weighted_{name}_z", "delta_weighted_m2_s": f"delta_weighted_{name}_s",
                     "targeting_ratio_S_vs_Z": f"ratio_{name}"})
        per = p if per is None else per.merge(p, on="draw_id")
        out["companions"][name] = {"effect_column": dcol, "S": _num(o["S"], 2),
                                   "Z": _dist(p[f"delta_weighted_{name}_z"]), "ratio_S_vs_Z": _dist(p[f"ratio_{name}"]),
                                   "n_nan_effect": o["n_nan_effect"]}
    return out, per


def evaluate_h5(selection_dir: Path | None = None, p_dir: Path | None = None, s_exact_dir: Path | None = None,
                grid_250_path: Path | None = None, segments_path: Path | None = None, out_dir: Path | None = None,
                pop_col: str | None = None, days=DAYS, primary_day: str = PRIMARY_DAY, rel_tol: float = REL_TOL) -> dict:
    selection_dir = Path(selection_dir) if selection_dir else H5_BASE / "selection"
    p_dir = Path(p_dir) if p_dir else H5_BASE / "P"
    s_exact_dir = Path(s_exact_dir) if s_exact_dir else H5_BASE / "S_exact"
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    segments_path = Path(segments_path) if segments_path else PROJECT_ROOT / "data/street/analysis/segments.gpkg"
    syn = "SYNTHETIC" in str(selection_dir).upper()
    prefix = "SYNTHETIC_" if syn else ""

    # ---- selection (both variants) ----
    sel = {}
    for v, (sfx, dcol, mcol) in VARIANTS.items():
        sp = selection_dir / f"{prefix}scenarios_summary{sfx}.json"
        if not sp.exists():
            raise FileNotFoundError(f"selection missing: {sp} (run select_scenarios.py)")
        summ = json.loads(sp.read_text(encoding="utf-8"))
        s_tab = pd.read_parquet(selection_dir / f"{prefix}scenario_S_segments{sfx}.parquet")
        z_tab = pd.read_parquet(selection_dir / f"{prefix}scenario_Z_draws{sfx}.parquet")
        for t, name in ((s_tab, "S"), (z_tab, "Z")):
            if "g250_id" not in t.columns or "prorate_frac" not in t.columns:
                raise KeyError(f"{name} table of variant {v} lacks g250_id / prorate_frac (re-run select_scenarios.py)")
        if summ.get("metric_col") != mcol:
            raise ValueError(f"variant {v}: selection metric {summ.get('metric_col')} != {mcol}")
        syn = syn or bool(summ.get("is_synthetic", False))
        sel[v] = {"summary": summ, "S": s_tab, "Z": z_tab, "sfx": sfx}
    if syn and ("SYNTHETIC" not in str(out_dir).upper() or "SYNTHETIC" not in str(selection_dir).upper()):
        raise RuntimeError("rule 6: synthetic H5 inputs/outputs must live under SYNTHETIC folders")
    if not syn and "SYNTHETIC" in str(out_dir).upper():
        raise RuntimeError("real H5 results must not be written into a SYNTHETIC folder")
    prefix = "SYNTHETIC_" if syn else ""
    vt_path = selection_dir / f"{prefix}s_exact_validation_tiles.csv"
    vtiles = pd.read_csv(vt_path)["tile_id"].astype(str).tolist()
    summ0 = sel["M2_38"]["summary"]
    run_status = summ0.get("run_status", "PARTIAL")
    sel_partial = bool(summ0.get("partial_run", True)) or run_status != f"COMPLETE ({N_TILES_TOTAL} tiles)"

    # ---- effects ----
    effP = {d: load_effects(p_dir, d, prefix, None, require_427=not syn, label="P") for d in days}
    effS = {d: load_effects(s_exact_dir, d, prefix, vtiles, require_427=False, label="S_exact") for d in days}
    bad = sorted(set(effS[primary_day]["tile_id"]) - set(vtiles))
    if bad:
        raise RuntimeError(f"S_exact effects contain tiles outside the validation list: {bad[:5]}")

    weights, pop_used = load_grid_weights(grid_250_path, pop_col=pop_col)
    if not syn and pop_col is None and pop_used != "pop_res":
        raise RuntimeError("A4: the primary weight needs pop_res; grid_250 has no pop_res column")

    # ---- outcomes per variant and day ----
    variants_out, draws_all, per_primary = {}, [], None
    for v, (sfx, dcol, mcol) in VARIANTS.items():
        vo = {"selection_metric": mcol, "effect_column": dcol, "tree_budget_N": sel[v]["summary"]["tree_budget_N"],
              "target_lcz": sel[v]["summary"]["scenario_Z"]["target_lcz"], "days": {}}
        for d in days:
            o = scenario_outcomes(sel[v]["S"], sel[v]["Z"], effP[d], weights, dcol)
            per = o["per_draw"]
            r = per["targeting_ratio_S_vs_Z"].to_numpy(float)
            med = float(np.median(r))
            vo["days"][d] = {"S_delta_weighted_m2": _num(o["S"], 2),
                             "Z_delta_weighted_m2": _dist(per["delta_weighted_m2_z"]),
                             "targeting_ratio_S_vs_Z": _dist(r), "supported": bool(med >= THRESHOLD),
                             "S_sum_dm2_unweighted": _num(o["S_sum_dm2_unweighted"], 2),
                             "n_nan_effect": o["n_nan_effect"], "n_missing_weight_S": o["n_missing_weight_S"],
                             "n_missing_weight_Z_rows": o["n_missing_weight_Z_rows"]}
            draws_all.append(per.assign(variant=v, day=d))
            if v == "M2_38" and d == primary_day:
                per_primary = per
        variants_out[v] = vo

    # ---- A11 descriptive items (primary selection): computed after the verdict inputs, never used by them ----
    a11 = {"descriptive_only": True, "selection": "M2_38 (primary)",
           "note": "A11: reported with the verdict in every case; does not change the H5 verdict.", "days": {}}
    for d in days:
        a11["days"][d], a11_per = a11_items(sel["M2_38"]["S"], sel["M2_38"]["Z"], effP[d], weights)
        for i, x in enumerate(draws_all):
            if x["variant"].iloc[0] == "M2_38" and x["day"].iloc[0] == d:
                draws_all[i] = x.merge(a11_per, on="draw_id")
        if d == primary_day:
            per_primary = per_primary.merge(a11_per, on="draw_id")

    # ---- A4 sensitivity: all-building population ----
    sens = None
    try:
        w_all, _ = load_grid_weights(grid_250_path, pop_col="pop")
        o = scenario_outcomes(sel["M2_38"]["S"], sel["M2_38"]["Z"], effP[primary_day], w_all, "dm2_38")
        sens = {"pop_column": "pop", "day": primary_day,
                "targeting_ratio_S_vs_Z": _dist(o["per_draw"]["targeting_ratio_S_vs_Z"])}
    except KeyError:
        pass

    # ---- A10 ruling 5 validation ----
    s38 = sel["M2_38"]["S"]
    val = {"tiles": vtiles, "n_tiles": len(vtiles), "rel_tol": rel_tol, "metric": "dm2_38", "days": {}}
    for d in days:
        for dcol in ("dm2_38", "dm2_32"):
            sv = s38[s38["tile_id"].isin(vtiles)]
            pv = effP[d].set_index("seg_id")[dcol].reindex(sv["seg_id"]).fillna(0.0).to_numpy()
            add = float((pv * sv["g250_id"].map(weights).fillna(0.0).to_numpy()).sum())
            ex = effS[d]
            g250 = segment_g250(segments_path, ex["seg_id"], s38)
            wv = ex["seg_id"].map(g250).map(weights).fillna(0.0).to_numpy()
            exact_all = float((ex[dcol].fillna(0.0).to_numpy() * wv).sum())
            in_s = ex["seg_id"].isin(set(s38["seg_id"])).to_numpy()
            exact_s = float((ex[dcol].fillna(0.0).to_numpy() * wv * in_s).sum())
            rel, status = validation_status(add, exact_all, rel_tol)
            val["days"].setdefault(d, {})[dcol] = {"additive": _num(add, 2), "exact_all_segments": _num(exact_all, 2),
                                                   "exact_S_segments_only": _num(exact_s, 2),
                                                   "rel_diff": _num(rel, 4), "status_if_rule_applied": status,
                                                   "n_S_segments_on_tiles": int(len(sv)),
                                                   "n_segments_on_tiles": int(len(ex))}
    vmain = val["days"][primary_day]["dm2_38"]
    verdict_status = vmain["status_if_rule_applied"]
    val["rel_diff"] = vmain["rel_diff"]
    val["additive"], val["exact"] = vmain["additive"], vmain["exact_all_segments"]
    val["verdict_status"] = verdict_status

    # ---- verdict (primary variant, primary day) ----
    pv = variants_out["M2_38"]["days"][primary_day]
    r_main = pv["targeting_ratio_S_vs_Z"]
    supported = bool(pv["supported"])
    is_partial = sel_partial or syn
    if is_partial:
        status = "PARTIAL — MACHINERY TEST ONLY"
        finding = None
        note = ("DO NOT REPORT AS A FINDING. Synthetic or partial selection; machinery test only."
                if syn else "DO NOT REPORT AS A FINDING. The selection was made on PARTIAL baseline metrics.")
    else:
        status = "H5 SUPPORTED" if supported else "H5 NOT SUPPORTED"
        finding = status if verdict_status == "test" else f"{status} (descriptive only)"
        note = ("Final H5 evaluation (427 tiles), additive outcome from scenario P (A10 ruling 4)."
                + ("" if verdict_status == "test" else
                   f" Additive vs exact differ by {vmain['rel_diff']:.3f} > {rel_tol}: reported as descriptive only, "
                   "not as a test (A10 ruling 5)."))
    n_tiles = {"selection_extraction": int(summ0.get("n_tiles_at_extraction", 0)),
               "P_effects": int(effP[primary_day]["n_tiles_extracted"].iloc[0]),
               "S_exact_effects": int(effS[primary_day]["n_tiles_extracted"].iloc[0])}
    report = {
        "hypothesis": "H5 Targeting (street-targeted S vs zone-targeted Z)",
        "run_status": run_status, "is_synthetic": syn, "is_partial": is_partial, "partial": is_partial,
        "machinery_test_only": is_partial, "n_tiles": n_tiles, "amendments": ["A4", "A5", "A6", "A10"],
        "authoritative_rule": "H5 supported if S reduces weighted heat-stress hours by >= 1.5 x Z (median of the 20 "
                              "per-draw ratios, 95 % CI = 2.5-97.5 % of the draws); primary M2_38 (A10 ruling 1)",
        "outcome_definition": "sum over scenario segments of dM2_seg(P) x pedestrian weight x prorate (A10 ruling 4)",
        "outcome_day": primary_day,
        "chosen_pedestrian_reach_field": CHOSEN_REACH_FIELD, "pedestrian_weight_pop_column": pop_used,
        "is_primary_residential_population": bool(pop_used == "pop_res"),
        "tree_budget_N": int(sel["M2_38"]["summary"]["tree_budget_N"]),
        "scenario_S": {"n_segments": int(len(s38)), "n_trees": int(s38["n_trees"].sum()),
                       "delta_weighted_m2": pv["S_delta_weighted_m2"]},
        "scenario_Z_20_draws": {"n_draws": int(len(per_primary)),
                                "median_delta_weighted_m2": pv["Z_delta_weighted_m2"]["median"],
                                "mean_delta_weighted_m2": pv["Z_delta_weighted_m2"]["mean"]},
        "targeting_ratio_S_vs_Z": r_main,
        "jaccard_overlap_priority_sets": summ0.get("jaccard_overlap", {}),
        "variants": variants_out,
        "validation": val,
        "a11": a11,
        "sensitivity_pop_all_buildings": sens,
        "verdict": "supported" if supported else "not supported",
        "verdict_status": verdict_status,
        "decision": {"status": status, "scientific_finding": finding, "threshold_condition_met": supported,
                     "verdict_status": verdict_status,
                     "condition_description": f"Median ratio {r_main['median']} {'>=' if supported else '<'} 1.5",
                     "note": note},
    }

    # ---- per-segment and per-tile outcomes (Figure 8 panel c) ----
    seg_out = segment_outcomes(effP, sel, weights, segments_path, days)
    tile_out = tile_outcomes(seg_out, effS, sel, weights, vtiles, segments_path, days)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}h5_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_primary.to_csv(out_dir / f"{prefix}h5_evaluation_summary.csv", index=False)
    pd.concat(draws_all, ignore_index=True).to_csv(out_dir / f"{prefix}h5_evaluation_draws_all.csv", index=False)
    seg_out.assign(is_synthetic=syn).to_parquet(out_dir / f"{prefix}h5_seg_outcomes.parquet", index=False)
    tile_out.assign(is_synthetic=syn).to_csv(out_dir / f"{prefix}h5_tile_outcomes.csv", index=False)
    for name in ("scenario_S_segments.parquet", "scenario_Z_draws.parquet", "scenarios_summary.json"):
        src = selection_dir / f"{prefix}{name}"
        if src.resolve() != (out_dir / f"{prefix}{name}").resolve():
            shutil.copy(src, out_dir / f"{prefix}{name}")

    print("\n=======================================================")
    print(f"  H5 evaluation — {run_status}{'  [MACHINERY TEST ONLY]' if is_partial else ''}")
    print("=======================================================")
    for v in VARIANTS:
        for d in days:
            x = variants_out[v]["days"][d]
            print(f"  {v} {d}: S = {x['S_delta_weighted_m2']}, Z median = {x['Z_delta_weighted_m2']['median']}, "
                  f"S/Z median {x['targeting_ratio_S_vs_Z']['median']} CI {x['targeting_ratio_S_vs_Z']['ci_95']}")
    print(f"  validation ({len(vtiles)} tiles, {primary_day}, dM2_38): additive {vmain['additive']} vs exact "
          f"{vmain['exact_all_segments']} -> rel diff {vmain['rel_diff']} -> {verdict_status}")
    print(f"  decision: {status} | verdict {report['verdict']} | verdict_status {verdict_status}")
    x = a11["days"][primary_day]
    print(f"  A11 ({primary_day}, descriptive): mean M2_38 S {x['baseline_mean_m2_38']['S']}, Z "
          f"{x['baseline_mean_m2_38']['Z_per_draw']['median']}; M2_38 = 0 S {x['share_m2_38_zero']['S']}, Z "
          f"{x['share_m2_38_zero']['Z_per_draw']['median']}; dM2_seg = 0 in P "
          f"{x['share_dm2_38_zero_in_P']['all_segments']}; S/Z cell "
          f"{x['companions']['cell_h38']['ratio_S_vs_Z']['median']}, UTCI "
          f"{x['companions']['mean_utci_b15']['ratio_S_vs_Z']['median']}")
    print(f"  written to {out_dir}")
    return report


def segment_g250(segments_path: Path, seg_ids, s_tab: pd.DataFrame) -> dict:
    """seg_id -> g250_id from segments.gpkg (cached)."""
    key = str(segments_path)
    if key not in _G250_CACHE:
        g = gpd.read_file(segments_path, layer="segments", columns=["seg_id", "g250_id", "tile_id"],
                          ignore_geometry=True)
        _G250_CACHE[key] = dict(zip(g["seg_id"].astype(np.int64), g["g250_id"]))
    return _G250_CACHE[key]


_G250_CACHE: dict = {}


def segment_outcomes(effP: dict, sel: dict, weights: dict, segments_path: Path, days) -> pd.DataFrame:
    base = effP[days[0]][["seg_id", "tile_id"]].copy()
    g250 = segment_g250(segments_path, base["seg_id"], sel["M2_38"]["S"])
    base["g250_id"] = base["seg_id"].map(g250)
    base["ped_weight"] = base["g250_id"].map(weights).fillna(0.0)
    for d in days:
        e = effP[d].set_index("seg_id")
        for c in ("dm2_38", "dm2_32", "dm1", "m2_38_base", "m2_32_base", "m1_base", "dcell_h38", "dmean_b15"):
            base[f"{c}_{d}"] = base["seg_id"].map(e[c])
        base[f"w_dm2_38_{d}"] = base[f"dm2_38_{d}"].fillna(0.0) * base["ped_weight"]
        base[f"w_dm2_32_{d}"] = base[f"dm2_32_{d}"].fillna(0.0) * base["ped_weight"]
    for v, tag in (("M2_38", ""), ("M2_32", "_M2_32")):
        s, z = sel[v]["S"], sel[v]["Z"]
        nd = z["draw_id"].nunique()
        base[f"in_S{tag}"] = base["seg_id"].isin(set(s["seg_id"]))
        base[f"n_z_draws{tag}"] = base["seg_id"].map(z.groupby("seg_id")["draw_id"].nunique()).fillna(0).astype(int)
        base[f"z_mean_prorate{tag}"] = base["seg_id"].map(z.groupby("seg_id")["prorate_frac"].sum() / nd).fillna(0.0)
    return base


def tile_outcomes(seg: pd.DataFrame, effS: dict, sel: dict, weights: dict, vtiles: list[str], segments_path: Path,
                  days) -> pd.DataFrame:
    rows = []
    for tid, g in seg.groupby("tile_id"):
        r = {"tile_id": tid, "n_segments": int(len(g)), "n_S": int(g["in_S"].sum()),
             "n_S_M2_32": int(g["in_S_M2_32"].sum()), "in_validation": tid in set(vtiles)}
        for d in days:
            r[f"S_additive_w_dm2_38_{d}"] = float((g[f"w_dm2_38_{d}"] * g["in_S"]).sum())
            r[f"Z_mean_additive_w_dm2_38_{d}"] = float((g[f"w_dm2_38_{d}"] * g["z_mean_prorate"]).sum())
            r[f"S_M2_32_additive_w_dm2_32_{d}"] = float((g[f"w_dm2_32_{d}"] * g["in_S_M2_32"]).sum())
            r[f"Z_M2_32_mean_additive_w_dm2_32_{d}"] = float((g[f"w_dm2_32_{d}"] * g["z_mean_prorate_M2_32"]).sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    g250 = segment_g250(segments_path, None, None)
    s_ids = set(sel["M2_38"]["S"]["seg_id"])
    for d in days:
        e = effS[d].copy()
        e["w"] = e["seg_id"].map(g250).map(weights).fillna(0.0)
        e["w_dm2_38"] = e["dm2_38"].fillna(0.0) * e["w"]
        ex_all = e.groupby("tile_id")["w_dm2_38"].sum()
        ex_s = e[e["seg_id"].isin(s_ids)].groupby("tile_id")["w_dm2_38"].sum()
        t[f"exact_all_w_dm2_38_{d}"] = t["tile_id"].map(ex_all)
        t[f"exact_S_only_w_dm2_38_{d}"] = t["tile_id"].map(ex_s).where(t["in_validation"])
        with np.errstate(divide="ignore", invalid="ignore"):
            t[f"rel_diff_additive_vs_exact_{d}"] = ((t[f"S_additive_w_dm2_38_{d}"] - t[f"exact_all_w_dm2_38_{d}"]).abs()
                                                    / t[f"exact_all_w_dm2_38_{d}"].abs()).where(t["in_validation"])
    return t.sort_values("tile_id").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate H5 (A10).")
    ap.add_argument("--selection", default=None, help="selection folder (default data/street/h5/selection)")
    ap.add_argument("--p-dir", default=None, help="scenario P folder (default data/street/h5/P)")
    ap.add_argument("--s-exact-dir", default=None, help="scenario S_exact folder (default data/street/h5/S_exact)")
    ap.add_argument("--grid", default=None, help="grid_250.gpkg (default data/street/analysis/grid_250.gpkg)")
    ap.add_argument("--segments", default=None, help="segments.gpkg (default data/street/analysis/segments.gpkg)")
    ap.add_argument("--pop-col", default=None, help="population column (default pop_res, A4)")
    ap.add_argument("--outdir", default=None, help="output folder (default outputs/street/analysis/h5)")
    a = ap.parse_args()
    evaluate_h5(a.selection, a.p_dir, a.s_exact_dir, a.grid, a.segments, a.outdir, a.pop_col)
