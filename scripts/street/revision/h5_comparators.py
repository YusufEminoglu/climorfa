"""CLIMORFA Street — post-hoc, descriptive H5 comparators (revision R1, reviewer request). NOT part of the lock.

Nothing here changes the registered H5 verdict. No new SOLWEIG run: every scenario is scored with the frozen additive
machinery of evaluate_h5.py (A10 ruling 4):

    outcome = sum over the scenario's segments of dM2_38_seg(P) x w(g250 cell of the segment) x prorate

dM2_38_seg = M2_38(baseline) - M2_38(P) from the all-planted run P (data/street/h5/P/seg_effects_<DAY>.parquet);
w = pop_res x network_reach_400m_road_length_m of the 250 m cell (load_grid_weights); capacity = floor(length_m / 8 m);
budget N = 67,770 trees (the frozen S budget); prorate = n_trees / capacity (only the last segment of a fill is < 1).

Candidate pool for every comparator = the frozen H5 pool (segments with an IMBAT M2_38 value, length_m >= 20, n_ps >= 25).
Rankings that use baseline M2_38 use the IMBAT selection metric m2_vstrong (the frozen S and Z are selected once on IMBAT
and evaluated on both days); a CALM-ranked sensitivity is reported for the two zone rules.

Step 1 (gate): rebuild the frozen S and Z with the select_scenarios functions, check them against the frozen tables and
reproduce the frozen S/Z medians (14.461 IMBAT, 11.166 CALM). The script stops if that fails.

Comparators (same N, same effects, same outcome):
  Z_pop        LCZ classes (lcz_primary, >= 30 pool segments, LCZ 0 excluded per A6) ranked by the w-weighted mean baseline
               M2_38; fill class by class (a class whose capacity <= remaining budget is planted entirely, otherwise
               length-weighted draws without replacement with draw_zone); 20 draws, draw k seeded 20261030 + k.
  Z_cell_mean  250 m cells ranked by the w-weighted mean baseline M2_38 of their pool segments (w is constant inside a cell,
               so this equals the plain cell mean); ties by cell w desc, then g250_id.
  Z_cell_total 250 m cells ranked by total w x M2_38 = w_cell x sum of the cell's baseline M2_38 (ties by cell mean, id).
               Both cell rules plant all pool segments of the top cells in rank order; inside the last cell the segments
               follow the frozen tie-breaking (M2_38 desc, M1 desc, seg_id asc) and the last segment is prorated.
  S_proxy      planner-available score, no modelled UTCI: rank_pct(svf_med) + rank_pct(1 - tcover15) (higher = more open sky,
               less tree cover within 15 m); top segments until N trees, last prorated; ties seg_id asc.
  R_city       length-weighted random segments city-wide (draw_zone on the full pool), 20 draws seeded 20261030 + k.
  S_LST        segments ranked by Landsat LST (lst_med, median of 30 m LST over the segment's PS cells), highest first.

Person-hour equivalent (resident-hours): sum dM2_38 x prorate x pop_res(cell) x length_seg / L_cell, with L_cell the total
length of all segments of that 250 m cell; i.e. each resident of a cell is assumed present on the cell's own street
network during the 10-18 h window, spread uniformly per metre of street. It drops the 400 m reach factor of w.

Outputs: outputs/street/analysis/revision_r1/h5_comparators.json, h5_comparators_table.csv, h5_comparators_draws.csv

    python -m scripts.street.revision.h5_comparators
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from scripts.street.h5.evaluate_h5 import DAYS, H5_BASE, load_effects, load_grid_weights, ratio, scenario_outcomes
from scripts.street.h5.select_scenarios import (DEFAULT_METRICS, DEFAULT_SEGMENTS, M1_COL, N_DRAWS, SEED,
                                                capacity_trees, draw_zone, lcz_class_table, rank_for_budget,
                                                select_hottest_share)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
FROZEN = PROJECT_ROOT / "outputs/street/analysis/h5"
OUT_DIR = PROJECT_ROOT / "outputs/street/analysis/revision_r1"
GRID = PROJECT_ROOT / "data/street/analysis/grid_250.gpkg"
METRIC = "m2_vstrong"          # baseline M2_38 on IMBAT (selection day)
SPACING = 8.0
MIN_LCZ_SEG = 30
FROZEN_MEDIAN = {"IMBAT": 14.461, "CALM": 11.166}
TOL = 5e-4
UNIT = ("h x resident x m: sum of dM2_38 (hours of the 10-18 h bands with segment-median UTCI >= 38 degC removed) x "
        "w (residents of the 250 m cell x road length in m within its 400 m network reach) x prorate")


# ---------------------------------------------------------------------------------------------------------------------
def load_pool():
    seg = pyogrio.read_dataframe(DEFAULT_SEGMENTS, layer="segments", read_geometry=False)
    met = pd.read_parquet(DEFAULT_METRICS)[["seg_id", "m2_strong", "m2_vstrong", M1_COL, "svf_med"]]
    merged = seg.merge(met, on="seg_id", how="inner", validate="one_to_one")
    valid = merged[merged[METRIC].notna() & (merged["length_m"] >= 20.0) & (merged["n_ps"] >= 25)].copy()
    valid["capacity_trees"] = capacity_trees(valid["length_m"], SPACING)
    l_cell = seg.groupby("g250_id")["length_m"].sum()
    return seg, valid, l_cell


def fill_ordered(ordered: pd.DataFrame, budget_n: int) -> pd.DataFrame:
    """Plant segments in the given order until N trees; the last segment is prorated."""
    cap = ordered["capacity_trees"].to_numpy(np.int64)
    cum = np.cumsum(cap)
    k = int(np.searchsorted(cum, budget_n, side="left"))
    if k >= len(ordered):
        raise ValueError(f"capacity {cum[-1]} < N {budget_n}")
    t = ordered.iloc[: k + 1].copy()
    take = cap[: k + 1].copy()
    take[-1] = budget_n - (cum[k - 1] if k else 0)
    t["n_trees"] = take
    t["prorate_frac"] = take / cap[: k + 1]
    t["planted_length_m"] = t["length_m"] * t["prorate_frac"]
    assert int(t["n_trees"].sum()) == budget_n
    return t


def full(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["n_trees"] = t["capacity_trees"]
    t["prorate_frac"] = 1.0
    t["planted_length_m"] = t["length_m"]
    return t


def fill_classes(valid: pd.DataFrame, order: list, budget_n: int, rng: np.random.Generator) -> pd.DataFrame:
    parts, rem = [], budget_n
    for c in order:
        cands = valid[valid["lcz_primary"] == c].reset_index(drop=True)
        cap = int(cands["capacity_trees"].sum())
        if cap <= rem:
            parts.append(full(cands))
            rem -= cap
        else:
            parts.append(draw_zone(cands, rem, rng))
            rem = 0
        if rem == 0:
            break
    t = pd.concat(parts, ignore_index=True)
    assert int(t["n_trees"].sum()) == budget_n
    return t


# ---------------------------------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, eff: pd.DataFrame, weights: dict, pop: dict, l_cell: pd.Series, s_ids: set, s_km: float):
        self.dm = eff.set_index("seg_id")["dm2_38"].fillna(0.0)
        self.base = eff.set_index("seg_id")["m2_38_base"]
        self.w, self.pop, self.l_cell, self.s_ids, self.s_km = weights, pop, l_cell, s_ids, s_km

    def __call__(self, t: pd.DataFrame) -> dict:
        d = self.dm.reindex(t["seg_id"]).fillna(0.0).to_numpy()
        w = t["g250_id"].map(self.w).fillna(0.0).to_numpy()
        p = t["prorate_frac"].to_numpy(float)
        pop = t["g250_id"].map(self.pop).fillna(0.0).to_numpy()
        lc = t["g250_id"].map(self.l_cell).to_numpy(float)
        share = np.where(lc > 0, t["length_m"].to_numpy(float) / lc, 0.0)
        pl = t["length_m"].to_numpy(float) * p
        in_s = t["seg_id"].isin(self.s_ids).to_numpy()
        return {"weighted_hours_removed": float((d * w * p).sum()),
                "unweighted_sum_dm2_38": float((d * p).sum()),
                "resident_hours_removed": float((d * p * pop * share).sum()),
                "n_segments": int(len(t)), "n_trees": int(t["n_trees"].sum()), "km_planted": float(pl.sum() / 1000),
                "share_segments_zero_weight": float((w == 0).mean()),
                "share_km_zero_weight": float(pl[w == 0].sum() / pl.sum()),
                "mean_baseline_m2_38": float(self.base.reindex(t["seg_id"]).mean()),
                "share_km_in_S": float(pl[in_s].sum() / pl.sum()),
                "n_tiles": int(t["tile_id"].nunique())}


def _q(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "ci_95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
            "min": float(x.min()), "max": float(x.max())}


def summarise(rows: list[dict], s_out: float, z_med: float, z_per_draw: np.ndarray | None) -> dict:
    df = pd.DataFrame(rows)
    o = df["weighted_hours_removed"].to_numpy()
    res = {k: (float(df[k].median()) if k not in ("n_segments", "n_trees", "n_tiles") or len(df) > 1
               else int(df[k].iloc[0])) for k in df.columns}
    res["n_draws"] = int(len(df))
    res["ratio_to_frozen_Z_median"] = float(np.median(o)) / z_med
    res["ratio_to_S"] = float(np.median(o)) / s_out
    res["S_over_scenario"] = ratio(s_out, float(np.median(o)))
    if len(df) > 1:
        res["weighted_hours_removed_dist"] = _q(o)
        res["ratio_to_frozen_Z_median_dist"] = _q(o / z_med)
        res["S_over_scenario_per_draw"] = _q(s_out / o)
        res["per_draw_medians_note"] = "every field above is the median over draws"
    return res


# ---------------------------------------------------------------------------------------------------------------------
def build_scenarios(valid: pd.DataFrame, weights: dict, budget_n: int, rank_metric: pd.Series, tag: str = "") -> dict:
    """Comparator planting tables; rank_metric = baseline M2_38 by seg_id used by the zone rules."""
    v = valid.copy()
    v["_m2"] = v["seg_id"].map(rank_metric)
    v["_w"] = v["g250_id"].map(weights).fillna(0.0)
    info, sc = {}, {}

    # Z_pop ----------------------------------------------------------------------------------------------------------
    g = v.groupby("lcz_primary")
    ct = pd.DataFrame({"n_segments": g.size(), "capacity_trees": g["capacity_trees"].sum(), "sum_w": g["_w"].sum(),
                       "mean_m2_38": g["_m2"].mean(),
                       "w_mean_m2_38": g.apply(lambda x: (x["_w"] * x["_m2"]).sum() / x["_w"].sum()
                                               if x["_w"].sum() > 0 else np.nan, include_groups=False)}).reset_index()
    ct["eligible"] = (ct["lcz_primary"] != 0) & (ct["n_segments"] >= MIN_LCZ_SEG)
    ct = ct.sort_values(["eligible", "w_mean_m2_38", "lcz_primary"], ascending=[False, False, True],
                        na_position="last").reset_index(drop=True)
    order = ct.loc[ct["eligible"] & ct["w_mean_m2_38"].notna(), "lcz_primary"].astype(int).tolist()
    elig = v[v["lcz_primary"].isin(order)]
    sc["Z_pop"] = [fill_classes(elig, order, budget_n, np.random.default_rng(SEED + k)) for k in range(N_DRAWS)]
    used = sorted({int(c) for t in sc["Z_pop"] for c in t["lcz_primary"].unique()})
    info["Z_pop"] = {"class_order_by_w_mean_m2_38": order, "classes_used": used,
                     "class_table": json.loads(ct.to_json(orient="records"))}

    # Z_cell ---------------------------------------------------------------------------------------------------------
    vc = v[v["g250_id"].notna()]
    gc = vc.groupby("g250_id")
    cells = pd.DataFrame({"w": gc["_w"].first(), "n_segments": gc.size(), "capacity_trees": gc["capacity_trees"].sum(),
                          "mean_m2_38": gc["_m2"].mean(), "sum_m2_38": gc["_m2"].sum()}).reset_index()
    cells["total_w_m2_38"] = cells["w"] * cells["sum_m2_38"]
    within = rank_for_budget(vc.assign(**{METRIC: vc["_m2"]}), METRIC)
    for name, keys in (("Z_cell_mean", ["mean_m2_38", "w", "g250_id"]),
                       ("Z_cell_total", ["total_w_m2_38", "mean_m2_38", "g250_id"])):
        cs = cells.sort_values(keys, ascending=[False, False, True]).reset_index(drop=True)
        cs["cell_rank"] = np.arange(len(cs))
        o = within.merge(cs[["g250_id", "cell_rank"]], on="g250_id")
        o = o.sort_values(["cell_rank", "rank"], kind="mergesort")
        t = fill_ordered(o, budget_n)
        sc[name] = [t]
        info[name] = {"n_cells_planted": int(t["g250_id"].nunique()),
                      "last_cell_partial": bool(t["g250_id"].iloc[-1] in set(t["g250_id"]) and
                                                int(t.loc[t["g250_id"] == t["g250_id"].iloc[-1], "n_trees"].sum())
                                                < int(cs.loc[cs["g250_id"] == t["g250_id"].iloc[-1],
                                                             "capacity_trees"].iloc[0])),
                      "ranking_key": keys,
                      "n_cells_in_pool": int(len(cells)),
                      "share_planted_cells_with_w_zero": float((cs.set_index("g250_id").loc[
                          t["g250_id"].unique(), "w"] == 0).mean())}
    if tag:
        return {"scenarios": {k: sc[k] for k in ("Z_pop", "Z_cell_mean", "Z_cell_total")}, "info": info}

    # S_proxy --------------------------------------------------------------------------------------------------------
    p = v.copy()
    p["score"] = p["svf_med"].rank(pct=True) + (1.0 - p["tcover15"]).rank(pct=True)
    sc["S_proxy"] = [fill_ordered(p.sort_values(["score", "seg_id"], ascending=[False, True], na_position="last"),
                                  budget_n)]
    info["S_proxy"] = {"score": "rank_pct(svf_med) + rank_pct(1 - tcover15), desc; ties seg_id asc",
                       "svf_source": "svf_med = median SOLWEIG sky view factor (DSM geometry only, day-invariant)",
                       "tree_cover_source": "tcover15 = tree-pixel share of land within 15 m (segments.gpkg, H2 driver)"}

    # R_city ---------------------------------------------------------------------------------------------------------
    rc = v.reset_index(drop=True)
    sc["R_city"] = [draw_zone(rc, budget_n, np.random.default_rng(SEED + k)) for k in range(N_DRAWS)]
    info["R_city"] = {"pool": "all frozen-pool segments (all LCZ incl. 0)", "draws": N_DRAWS, "seed_rule": "SEED + k"}

    # S_LST ----------------------------------------------------------------------------------------------------------
    sc["S_LST"] = [fill_ordered(v.sort_values(["lst_med", "seg_id"], ascending=[False, True], na_position="last"),
                                budget_n)]
    info["S_LST"] = {"ranking": "lst_med desc (median 30 m Landsat LST over PS cells), ties seg_id asc"}
    return {"scenarios": sc, "info": info}


# ---------------------------------------------------------------------------------------------------------------------
def main():
    seg, valid, l_cell = load_pool()
    weights, pop_used = load_grid_weights(GRID)
    if pop_used != "pop_res":
        raise RuntimeError("primary weight needs pop_res")
    g = pyogrio.read_dataframe(GRID, read_geometry=False)
    pop = dict(zip(g["g250_id"], g["pop_res"].fillna(0.0)))
    effP = {d: load_effects(H5_BASE / "P", d, "", None, require_427=True, label="P") for d in DAYS}

    # ---- Step 1: rebuild the frozen S and Z and reproduce the frozen medians ----------------------------------------
    s_fz = pd.read_parquet(FROZEN / "scenario_S_segments.parquet")
    z_fz = pd.read_parquet(FROZEN / "scenario_Z_draws.parquet")
    ranked = rank_for_budget(valid, METRIC)
    s_new, budget_n, _, _ = select_hottest_share(ranked, 0.10, SPACING)
    tab = lcz_class_table(valid, METRIC, budget_n, SPACING, MIN_LCZ_SEG)
    target = int(tab.loc[tab["is_target"], "lcz_primary"].iloc[0])
    cands = valid[valid["lcz_primary"] == target].reset_index(drop=True)
    z_new = pd.concat([draw_zone(cands, budget_n, np.random.default_rng(SEED + k)).assign(draw_id=k)
                       for k in range(N_DRAWS)], ignore_index=True)
    same_s = list(s_new["seg_id"]) == list(s_fz["seg_id"])
    same_z = all(list(z_new.loc[z_new.draw_id == k, "seg_id"]) == list(z_fz.loc[z_fz.draw_id == k, "seg_id"]) and
                 np.allclose(z_new.loc[z_new.draw_id == k, "prorate_frac"], z_fz.loc[z_fz.draw_id == k, "prorate_frac"])
                 for k in range(N_DRAWS))
    repro = {"budget_N": budget_n, "z_target_lcz": target, "S_rebuilt_identical": same_s,
             "Z_draws_rebuilt_identical": same_z, "days": {}}
    zmed, s_out_d, zdraws_d = {}, {}, {}
    for d in DAYS:
        o_fz = scenario_outcomes(s_fz, z_fz, effP[d], weights, "dm2_38")
        o_new = scenario_outcomes(s_new, z_new, effP[d], weights, "dm2_38")
        r_fz = float(np.median(o_fz["per_draw"]["targeting_ratio_S_vs_Z"]))
        r_new = float(np.median(o_new["per_draw"]["targeting_ratio_S_vs_Z"]))
        zmed[d] = float(np.median(o_fz["per_draw"]["delta_weighted_m2_z"]))
        s_out_d[d] = o_fz["S"]
        zdraws_d[d] = o_fz["per_draw"]["delta_weighted_m2_z"].to_numpy()
        ok = abs(r_fz - FROZEN_MEDIAN[d]) < TOL and abs(r_new - FROZEN_MEDIAN[d]) < TOL
        repro["days"][d] = {"frozen_reported": FROZEN_MEDIAN[d], "median_S_over_Z_frozen_tables": r_fz,
                            "median_S_over_Z_rebuilt_tables": r_new, "S_weighted": o_fz["S"],
                            "Z_median_weighted": zmed[d], "match": bool(ok)}
        print(f"[repro] {d}: S/Z median frozen tables {r_fz:.4f}, rebuilt {r_new:.4f}, reported {FROZEN_MEDIAN[d]}")
    if not (same_s and same_z and budget_n == 67770 and all(x["match"] for x in repro["days"].values())):
        raise SystemExit(f"REPRODUCTION FAILED: {json.dumps(repro, indent=1)}")
    print("[repro] OK: S and 20 Z draws rebuilt identically; medians reproduced.")

    # ---- comparators ------------------------------------------------------------------------------------------------
    rank_imbat = effP["IMBAT"].set_index("seg_id")["m2_38_base"]
    built = build_scenarios(valid, weights, budget_n, rank_imbat)
    sens_calm = build_scenarios(valid, weights, budget_n, effP["CALM"].set_index("seg_id")["m2_38_base"], tag="CALM")
    s_ids, s_km = set(s_fz["seg_id"]), float(s_fz["length_m"].sum() / 1000)
    z_list = [x for _, x in z_fz.groupby("draw_id")]
    allsc = {"S": [s_fz], "Z": z_list, **built["scenarios"]}

    days_out, table, draws_rows = {}, [], []
    for d in DAYS:
        sco = Scorer(effP[d], weights, pop, l_cell, s_ids, s_km)
        days_out[d] = {}
        for name, tabs in allsc.items():
            rows = [sco(t) for t in tabs]
            res = summarise(rows, s_out_d[d], zmed[d], None)
            days_out[d][name] = res
            for k, r in enumerate(rows):
                draws_rows.append({"day": d, "scenario": name, "draw_id": k, **r})
            ci = res.get("weighted_hours_removed_dist", {}).get("ci_95", [np.nan, np.nan])
            table.append({"day": d, "scenario": name, "n_draws": res["n_draws"],
                          "weighted_hours_removed": res["weighted_hours_removed"],
                          "weighted_ci95_lo": ci[0], "weighted_ci95_hi": ci[1],
                          "ratio_to_frozen_Z_median": res["ratio_to_frozen_Z_median"],
                          "ratio_to_S": res["ratio_to_S"], "S_over_scenario": res["S_over_scenario"],
                          "n_segments": res["n_segments"], "km_planted": res["km_planted"], "n_trees": res["n_trees"],
                          "unweighted_sum_dm2_38": res["unweighted_sum_dm2_38"],
                          "resident_hours_removed": res["resident_hours_removed"],
                          "share_segments_zero_weight": res["share_segments_zero_weight"],
                          "mean_baseline_m2_38": res["mean_baseline_m2_38"], "share_km_in_S": res["share_km_in_S"]})
        # unweighted S / Z reference
        u_s = days_out[d]["S"]["unweighted_sum_dm2_38"]
        u_z = np.array([r["unweighted_sum_dm2_38"] for r in draws_rows if r["day"] == d and r["scenario"] == "Z"])
        days_out[d]["unweighted_reference"] = {"S_unweighted": u_s, "Z_unweighted_median": float(np.median(u_z)),
                                               "S_over_Z_unweighted_median_of_draws": float(np.median(u_s / u_z))}

    # CALM-ranked zone-rule sensitivity, evaluated on CALM
    sco = Scorer(effP["CALM"], weights, pop, l_cell, s_ids, s_km)
    sens = {k: summarise([sco(t) for t in v], s_out_d["CALM"], zmed["CALM"], None)
            for k, v in sens_calm["scenarios"].items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "descriptive_only": True, "post_hoc": True, "reviewer_request": "R1",
        "note": "Post-hoc comparators; additive outcome from run P; do not change the registered H5 verdict.",
        "outcome_definition": "sum over the scenario's own segments of dM2_38(P) x w x prorate (A10 ruling 4)",
        "unit": UNIT,
        "person_hour_definition": ("resident_hours_removed = sum dM2_38 x prorate x pop_res(cell) x length_seg / "
                                   "L_cell, L_cell = total length of all 74,178 segments in the 250 m cell: every "
                                   "resident assumed on the cell's own streets during 10-18 h, uniform per metre"),
        "ratio_definitions": {"ratio_to_frozen_Z_median": "scenario outcome (median over draws) / frozen Z median "
                                                          "outcome of the same day",
                              "ratio_to_S": "scenario outcome (median over draws) / frozen S outcome of the same day",
                              "S_over_scenario": "frozen S outcome / scenario outcome (comparable to the S/Z ratio)"},
        "budget_N": budget_n, "pool": {"n_segments": int(len(valid)),
                                       "rule": "m2_vstrong (IMBAT) not NaN, length_m >= 20, n_ps >= 25 (frozen pool)"},
        "ranking_baseline": "zone rules rank on IMBAT baseline M2_38 (selection day of the frozen S and Z) for both "
                            "evaluation days; CALM-ranked sensitivity under sensitivity_zone_rules_ranked_on_CALM",
        "seed": SEED, "n_draws": N_DRAWS, "reproduction_check": repro,
        "scenario_info": built["info"], "days": days_out,
        "sensitivity_zone_rules_ranked_on_CALM": {"evaluated_day": "CALM", "scenarios": sens,
                                                  "info": sens_calm["info"]},
    }
    (OUT_DIR / "h5_comparators.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    t = pd.DataFrame(table)
    t.to_csv(OUT_DIR / "h5_comparators_table.csv", index=False, float_format="%.6g")
    pd.DataFrame(draws_rows).to_csv(OUT_DIR / "h5_comparators_draws.csv", index=False, float_format="%.6g")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print(t.drop(columns=["weighted_ci95_lo", "weighted_ci95_hi"]).to_string(index=False))
        print("CALM-ranked sensitivity:", {k: round(v["S_over_scenario"], 3) for k, v in sens.items()})
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
