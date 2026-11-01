"""CLIMORFA Street — descriptive decomposition of the H5 S/Z ratio. NOT part of the lock; written after the H5 verdict.

The locked outcome sums dM2_38 x pedestrian weight per segment, so the S/Z ratio mixes three things: how many segments the
same N trees reach, how many hours each segment loses and who walks there. This script separates them from the raw tables
(selection, seg_effects, grid_250) without importing evaluate_h5, and so also recomputes the verdict number independently.
Output: outputs/street/analysis/h5/h5_decomposition.json. Nothing here changes the H5 verdict.

    python -m scripts.street.h5.decompose_h5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

PROJECT_ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
SEL = PROJECT_ROOT / "data/street/h5/selection"
P_DIR = PROJECT_ROOT / "data/street/h5/P"
GRID = PROJECT_ROOT / "data/street/analysis/grid_250.gpkg"
OUT = PROJECT_ROOT / "outputs/street/analysis/h5/h5_decomposition.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def scenario_stats(t: pd.DataFrame, eff: pd.DataFrame, w: pd.Series) -> dict:
    d = t["seg_id"].map(eff["dm2_38"]).fillna(0.0)
    wt = t["g250_id"].map(w).fillna(0.0)
    p = t["prorate_frac"]
    km = float((t["length_m"] * p).sum() / 1000)
    ids = t["seg_id"]
    return dict(n_segments=int(len(t)), trees=float((t["n_trees"] * p).sum()), planted_km=km,
                mean_length_m=float(t["length_m"].mean()), sum_dm2=float((d * p).sum()), mean_dm2=float(d.mean()),
                mean_m2_38_base=float(ids.map(eff["m2_38_base"]).mean()),
                share_m2_38_zero_after=float((ids.map(eff["m2_38_scen"]) == 0).mean()),
                mean_dutci_b15=float(ids.map(eff["dmean_b15"]).mean()), mean_dm1=float(ids.map(eff["dm1"]).mean()),
                mean_weight=float(wt.mean()), share_weight_zero=float((wt == 0).mean()),
                outcome=float((d * wt * p).sum()), outcome_x_length=float((d * wt * p * t["length_m"]).sum()),
                outcome_per_km=float((d * wt * p).sum() / km))


def main():
    s = pd.read_parquet(SEL / "scenario_S_segments.parquet")
    z = pd.read_parquet(SEL / "scenario_Z_draws.parquet")
    g = pyogrio.read_dataframe(GRID, read_geometry=False)
    w = (g["reach"].fillna(0.0) * g["pop_res"].fillna(0.0)).set_axis(g["g250_id"])
    res = {"descriptive_only": True, "post_hoc": True, "note": "Written after the H5 verdict; does not change it.", "days": {}}
    for day in ("IMBAT", "CALM"):
        eff = pd.read_parquet(P_DIR / f"seg_effects_{day}.parquet").set_index("seg_id")
        a = scenario_stats(s, eff, w)
        draws = pd.DataFrame([scenario_stats(x, eff, w) for _, x in z.groupby("draw_id")])
        if len(draws) != 20:
            raise RuntimeError(f"expected 20 draws, found {len(draws)}")
        zm = draws.median().to_dict()
        ratio = a["outcome"] / draws["outcome"].to_numpy()
        res["days"][day] = {
            "S": a, "Z_median_of_draws": zm, "S_over_Z": {k: a[k] / zm[k] for k in a if zm[k]},
            "median_ratio_recomputed": float(np.median(ratio)),
            "ratio_ci95_of_draws": [float(x) for x in np.percentile(ratio, [2.5, 97.5])],
            "ratio_unweighted_sum_dm2": float(np.median(a["sum_dm2"] / draws["sum_dm2"].to_numpy())),
            "ratio_length_weighted": float(np.median(a["outcome_x_length"] / draws["outcome_x_length"].to_numpy()))}
    res["S_lcz_counts"] = {str(k): int(v) for k, v in s["lcz_primary"].value_counts().items()}
    res["Z_lcz_counts_all_draws"] = {str(k): int(v) for k, v in z["lcz_primary"].value_counts().items()}
    res["S_n_tiles"] = int(s["tile_id"].nunique())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
