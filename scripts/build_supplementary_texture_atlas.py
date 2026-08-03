"""Selection logic for the 6 x 8 supplementary texture/surface/canopy atlas
cells. The standalone texture-only figure this used to build directly was
retired and merged into `build_supplementary_texture_surface_canopy_atlas.py`
(same 48 cells, same selection rule below, now paired with matched surface model and
canopy rasters per cell instead of being a separate 48-panel figure) --
this module now only provides the deterministic cell-selection functions
that both the merged atlas and its own selected-tiles CSV depend on."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_figure_1_texture_atlas import CLASSES


ATLAS_METRICS = [
    "building_coverage_exact",
    "height_proxy_aw_mean_m",
    "dsm_elevation_m_std",
    "canopy_cover_gt2m_share",
    "road_density_exact_m_per_km2",
    "network_intersection_density_per_km2",
    "s2_ndvi_mean",
    "lst_c_median_mean",
]

COLUMNS = [
    ("Typical", "median fabric"),
    ("Coverage+", "built cover"),
    ("Height/Surface+", "vertical roughness"),
    ("Network+", "street grain"),
    ("Green+", "canopy + NDVI"),
    ("Thermal edge", "coolest / hottest"),
]


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "lcz_weak_label",
        "lcz_weak_confidence",
        "eligible_core",
        "building_count_exact",
        *ATLAS_METRICS,
    ]
    base = _numeric(df, cols)
    return base[
        base["lcz_weak_label"].isin(CLASSES)
        & (base["lcz_weak_confidence"] >= 0.60)
        & (base["eligible_core"] == 1)
        & (base["building_count_exact"] > 0)
    ].dropna(subset=ATLAS_METRICS)


def _score(pool: pd.DataFrame, cols: list[str]) -> pd.Series:
    z = (pool[cols] - pool[cols].mean()) / pool[cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
    return z.mean(axis=1)


def _pick_one(pool: pd.DataFrame, score: pd.Series, used: set[str], highest: bool = True) -> pd.Series:
    ranked = pool.assign(_score=score)
    ranked = ranked[~ranked["grid_id"].isin(used)]
    ranked = ranked.sort_values(["_score", "grid_id"], ascending=[not highest, True])
    row = ranked.iloc[0].drop(labels=["_score"])
    used.add(str(row["grid_id"]))
    return row


def _select_atlas_tiles(df: pd.DataFrame) -> pd.DataFrame:
    base = _candidate_frame(df)
    selections = []
    for klass in CLASSES:
        class_pool = base[base["lcz_weak_label"] == klass].copy()
        low_q = class_pool["lst_c_median_mean"].quantile(0.25)
        high_q = class_pool["lst_c_median_mean"].quantile(0.75)
        thermal_rows = [
            ("cool", class_pool[class_pool["lst_c_median_mean"] <= low_q].copy(), low_q),
            ("hot", class_pool[class_pool["lst_c_median_mean"] >= high_q].copy(), high_q),
        ]
        for thermal_group, pool, threshold in thermal_rows:
            if len(pool) < len(COLUMNS):
                raise RuntimeError(f"Not enough candidates for LCZ {klass} {thermal_group}: {len(pool)}")
            used: set[str] = set()
            med = pool[ATLAS_METRICS].median()
            sd = pool[ATLAS_METRICS].std(ddof=0).replace(0, 1.0).fillna(1.0)
            typical_score = -np.square((pool[ATLAS_METRICS] - med) / sd).sum(axis=1)
            rows = [
                ("Typical", _pick_one(pool, typical_score, used, highest=True)),
                ("Coverage+", _pick_one(pool, pool["building_coverage_exact"], used, highest=True)),
                ("Height/Surface+", _pick_one(pool, _score(pool, ["height_proxy_aw_mean_m", "dsm_elevation_m_std"]), used, highest=True)),
                ("Network+", _pick_one(pool, _score(pool, ["road_density_exact_m_per_km2", "network_intersection_density_per_km2"]), used, highest=True)),
                ("Green+", _pick_one(pool, _score(pool, ["canopy_cover_gt2m_share", "s2_ndvi_mean"]), used, highest=True)),
            ]
            thermal_score = pool["lst_c_median_mean"] if thermal_group == "hot" else -pool["lst_c_median_mean"]
            rows.append(("Thermal edge", _pick_one(pool, thermal_score, used, highest=True)))
            for col_index, (prototype, row) in enumerate(rows):
                item = row.to_dict()
                item["prototype"] = prototype
                item["prototype_detail"] = COLUMNS[col_index][1]
                item["thermal_group"] = thermal_group
                item["thermal_threshold_c"] = threshold
                item["lcz_weak_label"] = int(klass)
                item["row_label"] = f"LCZ {klass} {thermal_group}"
                item["row_index"] = len(selections) // len(COLUMNS)
                item["col_index"] = col_index
                selections.append(item)
    selected = pd.DataFrame(selections)
    keep = [
        "row_index",
        "col_index",
        "row_label",
        "prototype",
        "prototype_detail",
        "thermal_group",
        "thermal_threshold_c",
        "grid_id",
        "district",
        "lcz_weak_label",
        "lcz_weak_confidence",
        *ATLAS_METRICS,
    ]
    return selected[keep]


if __name__ == "__main__":
    df = pd.read_csv(Path(".") / "data/03_processed/grid_250m_model_features_v8.csv")
    print(json.dumps(_select_atlas_tiles(df).to_dict(orient="records"), indent=2))
