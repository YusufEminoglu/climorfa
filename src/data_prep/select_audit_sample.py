"""Select a spatially balanced stratified random audit sample.

Expected input: a 250 m grid layer for the Izmir functional urban region with
at least these fields:

- grid_id
- weak_lcz
- weak_lcz_confidence
- coastal_band
- district
- built_intensity_bin
- industrial_port_flag

The script writes a GeoPackage with the selected audit cells and a CSV summary.
It deliberately avoids a GNN workflow; network information is expected to be
attached later as interpretable tabular metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    import numpy as np
    import pandas as pd


DEFAULT_STRATA = [
    "weak_lcz",
    "coastal_band",
    "built_intensity_bin",
    "industrial_port_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="Input 250 m grid layer.")
    parser.add_argument("--out-gpkg", required=True, help="Selected sample output GeoPackage.")
    parser.add_argument("--out-summary", required=True, help="CSV summary output.")
    parser.add_argument("--target-n", type=int, default=400)
    parser.add_argument("--min-per-stratum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument(
        "--strata",
        nargs="*",
        default=DEFAULT_STRATA,
        help="Columns used to define sampling strata.",
    )
    parser.add_argument(
        "--min-distance-m",
        type=float,
        default=0.0,
        help="Optional minimum centroid distance inside each stratum.",
    )
    parser.add_argument(
        "--eligible-column",
        default="eligible_core",
        help="Optional 1/0 column used to filter eligible cells if present.",
    )
    return parser.parse_args()


def ensure_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def make_stratum(row: pd.Series, columns: list[str]) -> str:
    values = [str(row[column]) if pd.notna(row[column]) else "missing" for column in columns]
    return "|".join(values)


def select_with_min_distance(
    group: gpd.GeoDataFrame,
    target_n: int,
    rng: np.random.Generator,
    min_distance_m: float,
) -> gpd.GeoDataFrame:
    shuffled = group.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    if min_distance_m <= 0 or target_n >= len(shuffled):
        return shuffled.head(target_n)

    centroids = shuffled.geometry.centroid
    selected_indices: list[int] = []
    selected_points = []

    for idx, point in zip(shuffled.index, centroids):
        if len(selected_indices) >= target_n:
            break
        if all(point.distance(other) >= min_distance_m for other in selected_points):
            selected_indices.append(idx)
            selected_points.append(point)

    if len(selected_indices) < target_n:
        remainder = shuffled.drop(index=selected_indices).head(target_n - len(selected_indices))
        selected_indices.extend(remainder.index.tolist())

    return group.loc[selected_indices]


def main() -> None:
    args = parse_args()

    global gpd, np, pd
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(args.seed)

    grid_path = Path(args.grid)
    gdf = gpd.read_file(grid_path)
    ensure_columns(gdf, ["grid_id", *args.strata])
    if args.eligible_column in gdf.columns:
        gdf = gdf[gdf[args.eligible_column].astype(bool)].copy()

    gdf = gdf.copy()
    gdf["sample_stratum"] = gdf.apply(make_stratum, axis=1, columns=args.strata)
    total_n = len(gdf)
    if total_n == 0:
        raise ValueError("Input grid has no rows.")

    selected_parts = []
    for stratum, group in gdf.groupby("sample_stratum", dropna=False):
        proportional_n = int(round(args.target_n * len(group) / total_n))
        stratum_n = max(args.min_per_stratum, proportional_n)
        stratum_n = min(stratum_n, len(group))
        role = "certainty_rare" if len(group) <= args.min_per_stratum else "random_audit"

        selected = select_with_min_distance(group, stratum_n, rng, args.min_distance_m).copy()
        selected["sample_selected"] = True
        selected["sample_role"] = role
        selected["sample_seed"] = args.seed
        selected_parts.append(selected)

    selected_gdf = pd.concat(selected_parts, ignore_index=True)
    selected_gdf = gpd.GeoDataFrame(selected_gdf, geometry="geometry", crs=gdf.crs)

    out_gpkg = Path(args.out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    selected_gdf.to_file(out_gpkg, driver="GPKG", layer="audit_sample")

    summary = (
        selected_gdf.groupby(["sample_stratum", "sample_role"], dropna=False)
        .size()
        .reset_index(name="selected_n")
        .sort_values(["sample_stratum", "sample_role"])
    )
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    print(f"selected={len(selected_gdf)} strata={selected_gdf['sample_stratum'].nunique()}")
    print(f"out_gpkg={out_gpkg}")
    print(f"out_summary={out_summary}")


if __name__ == "__main__":
    main()
