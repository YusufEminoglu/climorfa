"""Attach lightweight preliminary features to 250 m grid cells.

These fields are for sampling v0 only. They help stratify the audit sample
before the full, exact feature pipeline and GEE exports are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", required=True)
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--buildings-gpkg", required=True)
    parser.add_argument("--buildings-layer", default="buildings_fua")
    parser.add_argument("--roads-gpkg", required=True)
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--districts", required=True)
    parser.add_argument("--out-gpkg", required=True)
    parser.add_argument("--out-layer", default="grid_250m_prelim")
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def add_district(grid: gpd.GeoDataFrame, districts_path: str) -> gpd.GeoDataFrame:
    districts = gpd.read_file(districts_path)
    if districts.crs != grid.crs:
        districts = districts.to_crs(grid.crs)
    district_col = "ADINUMARAS" if "ADINUMARAS" in districts.columns else districts.columns[0]
    centroids = grid[["grid_id", "geometry"]].copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(
        centroids,
        districts[[district_col, "geometry"]],
        how="left",
        predicate="within",
    )
    district = joined.groupby("grid_id")[district_col].first().rename("district")
    grid = grid.merge(district, on="grid_id", how="left")
    grid["district"] = grid["district"].fillna("unknown")
    return grid


def add_building_metrics(grid: gpd.GeoDataFrame, buildings_path: str, layer: str) -> gpd.GeoDataFrame:
    buildings = gpd.read_file(buildings_path, layer=layer)
    if buildings.crs != grid.crs:
        buildings = buildings.to_crs(grid.crs)
    buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()].copy()
    buildings["building_area_prelim_m2"] = buildings.geometry.area
    bcent = buildings[["building_area_prelim_m2", "geometry"]].copy()
    bcent["geometry"] = bcent.geometry.centroid
    joined = gpd.sjoin(
        bcent,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    grouped = joined.dropna(subset=["grid_id"]).groupby("grid_id").agg(
        building_count_prelim=("building_area_prelim_m2", "size"),
        building_area_prelim_m2=("building_area_prelim_m2", "sum"),
    )
    grid = grid.merge(grouped, on="grid_id", how="left")
    grid["building_count_prelim"] = grid["building_count_prelim"].fillna(0).astype(int)
    grid["building_area_prelim_m2"] = grid["building_area_prelim_m2"].fillna(0.0)
    grid["building_coverage_prelim"] = grid["building_area_prelim_m2"] / grid["cell_area_m2"]
    return grid


def add_road_metrics(grid: gpd.GeoDataFrame, roads_path: str, layer: str) -> gpd.GeoDataFrame:
    roads = gpd.read_file(roads_path, layer=layer)
    if roads.crs != grid.crs:
        roads = roads.to_crs(grid.crs)
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads["road_length_prelim_m"] = roads.geometry.length
    rcent = roads[["road_length_prelim_m", "geometry"]].copy()
    rcent["geometry"] = rcent.geometry.centroid
    joined = gpd.sjoin(
        rcent,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    grouped = joined.dropna(subset=["grid_id"]).groupby("grid_id").agg(
        road_length_prelim_m=("road_length_prelim_m", "sum"),
    )
    grid = grid.merge(grouped, on="grid_id", how="left")
    grid["road_length_prelim_m"] = grid["road_length_prelim_m"].fillna(0.0)
    grid["road_density_prelim_m_per_km2"] = grid["road_length_prelim_m"] / (grid["cell_area_m2"] / 1_000_000)
    return grid


def add_bins(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    eligible = grid["eligible_core"].astype(bool)
    values = grid.loc[eligible, "building_coverage_prelim"]
    if values.nunique() >= 3:
        bins = pd.qcut(values.rank(method="first"), q=3, labels=["low", "mid", "high"])
        grid.loc[eligible, "built_intensity_bin"] = bins.astype(str).to_numpy()
    else:
        grid.loc[eligible, "built_intensity_bin"] = "unbinned"
    grid.loc[~eligible, "built_intensity_bin"] = "edge"
    grid["industrial_port_flag"] = "unknown_pending_landuse"
    grid["coastal_band"] = "pending_coast_distance"
    return grid


def main() -> None:
    args = parse_args()
    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer)
    grid = add_district(grid, args.districts)
    grid = add_building_metrics(grid, args.buildings_gpkg, args.buildings_layer)
    grid = add_road_metrics(grid, args.roads_gpkg, args.roads_layer)
    grid = add_bins(grid)

    out_gpkg = Path(args.out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(out_gpkg, layer=args.out_layer, driver="GPKG")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    non_geom = grid.drop(columns="geometry")
    non_geom.to_csv(out_csv, index=False)

    print(f"rows={len(grid)}")
    print(f"eligible_core={int(grid['eligible_core'].sum())}")
    print(f"out_gpkg={out_gpkg}")
    print(f"out_csv={out_csv}")


if __name__ == "__main__":
    main()
