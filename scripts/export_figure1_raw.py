"""Export raw GIS/CSV layers for the manually designed Figure 1 (Figma).

Produces one GeoPackage per retained panel group under
paper/figures/manual/figure1_Raw/, using the same source layers and the same
deterministic texture-tile selection as scripts/build_figure_1_texture_atlas.py.
Panel (d) (RF out-of-fold confidence) is intentionally excluded: it is a model
result, not study-area/data content, and stays in the results figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_figure_1_texture_atlas import _select_texture_tiles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/figures/manual/figure1_Raw"

SOURCES = {
    "grid": ROOT / "data/03_processed/analysis_grids.gpkg",
    "boundary": ROOT / "data/02_interim/study_area_fua.gpkg",
    "coastline": ROOT / "data/02_interim/shoreline_epsg5253.gpkg",
    "buildings_roads": ROOT / "data/02_interim/akilli_sehir_fua.gpkg",
    "features": ROOT / "data/03_processed/grid_250m_model_features_v8.csv",
}


def _load_grid_with_features() -> gpd.GeoDataFrame:
    grid = gpd.read_file(SOURCES["grid"], layer="grid_250m")
    feat = pd.read_csv(SOURCES["features"])
    cols = ["grid_id", "lcz_weak_label", "lcz_weak_confidence", "lcz_mixed_flag", "lst_c_median_mean"]
    return grid.merge(feat[cols], on="grid_id", how="left")


def _panel_a(g: gpd.GeoDataFrame) -> None:
    path = OUT_DIR / "panel_a_study_frame.gpkg"
    if path.exists():
        path.unlink()
    g[["grid_id", "eligible_core", "geometry"]].to_file(path, layer="grid_250m", driver="GPKG")
    gpd.read_file(SOURCES["boundary"], layer="study_area_fua").to_file(path, layer="study_boundary", driver="GPKG")
    gpd.read_file(SOURCES["coastline"], layer="shoreline_epsg5253").to_file(path, layer="coastline", driver="GPKG")


def _panel_b(g: gpd.GeoDataFrame) -> None:
    path = OUT_DIR / "panel_b_weak_lcz.gpkg"
    if path.exists():
        path.unlink()
    cols = ["grid_id", "eligible_core", "lcz_weak_label", "lcz_weak_confidence", "lcz_mixed_flag", "geometry"]
    g[cols].to_file(path, layer="grid_250m_weak_lcz", driver="GPKG")


def _panel_c(g: gpd.GeoDataFrame) -> None:
    path = OUT_DIR / "panel_c_lst.gpkg"
    if path.exists():
        path.unlink()
    cols = ["grid_id", "eligible_core", "lst_c_median_mean", "geometry"]
    g[cols].to_file(path, layer="grid_250m_lst", driver="GPKG")


def _panels_e_h(root: Path) -> pd.DataFrame:
    df = pd.read_csv(SOURCES["features"])
    tiles = _select_texture_tiles(df)
    grid = gpd.read_file(SOURCES["grid"], layer="grid_250m")
    cell_lookup = grid.set_index("grid_id")
    buildings = gpd.read_file(SOURCES["buildings_roads"], layer="buildings_fua")
    roads = gpd.read_file(SOURCES["buildings_roads"], layer="roads_fua")

    cell_rows, building_rows, road_rows = [], [], []
    for _, row in tiles.iterrows():
        klass = int(row["lcz_weak_label"])
        cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
        geom = cell.geometry.iloc[0]
        cell = cell.assign(lcz_weak_label=klass)
        cell_rows.append(cell)
        b = buildings[buildings.intersects(geom)].clip(geom).assign(lcz_weak_label=klass, grid_id=row["grid_id"])
        r = roads[roads.intersects(geom)].clip(geom).assign(lcz_weak_label=klass, grid_id=row["grid_id"])
        building_rows.append(b)
        road_rows.append(r)

    path = OUT_DIR / "panel_e_h_texture_tiles.gpkg"
    if path.exists():
        path.unlink()
    pd.concat(cell_rows, ignore_index=True).pipe(gpd.GeoDataFrame, geometry="geometry", crs=grid.crs).to_file(
        path, layer="cells", driver="GPKG"
    )
    pd.concat(building_rows, ignore_index=True).pipe(gpd.GeoDataFrame, geometry="geometry", crs=grid.crs).to_file(
        path, layer="buildings", driver="GPKG"
    )
    pd.concat(road_rows, ignore_index=True).pipe(gpd.GeoDataFrame, geometry="geometry", crs=grid.crs).to_file(
        path, layer="roads", driver="GPKG"
    )
    tiles.to_csv(OUT_DIR / "panel_e_h_texture_tile_stats.csv", index=False)
    return tiles


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = _load_grid_with_features()
    _panel_a(g)
    _panel_b(g)
    _panel_c(g)
    tiles = _panels_e_h(ROOT)

    manifest = {
        "figure": "figure_1_manual_design",
        "purpose": "Raw GIS/CSV inputs for manual (Figma) redesign of Figure 1. Excludes panel (d) RF OOF confidence: that is a model result and belongs to the results figures, not study area.",
        "crs": "EPSG:5253",
        "panels": {
            "a_study_frame": {
                "file": "panel_a_study_frame.gpkg",
                "layers": ["grid_250m", "study_boundary", "coastline"],
                "sources": [str(SOURCES["grid"]), str(SOURCES["boundary"]), str(SOURCES["coastline"])],
            },
            "b_weak_lcz": {
                "file": "panel_b_weak_lcz.gpkg",
                "layers": ["grid_250m_weak_lcz"],
                "sources": [str(SOURCES["grid"]), str(SOURCES["features"])],
                "note": "lcz_weak_label/lcz_weak_confidence are the global WUDAPT-derived weak-label product, not model output.",
            },
            "c_lst": {
                "file": "panel_c_lst.gpkg",
                "layers": ["grid_250m_lst"],
                "sources": [str(SOURCES["grid"]), str(SOURCES["features"])],
                "note": "lst_c_median_mean is the Landsat 8/9 summer 2021-2025 median composite, not model output.",
            },
            "e_h_texture_tiles": {
                "file": "panel_e_h_texture_tiles.gpkg",
                "layers": ["cells", "buildings", "roads"],
                "stats_csv": "panel_e_h_texture_tile_stats.csv",
                "sources": [str(SOURCES["grid"]), str(SOURCES["buildings_roads"]), str(SOURCES["features"])],
                "selection_rule": "Deterministic class-median cell among confidence>=0.80 eligible-core building-bearing cells, matching scripts/build_figure_1_texture_atlas.py.",
                "grid_ids": tiles[["lcz_weak_label", "grid_id", "district"]].to_dict(orient="records"),
            },
        },
        "excluded_panel_d": "Spatial LightGBM out-of-fold maximum probability; already covered by Figure 5 and Figure 7 results.",
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
