"""Download OSM functional-context proxies and aggregate them to the grid.

This is a proxy layer, not an official zoning/land-use plan. It is intended to
flag industrial/port/airport, green-blue, commercial/institutional, and
transport-terminal contexts when municipal plan layers are not used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# On Windows with OSGeo4W, PROJ_LIB / GDAL_DATA may need to be set before
# importing GDAL-dependent packages. If your install lives elsewhere, set
# these environment variables before running the script.
if os.name == "nt":
    _proj = os.environ.get("PROJ_LIB", "")
    _gdal = os.environ.get("GDAL_DATA", "")
    _proj_default = r"C:\OSGeo4W\share\proj"
    _gdal_default = r"C:\OSGeo4W\share\gdal"
    if not _proj and os.path.isdir(_proj_default):
        os.environ["PROJ_LIB"] = _proj_default
    if not _gdal and os.path.isdir(_gdal_default):
        os.environ["GDAL_DATA"] = _gdal_default

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CONTEXT_CLASSES = [
    "airport",
    "industrial_port",
    "commercial_retail",
    "institutional_public",
    "green_open",
    "blue_water",
    "transport_terminal",
    "military_quarry",
    "construction_brownfield",
    "other_context",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-area", default="data/02_interim/study_area_fua.gpkg")
    parser.add_argument("--study-layer", default="study_area_fua")
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--raw-json", default="data/01_raw/osm/osm_functional_context_overpass.json")
    parser.add_argument("--out-gpkg", default="data/02_interim/osm_functional_context_fua.gpkg")
    parser.add_argument("--out-grid-gpkg", default="data/03_processed/grid_osm_functional_context.gpkg")
    parser.add_argument("--out-grid-layer", default="grid_250m_osm_functional_context")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_osm_functional_context.csv")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def load_study_area(path: str, layer: str) -> gpd.GeoDataFrame:
    study = gpd.read_file(path, layer=layer)
    if study.crs is None:
        raise RuntimeError("Study area CRS is missing")
    return study


def bbox_wgs84(study: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    study4326 = study.to_crs(4326)
    minx, miny, maxx, maxy = study4326.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    box = f"({south},{west},{north},{east})"
    return f"""
[out:json][timeout:240];
(
  nwr["landuse"~"^(industrial|commercial|retail|military|quarry|brownfield|construction|depot|garages|railway|grass|meadow|orchard|vineyard)$"]{box};
  nwr["aeroway"]{box};
  nwr["leisure"~"^(park|garden|recreation_ground|pitch|sports_centre|nature_reserve|marina)$"]{box};
  nwr["natural"~"^(wood|scrub|grassland|water|wetland|beach)$"]{box};
  nwr["amenity"~"^(university|school|college|hospital|clinic|bus_station|ferry_terminal|parking|marketplace)$"]{box};
  nwr["shop"~"^(mall|supermarket|department_store)$"]{box};
  nwr["industrial"]{box};
  nwr["harbour"]{box};
  nwr["seamark:type"~"^(harbour|port|dock|terminal|marina)$"]{box};
  nwr["man_made"~"^(pier|breakwater|wastewater_plant|works|storage_tank|silo)$"]{box};
  nwr["railway"~"^(station|yard|halt|tram_stop)$"]{box};
  nwr["public_transport"~"^(station|stop_position|platform)$"]{box};
);
out body geom;
"""


def download_overpass(query: str, out_path: Path, force: bool) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={
            "User-Agent": "izmir-lcz-deep-morphology/0.1 (research; contact yusuf.eminoglu@deu.edu.tr)",
            "Accept": "application/json",
        },
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def classify(tags: dict[str, Any]) -> tuple[str, str, str]:
    lower = {str(k): str(v) for k, v in tags.items()}

    def has(key: str, values: set[str] | None = None) -> bool:
        if key not in lower:
            return False
        return values is None or lower[key] in values

    if has("aeroway"):
        return "airport", "aeroway", lower["aeroway"]
    if has("landuse", {"industrial"}) or has("industrial") or has("harbour"):
        key = "landuse" if "landuse" in lower else ("industrial" if "industrial" in lower else "harbour")
        return "industrial_port", key, lower[key]
    if has("seamark:type", {"harbour", "port", "dock", "terminal", "marina"}):
        return "industrial_port", "seamark:type", lower["seamark:type"]
    if has("man_made", {"works", "storage_tank", "silo"}):
        return "industrial_port", "man_made", lower["man_made"]
    if has("landuse", {"commercial", "retail"}) or has("shop") or has("amenity", {"marketplace", "parking"}):
        key = "landuse" if "landuse" in lower else ("shop" if "shop" in lower else "amenity")
        return "commercial_retail", key, lower[key]
    if has("amenity", {"university", "school", "college", "hospital", "clinic"}):
        return "institutional_public", "amenity", lower["amenity"]
    if has("leisure") or has("landuse", {"grass", "meadow", "orchard", "vineyard"}) or has(
        "natural", {"wood", "scrub", "grassland", "wetland", "beach"}
    ):
        key = "leisure" if "leisure" in lower else ("landuse" if "landuse" in lower else "natural")
        return "green_open", key, lower[key]
    if has("natural", {"water"}):
        return "blue_water", "natural", lower["natural"]
    if has("amenity", {"bus_station", "ferry_terminal"}) or has("railway") or has("public_transport") or has(
        "man_made", {"pier", "breakwater"}
    ):
        key = (
            "amenity"
            if "amenity" in lower
            else ("railway" if "railway" in lower else ("public_transport" if "public_transport" in lower else "man_made"))
        )
        return "transport_terminal", key, lower[key]
    if has("landuse", {"military", "quarry"}):
        return "military_quarry", "landuse", lower["landuse"]
    if has("landuse", {"construction", "brownfield", "depot", "garages", "railway"}):
        return "construction_brownfield", "landuse", lower["landuse"]
    return "other_context", "", ""


def coords_to_linestring(coords: list[dict[str, float]]) -> LineString | None:
    if not coords or len(coords) < 2:
        return None
    return LineString([(pt["lon"], pt["lat"]) for pt in coords])


def element_geometry(element: dict[str, Any]):
    element_type = element.get("type")
    if element_type == "node":
        if "lon" not in element or "lat" not in element:
            return None
        return Point(float(element["lon"]), float(element["lat"]))

    if element_type == "way":
        line = coords_to_linestring(element.get("geometry", []))
        if line is None:
            return None
        coords = list(line.coords)
        if len(coords) >= 4 and coords[0] == coords[-1]:
            polygon = Polygon(coords)
            return polygon if polygon.is_valid else polygon.buffer(0)
        return line

    if element_type == "relation":
        rings = []
        lines = []
        for member in element.get("members", []):
            if member.get("type") != "way" or member.get("role") not in {"outer", ""}:
                continue
            line = coords_to_linestring(member.get("geometry", []))
            if line is None:
                continue
            coords = list(line.coords)
            if len(coords) >= 4 and coords[0] == coords[-1]:
                rings.append(Polygon(coords))
            else:
                lines.append(line)
        polygons = [poly if poly.is_valid else poly.buffer(0) for poly in rings]
        polygons.extend(list(polygonize(lines)))
        polygons = [poly for poly in polygons if not poly.is_empty]
        if not polygons:
            return None
        merged = unary_union(polygons)
        if isinstance(merged, (Polygon, MultiPolygon)):
            return merged
        return None

    return None


def elements_to_gdf(data: dict[str, Any]) -> gpd.GeoDataFrame:
    records = []
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        geom = element_geometry(element)
        if geom is None or geom.is_empty:
            continue
        context_class, matched_key, matched_value = classify(tags)
        records.append(
            {
                "osm_type": element.get("type"),
                "osm_id": int(element.get("id")),
                "context_class": context_class,
                "matched_key": matched_key,
                "matched_value": matched_value,
                "name": str(tags.get("name", "")),
                "tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=4326)


def clip_and_save(gdf: gpd.GeoDataFrame, study: gpd.GeoDataFrame, out_gpkg: Path) -> dict[str, int]:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    study_proj = study.to_crs(5253)
    gdf_proj = gdf.to_crs(5253)
    study_geom = study_proj.geometry.union_all()

    counts: dict[str, int] = {}
    for geom_type, layer_name in [
        (["Polygon", "MultiPolygon"], "osm_context_polygons"),
        (["LineString", "MultiLineString"], "osm_context_lines"),
        (["Point", "MultiPoint"], "osm_context_points"),
    ]:
        sub = gdf_proj[gdf_proj.geometry.geom_type.isin(geom_type)].copy()
        if sub.empty:
            empty = gpd.GeoDataFrame(columns=list(gdf_proj.columns), geometry="geometry", crs=5253)
            empty.to_file(out_gpkg, layer=layer_name, driver="GPKG")
            counts[layer_name] = 0
            continue
        sub["geometry"] = sub.geometry.intersection(study_geom)
        sub = sub[~sub.geometry.is_empty & sub.geometry.notna()].copy()
        sub.to_file(out_gpkg, layer=layer_name, driver="GPKG")
        counts[layer_name] = len(sub)
    return counts


def init_grid_rows(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = grid.drop(columns="geometry").copy()
    for cls in CONTEXT_CLASSES:
        rows[f"osm_area_{cls}_m2"] = 0.0
        rows[f"osm_share_{cls}"] = 0.0
        rows[f"osm_flag_{cls}"] = 0
        rows[f"osm_line_length_{cls}_m"] = 0.0
        rows[f"osm_point_count_{cls}"] = 0
    return rows


def aggregate_polygons(rows: pd.DataFrame, grid: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> None:
    if polygons.empty:
        return
    candidate = gpd.sjoin(
        polygons[["context_class", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if candidate.empty:
        return
    areas: dict[tuple[str, str], float] = {}
    for idx, row in candidate.iterrows():
        poly = polygons.loc[idx, "geometry"]
        grid_geom = grid.loc[row["index_right"], "geometry"]
        area = poly.intersection(grid_geom).area
        if area > 0:
            key = (row["grid_id"], row["context_class"])
            areas[key] = areas.get(key, 0.0) + float(area)
    for (grid_id, cls), area in areas.items():
        selector = rows["grid_id"] == grid_id
        rows.loc[selector, f"osm_area_{cls}_m2"] = area
        cell_area = float(rows.loc[selector, "cell_area_m2"].iloc[0])
        rows.loc[selector, f"osm_share_{cls}"] = min(area / cell_area, 1.0) if cell_area else 0.0
        rows.loc[selector, f"osm_flag_{cls}"] = 1


def aggregate_lines(rows: pd.DataFrame, grid: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> None:
    if lines.empty:
        return
    candidate = gpd.sjoin(
        lines[["context_class", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if candidate.empty:
        return
    lengths: dict[tuple[str, str], float] = {}
    for idx, row in candidate.iterrows():
        line = lines.loc[idx, "geometry"]
        grid_geom = grid.loc[row["index_right"], "geometry"]
        length = line.intersection(grid_geom).length
        if length > 0:
            key = (row["grid_id"], row["context_class"])
            lengths[key] = lengths.get(key, 0.0) + float(length)
    for (grid_id, cls), length in lengths.items():
        selector = rows["grid_id"] == grid_id
        rows.loc[selector, f"osm_line_length_{cls}_m"] = length
        rows.loc[selector, f"osm_flag_{cls}"] = 1


def aggregate_points(rows: pd.DataFrame, grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> None:
    if points.empty:
        return
    joined = gpd.sjoin(
        points[["context_class", "geometry"]],
        grid[["grid_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return
    grouped = joined.groupby(["grid_id", "context_class"]).size()
    for (grid_id, cls), count in grouped.items():
        selector = rows["grid_id"] == grid_id
        rows.loc[selector, f"osm_point_count_{cls}"] = int(count)
        rows.loc[selector, f"osm_flag_{cls}"] = 1


def write_grid_output(grid: gpd.GeoDataFrame, rows: pd.DataFrame, out_gpkg: Path, layer: str, out_csv: Path) -> None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_gdf = grid[["grid_id", "geometry"]].merge(rows, on="grid_id", how="left")
    out_gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    rows.to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    study = load_study_area(args.study_area, args.study_layer)
    bbox = bbox_wgs84(study)
    query = overpass_query(bbox)
    raw = download_overpass(query, Path(args.raw_json), args.force_download)
    gdf = elements_to_gdf(raw)
    if gdf.empty:
        raise RuntimeError("No OSM context geometries were parsed")

    out_gpkg = Path(args.out_gpkg)
    layer_counts = clip_and_save(gdf, study, out_gpkg)

    grid = gpd.read_file(args.grid_gpkg, layer=args.grid_layer)
    if grid.crs is None:
        raise RuntimeError("Grid CRS is missing")
    if grid.crs.to_epsg() != 5253:
        grid = grid.to_crs(5253)

    polygons = gpd.read_file(out_gpkg, layer="osm_context_polygons")
    lines = gpd.read_file(out_gpkg, layer="osm_context_lines")
    points = gpd.read_file(out_gpkg, layer="osm_context_points")
    rows = init_grid_rows(grid)
    aggregate_polygons(rows, grid, polygons)
    aggregate_lines(rows, grid, lines)
    aggregate_points(rows, grid, points)
    write_grid_output(grid, rows, Path(args.out_grid_gpkg), args.out_grid_layer, Path(args.out_csv))

    flagged = {cls: int(rows[f"osm_flag_{cls}"].sum()) for cls in CONTEXT_CLASSES}
    print(f"overpass_elements={len(raw.get('elements', []))}")
    print(f"parsed_features={len(gdf)}")
    print(f"layer_counts={layer_counts}")
    print(f"grid_cells={len(rows)}")
    print(f"flagged_cells={flagged}")
    print(f"raw_json={args.raw_json}")
    print(f"context_gpkg={args.out_gpkg}")
    print(f"grid_gpkg={args.out_grid_gpkg}")
    print(f"grid_csv={args.out_csv}")


if __name__ == "__main__":
    main()
