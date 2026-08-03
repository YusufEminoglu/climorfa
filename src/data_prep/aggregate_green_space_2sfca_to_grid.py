"""Build a network-based two-step floating catchment area green-space metric.

The observation unit remains the 250 m analysis grid. Demand is 2024 mahalle
population allocated to grid cells in proportion to a residential floor-area
proxy. Supply is the mapped area of OSM leisure=park/garden polygons. Travel
cost is shortest-path distance on a planarized municipal road-centreline proxy
after explicitly excluding YOLTIP=OTOYOL.

This is a planning/accessibility proxy. It is not a pedestrian routing product,
an official public-green-space inventory, or a substitute for grid-local street
morphology metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from osgeo import ogr, osr
from scipy.spatial import cKDTree
from shapely import from_wkb, get_parts, unary_union
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


ogr.UseExceptions()
osr.UseExceptions()

DEFAULT_COSTS = [400.0, 800.0, 1200.0]
DEFAULT_GREEN_VALUES = ["park", "garden"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--model-csv", default="data/03_processed/grid_250m_model_features_v7.csv")
    parser.add_argument(
        "--population-gpkg",
        default="data/01_raw/population/population_controls.gpkg",
    )
    parser.add_argument("--population-layer", default="mahalle_population_controls_2024")
    parser.add_argument("--population-field", default="pop_total")
    parser.add_argument("--population-key-field", default="join_key")
    parser.add_argument("--roads-gpkg", default="data/02_interim/buildings_roads_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--road-type-field", default="YOLTIP")
    parser.add_argument("--exclude-road-types", nargs="*", default=["OTOYOL"])
    parser.add_argument("--osm-gpkg", default="data/02_interim/osm_functional_context_fua.gpkg")
    parser.add_argument("--osm-polygon-layer", default="osm_context_polygons")
    parser.add_argument("--green-values", nargs="+", default=DEFAULT_GREEN_VALUES)
    parser.add_argument("--network-costs-m", nargs="+", type=float, default=DEFAULT_COSTS)
    parser.add_argument("--node-grid-size-m", type=float, default=0.05)
    parser.add_argument("--max-origin-snap-m", type=float, default=500.0)
    parser.add_argument("--max-supply-snap-m", type=float, default=300.0)
    parser.add_argument("--max-grid-cells", type=int)
    parser.add_argument("--max-supplies", type=int)
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_green_space_2sfca.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_green_space_2sfca")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_green_space_2sfca.csv")
    parser.add_argument("--out-supply-gpkg", default="data/03_processed/green_space_2sfca_supply_audit.gpkg")
    parser.add_argument("--out-supply-layer", default="green_space_2sfca_supply_audit")
    parser.add_argument("--out-manifest", default="data/03_processed/grid_green_space_2sfca_manifest.json")
    return parser.parse_args()


def cost_label(cost_m: float) -> str:
    return f"{int(cost_m)}m" if float(cost_m).is_integer() else f"{str(cost_m).replace('.', 'p')}m"


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def shapely_geometry(geom: ogr.Geometry):
    return from_wkb(bytes(geom.ExportToWkb()))


def read_model_rows(path: str) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"grid_id", "floor_area_proxy_m2", "type_mesken_area_share"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Missing demand-proxy fields in {path}: {sorted(required)}")
        return {str(row["grid_id"]): row for row in reader}


def as_nonnegative_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


def read_grid_rows(
    grid_layer: ogr.Layer,
    model_rows: dict[str, dict[str, str]],
    max_cells: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_layer.ResetReading()
    for feature in grid_layer:
        grid_id = str(feature.GetField("grid_id"))
        geom_ref = feature.GetGeometryRef()
        if geom_ref is None or geom_ref.IsEmpty() or grid_id not in model_rows:
            continue
        geom = geom_ref.Clone()
        centre = geom.PointOnSurface()
        model_row = model_rows[grid_id]
        floor_area = as_nonnegative_float(model_row.get("floor_area_proxy_m2"))
        residential_share = min(as_nonnegative_float(model_row.get("type_mesken_area_share")), 1.0)
        rows.append(
            {
                "grid_id": grid_id,
                "ogr_geometry": geom,
                "shapely_geometry": shapely_geometry(geom),
                "x": float(centre.GetX()),
                "y": float(centre.GetY()),
                "residential_floor_area_proxy_m2": floor_area * residential_share,
            }
        )
        if max_cells is not None and len(rows) >= max_cells:
            break
    grid_layer.ResetReading()
    return rows


def coordinate_transform(source: osr.SpatialReference, target: osr.SpatialReference) -> osr.CoordinateTransformation:
    source = source.Clone()
    target = target.Clone()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(source, target)


def allocate_population(
    grid_rows: list[dict[str, Any]],
    population_layer: ogr.Layer,
    grid_srs: osr.SpatialReference,
    population_field: str,
    key_field: str,
) -> dict[str, Any]:
    pop_srs = population_layer.GetSpatialRef()
    if pop_srs is None:
        raise RuntimeError("Population layer has no CRS")
    transform = coordinate_transform(pop_srs, grid_srs)
    pop_geometries = []
    pop_records: list[dict[str, Any]] = []
    population_layer.ResetReading()
    for feature in population_layer:
        geom_ref = feature.GetGeometryRef()
        population = as_nonnegative_float(feature.GetField(population_field))
        if geom_ref is None or geom_ref.IsEmpty() or population <= 0:
            continue
        geom = geom_ref.Clone()
        geom.Transform(transform)
        shp = shapely_geometry(geom)
        pop_geometries.append(shp)
        pop_records.append(
            {
                "key": str(feature.GetField(key_field) or ""),
                "population": population,
                "area": float(shp.area),
            }
        )
    population_layer.ResetReading()
    tree = STRtree(pop_geometries)

    grid_indices_by_pop: dict[int, list[int]] = defaultdict(list)
    unmatched_grid_cells = 0
    for grid_index, row in enumerate(grid_rows):
        point = Point(row["x"], row["y"])
        matches = list(tree.query(point, predicate="within"))
        if not matches:
            matches = list(tree.query(point, predicate="intersects"))
        if not matches:
            row["population_polygon_index"] = None
            row["population_mahalle_key"] = ""
            row["population_2024_dasymetric"] = 0.0
            row["population_allocation_method"] = "outside_population_geometry"
            unmatched_grid_cells += 1
            continue
        pop_index = min((int(index) for index in matches), key=lambda index: pop_records[index]["area"])
        row["population_polygon_index"] = pop_index
        row["population_mahalle_key"] = pop_records[pop_index]["key"]
        grid_indices_by_pop[pop_index].append(grid_index)

    allocated_population = 0.0
    floor_weighted_neighbourhoods = 0
    equal_share_neighbourhoods = 0
    for pop_index, grid_indices in grid_indices_by_pop.items():
        population = pop_records[pop_index]["population"]
        floor_total = sum(grid_rows[index]["residential_floor_area_proxy_m2"] for index in grid_indices)
        if floor_total > 0:
            floor_weighted_neighbourhoods += 1
            for grid_index in grid_indices:
                row = grid_rows[grid_index]
                row["population_2024_dasymetric"] = population * row["residential_floor_area_proxy_m2"] / floor_total
                row["population_allocation_method"] = "residential_floor_area_proxy"
        else:
            equal_share_neighbourhoods += 1
            share = population / len(grid_indices)
            for grid_index in grid_indices:
                grid_rows[grid_index]["population_2024_dasymetric"] = share
                grid_rows[grid_index]["population_allocation_method"] = "equal_grid_fallback"
        allocated_population += population

    return {
        "population_polygons_with_grid_centres": len(grid_indices_by_pop),
        "population_total_allocated_to_grid": allocated_population,
        "floor_weighted_neighbourhoods": floor_weighted_neighbourhoods,
        "equal_share_neighbourhoods": equal_share_neighbourhoods,
        "grid_cells_outside_population_geometry": unmatched_grid_cells,
    }


def iter_line_parts(geom: ogr.Geometry) -> Iterable[ogr.Geometry]:
    name = geom.GetGeometryName().upper()
    if name in {"LINESTRING", "LINEARRING"}:
        yield geom
        return
    if name in {"MULTILINESTRING", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_line_parts(child)


def build_planarized_graph(
    roads_layer: ogr.Layer,
    road_type_field: str,
    excluded_types: set[str],
    grid_size_m: float,
) -> tuple[nx.Graph, np.ndarray, np.ndarray, dict[str, Any]]:
    lines: list[LineString] = []
    source_parts = 0
    excluded_parts = 0
    roads_layer.ResetReading()
    for feature in roads_layer:
        road_type_raw = feature.GetField(road_type_field)
        road_type = str(road_type_raw).strip().upper() if road_type_raw not in (None, "") else ""
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        for part in iter_line_parts(geom):
            source_parts += 1
            if road_type in excluded_types:
                excluded_parts += 1
                continue
            shp = shapely_geometry(part)
            if isinstance(shp, LineString) and not shp.is_empty and shp.length > 0:
                lines.append(shp)
    roads_layer.ResetReading()
    if not lines:
        raise RuntimeError("No usable road lines after filtering")

    print(f"planarizing_road_parts={len(lines)}", flush=True)
    noded = unary_union(lines, grid_size=grid_size_m)
    graph = nx.Graph()
    node_ids: dict[tuple[float, float], int] = {}

    def node_id(coord: tuple[float, float]) -> int:
        key = (round(float(coord[0]) / grid_size_m) * grid_size_m, round(float(coord[1]) / grid_size_m) * grid_size_m)
        if key not in node_ids:
            node_ids[key] = len(node_ids)
        return node_ids[key]

    noded_parts = 0
    noded_segments = 0
    for part in get_parts(noded):
        if not isinstance(part, LineString) or part.is_empty or part.length <= 0:
            continue
        coordinates = list(part.coords)
        if len(coordinates) < 2:
            continue
        noded_parts += 1
        # Retain every source vertex as a graph node. Using only the endpoints
        # of a noded polyline makes points beside a long curving edge appear
        # hundreds of metres from the network even when the line is adjacent.
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            u = node_id(start)
            v = node_id(end)
            if u == v:
                continue
            length = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            if length <= 0:
                continue
            existing = graph.get_edge_data(u, v)
            if existing is None or length < float(existing["length_m"]):
                graph.add_edge(u, v, length_m=length)
            noded_segments += 1

    coordinates = np.empty((len(node_ids), 2), dtype=float)
    for coord, identifier in node_ids.items():
        coordinates[identifier] = coord
        graph.add_node(identifier, x=coord[0], y=coord[1])
    ids = np.arange(len(node_ids), dtype=int)
    components = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    audit = {
        "source_road_parts": source_parts,
        "excluded_road_parts": excluded_parts,
        "noded_road_parts": noded_parts,
        "noded_road_segments": noded_segments,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_components": len(components),
        "largest_component_nodes": components[0] if components else 0,
        "largest_component_share": components[0] / graph.number_of_nodes() if components and graph.number_of_nodes() else 0.0,
    }
    return graph, ids, coordinates, audit


def read_green_supplies(
    osm_layer: ogr.Layer,
    green_values: set[str],
    max_supplies: int | None,
) -> list[dict[str, Any]]:
    supplies: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    osm_layer.ResetReading()
    for feature in osm_layer:
        if str(feature.GetField("matched_key") or "").lower() != "leisure":
            continue
        matched_value = str(feature.GetField("matched_value") or "").lower()
        if matched_value not in green_values:
            continue
        try:
            tags = json.loads(str(feature.GetField("tags_json") or "{}"))
        except json.JSONDecodeError:
            tags = {}
        if str(tags.get("access", "")).lower() in {"private", "no"}:
            continue
        osm_type = str(feature.GetField("osm_type") or "")
        osm_id = str(feature.GetField("osm_id") or "")
        key = (osm_type, osm_id)
        if key in seen:
            continue
        geom_ref = feature.GetGeometryRef()
        if geom_ref is None or geom_ref.IsEmpty():
            continue
        geom = geom_ref.Clone()
        area = float(geom.Area())
        if area <= 0:
            continue
        supplies.append(
            {
                "supply_id": f"{osm_type}/{osm_id}",
                "osm_type": osm_type,
                "osm_id": osm_id,
                "green_type": matched_value,
                "name": str(feature.GetField("name") or ""),
                "supply_area_m2": area,
                "ogr_geometry": geom,
                "shapely_geometry": shapely_geometry(geom),
            }
        )
        seen.add(key)
        if max_supplies is not None and len(supplies) >= max_supplies:
            break
    osm_layer.ResetReading()
    return supplies


def snap_origins(
    grid_rows: list[dict[str, Any]],
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    max_snap_m: float,
) -> dict[str, Any]:
    tree = cKDTree(coordinates)
    points = np.array([[row["x"], row["y"]] for row in grid_rows], dtype=float)
    distances, indices = tree.query(points, k=1)
    available = 0
    for row, distance, index in zip(grid_rows, distances, indices):
        row["network_snap_distance_m"] = float(distance)
        if float(distance) <= max_snap_m:
            row["network_node_id"] = int(node_ids[int(index)])
            available += 1
        else:
            row["network_node_id"] = None
    return {
        "origin_network_available_cells": available,
        "origin_network_unavailable_cells": len(grid_rows) - available,
        "origin_snap_distance_median": float(np.median(distances)),
        "origin_snap_distance_p95": float(np.quantile(distances, 0.95)),
        "origin_snap_distance_max": float(np.max(distances)),
    }


def snap_supplies(
    supplies: list[dict[str, Any]],
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    max_snap_m: float,
) -> dict[str, Any]:
    node_points = [Point(float(x), float(y)) for x, y in coordinates]
    tree = STRtree(node_points)
    available = 0
    snap_distances: list[float] = []
    for supply in supplies:
        indices, distances = tree.query_nearest(supply["shapely_geometry"], return_distance=True)
        node_index = int(np.atleast_1d(indices)[0])
        distance = float(np.atleast_1d(distances)[0])
        supply["network_snap_distance_m"] = distance
        snap_distances.append(distance)
        if distance <= max_snap_m:
            supply["network_node_id"] = int(node_ids[node_index])
            available += 1
        else:
            supply["network_node_id"] = None
    return {
        "green_supplies_total": len(supplies),
        "green_supplies_network_available": available,
        "green_supplies_network_unavailable": len(supplies) - available,
        "green_supply_area_m2_total": sum(row["supply_area_m2"] for row in supplies),
        "green_supply_area_m2_network_available": sum(
            row["supply_area_m2"] for row in supplies if row["network_node_id"] is not None
        ),
        "supply_snap_distance_median": float(np.median(snap_distances)) if snap_distances else None,
        "supply_snap_distance_p95": float(np.quantile(snap_distances, 0.95)) if snap_distances else None,
        "supply_snap_distance_max": max(snap_distances) if snap_distances else None,
    }


def calculate_2sfca(
    grid_rows: list[dict[str, Any]],
    supplies: list[dict[str, Any]],
    graph: nx.Graph,
    costs: list[float],
) -> dict[str, Any]:
    origins_by_node: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(grid_rows):
        node = row["network_node_id"]
        if node is not None:
            origins_by_node[int(node)].append(index)
        for cost in costs:
            label = cost_label(cost)
            row[f"green_2sfca_{label}_access_m2_per_person"] = 0.0 if node is not None else None
            row[f"green_2sfca_{label}_accessible_supply_count"] = 0 if node is not None else None
            row[f"green_2sfca_{label}_accessible_supply_area_m2"] = 0.0 if node is not None else None

    used_supplies_by_cost = {cost: 0 for cost in costs}
    zero_demand_supplies_by_cost = {cost: 0 for cost in costs}
    max_cost = max(costs)
    usable_supplies = [row for row in supplies if row["network_node_id"] is not None]

    for supply_index, supply in enumerate(usable_supplies, start=1):
        if supply_index % 100 == 0:
            print(f"processed_2sfca_supplies={supply_index}/{len(usable_supplies)}", flush=True)
        supply_snap = float(supply["network_snap_distance_m"])
        cutoff = max_cost - supply_snap
        if cutoff < 0:
            continue
        distances = nx.single_source_dijkstra_path_length(
            graph,
            int(supply["network_node_id"]),
            cutoff=cutoff,
            weight="length_m",
        )
        reachable_by_cost: dict[float, list[int]] = {cost: [] for cost in costs}
        denominator_by_cost = {cost: 0.0 for cost in costs}
        for node, network_distance in distances.items():
            for grid_index in origins_by_node.get(int(node), []):
                row = grid_rows[grid_index]
                total_distance = supply_snap + float(network_distance) + float(row["network_snap_distance_m"])
                for cost in costs:
                    if total_distance <= cost:
                        reachable_by_cost[cost].append(grid_index)
                        denominator_by_cost[cost] += float(row["population_2024_dasymetric"])

        supply["catchment_population"] = {}
        supply["supply_ratio_m2_per_person"] = {}
        for cost in costs:
            denominator = denominator_by_cost[cost]
            supply["catchment_population"][cost] = denominator
            if denominator <= 0:
                supply["supply_ratio_m2_per_person"][cost] = None
                zero_demand_supplies_by_cost[cost] += 1
                continue
            ratio = float(supply["supply_area_m2"]) / denominator
            supply["supply_ratio_m2_per_person"][cost] = ratio
            used_supplies_by_cost[cost] += 1
            for grid_index in reachable_by_cost[cost]:
                row = grid_rows[grid_index]
                label = cost_label(cost)
                row[f"green_2sfca_{label}_access_m2_per_person"] += ratio
                row[f"green_2sfca_{label}_accessible_supply_count"] += 1
                row[f"green_2sfca_{label}_accessible_supply_area_m2"] += float(supply["supply_area_m2"])

    return {
        "usable_green_supplies": len(usable_supplies),
        "used_supplies_by_cost": {cost_label(cost): used_supplies_by_cost[cost] for cost in costs},
        "zero_demand_supplies_by_cost": {cost_label(cost): zero_demand_supplies_by_cost[cost] for cost in costs},
    }


def output_fields(costs: list[float]) -> list[str]:
    fields = [
        "population_mahalle_key",
        "population_2024_dasymetric",
        "population_allocation_method",
        "residential_floor_area_proxy_m2",
        "green_2sfca_network_available_flag",
        "green_2sfca_network_snap_distance_m",
    ]
    for cost in costs:
        label = cost_label(cost)
        fields.extend(
            [
                f"green_2sfca_{label}_access_m2_per_person",
                f"green_2sfca_{label}_access_log1p",
                f"green_2sfca_{label}_accessible_supply_count",
                f"green_2sfca_{label}_accessible_supply_area_m2",
            ]
        )
    return fields


def row_output(row: dict[str, Any], costs: list[float]) -> dict[str, Any]:
    result = {
        "population_mahalle_key": row["population_mahalle_key"],
        "population_2024_dasymetric": row["population_2024_dasymetric"],
        "population_allocation_method": row["population_allocation_method"],
        "residential_floor_area_proxy_m2": row["residential_floor_area_proxy_m2"],
        "green_2sfca_network_available_flag": 1 if row["network_node_id"] is not None else 0,
        "green_2sfca_network_snap_distance_m": row["network_snap_distance_m"],
    }
    for cost in costs:
        label = cost_label(cost)
        raw_access = row[f"green_2sfca_{label}_access_m2_per_person"]
        result[f"green_2sfca_{label}_access_log1p"] = math.log1p(raw_access) if raw_access is not None else None
        for suffix in ("access_m2_per_person", "accessible_supply_count", "accessible_supply_area_m2"):
            field = f"green_2sfca_{label}_{suffix}"
            result[field] = row[field]
    return result


def field_definition(name: str) -> ogr.FieldDefn:
    if name in {"population_mahalle_key", "population_allocation_method"}:
        field = ogr.FieldDefn(name, ogr.OFTString)
        field.SetWidth(120)
        return field
    if name.endswith("_flag") or name.endswith("_count"):
        return ogr.FieldDefn(name, ogr.OFTInteger64)
    return ogr.FieldDefn(name, ogr.OFTReal)


def write_grid_outputs(
    grid_rows: list[dict[str, Any]],
    grid_srs: osr.SpatialReference,
    costs: list[float],
    out_gpkg: Path,
    out_layer_name: str,
    out_csv: Path,
) -> None:
    fields = output_fields(costs)
    driver = ogr.GetDriverByName("GPKG")
    if out_gpkg.exists():
        driver.DeleteDataSource(str(out_gpkg))
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    ds = driver.CreateDataSource(str(out_gpkg))
    layer = ds.CreateLayer(out_layer_name, srs=grid_srs, geom_type=ogr.wkbPolygon)
    grid_id_field = ogr.FieldDefn("grid_id", ogr.OFTString)
    grid_id_field.SetWidth(40)
    layer.CreateField(grid_id_field)
    for name in fields:
        layer.CreateField(field_definition(name))
    definition = layer.GetLayerDefn()
    for row in grid_rows:
        values = row_output(row, costs)
        feature = ogr.Feature(definition)
        feature.SetField("grid_id", row["grid_id"])
        for name, value in values.items():
            if value is not None:
                feature.SetField(name, value)
        feature.SetGeometry(row["ogr_geometry"])
        layer.CreateFeature(feature)
        feature = None
    layer.SyncToDisk()
    ds = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_id", *fields])
        writer.writeheader()
        for row in grid_rows:
            writer.writerow({"grid_id": row["grid_id"], **row_output(row, costs)})


def write_supply_audit(
    supplies: list[dict[str, Any]],
    srs: osr.SpatialReference,
    costs: list[float],
    out_path: Path,
    layer_name: str,
) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        driver.DeleteDataSource(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = driver.CreateDataSource(str(out_path))
    layer = ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbUnknown)
    for name, field_type, width in (
        ("supply_id", ogr.OFTString, 80),
        ("green_type", ogr.OFTString, 30),
        ("name", ogr.OFTString, 160),
        ("supply_area_m2", ogr.OFTReal, 0),
        ("network_available_flag", ogr.OFTInteger, 0),
        ("network_snap_distance_m", ogr.OFTReal, 0),
    ):
        field = ogr.FieldDefn(name, field_type)
        if width:
            field.SetWidth(width)
        layer.CreateField(field)
    for cost in costs:
        layer.CreateField(ogr.FieldDefn(f"pop_{cost_label(cost)}", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn(f"ratio_{cost_label(cost)}", ogr.OFTReal))
    definition = layer.GetLayerDefn()
    for row in supplies:
        feature = ogr.Feature(definition)
        feature.SetField("supply_id", row["supply_id"])
        feature.SetField("green_type", row["green_type"])
        feature.SetField("name", row["name"])
        feature.SetField("supply_area_m2", row["supply_area_m2"])
        feature.SetField("network_available_flag", 1 if row["network_node_id"] is not None else 0)
        feature.SetField("network_snap_distance_m", row["network_snap_distance_m"])
        for cost in costs:
            population = row.get("catchment_population", {}).get(cost)
            ratio = row.get("supply_ratio_m2_per_person", {}).get(cost)
            if population is not None:
                feature.SetField(f"pop_{cost_label(cost)}", population)
            if ratio is not None:
                feature.SetField(f"ratio_{cost_label(cost)}", ratio)
        feature.SetGeometry(row["ogr_geometry"])
        layer.CreateFeature(feature)
        feature = None
    layer.SyncToDisk()
    ds = None


def summarize_outputs(grid_rows: list[dict[str, Any]], costs: list[float]) -> dict[str, Any]:
    population_total = sum(float(row["population_2024_dasymetric"]) for row in grid_rows)
    population_network_available = sum(
        float(row["population_2024_dasymetric"])
        for row in grid_rows
        if row["network_node_id"] is not None
    )
    populated_cells = sum(1 for row in grid_rows if float(row["population_2024_dasymetric"]) > 0)
    populated_available_cells = sum(
        1
        for row in grid_rows
        if float(row["population_2024_dasymetric"]) > 0 and row["network_node_id"] is not None
    )
    summary: dict[str, Any] = {
        "population_total": population_total,
        "population_network_available": population_network_available,
        "population_network_available_share": population_network_available / population_total if population_total > 0 else None,
        "populated_cells": populated_cells,
        "populated_network_available_cells": populated_available_cells,
        "populated_network_available_share": populated_available_cells / populated_cells if populated_cells > 0 else None,
        "by_threshold": {},
    }
    for cost in costs:
        label = cost_label(cost)
        field = f"green_2sfca_{label}_access_m2_per_person"
        values = np.array([row[field] for row in grid_rows if row[field] is not None], dtype=float)
        population_with_positive_access = sum(
            float(row["population_2024_dasymetric"])
            for row in grid_rows
            if row[field] is not None and float(row[field]) > 0
        )
        summary["by_threshold"][label] = {
            "available_cells": int(len(values)),
            "positive_access_cells": int(np.sum(values > 0)),
            "access_m2_per_person_median": float(np.median(values)) if len(values) else None,
            "access_m2_per_person_p95": float(np.quantile(values, 0.95)) if len(values) else None,
            "access_m2_per_person_max": float(np.max(values)) if len(values) else None,
            "population_with_positive_access_share_of_network_available": (
                population_with_positive_access / population_network_available
                if population_network_available > 0
                else None
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    costs = sorted(set(float(value) for value in args.network_costs_m))
    if not costs or min(costs) <= 0:
        raise RuntimeError("Network costs must be positive")

    grid_ds, grid_layer = open_layer(args.grid_gpkg, args.grid_layer)
    population_ds, population_layer = open_layer(args.population_gpkg, args.population_layer)
    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)
    osm_ds, osm_layer = open_layer(args.osm_gpkg, args.osm_polygon_layer)
    grid_srs = grid_layer.GetSpatialRef()
    if grid_srs is None:
        raise RuntimeError("Grid layer has no CRS")

    model_rows = read_model_rows(args.model_csv)
    grid_rows = read_grid_rows(grid_layer, model_rows, args.max_grid_cells)
    population_audit = allocate_population(
        grid_rows,
        population_layer,
        grid_srs,
        args.population_field,
        args.population_key_field,
    )
    graph, node_ids, coordinates, graph_audit = build_planarized_graph(
        roads_layer,
        args.road_type_field,
        {value.strip().upper() for value in args.exclude_road_types},
        args.node_grid_size_m,
    )
    origin_audit = snap_origins(grid_rows, node_ids, coordinates, args.max_origin_snap_m)
    supplies = read_green_supplies(
        osm_layer,
        {value.strip().lower() for value in args.green_values},
        args.max_supplies,
    )
    supply_audit = snap_supplies(supplies, node_ids, coordinates, args.max_supply_snap_m)
    sfca_audit = calculate_2sfca(grid_rows, supplies, graph, costs)
    output_summary = summarize_outputs(grid_rows, costs)

    write_grid_outputs(
        grid_rows,
        grid_srs,
        costs,
        Path(args.out_gpkg),
        args.out_layer,
        Path(args.out_csv),
    )
    write_supply_audit(
        supplies,
        grid_srs,
        costs,
        Path(args.out_supply_gpkg),
        args.out_supply_layer,
    )

    manifest = {
        "schema_version": "climorfa.green_space_2sfca.v1",
        "method": "classic two-step floating catchment area calculated separately at each network-distance threshold",
        "grid": {"path": args.grid_gpkg, "layer": args.grid_layer, "rows": len(grid_rows)},
        "demand": {
            "population_path": args.population_gpkg,
            "population_layer": args.population_layer,
            "population_field": args.population_field,
            "allocation_weight": "floor_area_proxy_m2 * type_mesken_area_share",
            "allocation_rule": "mahalle totals allocated to grid point-on-surface locations in proportion to residential floor-area proxy; equal-grid fallback only where the neighbourhood proxy sum is zero",
        },
        "supply": {
            "path": args.osm_gpkg,
            "layer": args.osm_polygon_layer,
            "rule": "OSM polygon matched_key=leisure and matched_value in selected green values; access=private/no excluded",
            "green_values": sorted({value.strip().lower() for value in args.green_values}),
            "capacity": "mapped polygon area in square metres",
        },
        "network": {
            "path": args.roads_gpkg,
            "layer": args.roads_layer,
            "excluded_road_types": sorted({value.strip().upper() for value in args.exclude_road_types}),
            "topology": "planarized undirected road-centreline proxy",
            "thresholds_m": costs,
            "origin_snap_distance_is_added_to_cost": True,
            "supply_snap_distance_is_added_to_cost": True,
        },
        "formula": {
            "step_1": "R_j = S_j / sum(P_i for d_ij <= d0)",
            "step_2": "A_i = sum(R_j for d_ij <= d0)",
            "output_unit": "mapped green-space square metres per dasymetrically allocated person",
        },
        "guardrails": [
            "This is not an official public-green-space inventory.",
            "The municipal road centrelines are a pedestrian-network proxy and do not encode crossings, access restrictions, slope, or travel time.",
            "Planarization can connect grade-separated non-motorway crossings; YOLTIP=OTOYOL is excluded explicitly.",
            "OSM completeness and park access tags are uncertain.",
            "The 2024 population allocation uses a residential floor-area proxy, not address-level population.",
            "The 2SFCA feature is contextual and should not replace grid-local street morphology metrics.",
        ],
        "audit": {
            "population": population_audit,
            "graph": graph_audit,
            "origins": origin_audit,
            "supplies": supply_audit,
            "two_step_fca": sfca_audit,
            "outputs": output_summary,
        },
        "outputs": {
            "grid_gpkg": args.out_gpkg,
            "grid_csv": args.out_csv,
            "supply_audit_gpkg": args.out_supply_gpkg,
        },
    }
    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest["audit"], indent=2, ensure_ascii=False))
    print(f"out_csv={args.out_csv}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_supply_gpkg={args.out_supply_gpkg}")
    print(f"out_manifest={args.out_manifest}")

    grid_ds = None
    population_ds = None
    roads_ds = None
    osm_ds = None


if __name__ == "__main__":
    main()
