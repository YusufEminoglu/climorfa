"""Aggregate network-distance service-area metrics around grid-cell centroids."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from osgeo import ogr
from scipy.spatial import cKDTree

ogr.UseExceptions()


DEFAULT_COSTS = [250.0, 400.0, 800.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--roads-gpkg", default="data/02_interim/buildings_roads_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--road-type-field", default="YOLTIP")
    parser.add_argument("--exclude-road-types", nargs="*", default=["OTOYOL"])
    parser.add_argument("--network-costs-m", nargs="+", type=float, default=DEFAULT_COSTS)
    parser.add_argument("--node-snap-m", type=float, default=1.0)
    parser.add_argument("--max-centroid-snap-m", type=float, default=250.0)
    parser.add_argument("--orientation-bins", type=int, default=12)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_road_network_service_area_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_road_network_service_area_features")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_road_network_service_area_features.csv")
    return parser.parse_args()


def cost_label(cost_m: float) -> str:
    if float(cost_m).is_integer():
        return f"{int(cost_m)}m"
    return f"{str(cost_m).replace('.', 'p')}m"


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def iter_line_parts(geom: ogr.Geometry) -> Iterable[ogr.Geometry]:
    geom_name = geom.GetGeometryName().upper()
    if geom_name in {"LINESTRING", "LINEARRING"}:
        yield geom
        return
    if geom_name in {"MULTILINESTRING", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_line_parts(child)


def iter_segments(line: ogr.Geometry) -> Iterable[tuple[float, float, float, float]]:
    for index in range(max(line.GetPointCount() - 1, 0)):
        x1, y1, _ = line.GetPoint(index)
        x2, y2, _ = line.GetPoint(index + 1)
        yield float(x1), float(y1), float(x2), float(y2)


def node_key(x: float, y: float, snap_m: float) -> tuple[int, int]:
    return int(round(x / snap_m)), int(round(y / snap_m))


def orientation_lengths(line: ogr.Geometry, bins: int) -> list[float]:
    values = [0.0] * bins
    width = 180.0 / bins
    for x1, y1, x2, y2 in iter_segments(line):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        index = min(int(angle // width), bins - 1)
        values[index] += length
    return values


def build_graph(
    roads_layer: ogr.Layer,
    road_type_field: str,
    excluded_types: set[str],
    snap_m: float,
    orientation_bins: int,
) -> tuple[nx.Graph, list[dict[str, Any]], dict[int, list[int]], dict[str, int]]:
    edge_candidates: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    source_line_parts = 0
    excluded_line_parts = 0
    invalid_line_parts = 0

    roads_layer.ResetReading()
    for feature in roads_layer:
        road_type_raw = feature.GetField(road_type_field)
        road_type = str(road_type_raw).strip().upper() if road_type_raw not in (None, "") else ""
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        for line_ref in iter_line_parts(geom):
            source_line_parts += 1
            if road_type in excluded_types:
                excluded_line_parts += 1
                continue
            line = line_ref.Clone()
            if line.GetPointCount() < 2:
                invalid_line_parts += 1
                continue
            x1, y1, _ = line.GetPoint(0)
            x2, y2, _ = line.GetPoint(line.GetPointCount() - 1)
            key_a = node_key(float(x1), float(y1), snap_m)
            key_b = node_key(float(x2), float(y2), snap_m)
            if key_a == key_b:
                invalid_line_parts += 1
                continue
            length = float(line.Length())
            if length <= 0:
                invalid_line_parts += 1
                continue
            pair = tuple(sorted((key_a, key_b)))
            candidate = {
                "key_a": key_a,
                "key_b": key_b,
                "length_m": length,
                "orientation_lengths": orientation_lengths(line, orientation_bins),
            }
            existing = edge_candidates.get(pair)
            if existing is None or length < float(existing["length_m"]):
                edge_candidates[pair] = candidate
    roads_layer.ResetReading()

    node_keys = sorted({key for pair in edge_candidates for key in pair})
    node_ids = {key: index for index, key in enumerate(node_keys)}
    graph = nx.Graph()
    for key, node_id in node_ids.items():
        graph.add_node(node_id, x=key[0] * snap_m, y=key[1] * snap_m)

    edges: list[dict[str, Any]] = []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for candidate in edge_candidates.values():
        u = node_ids[candidate["key_a"]]
        v = node_ids[candidate["key_b"]]
        edge_id = len(edges)
        edge = {
            "u": u,
            "v": v,
            "length_m": float(candidate["length_m"]),
            "orientation_lengths": candidate["orientation_lengths"],
        }
        edges.append(edge)
        adjacency[u].append(edge_id)
        adjacency[v].append(edge_id)
        graph.add_edge(u, v, length_m=edge["length_m"], edge_id=edge_id)

    audit = {
        "source_line_parts": source_line_parts,
        "excluded_line_parts": excluded_line_parts,
        "invalid_line_parts": invalid_line_parts,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }
    return graph, edges, adjacency, audit


def empty_cost_metrics(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_reachable_road_length_m": None,
        f"{prefix}_reachable_edge_count": None,
        f"{prefix}_reachable_node_count": None,
        f"{prefix}_intersection_node_count": None,
        f"{prefix}_dead_end_node_count": None,
        f"{prefix}_intersections_per_network_km": None,
        f"{prefix}_dead_ends_per_network_km": None,
        f"{prefix}_mean_node_degree": None,
        f"{prefix}_orientation_entropy_norm": None,
        f"{prefix}_orientation_dominant_bin_share": None,
    }


def finalize_orientation(lengths: list[float]) -> tuple[float | None, float | None]:
    total = sum(lengths)
    if total <= 0:
        return None, None
    probabilities = [value / total for value in lengths if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probabilities)
    entropy_norm = entropy / math.log(len(lengths)) if len(lengths) > 1 else 0.0
    return entropy_norm, max(lengths) / total


def service_metrics_for_source(
    graph: nx.Graph,
    edges: list[dict[str, Any]],
    adjacency: dict[int, list[int]],
    source: int,
    costs: list[float],
    orientation_bins: int,
) -> dict[str, Any]:
    max_cost = max(costs)
    distances = nx.single_source_dijkstra_path_length(graph, source, cutoff=max_cost, weight="length_m")
    row: dict[str, Any] = {}

    for cost in costs:
        prefix = f"network_service_{cost_label(cost)}"
        reachable_nodes = [node for node, distance in distances.items() if distance <= cost]
        edge_ids: set[int] = set()
        for node in reachable_nodes:
            edge_ids.update(adjacency.get(node, []))

        road_length = 0.0
        reachable_edge_count = 0
        orientation = [0.0] * orientation_bins
        for edge_id in edge_ids:
            edge = edges[edge_id]
            length = float(edge["length_m"])
            du = distances.get(int(edge["u"]), math.inf)
            dv = distances.get(int(edge["v"]), math.inf)
            from_u = max(cost - du, 0.0) if math.isfinite(du) else 0.0
            from_v = max(cost - dv, 0.0) if math.isfinite(dv) else 0.0
            reachable_length = min(length, from_u + from_v)
            if reachable_length <= 0:
                continue
            reachable_edge_count += 1
            road_length += reachable_length
            scale = reachable_length / length
            for index, value in enumerate(edge["orientation_lengths"]):
                orientation[index] += float(value) * scale

        degrees = [int(graph.degree(node)) for node in reachable_nodes]
        intersection_count = sum(1 for degree in degrees if degree >= 3)
        dead_end_count = sum(1 for degree in degrees if degree == 1)
        network_km = road_length / 1000.0
        entropy, dominant_share = finalize_orientation(orientation)

        row[f"{prefix}_reachable_road_length_m"] = road_length
        row[f"{prefix}_reachable_edge_count"] = reachable_edge_count
        row[f"{prefix}_reachable_node_count"] = len(reachable_nodes)
        row[f"{prefix}_intersection_node_count"] = intersection_count
        row[f"{prefix}_dead_end_node_count"] = dead_end_count
        row[f"{prefix}_intersections_per_network_km"] = intersection_count / network_km if network_km > 0 else None
        row[f"{prefix}_dead_ends_per_network_km"] = dead_end_count / network_km if network_km > 0 else None
        row[f"{prefix}_mean_node_degree"] = sum(degrees) / len(degrees) if degrees else None
        row[f"{prefix}_orientation_entropy_norm"] = entropy
        row[f"{prefix}_orientation_dominant_bin_share"] = dominant_share
    return row


def read_grid_centroids(grid_layer: ogr.Layer, max_cells: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_layer.ResetReading()
    for feature in grid_layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        centroid = geom.Centroid()
        rows.append(
            {
                "grid_id": str(feature.GetField("grid_id")),
                "x": float(centroid.GetX()),
                "y": float(centroid.GetY()),
            }
        )
        if max_cells is not None and len(rows) >= max_cells:
            break
    grid_layer.ResetReading()
    return rows


def aggregate(
    grid_rows: list[dict[str, Any]],
    graph: nx.Graph,
    edges: list[dict[str, Any]],
    adjacency: dict[int, list[int]],
    costs: list[float],
    max_snap_m: float,
    orientation_bins: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    node_ids = np.array(list(graph.nodes()), dtype=int)
    coordinates = np.array([[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in node_ids], dtype=float)
    tree = cKDTree(coordinates)
    query_points = np.array([[row["x"], row["y"]] for row in grid_rows], dtype=float)
    snap_distances, nearest_indices = tree.query(query_points, k=1)

    assignments: list[tuple[int | None, float]] = []
    unique_sources: set[int] = set()
    for snap_distance, nearest_index in zip(snap_distances, nearest_indices):
        source = int(node_ids[int(nearest_index)])
        if float(snap_distance) > max_snap_m:
            assignments.append((None, float(snap_distance)))
        else:
            assignments.append((source, float(snap_distance)))
            unique_sources.add(source)

    cache: dict[int, dict[str, Any]] = {}
    total_sources = len(unique_sources)
    for index, source in enumerate(sorted(unique_sources), start=1):
        if index % 500 == 0:
            print(f"processed_unique_service_sources={index}/{total_sources}")
        cache[source] = service_metrics_for_source(
            graph=graph,
            edges=edges,
            adjacency=adjacency,
            source=source,
            costs=costs,
            orientation_bins=orientation_bins,
        )

    component_sizes: dict[int, int] = {}
    for component in nx.connected_components(graph):
        size = len(component)
        for node in component:
            component_sizes[int(node)] = size

    stats: dict[str, dict[str, Any]] = {}
    for grid_row, (source, snap_distance) in zip(grid_rows, assignments):
        row: dict[str, Any] = {
            "network_service_available_flag": 1 if source is not None else 0,
            "network_service_snap_distance_m": snap_distance,
            "network_service_source_node_id": source,
            "network_service_component_node_count": component_sizes.get(source) if source is not None else None,
        }
        if source is None:
            for cost in costs:
                row.update(empty_cost_metrics(f"network_service_{cost_label(cost)}"))
        else:
            row.update(cache[source])
        stats[str(grid_row["grid_id"])] = row

    audit = {
        "grid_cells": len(grid_rows),
        "available_cells": sum(1 for source, _ in assignments if source is not None),
        "unavailable_cells": sum(1 for source, _ in assignments if source is None),
        "unique_service_sources": len(unique_sources),
        "snap_distance_median": float(np.median(snap_distances)),
        "snap_distance_p95": float(np.quantile(snap_distances, 0.95)),
        "snap_distance_max": float(np.max(snap_distances)),
    }
    return stats, audit


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if name.endswith("_flag") or name.endswith("_count") or name.endswith("_node_id"):
        return ogr.FieldDefn(name, ogr.OFTInteger64)
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger64)
    return ogr.FieldDefn(name, ogr.OFTReal)


def write_gpkg(grid_layer: ogr.Layer, stats: dict[str, dict[str, Any]], out_path: Path, out_layer_name: str) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        driver.DeleteDataSource(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")
    out_layer = out_ds.CreateLayer(out_layer_name, srs=grid_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)

    in_defn = grid_layer.GetLayerDefn()
    for index in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(index))
    fields = list(next(iter(stats.values())).keys())
    for field in fields:
        sample = next((row.get(field) for row in stats.values() if row.get(field) is not None), None)
        out_layer.CreateField(create_field(field, sample))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        grid_id = str(feature.GetField("grid_id"))
        if grid_id not in stats:
            continue
        out_feature = ogr.Feature(out_defn)
        for index in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(index).GetName(), feature.GetField(index))
        for field, value in stats[grid_id].items():
            if value is not None:
                out_feature.SetField(field, value)
        geom = feature.GetGeometryRef()
        if geom is not None:
            out_feature.SetGeometry(geom.Clone())
        out_layer.CreateFeature(out_feature)
        out_feature = None
    out_layer.SyncToDisk()
    out_ds = None
    grid_layer.ResetReading()


def write_csv(out_path: Path, stats: dict[str, dict[str, Any]]) -> None:
    fields = list(next(iter(stats.values())).keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_id", *fields])
        writer.writeheader()
        for grid_id, row in stats.items():
            writer.writerow({"grid_id": grid_id, **row})


def main() -> None:
    args = parse_args()
    grid_ds, grid_layer = open_layer(args.grid_gpkg, args.grid_layer)
    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)
    road_defn = roads_layer.GetLayerDefn()
    road_fields = {road_defn.GetFieldDefn(index).GetName() for index in range(road_defn.GetFieldCount())}
    if args.road_type_field not in road_fields:
        raise RuntimeError(f"Road type field not found: {args.road_type_field}")

    excluded_types = {value.strip().upper() for value in args.exclude_road_types}
    graph, edges, adjacency, graph_audit = build_graph(
        roads_layer=roads_layer,
        road_type_field=args.road_type_field,
        excluded_types=excluded_types,
        snap_m=args.node_snap_m,
        orientation_bins=args.orientation_bins,
    )
    grid_rows = read_grid_centroids(grid_layer, args.max_cells)
    stats, service_audit = aggregate(
        grid_rows=grid_rows,
        graph=graph,
        edges=edges,
        adjacency=adjacency,
        costs=sorted(args.network_costs_m),
        max_snap_m=args.max_centroid_snap_m,
        orientation_bins=args.orientation_bins,
    )
    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    print(f"excluded_road_types={','.join(sorted(excluded_types))}")
    for key, value in graph_audit.items():
        print(f"{key}={value}")
    for key, value in service_audit.items():
        print(f"{key}={value}")
    for cost in sorted(args.network_costs_m):
        prefix = f"network_service_{cost_label(cost)}"
        lengths = [
            float(row[f"{prefix}_reachable_road_length_m"])
            for row in stats.values()
            if row[f"{prefix}_reachable_road_length_m"] is not None
        ]
        print(f"{prefix}_mean_reachable_road_length_m={sum(lengths) / len(lengths):.2f}" if lengths else "")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    roads_ds = None


if __name__ == "__main__":
    main()
