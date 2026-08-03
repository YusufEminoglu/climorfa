"""Aggregate exact road and endpoint-network metrics to 250 m grid cells."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from osgeo import ogr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--roads-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_road_network_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_road_network_features")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_road_network_features.csv")
    parser.add_argument("--node-snap-m", type=float, default=1.0)
    parser.add_argument("--orientation-bins", type=int, default=12)
    return parser.parse_args()


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
    point_count = line.GetPointCount()
    if point_count < 2:
        return
    for index in range(point_count - 1):
        x1, y1, _ = line.GetPoint(index)
        x2, y2, _ = line.GetPoint(index + 1)
        yield float(x1), float(y1), float(x2), float(y2)


def safe_intersection(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        inter = a.Intersection(b)
    except RuntimeError:
        inter = a.MakeValid().Intersection(b.MakeValid())
    if inter is None or inter.IsEmpty():
        return None
    return inter


def node_key(x: float, y: float, snap_m: float) -> tuple[int, int]:
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def endpoint_records(roads_layer: ogr.Layer, snap_m: float) -> dict[tuple[int, int], dict[str, float]]:
    nodes: dict[tuple[int, int], dict[str, float]] = {}
    roads_layer.ResetReading()
    for feature in roads_layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        for line in iter_line_parts(geom):
            point_count = line.GetPointCount()
            if point_count < 2:
                continue
            for point_index in (0, point_count - 1):
                x, y, _ = line.GetPoint(point_index)
                key = node_key(float(x), float(y), snap_m)
                if key not in nodes:
                    nodes[key] = {"x": float(x), "y": float(y), "degree": 0.0}
                nodes[key]["degree"] += 1.0
    roads_layer.ResetReading()
    return nodes


def create_grid_index(grid_layer: ogr.Layer, cell_m: float) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int], list[str]]]:
    grids: dict[str, dict[str, Any]] = {}
    bins: dict[tuple[int, int], list[str]] = defaultdict(list)
    grid_layer.ResetReading()
    for feature in grid_layer:
        grid_id = str(feature.GetField("grid_id"))
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        min_x, max_x, min_y, max_y = geom.GetEnvelope()
        grids[grid_id] = {
            "geometry": geom,
            "cell_area_m2": float(feature.GetField("cell_area_m2") or geom.Area()),
            "envelope": (min_x, max_x, min_y, max_y),
        }
        x0 = int(math.floor(min_x / cell_m))
        x1 = int(math.floor(max_x / cell_m))
        y0 = int(math.floor(min_y / cell_m))
        y1 = int(math.floor(max_y / cell_m))
        for bx in range(x0, x1 + 1):
            for by in range(y0, y1 + 1):
                bins[(bx, by)].append(grid_id)
    grid_layer.ResetReading()
    return grids, bins


def assign_nodes_to_grids(
    nodes: dict[tuple[int, int], dict[str, float]],
    grids: dict[str, dict[str, Any]],
    bins: dict[tuple[int, int], list[str]],
    cell_m: float,
) -> dict[str, list[dict[str, float]]]:
    assigned: dict[str, list[dict[str, float]]] = defaultdict(list)
    point = ogr.Geometry(ogr.wkbPoint)
    for node in nodes.values():
        x = node["x"]
        y = node["y"]
        bx = int(math.floor(x / cell_m))
        by = int(math.floor(y / cell_m))
        point.Empty()
        point.AddPoint(x, y)
        for grid_id in bins.get((bx, by), []):
            grid = grids[grid_id]
            min_x, max_x, min_y, max_y = grid["envelope"]
            if x < min_x or x > max_x or y < min_y or y > max_y:
                continue
            if grid["geometry"].Intersects(point):
                assigned[grid_id].append(node)
                break
    return assigned


def empty_row(orientation_bins: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "road_length_exact_m": 0.0,
        "road_density_exact_m_per_km2": 0.0,
        "road_segment_intersection_count": 0,
        "road_segment_intersection_density_per_km2": 0.0,
        "road_mean_intersection_length_m": 0.0,
        "road_orientation_entropy_norm": None,
        "road_orientation_dominant_bin_share": None,
        "road_orientation_bins_used": 0,
        "network_endpoint_node_count": 0,
        "network_endpoint_node_density_per_km2": 0.0,
        "network_intersection_node_count": 0,
        "network_intersection_density_per_km2": 0.0,
        "network_dead_end_node_count": 0,
        "network_dead_end_density_per_km2": 0.0,
        "network_mean_endpoint_degree": None,
        "network_dead_end_share": None,
    }
    for index in range(orientation_bins):
        row[f"road_orientation_bin_{index:02d}_length_m"] = 0.0
    return row


def add_orientation(row: dict[str, Any], geom: ogr.Geometry, orientation_bins: int) -> None:
    bin_width = 180.0 / orientation_bins
    for line in iter_line_parts(geom):
        for x1, y1, x2, y2 in iter_segments(line):
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            bin_index = min(int(angle // bin_width), orientation_bins - 1)
            row[f"road_orientation_bin_{bin_index:02d}_length_m"] += length


def finalize_orientation(row: dict[str, Any], orientation_bins: int) -> None:
    lengths = [float(row[f"road_orientation_bin_{index:02d}_length_m"]) for index in range(orientation_bins)]
    total = sum(lengths)
    if total <= 0:
        return
    probs = [value / total for value in lengths if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    row["road_orientation_entropy_norm"] = entropy / math.log(orientation_bins) if orientation_bins > 1 else 0.0
    row["road_orientation_dominant_bin_share"] = max(lengths) / total
    row["road_orientation_bins_used"] = len(probs)


def aggregate_road_lengths(
    grid_layer: ogr.Layer,
    roads_layer: ogr.Layer,
    orientation_bins: int,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for index, grid_feature in enumerate(grid_layer, start=1):
        if index % 1000 == 0:
            print(f"processed_road_grid_cells={index}/{total}")
        grid_id = str(grid_feature.GetField("grid_id"))
        cell_geom = grid_feature.GetGeometryRef()
        if cell_geom is None or cell_geom.IsEmpty():
            continue
        cell_geom = cell_geom.Clone()
        cell_area_m2 = float(grid_feature.GetField("cell_area_m2") or cell_geom.Area())
        row = empty_row(orientation_bins)
        roads_layer.SetSpatialFilter(cell_geom)
        roads_layer.ResetReading()
        for road_feature in roads_layer:
            road_geom = road_feature.GetGeometryRef()
            if road_geom is None or road_geom.IsEmpty():
                continue
            inter = safe_intersection(cell_geom, road_geom)
            if inter is None:
                continue
            length = float(inter.Length())
            if length <= 0:
                continue
            row["road_length_exact_m"] += length
            row["road_segment_intersection_count"] += 1
            add_orientation(row, inter, orientation_bins)
        roads_layer.SetSpatialFilter(None)
        km2 = cell_area_m2 / 1_000_000.0 if cell_area_m2 else 0.0
        if km2 > 0:
            row["road_density_exact_m_per_km2"] = row["road_length_exact_m"] / km2
            row["road_segment_intersection_density_per_km2"] = row["road_segment_intersection_count"] / km2
        if row["road_segment_intersection_count"] > 0:
            row["road_mean_intersection_length_m"] = (
                row["road_length_exact_m"] / row["road_segment_intersection_count"]
            )
        finalize_orientation(row, orientation_bins)
        stats[grid_id] = row
    grid_layer.ResetReading()
    roads_layer.ResetReading()
    return stats


def add_node_metrics(
    stats: dict[str, dict[str, Any]],
    assigned_nodes: dict[str, list[dict[str, float]]],
    grids: dict[str, dict[str, Any]],
) -> None:
    for grid_id, row in stats.items():
        nodes = assigned_nodes.get(grid_id, [])
        node_count = len(nodes)
        intersection_count = sum(1 for node in nodes if node["degree"] >= 3)
        dead_end_count = sum(1 for node in nodes if node["degree"] == 1)
        km2 = grids[grid_id]["cell_area_m2"] / 1_000_000.0 if grids[grid_id]["cell_area_m2"] else 0.0
        row["network_endpoint_node_count"] = node_count
        row["network_intersection_node_count"] = intersection_count
        row["network_dead_end_node_count"] = dead_end_count
        if km2 > 0:
            row["network_endpoint_node_density_per_km2"] = node_count / km2
            row["network_intersection_density_per_km2"] = intersection_count / km2
            row["network_dead_end_density_per_km2"] = dead_end_count / km2
        if node_count > 0:
            row["network_mean_endpoint_degree"] = sum(node["degree"] for node in nodes) / node_count
            row["network_dead_end_share"] = dead_end_count / node_count


def create_field(name: str, value: Any) -> ogr.FieldDefn:
    if isinstance(value, int):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if isinstance(value, float) or value is None:
        return ogr.FieldDefn(name, ogr.OFTReal)
    field = ogr.FieldDefn(name, ogr.OFTString)
    field.SetWidth(80)
    return field


def write_gpkg(grid_layer: ogr.Layer, stats: dict[str, dict[str, Any]], out_path: Path, out_layer_name: str) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        out_ds = driver.Open(str(out_path), update=1)
        if out_ds is None:
            raise RuntimeError(f"Could not open existing output {out_path}")
        existing = out_ds.GetLayerByName(out_layer_name)
        if existing is not None:
            out_ds.DeleteLayer(out_layer_name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")

    out_layer = out_ds.CreateLayer(out_layer_name, srs=grid_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
    in_defn = grid_layer.GetLayerDefn()
    for index in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(index))

    fields = list(next(iter(stats.values())).keys())
    sample_values: dict[str, Any] = {}
    for field in fields:
        for row in stats.values():
            value = row.get(field)
            if value is not None:
                sample_values[field] = value
                break
    for field in fields:
        out_layer.CreateField(create_field(field, sample_values.get(field)))

    out_defn = out_layer.GetLayerDefn()
    grid_layer.ResetReading()
    for feature in grid_layer:
        out_feature = ogr.Feature(out_defn)
        for index in range(in_defn.GetFieldCount()):
            out_feature.SetField(in_defn.GetFieldDefn(index).GetName(), feature.GetField(index))
        grid_id = str(feature.GetField("grid_id"))
        row = stats.get(grid_id, {})
        for field in fields:
            value = row.get(field)
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

    cell_m = float(grid_layer.GetFeature(1).GetField("cell_m") or 250.0)
    grids, grid_bins = create_grid_index(grid_layer, cell_m)
    nodes = endpoint_records(roads_layer, args.node_snap_m)
    assigned_nodes = assign_nodes_to_grids(nodes, grids, grid_bins, cell_m)
    stats = aggregate_road_lengths(grid_layer, roads_layer, args.orientation_bins)
    add_node_metrics(stats, assigned_nodes, grids)

    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    road_cells = sum(1 for row in stats.values() if row["road_length_exact_m"] > 0)
    node_cells = sum(1 for row in stats.values() if row["network_endpoint_node_count"] > 0)
    total_road_length = sum(float(row["road_length_exact_m"]) for row in stats.values())
    total_intersections = sum(int(row["network_intersection_node_count"]) for row in stats.values())
    total_dead_ends = sum(int(row["network_dead_end_node_count"]) for row in stats.values())
    print(f"grid_cells={len(stats)}")
    print(f"road_cells={road_cells}")
    print(f"node_cells={node_cells}")
    print(f"network_endpoint_nodes={len(nodes)}")
    print(f"network_intersection_nodes_assigned={total_intersections}")
    print(f"network_dead_end_nodes_assigned={total_dead_ends}")
    print(f"road_length_exact_m={total_road_length:.2f}")
    print(f"node_snap_m={args.node_snap_m}")
    print(f"orientation_bins={args.orientation_bins}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    roads_ds = None


if __name__ == "__main__":
    main()
