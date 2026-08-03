"""Aggregate 250/400/800 m road-network context around grid centroids.

These are Euclidean centroid-buffer context metrics, not true network-distance
reach metrics. They provide the grid-vs-context sensitivity layer requested for
CLIMORFA while keeping the current no-GNN, interpretable-network design.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from osgeo import ogr


DEFAULT_RADII = [250.0, 400.0, 800.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--roads-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--study-area-gpkg", default="data/02_interim/study_area_fua.gpkg")
    parser.add_argument("--study-area-layer", default="study_area_fua")
    parser.add_argument("--out-gpkg", default="data/03_processed/grid_road_network_reach_features.gpkg")
    parser.add_argument("--out-layer", default="grid_250m_road_network_reach_features")
    parser.add_argument("--out-csv", default="data/03_processed/grid_250m_road_network_reach_features.csv")
    parser.add_argument("--radii-m", nargs="+", type=float, default=DEFAULT_RADII)
    parser.add_argument("--node-snap-m", type=float, default=1.0)
    parser.add_argument("--node-bin-m", type=float, default=250.0)
    parser.add_argument("--orientation-bins", type=int, default=12)
    parser.add_argument("--buffer-segments", type=int, default=24)
    return parser.parse_args()


def radius_label(radius_m: float) -> str:
    if float(radius_m).is_integer():
        return f"{int(radius_m)}m"
    return f"{str(radius_m).replace('.', 'p')}m"


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def read_union_geometry(layer: ogr.Layer) -> ogr.Geometry:
    union_geom: ogr.Geometry | None = None
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        union_geom = geom if union_geom is None else union_geom.Union(geom)
    layer.ResetReading()
    if union_geom is None or union_geom.IsEmpty():
        raise RuntimeError("Study area geometry is empty")
    return union_geom


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


def safe_area(geom: ogr.Geometry | None) -> float:
    if geom is None or geom.IsEmpty():
        return 0.0
    return float(geom.Area())


def point_from_xy(x: float, y: float) -> ogr.Geometry:
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(float(x), float(y))
    return point


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


def build_node_bins(
    nodes: dict[tuple[int, int], dict[str, float]],
    bin_m: float,
) -> dict[tuple[int, int], list[dict[str, float]]]:
    bins: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for node in nodes.values():
        bx = int(math.floor(node["x"] / bin_m))
        by = int(math.floor(node["y"] / bin_m))
        bins[(bx, by)].append(node)
    return bins


def nodes_in_geometry(
    geom: ogr.Geometry,
    node_bins: dict[tuple[int, int], list[dict[str, float]]],
    bin_m: float,
) -> list[dict[str, float]]:
    min_x, max_x, min_y, max_y = geom.GetEnvelope()
    x0 = int(math.floor(min_x / bin_m))
    x1 = int(math.floor(max_x / bin_m))
    y0 = int(math.floor(min_y / bin_m))
    y1 = int(math.floor(max_y / bin_m))
    selected: list[dict[str, float]] = []
    point = ogr.Geometry(ogr.wkbPoint)
    seen: set[tuple[float, float]] = set()
    for bx in range(x0, x1 + 1):
        for by in range(y0, y1 + 1):
            for node in node_bins.get((bx, by), []):
                key = (node["x"], node["y"])
                if key in seen:
                    continue
                point.Empty()
                point.AddPoint(node["x"], node["y"])
                if geom.Intersects(point):
                    selected.append(node)
                    seen.add(key)
    return selected


def add_orientation_lengths(row: dict[str, float], prefix: str, geom: ogr.Geometry, orientation_bins: int) -> None:
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
            row[f"{prefix}_orientation_bin_{bin_index:02d}_length_m"] += length


def finalize_orientation(row: dict[str, Any], prefix: str, orientation_bins: int) -> None:
    lengths = [float(row[f"{prefix}_orientation_bin_{index:02d}_length_m"]) for index in range(orientation_bins)]
    total = sum(lengths)
    if total <= 0:
        return
    probs = [value / total for value in lengths if value > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    row[f"{prefix}_orientation_entropy_norm"] = entropy / math.log(orientation_bins) if orientation_bins > 1 else 0.0
    row[f"{prefix}_orientation_dominant_bin_share"] = max(lengths) / total
    row[f"{prefix}_orientation_bins_used"] = len(probs)


def empty_radius_metrics(prefix: str, orientation_bins: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{prefix}_buffer_area_m2": 0.0,
        f"{prefix}_fua_area_m2": 0.0,
        f"{prefix}_fua_coverage_share": None,
        f"{prefix}_road_length_m": 0.0,
        f"{prefix}_road_density_m_per_km2": 0.0,
        f"{prefix}_road_segment_intersection_count": 0,
        f"{prefix}_road_segment_intersection_density_per_km2": 0.0,
        f"{prefix}_orientation_entropy_norm": None,
        f"{prefix}_orientation_dominant_bin_share": None,
        f"{prefix}_orientation_bins_used": 0,
        f"{prefix}_endpoint_node_count": 0,
        f"{prefix}_endpoint_node_density_per_km2": 0.0,
        f"{prefix}_intersection_node_count": 0,
        f"{prefix}_intersection_density_per_km2": 0.0,
        f"{prefix}_dead_end_node_count": 0,
        f"{prefix}_dead_end_density_per_km2": 0.0,
        f"{prefix}_mean_endpoint_degree": None,
        f"{prefix}_dead_end_share": None,
    }
    for index in range(orientation_bins):
        row[f"{prefix}_orientation_bin_{index:02d}_length_m"] = 0.0
    return row


def aggregate_one_buffer(
    roads_layer: ogr.Layer,
    buffer_geom: ogr.Geometry,
    context_geom: ogr.Geometry,
    node_bins: dict[tuple[int, int], list[dict[str, float]]],
    node_bin_m: float,
    prefix: str,
    orientation_bins: int,
) -> dict[str, Any]:
    row = empty_radius_metrics(prefix, orientation_bins)
    buffer_area = float(buffer_geom.Area())
    context_area = safe_area(context_geom)
    row[f"{prefix}_buffer_area_m2"] = buffer_area
    row[f"{prefix}_fua_area_m2"] = context_area
    row[f"{prefix}_fua_coverage_share"] = context_area / buffer_area if buffer_area > 0 else None

    if context_area <= 0:
        return row

    roads_layer.SetSpatialFilter(context_geom)
    roads_layer.ResetReading()
    for road_feature in roads_layer:
        road_geom = road_feature.GetGeometryRef()
        if road_geom is None or road_geom.IsEmpty():
            continue
        inter = safe_intersection(context_geom, road_geom)
        if inter is None:
            continue
        length = float(inter.Length())
        if length <= 0:
            continue
        row[f"{prefix}_road_length_m"] += length
        row[f"{prefix}_road_segment_intersection_count"] += 1
        add_orientation_lengths(row, prefix, inter, orientation_bins)
    roads_layer.SetSpatialFilter(None)

    nodes = nodes_in_geometry(context_geom, node_bins, node_bin_m)
    node_count = len(nodes)
    intersection_count = sum(1 for node in nodes if node["degree"] >= 3)
    dead_end_count = sum(1 for node in nodes if node["degree"] == 1)
    km2 = context_area / 1_000_000.0

    row[f"{prefix}_endpoint_node_count"] = node_count
    row[f"{prefix}_intersection_node_count"] = intersection_count
    row[f"{prefix}_dead_end_node_count"] = dead_end_count
    if km2 > 0:
        row[f"{prefix}_road_density_m_per_km2"] = row[f"{prefix}_road_length_m"] / km2
        row[f"{prefix}_road_segment_intersection_density_per_km2"] = (
            row[f"{prefix}_road_segment_intersection_count"] / km2
        )
        row[f"{prefix}_endpoint_node_density_per_km2"] = node_count / km2
        row[f"{prefix}_intersection_density_per_km2"] = intersection_count / km2
        row[f"{prefix}_dead_end_density_per_km2"] = dead_end_count / km2
    if node_count > 0:
        row[f"{prefix}_mean_endpoint_degree"] = sum(node["degree"] for node in nodes) / node_count
        row[f"{prefix}_dead_end_share"] = dead_end_count / node_count

    finalize_orientation(row, prefix, orientation_bins)
    return row


def aggregate(
    grid_layer: ogr.Layer,
    roads_layer: ogr.Layer,
    study_area_geom: ogr.Geometry,
    node_bins: dict[tuple[int, int], list[dict[str, float]]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    grid_layer.ResetReading()
    total = grid_layer.GetFeatureCount()
    for index, grid_feature in enumerate(grid_layer, start=1):
        if index % 500 == 0:
            print(f"processed_reach_grid_cells={index}/{total}")
        grid_id = str(grid_feature.GetField("grid_id"))
        geom = grid_feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        centroid = geom.Centroid()
        row: dict[str, Any] = {}
        for radius_m in args.radii_m:
            label = radius_label(radius_m)
            prefix = f"network_reach_{label}"
            buffer_geom = centroid.Buffer(float(radius_m), int(args.buffer_segments))
            context_geom = safe_intersection(buffer_geom, study_area_geom)
            if context_geom is None:
                context_geom = ogr.Geometry(ogr.wkbPolygon)
            row.update(
                aggregate_one_buffer(
                    roads_layer=roads_layer,
                    buffer_geom=buffer_geom,
                    context_geom=context_geom,
                    node_bins=node_bins,
                    node_bin_m=args.node_bin_m,
                    prefix=prefix,
                    orientation_bins=args.orientation_bins,
                )
            )
        stats[grid_id] = row
    grid_layer.ResetReading()
    return stats


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
    study_ds, study_layer = open_layer(args.study_area_gpkg, args.study_area_layer)

    study_area_geom = read_union_geometry(study_layer)
    nodes = endpoint_records(roads_layer, args.node_snap_m)
    node_bins = build_node_bins(nodes, args.node_bin_m)
    stats = aggregate(grid_layer, roads_layer, study_area_geom, node_bins, args)

    write_gpkg(grid_layer, stats, Path(args.out_gpkg), args.out_layer)
    write_csv(Path(args.out_csv), stats)

    print(f"grid_cells={len(stats)}")
    print(f"endpoint_nodes={len(nodes)}")
    print(f"radii_m={','.join(str(radius) for radius in args.radii_m)}")
    for radius_m in args.radii_m:
        label = radius_label(radius_m)
        prefix = f"network_reach_{label}"
        cells_with_roads = sum(1 for row in stats.values() if float(row[f"{prefix}_road_length_m"]) > 0)
        cells_with_nodes = sum(1 for row in stats.values() if int(row[f"{prefix}_endpoint_node_count"]) > 0)
        road_length = sum(float(row[f"{prefix}_road_length_m"]) for row in stats.values())
        print(f"{prefix}_cells_with_roads={cells_with_roads}")
        print(f"{prefix}_cells_with_nodes={cells_with_nodes}")
        print(f"{prefix}_summed_road_length_m={road_length:.2f}")
    print(f"out_gpkg={args.out_gpkg}")
    print(f"out_csv={args.out_csv}")

    grid_ds = None
    roads_ds = None
    study_ds = None


if __name__ == "__main__":
    main()
