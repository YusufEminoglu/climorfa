"""Validate endpoint-derived road-network node assumptions.

The current CLIMORFA network metrics use snapped line endpoints as node proxies.
This script documents the degree distribution and snap-distance sensitivity so
the manuscript can describe the limitation honestly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from osgeo import ogr, osr


DEFAULT_SNAP_SENSITIVITY_M = [0.5, 1.0, 2.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads-gpkg", default="data/02_interim/buildings_roads_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--primary-snap-m", type=float, default=1.0)
    parser.add_argument("--snap-sensitivity-m", nargs="+", type=float, default=DEFAULT_SNAP_SENSITIVITY_M)
    parser.add_argument("--out-dir", default="outputs/diagnostics/network_endpoint_validation_2026-06-15")
    parser.add_argument("--out-nodes-gpkg", default="data/03_processed/road_endpoint_nodes_validation.gpkg")
    parser.add_argument("--out-nodes-layer", default="road_endpoint_nodes_snap1m")
    parser.add_argument("--report-md", default="docs/methodology/network_endpoint_validation_2026-06-15.md")
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


def node_key(x: float, y: float, snap_m: float) -> tuple[int, int]:
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def node_class(degree: int) -> str:
    if degree <= 0:
        return "invalid"
    if degree == 1:
        return "dead_end"
    if degree == 2:
        return "through_or_segment_join"
    if degree <= 4:
        return "intersection"
    return "high_degree_intersection"


def collect_nodes(layer: ogr.Layer, snap_m: float) -> tuple[dict[tuple[int, int], dict[str, float]], dict[str, float]]:
    nodes: dict[tuple[int, int], dict[str, float]] = {}
    line_parts = 0
    raw_endpoints = 0
    total_length_m = 0.0
    layer.ResetReading()
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        geom = geom.Clone()
        for line in iter_line_parts(geom):
            point_count = line.GetPointCount()
            if point_count < 2:
                continue
            line_parts += 1
            total_length_m += float(line.Length())
            for point_index in (0, point_count - 1):
                x, y, _ = line.GetPoint(point_index)
                raw_endpoints += 1
                key = node_key(float(x), float(y), snap_m)
                if key not in nodes:
                    nodes[key] = {
                        "x": float(x),
                        "y": float(y),
                        "degree": 0.0,
                        "raw_endpoint_count": 0.0,
                    }
                nodes[key]["degree"] += 1.0
                nodes[key]["raw_endpoint_count"] += 1.0
    layer.ResetReading()
    meta = {
        "line_parts": line_parts,
        "raw_endpoints": raw_endpoints,
        "total_length_m": total_length_m,
    }
    return nodes, meta


def summarize_nodes(nodes: dict[tuple[int, int], dict[str, float]], meta: dict[str, float], snap_m: float) -> dict[str, float]:
    degrees = [int(node["degree"]) for node in nodes.values()]
    degree_counts = Counter(degrees)
    node_count = len(degrees)
    summary: dict[str, float] = {
        "snap_m": snap_m,
        "line_parts": int(meta["line_parts"]),
        "raw_endpoints": int(meta["raw_endpoints"]),
        "unique_snapped_nodes": node_count,
        "endpoint_merge_ratio": 1.0 - (node_count / meta["raw_endpoints"]) if meta["raw_endpoints"] else 0.0,
        "total_length_m": float(meta["total_length_m"]),
        "degree_1_dead_end_nodes": degree_counts.get(1, 0),
        "degree_2_through_or_segment_join_nodes": degree_counts.get(2, 0),
        "degree_3_nodes": degree_counts.get(3, 0),
        "degree_4_nodes": degree_counts.get(4, 0),
        "degree_ge_5_nodes": sum(count for degree, count in degree_counts.items() if degree >= 5),
        "max_degree": max(degrees) if degrees else 0,
    }
    if node_count:
        summary["dead_end_share"] = summary["degree_1_dead_end_nodes"] / node_count
        summary["intersection_share_degree_ge_3"] = (
            summary["degree_3_nodes"] + summary["degree_4_nodes"] + summary["degree_ge_5_nodes"]
        ) / node_count
        summary["mean_degree"] = sum(degrees) / node_count
    else:
        summary["dead_end_share"] = 0.0
        summary["intersection_share_degree_ge_3"] = 0.0
        summary["mean_degree"] = 0.0
    return summary


def write_nodes_gpkg(
    out_path: Path,
    layer_name: str,
    srs: osr.SpatialReference | None,
    nodes: dict[tuple[int, int], dict[str, float]],
) -> None:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        out_ds = driver.Open(str(out_path), update=1)
        if out_ds is None:
            raise RuntimeError(f"Could not open existing output {out_path}")
        existing = out_ds.GetLayerByName(layer_name)
        if existing is not None:
            out_ds.DeleteLayer(layer_name)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")

    layer = out_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("node_id", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("degree", ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn("node_class", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("raw_endpoint_count", ogr.OFTInteger))

    defn = layer.GetLayerDefn()
    for index, (key, node) in enumerate(nodes.items(), start=1):
        feature = ogr.Feature(defn)
        feature.SetField("node_id", f"node_{index:07d}")
        degree = int(node["degree"])
        feature.SetField("degree", degree)
        feature.SetField("node_class", node_class(degree))
        feature.SetField("raw_endpoint_count", int(node["raw_endpoint_count"]))
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(node["x"], node["y"])
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
        feature = None
    layer.SyncToDisk()
    out_ds = None


def write_node_csv(out_path: Path, nodes: dict[tuple[int, int], dict[str, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "x", "y", "degree", "node_class", "raw_endpoint_count"],
        )
        writer.writeheader()
        for index, node in enumerate(nodes.values(), start=1):
            degree = int(node["degree"])
            writer.writerow(
                {
                    "node_id": f"node_{index:07d}",
                    "x": node["x"],
                    "y": node["y"],
                    "degree": degree,
                    "node_class": node_class(degree),
                    "raw_endpoint_count": int(node["raw_endpoint_count"]),
                }
            )


def markdown_table(rows: list[dict[str, float]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if "share" in column or "ratio" in column:
                    values.append(f"{value * 100:.2f}%")
                elif abs(value) >= 1000:
                    values.append(f"{value:,.2f}")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    summary_rows: list[dict[str, float]],
    out_dir: Path,
    nodes_gpkg: Path,
    nodes_csv: Path,
) -> None:
    primary = next(row for row in summary_rows if math.isclose(float(row["snap_m"]), 1.0))
    columns = [
        "snap_m",
        "unique_snapped_nodes",
        "endpoint_merge_ratio",
        "degree_1_dead_end_nodes",
        "degree_2_through_or_segment_join_nodes",
        "degree_3_nodes",
        "degree_4_nodes",
        "degree_ge_5_nodes",
        "dead_end_share",
        "intersection_share_degree_ge_3",
        "mean_degree",
        "max_degree",
    ]
    text = "\n".join(
        [
            "# Network Endpoint Validation",
            "",
            "Date: 2026-06-15",
            "",
            "## Purpose",
            "",
            "Validate the endpoint-derived node and intersection proxy used by the CLIMORFA road/network feature pipeline.",
            "",
            "## Primary 1 m Snap Summary",
            "",
            f"- Unique snapped endpoint nodes: {int(primary['unique_snapped_nodes']):,}",
            f"- Raw line endpoints: {int(primary['raw_endpoints']):,}",
            f"- Endpoint merge ratio: {float(primary['endpoint_merge_ratio']) * 100:.2f}%",
            f"- Dead-end nodes, degree 1: {int(primary['degree_1_dead_end_nodes']):,}",
            f"- Through/segment-join nodes, degree 2: {int(primary['degree_2_through_or_segment_join_nodes']):,}",
            f"- Intersection nodes, degree >= 3: {int(primary['degree_3_nodes'] + primary['degree_4_nodes'] + primary['degree_ge_5_nodes']):,}",
            f"- Mean endpoint degree: {float(primary['mean_degree']):.3f}",
            f"- Max endpoint degree: {int(primary['max_degree'])}",
            "",
            "## Snap Sensitivity",
            "",
            markdown_table(summary_rows, columns),
            "",
            "## Interpretation",
            "",
            "- Endpoint-derived nodes are useful as an interpretable proxy for local network structure.",
            "- Degree-2 nodes are not necessarily intersections; many are line segmentation joins.",
            "- Crossings that are not split in the source data may be missed by an endpoint-only approach.",
            "- High-degree nodes may represent real complex intersections or source segmentation artifacts.",
            "- The manuscript should describe these fields as endpoint-derived proxies unless a full planar topology build is added.",
            "",
            "## Outputs",
            "",
            f"- Node validation GPKG: `{nodes_gpkg.as_posix()}`",
            f"- Node validation CSV: `{nodes_csv.as_posix()}`",
            f"- Summary CSV: `{(out_dir / 'snap_sensitivity_summary.csv').as_posix()}`",
            f"- Summary JSON: `{(out_dir / 'network_endpoint_validation_summary.json').as_posix()}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_md)
    nodes_gpkg = Path(args.out_nodes_gpkg)
    nodes_csv = out_dir / "road_endpoint_nodes_snap1m.csv"

    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)
    summary_rows: list[dict[str, float]] = []
    primary_nodes: dict[tuple[int, int], dict[str, float]] | None = None
    for snap_m in args.snap_sensitivity_m:
        nodes, meta = collect_nodes(roads_layer, snap_m)
        summary = summarize_nodes(nodes, meta, snap_m)
        summary_rows.append(summary)
        if math.isclose(snap_m, args.primary_snap_m):
            primary_nodes = nodes
    if primary_nodes is None:
        primary_nodes, _ = collect_nodes(roads_layer, args.primary_snap_m)

    summary_csv = out_dir / "snap_sensitivity_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = out_dir / "network_endpoint_validation_summary.json"
    summary_json.write_text(json.dumps({"snap_sensitivity": summary_rows}, indent=2), encoding="utf-8")

    write_nodes_gpkg(nodes_gpkg, args.out_nodes_layer, roads_layer.GetSpatialRef(), primary_nodes)
    write_node_csv(nodes_csv, primary_nodes)
    write_report(report_path, summary_rows, out_dir, nodes_gpkg, nodes_csv)

    primary_summary = next(row for row in summary_rows if math.isclose(float(row["snap_m"]), args.primary_snap_m))
    print(f"primary_snap_m={args.primary_snap_m}")
    print(f"line_parts={int(primary_summary['line_parts'])}")
    print(f"raw_endpoints={int(primary_summary['raw_endpoints'])}")
    print(f"unique_snapped_nodes={int(primary_summary['unique_snapped_nodes'])}")
    print(f"degree_1_dead_end_nodes={int(primary_summary['degree_1_dead_end_nodes'])}")
    print(
        "intersection_nodes_degree_ge_3="
        f"{int(primary_summary['degree_3_nodes'] + primary_summary['degree_4_nodes'] + primary_summary['degree_ge_5_nodes'])}"
    )
    print(f"out_report={report_path}")
    print(f"out_nodes_gpkg={nodes_gpkg}")
    print(f"out_dir={out_dir}")

    roads_ds = None


if __name__ == "__main__":
    main()
