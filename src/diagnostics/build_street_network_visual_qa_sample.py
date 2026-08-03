"""Build a visual QA sample for street-facing and network-reach proxy metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from osgeo import ogr

ogr.UseExceptions()


SAMPLE_FIELDS = [
    "grid_id",
    "qa_category",
    "qa_rank",
    "qa_reason",
    "street_frontage_continuity_proxy",
    "street_frontage_open_buffer_share",
    "block_permeability_street_open_network_proxy",
    "street_frontage_road_length_m",
    "street_frontage_building_count",
    "network_reach_250m_road_density_m_per_km2",
    "network_reach_400m_road_density_m_per_km2",
    "network_reach_800m_road_density_m_per_km2",
    "network_reach_250m_intersection_density_per_km2",
    "network_reach_400m_intersection_density_per_km2",
    "network_reach_800m_intersection_density_per_km2",
    "road_density_exact_m_per_km2",
    "network_intersection_density_per_km2",
    "lcz_weak_label",
    "district",
]


REPORT_COLUMNS = [
    "qa_category",
    "grid_id",
    "street_frontage_continuity_proxy",
    "street_frontage_open_buffer_share",
    "block_permeability_street_open_network_proxy",
    "network_reach_250m_road_density_m_per_km2",
    "network_reach_400m_road_density_m_per_km2",
    "network_reach_800m_road_density_m_per_km2",
    "qa_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", default="data/03_processed/grid_250m_model_features_v6.csv")
    parser.add_argument("--grid-gpkg", default="data/03_processed/analysis_grids.gpkg")
    parser.add_argument("--grid-layer", default="grid_250m")
    parser.add_argument("--roads-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--roads-layer", default="roads_fua")
    parser.add_argument("--buildings-gpkg", default="data/02_interim/akilli_sehir_fua.gpkg")
    parser.add_argument("--buildings-layer", default="buildings_fua")
    parser.add_argument("--out-dir", default="outputs/diagnostics/street_network_visual_qa_2026-06-15")
    parser.add_argument("--out-gpkg", default="data/03_processed/street_network_visual_qa_cells.gpkg")
    parser.add_argument("--out-report", default="docs/methodology/street_network_visual_qa_2026-06-15.md")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--street-buffer-m", type=float, default=20.0)
    return parser.parse_args()


def open_layer(path: str, layer_name: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer not found: {layer_name} in {path}")
    return ds, layer


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    parsed = parse_float(row.get(key))
    return default if parsed is None else parsed


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"No header in {path}")
        return list(reader)


def candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if numeric(row, "street_frontage_road_length_m") >= 150.0
    ]


def sort_key(row: dict[str, str], field: str, descending: bool) -> tuple[float, float, str]:
    value = numeric(row, field, default=-math.inf if descending else math.inf)
    road_length = numeric(row, "street_frontage_road_length_m")
    if descending:
        return (-value, -road_length, row["grid_id"])
    return (value, -road_length, row["grid_id"])


def add_category(
    selected: list[dict[str, Any]],
    used: set[str],
    candidates: list[dict[str, str]],
    category: str,
    reason: str,
    field: str,
    descending: bool,
    per_category: int,
) -> None:
    picked = 0
    for row in sorted(candidates, key=lambda item: sort_key(item, field, descending)):
        grid_id = row["grid_id"]
        if grid_id in used:
            continue
        sample = {key: row.get(key, "") for key in SAMPLE_FIELDS if key not in {"qa_category", "qa_rank", "qa_reason"}}
        sample["qa_category"] = category
        sample["qa_rank"] = picked + 1
        sample["qa_reason"] = reason
        selected.append(sample)
        used.add(grid_id)
        picked += 1
        if picked >= per_category:
            return


def build_selection(rows: list[dict[str, str]], per_category: int) -> list[dict[str, Any]]:
    road_rows = candidate_rows(rows)
    built_street_rows = [
        row
        for row in road_rows
        if numeric(row, "street_frontage_building_count") >= 5.0
        and parse_float(row.get("street_frontage_continuity_proxy")) is not None
    ]
    network_rows = [
        row
        for row in road_rows
        if parse_float(row.get("block_permeability_street_open_network_proxy")) is not None
    ]
    reach_rows = [
        row
        for row in rows
        if numeric(row, "network_reach_800m_road_density_m_per_km2") > 0.0
    ]
    for row in reach_rows:
        n250 = numeric(row, "network_reach_250m_road_density_m_per_km2")
        n400 = numeric(row, "network_reach_400m_road_density_m_per_km2")
        n800 = numeric(row, "network_reach_800m_road_density_m_per_km2")
        row["_reach_min_density"] = str(min(n250, n400, n800))
        row["_reach_800_minus_250"] = str(n800 - n250)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    add_category(
        selected,
        used,
        built_street_rows,
        "high_frontage_continuity",
        "Top road-buffer building-edge continuity among cells with roads and at least five street-buffer buildings.",
        "street_frontage_continuity_proxy",
        True,
        per_category,
    )
    add_category(
        selected,
        used,
        built_street_rows,
        "low_frontage_continuity_with_buildings",
        "Lowest road-buffer continuity among cells that still have roads and at least five street-buffer buildings.",
        "street_frontage_continuity_proxy",
        False,
        per_category,
    )
    add_category(
        selected,
        used,
        network_rows,
        "high_block_permeability_proxy",
        "Highest combined street-open and endpoint-network permeability proxy.",
        "block_permeability_street_open_network_proxy",
        True,
        per_category,
    )
    add_category(
        selected,
        used,
        network_rows,
        "low_block_permeability_with_roads",
        "Lowest combined permeability proxy among cells with road length.",
        "block_permeability_street_open_network_proxy",
        False,
        per_category,
    )
    add_category(
        selected,
        used,
        reach_rows,
        "high_network_reach_all_scales",
        "High road density across 250 m, 400 m, and 800 m centroid-buffer reach contexts.",
        "_reach_min_density",
        True,
        per_category,
    )
    add_category(
        selected,
        used,
        reach_rows,
        "high_800m_context_low_250m_local",
        "Strong broader 800 m road-density context relative to the 250 m local context.",
        "_reach_800_minus_250",
        True,
        per_category,
    )
    add_category(
        selected,
        used,
        reach_rows,
        "high_local_250m_network_reach",
        "High 250 m centroid-buffer road density for local reach QA.",
        "network_reach_250m_road_density_m_per_km2",
        True,
        per_category,
    )
    add_category(
        selected,
        used,
        reach_rows,
        "sparse_local_reach_context",
        "Low 250 m road density but non-zero broader network context.",
        "network_reach_250m_road_density_m_per_km2",
        False,
        per_category,
    )

    return selected


def make_field(name: str, value: Any) -> ogr.FieldDefn:
    if name == "qa_rank" or name.endswith("_count"):
        return ogr.FieldDefn(name, ogr.OFTInteger)
    if parse_float(str(value)) is not None and name not in {"grid_id", "lcz_weak_label", "district", "qa_category", "qa_reason"}:
        return ogr.FieldDefn(name, ogr.OFTReal)
    field = ogr.FieldDefn(name, ogr.OFTString)
    field.SetWidth(240 if name == "qa_reason" else 96)
    return field


def set_field(feature: ogr.Feature, field: str, value: Any) -> None:
    if value is None or value == "":
        return
    if field == "qa_rank" or field.endswith("_count"):
        feature.SetField(field, int(float(value)))
        return
    parsed = parse_float(str(value))
    if parsed is not None and field not in {"grid_id", "lcz_weak_label", "district", "qa_category", "qa_reason"}:
        feature.SetField(field, parsed)
    else:
        feature.SetField(field, str(value))


def safe_intersection(a: ogr.Geometry, b: ogr.Geometry) -> ogr.Geometry | None:
    try:
        inter = a.Intersection(b)
    except RuntimeError:
        inter = a.MakeValid().Intersection(b.MakeValid())
    if inter is None or inter.IsEmpty():
        return None
    return inter


def iter_line_parts(geom: ogr.Geometry | None) -> Iterable[ogr.Geometry]:
    if geom is None or geom.IsEmpty():
        return
    geom_name = geom.GetGeometryName().upper()
    if geom_name in {"LINESTRING", "LINEARRING"}:
        yield geom
        return
    if geom_name in {"MULTILINESTRING", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_line_parts(child)


def iter_polygon_parts(geom: ogr.Geometry | None) -> Iterable[ogr.Geometry]:
    if geom is None or geom.IsEmpty():
        return
    geom_name = geom.GetGeometryName().upper()
    if geom_name in {"POLYGON", "CURVEPOLYGON"}:
        yield geom
        return
    if geom_name in {"MULTIPOLYGON", "GEOMETRYCOLLECTION"}:
        for index in range(geom.GetGeometryCount()):
            child = geom.GetGeometryRef(index)
            if child is not None and not child.IsEmpty():
                yield from iter_polygon_parts(child)


def geometry_union(geoms: list[ogr.Geometry]) -> ogr.Geometry | None:
    if not geoms:
        return None
    collection = ogr.Geometry(ogr.wkbMultiPolygon)
    for geom in geoms:
        for part in iter_polygon_parts(geom):
            collection.AddGeometry(part.Clone())
    if collection.GetGeometryCount() == 0:
        return None
    try:
        union = collection.UnionCascaded()
    except (AttributeError, RuntimeError):
        union = geoms[0].Clone()
        for geom in geoms[1:]:
            union = union.Union(geom)
    if union is None or union.IsEmpty():
        return None
    return union


def selected_grid_geometries(grid_layer: ogr.Layer, selected: list[dict[str, Any]]) -> dict[str, ogr.Geometry]:
    selected_ids = {str(row["grid_id"]) for row in selected}
    geoms: dict[str, ogr.Geometry] = {}
    grid_layer.ResetReading()
    for feature in grid_layer:
        grid_id = str(feature.GetField("grid_id"))
        if grid_id not in selected_ids:
            continue
        geom = feature.GetGeometryRef()
        if geom is not None and not geom.IsEmpty():
            geoms[grid_id] = geom.Clone()
    grid_layer.ResetReading()
    missing = selected_ids - set(geoms)
    if missing:
        raise RuntimeError(f"Selected grid IDs missing from grid layer: {sorted(missing)[:10]}")
    return geoms


def create_output_ds(out_path: Path) -> ogr.DataSource:
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        driver.DeleteDataSource(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output {out_path}")
    return out_ds


def write_qa_cells(
    out_ds: ogr.DataSource,
    selected: list[dict[str, Any]],
    geoms: dict[str, ogr.Geometry],
    srs: ogr.SpatialReference,
) -> None:
    layer = out_ds.CreateLayer("qa_cells", srs=srs, geom_type=ogr.wkbPolygon)
    for field in SAMPLE_FIELDS:
        sample_value = next((row.get(field, "") for row in selected if row.get(field, "") != ""), "")
        layer.CreateField(make_field(field, sample_value))
    defn = layer.GetLayerDefn()
    for row in selected:
        feature = ogr.Feature(defn)
        for field in SAMPLE_FIELDS:
            set_field(feature, field, row.get(field, ""))
        feature.SetGeometry(geoms[str(row["grid_id"])].Clone())
        layer.CreateFeature(feature)
        feature = None
    layer.SyncToDisk()


def write_context_buffers(
    out_ds: ogr.DataSource,
    selected: list[dict[str, Any]],
    geoms: dict[str, ogr.Geometry],
    srs: ogr.SpatialReference,
    radii: tuple[int, ...] = (250, 400, 800),
) -> dict[int, list[ogr.Geometry]]:
    buffers_by_radius: dict[int, list[ogr.Geometry]] = {radius: [] for radius in radii}
    for radius in radii:
        layer = out_ds.CreateLayer(f"qa_centroid_buffer_{radius}m", srs=srs, geom_type=ogr.wkbPolygon)
        for name in ("grid_id", "qa_category", "buffer_m"):
            layer.CreateField(make_field(name, radius if name == "buffer_m" else ""))
        defn = layer.GetLayerDefn()
        for row in selected:
            grid_id = str(row["grid_id"])
            centroid = geoms[grid_id].Centroid()
            buffer_geom = centroid.Buffer(float(radius))
            buffers_by_radius[radius].append(buffer_geom.Clone())
            feature = ogr.Feature(defn)
            set_field(feature, "grid_id", grid_id)
            set_field(feature, "qa_category", row["qa_category"])
            set_field(feature, "buffer_m", radius)
            feature.SetGeometry(buffer_geom)
            layer.CreateFeature(feature)
            feature = None
        layer.SyncToDisk()
    return buffers_by_radius


def write_street_buffers(
    out_ds: ogr.DataSource,
    selected: list[dict[str, Any]],
    geoms: dict[str, ogr.Geometry],
    roads_layer: ogr.Layer,
    srs: ogr.SpatialReference,
    street_buffer_m: float,
) -> int:
    layer = out_ds.CreateLayer("qa_street_buffer_20m", srs=srs, geom_type=ogr.wkbUnknown)
    for name in ("grid_id", "qa_category", "street_buffer_m"):
        layer.CreateField(make_field(name, street_buffer_m if name == "street_buffer_m" else ""))
    defn = layer.GetLayerDefn()
    count = 0
    for row in selected:
        grid_id = str(row["grid_id"])
        cell_geom = geoms[grid_id]
        road_buffers: list[ogr.Geometry] = []
        roads_layer.SetSpatialFilter(cell_geom)
        roads_layer.ResetReading()
        for road_feature in roads_layer:
            road_geom = road_feature.GetGeometryRef()
            if road_geom is None or road_geom.IsEmpty():
                continue
            road_in_cell = safe_intersection(cell_geom, road_geom)
            if road_in_cell is None:
                continue
            for line in iter_line_parts(road_in_cell):
                if line.Length() <= 0:
                    continue
                clipped = safe_intersection(cell_geom, line.Buffer(street_buffer_m))
                if clipped is not None:
                    road_buffers.append(clipped)
        roads_layer.SetSpatialFilter(None)
        buffer_union = geometry_union(road_buffers)
        if buffer_union is None:
            continue
        feature = ogr.Feature(defn)
        set_field(feature, "grid_id", grid_id)
        set_field(feature, "qa_category", row["qa_category"])
        set_field(feature, "street_buffer_m", street_buffer_m)
        feature.SetGeometry(buffer_union)
        layer.CreateFeature(feature)
        feature = None
        count += 1
    layer.SyncToDisk()
    roads_layer.ResetReading()
    return count


def write_clipped_context_layer(
    out_ds: ogr.DataSource,
    source_layer: ogr.Layer,
    clip_geom: ogr.Geometry,
    srs: ogr.SpatialReference,
    layer_name: str,
    geom_type: int,
    max_features: int | None = None,
) -> int:
    layer = out_ds.CreateLayer(layer_name, srs=srs, geom_type=geom_type)
    layer.CreateField(make_field("source_fid", 0))
    defn = layer.GetLayerDefn()
    count = 0
    source_layer.SetSpatialFilter(clip_geom)
    source_layer.ResetReading()
    for source_feature in source_layer:
        source_geom = source_feature.GetGeometryRef()
        if source_geom is None or source_geom.IsEmpty():
            continue
        clipped = safe_intersection(clip_geom, source_geom)
        if clipped is None:
            continue
        feature = ogr.Feature(defn)
        feature.SetField("source_fid", int(source_feature.GetFID()))
        feature.SetGeometry(clipped)
        layer.CreateFeature(feature)
        feature = None
        count += 1
        if max_features is not None and count >= max_features:
            break
    source_layer.SetSpatialFilter(None)
    source_layer.ResetReading()
    layer.SyncToDisk()
    return count


def write_csv(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row.get(field, "") for field in SAMPLE_FIELDS})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            parsed = parse_float(str(value))
            if parsed is not None:
                values.append(f"{parsed:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    selected: list[dict[str, Any]],
    out_csv: Path,
    out_gpkg: Path,
    summary: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Street and Network Visual QA Sample",
        "",
        "Date: 2026-06-15",
        "",
        "## Purpose",
        "",
        "This QA package selects representative cells for visual inspection of the v6 street-facing morphology and 250/400/800 m network-reach proxy metrics.",
        "",
        "The package is for method QA, not for model training. Its main job is to catch proxy failures before the variables are interpreted in the manuscript.",
        "",
        "## Outputs",
        "",
        f"- QA cell table: `{out_csv.as_posix()}`",
        f"- QGIS package: `{out_gpkg.as_posix()}`",
        "",
        "QGIS package layers:",
        "",
        "- `qa_cells`: selected 250 m cells with QA category and key metrics.",
        "- `qa_street_buffer_20m`: road-buffer area used for the street-frontage proxy.",
        "- `qa_centroid_buffer_250m`, `qa_centroid_buffer_400m`, `qa_centroid_buffer_800m`: visual context buffers for network-reach interpretation.",
        "- `qa_roads_800m_context`: roads clipped to the union of selected 800 m context buffers.",
        "- `qa_buildings_cells_context`: building footprints clipped to the selected 250 m cells.",
        "",
        "## Selection Summary",
        "",
        f"- Selected cells: {summary['selected_cells']}",
        f"- QA categories: {summary['category_count']}",
        f"- Street-buffer features exported: {summary['street_buffer_features']}",
        f"- Road context features exported: {summary['roads_context_features']}",
        f"- Building context features exported: {summary['building_context_features']}",
        f"- Street buffer distance: {summary['street_buffer_m']} m",
        "",
        "## Selected Cells",
        "",
        markdown_table(selected, REPORT_COLUMNS),
        "",
        "## Review Checklist",
        "",
        "For each selected cell, inspect:",
        "",
        "1. Whether the 20 m street buffer captures the intended road-edge condition.",
        "2. Whether high `street_frontage_continuity_proxy` cells visibly have continuous built edges along roads.",
        "3. Whether low-continuity cells are true gaps, open campuses/industrial land, sparse roads, or source-geometry artifacts.",
        "4. Whether high `block_permeability_street_open_network_proxy` cells combine road-adjacent openness with plausible network connectivity.",
        "5. Whether 250/400/800 m buffers show the expected local-versus-broader network context contrast.",
        "",
        "## Guardrails",
        "",
        "- Treat this as a visual QA sample, not a statistical validation set.",
        "- Street-frontage fields are road-buffer proxies, not cadastral frontage or measured setbacks.",
        "- Block-permeability fields are endpoint-network and road-buffer openness proxies, not true block topology.",
        "- If several selected cells fail visual inspection, rerun the street-facing aggregation with alternative buffer distances and keep the sensitivity result in the supplement.",
        "",
        "## Rebuild",
        "",
        "```powershell",
        'python src\\diagnostics\\build_street_network_visual_qa_sample.py',
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_csv = out_dir / "street_network_visual_qa_cells.csv"
    out_summary = out_dir / "street_network_visual_qa_summary.json"
    out_gpkg = Path(args.out_gpkg)
    report_path = Path(args.out_report)

    rows = read_matrix(Path(args.matrix_csv))
    selected = build_selection(rows, args.per_category)
    if not selected:
        raise RuntimeError("No QA cells selected")

    grid_ds, grid_layer = open_layer(args.grid_gpkg, args.grid_layer)
    roads_ds, roads_layer = open_layer(args.roads_gpkg, args.roads_layer)
    buildings_ds, buildings_layer = open_layer(args.buildings_gpkg, args.buildings_layer)

    selected_geoms = selected_grid_geometries(grid_layer, selected)
    out_ds = create_output_ds(out_gpkg)
    srs = grid_layer.GetSpatialRef()
    write_qa_cells(out_ds, selected, selected_geoms, srs)
    buffers_by_radius = write_context_buffers(out_ds, selected, selected_geoms, srs)
    street_buffer_features = write_street_buffers(out_ds, selected, selected_geoms, roads_layer, srs, args.street_buffer_m)

    union_800 = geometry_union(buffers_by_radius[800])
    if union_800 is None:
        raise RuntimeError("Could not build 800 m context union")
    cell_union = geometry_union([geom.Clone() for geom in selected_geoms.values()])
    if cell_union is None:
        raise RuntimeError("Could not build selected-cell union")
    road_features = write_clipped_context_layer(
        out_ds,
        roads_layer,
        union_800,
        srs,
        "qa_roads_800m_context",
        ogr.wkbUnknown,
    )
    building_features = write_clipped_context_layer(
        out_ds,
        buildings_layer,
        cell_union,
        srs,
        "qa_buildings_cells_context",
        ogr.wkbUnknown,
    )
    out_ds = None
    grid_ds = None
    roads_ds = None
    buildings_ds = None

    write_csv(out_csv, selected)
    category_counts: dict[str, int] = {}
    for row in selected:
        category_counts[str(row["qa_category"])] = category_counts.get(str(row["qa_category"]), 0) + 1
    summary = {
        "matrix_csv": args.matrix_csv,
        "selected_cells": len(selected),
        "category_count": len(category_counts),
        "category_counts": category_counts,
        "street_buffer_m": args.street_buffer_m,
        "street_buffer_features": street_buffer_features,
        "roads_context_features": road_features,
        "building_context_features": building_features,
        "outputs": {
            "csv": out_csv.as_posix(),
            "gpkg": out_gpkg.as_posix(),
            "report": report_path.as_posix(),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(report_path, selected, out_csv, out_gpkg, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
