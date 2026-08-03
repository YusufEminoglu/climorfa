"""Render per-cell street/network QA panels and contact sheets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from osgeo import ogr
from PIL import Image, ImageDraw
from shapely import wkb
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon

ogr.UseExceptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-gpkg", default="data/03_processed/street_network_visual_qa_cells.gpkg")
    parser.add_argument(
        "--qa-csv",
        default="outputs/diagnostics/street_network_visual_qa_2026-06-15/street_network_visual_qa_cells.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/diagnostics/street_network_visual_review_2026-06-18",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_shapely(geom: ogr.Geometry | None):
    if geom is None or geom.IsEmpty():
        return None
    return wkb.loads(bytes(geom.ExportToWkb()))


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms
    elif isinstance(geom, GeometryCollection):
        for child in geom.geoms:
            yield from iter_polygons(child)


def iter_lines(geom) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from geom.geoms
    elif isinstance(geom, Polygon):
        yield LineString(geom.exterior.coords)
        for ring in geom.interiors:
            yield LineString(ring.coords)
    elif isinstance(geom, MultiPolygon):
        for polygon in geom.geoms:
            yield from iter_lines(polygon)
    elif isinstance(geom, GeometryCollection):
        for child in geom.geoms:
            yield from iter_lines(child)


def plot_polygon_fill(ax, geom, facecolor: str, edgecolor: str, alpha: float, linewidth: float, zorder: int) -> None:
    for polygon in iter_polygons(geom):
        patch = MplPolygon(
            list(polygon.exterior.coords),
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
        ax.add_patch(patch)
        for ring in polygon.interiors:
            hole = MplPolygon(
                list(ring.coords),
                closed=True,
                facecolor=ax.get_facecolor(),
                edgecolor=edgecolor,
                alpha=1.0,
                linewidth=max(linewidth * 0.6, 0.2),
                zorder=zorder + 0.1,
            )
            ax.add_patch(hole)


def plot_lines(ax, geom, color: str, linewidth: float, alpha: float, zorder: int, linestyle: str = "-") -> None:
    for line in iter_lines(geom):
        xs, ys = line.xy
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, linestyle=linestyle)


def get_feature_geom(layer: ogr.Layer, grid_id: str):
    layer.SetAttributeFilter(f"grid_id = '{grid_id}'")
    layer.ResetReading()
    feature = layer.GetNextFeature()
    geom = to_shapely(feature.GetGeometryRef()) if feature is not None else None
    layer.SetAttributeFilter(None)
    layer.ResetReading()
    return geom


def spatial_geometries(layer: ogr.Layer, filter_geom: ogr.Geometry) -> list:
    geoms = []
    layer.SetSpatialFilter(filter_geom)
    layer.ResetReading()
    for feature in layer:
        geom = to_shapely(feature.GetGeometryRef())
        if geom is not None and not geom.is_empty:
            geoms.append(geom)
    layer.SetSpatialFilter(None)
    layer.ResetReading()
    return geoms


def ogr_filter_geom(layer: ogr.Layer, grid_id: str) -> ogr.Geometry:
    layer.SetAttributeFilter(f"grid_id = '{grid_id}'")
    layer.ResetReading()
    feature = layer.GetNextFeature()
    if feature is None or feature.GetGeometryRef() is None:
        raise RuntimeError(f"Geometry not found for {grid_id} in {layer.GetName()}")
    geom = feature.GetGeometryRef().Clone()
    layer.SetAttributeFilter(None)
    layer.ResetReading()
    return geom


def set_extent(ax, geom, pad: float) -> None:
    min_x, min_y, max_x, max_y = geom.bounds
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def render_cell(ds: ogr.DataSource, row: dict[str, str], out_path: Path, dpi: int) -> None:
    grid_id = row["grid_id"]
    cells = ds.GetLayerByName("qa_cells")
    street_buffers = ds.GetLayerByName("qa_street_buffer_20m")
    buffer_250_layer = ds.GetLayerByName("qa_centroid_buffer_250m")
    buffer_400_layer = ds.GetLayerByName("qa_centroid_buffer_400m")
    buffer_800_layer = ds.GetLayerByName("qa_centroid_buffer_800m")
    roads_layer = ds.GetLayerByName("qa_roads_800m_context")
    buildings_layer = ds.GetLayerByName("qa_buildings_cells_context")

    cell_ogr = ogr_filter_geom(cells, grid_id)
    buffer_800_ogr = ogr_filter_geom(buffer_800_layer, grid_id)
    cell = to_shapely(cell_ogr)
    street_buffer = get_feature_geom(street_buffers, grid_id)
    buffer_250 = get_feature_geom(buffer_250_layer, grid_id)
    buffer_400 = get_feature_geom(buffer_400_layer, grid_id)
    buffer_800 = to_shapely(buffer_800_ogr)
    local_roads = spatial_geometries(roads_layer, cell_ogr)
    context_roads = spatial_geometries(roads_layer, buffer_800_ogr)
    local_buildings = spatial_geometries(buildings_layer, cell_ogr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#f7f8fa")
    for ax in axes:
        ax.set_facecolor("#f7f8fa")

    local_ax, context_ax = axes
    for geom in local_buildings:
        plot_polygon_fill(local_ax, geom, "#697386", "#4a5568", 0.72, 0.35, 3)
    if street_buffer is not None:
        plot_polygon_fill(local_ax, street_buffer, "#f4c95d", "#c79a20", 0.32, 0.8, 2)
    for geom in local_roads:
        plot_lines(local_ax, geom, "#c23b3b", 1.2, 0.95, 4)
    plot_lines(local_ax, cell, "#111827", 2.1, 1.0, 6)
    set_extent(local_ax, cell, 28.0)
    local_ax.set_title("Local 250 m street context", fontsize=12, weight="bold")
    if street_buffer is None:
        local_ax.text(
            0.5,
            0.06,
            "No in-cell road / no 20 m street buffer",
            transform=local_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#8b1e1e",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#d8dee8", "pad": 4},
        )

    for geom in context_roads:
        plot_lines(context_ax, geom, "#48576a", 0.55, 0.72, 2)
    plot_lines(context_ax, buffer_800, "#b33f40", 1.6, 0.95, 3, "--")
    plot_lines(context_ax, buffer_400, "#34699a", 1.6, 0.95, 4, "--")
    plot_lines(context_ax, buffer_250, "#2a7f62", 1.8, 0.95, 5, "--")
    plot_lines(context_ax, cell, "#111827", 2.0, 1.0, 6)
    set_extent(context_ax, buffer_800, 35.0)
    context_ax.set_title("Network reach context", fontsize=12, weight="bold")

    district = row.get("district", "unknown") or "unknown"
    fig.suptitle(
        f"{row['qa_category']} | {grid_id} | {district}",
        fontsize=15,
        weight="bold",
        y=0.965,
    )
    continuity = as_float(row, "street_frontage_continuity_proxy")
    open_share = as_float(row, "street_frontage_open_buffer_share")
    permeability = as_float(row, "block_permeability_street_open_network_proxy")
    reach_250 = as_float(row, "network_reach_250m_road_density_m_per_km2")
    reach_400 = as_float(row, "network_reach_400m_road_density_m_per_km2")
    reach_800 = as_float(row, "network_reach_800m_road_density_m_per_km2")
    footer = (
        f"continuity={fmt(continuity)}   open_share={fmt(open_share)}   permeability={fmt(permeability)}\n"
        f"road density: 250 m={fmt(reach_250, 0)}   400 m={fmt(reach_400, 0)}   800 m={fmt(reach_800, 0)} m/km2"
    )
    fig.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=10.5, color="#263241")

    legend = [
        Patch(facecolor="#697386", edgecolor="#4a5568", alpha=0.72, label="Buildings"),
        Patch(facecolor="#f4c95d", edgecolor="#c79a20", alpha=0.32, label="20 m street buffer"),
        Line2D([0], [0], color="#c23b3b", linewidth=1.5, label="Roads"),
        Line2D([0], [0], color="#2a7f62", linewidth=1.8, linestyle="--", label="250 m"),
        Line2D([0], [0], color="#34699a", linewidth=1.6, linestyle="--", label="400 m"),
        Line2D([0], [0], color="#b33f40", linewidth=1.6, linestyle="--", label="800 m"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.075), fontsize=9)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.17, wspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def contact_sheet(image_paths: list[Path], out_path: Path, columns: int = 2) -> None:
    thumb_w, thumb_h = 900, 450
    rows = math.ceil(len(image_paths) / columns)
    canvas = Image.new("RGB", (thumb_w * columns, thumb_h * rows), "#e9edf2")
    draw = ImageDraw.Draw(canvas)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
            y = (index // columns) * thumb_h + (thumb_h - image.height) // 2
            canvas.paste(image, (x, y))
        draw.rectangle(
            [
                (index % columns) * thumb_w,
                (index // columns) * thumb_h,
                (index % columns + 1) * thumb_w - 1,
                (index // columns + 1) * thumb_h - 1,
            ],
            outline="#c8d0da",
            width=2,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.qa_csv))
    ds = ogr.Open(args.qa_gpkg)
    if ds is None:
        raise RuntimeError(f"Could not open {args.qa_gpkg}")

    out_dir = Path(args.out_dir)
    cells_dir = out_dir / "cells"
    image_paths: list[Path] = []
    for index, row in enumerate(rows, start=1):
        safe_category = row["qa_category"].replace("/", "_")
        out_path = cells_dir / f"{index:02d}_{safe_category}_{row['grid_id']}.png"
        render_cell(ds, row, out_path, args.dpi)
        image_paths.append(out_path)
        print(f"rendered={index}/{len(rows)} {out_path.as_posix()}")

    midpoint = math.ceil(len(image_paths) / 2)
    contact_sheet(image_paths[:midpoint], out_dir / "contact_sheet_01.png")
    contact_sheet(image_paths[midpoint:], out_dir / "contact_sheet_02.png")
    print(f"cell_images={len(image_paths)}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
