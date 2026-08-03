"""Build regular analysis grids for the Izmir FUA study area.

The script reads a polygon boundary layer, creates a regular grid, and writes
all cells intersecting the boundary with intersection area and ratio fields.
It uses GDAL/OGR so it can run in the OSGeo4W Python environment.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from osgeo import ogr, osr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", required=True, help="Boundary dataset.")
    parser.add_argument("--boundary-layer", required=True, help="Boundary layer name.")
    parser.add_argument("--out-gpkg", required=True, help="Output GeoPackage.")
    parser.add_argument("--out-layer", default="grid_250m")
    parser.add_argument("--cell-size", type=float, default=250.0)
    parser.add_argument("--min-ratio", type=float, default=0.0)
    return parser.parse_args()


def get_union_geometry(layer: ogr.Layer) -> ogr.Geometry:
    union = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        if union is None:
            union = geom
        else:
            union = union.Union(geom)
    if union is None:
        raise ValueError("Boundary layer has no valid geometry.")
    return union.MakeValid()


def make_cell(x0: float, y0: float, size: float) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(x0, y0)
    ring.AddPoint_2D(x0 + size, y0)
    ring.AddPoint_2D(x0 + size, y0 + size)
    ring.AddPoint_2D(x0, y0 + size)
    ring.AddPoint_2D(x0, y0)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def create_output_layer(out_path: Path, layer_name: str, srs: osr.SpatialReference):
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        ds_existing = driver.Open(str(out_path), update=1)
        if ds_existing is not None:
            existing = ds_existing.GetLayerByName(layer_name)
            if existing is not None:
                ds_existing.DeleteLayer(layer_name)
            ds_existing = None
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = driver.CreateDataSource(str(out_path)) if not out_path.exists() else driver.Open(str(out_path), update=1)
    if ds is None:
        raise RuntimeError(f"Could not create/open {out_path}")

    layer = ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    fields = [
        ("grid_id", ogr.OFTString, 40),
        ("cell_m", ogr.OFTReal, None),
        ("cell_area_m2", ogr.OFTReal, None),
        ("fua_area_m2", ogr.OFTReal, None),
        ("fua_ratio", ogr.OFTReal, None),
        ("eligible_core", ogr.OFTInteger, None),
        ("row_id", ogr.OFTInteger, None),
        ("col_id", ogr.OFTInteger, None),
    ]
    for name, ftype, width in fields:
        field = ogr.FieldDefn(name, ftype)
        if width:
            field.SetWidth(width)
        layer.CreateField(field)
    return ds, layer


def main() -> None:
    args = parse_args()
    boundary_ds = ogr.Open(args.boundary)
    if boundary_ds is None:
        raise RuntimeError(f"Could not open boundary dataset: {args.boundary}")
    boundary_layer = boundary_ds.GetLayerByName(args.boundary_layer)
    if boundary_layer is None:
        raise RuntimeError(f"Boundary layer not found: {args.boundary_layer}")

    srs = boundary_layer.GetSpatialRef()
    if srs is None:
        raise RuntimeError("Boundary layer has no CRS.")

    boundary_geom = get_union_geometry(boundary_layer)
    minx, maxx, miny, maxy = boundary_geom.GetEnvelope()
    size = args.cell_size

    start_x = math.floor(minx / size) * size
    end_x = math.ceil(maxx / size) * size
    start_y = math.floor(miny / size) * size
    end_y = math.ceil(maxy / size) * size

    out_path = Path(args.out_gpkg)
    out_ds, out_layer = create_output_layer(out_path, args.out_layer, srs)
    defn = out_layer.GetLayerDefn()

    created = 0
    rows = int(round((end_y - start_y) / size))
    cols = int(round((end_x - start_x) / size))
    cell_area = size * size

    for row in range(rows):
        y0 = start_y + row * size
        for col in range(cols):
            x0 = start_x + col * size
            cell = make_cell(x0, y0, size)
            if not cell.Intersects(boundary_geom):
                continue
            inter = cell.Intersection(boundary_geom)
            fua_area = inter.GetArea() if inter is not None and not inter.IsEmpty() else 0.0
            fua_ratio = fua_area / cell_area
            if fua_ratio <= args.min_ratio:
                continue

            feature = ogr.Feature(defn)
            feature.SetField("grid_id", f"g{int(size)}_{row:05d}_{col:05d}")
            feature.SetField("cell_m", size)
            feature.SetField("cell_area_m2", cell_area)
            feature.SetField("fua_area_m2", fua_area)
            feature.SetField("fua_ratio", fua_ratio)
            feature.SetField("eligible_core", int(fua_ratio >= 0.5))
            feature.SetField("row_id", row)
            feature.SetField("col_id", col)
            feature.SetGeometry(cell)
            out_layer.CreateFeature(feature)
            feature = None
            created += 1

    out_layer.SyncToDisk()
    out_ds = None
    print(f"grid={args.out_layer} cell_size={size} rows={rows} cols={cols} features={created}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
