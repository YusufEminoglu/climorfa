"""Download an authoritative OSM coastline for the FUR and reproject it to the
analysis CRS, replacing the CRS-unverified local shoreline snapshot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
import requests
from shapely.geometry import LineString

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-area", default="data/02_interim/study_area_fua.gpkg")
    parser.add_argument("--study-layer", default="study_area_fua")
    parser.add_argument("--raw-json", default="data/01_raw/osm/osm_coastline_overpass.json")
    parser.add_argument("--out-gpkg", default="data/02_interim/shoreline_osm_epsg5253.gpkg")
    parser.add_argument("--out-layer", default="shoreline_osm_epsg5253")
    parser.add_argument("--target-epsg", type=int, default=5253)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--margin-deg", type=float, default=0.15)
    return parser.parse_args()


def bbox_wgs84(study: gpd.GeoDataFrame, margin_deg: float) -> tuple[float, float, float, float]:
    study4326 = study.to_crs(4326)
    minx, miny, maxx, maxy = study4326.total_bounds
    return (
        float(minx) - margin_deg,
        float(miny) - margin_deg,
        float(maxx) + margin_deg,
        float(maxy) + margin_deg,
    )


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    box = f"({south},{west},{north},{east})"
    return f"""
[out:json][timeout:240];
(
  way["natural"="coastline"]{box};
);
out geom;
"""


def fetch_coastline(raw_path: Path, bbox: tuple[float, float, float, float], force: bool) -> dict:
    if raw_path.exists() and not force:
        return json.loads(raw_path.read_text(encoding="utf-8"))
    query = overpass_query(bbox)
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={
            "User-Agent": "izmir-lcz-deep-morphology/0.1 (research; contact yusuf.eminoglu@deu.edu.tr)",
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_lines(payload: dict) -> list[LineString]:
    lines = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        coords = [(node["lon"], node["lat"]) for node in geometry]
        lines.append(LineString(coords))
    return lines


def main() -> None:
    args = parse_args()
    study = gpd.read_file(args.study_area, layer=args.study_layer)
    if study.crs is None:
        raise RuntimeError("Study area CRS is missing")

    bbox = bbox_wgs84(study, args.margin_deg)
    print(f"bbox_wgs84={bbox}")

    payload = fetch_coastline(Path(args.raw_json), bbox, args.force_download)
    lines = build_lines(payload)
    if not lines:
        raise RuntimeError("No coastline ways returned by Overpass")
    print(f"osm_coastline_ways={len(lines)}")

    gdf = gpd.GeoDataFrame({"osm_way_count": [len(lines)] * len(lines)}, geometry=lines, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=args.target_epsg)

    out_path = Path(args.out_gpkg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_proj.to_file(out_path, layer=args.out_layer, driver="GPKG")

    total_bounds = gdf_proj.total_bounds
    print(f"out_gpkg={out_path}")
    print(f"out_layer={args.out_layer}")
    print(f"projected_extent={tuple(total_bounds)}")
    print(f"crs={gdf_proj.crs}")


if __name__ == "__main__":
    main()
