"""WP3.0 shared constants and helpers (street units, PS index, segment metrics).

CRS EPSG:5253. Tiles per docs/climorfa_street/grid_spec.md. Analysis grid per docs/climorfa_street/analysis_schema.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# make PROJ / GDAL data discoverable when launched outside an activated conda env
for _p in (Path(sys.prefix) / "Library/share/proj", Path(sys.prefix) / "share/proj"):
    if _p.exists():
        os.environ.setdefault("PROJ_LIB", str(_p))
        os.environ.setdefault("PROJ_DATA", str(_p))
        break
for _g in (Path(sys.prefix) / "Library/share/gdal", Path(sys.prefix) / "share/gdal"):
    if _g.exists():
        os.environ.setdefault("GDAL_DATA", str(_g))
        break

ROOT = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ANALYSIS_DIR = ROOT / "data/street/analysis"
PS_DIR = ANALYSIS_DIR / "ps_index"
SEGMENTS_GPKG = ANALYSIS_DIR / "segments.gpkg"
TILES_DONE_CSV = ANALYSIS_DIR / "WP3.0_tiles_done.csv"
OUT_DIR = ROOT / "outputs/street/analysis/wp30"
PRE_PS_GPKG = OUT_DIR / "segments_pre_ps.gpkg"
TILES_CORE = ROOT / "data/street/grid/tiles_core.gpkg"

ROADS_GPKG = ROOT / "data/02_interim/akilli_sehir_fua.gpkg"
SHORELINE_GPKG = ROOT / "data/02_interim/shoreline_osm_epsg5253.gpkg"
WEAK_GPKG = ROOT / "data/03_processed/grid_lcz_weak_labels.gpkg"
LST_TIF = ROOT / "data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif"
LCZ_TIF = ROOT / "data/01_raw/gee/lcz/izmir_fua_rub_lcz_latest.tif"

TILE_DIR = ROOT / "data/street/tiles/2m"
SOLWEIG_DIR = ROOT / "data/street/solweig"
UTCI_DIR = ROOT / "data/street/utci"
RATIO16_DIR = ROOT / "data/street/wind/ratio16"
MET_FILES = {
    "IMBAT": ROOT / "data/street/forcing_summer/met_DESIGN_IMBAT_2022-07-05_hourly.txt",
    "CALM": ROOT / "data/street/forcing_summer/met_DESIGN_CALM_2023-07-26_hourly.txt",
}
DAYS = ("IMBAT", "CALM")

PS_BUFFER_M = 15.0
MIN_PS_CELLS = 25
MIN_SEG_LEN_M = 20.0
TILE_M = 1000
BUF_PX = 50           # 100 m buffer at 2 m
INNER_PX = 500
RES = 2.0


def tile_xy(tile_id: str) -> tuple[int, int]:
    """Lower-left corner (x0, y0) in metres of the inner square."""
    _, a, b = tile_id.split("_")
    return int(a) * 1000, int(b) * 1000


def tile_id_from_xy(x, y) -> str:
    import numpy as np
    kx = int(np.floor(x / TILE_M))
    ky = int(np.floor(y / TILE_M))
    return f"T_{kx:04d}_{ky:04d}"


def lower_priority():
    """Run below normal priority so the SOLWEIG / UTCI jobs are never slowed."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
