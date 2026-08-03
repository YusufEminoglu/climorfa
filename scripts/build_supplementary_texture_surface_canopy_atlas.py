"""Build the merged supplementary atlas: for each of the same 48 selected
cells used throughout this paper's texture references, one tile now shows
figure-ground texture, 5 m surface elevation, and 10 m canopy height side by
side, instead of the two separate 48-panel figures (S1 texture-only, S2
surface/canopy-only) this replaces. Same 8x6 selection grid and selection rule
as before (see `_select_atlas_tiles`); merging removes a whole redundant
supplementary figure from the paper while giving each example cell richer,
directly comparable evidence in one place rather than requiring a reader to
flip between two atlases keyed by the same grid_id.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from build_figure_1_texture_atlas import CLASS_COLORS, PALETTE, _clip_to_cell, _safe_read_layer
from build_supplementary_texture_atlas import COLUMNS, _select_atlas_tiles

DSM_PATH = "data/02_interim/rasters/izmir_fua_dsm5m_epsg5253.tif"
CANOPY_PATH = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"


def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


DSM_CMAP = _cmap("climorfa_dsm", [PALETTE["taupe_light"], PALETTE["teal"], PALETTE["teal_dark"]])
CANOPY_CMAP = _cmap("climorfa_canopy", [PALETTE["taupe_light"], PALETTE["orange"], PALETTE["teal"]])


def _read_raster_tile(src: rasterio.io.DatasetReader, geom) -> np.ndarray:
    arr, _ = mask(src, [geom], crop=True, filled=True)
    data = arr[0].astype("float32")
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data


def _collect_rasters(root: Path, selected: pd.DataFrame, grid: gpd.GeoDataFrame) -> tuple[dict[str, dict], dict[str, float]]:
    cell_lookup = grid.set_index("grid_id")
    rasters: dict[str, dict] = {}
    dsm_values, canopy_values = [], []
    with rasterio.open(root / DSM_PATH) as dsm, rasterio.open(root / CANOPY_PATH) as canopy:
        for _, row in selected.iterrows():
            cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
            geom = cell.geometry.iloc[0]
            dsm_arr = _read_raster_tile(dsm, geom)
            canopy_arr = _read_raster_tile(canopy, geom)
            dsm_values.append(dsm_arr[np.isfinite(dsm_arr)])
            canopy_values.append(canopy_arr[np.isfinite(canopy_arr)])
            rasters[str(row["grid_id"])] = {"cell": cell, "dsm": dsm_arr, "canopy": canopy_arr}
    dsm_all = np.concatenate([x for x in dsm_values if len(x)])
    canopy_all = np.concatenate([x for x in canopy_values if len(x)])
    limits = {
        "dsm_p02": float(np.nanpercentile(dsm_all, 2)),
        "dsm_p98": float(np.nanpercentile(dsm_all, 98)),
        "canopy_p02": max(0.0, float(np.nanpercentile(canopy_all, 2))),
        "canopy_p98": max(5.0, float(np.nanpercentile(canopy_all, 98))),
    }
    return rasters, limits


def _plot_merged_tile(
    ax: plt.Axes,
    root: Path,
    row: pd.Series,
    raster: dict,
    limits: dict[str, float],
    row_index: int,
    col_index: int,
) -> None:
    klass = int(row["lcz_weak_label"])
    ax.axis("off")
    ax.set_facecolor(PALETTE["surface"])

    # Keep the three matched sub-panels visually contiguous while retaining a
    # hairline separator so their borders remain distinguishable at print size.
    tex = ax.inset_axes([0.00, 0.12, 0.326, 0.82])
    dsm_ax = ax.inset_axes([0.337, 0.12, 0.326, 0.82])
    canopy_ax = ax.inset_axes([0.674, 0.12, 0.326, 0.82])

    cell = raster["cell"]
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)
    roads = _clip_to_cell(_safe_read_layer(root, "roads_fua", bounds), geom)
    if not roads.empty:
        roads.plot(ax=tex, color=PALETTE["taupe_dark"], linewidth=0.45, alpha=0.85, zorder=2)
    if not buildings.empty:
        buildings.plot(ax=tex, facecolor=CLASS_COLORS[klass], edgecolor=PALETTE["ink"], linewidth=0.08, alpha=0.92, zorder=3)
    cell.boundary.plot(ax=tex, color=PALETTE["ink"], linewidth=0.7, zorder=4)
    tex.set_xlim(bounds[0], bounds[2])
    tex.set_ylim(bounds[1], bounds[3])
    tex.set_aspect("equal")
    tex.set_xticks([])
    tex.set_yticks([])
    for spine in tex.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
        spine.set_linewidth(0.5)

    dsm_ax.imshow(raster["dsm"], cmap=DSM_CMAP, vmin=limits["dsm_p02"], vmax=limits["dsm_p98"])
    canopy_ax.imshow(raster["canopy"], cmap=CANOPY_CMAP, vmin=limits["canopy_p02"], vmax=limits["canopy_p98"])
    for child in (dsm_ax, canopy_ax):
        child.set_xticks([])
        child.set_yticks([])
        for spine in child.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
            spine.set_linewidth(0.5)

    if row_index == 0:
        tex.set_title("Texture", fontsize=5.6, pad=1.5)
        dsm_ax.set_title("Surface", fontsize=5.6, pad=1.5)
        canopy_ax.set_title("Canopy", fontsize=5.6, pad=1.5)

    district = str(row["district"])
    if len(district) > 13:
        district = district[:12] + "."
    label = (
        f"{row['grid_id']} | {district} | cov {100 * row['building_coverage_exact']:.0f}%  "
        f"h {row['height_proxy_aw_mean_m']:.1f}m  DSMsd {row['dsm_elevation_m_std']:.1f}m\n"
        f"canopy {100 * row['canopy_cover_gt2m_share']:.0f}%  NDVI {row['s2_ndvi_mean']:.2f}  "
        f"LST {row['lst_c_median_mean']:.1f} C"
    )
    ax.text(
        0.01,
        0.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.4,
        color=PALETTE["ink"],
        linespacing=1.05,
        bbox={"facecolor": "white", "edgecolor": CLASS_COLORS[klass], "linewidth": 0.55, "alpha": 0.93, "pad": 0.8},
        zorder=50,
        clip_on=False,
    )
    if col_index == 0:
        thermal = "cool LST Q1" if row["thermal_group"] == "cool" else "hot LST Q4"
        ax.text(
            -0.09,
            0.56,
            f"LCZ {klass}\n{thermal}",
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": CLASS_COLORS[klass], "linewidth": 0.8, "alpha": 0.96, "pad": 2.6},
        )


def build_texture_dsm_canopy_atlas(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    out_dir = root / "outputs/diagnostics/supplementary_texture_dsm_canopy_atlas_2026-08-03"
    fig_dir = root / "paper/figures/supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    selected = _select_atlas_tiles(df)
    selected.to_csv(out_dir / "fig_s1_texture_dsm_canopy_atlas_selected_tiles.csv", index=False)
    rasters, limits = _collect_rasters(root, selected, grid)

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    # A shorter canvas removes the artificial vertical bands between the
    # eight evidence rows while preserving the six-column reading width.
    fig = plt.figure(figsize=(18.4, 19.0), constrained_layout=False)
    # The label box belongs to the row above; remove inter-row padding so the
    # eight evidence rows read as one compact atlas rather than separated bands.
    gs = fig.add_gridspec(8, 6, left=0.055, right=0.99, top=0.955, bottom=0.055, wspace=0.035, hspace=-0.20)

    for r in range(8):
        for c in range(6):
            ax = fig.add_subplot(gs[r, c])
            row = selected[(selected["row_index"] == r) & (selected["col_index"] == c)].iloc[0]
            raster = rasters[str(row["grid_id"])]
            _plot_merged_tile(ax, root, row, raster, limits, r, c)
            if r == 0:
                title, subtitle = COLUMNS[c]
                ax.text(0.5, 1.09, f"{title}\n{subtitle}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, color=PALETTE["ink"])

    dsm_norm = mpl.colors.Normalize(vmin=limits["dsm_p02"], vmax=limits["dsm_p98"])
    canopy_norm = mpl.colors.Normalize(vmin=limits["canopy_p02"], vmax=limits["canopy_p98"])
    cax1 = fig.add_axes([0.30, 0.040, 0.18, 0.007])
    cax2 = fig.add_axes([0.54, 0.040, 0.18, 0.007])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=dsm_norm, cmap=DSM_CMAP), cax=cax1, orientation="horizontal")
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=canopy_norm, cmap=CANOPY_CMAP), cax=cax2, orientation="horizontal")
    cb1.set_label("Surface elevation (m), common p02-p98 scale", fontsize=7)
    cb2.set_label("Canopy height (m), common p02-p98 scale", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    cb2.ax.tick_params(labelsize=6)

    fig.text(
        0.5,
        0.008,
        "Sources: grid_250m_model_features_v8.csv, analysis_grids.gpkg, local building/road vectors,\n"
        f"{DSM_PATH}, {CANOPY_PATH}. Surface model is surface elevation/roughness evidence, not validated building height.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=PALETTE["muted"],
        linespacing=1.4,
    )

    png = fig_dir / "fig_s1_texture_dsm_canopy_atlas_48_cells.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    qa = {
        "figure": "fig_s1_texture_dsm_canopy_atlas_48_cells",
        "date": "2026-08-03",
        "panel_count": 48,
        "layout": "6 columns x 8 rows, 3 raster/vector subpanels (texture, DSM, canopy) per selected cell",
        "revision_2026-08-03": "Merges the former separate fig_s1_texture_atlas_48_cells and fig_s2_dsm_canopy_atlas_48_cells into one figure per user request to reduce total supplementary figure count -- same 48 cells, same selection rule, now shown together per cell instead of requiring a reader to cross-reference two atlases by grid_id.",
        "row_rule": "For each LCZ 3/6/8/9, rows are within-class LST Q1 and Q4 among analysis candidates.",
        "column_rule": "Six deterministic prototypes: median fabric, high coverage, high height/DSM roughness, high network density, high green signal, and thermal edge.",
        "raster_limits": limits,
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "selected_tiles_csv": str(out_dir / "fig_s1_texture_dsm_canopy_atlas_selected_tiles.csv"),
        "png": str(png),
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/akilli_sehir_fua.gpkg",
            DSM_PATH,
            CANOPY_PATH,
        ],
        "claim_boundary": "Surface panels show surface elevation/roughness context; they are not interpreted as validated building heights.",
    }
    (out_dir / "fig_s1_texture_dsm_canopy_atlas_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_s1_texture_dsm_canopy_atlas_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_texture_dsm_canopy_atlas(Path(".")), indent=2))
