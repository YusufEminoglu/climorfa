"""Build the merged supplementary atlas: for each of the same 32 selected
cells used throughout this paper's texture references, one tile shows
figure-ground texture, Landsat 30 m LST (°C), and 10 m canopy height side by
side. Same 8x4 selection grid and selection rule as before.
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

LST_PATH = "data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif"
CANOPY_PATH = "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"


def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


LST_CMAP = _cmap(
    "climorfa_lst_purple_orange",
    [
        "#240038",  # Deep royal purple (Coolest)
        "#4c126b",  # Dark violet
        "#7b277d",  # Purple magenta
        "#b33d79",  # Deep rose
        "#e05c5c",  # Coral pink
        "#f48642",  # Bright orange
        "#f9b353",  # Warm gold
        "#fef1a0",  # Light amber (Hottest)
    ],
)
CANOPY_CMAP = _cmap(
    "climorfa_canopy_green",
    ["#f7fcf5", "#d5efda", "#8fd496", "#32a051", "#055222"],
)


import matplotlib.patheffects as path_effects


def _read_raster_tile(src: rasterio.io.DatasetReader, geom, min_val: float | None = None, max_val: float | None = None) -> np.ndarray:
    # Read using bounding box envelope so the raster chip fills 100% of the 250m cell without edge gaps
    arr, _ = mask(src, [geom.envelope], crop=True, filled=True)
    data = arr[0].astype("float32")
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    if min_val is not None:
        data[data < min_val] = np.nan
    if max_val is not None:
        data[data > max_val] = np.nan
    
    # Fill any edge NaNs cleanly with nearest valid values
    if np.isnan(data).any():
        df_d = pd.DataFrame(data).ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
        data = df_d.to_numpy()
    return data


def _collect_rasters(root: Path, selected: pd.DataFrame, grid: gpd.GeoDataFrame) -> tuple[dict[str, dict], dict[str, float]]:
    cell_lookup = grid.set_index("grid_id")
    rasters: dict[str, dict] = {}
    lst_values, canopy_values = [], []
    with rasterio.open(root / LST_PATH) as lst, rasterio.open(root / CANOPY_PATH) as canopy:
        for _, row in selected.iterrows():
            cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
            geom = cell.geometry.iloc[0]
            lst_arr = _read_raster_tile(lst, geom, min_val=15.0, max_val=60.0)
            canopy_arr = _read_raster_tile(canopy, geom)
            lst_values.append(lst_arr[np.isfinite(lst_arr)])
            canopy_values.append(canopy_arr[np.isfinite(canopy_arr)])
            rasters[str(row["grid_id"])] = {"cell": cell, "lst": lst_arr, "canopy": canopy_arr}
    lst_all = np.concatenate([x for x in lst_values if len(x)])
    canopy_all = np.concatenate([x for x in canopy_values if len(x)])
    limits = {
        "lst_p02": float(np.nanpercentile(lst_all, 2)),
        "lst_p98": float(np.nanpercentile(lst_all, 98)),
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
    ax.set_facecolor("none")

    # Keep the three matched sub-panels visually contiguous with exact 1:1 aspect matching
    h_top = 0.98 if row_index > 0 else 0.91
    h_span = h_top - 0.01

    # Draw a distinct dashed container card background framing the 3-panel group as a single unit
    card = plt.Rectangle(
        (-0.012, 0.00),
        1.024,
        h_top,
        transform=ax.transAxes,
        facecolor="#faf8f5",
        edgecolor="#8c867e",
        linewidth=0.80,
        linestyle=(0, (4, 3)),
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(card)

    tex = ax.inset_axes([0.000, 0.01, 0.324, h_span])
    lst_ax = ax.inset_axes([0.338, 0.01, 0.324, h_span])
    canopy_ax = ax.inset_axes([0.676, 0.01, 0.324, h_span])

    cell = raster["cell"]
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)
    roads = _clip_to_cell(_safe_read_layer(root, "roads_fua", bounds), geom)
    if not roads.empty:
        roads.plot(ax=tex, color=PALETTE["taupe_dark"], linewidth=0.50, alpha=0.85, zorder=2)
    if not buildings.empty:
        buildings.plot(ax=tex, facecolor=CLASS_COLORS[klass], edgecolor=PALETTE["ink"], linewidth=0.10, alpha=0.92, zorder=3)
    cell.boundary.plot(ax=tex, color=PALETTE["ink"], linewidth=0.75, zorder=4)
    tex.set_xlim(bounds[0], bounds[2])
    tex.set_ylim(bounds[1], bounds[3])
    tex.set_aspect("equal")
    tex.set_xticks([])
    tex.set_yticks([])
    for spine in tex.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
        spine.set_linewidth(0.55)

    lst_ax.imshow(
        raster["lst"],
        cmap=LST_CMAP,
        vmin=limits["lst_p02"],
        vmax=limits["lst_p98"],
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        interpolation="bilinear",
        origin="upper",
    )
    lst_ax.set_xlim(bounds[0], bounds[2])
    lst_ax.set_ylim(bounds[1], bounds[3])
    lst_ax.set_aspect("equal")
    
    # Overlay high-contrast temperature contour lines (isotherms / isohips)
    lst_data = raster["lst"]
    valid_lst = lst_data[np.isfinite(lst_data)]
    if len(valid_lst) > 0 and (valid_lst.max() - valid_lst.min()) >= 0.15:
        try:
            levels = np.linspace(valid_lst.min() + 0.10, valid_lst.max() - 0.10, 3)
            cs = lst_ax.contour(
                lst_data,
                levels=levels,
                extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                colors="#111111",
                linewidths=0.55,
                alpha=0.75,
                linestyles="-",
            )
            clabels = lst_ax.clabel(cs, fmt="%.1f°C", fontsize=4.8, inline=True, inline_spacing=3, colors="black")
            if clabels:
                for txt in clabels:
                    txt.set_fontweight("bold")
                    txt.set_color("#111111")
                    txt.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])
        except Exception:
            pass

    canopy_ax.imshow(
        raster["canopy"],
        cmap=CANOPY_CMAP,
        vmin=limits["canopy_p02"],
        vmax=limits["canopy_p98"],
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
        interpolation="nearest",
        origin="upper",
    )
    canopy_ax.set_xlim(bounds[0], bounds[2])
    canopy_ax.set_ylim(bounds[1], bounds[3])
    canopy_ax.set_aspect("equal")
    for child in (lst_ax, canopy_ax):
        child.set_xticks([])
        child.set_yticks([])
        for spine in child.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
            spine.set_linewidth(0.55)

    if row_index == 0:
        tex.set_title("Texture", fontsize=6.8, pad=2.5, fontweight="bold", color=PALETTE["ink"])
        lst_ax.set_title("LST (°C)", fontsize=6.8, pad=2.5, fontweight="bold", color=PALETTE["ink"])
        canopy_ax.set_title("Canopy", fontsize=6.8, pad=2.5, fontweight="bold", color=PALETTE["ink"])

    if col_index == 0:
        thermal = "cool LST Q1" if row["thermal_group"] == "cool" else "hot LST Q4"
        ax.text(
            -0.08,
            0.50,
            f"LCZ {klass}\n{thermal}",
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": CLASS_COLORS[klass], "linewidth": 0.85, "alpha": 0.98, "pad": 3.0},
        )


def build_texture_dsm_canopy_atlas(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    out_dir = root / "outputs/diagnostics/supplementary_texture_lst_canopy_atlas_2026-08-03"
    fig_dir = root / "paper/figures/supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    selected = _select_atlas_tiles(df)
    selected.to_csv(out_dir / "fig_s1_texture_lst_canopy_atlas_selected_tiles.csv", index=False)
    rasters, limits = _collect_rasters(root, selected, grid)

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15.5, 9.8), constrained_layout=False)
    gs = fig.add_gridspec(8, 4, left=0.040, right=0.995, top=0.930, bottom=0.040, wspace=0.060, hspace=0.045)

    grid_axes = {}
    for r in range(8):
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            grid_axes[(r, c)] = ax
            row = selected[(selected["row_index"] == r) & (selected["col_index"] == c)].iloc[0]
            raster = rasters[str(row["grid_id"])]
            _plot_merged_tile(ax, root, row, raster, limits, r, c)
            if r == 0:
                title, subtitle = COLUMNS[c]
                ax.text(
                    0.5,
                    1.20,
                    f"{title}\n{subtitle}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color=PALETTE["ink"],
                    linespacing=1.25,
                    bbox={
                        "facecolor": "#f2efe9",
                        "edgecolor": "#8c867e",
                        "linewidth": 0.85,
                        "linestyle": (0, (4, 3)),
                        "boxstyle": "round,pad=0.45",
                        "alpha": 0.98,
                    },
                )

    # Add vertical dashed lines between the 4 major 3-panel prototype columns
    for c in range(3):
        ax_left = grid_axes[(0, c)]
        ax_right = grid_axes[(0, c + 1)]
        b_left = ax_left.get_position()
        b_right = ax_right.get_position()
        x_mid = (b_left.x1 + b_right.x0) / 2.0
        line = mpl.lines.Line2D(
            [x_mid, x_mid],
            [0.038, 0.942],
            transform=fig.transFigure,
            linestyle=(0, (6, 4)),
            linewidth=1.0,
            color="#9e988f",
            alpha=0.85,
        )
        fig.add_artist(line)

    lst_norm = mpl.colors.Normalize(vmin=limits["lst_p02"], vmax=limits["lst_p98"])
    canopy_norm = mpl.colors.Normalize(vmin=limits["canopy_p02"], vmax=limits["canopy_p98"])
    cax1 = fig.add_axes([0.22, 0.014, 0.24, 0.011])
    cax2 = fig.add_axes([0.54, 0.014, 0.24, 0.011])
    cb1 = fig.colorbar(mpl.cm.ScalarMappable(norm=lst_norm, cmap=LST_CMAP), cax=cax1, orientation="horizontal")
    cb2 = fig.colorbar(mpl.cm.ScalarMappable(norm=canopy_norm, cmap=CANOPY_CMAP), cax=cax2, orientation="horizontal")
    
    cb1.ax.xaxis.set_ticks_position("bottom")
    cb1.ax.xaxis.set_label_position("top")
    cb2.ax.xaxis.set_ticks_position("bottom")
    cb2.ax.xaxis.set_label_position("top")
    
    cb1.set_label("Land-surface temperature LST (°C), common p02-p98 scale", fontsize=7.8, fontweight="bold", labelpad=3)
    cb2.set_label("Canopy height (m), common p02-p98 scale", fontsize=7.8, fontweight="bold", labelpad=3)
    cb1.ax.tick_params(labelsize=6.8)
    cb2.ax.tick_params(labelsize=6.8)

    png = fig_dir / "fig_s1_texture_lst_canopy_atlas_32_cells.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    qa = {
        "figure": "fig_s1_texture_lst_canopy_atlas_32_cells",
        "date": "2026-08-05",
        "panel_count": 32,
        "layout": "4 columns x 8 rows, 3 raster/vector subpanels (texture, LST, canopy) per selected cell",
        "row_rule": "For each LCZ 3/6/8/9, rows are within-class LST Q1 and Q4 among analysis candidates.",
        "column_rule": "Four deterministic prototypes: median fabric, high coverage, high network density, and high green signal.",
        "raster_limits": limits,
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "selected_tiles_csv": str(out_dir / "fig_s1_texture_lst_canopy_atlas_selected_tiles.csv"),
        "png": str(png),
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/akilli_sehir_fua.gpkg",
            LST_PATH,
            CANOPY_PATH,
        ],
    }
    (out_dir / "fig_s1_texture_lst_canopy_atlas_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_s1_texture_lst_canopy_atlas_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_texture_dsm_canopy_atlas(Path(".")), indent=2))

