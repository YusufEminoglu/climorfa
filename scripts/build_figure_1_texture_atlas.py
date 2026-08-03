"""Build the multi-panel Figure 1 study-area and texture atlas."""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CLASSES = [3, 6, 8, 9]
CLASS_NAMES = {
    3: "LCZ 3 compact low-rise",
    6: "LCZ 6 open low-rise",
    8: "LCZ 8 large low-rise",
    9: "LCZ 9 sparsely built",
}
CLASS_SHORT = {
    3: "LCZ 3",
    6: "LCZ 6",
    8: "LCZ 8",
    9: "LCZ 9",
}
PALETTE = {
    "teal": "#87D8CD",
    "pink": "#F978A5",
    "orange": "#F4AB55",
    "taupe": "#ACA49C",
    "teal_dark": "#2F756E",
    "pink_dark": "#9E365F",
    "orange_dark": "#985D18",
    "taupe_dark": "#5F5A56",
    "teal_light": "#E7F7F5",
    "pink_light": "#FEE8EF",
    "orange_light": "#FDEEDC",
    "taupe_light": "#F0EEEC",
    "surface": "#FAF9F7",
    "grid": "#E8E4E0",
    "ink": "#252526",
    "muted": "#6E6A67",
}
CLASS_COLORS = {
    3: PALETTE["teal"],
    6: PALETTE["pink"],
    8: PALETTE["orange"],
    9: PALETTE["taupe"],
}
SELECTION_FEATURES = [
    "building_coverage_exact",
    "height_proxy_aw_mean_m",
    "dsm_elevation_m_std",
    "canopy_cover_gt2m_share",
    "road_density_exact_m_per_km2",
    "s2_ndvi_mean",
    "lst_c_median_mean",
]
TEXTURE_METRICS = [
    "grid_id",
    "district",
    "lcz_weak_label",
    "lcz_weak_confidence",
    "building_coverage_exact",
    "height_proxy_aw_mean_m",
    "dsm_elevation_m_std",
    "canopy_cover_gt2m_share",
    "road_density_exact_m_per_km2",
    "s2_ndvi_mean",
    "lst_c_median_mean",
    "median_distance",
]


def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


LST_CMAP = _cmap(
    "climorfa_lst",
    [PALETTE["taupe_light"], PALETTE["taupe"], PALETTE["orange"], PALETTE["pink"]],
)
CONF_CMAP = _cmap(
    "climorfa_confidence",
    [PALETTE["taupe_light"], PALETTE["taupe"], PALETTE["teal"]],
)


def _read_inputs(root: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    study = gpd.read_file(root / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    shoreline = gpd.read_file(root / "data/02_interim/shoreline_epsg5253.gpkg", layer="shoreline_epsg5253")
    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    pred = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv")
    return grid, study, shoreline, df, pred


def _select_texture_tiles(df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    base = df.copy()
    base["lcz_weak_label"] = pd.to_numeric(base["lcz_weak_label"], errors="coerce")
    for col in SELECTION_FEATURES:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    for klass in CLASSES:
        pool = base[
            (base["lcz_weak_label"] == klass)
            & (base["lcz_weak_confidence"] >= 0.80)
            & (base["eligible_core"] == 1)
            & (base["building_count_exact"] > 0)
        ].copy()
        pool = pool.dropna(subset=SELECTION_FEATURES)
        if pool.empty:
            raise RuntimeError(f"No eligible texture tile candidates for LCZ {klass}.")
        med = pool[SELECTION_FEATURES].median()
        sd = pool[SELECTION_FEATURES].std(ddof=0).replace(0, 1.0).fillna(1.0)
        z = (pool[SELECTION_FEATURES] - med) / sd
        pool["median_distance"] = np.square(z).sum(axis=1)
        selected.append(pool.sort_values(["median_distance", "grid_id"]).iloc[0])

    return pd.DataFrame(selected)[TEXTURE_METRICS].reset_index(drop=True)


def _panel_label(ax: plt.Axes, label: str, color: str = PALETTE["ink"]) -> None:
    ax.text(
        0.012,
        0.986,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={
            "facecolor": "white",
            "edgecolor": color,
            "boxstyle": "round,pad=0.16",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
        zorder=20,
    )


def _finalize_panel_labels(fig: plt.Figure) -> None:
    """Render bare panel letters (e.g. 'A') as lowercase-parenthesized (e.g. '(a)')."""
    for ax in fig.axes:
        for child in ax.texts:
            text = child.get_text()
            if re.fullmatch(r"[A-Za-z]", text):
                child.set_text(f"({text.lower()})")


def _set_map_extent(ax: plt.Axes, bounds: np.ndarray) -> None:
    xmin, ymin, xmax, ymax = bounds
    pad_x = (xmax - xmin) * 0.015
    pad_y = (ymax - ymin) * 0.015
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")


def _add_source_note(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.018,
        "Sources: data/03_processed/grid_250m_model_features_v8.csv; "
        "data/03_processed/analysis_grids.gpkg; local building/road vector layer; "
        "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["muted"],
    )


def _plot_study_frame(
    ax: plt.Axes,
    g: gpd.GeoDataFrame,
    study: gpd.GeoDataFrame,
    shoreline: gpd.GeoDataFrame,
) -> None:
    g.plot(ax=ax, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)
    core = g[g["eligible_core"] == 1]
    edge = g[g["eligible_core"] != 1]
    core.plot(ax=ax, color=PALETTE["teal_light"], edgecolor="none", linewidth=0)
    if not edge.empty:
        edge.plot(ax=ax, color=PALETTE["grid"], edgecolor="none", linewidth=0)
    study.boundary.plot(ax=ax, color=PALETTE["ink"], linewidth=0.8)
    shoreline.plot(ax=ax, color=PALETTE["teal_dark"], linewidth=0.55)
    _panel_label(ax, "A", PALETTE["teal"])
    ax.set_title("FUR analytical frame, 250 m grid, and coastline", fontsize=10, pad=4)
    _set_map_extent(ax, g.total_bounds)
    ax.text(
        0.025,
        0.03,
        f"{len(g):,} FUR-intersecting cells\n{len(core):,} eligible-core cells",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 3},
    )


def _plot_lcz_map(ax: plt.Axes, g: gpd.GeoDataFrame) -> None:
    g.plot(ax=ax, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)
    handles = []
    for klass in CLASSES:
        subset = g[(g["lcz_weak_label"] == klass) & (g["lcz_weak_confidence"] >= 0.60)]
        subset.plot(ax=ax, color=CLASS_COLORS[klass], edgecolor="none", linewidth=0)
        handles.append(Patch(facecolor=CLASS_COLORS[klass], edgecolor="none", label=f"{CLASS_SHORT[klass]}: {len(subset):,}"))
    _panel_label(ax, "B", PALETTE["pink"])
    ax.set_title("High-confidence weak LCZ morphology labels", fontsize=10, pad=4)
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=7, title="Cells", title_fontsize=8)
    _set_map_extent(ax, g.total_bounds)


def _plot_lst_map(ax: plt.Axes, g: gpd.GeoDataFrame) -> None:
    g.plot(
        column="lst_c_median_mean",
        ax=ax,
        cmap=LST_CMAP,
        edgecolor="none",
        linewidth=0,
        legend=True,
        legend_kwds={"label": "Summer median LST (C)", "shrink": 0.72, "pad": 0.01},
    )
    _panel_label(ax, "C", PALETTE["orange"])
    ax.set_title("Landsat 8/9 summer 2021-2025 thermal response", fontsize=10, pad=4)
    _set_map_extent(ax, g.total_bounds)


def _plot_confidence_map(ax: plt.Axes, g: gpd.GeoDataFrame) -> None:
    g.plot(ax=ax, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)
    scored = g.dropna(subset=["predicted_probability"])
    scored.plot(
        column="predicted_probability",
        ax=ax,
        cmap=CONF_CMAP,
        vmin=0.25,
        vmax=1.0,
        edgecolor="none",
        linewidth=0,
        legend=True,
        legend_kwds={"label": "OOF maximum probability", "shrink": 0.72, "pad": 0.01},
    )
    low = scored[scored["predicted_probability"] < 0.55]
    if not low.empty:
        low.boundary.plot(ax=ax, color=PALETTE["pink_dark"], linewidth=0.12, alpha=0.75)
    _panel_label(ax, "D", PALETTE["teal"])
    ax.set_title("Spatial LightGBM out-of-fold confidence", fontsize=10, pad=4)
    _set_map_extent(ax, g.total_bounds)
    ax.text(
        0.025,
        0.03,
        f"Scored cells: {len(scored):,}\nLow confidence <0.55: {len(low):,}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 3},
    )


def _safe_read_layer(root: Path, layer: str, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    path = root / "data/02_interim/akilli_sehir_fua.gpkg"
    return gpd.read_file(path, layer=layer, bbox=bbox)


def _clip_to_cell(features: gpd.GeoDataFrame, cell_geom) -> gpd.GeoDataFrame:
    if features.empty:
        return features
    features = features[features.intersects(cell_geom)].copy()
    if features.empty:
        return features
    features["geometry"] = features.geometry.intersection(cell_geom)
    return features[~features.geometry.is_empty]


def _plot_texture_tile(
    ax: plt.Axes,
    root: Path,
    cell: gpd.GeoDataFrame,
    row: pd.Series,
    panel: str,
) -> None:
    klass = int(row["lcz_weak_label"])
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)
    roads = _clip_to_cell(_safe_read_layer(root, "roads_fua", bounds), geom)

    ax.set_facecolor(PALETTE["surface"])
    if not roads.empty:
        roads.plot(ax=ax, color=PALETTE["taupe_dark"], linewidth=0.7, alpha=0.82, zorder=2)
    if not buildings.empty:
        buildings.plot(
            ax=ax,
            facecolor=CLASS_COLORS[klass],
            edgecolor=PALETTE["ink"],
            linewidth=0.16,
            alpha=0.92,
            zorder=3,
        )
    cell.boundary.plot(ax=ax, color=PALETTE["ink"], linewidth=1.15, zorder=4)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    _panel_label(ax, panel, CLASS_COLORS[klass])
    ax.set_title(CLASS_NAMES[klass], fontsize=9.5, pad=4)
    district = str(row["district"])
    if len(district) > 18:
        district = district[:17] + "."
    metrics = (
        f"{row['grid_id']} | {district}\n"
        f"conf {row['lcz_weak_confidence']:.2f}  cov {100 * row['building_coverage_exact']:.1f}%  "
        f"h {row['height_proxy_aw_mean_m']:.1f} m\n"
        f"Surf.sd {row['dsm_elevation_m_std']:.1f} m  NDVI {row['s2_ndvi_mean']:.2f}  "
        f"LST {row['lst_c_median_mean']:.1f} C"
    )
    ax.text(
        0.022,
        0.026,
        metrics,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": CLASS_COLORS[klass], "alpha": 0.93, "pad": 2.7},
        zorder=8,
    )


def build_figure_1(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    out_dir = root / "outputs/diagnostics/figure_1_texture_atlas_2026-06-20"
    fig_dir = root / "paper/figures/main"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid, study, shoreline, df, pred = _read_inputs(root)
    keep_cols = [
        "grid_id",
        "district",
        "lcz_weak_label",
        "lcz_weak_confidence",
        "lst_c_median_mean",
        "coastal_5km_flag",
    ]
    pred_800 = pred[
        (pred["model"] == "lightgbm")
        & (pred["recipe"] == "baseline_d_full_proxy_context")
    ][["grid_id", "predicted_probability", "predicted_label"]]
    g = grid.merge(df[keep_cols], on="grid_id", how="left").merge(pred_800, on="grid_id", how="left")
    tiles = _select_texture_tiles(df)
    tiles.to_csv(out_dir / "figure_1_selected_tiles.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
        }
    )
    fig = plt.figure(figsize=(15.8, 11.6), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.12, 1.12, 0.98],
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.065,
        hspace=0.16,
        wspace=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    tile_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]

    _plot_study_frame(ax_a, g, study, shoreline)
    _plot_lcz_map(ax_b, g)
    _plot_lst_map(ax_c, g)
    _plot_confidence_map(ax_d, g)

    cell_lookup = grid.set_index("grid_id")
    for ax, (_, row), panel in zip(tile_axes, tiles.iterrows(), ["E", "F", "G", "H"]):
        cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
        _plot_texture_tile(ax, root, cell, row, panel)

    _add_source_note(fig)
    _finalize_panel_labels(fig)

    png = fig_dir / "fig_1_study_area_texture_atlas.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    qa = {
        "figure": "fig_1_study_area_texture_atlas",
        "date": "2026-06-20",
        "panel_count": 8,
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "selection_rule": (
            "For each LCZ 3/6/8/9, choose the confidence >=0.80 eligible-core "
            "building-bearing cell with complete selection metrics and minimum "
            "squared standardized distance to the class median."
        ),
        "selected_tiles_csv": str(out_dir / "figure_1_selected_tiles.csv"),
        "png": str(png),
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/study_area_fua.gpkg",
            "data/02_interim/shoreline_epsg5253.gpkg",
            "data/02_interim/akilli_sehir_fua.gpkg",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv",
        ],
        "selected_grid_ids": tiles[["lcz_weak_label", "grid_id", "district"]].to_dict(orient="records"),
    }
    (out_dir / "figure_1_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "figure_1_qa.json")}


if __name__ == "__main__":
    result = build_figure_1(Path("."))
    print(json.dumps(result, indent=2))
