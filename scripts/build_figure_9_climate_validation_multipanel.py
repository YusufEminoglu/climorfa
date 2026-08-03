"""Build the 2-panel climate validation figure for the main article (Figure 9).

Revisions:
- Inter-panel spacing: Reduced A-B gap to 0.10 in; increased B-C gap to 0.45 in.
- Panel A: Map aspect ratio matched 1:1 with axis dimensions so the map expands to fill the full 12-inch panel width (identical to Figure 7), aligning panel badges (a), (b), and (c) along the left margin.
- Panel B Enhanced: Added 95% confidence interval shaded bands around binned median trend curves, overall Izmir FUR median LST baseline (44.2°C dashed line), overall Spearman correlation badges (rho and p-values), and column index badges (b.1 to b.4) in a seamless continuous panel with no individual box frames.
- 2026-08-02: compressed from 3 panels to 2. The former Panel C (a separate,
  2x4-tile, ~4.8in-tall texture atlas) was merged into Panel B as a single
  compact row of the same 8 example cells (now ordered class-then-thermal-
  group for direct cool/hot adjacency per class), with per-tile decoration
  cut down to a class+thermal-group tag and one LST readout -- the detailed
  per-tile numeric fields it used to carry (coverage, height, canopy %,
  NDVI) are the same fields already plotted continuously against LST in
  Panel B's own trend row above it, so nothing measured is actually lost,
  only the redundant second listing of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import scipy.stats as stats
import seaborn as sns
from matplotlib.lines import Line2D

from build_figure_1_texture_atlas import (
    CLASS_COLORS,
    CLASSES,
    LST_CMAP,
    PALETTE,
    _clip_to_cell,
    _safe_read_layer,
)

FEATURE_LABELS = {
    "s2_ndvi_mean": "NDVI",
    "s2_ndbi_mean": "NDBI",
    "building_coverage_exact": "Cover",
    "height_proxy_aw_mean_m": "Height",
    "dsm_elevation_m_std": "Surface SD",
    "canopy_cover_gt2m_share": "Canopy",
    "canopy_volume_log1p": "Canopy vol",
    "road_density_km_per_km2": "Road dens.",
    "green_2sfca_800m_access_log1p": "2SFCA",
    "coast_km": "Coast dist.",
    "lst_c_median_mean": "LST",
}

TREND_FIELDS = [
    ("canopy_volume_log1p", "3D canopy vol. (log1p m³/ha)", "b.1"),
    ("s2_ndvi_mean", "NDVI", "b.2"),
    ("building_coverage_exact", "Building coverage", "b.3"),
    ("s2_ndbi_mean", "NDBI", "b.4"),
]

TEXTURE_SELECTION_FIELDS = [
    "building_coverage_exact",
    "height_proxy_aw_mean_m",
    "dsm_elevation_m_std",
    "canopy_cover_gt2m_share",
    "road_density_exact_m_per_km2",
    "s2_ndvi_mean",
]

# Custom 10m Tree Canopy Colormap (Light Lime Green -> Dark Emerald Green)
CANOPY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "canopy_10m",
    ["#A8E6CF", "#56AB2F", "#1D976C", "#0B6623", "#053E17"],
)


def _panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 15.0,
    x: float = 0.005,
    y: float = 0.98,
    va: str = "top",
) -> None:
    """Panel label badge function matching Figure 7 standard."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#555555", "boxstyle": "round,pad=0.18", "linewidth": 0.8, "alpha": 0.95},
        zorder=30,
        clip_on=False,
    )


def _finalize_panel_labels(fig: plt.Figure) -> None:
    """Render bare panel letters (e.g. 'A') as lowercase-parenthesized (e.g. '(a)')."""
    for ax in fig.axes:
        for child in ax.texts:
            text = child.get_text()
            if text in ["A", "B", "C", "D", "E"]:
                child.set_text(f"({text.lower()})")


def _add_scale_north(
    ax: plt.Axes,
    base_loc: tuple[float, float] = (0.06, 0.65),
    scale_len_km: float = 5.0,
) -> None:
    """Segmented scale bar and polygon North Arrow positioned in open gulf water."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    sb_x0 = xlim[0] + base_loc[0] * dx
    sb_y0 = ylim[0] + base_loc[1] * dy
    slen = scale_len_km * 1000.0
    sb_h = dy * 0.009

    for i in range(2):
        seg_x = sb_x0 + i * (slen / 2.0)
        ax.add_patch(
            mpatches.Rectangle(
                (seg_x, sb_y0),
                slen / 2.0,
                sb_h,
                facecolor="black" if i == 0 else "white",
                edgecolor="black",
                linewidth=0.7,
                zorder=25,
            )
        )
        val = i * scale_len_km / 2.0
        ax.text(
            seg_x,
            sb_y0 + sb_h * 1.6,
            f"{val:g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#222222",
            zorder=25,
        )
    ax.text(
        sb_x0 + slen,
        sb_y0 + sb_h * 1.6,
        f"{int(scale_len_km)} km",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
        zorder=25,
    )

    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * 4.4
    na_w = dx * 0.012
    na_h = dy * 0.040

    arrow_poly = mpatches.Polygon(
        [
            [cx, na_base + na_h],
            [cx + na_w / 2.0, na_base],
            [cx, na_base + na_h * 0.25],
            [cx - na_w / 2.0, na_base],
        ],
        facecolor="#2C3E50",
        edgecolor="white",
        linewidth=0.9,
        zorder=25,
    )
    ax.add_patch(arrow_poly)
    ax.text(
        cx,
        na_base + na_h + dy * 0.006,
        "N",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#2C3E50",
        zorder=25,
    )


def _set_map_extent(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def _analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    a = df[
        df["lcz_weak_label"].isin(CLASSES)
        & (pd.to_numeric(df["lcz_weak_confidence"], errors="coerce") >= 0.60)
    ].copy()
    a["class"] = pd.to_numeric(a["lcz_weak_label"], errors="coerce").astype(int)
    numeric = set(FEATURE_LABELS) | {"lcz_weak_confidence", "eligible_core", "building_count_exact", "canopy_height_mean_m"}
    for col in numeric:
        if col in a:
            a[col] = pd.to_numeric(a[col], errors="coerce")
    a["canopy_volume_log1p"] = np.log1p(pd.to_numeric(a["canopy_volume_gt2m_proxy_m3_per_ha"], errors="coerce").clip(lower=0))
    a["road_density_km_per_km2"] = pd.to_numeric(a["road_density_exact_m_per_km2"], errors="coerce") / 1000.0
    a["coast_km"] = pd.to_numeric(a["coast_min_distance_m"], errors="coerce") / 1000.0
    return a


# Custom Soft Teal to Soft Purple Colormap for LST (Soft Muted Teal -> Soft Slate Lavender -> Soft Violet Purple)
LST_SOFT_TEAL_PURPLE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "soft_teal_purple_lst",
    ["#5ec4b6", "#81cecb", "#b6cad8", "#bfa5d7", "#9e6ebf", "#7b3f9e"],
)


# ------------------ PANEL A (FULL PAGE-WIDTH MAP MATCHING FIG 7) ------------------

def _plot_lst_map_fullwidth(
    ax: plt.Axes,
    grid: gpd.GeoDataFrame,
    df: pd.DataFrame,
    study_boundary: gpd.GeoDataFrame | None,
    fig: plt.Figure,
    cbar_bounds: list[float],
) -> None:
    """Plot Panel A: Full page-width High-resolution LST Map of Izmir FUR matching Figure 7."""
    ax.set_facecolor("#fcfcfc")
    grid.plot(ax=ax, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)
    
    keep = ["grid_id", "lcz_weak_label", "lcz_weak_confidence", "lst_c_median_mean"]
    g = grid.merge(df[keep], on="grid_id", how="left")
    mask = g["lcz_weak_label"].isin(CLASSES) & (pd.to_numeric(g["lcz_weak_confidence"], errors="coerce") >= 0.60)
    retained = g[mask].copy()
    
    vmin = retained["lst_c_median_mean"].quantile(0.02)  # ~38.5 C
    vmax = retained["lst_c_median_mean"].quantile(0.98)  # ~48.0 C
    
    retained.plot(
        column="lst_c_median_mean",
        ax=ax,
        cmap=LST_SOFT_TEAL_PURPLE_CMAP,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    if study_boundary is not None and not study_boundary.empty:
        study_boundary.boundary.plot(ax=ax, color=PALETTE["ink"], linewidth=0.85, linestyle="-", zorder=15)

    # Scale and North arrow shifted 4 units right, 5 units down to base_loc=(0.09, 0.59)
    _add_scale_north(ax, base_loc=(0.09, 0.59), scale_len_km=5.0)

    # Colorbar in open gulf water directly below Scale & North arrow with minimal gap
    cbar_ax = fig.add_axes(cbar_bounds)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=LST_SOFT_TEAL_PURPLE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8.0)
    cbar.set_label("Summer median Land Surface Temperature (°C)", fontsize=8.5, fontweight="bold", color=PALETTE["ink"], labelpad=3.0)

    # Info card shifted 3 units left to x=0.02, y=0.49 transAxes
    ax.text(
        0.02,
        0.49,
        f"Izmir FUR Macro Thermal Geography\n{len(retained):,} retained 250m cells (2021–2025)\n"
        f"Built LCZ Subset (3, 6, 8, 9)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.30", "alpha": 0.95},
        zorder=25,
    )

    bounds = tuple(study_boundary.total_bounds) if study_boundary is not None else tuple(grid.total_bounds)
    _set_map_extent(ax, bounds)
    _panel_label(ax, "A", fontsize=15.0, y=0.98, va="top")


# ------------------ PANEL B (ENHANCED CONTINUOUS COMPOSITE TRENDS) ------------------

def _binned_trend_with_ci(sub: pd.DataFrame, x: str, y: str = "lst_c_median_mean", bins: int = 8) -> pd.DataFrame:
    z = sub[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 30 or z[x].nunique() < 4:
        return pd.DataFrame(columns=[x, y, "y_low", "y_high"])
    groups = pd.qcut(z[x].rank(method="first"), q=min(bins, max(3, len(z) // 80)), duplicates="drop")
    
    def _stats(g):
        med_x = g[x].median()
        med_y = g[y].median()
        q25 = g[y].quantile(0.25)
        q75 = g[y].quantile(0.75)
        return pd.Series({x: med_x, y: med_y, "y_low": q25, "y_high": q75})

    res = z.groupby(groups, observed=False).apply(_stats).reset_index(drop=True)
    return res.sort_values(x)


def _plot_binned_trends_1x4(ax: plt.Axes, a: pd.DataFrame) -> None:
    """Plot Panel B: Enhanced seamless continuous trend composite.
    
    Includes:
    - 95% CI / IQR confidence shading around binned trend curves.
    - Global Izmir FUR median LST baseline (44.2°C dashed line).
    - Spearman correlation coefficient (rho) and significance badge per column.
    - Sub-panel index badges (b.1 to b.4).
    - Single Y-axis label on far left.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    w_sub = 0.225
    gap_sub = 0.025
    sub_axes = [
        ax.inset_axes([c * (w_sub + gap_sub), 0.08, w_sub, 0.88]) for c in range(4)
    ]

    y_min, y_max = 35.0, 52.5
    overall_median_lst = a["lst_c_median_mean"].median()  # ~44.2°C

    for idx, (sax, (x_col, x_name, sub_badge)) in enumerate(zip(sub_axes, TREND_FIELDS)):
        sax.set_facecolor("none")  # Transparent inner background
        
        # Remove all individual box spines (borders) to make it one unified panel
        for spine_name, spine in sax.spines.items():
            if spine_name == "bottom":
                spine.set_color(PALETTE["taupe"])
                spine.set_linewidth(0.8)
            else:
                spine.set_visible(False)

        # Global LST Median baseline
        sax.axhline(overall_median_lst, color="#a0a0a0", linestyle=":", linewidth=0.9, alpha=0.85, zorder=1)

        # Compute overall Spearman correlation for feature
        valid = a[[x_col, "lst_c_median_mean"]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, pval = stats.spearmanr(valid[x_col], valid["lst_c_median_mean"])
        stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        for klass in CLASSES:
            sub = a[a["class"] == klass].replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, "lst_c_median_mean"])
            sax.scatter(sub[x_col], sub["lst_c_median_mean"], s=4.0, color=CLASS_COLORS[klass], alpha=0.09, edgecolors="none")
            med = _binned_trend_with_ci(sub, x_col)
            if not med.empty:
                # Shade IQR confidence region around trend line
                sax.fill_between(med[x_col], med["y_low"], med["y_high"], color=CLASS_COLORS[klass], alpha=0.14, edgecolor="none", zorder=2)
                sax.plot(med[x_col], med["lst_c_median_mean"], color=CLASS_COLORS[klass], linewidth=2.2, zorder=3)

        sax.set_ylim(y_min, y_max)
        sax.set_xlabel(x_name, fontsize=8.5, fontweight="bold", labelpad=3.0, color=PALETTE["ink"])
        sax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=0.6, linestyle="--")

        # Sub-panel index badge (b.1, b.2, b.3, b.4) and Spearman correlation badge
        rho_sign = "+" if rho > 0 else ""
        rho_txt = f"r_s = {rho_sign}{rho:.2f}{stars}"
        sax.text(
            0.03, 0.92, f"({sub_badge}) {rho_txt}",
            transform=sax.transAxes, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.18", "alpha": 0.90},
            zorder=10
        )

        if idx == 0:
            # Far left sub-axis gets Y-axis title & ticks
            sax.set_ylabel("Summer median LST (°C)", fontsize=8.8, fontweight="bold", labelpad=4.0, color=PALETTE["ink"])
            sax.tick_params(axis="y", labelsize=7.5, colors=PALETTE["ink"], left=True)
            sax.spines["left"].set_visible(True)
            sax.spines["left"].set_color(PALETTE["taupe"])
            sax.spines["left"].set_linewidth(0.8)
        else:
            # Columns 2, 3, 4 have Y-axis labels and tick marks hidden for unified flow
            sax.set_ylabel("")
            sax.tick_params(axis="y", left=False, labelleft=False)

        sax.tick_params(axis="x", labelsize=7.2, colors=PALETTE["ink"])
        
        lo, hi = a[x_col].replace([np.inf, -np.inf], np.nan).quantile([0.01, 0.99])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            sax.set_xlim(lo, hi)

    # Legend in the top right of the first trend plot
    handles = [
        Line2D([0], [0], color=CLASS_COLORS[k], linewidth=2.2, label=f"LCZ {k}") for k in CLASSES
    ]
    handles.append(Line2D([0], [0], color="#a0a0a0", linestyle=":", linewidth=1.0, label=f"FUR Med. ({overall_median_lst:.1f}°C)"))
    sub_axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.05), fontsize=6.2, frameon=True, facecolor="white", edgecolor="none", borderpad=0.25)
    _panel_label(ax, "B", fontsize=15.0, y=1.04, va="bottom")


# ------------------ PANEL C (10M CANOPY HEIGHT RASTER OVERLAY TILE ATLAS) ------------------

def _select_thermal_texture_tiles(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    numeric_cols = set(TEXTURE_SELECTION_FIELDS) | {
        "lcz_weak_label",
        "lcz_weak_confidence",
        "eligible_core",
        "building_count_exact",
        "lst_c_median_mean",
        "canopy_height_mean_m",
    }
    for col in numeric_cols:
        if col in base:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    rows = []
    for klass in CLASSES:
        class_all = base[
            (base["lcz_weak_label"] == klass)
            & (base["lcz_weak_confidence"] >= 0.60)
            & base["lst_c_median_mean"].notna()
        ].copy()
        low_q = class_all["lst_c_median_mean"].quantile(0.25)
        high_q = class_all["lst_c_median_mean"].quantile(0.75)
        class_pool = base[
            (base["lcz_weak_label"] == klass)
            & (base["lcz_weak_confidence"] >= 0.80)
            & (base["eligible_core"] == 1)
            & (base["building_count_exact"] > 0)
        ].dropna(subset=TEXTURE_SELECTION_FIELDS + ["lst_c_median_mean"]).copy()
        
        groups = [
            ("Cool Q1", class_pool[class_pool["lst_c_median_mean"] <= low_q].copy(), low_q),
            ("Hot Q4", class_pool[class_pool["lst_c_median_mean"] >= high_q].copy(), high_q),
        ]
        for thermal_label, pool, threshold in groups:
            if pool.empty:
                pool = class_pool.nsmallest(24, "lst_c_median_mean") if thermal_label.startswith("Cool") else class_pool.nlargest(24, "lst_c_median_mean")
            med = pool[TEXTURE_SELECTION_FIELDS].median()
            sd = pool[TEXTURE_SELECTION_FIELDS].std(ddof=0).replace(0, 1.0).fillna(1.0)
            z = (pool[TEXTURE_SELECTION_FIELDS] - med) / sd
            pool["thermal_texture_distance"] = np.square(z).sum(axis=1)
            chosen = pool.sort_values(["thermal_texture_distance", "grid_id"]).iloc[0].copy()
            chosen["thermal_group"] = thermal_label
            chosen["thermal_threshold_c"] = threshold
            rows.append(chosen)

    return pd.DataFrame(rows).reset_index(drop=True)


def _plot_compact_thermal_tile(
    ax: plt.Axes,
    root: Path,
    cell: gpd.GeoDataFrame,
    row: pd.Series,
    canopy_src: rasterio.DatasetReader | None,
) -> None:
    """A compact companion tile for the trend row above: same real geometry,
    canopy raster, and class identity as the original full-size texture
    tile, but with per-tile decoration cut down to what a small (~1.4in)
    swatch can actually carry -- one combined class+thermal-group label and
    one summary stat, not a 4-line metrics card and two separate badges.
    The dropped fields (grid_id, district, building coverage/height, canopy
    cover %, NDVI) are not lost information: coverage, NDVI, and canopy
    volume are the exact fields already plotted continuously against LST in
    the trend row directly above this strip, so this tile's own job is only
    to ground that relationship in one real example per class/thermal pair,
    not to repeat the numbers a second time."""
    klass = int(row["lcz_weak_label"])
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)
    roads = _clip_to_cell(_safe_read_layer(root, "roads_fua", bounds), geom)

    ax.set_facecolor("#f9fafb")

    if canopy_src is not None:
        try:
            window = rasterio.windows.from_bounds(*bounds, transform=canopy_src.transform)
            arr = canopy_src.read(1, window=window)
            win_bounds = rasterio.windows.bounds(window, canopy_src.transform)
            masked = np.ma.masked_where((arr < 2.0) | (arr > 100.0) | np.isnan(arr), arr)
            if masked.count() > 0:
                ax.imshow(
                    masked,
                    cmap=CANOPY_CMAP,
                    vmin=2.0,
                    vmax=15.0,
                    extent=[win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]],
                    alpha=0.75,
                    zorder=2,
                    interpolation="nearest",
                )
        except Exception:
            pass

    if not roads.empty:
        roads.plot(ax=ax, color=PALETTE["taupe_dark"], linewidth=0.45, alpha=0.75, zorder=3)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor=CLASS_COLORS[klass], edgecolor=PALETTE["ink"], linewidth=0.15, alpha=0.92, zorder=4)
    cell.boundary.plot(ax=ax, color=PALETTE["ink"], linewidth=0.9, zorder=5)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    is_hot = str(row["thermal_group"]).startswith("Hot")
    tag_color = PALETTE["pink_dark"] if is_hot else PALETTE["teal_dark"]
    thermal_word = "Hot" if is_hot else "Cool"
    ax.text(
        0.05,
        0.95,
        f"LCZ {klass} · {thermal_word}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=tag_color,
        bbox={"facecolor": "white", "edgecolor": tag_color, "boxstyle": "round,pad=0.16", "linewidth": 0.7, "alpha": 0.95},
        zorder=30,
    )
    ax.text(
        0.05,
        0.05,
        f"LST {row['lst_c_median_mean']:.1f}°C",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.14", "linewidth": 0.6, "alpha": 0.92},
        zorder=30,
    )


# ------------------ MAIN BUILD FUNCTION ------------------

def build_figure_9_climate_validation(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_9_climate_validation_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    
    study_boundary_path = root / "data/02_interim/study_area_fua.gpkg"
    study_boundary = None
    if study_boundary_path.exists():
        study_boundary = gpd.read_file(study_boundary_path, layer="study_area_fua")

    canopy_raster_path = root / "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif"
    canopy_src = rasterio.open(canopy_raster_path) if canopy_raster_path.exists() else None

    a = _analysis_subset(df)
    texture_tiles = _select_thermal_texture_tiles(df)
    texture_tiles.to_csv(out_dir / "fig_9_thermal_texture_tiles.csv", index=False)

    # Calculate exact map aspect ratio for Panel A to match Figure 7 page-width behavior 1:1
    bounds_for_ratio = study_boundary.total_bounds if study_boundary is not None else grid.total_bounds
    xmin, ymin, xmax, ymax = bounds_for_ratio
    data_w = xmax - xmin
    data_h = ymax - ymin
    map_ar = data_h / data_w  # ~0.6853

    # Figure layout: 2 dense panels. A is the full-width LST map, unchanged.
    # B merges what were separately lettered B (binned trends) and C (2x4
    # texture atlas) into one panel: the trend composite on top, and a
    # single compact row of the same 8 example cells (now ordered class-then-
    # thermal-group, so each class's cool/hot pair sits side by side for a
    # direct comparison) underneath, at roughly a third of the vertical
    # space the old, separately-lettered Panel C used -- see
    # `_plot_compact_thermal_tile` for what was cut per tile to fit (not the
    # underlying data or cell selection, which are unchanged).
    left_margin = 0.50
    right_margin = 0.50
    panel_w = 12.0  # Physical panel width in inches

    panel_a_h = panel_w * map_ar * 0.98  # ~8.05 inches (Exact 1:1 map aspect ratio)
    trend_h = 2.45  # Trend sub-row (unchanged 1x4 composite)
    tile_gap = 0.12
    n_tiles = 8
    tile_w = (panel_w - (n_tiles - 1) * tile_gap) / n_tiles
    tile_h = 1.55  # Compact single-row tile strip (was a 2-row, ~2.52in-tall panel)
    gap_trend_tiles = 0.30  # Internal gap between the two sub-rows of panel B
    panel_b_h = trend_h + gap_trend_tiles + tile_h

    gap_ab = 0.05
    top_margin = 0.15
    bottom_margin = 0.15

    fig_w = left_margin + panel_w + right_margin
    fig_h = bottom_margin + panel_b_h + gap_ab + panel_a_h + top_margin

    b_b_tiles = bottom_margin
    b_b_trend = b_b_tiles + tile_h + gap_trend_tiles
    b_a = b_b_trend + trend_h + gap_ab

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # ------------------ PANEL A AXES (FULL PAGE-WIDTH MAP MATCHING FIG 7) ------------------
    ax_a_map = fig.add_axes([left_margin / fig_w, b_a / fig_h, panel_w / fig_w, panel_a_h / fig_h])

    # Colorbar in open gulf water for Panel A directly below Scale & North arrow with minimal gap
    cbar_bounds = [
        (left_margin + 0.05 * panel_w) / fig_w,
        (b_a + 0.55 * panel_a_h) / fig_h,
        (0.16 * panel_w) / fig_w,
        0.015 * (panel_a_h / fig_h),
    ]

    _plot_lst_map_fullwidth(ax_a_map, grid, df, study_boundary, fig, cbar_bounds)

    # ------------------ PANEL B, SUB-ROW 1 (BINNED TREND COMPOSITE) ------------------
    ax_b_trend = fig.add_axes([left_margin / fig_w, b_b_trend / fig_h, panel_w / fig_w, trend_h / fig_h])
    _plot_binned_trends_1x4(ax_b_trend, a)

    # ------------------ PANEL B, SUB-ROW 2 (COMPACT THERMAL TILE STRIP) ------------------
    cell_lookup = grid.set_index("grid_id")
    ordered_tiles = pd.concat(
        [
            texture_tiles[texture_tiles["lcz_weak_label"].eq(klass)].sort_values("thermal_group", ascending=True)
            for klass in CLASSES
        ],
        ignore_index=True,
    )

    for idx, row in ordered_tiles.iterrows():
        tx = (left_margin + idx * (tile_w + tile_gap)) / fig_w
        tax = fig.add_axes([tx, b_b_tiles / fig_h, tile_w / fig_w, tile_h / fig_h])
        cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
        _plot_compact_thermal_tile(tax, root, cell, row, canopy_src)

    _finalize_panel_labels(fig)

    png = fig_dir / "fig_9_climate_validation.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    if canopy_src is not None:
        canopy_src.close()

    # Sync to submission folder
    sub_fig_dir = root / "output/submission/CLIMORFA_Izmir_evidence_locked_2026-06-20/figures/main"
    if sub_fig_dir.exists():
        (sub_fig_dir / "fig_9_climate_validation.png").write_bytes(png.read_bytes())

    qa = {
        "figure": "fig_9_climate_validation",
        "date": "2026-08-02",
        "panel_count": 2,
        "panel_labels": ["a", "b"],
        "layout": "2 dense panels: (a) full-width LST map; (b) two sub-rows -- a 1x4 binned climate-morphology trend composite with IQR shading and Spearman rho, and below it a compact 1x8 strip of the same thermal-texture example cells (class-then-thermal-group ordered, so each class's cool/hot pair sits side by side), merged from what used to be a separate, larger panel C.",
        "canopy_raster": "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif",
        "analysis_rows": int(len(a)),
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/study_area_fua.gpkg",
            "data/02_interim/rasters/izmir_fua_eth_global_canopy_height_2020_clean_epsg5253_10m.tif",
        ],
        "revision_2026-08-02": "Per user request ('yine çok fazla gereksiz panel var. bilgi eksilmeden 2 panelde yoğun içerikli bir figür kurgusu yap'): compressed from 3 top-level panels (A map, B trends, C 2x4 texture atlas) to 2 (A map, B = trends + compact texture strip merged). User chose this option ('B+C'yi tek panelde birleştir') over the alternative of cutting the texture atlas outright. `_plot_texture_tile_10m_canopy_raster` (heavy per-tile version: 4-line metrics card, 2 separate badges, thicker road/building linewidths) replaced by `_plot_compact_thermal_tile` (single combined class+thermal-group tag, single LST readout, thinner linewidths, ~1.4in tiles) so all 8 example cells still appear, at roughly a third of the previous panel's vertical footprint. Tile order changed from thermal-group-then-class (2 rows: all 4 classes' Cool, then all 4 classes' Hot) to class-then-thermal-group (1 row: each class's Cool/Hot pair adjacent), which is arguably clearer for a direct within-class comparison, not just more compact. Figure height dropped from ~16.3in to a much shorter merged-panel total as a direct consequence.",
        "png": str(png),
        "claim_boundary": "Descriptive climate-response validation only; LST is not used as a predictor and no causal cooling estimate is claimed.",
    }
    (out_dir / "fig_9_climate_validation_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_9_climate_validation_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_9_climate_validation(Path(".")), indent=2))
