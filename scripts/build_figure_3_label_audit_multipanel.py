"""Build a multi-panel weak-label, transition-cell, and audit-frame figure."""
from __future__ import annotations

import json
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyproj import Transformer
from rasterio.windows import from_bounds as rio_window_from_bounds

from build_figure_1_texture_atlas import CLASS_COLORS, CLASSES, PALETTE, _clip_to_cell, _safe_read_layer
from build_figure_2_methodology_multipanel import _add_scale_north_above_legend, _finalize_panel_labels, _panel_label

LST_RASTER = "data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif"
LST_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "fig3_lst", [PALETTE["teal"], PALETTE["taupe_light"], PALETTE["orange"], PALETTE["pink_dark"]]
)
M_PER_FLOOR = 3.2  # same above-ground-floor height proxy used throughout the paper
HEIGHT_EXAGGERATION = 1.2  # visual-only vertical exaggeration for the 3D quadrant; the
# reported height in the metrics text is never multiplied by this.
_TO_WEBMERC = Transformer.from_crs("EPSG:5253", "EPSG:3857", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:5253", "EPSG:4326", always_xy=True)


def _set_map_extent(ax: plt.Axes, bounds: np.ndarray) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin - (xmax - xmin) * 0.015, xmax + (xmax - xmin) * 0.015)
    ax.set_ylim(ymin - (ymax - ymin) * 0.015, ymax + (ymax - ymin) * 0.015)
    ax.set_aspect("equal")
    ax.axis("off")


# One example district each, as requested: a central, recognizable district
# per panel rather than three arbitrary/peripheral cells.
EXAMPLE_DISTRICTS = {"confident": "KONAK", "transition": "BAYRAKLI", "pass1": "BORNOVA"}


def _select_example_cells(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Pick a near-certain cell, a near-threshold (mixed) cell, and a Pass-1
    audit-sample cell, one each from Konak, Bayrakli, and Bornova.

    All three are drawn from the same four-class, eligible-core,
    building-bearing population used throughout the diagnostic, and each
    district's pick is itself a deterministic rule (not a hand-picked
    grid_id), so the within-district selection stays reproducible.
    """
    metric_cols = [
        "lcz_weak_confidence",
        "building_coverage_exact",
        "height_proxy_aw_mean_m",
        "dsm_elevation_m_std",
        "s2_ndvi_mean",
        "lst_c_median_mean",
    ]
    base = df.copy()
    base["lcz_weak_label"] = pd.to_numeric(base["lcz_weak_label"], errors="coerce")
    for col in metric_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    pool = base[
        base["lcz_weak_label"].isin(CLASSES)
        & (base["eligible_core"] == 1)
        & (base["building_count_exact"] > 0)
    ].dropna(subset=metric_cols)

    confident_pool = pool[pool["district"] == EXAMPLE_DISTRICTS["confident"]]
    confident = confident_pool.sort_values(
        ["lcz_weak_confidence", "building_coverage_exact", "grid_id"], ascending=[False, False, True]
    ).iloc[0]

    # Among cells split closest to an even 50/50 mix, prefer the one with the
    # most building coverage so the tile actually shows legible mixed texture
    # rather than a near-empty cell.
    mixed_pool = pool[(pool["district"] == EXAMPLE_DISTRICTS["transition"]) & (pool["lcz_mixed_flag"] == 1)].copy()
    mixed_pool["balance_distance"] = (mixed_pool["lcz_weak_confidence"] - 0.50).abs()
    transition = mixed_pool.sort_values(
        ["balance_distance", "building_coverage_exact", "grid_id"], ascending=[True, False, True]
    ).iloc[0]

    pass1_ids = set(audit.loc[audit["audit_priority_pass"] == "pass_1_first_150", "grid_id"])
    pass1_pool = pool[(pool["district"] == EXAMPLE_DISTRICTS["pass1"]) & (pool["grid_id"].isin(pass1_ids))]
    pass1 = pass1_pool.sort_values(["building_coverage_exact", "grid_id"], ascending=[False, True]).iloc[0]

    cols = ["grid_id", "district", "lcz_weak_label"] + metric_cols
    return pd.DataFrame([confident, transition, pass1])[cols].reset_index(drop=True)


def _plot_plan_subview(
    ax: plt.Axes,
    root: Path,
    cell: gpd.GeoDataFrame,
    row: pd.Series,
    panel: str,
    extra_note: str | None,
    fill_color: str,
) -> None:
    """Top-left quadrant: the original plan view, tinted by the panel's own
    identity color (audit-pass color for the Pass-1 example, a dedicated
    call-out color otherwise), carrying the panel label and metrics text."""
    klass = int(row["lcz_weak_label"])
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)
    roads = _clip_to_cell(_safe_read_layer(root, "roads_fua", bounds), geom)

    ax.set_facecolor(PALETTE["surface"])
    if not roads.empty:
        roads.plot(ax=ax, color=PALETTE["taupe_dark"], linewidth=0.7, alpha=0.82, zorder=2)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor=fill_color, edgecolor=PALETTE["ink"], linewidth=0.16, alpha=0.6, zorder=3)
    cell.boundary.plot(ax=ax, color=fill_color, linewidth=1.6, zorder=4)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    _panel_label(ax, panel, y=1.045)
    district = str(row["district"])
    if len(district) > 18:
        district = district[:17] + "."
    metrics = (
        f"{row['grid_id']} | {district}\n"
        f"weak LCZ {klass}  conf {row['lcz_weak_confidence']:.2f}  cov {100 * row['building_coverage_exact']:.1f}%\n"
        f"h {row['height_proxy_aw_mean_m']:.1f} m  NDVI {row['s2_ndvi_mean']:.2f}  LST {row['lst_c_median_mean']:.1f} C"
    )
    if extra_note:
        metrics += f"\n{extra_note}"
    ax.text(
        0.03,
        0.03,
        metrics,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": fill_color, "alpha": 0.93, "pad": 2.0},
        zorder=8,
    )


def _plot_lst_subview(ax: plt.Axes, root: Path, bounds: tuple[float, float, float, float]) -> None:
    """Top-right quadrant: Landsat summer LST for this exact cell."""
    with rasterio.open(root / LST_RASTER) as src:
        window = rio_window_from_bounds(*bounds, transform=src.transform)
        arr = src.read(1, window=window, boundless=True)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    ax.imshow(arr, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), origin="upper", cmap=LST_CMAP, interpolation="nearest")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["ink"])
        spine.set_linewidth(0.8)
    ax.text(0.03, 0.97, "LST", transform=ax.transAxes, ha="left", va="top", fontsize=6.0, fontweight="bold",
            color=PALETTE["ink"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.4}, zorder=8)


def _plot_basemap_subview(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Bottom-left quadrant: Esri World Imagery satellite basemap, cropped
    exactly to the cell (contextily returns whole map tiles, which cover a
    larger area than the 250 m cell, so the view is clipped after fetch)."""
    xmin_ll, ymin_ll = _TO_WGS84.transform(bounds[0], bounds[1])
    xmax_ll, ymax_ll = _TO_WGS84.transform(bounds[2], bounds[3])
    img, ext = cx.bounds2img(xmin_ll, ymin_ll, xmax_ll, ymax_ll, ll=True, source=cx.providers.Esri.WorldImagery, zoom=19)
    ax.imshow(img, extent=ext, origin="upper")
    xmin_m, ymin_m = _TO_WEBMERC.transform(bounds[0], bounds[1])
    xmax_m, ymax_m = _TO_WEBMERC.transform(bounds[2], bounds[3])
    ax.set_xlim(xmin_m, xmax_m)
    ax.set_ylim(ymin_m, ymax_m)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["ink"])
        spine.set_linewidth(0.8)
    ax.text(0.03, 0.97, "Esri World Imagery", transform=ax.transAxes, ha="left", va="top", fontsize=5.0,
            color="white", bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 1.4}, zorder=8)


def _plot_3d_subview(ax3d: plt.Axes, root: Path, cell: gpd.GeoDataFrame, bounds: tuple[float, float, float, float], roof_color: str) -> None:
    """Bottom-right quadrant: a real building-massing extrusion (matplotlib
    3D, not the osm_quick_3d QGIS plugin, which needs a live QGIS Desktop 3D
    canvas and cannot be driven headlessly). Height per building is
    ZEMINUSTUK (above-ground floor count) times 3.2 m, the same height proxy
    used everywhere else in the paper. Coordinates are shifted to the cell's
    own local origin (0-250 m) to avoid float overflow in mplot3d's internal
    margin math at raw EPSG:5253 magnitudes."""
    geom = cell.geometry.iloc[0]
    x0, y0 = bounds[0], bounds[1]
    buildings = _clip_to_cell(_safe_read_layer(root, "buildings_fua", bounds), geom)

    # Per-face RGBA so ground/roof/walls can carry different alphas at once;
    # Poly3DCollection's own `alpha` kwarg would override these if set, so it
    # is deliberately left unset. Walls are tinted the panel's own identity
    # color (not white, which was invisible against the white background)
    # and kept semi-transparent so the massing reads as glass-like rather
    # than solid blocks; roofs stay closer to opaque so the color still reads
    # clearly from above.
    ground_rgba = mpl.colors.to_rgba(PALETTE["surface"], 1.0)
    roof_rgba = mpl.colors.to_rgba(roof_color, 0.88)
    wall_rgba = mpl.colors.to_rgba(roof_color, 0.35)

    # The ground plane is kept in its own Poly3DCollection, separate from
    # the buildings: mplot3d sorts faces for draw order by each face's own
    # average view-depth rather than true z-buffering, and a single huge
    # flat face spanning the whole cell can sort ahead of (and silently
    # hide) smaller building faces depending on where they sit relative to
    # the camera -- this was leaving real gaps of bare white ground where
    # the plan-view quadrant clearly shows buildings. `computed_zorder =
    # False` disables that automatic depth sort in favor of a fixed,
    # explicit draw order (ground first, buildings on top), which is the
    # standard workaround for this known mplot3d limitation.
    ax3d.computed_zorder = False

    ground_faces: list[list[tuple[float, float, float]]] = [[(0, 0, 0), (bounds[2] - x0, 0, 0), (bounds[2] - x0, bounds[3] - y0, 0), (0, bounds[3] - y0, 0)]]
    ground_colors: list[tuple[float, float, float, float]] = [ground_rgba]

    all_faces: list[list[tuple[float, float, float]]] = []
    all_colors: list[tuple[float, float, float, float]] = []
    w, h = bounds[2] - x0, bounds[3] - y0

    max_h = M_PER_FLOOR
    for _, b in buildings.iterrows():
        floors = b.get("ZEMINUSTUK", 1)
        floors = 1 if floors is None or (isinstance(floors, float) and np.isnan(floors)) else floors
        height = max(float(floors), 1.0) * M_PER_FLOOR * HEIGHT_EXAGGERATION
        max_h = max(max_h, height)
        poly = b.geometry
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for p in polys:
            xs, ys = p.exterior.coords.xy
            xs = [x - x0 for x in xs]
            ys = [y - y0 for y in ys]
            n = len(xs) - 1
            if n < 3:
                continue
            all_faces.append([(xs[i], ys[i], height) for i in range(n)])
            all_colors.append(roof_rgba)
            for i in range(n):
                j = (i + 1) % n
                all_faces.append([(xs[i], ys[i], 0), (xs[j], ys[j], 0), (xs[j], ys[j], height), (xs[i], ys[i], height)])
                all_colors.append(wall_rgba)

    ground_coll = Poly3DCollection(ground_faces, facecolor=ground_colors, edgecolor="none")
    ground_coll.set_zorder(1)
    ax3d.add_collection3d(ground_coll)

    coll = Poly3DCollection(all_faces, facecolor=all_colors, edgecolor=PALETTE["ink"], linewidth=0.08)
    coll.set_zorder(2)
    ax3d.add_collection3d(coll)
    ax3d.set_xlim(0, w)
    ax3d.set_ylim(0, h)
    zmax = max(max_h * 1.15, 12.0)
    ax3d.set_zlim(0, zmax)
    # zoom=1.45 was pushing the tallest rooftops past the axes' own frame
    # (visibly clipped at the top edge in panel B); pulled back to 1.20 so
    # the full massing -- including the tallest building -- stays inside
    # the panel.
    ax3d.set_box_aspect((w, h, zmax), zoom=1.20)
    ax3d.view_init(elev=32, azim=-60)
    ax3d.set_axis_off()

    # Frame to match the LST/basemap quadrants' borders: Axes3D has no 2D
    # spines, so draw a plain rectangle around its actual figure position.
    pos = ax3d.get_position()
    frame = Rectangle(
        (pos.x0, pos.y0), pos.width, pos.height,
        transform=ax3d.figure.transFigure, fill=False,
        edgecolor=PALETTE["ink"], linewidth=0.8, zorder=30,
    )
    ax3d.figure.add_artist(frame)


def _plot_example_supercell(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    root: Path,
    cell: gpd.GeoDataFrame,
    row: pd.Series,
    panel: str,
    identity_color: str,
    extra_note: str | None = None,
) -> None:
    """Fill one super-panel's 2x2 quadrants: plan (own color), LST, basemap, 3D."""
    geom = cell.geometry.iloc[0]
    bounds = tuple(geom.bounds)
    _plot_plan_subview(axes["plan"], root, cell, row, panel, extra_note, identity_color)
    _plot_lst_subview(axes["lst"], root, bounds)
    _plot_basemap_subview(axes["basemap"], bounds)
    _plot_3d_subview(axes["3d"], root, cell, bounds, identity_color)


def _add_group_callout(
    fig: plt.Figure,
    ax_map: plt.Axes,
    xy: tuple[float, float],
    group_box_fig: tuple[float, float, float, float],
    color: str,
) -> None:
    """Mark a cell's location on the map panel, draw a dashed frame around the
    whole 2x2 super-panel that shows it (not just one quadrant), and a dashed
    leader line connecting the two, so (a)/(b)/(c) read as call-outs from (d)
    at the level of the whole group."""
    left, bottom, width, height = group_box_fig
    frame = Rectangle(
        (left, bottom),
        width,
        height,
        transform=fig.transFigure,
        fill=False,
        edgecolor=color,
        linewidth=2.0,
        linestyle="--",
        alpha=0.9,
        zorder=25,
    )
    fig.add_artist(frame)

    ax_map.scatter(
        [xy[0]], [xy[1]], s=110, marker="*", facecolor=color, edgecolor="white", linewidth=0.9, zorder=20
    )
    target = (left + width / 2, bottom)
    # Right-angle elbow (straight up from the star, then straight across into
    # the panel's bottom-mid), not a direct diagonal, with a sharp corner
    # (rad=0) and no arrowhead.
    con = ConnectionPatch(
        xyA=xy,
        coordsA=ax_map.transData,
        xyB=target,
        coordsB=fig.transFigure,
        color=color,
        linewidth=1.9,
        linestyle="--",
        alpha=0.85,
        zorder=25,
        arrowstyle="-",
        connectionstyle="angle,angleA=0,angleB=90,rad=0",
    )
    fig.add_artist(con)

    # Small round node marking the exact bottom-mid connection point.
    node = plt.Circle(
        target, radius=0.0065, transform=fig.transFigure, facecolor=color,
        edgecolor="white", linewidth=0.8, zorder=26,
    )
    fig.add_artist(node)


def _plot_fur_backdrop(ax: plt.Axes, root: Path, grid: gpd.GeoDataFrame) -> None:
    """Shared base layer for both full-width map panels: faint FUR grid
    texture, boundary, and very faint real building/road context."""
    grid.plot(ax=ax, facecolor="none", edgecolor="#B0B0B0", linewidth=0.15, alpha=0.5, zorder=1)
    fur_boundary = grid.unary_union
    gpd.GeoSeries([fur_boundary], crs=grid.crs).boundary.plot(
        ax=ax, color="#555555", linewidth=0.5, linestyle="-", zorder=2
    )
    bounds = tuple(grid.total_bounds)
    roads = _safe_read_layer(root, "roads_fua", bounds)
    if not roads.empty:
        roads.plot(ax=ax, color=PALETTE["taupe_dark"], linewidth=0.12, alpha=0.30, zorder=3)
    buildings = _safe_read_layer(root, "buildings_fua", bounds)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor=PALETTE["taupe_dark"], edgecolor="none", alpha=0.30, zorder=4)


def _plot_audit_map(ax: plt.Axes, root: Path, grid: gpd.GeoDataFrame, audit: pd.DataFrame) -> None:
    # Layer order requested: FUR at the bottom, then roads and buildings
    # (very faint, real geometry, giving surrounding-morphology context the
    # same way the single-cell audit tiles do), then audit cells on top.
    _plot_fur_backdrop(ax, root, grid)

    # Audit cells, hollow pass-colored outlines, on top.
    base = grid.merge(audit[["grid_id", "audit_priority_pass"]], on="grid_id", how="inner")
    pass_labels = {"pass_1_first_150": "Pass 1", "pass_2_next_200": "Pass 2", "pass_3_remaining": "Pass 3"}
    colors = {"pass_1_first_150": PALETTE["pink"], "pass_2_next_200": PALETTE["orange"], "pass_3_remaining": PALETTE["teal"]}
    handles = []
    for key, color in colors.items():
        sub = base[base["audit_priority_pass"].eq(key)]
        sub.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=0.9, zorder=5)
        handles.append(Patch(facecolor="none", edgecolor=color, linewidth=1.4, label=f"{pass_labels[key]} (n={len(sub):,})"))
    ax.legend(handles=handles, fontsize=11.5, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none", loc="lower left", bbox_to_anchor=(0.02, 0.35), handletextpad=0.6, labelspacing=0.55)

    _set_map_extent(ax, grid.total_bounds)
    _add_scale_north_above_legend(ax, base_loc=(0.035, 0.49), scale_len_km=5.0)
    _panel_label(ax, "D", x=0.012, y=0.955, va="top")


QUALITY_LABELS = {"high": "High", "medium": "Medium", "low": "Low", "reject_edge": "Reject/edge"}
QUALITY_COLORS = {"high": PALETTE["teal"], "medium": PALETTE["orange"], "low": "#9E365F", "reject_edge": PALETTE["muted"]}
QUALITY_ORDER = ["high", "medium", "low", "reject_edge"]


PASS_LABELS = {"pass_1_first_150": "Pass 1", "pass_2_next_200": "Pass 2", "pass_3_remaining": "Pass 3"}
PASS_ORDER = ["pass_1_first_150", "pass_2_next_200", "pass_3_remaining"]


PASS_COLORS = {"pass_1_first_150": PALETTE["pink"], "pass_2_next_200": PALETTE["orange"], "pass_3_remaining": PALETTE["teal"]}


def _v_half_violin(ax: plt.Axes, x0: float, data: np.ndarray, color: str, max_width: float = 0.15) -> None:
    """A violin bulging rightward from x0, matching Figure 5's raincloud
    (_half_violin) but rotated 90 degrees for a dodged categorical x-axis."""
    if len(data) < 3 or np.std(data) == 0:
        return
    ys = np.linspace(data.min(), data.max(), 200)
    kde = gaussian_kde(data, bw_method=0.35)
    density = kde(ys)
    density = density / density.max() * max_width
    ax.fill_betweenx(ys, x0, x0 + density, color=color, alpha=0.32, linewidth=0, zorder=2)
    ax.plot(x0 + density, ys, color=color, linewidth=1.0, alpha=0.85, zorder=3)


def _v_mini_box(ax: plt.Axes, x0: float, data: np.ndarray, color: str, width: float = 0.10) -> None:
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)
    ax.plot([x0, x0], [lo, hi], color=color, linewidth=1.0, zorder=4)
    ax.add_patch(plt.Rectangle((x0 - width / 2, q1), width, q3 - q1, facecolor=color, alpha=0.62, edgecolor=PALETTE["ink"], linewidth=0.6, zorder=5))
    ax.plot([x0 - width / 2, x0 + width / 2], [med, med], color="white", linewidth=1.2, zorder=6)


def _plot_quality_chart(ax: plt.Axes, audit: pd.DataFrame, audit_quality: pd.DataFrame, label_x: float = 0.005) -> None:
    """A raincloud, in the same visual language as Figure 5b's confidence
    raincloud (half-violin + mini box + jittered strip), rotated to a
    vertical/dodged layout: one mini-raincloud per (final quality flag,
    review pass) combination. The continuous variable is each cell's own
    audit_priority_score (the triage heuristic that ordered the queue into
    passes 1-3) -- testing whether the queue's own priority score actually
    predicted which cells would end up lower quality, and whether that
    relationship held the same way in every pass, rather than repeating
    (d)'s plain review-order geography."""
    merged = audit[["grid_id", "audit_priority_pass", "audit_priority_score"]].merge(
        audit_quality[["grid_id", "label_quality"]], on="grid_id", how="inner"
    )

    quality_order = [QUALITY_LABELS[k] for k in QUALITY_ORDER]
    offsets = {PASS_ORDER[0]: -0.27, PASS_ORDER[1]: 0.0, PASS_ORDER[2]: 0.27}
    rng = np.random.default_rng(11)

    for qi, qkey in enumerate(QUALITY_ORDER):
        for pkey in PASS_ORDER:
            values = merged.loc[
                (merged["label_quality"] == qkey) & (merged["audit_priority_pass"] == pkey), "audit_priority_score"
            ].to_numpy()
            if len(values) == 0:
                continue
            x0 = qi + offsets[pkey]
            color = PASS_COLORS[pkey]
            _v_half_violin(ax, x0 + 0.05, values, color, max_width=0.15)
            _v_mini_box(ax, x0 - 0.03, values, color, width=0.09)
            jitter = rng.uniform(-0.10, -0.06, size=len(values))
            ax.scatter(x0 + jitter, values, s=4.5, color=color, alpha=0.35, edgecolors="none", zorder=1)

    # Tight, symmetric margins around the dodge groups' real reach (not the
    # arbitrary -0.6/+0.4 used before) so the plotted content -- not just the
    # invisible axes box -- runs edge to edge with (d) above it, on both the
    # left and the right, regardless of which quality/pass cells happen to
    # be empty (e.g. no Pass-3 Reject/edge cells).
    ax.set_xlim(-0.42, len(quality_order) - 1 + 0.42)
    ax.set_xticks(range(len(quality_order)))
    ax.set_xticklabels(quality_order, fontsize=10)
    ax.set_xlabel("Final quality flag", fontsize=10)
    ax.set_ylabel("Priority score", fontsize=10, labelpad=4)
    ax.tick_params(axis="y", labelsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, zorder=0)
    handles = [Patch(facecolor=PASS_COLORS[k], alpha=0.75, edgecolor=PALETTE["ink"], linewidth=0.6, label=PASS_LABELS[k]) for k in PASS_ORDER]
    ax.legend(handles=handles, title=None, ncol=3, fontsize=9.5, frameon=True, loc="upper right")
    _panel_label(ax, "E", x=label_x, y=1.06)


def build_figure_3_label_audit(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_3_label_audit_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    audit = pd.read_csv(root / "data/04_training_labels/audit_sample_v1_priority_queue.csv")
    audit_quality = pd.read_csv(root / "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")

    examples = _select_example_cells(df, audit)
    cell_lookup = grid.set_index("grid_id")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})

    # Map aspect ratio dy/dx, matching Figure 2's construction so the two
    # figures share the bottom-row map's proportions.
    map_ar = 0.685279

    # Each of the 3 top-row super-panels (a, b, c) is a 2x2 grid of square
    # sub-cells: plan view (own identity color), LST, satellite basemap, and
    # a real 3D building-massing extrusion.
    sub_w = 2.00
    sub_gap = 0.12
    super_w = 2 * sub_w + sub_gap
    h_gap = 0.35
    # Per user request: gap between panel D and panel E reduced by 1/2
    # (0.40 -> 0.20), and gap between row 1 (A-C) and panel D reduced by
    # 2/3 (0.40 -> 0.1333), independently of each other.
    v_gap_d_e = 0.40 * 0.5
    v_gap_row1_d = 0.40 * (1 / 3)

    left_margin = 0.62
    right_margin = 0.60
    top_margin = 0.45
    bottom_margin = 0.38

    w_total = 3 * super_w + 2 * h_gap

    # Row 2: panel D spans w_total, height matches the true map aspect ratio.
    panel_d_w = w_total
    panel_d_h = panel_d_w * map_ar

    # Row 3: panel E is a chart, not a map, so it gets its own aspect ratio
    # rather than map_ar, matching Figure 4(e)'s boxplot-plus-stripplot
    # panel construction (legend sits inside the axes, no extra reserve
    # needed below it).
    panel_e_w = w_total
    panel_e_h = 1.95

    fig_w = left_margin + w_total + right_margin
    fig_h = bottom_margin + panel_e_h + v_gap_d_e + panel_d_h + v_gap_row1_d + super_w + top_margin

    b_row3 = bottom_margin
    b_row2 = b_row3 + panel_e_h + v_gap_d_e
    b_row1 = b_row2 + panel_d_h + v_gap_row1_d

    left_a = left_margin
    left_b = left_a + super_w + h_gap
    left_c = left_b + super_w + h_gap

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    def _super_panel_axes(left: float) -> dict[str, plt.Axes]:
        nw = fig.add_axes([left / fig_w, (b_row1 + sub_w + sub_gap) / fig_h, sub_w / fig_w, sub_w / fig_h])
        ne = fig.add_axes([(left + sub_w + sub_gap) / fig_w, (b_row1 + sub_w + sub_gap) / fig_h, sub_w / fig_w, sub_w / fig_h])
        sw = fig.add_axes([left / fig_w, b_row1 / fig_h, sub_w / fig_w, sub_w / fig_h])
        se = fig.add_axes([(left + sub_w + sub_gap) / fig_w, b_row1 / fig_h, sub_w / fig_w, sub_w / fig_h], projection="3d")
        return {"plan": nw, "lst": ne, "basemap": sw, "3d": se}

    axes_a = _super_panel_axes(left_a)
    axes_b = _super_panel_axes(left_b)
    axes_c = _super_panel_axes(left_c)
    ax_d = fig.add_axes([left_margin / fig_w, b_row2 / fig_h, panel_d_w / fig_w, panel_d_h / fig_h])
    # (d)'s real FUR polygon does not quite reach its own box's right edge
    # (irregular coastline shape, ~1.4% short even with equal-aspect
    # padding), so panel (e)'s box -- which would otherwise fill 100% of
    # the same width and visibly overshoot (d)'s drawn content on the right
    # -- is trimmed by that same measured fraction to end at the same
    # visual point, keeping both panels aligned left AND right.
    # Per user request, panel E's box shifts right a bit (its width is
    # unchanged -- a pure horizontal translation, not a resize), while its
    # own panel-letter tag stays put at its original absolute position.
    # Since `_panel_label` anchors to the axes' own transAxes, shifting the
    # axes box would drag the tag along with it unless compensated: the
    # tag's x offset below is reduced by the same distance (converted to
    # this box's own width-fraction) that the box itself moves right.
    panel_e_actual_w = panel_e_w * 0.986
    panel_e_shift_right = 0.25
    ax_e = fig.add_axes([(left_margin + panel_e_shift_right) / fig_w, b_row3 / fig_h, panel_e_actual_w / fig_w, panel_e_h / fig_h])

    confident_row, transition_row, pass1_row = examples.iloc[0], examples.iloc[1], examples.iloc[2]
    cell_confident = cell_lookup.loc[[confident_row["grid_id"]]].reset_index()
    cell_transition = cell_lookup.loc[[transition_row["grid_id"]]].reset_index()
    cell_pass1 = cell_lookup.loc[[pass1_row["grid_id"]]].reset_index()

    # Identity colors are drawn from the same 3-color audit-pass palette used
    # in (d) (pink/orange/teal) rather than inventing new ones, applied to
    # plan view AND the 3D roof alike. Panel C is literally a Pass-1 cell, so
    # it keeps pink; A and B borrow the other two pass colors purely for a
    # consistent, in-palette visual identity (they are not audit-sample
    # members themselves).
    color_a, color_b, color_c = PALETTE["teal"], PALETTE["orange"], PALETTE["pink"]

    _plot_example_supercell(fig, axes_a, root, cell_confident, confident_row, "A", color_a)
    _plot_example_supercell(fig, axes_b, root, cell_transition, transition_row, "B", color_b)
    _plot_example_supercell(fig, axes_c, root, cell_pass1, pass1_row, "C", color_c, extra_note="Audit sample: Pass 1")
    _plot_audit_map(ax_d, root, grid, audit)
    _plot_quality_chart(ax_e, audit, audit_quality, label_x=0.005 - panel_e_shift_right / panel_e_actual_w)

    def _group_box(left: float) -> tuple[float, float, float, float]:
        return (left / fig_w, b_row1 / fig_h, super_w / fig_w, super_w / fig_h)

    confident_xy = (cell_confident.geometry.iloc[0].centroid.x, cell_confident.geometry.iloc[0].centroid.y)
    transition_xy = (cell_transition.geometry.iloc[0].centroid.x, cell_transition.geometry.iloc[0].centroid.y)
    pass1_xy = (cell_pass1.geometry.iloc[0].centroid.x, cell_pass1.geometry.iloc[0].centroid.y)
    _add_group_callout(fig, ax_d, confident_xy, _group_box(left_a), color_a)
    _add_group_callout(fig, ax_d, transition_xy, _group_box(left_b), color_b)
    _add_group_callout(fig, ax_d, pass1_xy, _group_box(left_c), color_c)

    _finalize_panel_labels(fig)

    # Per user request, crop to the figure's own actual content extent
    # (same technique as Figures 2/4/7) -- the right margin's nominal
    # 0.60in, plus (d)/(e)'s own ~1.4% right-edge trim for the coastline
    # shape, left more blank space on the right than the left margin has.
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig_w_in, fig_h_in = fig.get_size_inches()
    pad_in = 0.10
    crop_bbox = mpl.transforms.Bbox.from_extents(
        max(tight_bbox.x0 - pad_in, 0.0),
        max(tight_bbox.y0 - pad_in, 0.0),
        min(tight_bbox.x1 + pad_in, fig_w_in),
        min(tight_bbox.y1 + pad_in, fig_h_in),
    )

    png = fig_dir / "fig_3_data_readiness.png"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches=crop_bbox)
    plt.close(fig)

    examples.to_csv(out_dir / "fig_3_example_cells.csv", index=False)
    qa = {
        "figure": "fig_3_data_readiness",
        "date": "2026-08-01",
        "panel_count": 5,
        "layout": (
            "Adapts Figure 2's construction: same _panel_label style, no per-panel "
            "titles, 1 full-width faint-backdrop map panel (d) reused from "
            "build_figure_2_methodology_multipanel in row 2. Row 1's 3 "
            "super-panels (a-c) are each a 2x2 grid: plan view (own identity color, "
            "real building footprints/roads), Landsat summer LST, Esri World Imagery "
            "satellite basemap (fetched live via contextily), and a real matplotlib "
            "3D building-massing extrusion (height = ZEMINUSTUK floor count x 3.2 m, "
            "1.2x vertical exaggeration for legibility). Star markers and dashed "
            "leader lines in (d) connect each super-panel's plan quadrant to its "
            "real location on the map. Row 3's panel (e) is a chart, not a third "
            "map: it is a raincloud (half-violin + mini box + jittered strip per "
            "group), in the same visual construction as Figure 5b's confidence "
            "raincloud (_half_violin/_mini_box), rotated to a dodged categorical "
            "x-axis. It plots audit_priority_score (the triage heuristic that "
            "ordered cells into passes 1-3) against final label_quality, dodged "
            "by review pass -- testing whether the queue's own priority score "
            "predicted final quality (it did not, within a pass) rather than "
            "just repeating (d)'s review-order geography."
        ),
        "third_d_visualization_note": (
            "The osm_quick_3d QGIS plugin was investigated but not used: it builds a live QGIS "
            "Desktop Qt3D dock canvas (needs `iface`/a running QGIS session), and "
            "this QGIS 4.2 build's qgis._3d module has no headless scene-to-image "
            "capture API (Qgs3DMapExportSettings exports a 3D model file, not a "
            "raster). The 3D quadrant is instead a from-scratch matplotlib "
            "Poly3DCollection extrusion using the project's own building/height data."
        ),
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/akilli_sehir_fua.gpkg",
            "data/02_interim/rasters/izmir_fua_landsat_summer_2021_2025_lst_epsg5253_30m.tif",
            "data/04_training_labels/audit_sample_v1_priority_queue.csv",
            "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv",
            "Esri World Imagery (live web basemap via contextily, not a tracked local file)",
        ],
        "png": str(png),
        "claim_boundary": "Manual audit is complete; agreement figures live in the audited-validation figure/table, not here.",
        "example_cell_districts": EXAMPLE_DISTRICTS,
        "example_cell_selection_rule": (
            "One example each from Konak (confident), Bayrakli (transition), and "
            "Bornova (Pass-1 audit sample), all from the four-class eligible-core "
            "building-bearing population. Confident: max lcz_weak_confidence within "
            "district (tie-break: max building_coverage_exact). Transition: "
            "lcz_mixed_flag==1 within district, min |lcz_weak_confidence-0.50| "
            "(tie-break: max building_coverage_exact). Pass 1: audit_priority_pass=="
            "'pass_1_first_150' within district, max building_coverage_exact."
        ),
        "example_cells_csv": str(out_dir / "fig_3_example_cells.csv"),
    }
    (out_dir / "fig_3_label_audit_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_3_label_audit_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_3_label_audit(), indent=2))
