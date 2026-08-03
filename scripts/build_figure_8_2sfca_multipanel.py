"""Build a compact multi-panel 2SFCA access-geography and scale-sensitivity figure."""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
from matplotlib.path import Path as MplPath
from matplotlib.patches import ConnectionPatch, PathPatch, Rectangle
from scipy.spatial import cKDTree

from build_figure_1_texture_atlas import PALETTE


THRESHOLD_FIELDS = {
    "400 m": "green_2sfca_400m_access_log1p",
    "800 m": "green_2sfca_800m_access_log1p",
    "1200 m": "green_2sfca_1200m_access_log1p",
}

DELTA_COL = "access_delta_1200_400"
ZOOM_HALF_M = 900.0


def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


ACCESS_CMAP = _cmap("climorfa_2sfca", [PALETTE["taupe_light"], PALETTE["orange"], PALETTE["pink"]])
DELTA_CMAP = _cmap("climorfa_2sfca_delta", [PALETTE["teal_dark"], PALETTE["taupe_light"], PALETTE["pink"]])


def _panel_label(ax: plt.Axes, label: str, y: float = 1.03, fontsize: float = 14.0) -> None:
    # Above the axes (not inside its top-left corner) with a neutral
    # border, matching the house convention set by Figure 2
    # (build_figure_2_methodology_multipanel.py's own _panel_label). `y` is
    # in axes-fraction units, so the same fraction is a much bigger absolute
    # gap on panel D (~3x taller than A-C) than on A-C -- callers on a
    # much-taller axes should pass a smaller fraction to keep the same
    # visual gap.
    ax.text(
        0.005,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
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
            if re.fullmatch(r"[A-Za-z]", text):
                child.set_text(f"({text.lower()})")


def _set_map(ax: plt.Axes, study_area: gpd.GeoDataFrame) -> None:
    # Extent comes from the study-area boundary, not just the plotted
    # subset's own bounds, so the boundary line drawn below is never
    # clipped even where the colored analysis subset falls short of it.
    xmin, ymin, xmax, ymax = study_area.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    # Working-area (FUR) boundary on every map, so a reader can place the
    # colored analysis subset against the actual study extent rather than
    # reading the axes limits (which are just this subset's own bounds) as
    # if they were the boundary.
    study_area.boundary.plot(ax=ax, color="#555555", linewidth=0.6, linestyle="-", zorder=10)


def _plot_access_map(
    ax: plt.Axes,
    cax: plt.Axes,
    g: gpd.GeoDataFrame,
    study_area: gpd.GeoDataFrame,
    field: str,
    threshold_label: str,
    sub_label: str,
    vmax: float,
) -> None:
    g.plot(column=field, ax=ax, cmap=ACCESS_CMAP, vmin=0, vmax=vmax, edgecolor="none", linewidth=0)
    _set_map(ax, study_area)
    # Inside-corner roman-numeral sub-label (Figure 1/4's small-multiple
    # convention), not the above-axes top-level letter: these three maps are
    # now sub-panels of a single group panel A (see build_figure_8_2sfca's
    # own call to `_panel_label(ax_a, "A", ...)`), and a visible title
    # above each one carries its own threshold explicitly.
    _sub_panel_label(ax, sub_label)
    ax.set_title(threshold_label, fontsize=11.5, pad=6, fontweight="bold", color="#222222")
    # `cax` is a standalone figure axes placed just outside (left of) the
    # map, in the same column as this group's own panel letter -- not an
    # inset over the map itself. An inset colorbar was tried first but a
    # vertical bar tall enough to read well inevitably crossed the
    # coastline/map content somewhere, however it was positioned; a bar
    # drawn outside the map has no such collision.
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=vmax), cmap=ACCESS_CMAP)
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label(f"{threshold_label} access, log1p m$^2$/person", fontsize=6.4, labelpad=3)


def _plot_delta_map(
    ax: plt.Axes,
    cax: plt.Axes,
    g: gpd.GeoDataFrame,
    study_area: gpd.GeoDataFrame,
    vmax: float,
    panel_label_down_points: float = 0.0,
) -> dict[str, float]:
    g[DELTA_COL] = g["green_2sfca_1200m_access_log1p"] - g["green_2sfca_400m_access_log1p"]
    g.plot(column=DELTA_COL, ax=ax, cmap=DELTA_CMAP, vmin=-vmax, vmax=vmax, edgecolor="none", linewidth=0)
    _set_map(ax, study_area)
    # This panel's axes is ~3.3x taller than group panel A's (same total
    # width, but one row instead of three side by side), so the shared 1.03
    # axes-fraction used on A's sub-panels would put its letter tag ~3.3x
    # further above the axes in absolute inches -- visibly higher/floatier
    # than A's own tags. Scaled down to land at the same absolute gap
    # instead.
    # Convert the requested physical downward shift to axes-fraction units so
    # it remains exactly 5 points on the rendered figure.
    axes_height_in = ax.get_position().height * ax.figure.get_figheight()
    label_y = 1.005 - (panel_label_down_points / 72.0) / axes_height_in
    _panel_label(ax, "B", y=label_y)

    # Same external, left-column-aligned colorbar treatment as group panel
    # A's sub-panels, sized the same as theirs (not stretched to this
    # panel's much taller extent) and anchored at the top of its own
    # column, directly under this panel's own tag -- consistent scale and
    # position rather than a panel-relative one. A more defined legend than a bare numeric
    # colorbar: an explicit units label plus which-threshold-wins text at
    # each pole, since "-4"/"4" alone do not say which direction is which
    # threshold.
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=DELTA_CMAP)
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("$\\Delta$ 2SFCA access, log1p (1200 m $-$ 400 m)", fontsize=6.4, labelpad=3)
    cb.ax.text(0.5, 1.05, "more access\nat 1200 m", transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=5.6, color=PALETTE["pink_dark"], fontweight="bold", linespacing=1.15)
    cb.ax.text(0.5, -0.05, "more access\nat 400 m", transform=cb.ax.transAxes, ha="center", va="top", fontsize=5.6, color=PALETTE["teal_dark"], fontweight="bold", linespacing=1.15)

    # Improved addition: the map alone shows *where* the sign flips, but not
    # how common each direction actually is. A wider catchment adds both
    # supply and competing population, so access is not guaranteed to rise
    # with distance -- quantifying the split turns that into a checkable
    # number instead of a purely visual impression.
    delta = g[DELTA_COL]
    share_gain = float((delta > 1e-9).mean()) * 100
    share_loss = float((delta < -1e-9).mean()) * 100
    share_flat = 100 - share_gain - share_loss
    ax.text(
        0.988,
        0.02,
        f"{share_gain:.0f}% of cells gain access at 1200 m,\n{share_loss:.0f}% lose access (population growth\noutpaces new supply), {share_flat:.0f}% unchanged.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["ink"],
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 3.5},
        zorder=20,
    )
    return {"share_gain_pct": share_gain, "share_loss_pct": share_loss, "share_flat_pct": share_flat}


def _select_delta_hotspots(g: gpd.GeoDataFrame, zoom_half_m: float = ZOOM_HALF_M) -> list[dict]:
    """Deterministically pick 3 non-overlapping locations off the actual delta
    field panel D already plots, each making a different scale-sensitivity
    point: the strongest 1200 m gain, the strongest 400 m advantage, and the
    most locally contested cell (largest local spread -- gain and loss cells
    sitting right next to each other). Not hand-picked or eyeballed off the
    rendered map. Local stats are computed over the exact same square window
    (side = 2 * zoom_half_m) that `_plot_zoom_panel` later crops and displays
    -- an earlier version scored candidates over a smaller circular
    neighborhood, which could pick a point whose immediate few cells were an
    extreme but whose eventual, larger square crop was actually dominated by
    the opposite sign (e.g. a 'strongest 400 m advantage' pick whose crop
    came out net positive) -- using the same window for both keeps the title
    and the crop honest with each other."""
    centroids = g.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    delta = g[DELTA_COL].to_numpy()
    tree = cKDTree(xy)
    # Circumscribing radius so a square-box filter afterward never misses a
    # point that's inside the box but outside a same-radius circle.
    search_r = zoom_half_m * np.sqrt(2)
    candidate_idx = tree.query_ball_point(xy, r=search_r)
    n_points = len(xy)
    local_n = np.zeros(n_points, dtype=int)
    local_mean = np.full(n_points, np.nan)
    local_std = np.full(n_points, np.nan)
    for i, idx in enumerate(candidate_idx):
        idx = np.asarray(idx)
        in_box = idx[(np.abs(xy[idx, 0] - xy[i, 0]) <= zoom_half_m) & (np.abs(xy[idx, 1] - xy[i, 1]) <= zoom_half_m)]
        local_n[i] = len(in_box)
        if len(in_box):
            local_mean[i] = delta[in_box].mean()
            local_std[i] = delta[in_box].std()
    valid = local_n >= 10

    specs = [
        ("Strongest 1200 m gain", PALETTE["pink_dark"], np.argsort(-np.where(valid, local_mean, -np.inf))),
        ("Strongest 400 m advantage", PALETTE["teal_dark"], np.argsort(np.where(valid, local_mean, np.inf))),
        ("Most locally contested", PALETTE["orange"], np.argsort(-np.where(valid, local_std, -np.inf))),
    ]
    chosen: list[dict] = []
    min_sep = zoom_half_m * 2.5
    for title, color, order in specs:
        for i in order:
            if not valid[i]:
                continue
            x, y = xy[i]
            if any(np.hypot(x - c["x"], y - c["y"]) < min_sep for c in chosen):
                continue
            chosen.append({"x": float(x), "y": float(y), "geom": g.geometry.iloc[i], "title": title, "color": color})
            break
    return chosen


def _sub_panel_label(ax: plt.Axes, label: str) -> None:
    # Inside the axes' own top-left corner (not above it), matching Figure 1/
    # Figure 4's roman-numeral sub-panel convention for a small-multiple
    # within a single lettered panel -- distinct from the above-axes
    # top-level `_panel_label` convention, so it never competes for the same
    # vertical space as the group letter or each zoom's own descriptive
    # title (both of which sit above the axes).
    ax.text(
        0.035,
        0.955,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#555555", "boxstyle": "round,pad=0.15", "linewidth": 0.7, "alpha": 0.95},
        zorder=30,
    )


def _highlight_cell(ax: plt.Axes, geom, color: str) -> None:
    """Highlight the exact hotspot grid cell by lightening it from its own
    edges inward (a radial white gradient, strongest at the cell's center),
    rather than marking it with a star drawn on top -- the cell itself
    already reads as distinct once its center visibly lightens, and the
    gradient stays entirely inside that one cell's own polygon so it never
    bleeds into its neighbors. The cell's own boundary is redrawn in the
    callout's color, tying it back to the dashed frame/leader line without
    a separate marker."""
    minx, miny, maxx, maxy = geom.bounds
    # A soft colored halo bleeding slightly into the neighboring cells, so
    # the highlight stays visible at a consistent strength regardless of the
    # target cell's own fill color -- a cell near zero delta (the "most
    # locally contested" pick, for instance) sits on the colormap's own
    # near-white neutral, and without this the white gradient above reads as
    # much fainter there than on a strongly colored cell, even though the
    # cell and its outline are drawn at the identical pixel size every time.
    pad = (maxx - minx) * 0.22
    ax.add_patch(Rectangle((minx - pad, miny - pad), (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad, facecolor=color, alpha=0.16, edgecolor="none", zorder=13))
    n = 80
    xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    dist = np.sqrt(xx**2 + yy**2)
    alpha = np.clip(1 - dist, 0, 1) ** 1.4
    rgba = np.ones((n, n, 4))
    rgba[..., 3] = alpha * 0.92
    im = ax.imshow(rgba, extent=(minx, maxx, miny, maxy), origin="lower", zorder=15, interpolation="bilinear")
    path = MplPath(np.array(geom.exterior.coords))
    clip_patch = PathPatch(path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    im.set_clip_path(clip_patch)
    gx, gy = geom.exterior.xy
    ax.plot(gx, gy, color=color, linewidth=2.8, zorder=16)


def _plot_zoom_panel(
    ax: plt.Axes,
    g: gpd.GeoDataFrame,
    study_area: gpd.GeoDataFrame,
    hotspot: dict,
    delta_vmax: float,
    sub_label: str,
    zoom_half_m: float = ZOOM_HALF_M,
) -> dict[str, float]:
    x0, y0 = hotspot["x"], hotspot["y"]
    xmin, xmax = x0 - zoom_half_m, x0 + zoom_half_m
    ymin, ymax = y0 - zoom_half_m, y0 + zoom_half_m
    crop = g.cx[xmin:xmax, ymin:ymax]

    crop.plot(column=DELTA_COL, ax=ax, cmap=DELTA_CMAP, vmin=-delta_vmax, vmax=delta_vmax, edgecolor="white", linewidth=0.2, zorder=5)
    study_area.boundary.plot(ax=ax, color="#555555", linewidth=0.6, linestyle="-", zorder=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # The exact hotspot cell is highlighted directly (inward white gradient
    # + a colored outline) rather than marked with a star -- unlike panel B,
    # where the same cell is a single indistinguishable pixel among
    # thousands and a star is the only way to find it, here the cell is
    # large enough on screen that lightening it in place reads clearly on
    # its own.
    _highlight_cell(ax, hotspot["geom"], hotspot["color"])
    _sub_panel_label(ax, sub_label)
    # A large pad (default is ~3pt) is needed here specifically: the
    # callout's connection node sits right at this axes' top edge (see
    # `_add_zoom_callout`'s `target`), and a title with the default small
    # pad renders at almost the same height, so the node visually sat on
    # top of the title text. Pushing the title well clear of the axes top
    # keeps the node/line, the title, and the axes content in a clean
    # top-to-bottom stack instead.
    ax.set_title(hotspot["title"], fontsize=8.6, pad=17, color=hotspot["color"], fontweight="bold")

    # Advanced/computed values for this exact crop (not the wider selection
    # radius used to find it) -- the same real-number-over-decoration
    # convention already used for panel D's own stat box.
    vals = crop[DELTA_COL].to_numpy()
    n = len(vals)
    mean_delta = float(np.mean(vals)) if n else float("nan")
    share_gain = float((vals > 1e-9).mean()) * 100 if n else 0.0
    share_loss = float((vals < -1e-9).mean()) * 100 if n else 0.0
    share_flat = 100 - share_gain - share_loss if n else 0.0
    ax.text(
        0.03,
        0.03,
        f"n={n} cells, mean $\\Delta$={mean_delta:+.2f}\n{share_gain:.0f}% gain / {share_loss:.0f}% loss / {share_flat:.0f}% flat",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=PALETTE["ink"],
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 2.6},
        zorder=20,
    )
    return {"n_cells": n, "mean_delta": mean_delta, "share_gain_pct": share_gain, "share_loss_pct": share_loss, "share_flat_pct": share_flat}


def _add_zoom_callout(fig: plt.Figure, ax_map: plt.Axes, xy: tuple[float, float], zoom_box_fig: tuple[float, float, float, float], color: str) -> None:
    """Mirrors Figure 3's `_add_group_callout` (star on the source map +
    dashed frame around the zoomed group + a right-angle dashed leader line
    connecting the two), flipped vertically: there the zoomed group sat
    above its source map, here the zoom row sits below panel D, so the
    leader line's target is the zoom box's top-mid edge instead of its
    bottom-mid."""
    left, bottom, width, height = zoom_box_fig
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

    ax_map.scatter([xy[0]], [xy[1]], s=140, marker="*", facecolor=color, edgecolor="white", linewidth=1.0, zorder=22)
    target = (left + width / 2, bottom + height)
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

    node = plt.Circle(target, radius=0.0055, transform=fig.transFigure, facecolor=color, edgecolor="white", linewidth=0.8, zorder=26)
    fig.add_artist(node)


def build_figure_8_2sfca(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_8_2sfca_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    green = gpd.read_file(root / "data/03_processed/grid_green_space_2sfca.gpkg", layer="grid_250m_green_space_2sfca")
    study_area = gpd.read_file(root / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    features = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    keep = ["grid_id", "eligible_core", "lcz_weak_label", "lcz_weak_confidence", "lst_c_median_mean"]
    g = green.merge(features[keep], on="grid_id", how="left")
    analysis = g[(g["eligible_core"] == 1) & (g["lcz_weak_label"].isin([3, 6, 8, 9])) & (g["lcz_weak_confidence"] >= 0.60)].copy()
    vmax = float(np.nanpercentile(analysis[list(THRESHOLD_FIELDS.values())].to_numpy(), 98))
    delta_vmax = float(np.nanpercentile(np.abs(analysis["green_2sfca_1200m_access_log1p"] - analysis["green_2sfca_400m_access_log1p"]), 98))
    delta_vmax = max(delta_vmax, 0.1)

    mpl.rcParams.update({"font.family": "DejaVu Sans"})

    # Same explicit fig.add_axes() composition as Figure 2 (build_figure_2_
    # methodology_multipanel.py): equal-width map panels in a top row, sized
    # to the data's own aspect ratio rather than a generic square GridSpec
    # cell, with one panel spanning the same total width below -- here 3
    # access-map panels on top (one fewer than Figure 2's 4, since this
    # figure only has three thresholds), the delta map spanning below that,
    # and a third row of 3 zoom-in crops (panel E) spanning the same width
    # again at the bottom. h_gap is widened from the original 0.35in to fit
    # each panel's standalone colorbar column between it and its neighbor.
    xmin, ymin, xmax, ymax = study_area.total_bounds
    map_ar = (ymax - ymin) / (xmax - xmin)

    panel_w = 3.60
    panel_h = panel_w * map_ar
    h_gap = 0.55
    v_gap = 0.50
    left_margin = 0.95
    # Panel A ramps now sit to the right of their maps; leave enough canvas
    # for the rightmost ramp's vertical label and tick labels.
    right_margin = 1.10
    top_margin = 0.55
    bottom_margin = 0.45

    w_total = 3 * panel_w + 2 * h_gap
    panel_d_w = w_total
    panel_d_h = panel_d_w * map_ar

    # Row 3 (panel E): 3 square zoom-in crops spanning the same total width
    # as rows 1-2, with a wider gap than h_gap so each crop's dashed
    # callout frame has clear room to sit in without touching its neighbor.
    zoom_gap = 0.45
    zoom_w = (w_total - 2 * zoom_gap) / 3
    zoom_h = zoom_w
    v_gap2 = 0.55

    fig_w = left_margin + w_total + right_margin
    fig_h = bottom_margin + zoom_h + v_gap2 + panel_d_h + v_gap + panel_h + top_margin

    b_row3 = bottom_margin
    b_row2 = b_row3 + zoom_h + v_gap2
    # Per user request, panel A (row 1: the 3 access maps) shifts down
    # within the existing canvas -- fig_h is unchanged, so this both opens
    # up more headroom above panel A's own letter/titles and shrinks the
    # nominal v_gap between row 1 and row 2 (panel B, the delta map) by the
    # same amount.
    panel_a_shift_down = 0.20
    b_row1 = b_row2 + panel_d_h + v_gap - panel_a_shift_down

    left_a = left_margin
    left_b = left_a + panel_w + h_gap
    left_c = left_b + panel_w + h_gap

    left_e0 = left_margin
    left_e1 = left_e0 + zoom_w + zoom_gap
    left_e2 = left_e1 + zoom_w + zoom_gap

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    ax_a = fig.add_axes([left_a / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_b = fig.add_axes([left_b / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_c = fig.add_axes([left_c / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_d = fig.add_axes([left_margin / fig_w, b_row2 / fig_h, panel_d_w / fig_w, panel_d_h / fig_h])
    ax_e0 = fig.add_axes([left_e0 / fig_w, b_row3 / fig_h, zoom_w / fig_w, zoom_h / fig_h])
    ax_e1 = fig.add_axes([left_e1 / fig_w, b_row3 / fig_h, zoom_w / fig_w, zoom_h / fig_h])
    ax_e2 = fig.add_axes([left_e2 / fig_w, b_row3 / fig_h, zoom_w / fig_w, zoom_h / fig_h])

    # Each colorbar is its own figure axes, standing just outside its map,
    # capped to
    # a shared, legible absolute height and anchored to the top of its row
    # (i.e. right under the letter) rather than stretched to fill the row --
    # panel D's row is ~3x taller than A-C's, and a colorbar stretched to
    # match would be a huge, hard-to-read bar out of proportion to its tick
    # range.
    cbar_w = 0.13
    cbar_gap = 0.14
    cbar_h = min(panel_h * 0.72, 1.9)
    cbar_top_pad = 0.06

    def _cbar_axes(
        map_left: float,
        row_top: float,
        top_pad: float = cbar_top_pad,
        w: float = cbar_w,
        h: float = cbar_h,
        side: str = "left",
        x_shift: float = 0.0,
        y_shift: float = 0.0,
    ) -> plt.Axes:
        if side == "right":
            x0 = map_left + panel_w + cbar_gap + x_shift
        else:
            x0 = map_left - cbar_gap - w + x_shift
        y1 = row_top - top_pad + y_shift
        y0 = y1 - h
        return fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])

    # Panel A's three ramps sit on the right side of their corresponding maps.
    cax_a = _cbar_axes(left_a, b_row1 + panel_h, side="right")
    cax_b = _cbar_axes(left_b, b_row1 + panel_h, side="right")
    cax_c = _cbar_axes(left_c, b_row1 + panel_h, side="right")
    # Bigger and nudged further down than A-C's shared size/top_pad: D's row
    # has ample headroom (~3.3x A-C's height) to grow into, and sitting
    # right at the very top of that tall row read as floating relative to
    # the "D" tag right above it.
    # Panel B's ramp is aligned immediately to the right of the B tag.
    # Keep its current vertical position unchanged (10 points down).
    ten_points = 10.0 / 72.0
    panel_b_label_right_x_shift = 0.56
    cax_d = _cbar_axes(
        left_margin,
        b_row2 + panel_d_h,
        top_pad=0.68,
        w=0.19,
        h=2.7,
        x_shift=panel_b_label_right_x_shift,
        y_shift=-ten_points,
    )

    # Panel A: 3 access maps (400/800/1200 m) as roman-numeral sub-panels of
    # a single group panel, each with its own visible threshold title, so
    # the group reads as one panel rather than three separately lettered
    # ones -- the group letter itself is added once below, on the leftmost
    # sub-panel only.
    _plot_access_map(ax_a, cax_a, analysis, study_area, "green_2sfca_400m_access_log1p", "400 m", "(i)", vmax)
    _plot_access_map(ax_b, cax_b, analysis, study_area, "green_2sfca_800m_access_log1p", "800 m", "(ii)", vmax)
    _plot_access_map(ax_c, cax_c, analysis, study_area, "green_2sfca_1200m_access_log1p", "1200 m", "(iii)", vmax)
    _panel_label(ax_a, "A")
    delta_stats = _plot_delta_map(
        ax_d,
        cax_d,
        analysis,
        study_area,
        delta_vmax,
        panel_label_down_points=5.0,
    )

    # Panel C: 3 zoom-in crops of panel B's own delta field at the locations
    # that make each scale-sensitivity point concretely (strongest gain,
    # strongest loss, most contested), with a dashed star+leader-line callout
    # back to that exact spot on panel B above -- same visual grammar as
    # Figure 3's group callouts (_add_group_callout there), mirrored since
    # the zoom row sits below its source map instead of above it.
    hotspots = _select_delta_hotspots(analysis)
    zoom_axes = [ax_e0, ax_e1, ax_e2]
    zoom_lefts = [left_e0, left_e1, left_e2]
    sub_labels = ["(i)", "(ii)", "(iii)"]
    zoom_stats = []
    for ax_zoom, left, hotspot, sub_label in zip(zoom_axes, zoom_lefts, hotspots, sub_labels):
        stats = _plot_zoom_panel(ax_zoom, analysis, study_area, hotspot, delta_vmax, sub_label)
        zoom_stats.append({"title": hotspot["title"], **stats})
        zoom_box_fig = (left / fig_w, b_row3 / fig_h, zoom_w / fig_w, zoom_h / fig_h)
        _add_zoom_callout(fig, ax_d, (hotspot["x"], hotspot["y"]), zoom_box_fig, hotspot["color"])
    _panel_label(ax_e0, "C", y=1.06)

    _finalize_panel_labels(fig)
    png = fig_dir / "fig_8_2sfca_sensitivity.png"
    # Trim the outer canvas to the actual figure artists so the exported
    # raster has no meaningless white border when placed in the manuscript.
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    # Matplotlib's tight bbox can omit artists deliberately drawn outside an
    # axes (panel tags and colorbar annotations). Crop the already-rendered
    # raster instead, so every visible label is retained while the outer
    # white canvas is removed exactly.
    with Image.open(png) as rendered:
        rgb = rendered.convert("RGB")
        bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        if bbox is not None:
            cropped = rgb.crop(bbox)
            # Add an exact 0.5 cm white margin on every side at 300 dpi.
            margin_px = round((0.5 / 2.54) * 300)
            ImageOps.expand(cropped, border=margin_px, fill="white").save(png, dpi=(300, 300))

    qa = {
        "figure": "fig_8_2sfca_sensitivity",
        "date": "2026-06-20",
        "panel_count": 3,
        "panel_labels": ["A", "B", "C"],
        "layout": "Figure-2-style explicit axes: panel A (sub-panels i-iii) is 3 equal-width access maps (400/800/1200 m, each titled with its own threshold) in a top row sized to the data's own aspect ratio; panel B is 1 wide delta map spanning the same total width below; panel C (sub-panels i-iii) is a 3rd row of square zoom-in crops of B's most important locations, each with a dashed star+leader-line callout back to its exact spot on B -- same visual grammar as Figure 3's group callouts, mirrored since the zoom row sits below its source map. Each access/delta map has its own standalone vertical colorbar axes to its left, column-aligned with the group's panel letter.",
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "data/03_processed/grid_green_space_2sfca.gpkg",
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/02_interim/study_area_fua.gpkg",
        ],
        "access_vmax_p98": vmax,
        "delta_vmax_p98_abs": delta_vmax,
        "delta_direction_shares": delta_stats,
        "zoom_half_m": ZOOM_HALF_M,
        "zoom_hotspot_stats": zoom_stats,
        "revision_2026-08-02b": "Reverted the same-day mechanism-panel and F-I additions per explicit user request: panel A (buffer-vs-network mechanism map) and panels F-I (threshold distribution violin, paired spatial-fold model contribution, official-vs-OSM count/area triangulation scatters) were all cut. Figure is now 4 panels only (A-D: the three access maps plus the delta map), relettered contiguously, and rebuilt with the same explicit fig.add_axes() composition pattern as Figure 2 (equal-width top-row map panels sized to the data's own aspect ratio, one full-width panel below) instead of the previous 3x3 GridSpec. `_build_local_road_graph`, `_select_mechanism_example`, `_plot_mechanism`, `_plot_distribution`, `_plot_fold_deltas`, and `_plot_official_scatter` were deleted outright as dead code, along with their now-unused data reads (roads_fua, green_space_2sfca_supply_audit.gpkg, 2sfca_fold_deltas.csv, district_comparison.csv) and imports (networkx, shapely ops/geometry, scipy.spatial/stats, seaborn, matplotlib.lines/patches).",
        "revision_2026-08-02c": "Per user request ('her birine çalışma alan sınırı koyalım. panel d için daha tanımlı lejant ve geliştirilmiş bir ekleme yapalım'): (1) added the same study-area (FUR) boundary outline used in Figures 1/2 to all four panels, read from data/02_interim/study_area_fua.gpkg. (2) Panel D's legend was made explicit: an axis-labeled colorbar plus which-threshold-wins text at each pole, replacing a bare numeric -4..4 scale. (3) Added a computed stat annotation to panel D (delta_direction_shares below) quantifying what the map only shows visually: access is not guaranteed to rise with a wider catchment, since a larger radius adds both new supply and new competing population.",
        "revision_2026-08-02d": "Per user request ('panel numaralandırmasının stilini figure 2 kurallarına göre yap, coloramp panel numaralandırmasına align olacak şekilde dikey konumlansın her bir map paneli için'): panel letters moved from inside each axes' top-left corner to above the axes with a neutral border, matching Figure 2's own _panel_label exactly (x=0.005, y=1.03, va='bottom', fontsize=14, edgecolor #555555). All four colorbars converted from horizontal insets to vertical, standalone figure axes placed just outside (left of) each map, in the same left column as that map's own panel letter -- an inset vertical colorbar was tried first but any bar tall enough to read inevitably crossed the coastline/map content somewhere inside the axes; moving it outside the map entirely removed that collision. Per-panel titles (previously 'X m network 2SFCA access' etc.) were dropped, since Figure 2's own panels carry no titles either and the letter+colorbar now sit where the title used to be; the threshold value itself was folded into each colorbar's own label instead, so panels A/B/C are still distinguishable without it. Panel D's colorbar is capped to the same absolute height as A-C's (not stretched to D's ~3x-taller row) and anchored at the top of its column, directly under the D tag, for a consistent scale and position across all four panels. h_gap widened 0.35in->0.55in and left_margin 0.60in->0.95in to fit the new colorbar columns.",
        "revision_2026-08-02e": "Per user request ('panel d numarası ve coloramp biraz aşağı insin'): panel D's letter tag and colorbar both nudged down. Root cause: D's axes is ~3.3x taller than A-C's, so the shared y=1.03 axes-fraction letter offset (and the shared 0.06in colorbar top_pad) landed ~3.3x further above D's own axes top in absolute inches than the same values did for A-C -- D's tag/colorbar read as floating higher than A-C's despite using the *same* relative settings. Fixed with a `_panel_label(..., y=1.009)` override for D (same absolute gap as A-C's 1.03 fraction on their own, much shorter, axes) and a `_cbar_axes(..., top_pad=0.42)` override for D's colorbar specifically, both parameterized rather than hardcoded so future panels of differing heights can be tuned the same way.",
        "revision_2026-08-02f": "Per user request ('panel e için 3-4 tane zoom-in olan ve panel d için en önemli yeri gösteren yerler ve o zoom-in alanlarının advanced değerleri draw callout çizgileri ile panel d'nin altına konumlandırılabilir mi? calloutlar aynı figure 3 panel d gibi kurgulanabilir'): added a new panel E, a 3rd row of 3 square (1.8 km-wide) zoom-in crops of panel D's own delta field, positioned below D, spanning the same total width as rows 1-2. Locations are not hand-picked: `_select_delta_hotspots` computes, from the same delta_col panel D already plots, the strongest 1200 m gain, the strongest 400 m advantage, and the most locally contested cell (largest local std -- gain and loss cells sitting right next to each other), enforcing a minimum pairwise separation so their crop windows never overlap. Each zoom shows a real computed stat box (n cells, mean delta, gain/loss/flat share within that exact crop -- the same convention as D's own stat annotation) and a 500 m scale bar. Callouts (`_add_zoom_callout`) copy Figure 3's `_add_group_callout` construction exactly (star marker at the source location + dashed colored frame around the target group + a right-angle dashed leader line + a small circular node at the connection point), mirrored vertically since here the zoomed group sits below its source map instead of above it. Sub-panels are labeled (i)/(ii)/(iii) directly (not bare 'i'/'ii'/'iii') to avoid a real quirk in `_finalize_panel_labels`: its single-character regex would otherwise silently re-parenthesize a bare 'i' into '(i)' while leaving 'ii'/'iii' untouched, since only 'i' is exactly one character -- passing already-parenthesized text sidesteps that inconsistency. `_panel_label` gained optional `fontsize` so these smaller sub-labels don't use the top-level 14pt box size.",
        "revision_2026-08-02g": "Two bugs caught while checking panel E's rendered output against its own computed numbers (not by eyeballing the map): (1) `_select_delta_hotspots`' first version scored candidates over a smaller circular neighborhood than the square crop `_plot_zoom_panel` actually displays, so the 'strongest 400 m advantage' pick's title didn't match its own crop's sign (crop came out net +0.94, mostly gain, while labeled the loss example) -- fixed by scoring every candidate over the exact same square window (side = 2*zoom_half_m) that gets displayed, using a circumscribing-radius KD-tree query plus an exact box filter, so the title and the crop's stats always agree. (2) each zoom's `ax.set_title` and its callout's connection node (anchored at that axes' top edge) rendered at almost the same height with the default ~3pt title pad, visually overlapping -- fixed with `pad=17` so the title clears the node.",
        "revision_2026-08-02h": "Per user request ('panel d colorramp biraz büyüsün ve biraz aşağı kaydırılsın'): panel D's colorbar grown from 0.13in x ~1.78in to 0.19in x 2.7in and its top_pad increased from 0.42in to 0.68in (shifted further down within its column, below the D tag). `_cbar_axes` gained optional `w`/`h` overrides (previously only `top_pad` was per-panel-tunable) so D's colorbar could grow independently of A-C's shared, smaller size without touching their geometry.",
        "revision_2026-08-02i": "Per user request ('panel a-b-c bence panel a olarak yeniden numaralanabilir. bir de her birinin üstüne 1200 400 800 diye yaz ne ise o'): the 3 access maps (formerly separately lettered A/B/C) are now roman-numeral sub-panels (i)/(ii)/(iii) of a single group panel A, matching Figure 1/4's small-multiple convention -- the group letter is added once, on the leftmost sub-panel only (`_panel_label(ax_a, 'A')`), and each sub-panel's own `_sub_panel_label` moved to the inside-corner convention instead of the above-axes one, since the above-axes slot is now used by a real visible title per user's second request: each sub-panel gained `ax.set_title('400 m'/'800 m'/'1200 m', ...)` stating its own threshold directly above the map (previously the threshold only lived inside the colorbar's own label, never as a title). The delta map (formerly D) and the zoom-in row (formerly E) relettered to B and C to stay contiguous. panel_count 5->3 (A and C now carry sub-panels rather than being separate top-level letters).",
        "revision_2026-08-02j": "Per user request ('panel c ölçek barlarını kaldır bence. bir de yıldız ne anlama geliyor? ilgili gride denk gelmesi anlam ifade ediyorsa o grid içe doğru gradyan açık renge dönüşsün vurgulamak için'): removed the 500 m scale bar from each of panel C's 3 zoom-ins (`_add_zoom_scale_bar` deleted outright as dead code once its only call site was removed). The star marker inside each zoom crop (which did mark that exact hotspot grid cell) was replaced with `_highlight_cell`: the cell's own polygon is redrawn with an inward radial white gradient (strongest at its center, fading to nothing at its edges) plus a colored outline matching that hotspot's callout color, so the specific cell reads as distinct by lightening in place rather than by a symbol drawn on top of it. `_select_delta_hotspots` now also returns each chosen cell's actual polygon geometry (not just its centroid x/y), needed to clip the gradient to that one cell exactly. Panel B's own star (marking the same hotspot within the full, ~49 km-wide map, where the cell is a single indistinguishable pixel among thousands) was deliberately left unchanged -- a gradient on a cell that small would be imperceptible at that scale, so the star is still the right marker there; the gradient replacement only made sense in panel C's zoom crops, where the cell is large enough on screen for the effect to read.",
        "revision_2026-08-02k": "Per user question ('panel c III numaralı yıldız küçük?'): the (iii) cell highlight was never actually smaller in pixels -- measured directly against (i)/(ii) in the rendered PNG, all three highlighted cells and their outlines are pixel-identical in size (same zoom_half_m crop, same axes size). The apparent size difference was a contrast artifact: (iii)'s 'most locally contested' cell has a near-zero local mean delta, so it sits on the colormap's own near-white neutral fill, and the white radial gradient highlight read much fainter there than on (i)/(ii)'s strongly colored (pink_dark/teal_dark) cells. Fixed with a soft colored halo (a low-alpha rectangle in the hotspot's own color, padded ~22% beyond the cell into its immediate neighbors, drawn behind the gradient) so every highlighted cell carries a consistent visual weight regardless of its own fill color; outline linewidth also bumped 2.2->2.8 for the same reason.",
        "png": str(png),
        "claim_boundary": "2SFCA access geography is descriptive; scale sensitivity across thresholds is shown as access-value geography, not a model-performance claim.",
    }
    (out_dir / "fig_8_2sfca_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_8_2sfca_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_8_2sfca(Path(".")), indent=2))
