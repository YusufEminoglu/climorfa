"""Build a multi-panel morphology phenotype and texture comparison figure."""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

from build_figure_1_texture_atlas import (
    CLASS_COLORS,
    CLASS_NAMES,
    CLASS_SHORT,
    CLASSES,
    PALETTE,
    SELECTION_FEATURES,
    TEXTURE_METRICS,
    _plot_texture_tile,
    _select_texture_tiles,
)

CLASS_DARK_KEY = {3: "teal_dark", 6: "pink_dark", 8: "orange_dark", 9: "taupe_dark"}


def _select_distinct_texture_tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Same deterministic per-class median-distance ranking as Figure 1's
    texture atlas (`_select_texture_tiles`), but excludes Figure 1's own four
    selected cells so Figure 4's panel A shows different real building/street
    fabric instead of repeating the identical four grid cells across two
    main-text figures. Median/SD (and therefore each cell's median_distance)
    are computed on the full eligible pool, matching Figure 1 exactly; only
    the final pick is restricted to excluding Figure 1's grid_ids, so this
    returns each class's second-closest-to-median eligible cell."""
    exclude_ids = set(_select_texture_tiles(df)["grid_id"])
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
        pool = pool[~pool["grid_id"].isin(exclude_ids)]
        if pool.empty:
            raise RuntimeError(f"No distinct-from-Figure-1 texture tile candidates for LCZ {klass}.")
        selected.append(pool.sort_values(["median_distance", "grid_id"]).iloc[0])

    return pd.DataFrame(selected)[TEXTURE_METRICS].reset_index(drop=True)


PROFILE_FIELDS = {
    "building_coverage_exact": "Coverage",
    "height_proxy_aw_mean_m": "Height proxy",
    "dsm_elevation_m_std": "Surface SD",
    "canopy_cover_gt2m_share": "Canopy >2m",
    "s2_ndvi_mean": "NDVI",
    "road_density_exact_m_per_km2": "Road density",
    "morph_open_space_fragmentation_index": "Fragmentation",
    "green_2sfca_800m_access_log1p": "2SFCA",
    "lst_c_median_mean": "LST",
}
SCATTER_ALPHA = 0.20

CONTEXT_FIELDS = [
    ("building_coverage_exact", "Cov"),
    ("height_proxy_aw_mean_m", "H"),
    ("dsm_elevation_m_std", "Surf.sd"),
    ("s2_ndvi_mean", "NDVI"),
    ("lst_c_median_mean", "LST"),
]


def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


def _panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 14.0,
    x: float = 0.005,
    y: float = 1.02,
    va: str = "bottom",
) -> None:
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


def _apply_box_gradient(ax: plt.Axes, lighten: float = 0.55) -> None:
    """Redraw each box's fill as a horizontal gradient: base class colour at
    the edges, lightened toward the box's own centre for a subtle sheen."""
    # Each imshow() call re-triggers autoscale using only that image's own
    # (tiny) extent rather than the union of all axes content, so by the
    # last box the view would otherwise have zoomed into a sliver; pin the
    # limits first and restore them once every box has its gradient.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for patch in list(ax.patches):
        if not isinstance(patch, mpl.patches.PathPatch):
            continue
        base_rgb = np.array(mpl.colors.to_rgb(patch.get_facecolor()))
        light_rgb = base_rgb + (1.0 - base_rgb) * lighten
        v = patch.get_path().vertices
        x0, x1 = float(v[:, 0].min()), float(v[:, 0].max())
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        t = np.abs(np.linspace(-1.0, 1.0, 200))
        row = base_rgb[None, :] * t[:, None] + light_rgb[None, :] * (1.0 - t[:, None])
        grad = np.tile(row[None, :, :], (2, 1, 1))
        im = ax.imshow(
            grad, extent=(x0, x1, y0, y1), origin="lower", aspect="auto",
            zorder=1.5, interpolation="bilinear",
        )
        im.set_clip_path(patch)
        patch.set_facecolor("none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _finalize_panel_labels(fig: plt.Figure) -> None:
    """Render bare panel letters (e.g. 'A') as lowercase-parenthesized (e.g. '(a)')."""
    for ax in fig.axes:
        for child in ax.texts:
            text = child.get_text()
            if text in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
                child.set_text(f"({text.lower()})")


def _analysis_subset(df: pd.DataFrame) -> pd.DataFrame:
    a = df[df["lcz_weak_label"].isin(CLASSES) & (pd.to_numeric(df["lcz_weak_confidence"], errors="coerce") >= 0.60)].copy()
    a["class"] = a["lcz_weak_label"].astype(int)
    for col in PROFILE_FIELDS:
        a[col] = pd.to_numeric(a[col], errors="coerce")
    return a


def _plot_profile_boxplots(ax: plt.Axes, a: pd.DataFrame) -> None:
    # Fragmentation is excluded from this figure entirely: its class median
    # is exactly 0 for all four classes (51-100% of cells are exactly 0 per
    # class), so it carries no distributional signal in any of the summary
    # views used here (this median-based panel, the hot/cool contrast, or a
    # per-class hexbin), and panel H now covers 2SFCA-vs-LST instead.
    fields = [
        "building_coverage_exact",
        "height_proxy_aw_mean_m",
        "dsm_elevation_m_std",
        "canopy_cover_gt2m_share",
        "s2_ndvi_mean",
        "road_density_exact_m_per_km2",
        "lst_c_median_mean",
    ]
    rows = []
    for field in fields:
        lo, hi = np.nanpercentile(a[field], [1, 99])
        scaled = (a[field] - lo) / (hi - lo) if hi > lo else a[field] * 0
        rows.append(pd.DataFrame({"variable": PROFILE_FIELDS[field], "value": scaled.clip(0, 1), "class": a["class"]}))
    long = pd.concat(rows, ignore_index=True)
    order = [PROFILE_FIELDS[f] for f in fields]
    # A faint, sparse strip of the actual cells (subsampled so it reads as
    # texture, not clutter) drawn first so the boxplot's opaque fill sits on
    # top of it, visible mainly beyond the box in the tails/outliers.
    rng = np.random.default_rng(20260730)
    sampled = pd.concat(
        [grp.sample(min(120, len(grp)), random_state=int(rng.integers(1 << 31))) for _, grp in long.groupby(["variable", "class"])],
        ignore_index=True,
    )
    sns.stripplot(
        data=sampled, x="variable", y="value", hue="class", order=order, hue_order=CLASSES,
        palette=CLASS_COLORS, dodge=True, size=2.2, alpha=0.18, jitter=0.28, ax=ax, legend=False,
    )
    sns.boxplot(
        data=long,
        x="variable",
        y="value",
        hue="class",
        order=order,
        hue_order=CLASSES,
        palette=CLASS_COLORS,
        ax=ax,
        width=0.7,
        showfliers=False,
        linewidth=0.8,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7.5)
    # Extra headroom above the 0-1 data range keeps the legend clear of every
    # box, rather than floating it over whichever field happens to be tallest.
    ax.set_ylim(-0.05, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("")
    ax.set_ylabel("1st-99th percentile scaled value")
    ax.legend(title="LCZ", ncol=4, fontsize=7, title_fontsize=7.5, frameon=True, loc="upper center")
    _apply_box_gradient(ax)
    _panel_label(ax, "B")


def _scatter(ax: plt.Axes, a: pd.DataFrame, x: str, y: str, label: str, xscale: float = 1.0, legend_loc: str = "best") -> None:
    for klass in CLASSES:
        sub = a[a["class"] == klass]
        xv, yv = sub[x] / xscale, sub[y]
        ax.scatter(xv, yv, s=8, color=CLASS_COLORS[klass], alpha=SCATTER_ALPHA, edgecolors="none")
        # A class-specific density contour picks the real cluster core out of
        # an alpha-blended, heavily overplotted cloud; a centroid marker gives
        # a single, legend-able anchor point per class.
        valid = xv.notna() & yv.notna()
        if valid.sum() > 30:
            sns.kdeplot(x=xv[valid], y=yv[valid], ax=ax, levels=[0.3], thresh=0.3, color=CLASS_COLORS[klass], linewidths=1.3, alpha=0.9)
        ax.scatter(
            xv.median(), yv.median(), s=85, marker="D", color=CLASS_COLORS[klass], edgecolors=PALETTE["ink"], linewidths=0.9, zorder=10, label=f"LCZ {klass}",
        )
    ax.set_xlabel(PROFILE_FIELDS.get(x, x) + (" (km/km2)" if xscale == 1000 else ""))
    ax.set_ylabel(PROFILE_FIELDS.get(y, y))
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
    ax.legend(fontsize=6.5, frameon=True, loc=legend_loc, title="LCZ (diamond = median)", title_fontsize=6.5, markerscale=0.9)
    _panel_label(ax, label)


CLASS_COLOR_NAME = {3: "teal", 6: "pink", 8: "orange", 9: "taupe"}


def _plot_access_lst_boxplots(fig: plt.Figure, subplotspec, a: pd.DataFrame) -> list[plt.Axes]:
    # Every point/box-shaped view of this pairing ran into the same wall:
    # 2SFCA access is zero-inflated (49.8% overall, up to 84.2% for LCZ 9),
    # so any view built around raw point density (hexbin) or that puts a
    # single-point summary in front of a mostly-empty cloud (strip +
    # point-range) still reads as sparse for the access-poor classes, and a
    # boxplot here would just duplicate panel F's chart type next door.
    # Empirical CDFs sidestep all of that: a step function is exactly as
    # "full" whether it is built from 8 points or 1,075, it needs no binning
    # or jitter, and stacking the three access tiers on one axis turns the
    # comparison into a direct question of which curve sits left (cooler) or
    # right (warmer) of the others. A two-sample Kolmogorov-Smirnov test
    # (no-access vs. above-median-access) adds a number to what the eye sees.
    access_col = "green_2sfca_800m_access_log1p"
    xlo = min(a["lst_c_median_mean"].min(), 34)
    xhi = max(a["lst_c_median_mean"].max(), 54)
    order = ["no acc.", "below-med.", "above-med."]
    sub_gs = subplotspec.subgridspec(2, 2, wspace=0.22, hspace=0.5)
    axes = []
    for idx, klass in enumerate(CLASSES):
        ax = fig.add_subplot(sub_gs[idx // 2, idx % 2])
        axes.append(ax)
        sub = a[a["class"] == klass].copy()
        nz_median = sub.loc[sub[access_col] > 0, access_col].median()
        tier_idx = np.where(sub[access_col] <= 0, 0, np.where(sub[access_col] <= nz_median, 1, 2))
        # Lines (unlike box/bar fills) need real contrast against white at
        # every tier, so blend from this class's own mid colour to its dark
        # variant rather than starting from the near-white "_light" swatch.
        base_rgb = np.array(mpl.colors.to_rgb(CLASS_COLORS[klass]))
        dark_rgb = np.array(mpl.colors.to_rgb(PALETTE[CLASS_DARK_KEY[klass]]))
        shades = [tuple(base_rgb * (1 - f) + dark_rgb * f) for f in (0.0, 0.5, 1.0)]
        tiers = {}
        for t in range(3):
            vals = sub.loc[tier_idx == t, "lst_c_median_mean"].dropna()
            tiers[t] = vals
            if len(vals) == 0:
                continue
            n = len(vals)
            sns.ecdfplot(vals, ax=ax, color=shades[t], linewidth=2.0, label=f"{order[t]} (n={n:,})")
        if len(tiers[0]) >= 2 and len(tiers[2]) >= 2:
            ks_stat, ks_p = ks_2samp(tiers[0], tiers[2])
            sig = "p<0.001" if ks_p < 0.001 else f"p={ks_p:.3f}"
            ax.text(
                0.98, 0.02, f"no vs.\nabove-med.:\nKS D={ks_stat:.2f}, {sig}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, color=PALETTE["muted"],
            )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("LST (C)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.5)
        ax.legend(fontsize=5.6, frameon=True, loc="upper left", handlelength=1.4)
        if idx % 2 == 0:
            ax.set_ylabel("Cumulative share", fontsize=7.5)
        else:
            ax.set_ylabel("")
        if idx == 0:
            _panel_label(ax, "E")
    return axes


def _plot_hotcool_boxplots(fig: plt.Figure, subplotspec, a: pd.DataFrame) -> list[plt.Axes]:
    fields = [
        "building_coverage_exact",
        "height_proxy_aw_mean_m",
        "dsm_elevation_m_std",
        "canopy_cover_gt2m_share",
        "s2_ndvi_mean",
        "road_density_exact_m_per_km2",
        "green_2sfca_800m_access_log1p",
        "lst_c_median_mean",
    ]
    # Percentile bounds computed once over the full retained subset, so a
    # given field is scaled identically across all four class mini-panels.
    scale = {f: np.nanpercentile(a[f], [1, 99]) for f in fields}
    order = [PROFILE_FIELDS[f] for f in fields]
    sub_gs = subplotspec.subgridspec(2, 2, wspace=0.2, hspace=0.85)
    axes = []
    for idx, klass in enumerate(CLASSES):
        ax = fig.add_subplot(sub_gs[idx // 2, idx % 2])
        axes.append(ax)
        sub = a[a["class"] == klass].copy()
        q1 = sub["lst_c_median_mean"].quantile(0.25)
        q4 = sub["lst_c_median_mean"].quantile(0.75)
        group = pd.Series(np.where(sub["lst_c_median_mean"] <= q1, "cool", np.where(sub["lst_c_median_mean"] >= q4, "hot", None)), index=sub.index)
        rows = []
        for field in fields:
            lo, hi = scale[field]
            scaled = ((sub[field] - lo) / (hi - lo)).clip(0, 1) if hi > lo else sub[field] * 0
            tmp = pd.DataFrame({"variable": PROFILE_FIELDS[field], "value": scaled, "group": group})
            rows.append(tmp[tmp["group"].notna()])
        long = pd.concat(rows, ignore_index=True)
        sns.boxplot(
            data=long,
            x="variable",
            y="value",
            hue="group",
            order=order,
            hue_order=["cool", "hot"],
            palette={"cool": PALETTE["teal"], "hot": PALETTE["pink"]},
            ax=ax,
            width=0.6,
            fliersize=0.6,
            linewidth=0.6,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Scaled value" if idx % 2 == 0 else "", fontsize=7.5)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=6.2)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=6.2, frameon=True, loc="upper right", ncol=1, title=None)
        else:
            ax.get_legend().remove()
        if idx == 0:
            _panel_label(ax, "F")
    return axes


def _plot_tile_context_bars(ax: plt.Axes, row: pd.Series, a: pd.DataFrame, klass: int) -> None:
    """Small horizontal grouped bar chart to the right of a texture tile: this example
    cell's own percentile-scaled value versus its class's median, for the
    same fields already reported in the tile's metrics box. Ties panel A's
    single representative cell to panel B's class-level distributions,
    showing at a glance whether the pictured example is a typical or an
    unusual member of its class."""
    class_vals = a.loc[a["class"] == klass]
    y = np.arange(len(CONTEXT_FIELDS))
    height = 0.36
    cell_vals, med_vals = [], []
    for field, _ in CONTEXT_FIELDS:
        lo, hi = np.nanpercentile(a[field], [1, 99])
        span = hi - lo if hi > lo else 1.0
        cell_vals.append(float(np.clip((float(row[field]) - lo) / span, 0, 1)))
        med_vals.append(float(np.clip((class_vals[field].median() - lo) / span, 0, 1)))
    color = CLASS_COLORS[klass]
    y_pos = y[::-1]
    ax.barh(y_pos + height / 2, cell_vals, height, color=color, edgecolor=PALETTE["ink"], linewidth=0.4, label="this cell", zorder=3)
    ax.barh(y_pos - height / 2, med_vals, height, facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"], linewidth=0.5, hatch="////", label="class median", zorder=3)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(CONTEXT_FIELDS) - 0.4)
    ax.set_xlim(0, 0.72)
    ax.set_xticks([0, 0.35, 0.7])
    ax.tick_params(axis="x", labelsize=5.8, length=2.0, pad=1.5)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.4, zorder=0)
    for i_feat, (_, lab) in enumerate(CONTEXT_FIELDS):
        y_val = y_pos[i_feat]
        ax.text(
            0.015, y_val, lab,
            va="center", ha="left", fontsize=6.0, fontweight="bold", color="#111111",
            zorder=10,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.8},
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)


def build_figure_4_morphology(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_4_morphology_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    a = _analysis_subset(df)
    tiles = _select_distinct_texture_tiles(df)
    tiles.to_csv(out_dir / "fig_4_texture_tiles.csv", index=False)
    cell_lookup = grid.set_index("grid_id")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(16.2, 15.5), constrained_layout=False)
    # Panel A (gs_top), Panel B (gs_b), Panels C/D (gs_cd) and Panels E/F
    # (gs_ef) each get their own GridSpec so every inter-row gap (A-B,
    # B-C/D, C/D-E/F) can be tightened independently -- a single GridSpec
    # can only apply one uniform hspace to all of its row gaps. Panel/row
    # heights are unchanged from the original layout throughout; only the
    # whitespace gaps between rows have been narrowed.
    gs_top = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.940, bottom=0.675, wspace=0.08)
    gs_b = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.650, bottom=0.488, wspace=0.08)
    gs_cd = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.451, bottom=0.2888, wspace=0.08)
    gs_ef = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.2477, bottom=0.0855, wspace=0.08)

    # The 4 texture tiles are sub-panels of a single panel A (small-multiple,
    # one tile per class) rather than 4 separately lettered top-level panels;
    # each keeps its own roman-numeral sub-index since the class-name title
    # already distinguishes it, and only the group as a whole is lettered.
    # Each tile is itself split into the texture map plus a small horizontal
    # grouped bar chart to its right (this cell vs. its class median, same fields as the
    # metrics box) so panel A is tied to panel B's distributions rather than
    # standing alone as an isolated snapshot.
    sub_labels = ["i", "ii", "iii", "iv"]
    ax_a0 = None
    tile_pairs = []
    for i, (_, row) in enumerate(tiles.iterrows()):
        tile_gs = gs_top[0, i].subgridspec(1, 2, width_ratios=[1.7, 0.85], wspace=0.04)
        ax = fig.add_subplot(tile_gs[0, 0])
        ax_bar = fig.add_subplot(tile_gs[0, 1])
        cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
        _plot_texture_tile(ax, root, cell, row, sub_labels[i])
        # _plot_texture_tile's equal-aspect map defaults to centering within
        # its (here, tall and narrow) box, leaving blank figure background
        # split evenly above and below the actual map -- with Panel B sitting
        # right underneath, that bottom half of the split reads as an
        # oversized A-B gap. Anchoring south instead pushes all of that
        # slack above the map (against the row's own title), so the map's
        # true bottom edge lines up with the row's own bottom edge.
        ax.set_anchor("S")
        klass = int(row["lcz_weak_label"])
        _plot_tile_context_bars(ax_bar, row, a, klass)
        tile_pairs.append((ax, ax_bar))
        if i == 0:
            ax_a0 = ax
            ax_bar.legend(fontsize=5.4, frameon=True, loc="upper right", handlelength=1.1, borderpad=0.3, labelspacing=0.25, bbox_to_anchor=(1.02, 1.18))

    # Force drawing to compute exact aspect-scaled bounding boxes for grid tiles,
    # then lock each bar chart's vertical span (y0, height) to match its grid tile exactly.
    fig.canvas.draw()
    for ax, ax_bar in tile_pairs:
        pos_map = ax.get_position()
        pos_bar = ax_bar.get_position()
        ax_bar.set_position([pos_bar.x0, pos_map.y0, pos_bar.width, pos_map.height])

    _panel_label(ax_a0, "A", y=1.06)

    # The former panel E (a standardized class-median heatmap) was removed: it
    # duplicated Table 4's raw class-median values under a second,
    # colour-coded presentation. Remaining panels are relettered B-F
    # (contiguous, matching Figure 2's convention) rather than leaving a gap.
    ax_e = fig.add_subplot(gs_b[0, 0:4])
    ax_f = fig.add_subplot(gs_cd[0, 0:2])
    ax_g = fig.add_subplot(gs_cd[0, 2:4])

    _plot_profile_boxplots(ax_e, a)
    _scatter(ax_f, a, "building_coverage_exact", "height_proxy_aw_mean_m", "C")
    ax_f.set_ylim(0, min(45, np.nanpercentile(a["height_proxy_aw_mean_m"], 99.5)))
    # D pairs the two strongest remaining class-discriminators not already
    # spent on C (eta^2: coverage 0.67, road density 0.60, NDVI 0.55, versus
    # surface SD/height proxy ~0.09 and 2SFCA/canopy ~0.06) -- building coverage
    # (impervious fraction) versus NDVI (vegetation fraction), the classic
    # LCZ built-vs-green trade-off axis. Earlier candidates for this slot
    # were both cut: road-density-vs-surface-SD (surface SD's weak eta^2 meant nearly
    # all the visible separation came from road density alone, already shown
    # univariately in panel B) and NDVI-vs-LST (duplicated Figure 8's own
    # dedicated NDVI-LST panel). Canopy cover is excluded from any scatter
    # axis in this figure: it is saturated near 1.0 for all four classes and
    # showed no visible class separation in every pairing tried.
    _scatter(ax_g, a, "building_coverage_exact", "s2_ndvi_mean", "D")

    # Inward trim requested for the two middle scatter panels: panel C loses
    # 0.5 cm from its right edge, while panel D loses 0.5 cm from its left edge.
    # Their data limits and vertical spans stay unchanged; only the central
    # gap between the two axes becomes wider.
    half_cm_frac = (0.5 / 2.54) / fig.get_figwidth()
    pos_f = ax_f.get_position()
    pos_g = ax_g.get_position()
    ax_f.set_position([pos_f.x0, pos_f.y0, pos_f.width - half_cm_frac, pos_f.height])
    ax_g.set_position([pos_g.x0 + half_cm_frac, pos_g.y0, pos_g.width - half_cm_frac, pos_g.height])
    # E and F are each a 2x2 small-multiple (one mini-panel per class) rather
    # than a single axes: a shared class-colour hexbin/boxplot grid reads more
    # naturally than one overloaded panel, and matches the small-multiples
    # idiom already used for the texture tiles and panel B. E pairs network
    # green-space access (2SFCA) with LST, tying to the paper's network-
    # evidence branch.
    e_axes = _plot_access_lst_boxplots(fig, gs_ef[0, 0:2], a)
    f_axes = _plot_hotcool_boxplots(fig, gs_ef[0, 2:4], a)

    # Apply the same 0.5 cm inward trim to the two bottom small-multiple
    # groups: panel E narrows from the right, panel F from the left. Only the
    # outer mini-panel boxes are shortened; their internal data and spacing
    # remain unchanged.
    e_right_axes = [ax for ax in e_axes if ax.get_position().x0 > np.mean([a.get_position().x0 for a in e_axes])]
    f_left_axes = [ax for ax in f_axes if ax.get_position().x0 < np.mean([a.get_position().x0 for a in f_axes])]
    for ax in e_right_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width - half_cm_frac, pos.height])
    for ax in f_left_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + half_cm_frac, pos.y0, pos.width - half_cm_frac, pos.height])

    _finalize_panel_labels(fig)

    # Panel A's aspect-locked tile maps leave real, uneven blank slack above
    # (anchor="S", see above) and the bottom row's tick labels eat into its
    # nominal margin -- rather than re-deriving the gridspec fractions to
    # chase this by hand, crop to the figure's own actual content extent
    # (full width kept as-is, only top/bottom tightened) with a small pad.
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig_w, fig_h = fig.get_size_inches()
    pad_in = 0.10
    crop_bbox = mpl.transforms.Bbox.from_extents(0.0, max(tight_bbox.y0 - pad_in, 0.0), fig_w, min(tight_bbox.y1 + pad_in, fig_h))

    png = fig_dir / "fig_4_class_profiles.png"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches=crop_bbox)
    plt.close(fig)

    qa = {
        "figure": "fig_4_class_profiles",
        "date": "2026-07-30",
        "panel_count": 6,
        "layout": "panel A is a top-row 1x4 small-multiple of 4 texture tiles (sub-indexed i-iv, one per class); full-width boxplot panel B; 2x2 scatter panels C,D; panels E and F are each a further 2x2 small-multiple, one mini-panel per class",
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/02_interim/akilli_sehir_fua.gpkg",
        ],
        "texture_tiles_csv": str(out_dir / "fig_4_texture_tiles.csv"),
        "png": str(png),
        "claim_boundary": "Weak-label morphology phenotype comparison; not audited local LCZ subtype proof.",
        "excluded_field_note": "morph_open_space_fragmentation_index is dropped from this figure entirely: its class median is exactly 0 for all four classes (51-100% zero share per class), making median/distribution-based comparisons degenerate everywhere it was tried (panel E, the hot/cool contrast, and a per-class hexbin against road density). Panel H now shows 2SFCA-vs-LST instead.",
        "revision_2026-07-30": "Original panel E (standardized class-median heatmap) removed: it duplicated Table 4's raw class medians under a second, colour-coded presentation ('dashboard' redundancy). The line/parallel-coordinate panel converted to grouped boxplots (1st-99th percentile scaled) per field, now lettered E. Remaining panels relettered F-I contiguously (no gap left for the retired E), matching Figure 2's panel-lettering convention. The scatter panel formerly at H (surface-SD-vs-canopy, no visible class separation because canopy is uniformly high across classes) is now G, using NDVI-vs-LST (eta^2 = 0.55 and 0.27, the strongest remaining discriminating fields not already used in F or H).",
        "revision_2026-07-30b": "Panels H and I were each converted from one overloaded axes to a 2x2 small-multiple, one mini-panel per class. H (road density vs fragmentation) was a near-empty scatter because fragmentation is exactly 0 for 51-100% of cells per class; it is now a per-class log-count hexbin (per-class '% fragmented' annotated in each mini-title), which shows the real sparse/skewed structure instead of an overplotted mass at y=0. I (within-class hot-vs-cool contrast) was a bare, unannotated colour heatmap that read as inconsistent with the rest of the figure; it is now a per-class paired boxplot (cool=teal, hot=pink, matching the palette's existing climate-response semantics) of the actual hot/cool subsets for all 8 fields, percentile-scaled per field across the full retained subset for cross-class comparability.",
        "revision_2026-07-30c": "Panel H's y-axis (fragmentation) was linear and looked empty: even non-zero cells are mostly tiny (per-class non-zero median 0.003-0.018 against a max near 0.9), so nearly all populated hexagons were crushed into the bottom of the panel. Switched to an arcsinh-transformed y-axis (linear near zero, log-like further out; scale=0.01), with hand-picked ticks at 0/0.01/0.05/0.2/0.9, which handles the exact zeros and spreads out the populated low range without discarding the long tail. Panels F and G were enhanced with a per-class 30%-density KDE contour (isolates each class's cluster core from the alpha-blended scatter), a per-class median-cell diamond marker doubling as the legend handle, an explicit legend (previously absent), and per-class Spearman rho annotated below each panel, quantifying the coverage-height and NDVI-LST relationships instead of leaving them purely visual.",
        "revision_2026-07-30d": "Panel H replaced again: road-density-vs-fragmentation was judged still weak (fragmentation is a structurally uninteresting field even when made visible). Now shows green-space network access (2SFCA) versus thermal response (LST), one per-class log-count hexbin, ties to the paper's network-evidence branch. 2SFCA is itself zero-inflated (49.8% zero overall; 8.6% for LCZ 3 up to 84.2% for LCZ 9), so the x-axis uses the same arcsinh treatment (scale=0.3, ticks at 0/0.3/1/3/8) for the same reason as the retired fragmentation panel; LST stays linear. morph_open_space_fragmentation_index is no longer plotted anywhere in Figure 4.",
        "revision_2026-07-30e": "Panel H's hexbin still looked empty for access-poor classes (LCZ 9 especially, 84.2% zero access, so nearly all its mass sits in one hexbin column regardless of axis scaling). Replaced the joint density view with a per-class 3-tier ordinal split of LST (no access / below-median access / above-median access, split computed per class among its own non-zero cells, n annotated per tier), which stays legible no matter how lopsided the underlying counts are. `_plot_access_lst_boxplots` (function renamed from `_plot_2sfca_lst_hexbins`) uses each class's own light/base/dark palette shade for the three tiers.",
        "revision_2026-07-30f": "Panel H's tiers were first drawn as boxplots, but panel I is already boxplots and sits directly beside it in the layout, so the two panels read as visually repetitive. Redrawn as a jittered strip (every cell plotted, so it can never look empty regardless of n, e.g. LCZ 9's below-median tier at n=8) plus a median-and-IQR point-range diamond marker, reusing the diamond-median motif from panels F/G instead of duplicating panel I's chart type.",
        "revision_2026-07-30g": "The strip+point-range still wasn't the right form. Replaced with an empirical CDF of LST per access tier (one step curve per tier, all three overlaid per class mini-panel), plus a two-sample Kolmogorov-Smirnov statistic (no-access vs. above-median-access) annotated per panel. An ECDF is a complete curve regardless of how few or many points built it, so it is the only form tried that never looks sparse for the access-poor classes, and it adds a formal test rather than only a visual impression. Line colours were changed from a light/base/dark palette blend to a base-to-dark blend only (skipping the near-white '_light' swatch), since thin ECDF lines (unlike box or bar fills) need real contrast against white at every tier. Also fixed: ks_2samp returned NaN for LCZ 9 because scipy does not drop NaNs internally (one LST value was missing in that class); both tier samples are now explicitly .dropna()'d before the test.",
        "revision_2026-07-30h": "Polish pass per user request: (1) removed the per-mini-panel titles in H and I entirely (previously 'LCZ X: Y% no access' / 'LCZ X (n=...)') -- all small-multiples in this figure use the same LCZ 3/6/8/9 top-left/top-right/bottom-left/bottom-right order as the texture tiles (a-d), so the class labels were redundant once a reader has seen that order once; (2) H's KS-statistic text moved from bottom-left to top-right, split across 3 lines; (3) H's per-tier legend standardized to upper-left (was alternating loc by class) with abbreviated tier labels ('no acc.' / 'below-med.' / 'above-med.') so it fits; (4) panel E now draws a faint (alpha=0.18), sparse (120 cells per field-class combination, subsampled) jittered strip behind the boxplots via sns.stripplot before sns.boxplot, giving a raincloud-lite texture; boxplot fliers were turned off (showfliers=False) to avoid double-plotting the same outlier points.",
        "revision_2026-07-30i": "Second polish pass per user request: removed the remaining descriptive titles from panels E ('Distribution of key morphology...'), F ('Coverage versus height proxy'), G ('Vegetation greenness versus thermal response'), and the group-level titles above H ('Network green-space access...') and I ('Within-class hot...') -- axis labels and the caption already state what each panel plots, so these were redundant. `_scatter()`'s now-unused `title` parameter was removed (both call sites updated). H's KS-statistic annotation moved again, from top-right to bottom-right.",
        "revision_2026-07-30j": "Removed the figure-wide footer note ('Texture tiles are deterministic class-median cells among confidence >=0.80 eligible-core building-bearing cells. Scatter and heatmap panels use the 5,339-cell weak-label diagnostic subset.') per user request. This methodological detail is not lost: it is already stated independently in the methodology and study-area-data sections of the manuscript, so the figure-level repetition was redundant rather than the only place this fact appeared.",
        "revision_2026-08-01": "Panel relettering per user request: the 4 texture tiles (formerly separately lettered A-D) are now sub-panels of a single panel A (roman-numeral sub-indices i-iv; each tile's own class-name title already distinguishes it), and the remaining panels shift down contiguously: former E->B (profile boxplots), F->C (coverage-vs-height scatter), G->D (NDVI-vs-LST scatter), H->E (2SFCA-access ECDF small-multiple), I->F (hot/cool boxplot small-multiple). Panel B's boxplot fills were also redrawn as a horizontal gradient (base class colour at each box's edges, lightened toward its own centre) for a subtler, glossier fill than the former flat colour; implemented by replacing each seaborn PathPatch's facecolor with a clipped gradient image rather than a flat colour.",
        "revision_2026-08-01b": "Panel A was judged too shallow as a standalone snapshot (single example cell per class, no link to the rest of the figure). Each of the 4 tiles now sits above a small vertical grouped bar chart (_plot_tile_context_bars): for the same 5 fields already in the tile's metrics box (coverage, height proxy, surface SD, NDVI, LST), two grouped bars per field show this example cell's own percentile-scaled value versus its class's median, using the identical [1,99] percentile scaling as panel B so the two panels are directly comparable. This ties the single 'representative' cell back to panel B's class-level distributions and shows at a glance whether the pictured example is typical or an outlier within its class. Each tile's gridspec cell was split into a 2-row subgridspec (texture map + bar chart, height ratio 2.55:1); the top-level figure height and row-0 height ratio were both increased to keep the texture maps from shrinking. The panel-A group label was moved from an axes-fraction position (transAxes) to a figure-fraction position (transFigure, still attached to the first tile's own ax.texts so _finalize_panel_labels still converts it) because the tile axes' own fraction shrank once the bar chart was nested below it, which had caused the label to collide with the tile's class-name title.",
        "revision_2026-08-02": "Cross-figure redundancy pass (Figure 4 reviewed against its predecessor Figure 1 and successor Figure 8/climate-validation, without editing either of those figures' own scripts). Two duplications were found and fixed entirely within this script: (1) Panel A's four texture tiles were, cell for cell, identical to Figure 1's four representative tiles -- both scripts called the same deterministic `_select_texture_tiles` on the same dataframe, so they always picked the same closest-to-median grid cell per class. Added `_select_distinct_texture_tiles`, which reuses Figure 1's exact median/SD-based ranking (so distances stay comparable) but excludes Figure 1's own four grid_ids from the final pick, yielding each class's second-closest-to-median eligible cell instead. Verified via `fig_4_texture_tiles.csv`: all four grid_ids now differ from Figure 1's selected set. (2) Panel D plotted NDVI versus LST, which Figure 8 (climate validation) already covers as its own dedicated NDVI-LST trend panel -- the same class-level claim shown twice across two main-text figures. Initially replaced with road density versus surface SD (see revision_2026-08-02b: this pairing was then cut outright).",
        "revision_2026-08-02b": "Panel-necessity review, per user request, of the resulting 6-panel figure. Panel D (road density vs. surface SD, added in revision_2026-08-02) was judged low-value and cut: surface SD is a weak class-discriminator (eta^2=0.09), so nearly all of the panel's visible class separation came from road density alone, which is already shown as a univariate distribution in panel B -- the bivariate view added little beyond what B already covers. Panel C (coverage vs. height proxy) briefly spanned the full row in D's place; figure was 5 panels (A-E).",
        "revision_2026-08-02c": "Per user request, panel D was restored (figure back to 6 panels, A-F) but filled with a different pairing: building coverage vs. NDVI, the two strongest remaining class-discriminators not already spent on panel C (eta^2: coverage 0.67, road density 0.60, NDVI 0.55, vs. surface SD/height proxy ~0.09 and 2SFCA/canopy ~0.06) -- the classic LCZ built-fraction-vs-green-fraction trade-off axis, not used as a pairing anywhere else in the paper. Canopy cover remains excluded from every scatter axis tried in this figure (saturated near 1.0 for all four classes, no visible class separation).",
        "revision_2026-08-03": "Panel C and panel D were each narrowed by 1 cm toward the central gap per user request: C trims from the right edge, D trims from the left edge. Data limits and vertical spans are unchanged.",
        "revision_2026-08-03b": "Reduced the C/D inward trim from 1 cm to 0.5 cm. Applied the same 0.5 cm inward trim to the bottom small-multiple groups: panel E from its right edge and panel F from its left edge, preserving the mini-panel data limits and internal spacing.",
    }
    (out_dir / "fig_4_morphology_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_4_morphology_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_4_morphology(Path(".")), indent=2))
