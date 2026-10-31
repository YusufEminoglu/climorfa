"""CLIMORFA house figure style, extracted from the ORIGINAL paper's figure scripts.

Purpose: let the street-scale paper's figures reproduce the original paper's
look exactly.  Nothing here is a new design.  Every constant carries a
``# src: <file>:<line>`` pointer to the original script that defines it, and
``verify_against_originals()`` re-imports those scripts and asserts that the
copies below have not drifted.

HOW THE ORIGINAL FIGURES WERE REALLY RENDERED (important, not obvious from
the per-figure scripts alone)
---------------------------------------------------------------------------
All main figures are produced by ``scripts/generate_manuscript_evidence.py``
in ONE Python process.  The matplotlib state that every figure inherits is:

1. ``sns.set_theme(style='whitegrid', context='paper', font_scale=1.05)``
   (generate_manuscript_evidence.py:36, executed at import time).
2. ``build_figure_1`` runs first and PERMANENTLY updates rcParams
   (axes.edgecolor / axes.labelcolor / xtick.color / ytick.color,
   build_figure_1_texture_atlas.py:369-377); later figures inherit this.
3. Each figure then does ``rcParams.update({"font.family": "DejaVu Sans"})``.

So the grey light frames, light grid, muted-grey tick labels, 9.24 pt tick
labels, 10.08 pt axis labels, rounded line caps and white patch edges seen in
the originals come from steps 1-2, NOT from the individual figure scripts.
``apply_style()`` replays exactly this sequence.

Usage::

    from scripts.street.figures import climorfa_style as S
    S.apply_style()
    fig = plt.figure(figsize=(S.PRESET_WIDE_13.fig_w, 8))
    ...
    S.save_figure(fig, "out.png")
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Make the original scripts importable (they import each other as top-level
# modules) so that helpers can be re-used instead of re-implemented.
# --------------------------------------------------------------------------
ORIG_SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # <repo>/scripts
if str(ORIG_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ORIG_SCRIPTS_DIR))

import build_figure_1_texture_atlas as _f1  # noqa: E402
import build_figure_2_methodology_multipanel as _f2  # noqa: E402
import build_figure_4_morphology_multipanel as _f4  # noqa: E402
import build_figure_9_climate_validation_multipanel as _f9  # noqa: E402

# ==========================================================================
# 1. rcParams
# ==========================================================================
SEABORN_THEME = dict(style="whitegrid", context="paper", font_scale=1.05)  # src: generate_manuscript_evidence.py:36
FIG1_RC_UPDATE = {  # src: build_figure_1_texture_atlas.py:369-377 (persists for all later figures)
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#E8E4E0",   # PALETTE["grid"]
    "axes.labelcolor": "#252526",  # PALETTE["ink"]
    "xtick.color": "#6E6A67",      # PALETTE["muted"]
    "ytick.color": "#6E6A67",
}
FONT_FAMILY = "DejaVu Sans"  # src: every build_figure_*.py: mpl.rcParams.update({"font.family": "DejaVu Sans"})

# The *resolved* values (matplotlib 3.11 / seaborn 0.13.2) after the replay,
# for documentation and for the self-test.  Only keys that differ from
# matplotlib defaults are listed.  axes.prop_cycle = seaborn "deep".
RESOLVED_RC = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#E8E4E0",
    "axes.grid": True,
    "axes.labelcolor": "#252526",
    "axes.labelsize": 10.08,
    "axes.linewidth": 1.0,
    "axes.titlesize": 10.08,
    "font.family": ["DejaVu Sans"],
    "font.size": 10.08,
    "grid.color": ".8",
    "image.cmap": "rocket",
    "legend.fontsize": 9.24,
    "legend.title_fontsize": 10.08,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.8,
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "patch.linewidth": 0.8,
    "text.color": ".15",
    "xtick.bottom": False,
    "xtick.color": "#6E6A67",
    "xtick.labelsize": 9.24,
    "xtick.major.size": 4.8,
    "xtick.major.width": 1.0,
    "ytick.color": "#6E6A67",
    "ytick.labelsize": 9.24,
    "ytick.left": False,
    "ytick.major.size": 4.8,
    "ytick.major.width": 1.0,
}

DPI = 300                 # src: every fig.savefig(..., dpi=300, facecolor="white")
FACECOLOR = "white"
CROP_PAD_IN = 0.10        # src: build_figure_2_methodology_multipanel.py:460, build_figure_4...:541


def apply_style() -> None:
    """Replay the exact rcParams sequence the original figures were drawn under."""
    sns.set_theme(**SEABORN_THEME)
    mpl.rcParams.update(FIG1_RC_UPDATE)
    mpl.rcParams.update({"font.family": FONT_FAMILY})


def verify_rc() -> list[str]:
    """Return a list of mismatches between live rcParams and RESOLVED_RC (empty = OK)."""
    bad = []
    for k, v in RESOLVED_RC.items():
        live = mpl.rcParams[k]
        ok = np.isclose(live, v) if isinstance(v, float) else (live == v)
        if not ok:
            bad.append(f"{k}: live={live!r} expected={v!r}")
    return bad


# ==========================================================================
# 2. Palettes
# ==========================================================================
PALETTE = dict(_f1.PALETTE)  # src: build_figure_1_texture_atlas.py:29-46
# roots: teal #87D8CD, pink #F978A5, orange #F4AB55, taupe #ACA49C
# dark:  teal #2F756E, pink #9E365F, orange #985D18, taupe #5F5A56
# light: teal #E7F7F5, pink #FEE8EF, orange #FDEEDC, taupe #F0EEEC
# neutrals: surface #FAF9F7, grid #E8E4E0, ink #252526, muted #6E6A67
CLASSES = list(_f1.CLASSES)                  # [3, 6, 8, 9]   src: build_figure_1...:16
CLASS_NAMES = dict(_f1.CLASS_NAMES)          # src: :17-22
CLASS_SHORT = dict(_f1.CLASS_SHORT)          # src: :23-28
CLASS_COLORS = dict(_f1.CLASS_COLORS)        # src: :47-52  3 teal / 6 pink / 8 orange / 9 taupe
CLASS_DARK_KEY = dict(_f4.CLASS_DARK_KEY)    # src: build_figure_4...:28
CLASS_COLOR_NAME = {3: "teal", 6: "pink", 8: "orange", 9: "taupe"}  # src: build_figure_4...:245

# Neutral furniture colours (hard-coded hex in the originals)
PANEL_EDGE = "#555555"      # panel-letter box border, study outline (fig2/4/8/9/10)
PANEL_TEXT = "#222222"      # panel-letter / scale-bar text
NORTH_ARROW = "#2C3E50"     # north arrow + "N"   src: build_figure_9...:179,193
MAP_BG = "#fcfcfc"          # set_facecolor on map axes (invisible after axis("off")) src: fig9:240
TILE_BG_FIG9 = "#f9fafb"    # compact tile background   src: fig9:485
SPINE_GREY = "#a0a0a0"      # FUR-median dotted baseline  src: fig9:352
FOLD_NEUTRALS = ("#5C6370", "#A39E98")  # Fig 2 fold slate / muted grey  src: fig2:342,345

# Categorical palettes beyond the LCZ classes
FOLD_COLORS = [  # src: build_figure_2_methodology_multipanel.py:341-347
    "#5C6370", PALETTE["orange"], PALETTE["pink"], "#A39E98", PALETTE["teal"],
]
FAMILY_COLORS_FIG2 = dict(_f2.FAMILY_COLORS)  # src: fig2:29-36 (Morphology/Vegetation/Green-blue/Network/Climate/Gate)
FAMILY_COLORS_FIG6 = {  # src: build_figure_6_explainability_multipanel.py:38-47
    "Remote semantic": PALETTE["pink"],
    "2D form": PALETTE["teal"],
    "3D / Surface": PALETTE["orange"],
    "Vegetation": "#4FB286",
    "Street / network": PALETTE["taupe"],
    "Green access": "#61A0AF",
    "Context": "#C07C7C",
    "Other": "#B8B8B8",
}
HOT_COOL_COLORS = {"cool": PALETTE["teal"], "hot": PALETTE["pink"]}  # src: fig4:347 (climate semantics)
THRESHOLD_COLORS = {"400 m": PALETTE["teal"], "800 m": PALETTE["orange"], "1200 m": PALETTE["pink"]}  # src: fig2:214-216


def class_line_shades(klass: int) -> list[tuple]:
    """Three shades (base -> dark) of a class colour, as used for the ECDF tiers.
    src: build_figure_4_morphology_multipanel.py:276-278 (skips the near-white _light swatch)."""
    base = np.array(mpl.colors.to_rgb(CLASS_COLORS[klass]))
    dark = np.array(mpl.colors.to_rgb(PALETTE[CLASS_DARK_KEY[klass]]))
    return [tuple(base * (1 - f) + dark * f) for f in (0.0, 0.5, 1.0)]


# ==========================================================================
# 3. Continuous colormaps  (colour lists copied verbatim; verified by
#    verify_against_originals)
# ==========================================================================
def _cmap(name: str, colors: list[str]) -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(name, colors)


# CANONICAL LST ramp (Fig 9 panel a, also Fig 10 MAP_CMAP): teal -> lavender -> purple
LST_TEAL_PURPLE_COLORS = ["#5ec4b6", "#81cecb", "#b6cad8", "#bfa5d7", "#9e6ebf", "#7b3f9e"]
# src: build_figure_9_climate_validation_multipanel.py:223-226 ; build_figure_10...:25-28
LST_TEAL_PURPLE_CMAP = _cmap("soft_teal_purple_lst", LST_TEAL_PURPLE_COLORS)
# vmin/vmax convention for the LST map: 2nd / 98th percentile of the retained cells
# (LCZ in {3,6,8,9} and weak-label confidence >= 0.60).   src: fig9:248-249
LST_QUANTILES = (0.02, 0.98)
LCZ_RETAIN_CONFIDENCE = 0.60  # src: fig9:245

# Earlier / secondary ramps (kept so any original figure can be matched)
LST_CMAP_FIG1 = _cmap("climorfa_lst", [PALETTE["taupe_light"], PALETTE["taupe"], PALETTE["orange"], PALETTE["pink"]])  # fig1:82-85 (no vmin/vmax: data range)
CONF_CMAP = _cmap("climorfa_confidence", [PALETTE["taupe_light"], PALETTE["taupe"], PALETTE["teal"]])  # fig1:86-89, vmin .25 vmax 1.0 (fig1:247-248)
CONF_VMIN, CONF_VMAX = 0.25, 1.0
LST_CMAP_FIG3 = _cmap("fig3_lst", [PALETTE["teal"], PALETTE["taupe_light"], PALETTE["orange"], PALETTE["pink_dark"]])  # fig3:25-27
LST_PURPLE_ORANGE_COLORS = ["#240038", "#4c126b", "#7b277d", "#b33d79", "#e05c5c", "#f48642", "#f9b353", "#fef1a0"]
LST_CMAP_S1 = _cmap("climorfa_lst_purple_orange", LST_PURPLE_ORANGE_COLORS)  # supp. texture atlas :30-42 (30 m LST chips)
CANOPY_10M_COLORS = ["#A8E6CF", "#56AB2F", "#1D976C", "#0B6623", "#053E17"]
CANOPY_CMAP_10M = _cmap("canopy_10m", CANOPY_10M_COLORS)  # fig9:74-77; used with vmin=2.0, vmax=15.0, alpha=0.75, masked <2 or >100
CANOPY_VMIN, CANOPY_VMAX, CANOPY_ALPHA = 2.0, 15.0, 0.75  # fig9:498-500
CANOPY_CMAP_S1 = _cmap("climorfa_canopy_green", ["#f7fcf5", "#d5efda", "#8fd496", "#32a051", "#055222"])  # supp. atlas :43-46
ACCESS_CMAP = _cmap("climorfa_2sfca", [PALETTE["taupe_light"], PALETTE["orange"], PALETTE["pink"]])  # fig8:35, vmin=0, vmax=98th pct (fig8:411)
DELTA_CMAP = _cmap("climorfa_2sfca_delta", [PALETTE["teal_dark"], PALETTE["taupe_light"], PALETTE["pink"]])  # fig8:36, symmetric +-vmax (98th pct |delta|, min 0.1)
WARNING_CMAP = _cmap("purple_teal_clean", ["#f7f4f9", "#e7e1ef", "#d4b9da", "#998ec3", "#5e3c99", "#35978f", "#016c59", "#003c30"])  # fig7:27-29, vmin 0 vmax 0.65
WARNING_VMIN, WARNING_VMAX = 0.0, 0.65
TRANSFER_CMAP = _cmap("climorfa_transfer", [PALETTE["pink"], PALETTE["orange"], PALETTE["teal"]])  # fig10:22-24
CONTEXT_CMAP = _cmap("climorfa_context", [PALETTE["taupe"], PALETTE["teal"]])  # fig10:29-31
TRANSFER_VMIN, TRANSFER_VMAX = 0.40, 0.95  # fig10:32-33 (macro-F1 choropleth uses LST_TEAL_PURPLE_CMAP)


def lst_value_range(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """vmin/vmax of the Fig 9 LST map: 2nd and 98th percentiles of the retained cells."""
    s = pd.Series(values)
    return float(s.quantile(LST_QUANTILES[0])), float(s.quantile(LST_QUANTILES[1]))


# ==========================================================================
# 4. Figure-width presets  (inches, authoring size; manuscript includes every
#    figure at \textwidth = 384 pt = 5.33 in, i.e. scaled to ~0.33-0.41)
# ==========================================================================
@dataclass(frozen=True)
class WidthPreset:
    name: str
    panel_w: float
    left_margin: float
    right_margin: float
    top_margin: float
    bottom_margin: float
    source: str

    @property
    def fig_w(self) -> float:
        return self.left_margin + self.panel_w + self.right_margin


# Fig 7 / Fig 9: one 12 in panel column, 13.0 in figure
PRESET_WIDE_13 = WidthPreset("wide_13", 12.0, 0.50, 0.50, 0.15, 0.15, "fig9:588-606 (gap_ab=0.05 in, tile_gap=0.12 in)")
# Fig 2 / Fig 8: 4 x 3.6 in panels + 3 x 0.35 in gaps = 15.45 in, margins 0.60
PRESET_WIDE_FIG2 = WidthPreset("wide_fig2", 15.45, 0.60, 0.60, 0.55, 0.45, "fig2:405-424 (panel_w=3.60, h_gap=0.35, v_gap=0.50)")
# Fixed figsize presets used by gridspec-style figures (full width, no margins object)
FIGSIZES = {
    "fig1_atlas": (15.8, 11.6),      # fig1:378
    "fig4_profiles": (16.2, 15.5),   # fig4:422
    "fig5_validation": (15.6, 12.8),  # fig5:473
    "fig6_importance": (11.6, 6.6),  # fig6:233
    "fig10_lodo": (16.5, None),      # fig10:353 (height computed)
    "fig11_audit": (15.0, None),     # fig11:162
    "supp_atlas": (15.5, 9.8),       # supp. texture atlas:243
}
TEXTWIDTH_IN = 384.0 / 72.27  # \textwidth of main.tex (elsarticle review,1p), 5.31 in


def inch_axes(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    """fig.add_axes() specified in inches, the composition idiom of Figs 2/8/9
    (src: fig9:616 ``fig.add_axes([left/fig_w, b/fig_h, w/fig_w, h/fig_h])``)."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


# ==========================================================================
# 5. Panel labels
# ==========================================================================
PANEL_LABEL_BBOX = {  # src: fig2:57, fig4:113, fig9:99, fig10 (all identical)
    "facecolor": "white", "edgecolor": PANEL_EDGE, "boxstyle": "round,pad=0.18", "linewidth": 0.8, "alpha": 0.95,
}


def panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 15.0,   # fig2/fig9/fig10 = 15 ; fig4 uses 14.0
    x: float = 0.005,
    y: float = 1.02,          # above-axes convention (Fig 2). Maps with inside label: y=0.98, va="top" (fig9)
    va: str = "bottom",
) -> None:
    """Boxed bold letter above the axes with neutral grey border (Figure 2 convention).
    Pass a bare capital ("A"); call ``finalize_panel_labels(fig)`` once at the end to
    turn it into "(a)", exactly like the originals.  Sub-indices such as "(b.1)" are
    written literally (see ``subindex_badge``)."""
    ax.text(
        x, y, label, transform=ax.transAxes, ha="left", va=va, fontsize=fontsize,
        fontweight="bold", color=PANEL_TEXT, bbox=dict(PANEL_LABEL_BBOX), zorder=30, clip_on=False,
    )


_LETTERS = set("ABCDEFGHIJKLM")  # 'N' deliberately excluded: it is the north-arrow glyph


def finalize_panel_labels(fig: plt.Figure) -> None:
    """'A' -> '(a)' (fig2:63-76; visits inset child axes too)."""
    all_axes = list(fig.axes)
    for ax in fig.axes:
        all_axes.extend(ax.child_axes)
    for ax in all_axes:
        for child in ax.texts:
            if child.get_text() in _LETTERS:
                child.set_text(f"({child.get_text().lower()})")


def subpanel_label(ax: plt.Axes, label: str, edgecolor: str = PANEL_EDGE) -> None:
    """Inside-corner roman-numeral tag of small multiples (Fig 1/4 tiles). Re-uses the original."""
    _f1._panel_label(ax, label, edgecolor)  # src: fig1:127-146 (fontsize 12, top-left inside, coloured border)


def subindex_badge(ax: plt.Axes, text: str, x: float = 0.03, y: float = 0.92) -> None:
    """'(b.1) r_s = -0.67***' badge: fontsize 7.2 bold, taupe border (fig9:375-381)."""
    ax.text(
        x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.2, fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.18", "alpha": 0.90},
        zorder=10,
    )


# ==========================================================================
# 6. Map furniture
# ==========================================================================
def set_map_extent(ax: plt.Axes, bounds, pad: float = 0.015, exact: bool = False) -> None:
    """Equal aspect, axes off.  pad=1.5 % of the span (fig1:158-165, fig2:112-119, fig3);
    ``exact=True`` = no padding (fig9:198-203, bounds from the study-area polygon)."""
    xmin, ymin, xmax, ymax = bounds
    p = 0.0 if exact else pad
    ax.set_xlim(xmin - (xmax - xmin) * p, xmax + (xmax - xmin) * p)
    ax.set_ylim(ymin - (ymax - ymin) * p, ymax + (ymax - ymin) * p)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_study_outline(ax: plt.Axes, boundary, variant: str = "ink") -> None:
    """Study-area outline.  'ink' = Fig 9/1 (#252526, lw 0.85/0.8, zorder 15);
    'grey' = Fig 2/8 FUR perimeter (#555555, lw 0.5-0.6, zorder 10)."""
    if variant == "ink":
        boundary.boundary.plot(ax=ax, color=PALETTE["ink"], linewidth=0.85, linestyle="-", zorder=15)
    else:
        boundary.boundary.plot(ax=ax, color=PANEL_EDGE, linewidth=0.5, linestyle="-", zorder=10)


def draw_land_cells(ax: plt.Axes, grid) -> None:
    """Land / unclassified cells: taupe_light, no edge.  'Sea' is simply the white
    figure background (the map axes are switched off), nothing is drawn for water."""
    grid.plot(ax=ax, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)


@dataclass(frozen=True)
class ScaleNorthSpec:
    label_fs: float
    north_fs: float
    arrow_w_frac: float
    arrow_base_k: float   # north-arrow base = bar_y0 + sb_h * k
    label_k: float        # tick-label height above bar = sb_h * k


SCALE_NORTH_FIG9 = ScaleNorthSpec(7.5, 9.5, 0.012, 4.4, 1.6)  # fig9:114-195 (full-width LST map)
SCALE_NORTH_FIG2 = ScaleNorthSpec(6.8, 8.5, 0.014, 4.2, 1.5)  # fig2:241-327 (fold map, 4-column figure)


def add_scale_north(
    ax: plt.Axes,
    base_loc: tuple[float, float] = (0.09, 0.59),
    scale_len_km: float = 5.0,
    spec: ScaleNorthSpec = SCALE_NORTH_FIG9,
) -> None:
    """Segmented black/white 2-segment scale bar + polygon north arrow above it.
    base_loc is in axes fractions of the CURRENT xlim/ylim, so call AFTER set_map_extent.
    Geometry identical to fig9:114-195."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    sb_x0 = xlim[0] + base_loc[0] * dx
    sb_y0 = ylim[0] + base_loc[1] * dy
    slen = scale_len_km * 1000.0
    sb_h = dy * 0.009
    for i in range(2):
        seg_x = sb_x0 + i * (slen / 2.0)
        ax.add_patch(mpatches.Rectangle((seg_x, sb_y0), slen / 2.0, sb_h,
                                        facecolor="black" if i == 0 else "white",
                                        edgecolor="black", linewidth=0.7, zorder=25))
        ax.text(seg_x, sb_y0 + sb_h * spec.label_k, f"{i * scale_len_km / 2.0:g}", ha="center", va="bottom",
                fontsize=spec.label_fs, fontweight="bold", color=PANEL_TEXT, zorder=25)
    ax.text(sb_x0 + slen, sb_y0 + sb_h * spec.label_k, f"{int(scale_len_km)} km", ha="center", va="bottom",
            fontsize=spec.label_fs, fontweight="bold", color=PANEL_TEXT, zorder=25)
    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * spec.arrow_base_k
    na_w, na_h = dx * spec.arrow_w_frac, dy * 0.040
    ax.add_patch(mpatches.Polygon(
        [[cx, na_base + na_h], [cx + na_w / 2.0, na_base], [cx, na_base + na_h * 0.25], [cx - na_w / 2.0, na_base]],
        facecolor=NORTH_ARROW, edgecolor="white", linewidth=0.9, zorder=25))
    ax.text(cx, na_base + na_h + dy * 0.006, "N", ha="center", va="bottom", fontsize=spec.north_fs,
            fontweight="bold", color=NORTH_ARROW, zorder=25)


def add_colorbar(
    fig: plt.Figure,
    bounds_frac: list[float],
    cmap,
    vmin: float,
    vmax: float,
    label: str,
    orientation: str = "horizontal",
):
    """Stand-alone colourbar axes (never an inset) with the originals' font sizes.
    horizontal (fig9:268-275): tick labelsize 8.0, label 8.5 bold ink, labelpad 3.0.
    vertical   (fig8:113-116): tick labelsize 6, label 6.4, labelpad 3 (placed outside, left of map)."""
    cax = fig.add_axes(bounds_frac)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation=orientation)
    if orientation == "horizontal":
        cb.ax.tick_params(labelsize=8.0)
        cb.set_label(label, fontsize=8.5, fontweight="bold", color=PALETTE["ink"], labelpad=3.0)
    else:
        cb.ax.tick_params(labelsize=6)
        cb.set_label(label, fontsize=6.4, labelpad=3)
    return cb


def horizontal_cbar_bounds_in_water(panel_left_in: float, panel_bottom_in: float,
                                    panel_w: float, panel_h: float, fig_w: float, fig_h: float) -> list[float]:
    """fig9:619-624: 16 % of panel width, 1.5 % of panel height, starting 5 % in, 55 % up."""
    return [
        (panel_left_in + 0.05 * panel_w) / fig_w,
        (panel_bottom_in + 0.55 * panel_h) / fig_h,
        (0.16 * panel_w) / fig_w,
        0.015 * (panel_h / fig_h),
    ]


def info_card(ax: plt.Axes, text: str, x: float = 0.02, y: float = 0.49) -> None:
    """Computed read-out card (fig9:278-291): 8.2 pt bold ink, white box, taupe border,
    round pad 0.30, alpha 0.95, anchored top-left in axes fraction."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, fontweight="bold",
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.30", "alpha": 0.95},
            zorder=25)


def stat_note(ax: plt.Axes, text: str, x: float = 0.025, y: float = 0.03) -> None:
    """Small bottom-left counts box on maps (fig1:199-209): 7.5 pt, ink, white, grid border, pad 3."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 3})


def tile_tag(ax: plt.Axes, text: str, color: str, x: float = 0.05, y: float = 0.95) -> None:
    """Class/thermal tag on a texture tile (fig9:520-532): 6.6 pt bold coloured text in a box of the same colour."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=6.6, fontweight="bold", color=color,
            bbox={"facecolor": "white", "edgecolor": color, "boxstyle": "round,pad=0.16", "linewidth": 0.7, "alpha": 0.95},
            zorder=30)


def tile_readout(ax: plt.Axes, text: str, x: float = 0.05, y: float = 0.05) -> None:
    """LST read-out at the tile's lower-left (fig9:533-545)."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, fontweight="bold",
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.14", "linewidth": 0.6, "alpha": 0.92},
            zorder=30)


# ---- texture tiles / insets: re-used straight from the originals --------------------
plot_texture_tile = _f1._plot_texture_tile                    # fig1:288-342 (full tile: title + 3-line metrics box)
plot_compact_thermal_tile = _f9._plot_compact_thermal_tile    # fig9:461-545 (1.4 in tile with 10 m canopy raster)
plot_tile_context_bars = _f4._plot_tile_context_bars          # fig4:367-404 (cell vs class-median bars next to a tile)
clip_to_cell = _f1._clip_to_cell
safe_read_layer = _f1._safe_read_layer
style_inset = _f2._style_inset                                # fig2:79-86
add_inset_backing = _f2._add_inset_backing                    # fig2:89-109
apply_box_gradient = _f4._apply_box_gradient                  # fig4:119-145
add_scale_north_above_legend_fig2 = _f2._add_scale_north_above_legend  # direct re-export of the Fig 2 original

# Compact-tile geometry constants of Fig 9 (fig9:592-599)
TILE_STRIP = dict(n_tiles=8, tile_gap=0.12, tile_h=1.55, gap_trend_tiles=0.30, trend_h=2.45)

# ==========================================================================
# 7. Legend idioms (kwargs copied from the originals)
# ==========================================================================
LEGEND_BOXPLOT = dict(title="LCZ", ncol=4, fontsize=7, title_fontsize=7.5, frameon=True, loc="upper center")  # fig4:219
LEGEND_SCATTER = dict(fontsize=6.5, frameon=True, title="LCZ (diamond = median)", title_fontsize=6.5, markerscale=0.9)  # fig4:241
LEGEND_CHIPS_NOFRAME = dict(fontsize=6.5, frameon=False, handleheight=2.0)  # fig2:142 (map legend: patches, no frame)
LEGEND_LINES_SMALL = dict(fontsize=6.2, frameon=True, facecolor="white", edgecolor="none", borderpad=0.25)  # fig9:406


def class_line_legend_handles(overall_median: float | None = None) -> list[Line2D]:
    h = [Line2D([0], [0], color=CLASS_COLORS[k], linewidth=2.2, label=f"LCZ {k}") for k in CLASSES]
    if overall_median is not None:
        h.append(Line2D([0], [0], color=SPINE_GREY, linestyle=":", linewidth=1.0, label=f"FUR Med. ({overall_median:.1f}°C)"))
    return h


# ==========================================================================
# 8. Chart idioms
# ==========================================================================
def class_boxplot_panel(
    ax: plt.Axes,
    long: pd.DataFrame,
    order: list[str],
    ylabel: str = "1st-99th percentile scaled value",
    strip_n: int = 120,
    strip_seed: int = 20260730,
    class_col: str = "class",
    hue_order: list[int] | None = None,
    letter: str | None = "B",
) -> None:
    """Grouped class boxplots: faint sub-sampled strip UNDER 0.7-wide boxes with gradient fills,
    scaled 0-1 per field.  Constants copied from fig4:165-221 (``_plot_profile_boxplots``).
    ``long`` columns: variable, value, <class_col>."""
    hue_order = hue_order or CLASSES
    rng = np.random.default_rng(strip_seed)
    sampled = pd.concat(
        [grp.sample(min(strip_n, len(grp)), random_state=int(rng.integers(1 << 31)))
         for _, grp in long.groupby(["variable", class_col])],
        ignore_index=True,
    )
    sns.stripplot(data=sampled, x="variable", y="value", hue=class_col, order=order, hue_order=hue_order,
                  palette=CLASS_COLORS, dodge=True, size=2.2, alpha=0.18, jitter=0.28, ax=ax, legend=False)
    sns.boxplot(data=long, x="variable", y="value", hue=class_col, order=order, hue_order=hue_order,
                palette=CLASS_COLORS, ax=ax, width=0.7, showfliers=False, linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7.5)
    ax.set_ylim(-0.05, 1.32)   # headroom keeps the legend clear of every box
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(**LEGEND_BOXPLOT)
    apply_box_gradient(ax)
    if letter:
        panel_label(ax, letter, fontsize=14.0)  # Fig 4 uses 14 pt for its letters


def scaled_1_99(series: pd.Series) -> pd.Series:
    """Percentile scaling used by every profile boxplot (fig4:182-183)."""
    lo, hi = np.nanpercentile(series, [1, 99])
    return ((series - lo) / (hi - lo) if hi > lo else series * 0).clip(0, 1)


def scatter_class_layer(ax: plt.Axes, x: pd.Series, y: pd.Series, klass: int, alpha: float = 0.20) -> None:
    """Per-class scatter (s=8, alpha .20), 30 % KDE contour and median diamond (fig4:224-243)."""
    ax.scatter(x, y, s=8, color=CLASS_COLORS[klass], alpha=alpha, edgecolors="none")
    valid = x.notna() & y.notna()
    if valid.sum() > 30:
        sns.kdeplot(x=x[valid], y=y[valid], ax=ax, levels=[0.3], thresh=0.3, color=CLASS_COLORS[klass],
                    linewidths=1.3, alpha=0.9)
    ax.scatter(x.median(), y.median(), s=85, marker="D", color=CLASS_COLORS[klass], edgecolors=PALETTE["ink"],
               linewidths=0.9, zorder=10, label=f"LCZ {klass}")


def trend_axes_style(sax: plt.Axes, first: bool) -> None:
    """Seamless 1x4 trend composite sub-axes (fig9:340-399): only a taupe bottom spine
    (and left spine on the first column), dashed y-grid."""
    sax.set_facecolor("none")
    for name, sp in sax.spines.items():
        if name == "bottom":
            sp.set_color(PALETTE["taupe"]); sp.set_linewidth(0.8)
        else:
            sp.set_visible(False)
    sax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.5, alpha=0.6, linestyle="--")
    if first:
        sax.spines["left"].set_visible(True)
        sax.spines["left"].set_color(PALETTE["taupe"])
        sax.spines["left"].set_linewidth(0.8)
        sax.tick_params(axis="y", labelsize=7.5, colors=PALETTE["ink"], left=True)
    else:
        sax.tick_params(axis="y", left=False, labelleft=False)
    sax.tick_params(axis="x", labelsize=7.2, colors=PALETTE["ink"])


# ==========================================================================
# 9. Saving
# ==========================================================================
def tight_crop_bbox(fig: plt.Figure, pad_in: float = CROP_PAD_IN, full_width: bool = False) -> mpl.transforms.Bbox:
    """Crop to actual content extent (fig2:457-466 all sides; fig4:538-542 ``full_width=True``)."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = fig.get_size_inches()
    x0 = 0.0 if full_width else max(tb.x0 - pad_in, 0.0)
    x1 = fw if full_width else min(tb.x1 + pad_in, fw)
    return mpl.transforms.Bbox.from_extents(x0, max(tb.y0 - pad_in, 0.0), x1, min(tb.y1 + pad_in, fh))


def save_figure(fig: plt.Figure, path, crop: str | None = None, close: bool = True) -> Path:
    """dpi 300, white facecolor.  crop=None (fig9: no bbox_inches), 'all' (fig2) or 'vertical' (fig4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(dpi=DPI, facecolor=FACECOLOR)
    if crop:
        kw["bbox_inches"] = tight_crop_bbox(fig, full_width=(crop == "vertical"))
    fig.savefig(path, **kw)
    if close:
        plt.close(fig)
    return path


# ==========================================================================
# 9b. v2 print type scale (author decision 2026-10-30: enlarge type ~1.4x)
# ==========================================================================
# v1 figures were authored at 13.0 in and print at 190 mm (7.48 in) full width,
# i.e. at 0.575 scale, so an 8 pt label printed at ~4.6 pt.  v2 main figures are
# authored at V2_FIG_WIDTH_IN with the v1 point sizes and then pass through
# scale_text(), which multiplies every text size by TYPE_SCALE with a floor that
# prints at MIN_PRINT_PT.  Layout and overlap QA must run *after* scale_text().
V2_FIG_WIDTH_IN = 13.0
PRINT_WIDTH_IN = 190 / 25.4
TYPE_SCALE = 1.4
MIN_PRINT_PT = 6.0
MIN_AUTHOR_PT = MIN_PRINT_PT * V2_FIG_WIDTH_IN / PRINT_WIDTH_IN  # ~10.4 pt


def v2_fs(pt: float) -> float:
    """v1 point size -> v2 authored point size."""
    return max(pt * TYPE_SCALE, MIN_AUTHOR_PT)


def scale_text(fig: plt.Figure) -> int:
    """Apply v2_fs to every Text artist in ``fig`` (ticks, legends, colourbars,
    annotations, panel letters).  Idempotent: marks scaled artists.  Returns
    the number of artists changed."""
    w = fig.get_size_inches()[0]
    if abs(w - V2_FIG_WIDTH_IN) > 0.05:
        raise ValueError(f"v2 figures are authored at {V2_FIG_WIDTH_IN} in wide, got {w:.2f} in")
    fig.canvas.draw()  # materialise tick labels
    n = 0
    # Tick labels are regenerated on redraw, so scale them through tick_params.
    for ax in fig.axes:
        if getattr(ax, "_v2_ticks_scaled", False):
            continue
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            ticks = axis.get_major_ticks()
            if ticks:
                ax.tick_params(axis=name, which="both", labelsize=v2_fs(ticks[0].label1.get_fontsize()))
                for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                    tk.label1._v2_scaled = tk.label2._v2_scaled = True
                n += 1
        ax._v2_ticks_scaled = True
    for t in fig.findobj(mpl.text.Text):
        if getattr(t, "_v2_scaled", False) or not t.get_text():
            continue
        t.set_fontsize(v2_fs(t.get_fontsize()))
        t._v2_scaled = True
        n += 1
    return n


# ==========================================================================
# 10. Drift guard
# ==========================================================================
def _cmap_equal(a, b) -> bool:
    t = np.linspace(0, 1, 64)
    return np.allclose(a(t), b(t))


def verify_against_originals() -> dict[str, bool | None]:
    """Re-import the original scripts and compare every copied constant."""
    res: dict[str, bool | None] = {}
    res["PALETTE"] = PALETTE == _f1.PALETTE
    res["CLASS_COLORS"] = CLASS_COLORS == _f1.CLASS_COLORS
    res["LST_TEAL_PURPLE (fig9)"] = _cmap_equal(LST_TEAL_PURPLE_CMAP, _f9.LST_SOFT_TEAL_PURPLE_CMAP)
    res["CANOPY_10M (fig9)"] = _cmap_equal(CANOPY_CMAP_10M, _f9.CANOPY_CMAP)
    res["LST_CMAP_FIG1"] = _cmap_equal(LST_CMAP_FIG1, _f1.LST_CMAP)
    res["CONF_CMAP"] = _cmap_equal(CONF_CMAP, _f1.CONF_CMAP)
    for key, modname, attr, mine in [
        ("LST_CMAP_FIG3", "build_figure_3_label_audit_multipanel", "LST_CMAP", LST_CMAP_FIG3),
        ("MAP_CMAP (fig10)", "build_figure_10_leave_district_multipanel", "MAP_CMAP", LST_TEAL_PURPLE_CMAP),
        ("TRANSFER_CMAP (fig10)", "build_figure_10_leave_district_multipanel", "TRANSFER_CMAP", TRANSFER_CMAP),
        ("CONTEXT_CMAP (fig10)", "build_figure_10_leave_district_multipanel", "CONTEXT_CMAP", CONTEXT_CMAP),
        ("ACCESS_CMAP (fig8)", "build_figure_8_2sfca_multipanel", "ACCESS_CMAP", ACCESS_CMAP),
        ("DELTA_CMAP (fig8)", "build_figure_8_2sfca_multipanel", "DELTA_CMAP", DELTA_CMAP),
        ("WARNING_CMAP (fig7)", "build_figure_7_uncertainty_multipanel", "WARNING_CMAP", WARNING_CMAP),
        ("LST_CMAP_S1", "build_supplementary_texture_surface_canopy_atlas", "LST_CMAP", LST_CMAP_S1),
        ("CANOPY_CMAP_S1", "build_supplementary_texture_surface_canopy_atlas", "CANOPY_CMAP", CANOPY_CMAP_S1),
    ]:
        try:
            mod = __import__(modname)
            res[key] = _cmap_equal(mine, getattr(mod, attr))
        except Exception as exc:  # heavy optional deps missing -> report, do not fail silently
            res[key] = None  # original not importable in this env (missing optional dep): skipped, not verified
            print(f"  SKIPPED {key}: could not import {modname}: {exc!r}")
    try:
        f6 = __import__("build_figure_6_explainability_multipanel")
        res["FAMILY_COLORS_FIG6"] = FAMILY_COLORS_FIG6 == f6.FAMILY_COLORS
    except Exception as exc:
        res["FAMILY_COLORS_FIG6"] = None
        print(f"  SKIPPED FAMILY_COLORS_FIG6: {exc!r}")
    res["rcParams RESOLVED_RC"] = not verify_rc()
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    apply_style()
    for k, v in verify_against_originals().items():
        print(f"{'SKIP' if v is None else 'OK  ' if v else 'FAIL'} {k}")
