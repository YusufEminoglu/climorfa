"""Print-legible rebuild of main-text Figure 4 (Stage 1 class profiles).

The v1 figure (scripts/build_figure_4_morphology_multipanel.py) is authored at
16.2 in and printed at 190 mm, so its 5.4-7.5 pt text prints at ~3 pt.  This
script reuses the v1 data selection and plotting functions unchanged, authors
the same panels at 13.0 in, enlarges every text artist with
climorfa_style.scale_text (x1.4, floor 10.4 pt = 6 pt printed) and then
re-lays the figure out in inches so nothing overlaps.  Data, colours, chart
types, selected tiles and all numbers come from the v1 functions.

Layout changes relative to v1 (no data or wording changes beyond these):
* panel A: tile info boxes moved from inside the map to below it and wrapped
  onto 4 lines (same tokens); the "this cell / class median" legend moved to
  the panel-letter row above the tile titles; panel letter (a) raised above
  the tile titles.  The bar label for the height proxy reads "H" as defined
  in v1 CONTEXT_FIELDS (v1's letter finaliser turned it into "(h)").
* panel B: y label wrapped onto two lines.
* panels C/D: the same median-diamond legend in 2 columns (shallower box).
* panel E: per-tier legends (with n) moved below each mini-panel; KS read-out
  moved to the upper-left corner with "p<0.001" on its own line; x ticks every
  5 C; top-row x label dropped (the shared x axis is labelled on the bottom row).
* panel F: cool/hot legend moved out of the first mini-panel, next to (f).

Writes ONLY paper/figures/v2/main/fig_4_class_profiles.png.
Run from the project root:  python scripts/street/figures/build_v1print_fig4_class_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
for p in (str(SCRIPTS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import geopandas as gpd  # noqa: E402
import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_figure_4_morphology_multipanel as v1  # noqa: E402
from scripts.street.figures import climorfa_style as S  # noqa: E402

OUT = ROOT / "paper/figures/v2/main/fig_4_class_profiles.png"

# ---------------------------------------------------------------- layout (inches)
FW = S.V2_FIG_WIDTH_IN          # 13.0
L, R = 0.95, 0.25               # left margin holds panel B's two-line y label
CONTENT_W = FW - L - R

# panel A: 4 tile columns, each = map + context bars
A_COL_GAP = 0.30
A_COL_W = (CONTENT_W - 3 * A_COL_GAP) / 4
A_MAP_BAR_GAP = 0.06
A_MAP_W = A_COL_W * 0.64
A_BAR_W = A_COL_W - A_MAP_W - A_MAP_BAR_GAP
A_INFO_OFFSET = 0.30             # info box top, below map bottom (clears bar ticks)

# vertical stack, measured from the top of the sheet
TOP_PAD = 0.10
A_LETTER_ROW = 0.42              # (a) + bar legend
A_TITLE_ROW = 0.36               # class-name titles
A_INFO_H = 0.82                  # 4-line info box
GAP_A_B = 0.50                   # (b) letter
B_H = 2.50
GAP_B_CD = 0.80                  # B tick labels + (c)/(d) letters
CD_H = 2.80
GAP_CD_EF = 1.05                 # C/D tick + axis labels + (e)/(f) letters
EF_ROW_H = 1.75
E_TICK_H = 0.27                  # E tick numbers
E_XLABEL_H = 0.27                # E x label (bottom row only)
E_LEGEND_H = 0.66                # E per-tier legend, below the axes
EF_ROW_GAP = E_TICK_H + E_LEGEND_H + 0.12
BOTTOM = E_TICK_H + E_XLABEL_H + E_LEGEND_H + 0.05

A_MAP_TOP = TOP_PAD + A_LETTER_ROW + A_TITLE_ROW
A_MAP_BOTTOM = A_MAP_TOP + A_MAP_W
B_TOP = A_MAP_BOTTOM + A_INFO_OFFSET + A_INFO_H + GAP_A_B
B_BOTTOM = B_TOP + B_H
CD_TOP = B_BOTTOM + GAP_B_CD
CD_BOTTOM = CD_TOP + CD_H
E1_TOP = CD_BOTTOM + GAP_CD_EF
E1_BOTTOM = E1_TOP + EF_ROW_H
E2_TOP = E1_BOTTOM + EF_ROW_GAP
E2_BOTTOM = E2_TOP + EF_ROW_H
FH = E2_BOTTOM + BOTTOM

# C / D and E / F horizontal split
SPLIT_LEFT_END = 6.20            # right edge of C and of the E group
SPLIT_RIGHT_START = 7.35         # left axes edge of D and of the F group (room for its y label)
E_INNER_GAP = 0.50
F_INNER_GAP = 0.45


def rect(left: float, top: float, w: float, h: float) -> list[float]:
    """Inch rectangle measured from the sheet top -> figure fraction."""
    return [left / FW, (FH - top - h) / FH, w / FW, h / FH]


def finalize_letters(fig: plt.Figure) -> None:
    """'A'..'F' -> '(a)'..'(f)' (v1 convention), without touching the 'H' bar label."""
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text() in list("ABCDEF"):
                t.set_text(f"({t.get_text().lower()})")


def rewrap_info_box(ax: plt.Axes, klass: int) -> None:
    """Move v1's 3-line tile metrics box below the map, wrapped onto 4 lines (same tokens)."""
    box = next(t for t in ax.texts if "|" in t.get_text())
    l1, l2, l3 = box.get_text().split("\n")
    p2, p3 = l2.split("  "), l3.split("  ")     # [conf, cov, h], [Surf.sd, NDVI, LST]
    text = "\n".join([l1, "  ".join(p2[:2]), "  ".join([p2[2], p3[0]]), "  ".join(p3[1:])])
    box.remove()
    ax.text(
        0.0, -A_INFO_OFFSET / A_MAP_W, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color=v1.PALETTE["ink"], clip_on=False, linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": v1.CLASS_COLORS[klass], "alpha": 0.93, "pad": 2.7},
        zorder=8,
    )


def build() -> Path:
    S.apply_style()
    df = pd.read_csv(ROOT / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(ROOT / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    a = v1._analysis_subset(df)
    tiles = v1._select_distinct_texture_tiles(df)
    cell_lookup = grid.set_index("grid_id")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(FW, FH))

    # ---------------- panel A
    sub_labels = ["i", "ii", "iii", "iv"]
    ax_a0 = ax_bar0 = None
    for i, (_, row) in enumerate(tiles.iterrows()):
        x0 = L + i * (A_COL_W + A_COL_GAP)
        ax = fig.add_axes(rect(x0, A_MAP_TOP, A_MAP_W, A_MAP_W))
        ax_bar = fig.add_axes(rect(x0 + A_MAP_W + A_MAP_BAR_GAP, A_MAP_TOP, A_BAR_W, A_MAP_W))
        cell = cell_lookup.loc[[row["grid_id"]]].reset_index()
        v1._plot_texture_tile(ax, ROOT, cell, row, sub_labels[i])
        ax.set_anchor("S")
        klass = int(row["lcz_weak_label"])
        rewrap_info_box(ax, klass)
        v1._plot_tile_context_bars(ax_bar, row, a, klass)
        if i == 0:
            ax_a0, ax_bar0 = ax, ax_bar
    letter_y = 1 + (A_TITLE_ROW + 0.06) / A_MAP_W
    v1._panel_label(ax_a0, "A", y=letter_y)
    handles, labels = ax_bar0.get_legend_handles_labels()
    ax_bar0.legend(
        handles, labels, ncol=2, fontsize=5.4, frameon=True, loc="lower left", handlelength=1.6,
        borderpad=0.3, columnspacing=1.2,
        bbox_to_anchor=((L + 0.62) / FW, (FH - TOP_PAD - A_LETTER_ROW + 0.04) / FH),
        bbox_transform=fig.transFigure,
    )

    # ---------------- panels B, C, D
    ax_b = fig.add_axes(rect(L, B_TOP, CONTENT_W, B_H))
    v1._plot_profile_boxplots(ax_b, a)
    ax_b.set_ylabel("1st-99th percentile\nscaled value")

    ax_c = fig.add_axes(rect(L, CD_TOP, SPLIT_LEFT_END - L, CD_H))
    ax_d = fig.add_axes(rect(SPLIT_RIGHT_START, CD_TOP, FW - R - SPLIT_RIGHT_START, CD_H))
    v1._scatter(ax_c, a, "building_coverage_exact", "height_proxy_aw_mean_m", "C")
    ax_c.set_ylim(0, min(45, np.nanpercentile(a["height_proxy_aw_mean_m"], 99.5)))
    v1._scatter(ax_d, a, "building_coverage_exact", "s2_ndvi_mean", "D")
    # same legend, 2 columns: a shallower box over the sparse upper-right corner
    for ax_s in (ax_c, ax_d):
        ax_s.get_legend().remove()
        ax_s.legend(fontsize=6.5, frameon=True, loc="upper right", title="LCZ (diamond = median)",
                    title_fontsize=6.5, markerscale=0.9, ncol=2, columnspacing=1.0)

    # ---------------- panels E, F (2 x 2 small multiples, repositioned in inches)
    gs_tmp = fig.add_gridspec(1, 2, left=0.05, right=0.95, top=0.2, bottom=0.05)
    e_axes = v1._plot_access_lst_boxplots(fig, gs_tmp[0, 0], a)
    f_axes = v1._plot_hotcool_boxplots(fig, gs_tmp[0, 1], a)
    e_w = (SPLIT_LEFT_END - L - E_INNER_GAP) / 2
    f_w = (FW - R - SPLIT_RIGHT_START - F_INNER_GAP) / 2
    for idx, (ae, af) in enumerate(zip(e_axes, f_axes)):
        r, c = divmod(idx, 2)
        top = E1_TOP if r == 0 else E2_TOP
        ae.set_position(rect(L + c * (e_w + E_INNER_GAP), top, e_w, EF_ROW_H))
        af.set_position(rect(SPLIT_RIGHT_START + c * (f_w + F_INNER_GAP), top, f_w, EF_ROW_H))
        ae.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ae.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        if r == 0:
            ae.set_xlabel("")
        # KS read-out: same text, the "p" part on its own line, moved to the upper-left corner
        # (empty in all four classes once the legend sits below the axes; the lower-right
        # corner is too narrow for it at print size)
        for t in ae.texts:
            if "KS D=" in t.get_text():
                t.set_text(t.get_text().replace(", p", ",\np"))
                t.set_position((0.02, 0.97))
                t.set_ha("left")
                t.set_va("top")
        # per-tier legend (with n): moved below the axes, it cannot fit beside the curves at print size
        lg = ae.get_legend()
        h, lab = ae.get_legend_handles_labels()
        lg.remove()
        below = (E_TICK_H + (E_XLABEL_H if r == 1 else 0.0)) / EF_ROW_H
        ae.legend(h, lab, fontsize=5.6, frameon=True, loc="upper left", handlelength=1.4,
                  bbox_to_anchor=(0.0, -below), borderaxespad=0.0)
    # cool/hot legend: out of the first F mini-panel, onto the letter row
    hl = f_axes[0].get_legend()
    h, lab = f_axes[0].get_legend_handles_labels()
    hl.remove()
    f_axes[0].legend(h, lab, fontsize=6.2, frameon=True, ncol=2, loc="lower left",
                     bbox_to_anchor=(0.20, 1.0), borderaxespad=0.15, handlelength=1.6)

    finalize_letters(fig)
    S.scale_text(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    qa(fig)
    S.save_figure(fig, OUT, crop="vertical", close=False)
    plt.close(fig)
    return OUT


# ---------------------------------------------------------------- QA
def _texts(fig: plt.Figure):
    """Text artists that are actually drawn (ticks inside the view, axes that are on)."""
    cand = list(fig.texts)
    for ax in fig.axes:
        cand += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                cand.append(axis.label)
                for tk in axis._update_ticks():
                    if tk.label1.get_visible():
                        cand.append(tk.label1)
        lg = ax.get_legend()
        if lg is not None:
            cand += list(lg.get_texts()) + [lg.get_title()]
    return [t for t in cand if t.get_visible() and t.get_text().strip()]


def qa(fig: plt.Figure) -> None:
    r = fig.canvas.get_renderer()
    scale = S.PRINT_WIDTH_IN / FW
    texts = _texts(fig)
    sizes = [(t.get_fontsize(), t.get_text()) for t in texts]
    mn = min(sizes)
    print(f"text artists: {len(texts)}; min authored {mn[0]:.2f} pt -> printed {mn[0] * scale:.2f} pt ({mn[1]!r})")
    bad = [s for s in sizes if s[0] < S.MIN_AUTHOR_PT - 1e-6]
    print("below floor:", bad if bad else "none")
    # pairwise text-text overlaps
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    over = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if boxes[i][0].get_rotation() % 180 and boxes[j][0].get_rotation() % 180:
                continue  # parallel rotated tick labels: axis-aligned boxes always overlap; checked visually
            if a.overlaps(b):
                ix = min(a.x1, b.x1) - max(a.x0, b.x0)
                iy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ix > 1 and iy > 1:
                    over.append((boxes[i][0].get_text()[:25], boxes[j][0].get_text()[:25]))
    print(f"text-text overlaps: {len(over)}")
    for o in over:
        print("   ", o)
    fb = fig.bbox
    clip = [t.get_text()[:30] for t, b in boxes if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5]
    print("clipped at figure sides:", clip if clip else "none")
    print(f"figure {FW:.2f} x {FH:.2f} in (before vertical crop), ratio {FH / FW:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(build())
