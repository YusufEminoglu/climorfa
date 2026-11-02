"""Print-legible (v2 type scale) rebuild of v1 main-text Figure 5, Stage 1 baseline performance.

Re-uses the plotting functions of the v1 script ``scripts/build_figure_5_validation_multipanel.py``
unchanged (same panels, chart types, colours, data, numbers) and changes only the LAYOUT so that the
v2 type scale (``climorfa_style.scale_text``: x1.4, floor 10.4 pt at 13 in = 6 pt printed at 190 mm)
fits without overlaps.  Writes ONLY paper/figures/v2/main/fig_5_baseline_performance.png.

Run from the project root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_v1print_fig5_baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_figure_5_validation_multipanel as v1  # noqa: E402
from scripts.street.figures import climorfa_style as S  # noqa: E402

P = v1.PALETTE
MODEL_DIR = ROOT / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix"
OUT_PNG = ROOT / "paper/figures/v2/main/fig_5_baseline_performance.png"

# ---- layout (inches, authored at 13.0 in wide) --------------------------------------------------
FIG_W, FIG_H = S.V2_FIG_WIDTH_IN, 13.0
A_LEFT, A_RIGHT = 0.85, 0.85           # room for y-tick labels + "Macro-F1" / LGBM..ET labels
A_TOP, A_H = 0.80, 4.30                # sparkline titles + two-line zoom title above the axes
SPARK_GAP, ZOOM_GAP = 0.42, 0.55
B_TOP_GAP = 1.15                       # x-labels of row A + (b) tag
B_H = 5.80
CHORD_LEFT, CHORD_W = 0.10, 6.10
RAIN_LEFT = 7.45                       # leaves room for the "Correct (n=...)" labels left of the axes
RAIN_RIGHT = 0.15
CHORD_X_HALF = 1.95                    # wider data window so the enlarged node labels stay on the canvas


def _inset_ticks(lo: float, hi: float) -> list[float]:
    """At most three round 2-decimal ticks inside [lo, hi] (insets keep 2-3 ticks only)."""
    for step in (0.01, 0.02, 0.03, 0.05):
        t = np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step)
        if 2 <= len(t) <= 3:
            return [round(x, 2) for x in t]
    return [round(lo, 2), round(hi, 2)]


def build() -> Path:
    summary = pd.read_csv(MODEL_DIR / "summary_metrics.csv")
    fold = pd.read_csv(MODEL_DIR / "fold_metrics.csv")
    pred = pd.read_csv(MODEL_DIR / "out_of_fold_predictions.csv")
    class_metrics = pd.read_csv(MODEL_DIR / "pooled_class_metrics.csv")
    conf = pd.read_csv(MODEL_DIR / "pooled_confusion_matrix_long.csv")

    S.apply_style()  # same rc sequence the v1 figures were rendered under (generate_manuscript_evidence)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_in = lambda l, b, w, h: S.inch_axes(fig, l, b, w, h)  # noqa: E731

    # ---- row A: four sparklines + wider zoom --------------------------------------------------
    avail = FIG_W - A_LEFT - A_RIGHT - 3 * SPARK_GAP - ZOOM_GAP
    w = avail / 5.5
    a_bottom = FIG_H - A_TOP - A_H
    spark_axes = [ax_in(A_LEFT + i * (w + SPARK_GAP), a_bottom, w, A_H) for i in range(4)]
    zoom_ax = ax_in(A_LEFT + 4 * w + 3 * SPARK_GAP + ZOOM_GAP, a_bottom, 1.5 * w, A_H)

    # ---- row B: chord + raincloud -------------------------------------------------------------
    b_bottom = a_bottom - B_TOP_GAP - B_H
    ax_chord = ax_in(CHORD_LEFT, b_bottom, CHORD_W, B_H)
    ax_rain = ax_in(RAIN_LEFT, b_bottom, FIG_W - RAIN_LEFT - RAIN_RIGHT, B_H)

    v1._plot_ablation_small_multiples(spark_axes, zoom_ax, summary, fold)
    v1._plot_chord_confusion(ax_chord, conf, class_metrics, v1.PRIMARY_MODEL)
    v1._plot_confidence_raincloud(ax_rain, pred, v1.PRIMARY_MODEL)

    # (b) tag exactly as v1: x of (a), y = chord top + 13 pt
    pos_a0, pos_chord = spark_axes[0].get_position(), ax_chord.get_position()
    fig.text(pos_a0.x0 + 0.01 * pos_a0.width,
             pos_chord.y1 - 0.01 * pos_chord.height + (13.0 / 72.0) / fig.get_figheight(),
             "(b)", ha="left", va="top", fontsize=12, fontweight="bold", color=P["ink"],
             bbox={"facecolor": "white", "edgecolor": P["ink"], "boxstyle": "round,pad=0.18", "linewidth": 0.9,
                   "alpha": 0.96}, zorder=30)
    v1._finalize_panel_labels(fig)

    # ================= layout changes for the enlarged type ===================================
    # Sparklines: a little more x room right of recipe 7 for the end-value label.
    for ax in spark_axes:
        ax.set_xlim(0.55, len(v1.RECIPE_ORDER) + 1.15)
        # inset violins: slightly larger box, 2-3 round ticks only
        for ins in ax.child_axes:
            ins.set_position([0.60, 0.05, 0.37, 0.42])
            lo, hi = ins.get_ylim()
            ins.set_yticks(_inset_ticks(lo, hi))
            ins.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    # (a) tag: raised above the (now larger) sparkline titles instead of sitting on "LightGBM".
    for t in spark_axes[0].texts:
        if t.get_text() == "(a)":
            t.set_position((0.01, 1.0 + 0.66 / A_H))

    # Zoom panel: two-line title; majority-floor note on two lines in added headroom at the top.
    zoom_ax.set_title("Primary recipe (800 m):\nfold-level spread", fontsize=9.6, pad=5, linespacing=1.15)
    n_models = len(v1.MODEL_RANK_ORDER)
    zoom_ax.set_ylim(-0.6, n_models + 0.25)
    for t in zoom_ax.texts:
        if t.get_text().startswith("Majority-classifier floor"):
            t.set_text("Majority-classifier floor (macro-F1 0.128)\nis off-scale here.")
            t.set_position((0.797, n_models + 0.20))
            t.set_linespacing(1.2)

    # Chord: wider data window (same physical aspect, circle stays round); legend moved below the axes.
    bw, bh = CHORD_W, B_H
    ax_chord.set_xlim(-CHORD_X_HALF, CHORD_X_HALF)
    ax_chord.set_ylim(-CHORD_X_HALF * bh / bw, CHORD_X_HALF * bh / bw)
    leg = ax_chord.get_legend()
    handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    leg.remove()
    ax_chord.legend(handles=handles, labels=labels, fontsize=6.8, frameon=True, loc="upper center",
                    bbox_to_anchor=(0.5, 0.13), ncol=1, handlelength=1.4)

    # Raincloud: calibration card moved from the top-right corner (it would cover the Correct
    # violin peak at full size) to the top-left, above the flat part of the Correct density.
    for t in ax_rain.texts:
        if t.get_text().startswith("Pooled mean confidence"):
            t.set_position((0.015, 0.995))
            t.set_ha("left")
            t.set_multialignment("left")

    S.scale_text(fig)
    # inset (child) axes are not in fig.axes, so scale_text cannot reach their tick labels via
    # tick_params; give them the print floor explicitly.
    for ax in spark_axes:
        for ins in ax.child_axes:
            ins.tick_params(axis="y", labelsize=S.MIN_AUTHOR_PT, length=2.5, pad=1.5)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=S.DPI, facecolor=S.FACECOLOR)
    report(fig)
    plt.close(fig)
    return OUT_PNG


def report(fig: plt.Figure) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()  # tick labels of ticks outside the view interval are never drawn
    axes = list(fig.axes) + [c for a in fig.axes for c in a.child_axes]
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({id(tk.label1), id(tk.label2)})
    sizes = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_text() and t.get_visible() and id(t) not in hidden:
            sizes.append((t.get_fontsize(), t.get_text()[:40]))
            bb = t.get_window_extent(r)
            W, H = fig.bbox.width, fig.bbox.height
            if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
                print(f"OFF-CANVAS: {t.get_text()[:40]!r} {bb}")
    mn = min(sizes)
    print(f"text artists: {len(sizes)}; min authored {mn[0]:.2f} pt ({mn[1]!r}) "
          f"= {mn[0] * S.PRINT_WIDTH_IN / S.V2_FIG_WIDTH_IN:.2f} pt printed at 190 mm")


if __name__ == "__main__":
    print(build())
