"""Build a two-panel validation figure: additive-ablation trajectory (top)
and a primary-model error/confidence diagnostic (bottom, chord diagram +
raincloud)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Wedge, Patch
from scipy.stats import gaussian_kde

from build_figure_1_texture_atlas import PALETTE


MODEL_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "dummy_majority": "Majority",
}
RECIPE_LABELS = {
    "baseline_a_morphology_only": "Morphology",
    "baseline_b_morphology_plus_vegetation": "+ vegetation",
    "baseline_c_morphology_plus_green_blue_context": "+ green-blue",
    "baseline_d_no_green_2sfca": "Full, no 2SFCA",
    "baseline_d_green_2sfca_400m": "Full + 400 m",
    "baseline_d_full_proxy_context": "Full + 800 m",
    "baseline_d_green_2sfca_1200m": "Full + 1200 m",
}
RECIPE_ORDER = list(RECIPE_LABELS.values())
RECIPE_KEY_ORDER = list(RECIPE_LABELS.keys())
PRIMARY_RECIPE = "baseline_d_full_proxy_context"
PRIMARY_RECIPE_IDX = RECIPE_KEY_ORDER.index(PRIMARY_RECIPE)
PRIMARY_MODEL = "lightgbm"
COMPARISON_MODEL = "random_forest"
MODEL_COLORS = {
    "LightGBM": PALETTE["teal_dark"],
    "XGBoost": PALETTE["orange_dark"],
    "Random Forest": PALETTE["pink_dark"],
    "Extra Trees": PALETTE["taupe_dark"],
    "Majority": PALETTE["muted"],
}
CLASS_ORDER = [3, 6, 8, 9]
CLASS_COLORS = {3: PALETTE["teal"], 6: PALETTE["pink"], 8: PALETTE["orange"], 9: PALETTE["taupe"]}
CLASS_DARK = {3: PALETTE["teal_dark"], 6: PALETTE["pink_dark"], 8: PALETTE["orange_dark"], 9: PALETTE["taupe_dark"]}


def _lighten(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Blend a hex color toward white, e.g. for a softer fill echoing panel A's
    translucent ribbon look instead of a flat, fully saturated block."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


CLASS_COLORS_SOFT = {c: _lighten(hex_color, 0.35) for c, hex_color in CLASS_COLORS.items()}


def _panel_label(ax: plt.Axes, label: str, color: str = PALETTE["ink"], up_points: float = 0.0) -> None:
    axes_height_in = ax.get_position().height * ax.figure.get_figheight()
    y = 0.99 + (up_points / 72.0) / axes_height_in
    ax.text(
        0.01,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": color, "boxstyle": "round,pad=0.18", "linewidth": 0.9, "alpha": 0.96},
        zorder=30,
    )


def _finalize_panel_labels(fig: plt.Figure) -> None:
    """Render bare panel letters (e.g. 'A') as lowercase-parenthesized (e.g. '(a)')."""
    for ax in fig.axes:
        for child in ax.texts:
            text = child.get_text()
            if re.fullmatch(r"[A-Za-z]", text):
                child.set_text(f"({text.lower()})")


# ---------------------------------------------------------------------------
# Top panel: per-model ablation sparklines + a primary-recipe zoom
# ---------------------------------------------------------------------------

MODEL_RANK_ORDER = ["LightGBM", "XGBoost", "Random Forest", "Extra Trees"]
MODEL_ABBR = {"LightGBM": "LGBM", "XGBoost": "XGB", "Random Forest": "RF", "Extra Trees": "ET"}


def _plot_fold_violin_inset(ax: plt.Axes, fold_values: np.ndarray, model_label: str) -> None:
    # Small violin + jittered strip of this model's own 5 fold-level
    # macro-F1 values at the primary recipe, tucked in the corner that the
    # rising trajectory line never occupies. Deliberately kept even though
    # the zoom panel (E) shows the same numbers pooled across models: this
    # gives each sparkline a self-contained read on that model's own
    # variability without needing to cross-reference panel E.
    color = MODEL_COLORS[model_label]
    inset = ax.inset_axes([0.64, 0.06, 0.33, 0.40])
    if len(fold_values) >= 2 and np.std(fold_values) > 0:
        parts = inset.violinplot([fold_values], positions=[0], widths=0.85, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.30)
            body.set_edgecolor(color)
            body.set_linewidth(0.7)
    rng = np.random.default_rng(11)
    jitter = rng.uniform(-0.14, 0.14, size=len(fold_values))
    inset.scatter(jitter, fold_values, s=8, color=color, alpha=0.9, edgecolors="white", linewidth=0.3, zorder=5)
    pad = (fold_values.max() - fold_values.min()) * 0.35 + 0.0015
    inset.set_xlim(-0.65, 0.65)
    inset.set_ylim(fold_values.min() - pad, fold_values.max() + pad)
    inset.set_xticks([])
    inset.tick_params(axis="y", labelsize=4.6, length=1.5, pad=1, colors=PALETTE["muted"])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(PALETTE["muted"])


def _plot_model_sparkline(ax: plt.Axes, group: pd.DataFrame, model_label: str, show_ylabels: bool, fold_values: np.ndarray) -> None:
    color = MODEL_COLORS[model_label]
    x = group["recipe_pos"].to_numpy()
    y = group["macro_f1_mean"].to_numpy()
    sd = group["macro_f1_sd"].to_numpy()
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.20, zorder=1, linewidth=0)
    ax.plot(x, y, marker="o", ms=4.6, lw=2.0, color=color, zorder=5, mec="white", mew=0.5)

    primary_pos = PRIMARY_RECIPE_IDX + 1
    ax.axvline(primary_pos, color=PALETTE["grid"], linestyle="-", linewidth=6, zorder=0, alpha=0.7)
    final_val = y[-1]
    ax.annotate(
        f"{final_val:.3f}",
        (x[-1], final_val),
        xytext=(4, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        color=color,
    )

    ax.set_xlim(0.55, len(RECIPE_ORDER) + 0.75)
    ax.set_ylim(0.60, 0.87)
    ax.set_xticks(range(1, len(RECIPE_ORDER) + 1))
    ax.set_xticklabels(range(1, len(RECIPE_ORDER) + 1), fontsize=7.2, color=PALETTE["muted"])
    ax.tick_params(axis="x", length=2, pad=2)
    if show_ylabels:
        ax.set_ylabel("Macro-F1", fontsize=8.8)
        ax.tick_params(axis="y", labelsize=7.6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(model_label, fontsize=9.6, color=color, fontweight="bold", pad=5)

    _plot_fold_violin_inset(ax, fold_values, model_label)


def _plot_primary_recipe_zoom(ax: plt.Axes, summary: pd.DataFrame, fold: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    y_pos = {name: len(MODEL_RANK_ORDER) - 1 - i for i, name in enumerate(MODEL_RANK_ORDER)}

    f = fold[(fold["recipe"] == PRIMARY_RECIPE) & (fold["model"] != "dummy_majority")].copy()
    f["model_label"] = f["model"].map(MODEL_LABELS)
    for model_label, group in f.groupby("model_label"):
        y0 = y_pos[model_label]
        jitter = rng.uniform(-0.16, 0.16, size=len(group))
        ax.scatter(group["macro_f1"], y0 + jitter, s=22, color=MODEL_COLORS[model_label], alpha=0.55, edgecolors="none", zorder=3)

    s = summary[(summary["recipe"] == PRIMARY_RECIPE) & (summary["model"] != "dummy_majority")].copy()
    s["model_label"] = s["model"].map(MODEL_LABELS)
    for _, row in s.iterrows():
        y0 = y_pos[row["model_label"]]
        color = MODEL_COLORS[row["model_label"]]
        ax.plot([row["macro_f1_mean"] - row["macro_f1_sd"], row["macro_f1_mean"] + row["macro_f1_sd"]], [y0, y0], color=color, linewidth=1.4, zorder=4)
        ax.scatter([row["macro_f1_mean"]], [y0], marker="D", s=48, color=color, edgecolors="white", linewidth=0.8, zorder=6)
        ax.text(row["macro_f1_mean"], y0 + 0.30, f"{row['macro_f1_mean']:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=color)

    # Labels sit on the right, not the default left: at the default position
    # they would spill into the neighboring Extra Trees sparkline, since this
    # is the rightmost, widest column and there is no room to its left.
    # Abbreviated here since the full model names are already the sparkline
    # titles directly above, in the same colors.
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([MODEL_ABBR[name] for name in y_pos], fontsize=9.2, fontweight="bold")
    ax.yaxis.tick_right()
    abbr_to_name = {v: k for k, v in MODEL_ABBR.items()}
    for tick_label in ax.get_yticklabels():
        tick_label.set_color(MODEL_COLORS[abbr_to_name[tick_label.get_text()]])
    ax.set_ylim(-0.6, len(MODEL_RANK_ORDER) - 0.4)
    ax.set_xlim(0.795, 0.855)
    ax.set_xlabel("Macro-F1", fontsize=8.8)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Primary recipe (800 m): fold-level spread", fontsize=9.6, pad=5)
    ax.text(
        0.795,
        len(MODEL_RANK_ORDER) - 0.55,
        "Majority-classifier floor (macro-F1 0.128) is off-scale here.",
        ha="left",
        va="top",
        fontsize=6.8,
        color=PALETTE["muted"],
        style="italic",
    )


def _plot_ablation_small_multiples(
    spark_axes: list[plt.Axes],
    zoom_ax: plt.Axes,
    summary: pd.DataFrame,
    fold: pd.DataFrame,
) -> None:
    s = summary[summary["model"] != "dummy_majority"].copy()
    s["recipe_pos"] = s["recipe"].map({r: i + 1 for i, r in enumerate(RECIPE_KEY_ORDER)})
    s["model_label"] = s["model"].map(MODEL_LABELS)

    f_primary = fold[(fold["recipe"] == PRIMARY_RECIPE) & (fold["model"] != "dummy_majority")].copy()
    f_primary["model_label"] = f_primary["model"].map(MODEL_LABELS)

    for i, (ax, model_label) in enumerate(zip(spark_axes, MODEL_RANK_ORDER)):
        group = s[s["model_label"] == model_label].sort_values("recipe_pos")
        fold_values = f_primary.loc[f_primary["model_label"] == model_label, "macro_f1"].to_numpy()
        _plot_model_sparkline(ax, group, model_label, show_ylabels=(i == 0), fold_values=fold_values)

    _plot_primary_recipe_zoom(zoom_ax, summary, fold)

    _panel_label(spark_axes[0], "A", up_points=13.0)


# ---------------------------------------------------------------------------
# Bottom-left: chord diagram of the primary model's pooled confusion
# ---------------------------------------------------------------------------

def _arc_xy(deg_a: float, deg_b: float, r: float, n: int = 40) -> np.ndarray:
    degs = np.linspace(deg_a, deg_b, n)
    rad = np.radians(degs)
    return np.column_stack([r * np.cos(rad), r * np.sin(rad)])


def _chord_ribbon(ax: plt.Axes, seg_a: tuple[float, float], seg_b: tuple[float, float], r: float, color: str, alpha: float) -> None:
    a1, a2 = seg_a
    b1, b2 = seg_b
    arc_a = _arc_xy(a1, a2, r)
    arc_b = _arc_xy(b1, b2, r)
    verts = [arc_a[0]]
    codes = [MplPath.MOVETO]
    for pt in arc_a[1:]:
        verts.append(pt)
        codes.append(MplPath.LINETO)
    verts += [(0.0, 0.0), arc_b[0]]
    codes += [MplPath.CURVE3, MplPath.CURVE3]
    for pt in arc_b[1:]:
        verts.append(pt)
        codes.append(MplPath.LINETO)
    verts += [(0.0, 0.0), arc_a[0]]
    codes += [MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=2))


def _plot_chord_confusion(ax: plt.Axes, conf: pd.DataFrame, class_metrics: pd.DataFrame, model: str) -> None:
    cm = (
        conf[(conf["model"] == model) & (conf["recipe"] == PRIMARY_RECIPE)]
        .pivot(index="true_class", columns="predicted_class", values="count")
        .reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
        .fillna(0)
    )
    cmet = class_metrics[(class_metrics["model"] == model) & (class_metrics["recipe"] == PRIMARY_RECIPE)].set_index("class")

    r_in, r_out, bulge = 0.86, 1.0, 0.10
    gap_deg = 3.2
    node_total = {}
    for c in CLASS_ORDER:
        outs = sum(cm.loc[c, p] for p in CLASS_ORDER if p != c)
        ins = sum(cm.loc[t, c] for t in CLASS_ORDER if t != c)
        node_total[c] = outs + cm.loc[c, c] + ins
    grand_total = sum(node_total.values())
    scale = (360.0 - gap_deg * len(CLASS_ORDER)) / grand_total

    node_bounds: dict[int, tuple[float, float]] = {}
    seg_bounds: dict[tuple[str, int, int], tuple[float, float]] = {}
    current = 90.0
    for c in CLASS_ORDER:
        span = node_total[c] * scale
        start, end = current, current - span
        node_bounds[c] = (start, end)
        pos = start
        for p in CLASS_ORDER:
            if p == c:
                continue
            w = cm.loc[c, p] * scale
            seg_bounds[("out", c, p)] = (pos, pos - w)
            pos -= w
        w_self = cm.loc[c, c] * scale
        seg_bounds[("self", c, c)] = (pos, pos - w_self)
        pos -= w_self
        for t in CLASS_ORDER:
            if t == c:
                continue
            w = cm.loc[t, c] * scale
            seg_bounds[("in", c, t)] = (pos, pos - w)
            pos -= w
        current = end - gap_deg

    # Base rings, colored by class identity but softened toward white (like
    # panel A's translucent SD ribbons) rather than a flat saturated block;
    # the self-loop bulge below carries the fuller, darker accent color.
    for c in CLASS_ORDER:
        start, end = node_bounds[c]
        ax.add_patch(Wedge((0, 0), r_out, end, start, width=r_out - r_in, facecolor=CLASS_COLORS_SOFT[c], edgecolor="white", linewidth=0.6, zorder=3))

    # Self-loop bulges (correct predictions), dark class variant, outward of the ring.
    for c in CLASS_ORDER:
        s1, s2 = seg_bounds[("self", c, c)]
        ax.add_patch(Wedge((0, 0), r_out + bulge, s2, s1, width=bulge, facecolor=CLASS_DARK[c], edgecolor="white", linewidth=0.5, zorder=4))

    # Scale ticks around each node's outer rim (every 250 cells), circos-style,
    # so the ring reads as measured/data-rich rather than a plain flat band.
    tick_step = 250
    for c in CLASS_ORDER:
        start, _ = node_bounds[c]
        n_ticks = int(node_total[c] // tick_step)
        for k in range(n_ticks + 1):
            tick_deg = start - k * tick_step * scale
            rad = np.radians(tick_deg)
            r0, r1 = r_out + bulge, r_out + bulge + (0.028 if k % 2 == 0 else 0.016)
            ax.plot([r0 * np.cos(rad), r1 * np.cos(rad)], [r0 * np.sin(rad), r1 * np.sin(rad)], color=PALETTE["ink"], linewidth=0.6, alpha=0.55, zorder=5)

    # Cross ribbons (misclassifications), colored by the true class, translucent.
    for c in CLASS_ORDER:
        for p in CLASS_ORDER:
            if p == c or cm.loc[c, p] <= 0:
                continue
            _chord_ribbon(ax, seg_bounds[("out", c, p)], seg_bounds[("in", p, c)], r_in, CLASS_COLORS[c], alpha=0.60)

    # Class labels with F1 and support, placed just outside each node's bulge.
    for c in CLASS_ORDER:
        start, end = node_bounds[c]
        mid = np.radians((start + end) / 2)
        lx, ly = (r_out + bulge + 0.09) * np.cos(mid), (r_out + bulge + 0.09) * np.sin(mid)
        ha = "left" if np.cos(mid) >= 0 else "right"
        f1 = cmet.loc[c, "f1"] if c in cmet.index else np.nan
        support = int(cmet.loc[c, "support"]) if c in cmet.index else 0
        ax.text(
            lx,
            ly,
            f"LCZ {c}\nF1 {f1:.2f}  n={support:,}",
            ha=ha,
            va="center",
            fontsize=7.6,
            color=PALETTE["ink"],
            linespacing=1.25,
        )

    legend_handles = [
        Patch(facecolor=PALETTE["ink"], edgecolor="none", alpha=1.0, label="correct (dark outward rim, per class)"),
        Patch(facecolor=PALETTE["ink"], edgecolor="none", alpha=0.42, label="misclassified (translucent ribbon, true class color)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6.8, frameon=True, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1, handlelength=1.4)

    # Match the data-limit ratio to the axes box's actual physical aspect
    # ratio instead of calling set_aspect("equal"): that call would shrink
    # the box itself to keep the circle round, breaking left/right alignment
    # with the raincloud panel next to it. Pre-matching the ratio keeps the
    # circle round while letting the box fill its full allocated width.
    pos = ax.get_position()
    fig_w_in, fig_h_in = ax.figure.get_size_inches()
    box_w_in, box_h_in = pos.width * fig_w_in, pos.height * fig_h_in
    x_half = 1.62
    y_half = x_half * (box_h_in / box_w_in)
    ax.set_xlim(-x_half, x_half)
    ax.set_ylim(-y_half, y_half)
    ax.axis("off")
    # Panel-B tag is placed externally (build_figure_5_validation), pinned to
    # the same absolute x as panel A's tag, since this axes' own box is
    # intentionally wider than panel A's and left-anchoring the tag to this
    # box's own corner would pull it out of vertical alignment with (a).


# ---------------------------------------------------------------------------
# Bottom-right: raincloud of OOF confidence, correct vs. mismatched
# ---------------------------------------------------------------------------

def _half_violin(ax: plt.Axes, data: np.ndarray, base_y: float, color: str, height: float = 0.42) -> None:
    if len(data) < 3 or np.std(data) == 0:
        return
    xs = np.linspace(0.25, 1.0, 200)
    kde = gaussian_kde(data, bw_method=0.18)
    density = kde(xs)
    density = density / density.max() * height
    ax.fill_between(xs, base_y, base_y + density, color=color, alpha=0.30, linewidth=0, zorder=2)
    ax.plot(xs, base_y + density, color=color, linewidth=1.2, alpha=0.85, zorder=3)


def _mini_box(ax: plt.Axes, data: np.ndarray, y: float, color: str, height: float = 0.055) -> None:
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo = max(data.min(), q1 - 1.5 * iqr)
    hi = min(data.max(), q3 + 1.5 * iqr)
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.1, zorder=4)
    ax.add_patch(plt.Rectangle((q1, y - height / 2), q3 - q1, height, facecolor=color, edgecolor=PALETTE["ink"], linewidth=0.6, zorder=5))
    ax.plot([med, med], [y - height / 2, y + height / 2], color="white", linewidth=1.3, zorder=6)


def _plot_confidence_raincloud(ax: plt.Axes, pred: pd.DataFrame, model: str) -> None:
    p = pred[(pred["model"] == model) & (pred["recipe"] == PRIMARY_RECIPE)].copy()
    p["correct"] = p["true_label"] == p["predicted_label"]
    groups = [("Correct", p.loc[p["correct"], "predicted_probability"].to_numpy(), PALETTE["teal_dark"], 0.85), ("Mismatch", p.loc[~p["correct"], "predicted_probability"].to_numpy(), PALETTE["pink_dark"], 0.0)]
    rng = np.random.default_rng(7)
    for name, values, color, base_y in groups:
        _half_violin(ax, values, base_y + 0.10, color, height=0.30)
        _mini_box(ax, values, base_y - 0.02, color)
        jitter = rng.uniform(-0.075, 0.0, size=len(values))
        ax.scatter(values, base_y - 0.14 + jitter, s=5, color=color, alpha=0.25, edgecolors="none", zorder=1)
        n = len(values)
        ax.text(0.235, base_y + 0.10, f"{name}\n(n={n:,})", ha="right", va="center", fontsize=7.6, color=color, fontweight="bold", linespacing=1.2)

    mean_conf = p["predicted_probability"].mean()
    accuracy = p["correct"].mean()
    confidence_gap = mean_conf - accuracy
    ax.axvline(mean_conf, color=PALETTE["ink"], linestyle=":", linewidth=1.0, zorder=0)
    ax.axvline(accuracy, color=PALETTE["pink_dark"], linestyle="--", linewidth=1.0, alpha=0.85, zorder=0)
    ax.text(
        0.985,
        0.985,
        f"Pooled mean confidence {mean_conf:.3f}\nvs. pooled accuracy {accuracy:.3f}\nconfidence gap +{confidence_gap:.3f}\n(probabilities run more extreme\nthan accuracy justifies)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.9,
        color=PALETTE["ink"],
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": PALETTE["grid"], "alpha": 0.92, "pad": 3.5},
    )

    ax.set_xlim(0.22, 1.02)
    ax.set_ylim(-0.28, 1.32)
    ax.set_yticks([])
    ax.set_xlabel("OOF maximum probability", fontsize=8.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, zorder=0)


def build_figure_5_validation(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_5_validation_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix"
    summary = pd.read_csv(model_dir / "summary_metrics.csv")
    fold = pd.read_csv(model_dir / "fold_metrics.csv")
    pred = pd.read_csv(model_dir / "out_of_fold_predictions.csv")
    class_metrics = pd.read_csv(model_dir / "pooled_class_metrics.csv")
    conf = pd.read_csv(model_dir / "pooled_confusion_matrix_long.csv")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15.6, 12.8), constrained_layout=False)
    # Row A's margin is set by its own peripheral labels (y-tick labels on
    # the leftmost sparkline, LGBM/XGB/RF/ET labels on the zoom panel) and
    # cannot shrink further without clipping. Row B's chord + raincloud need
    # almost no peripheral margin, so row B is deliberately given a wider
    # box than row A here -- its content fills that box fully (same fix as
    # before), matching how fully panel A's content fills its own box. The
    # panel-B tag is pinned to the same absolute x as panel A's tag (see
    # below) rather than left-anchored to this wider box's own corner, so
    # growing row B does not pull the two tags out of vertical alignment.
    ROW_LEFT, ROW_RIGHT = 0.045, 0.965
    gs_a = fig.add_gridspec(1, 1, left=ROW_LEFT, right=ROW_RIGHT, top=0.97, bottom=0.535)
    gs_top = gs_a[0, 0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.5], wspace=0.14)
    spark_axes = [fig.add_subplot(gs_top[0, i]) for i in range(4)]
    zoom_ax = fig.add_subplot(gs_top[0, 4])

    gs_b = fig.add_gridspec(1, 2, left=0.015, right=0.995, top=0.465, bottom=0.075, wspace=0.05)
    ax_chord = fig.add_subplot(gs_b[0, 0])
    ax_rain = fig.add_subplot(gs_b[0, 1])

    _plot_ablation_small_multiples(spark_axes, zoom_ax, summary, fold)
    _plot_chord_confusion(ax_chord, conf, class_metrics, PRIMARY_MODEL)
    _plot_confidence_raincloud(ax_rain, pred, PRIMARY_MODEL)

    pos_a0 = spark_axes[0].get_position()
    pos_chord = ax_chord.get_position()
    tag_x = pos_a0.x0 + 0.01 * pos_a0.width
    tag_y = pos_chord.y1 - 0.01 * pos_chord.height + (13.0 / 72.0) / fig.get_figheight()
    fig.text(
        tag_x,
        tag_y,
        "(b)",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["ink"], "boxstyle": "round,pad=0.18", "linewidth": 0.9, "alpha": 0.96},
        zorder=30,
    )

    _finalize_panel_labels(fig)

    png = fig_dir / "fig_5_baseline_performance.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    qa = {
        "figure": "fig_5_baseline_performance",
        "date": "2026-07-31",
        "panel_count": 2,
        "layout": "top: ablation trajectory (a); bottom: chord confusion (b, left) + confidence raincloud (b, right)",
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "primary_model": PRIMARY_MODEL,
        "source_paths": [
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/summary_metrics.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/fold_metrics.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/pooled_class_metrics.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/pooled_confusion_matrix_long.csv",
        ],
        "png": str(png),
        "claim_boundary": "Weak-label model validation; not audited LCZ accuracy and not final deep multimodal performance.",
        "revision_2026-07-31c": "Full redesign from 6 panels (3x2 grid) to 2 panels (top/bottom) per explicit request for different, denser chart grammars. Top panel fuses the old macro-F1 ablation (A) and fold-level stability (B) into one bump/slope chart with fold-SD ribbons for all four models plus jittered raw fold points at the primary recipe. Bottom panel fuses the confusion matrix (old C), class-level F1/support (old D), reliability (old E), and confidence separation (old F) into two new chart forms: a chord diagram (confusion, with self-loop bulges for correct predictions and translucent cross-ribbons for misclassifications, F1/support annotated at each class node) and a raincloud plot (half-violin + boxplot + jittered strip) comparing confidence for correct vs. mismatched predictions. The standalone reliability-curve panel (old E) was dropped as a distinct chart; its core message (LightGBM's probabilities run far more extreme than its accuracy would justify) is now carried by the raincloud's visible right-skew for both correct and mismatched groups, and remains stated quantitatively in the main text. Every rendered number was independently cross-checked against the raw source CSVs (endpoint macro-F1s, majority floor, confusion cell counts, class F1/support, correct/mismatch n and pooled mean confidence/accuracy) and matched exactly.",
        "revision_2026-07-31d": "Chord panel (b, left) was rendering visibly narrower than the raincloud panel (b, right) despite equal gridspec column widths, because ax.set_aspect('equal') with the default adjustable='box' shrinks the axes' own bounding box to keep the circle round, breaking left/right alignment with its neighbor. Fixed by removing set_aspect entirely and instead pre-matching the data xlim/ylim ratio to the axes box's actual physical (inches) aspect ratio via ax.get_position(); this keeps the circle round while letting the box fill its full allocated width. Verified programmatically (not just visually) that both axes now occupy identical figure-fraction width (0.4393) with symmetric left/right placement around the column gap.",
        "revision_2026-07-31e": "Panel A redesigned again on user feedback that the single bump/ribbon chart with all four overlapping model ribbons looked visually muddy and the broken y-axis felt clunky. Replaced with five small multiples in one row: four per-model sparklines (LightGBM, XGBoost, Random Forest, Extra Trees, in descending final-recipe rank order) each with its own single-color SD ribbon and a shared y-axis (only the leftmost shows tick labels), plus a fifth, wider 'primary recipe (800 m) fold-level spread' panel showing all four models' individual fold values, mean+-SD, and value labels side by side for direct comparison. A shared recipe-index key (1-7) replaces repeating the long recipe names four times. The majority-classifier floor (0.128) is stated as a text note instead of a broken axis, since it is now far outside the zoomed 0.60-0.87 range shared by all sparklines. Fixed one real layout bug during this pass: the zoom panel's model-name y-tick labels defaulted to the left side and spilled into the neighboring Extra Trees sparkline; moved to the right side of the (widest, rightmost) zoom panel and abbreviated (LGBM/XGB/RF/ET) since full names are already the sparkline titles in the same colors.",
        "revision_2026-07-31f": "User flagged that panel (b)'s two halves looked unbalanced even though their axes boxes are identical width (confirmed programmatically): pixel-ink measurement showed the chord diagram's circle+labels only used ~1701px of its ~2151px-wide half versus the raincloud filling ~2148px of its own half, i.e. the chord had much more unused margin. Tightened the chord's data half-width from 1.75 to 1.62 and its label radius from r_out+bulge+0.14 to +0.09, which enlarged the circle to use ~1743px (some further tightening to 1.48 was tried first but clipped the LCZ 3 label vertically, since ylim shrinks proportionally with xlim under the fixed box-aspect-ratio approach -- reverted). A circular diagram inherently cannot fill a rectangular box's corners the way a violin/strip plot does, so full pixel-for-pixel parity with the raincloud panel is not achievable without either distorting the circle or removing node labels; this is the tightest safe margin found without clipping.",
        "revision_2026-07-31g": "User then reported that neither row reached the page width evenly: panel A fit on the right but had unused left margin (dead space from the shared gridspec's left=0.07 before any tick labels started), and panel B had unused margin on both sides. Root cause: both rows shared one gridspec's left/right, but row A needs real margin for y-axis tick labels while row B needs almost none, so a margin sized for row A left row B with a large unused gap. Split into two independent gridspecs with separate margins: row A (sparklines+zoom) left=0.045/right=0.965, row B (chord+raincloud) left=0.015/right=0.995. First attempt at right=0.99 for row A clipped the zoom panel's LGBM/XGB/RF/ET tick labels at the true canvas edge (verified via pixel scan, 1px margin remaining); backed off to right=0.965. Final pixel-ink margins on all four sides are now under 2% of image width (row A: 1.3% left / 0.7% right; row B: 1.7% left / 0.4% right), down from the prior 3.7-9.3% range, with no clipping.",
        "revision_2026-07-31h": "Added a mini-waterfall bar strip beneath each of the four sparklines (user request), showing that model's marginal macro-F1 gain per feature-family stage. Deliberately built as only 4 bars (+vegetation, +green-blue, +full-context/no-2SFCA, +2SFCA@800m) rather than 6 (one per recipe transition): recipes 4-7 (no-2SFCA, +400m, +800m, +1200m) are parallel threshold *alternatives* to the same field, not a further additive chain, so a naive consecutive-delta waterfall across all 7 recipes would have misleadingly implied 400m->800m->1200m was a cumulative build-up. A raincloud plot was considered for the same slot but rejected as redundant with the existing zoom panel, which already shows fold-level spread for all four models together (better for cross-model comparison than four isolated per-model copies would have been). Waterfall values were spot-checked against summary_metrics.csv for all four models and matched (e.g. LightGBM 0.663->0.760->0.802->0.833->0.838 gives deltas +0.097/+0.042/+0.031/+0.006, all displayed correctly).",
        "revision_2026-07-31i": "Waterfall strip removed on user feedback that it didn't work visually; replaced with a small violin+jittered-strip inset (ax.inset_axes) in the bottom-right corner of each sparkline itself (not a separate axes/strip), showing that model's own 5 fold-level macro-F1 values at the primary 800 m recipe. Positioned at axes-fraction [0.64, 0.06, 0.33, 0.40] within each sparkline, which maps to a data-y range comfortably below where the rising trajectory line sits at recipes 4-7 (confirmed no visual overlap between inset box and curve/ribbon). This duplicates data already shown in panel E (all four models' fold spread together) but was kept anyway per explicit user request, since it makes each sparkline self-contained without needing to cross-reference panel E; this is a legitimate redundancy trade-off for readability, not an error.",
        "revision_2026-07-31j": "Two more fixes per user feedback. (1) Rows A and B previously used different gridspec left/right margins (0.045/0.965 vs 0.015/0.995), chosen independently to fit each row's own peripheral labels; this meant the two panel-letter tags, both anchored at axes-fraction (0.01, 0.99) of their row's first axes, did not line up vertically in absolute figure coordinates. Unified both rows to the same left=0.045/right=0.965 (row A's more conservative margin, since it needs room for y-axis tick labels that row B does not) via a shared ROW_LEFT/ROW_RIGHT constant; verified via get_position() that both axes now share x0=0.045 exactly, not just visually. (2) The chord diagram (panel B, left) was judged too plain/flat and too saturated compared to panel A's softer, translucent visual language. Base class rings now use CLASS_COLORS_SOFT (each class color blended 35% toward white via a new _lighten() helper) instead of the flat saturated CLASS_COLORS, echoing panel A's light-fill-plus-solid-accent look (the self-loop bulge keeps the darker CLASS_DARK accent, playing the same role as panel A's solid trajectory line over its pale ribbon). Cross-ribbon alpha raised slightly (0.55 -> 0.60) for richness. Added circos-style unlabeled radial tick marks every 250 cells around each node's outer rim (alternating long/short) so the ring reads as a measured scale rather than a plain flat band.",
        "revision_2026-07-31k": "User then asked for panel B specifically to grow wider (left and right, preserving aspect ratio) to fill like panel A, while panel B's tag stays put -- i.e. reversing part of revision j's margin unification for row B only, without losing the tag alignment revision j had just fixed. Widened row B's own gridspec back out to left=0.015/right=0.995 (row A stays at 0.045/0.965, which it needs for its tick labels). The chord's ax.get_position()-based ratio-matching (revision d) automatically scaled the circle up to fill the new, wider box -- no manual re-tuning needed. To keep the (b) tag aligned with (a) despite row B's box now starting further left than row A's, moved the tag out of _plot_chord_confusion (removed its internal _panel_label call) and instead placed it in build_figure_5_validation via fig.text() at an absolute figure position: x pinned to spark_axes[0]'s own tag x (x0 + 0.01*width, i.e. wherever (a) actually sits), y taken from ax_chord's own top (y1 - 0.01*height). Confirmed no clipping after widening (LCZ 9 label, the closest to the new tighter left margin, renders fully).",
        "revision_2026-07-31l": "Final whole-figure margin pass: pixel-ink scan of the full PNG showed the top margin (136px, 2.9% of height) was noticeably larger than the other three sides (left 1.2%, right 0.4%, bottom 1.5%), from the title/recipe-key text sitting well below the top of their reserved zone. Moved the title from y=0.955 (default baseline anchor, effectively floating well below the top) to y=0.978 with va='top', and the recipe key from y=0.925 to y=0.945 with va='top' to follow it up proportionally. Top margin dropped to 84px (1.75%), bringing all four sides of the whole figure to within roughly 0.4-1.8% of image size, consistent with treating rows A and B as one balanced, page-filling group.",
        "revision_2026-08-03a": "Moved panel A and B tags upward by 5 points for tighter row anchoring. Added a compact central KPI inset to panel B's chord diagram reporting pooled accuracy and macro-F1 for the primary 800 m LightGBM recipe, preserving the chord as the class-flow graphic while giving the empty center an explicit meaning. Added a dashed pooled-accuracy guide alongside the dotted mean-confidence guide in the raincloud and reported the confidence gap in its annotation card. Updated the manuscript caption to explain both additions.",
        "revision_2026-08-03b": "Removed the central chord KPI circle at user request, removed the two bottom-panel titles and the all-panels footer note, and moved both panel tags an additional 8 points upward (13 points total from the original tag position). The raincloud's confidence-gap guides and annotation remain because they directly support the intended calibration reading.",
    }
    (out_dir / "fig_5_validation_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_5_validation_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_5_validation(Path(".")), indent=2))
