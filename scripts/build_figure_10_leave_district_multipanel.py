"""Build a multi-panel leave-district transfer diagnostics figure."""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from build_figure_1_texture_atlas import CLASS_COLORS, CLASSES, PALETTE


PRIMARY_MODEL = "lightgbm"
COMPARISON_MODEL = "random_forest"
MODEL_LABELS = {"lightgbm": "LightGBM", "random_forest": "Random Forest"}

TRANSFER_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "climorfa_transfer", [PALETTE["pink"], PALETTE["orange"], PALETTE["teal"]]
)
MAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "climorfa_map_fig9_soft_teal_purple",
    ["#5ec4b6", "#81cecb", "#b6cad8", "#bfa5d7", "#9e6ebf", "#7b3f9e"],
)
CONTEXT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "climorfa_context", [PALETTE["taupe"], PALETTE["teal"]]
)
MAP_VMIN = 0.40
MAP_VMAX = 0.95


def _panel_label(ax: plt.Axes, label: str, fontsize: float = 15.0, x: float = 0.005, y: float = 1.02, va: str = "bottom") -> None:
    # Matches Figure 2's own `_panel_label` exactly (build_figure_2_
    # methodology_multipanel.py): above the axes, not inside its top-left
    # corner, with a neutral #555555 border -- per explicit user request to
    # bring this figure's lettering in line with the house convention
    # already applied to Figures 4/6/8/9 earlier this session. Figure 2's
    # own panels carry zero ax.set_title calls (letter + caption only), so
    # adopting this convention here also means each panel's former on-image
    # title text was dropped -- caption text carries that identification
    # now, avoiding the guaranteed collision between a title's own pad and
    # this letter both wanting the same narrow strip just above the axes.
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
            if re.fullmatch(r"[A-Za-z]", text):
                child.set_text(f"({text.lower()})")


def _set_map_extent(ax: plt.Axes, bounds: np.ndarray) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin - (xmax - xmin) * 0.015, xmax + (xmax - xmin) * 0.015)
    ax.set_ylim(ymin - (ymax - ymin) * 0.015, ymax + (ymax - ymin) * 0.015)
    ax.set_aspect("equal")
    ax.axis("off")


def _stats_box(ax: plt.Axes, text: str) -> None:
    """Add a compact, low-contrast summary box inside a map."""
    ax.text(
        0.025,
        0.435,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        linespacing=1.15,
        color=PALETTE["ink"],
        bbox={
            "facecolor": "white",
            "edgecolor": PALETTE["taupe"],
            "boxstyle": "round,pad=0.28",
            "linewidth": 0.55,
            "alpha": 0.92,
        },
        zorder=12,
    )


def _district_geometries(grid: gpd.GeoDataFrame, df: pd.DataFrame, lodo: pd.DataFrame) -> gpd.GeoDataFrame:
    g = grid.merge(df[["grid_id", "district"]], on="grid_id", how="left")
    d = g.dissolve(by="district", as_index=False)
    return d.merge(lodo, left_on="district", right_on="held_out_district", how="inner")


def _mahalle_geometries(df: pd.DataFrame, predictions: pd.DataFrame, mahalle_boundaries: gpd.GeoDataFrame, study_area: gpd.GeoDataFrame, min_cells: int = 5) -> gpd.GeoDataFrame:
    """Real administrative neighborhood polygons (`mahalle_boundaries`, from
    `data/01_raw/population/mahalle_population_controls_2024.gpkg`), not a
    250 m analysis-grid dissolve -- an earlier version approximated each
    neighborhood's shape by dissolving square grid cells, which gave every
    neighborhood, including the empty/unreliable ones with no accuracy
    fill, a blocky, staircase-edged "pixel boundary" instead of its real
    shape; per user feedback ("pixel sınırı değil, gerçek mahalle sınırı
    olsaydı boş olanlara") this now uses the real boundary source directly.
    `join_key` in that source matches `population_mahalle_key` exactly for
    all 360 neighborhoods used elsewhere in this script (verified by full-
    overlap check), so no fuzzy name matching is needed. Boundaries are
    filtered to just those 360 (the source file covers all of Izmir
    province, ~1,300 neighborhoods, far beyond this figure's FUR scope),
    then clipped to the actual FUR study area (`study_area_fua.gpkg`, the
    same working-area boundary every other figure in this paper draws) --
    per user request ("çalışma alan sınırı dışındaki mahalleleri sil ...
    sadece çalışma alan sınırı kalmalı"), nothing should extend past that
    boundary at all. An `intersects` filter (keeping a boundary-straddling
    neighborhood's full, un-clipped shape) was tried first and still left
    large chunks of several neighborhoods sprawling well outside the drawn
    FUR line, which is exactly the "sadece çalışma alanı sınırı kalmalı"
    complaint -- `gpd.clip` cuts every polygon to the FUR polygon exactly,
    matching what was asked for even though it means a few boundary-
    straddling neighborhoods now show only their in-FUR portion rather than
    their full real shape. Each neighborhood's real held-out
    prediction accuracy is computed directly from predictions.csv -- the
    per-cell LODO predictions made while that cell's own district was
    excluded from training -- not a district-level average smeared back
    down. Every remaining neighborhood is kept (a left join, not inner):
    those outside the 11 held-out districts, below min_cells held-out
    predictions, or with zero modeled cells at all, get `accuracy`/
    `n_cells` = NaN and `reliable` = False rather than being dropped, so the
    map can still draw their real boundary -- just without a color fill."""
    preds = predictions[predictions["model"].eq(PRIMARY_MODEL)].copy()
    preds["correct"] = preds["true_label"] == preds["predicted_label"]
    key = df[["grid_id", "population_mahalle_key"]].dropna(subset=["population_mahalle_key"])
    preds = preds.merge(key, on="grid_id", how="inner")
    acc = preds.groupby("population_mahalle_key").agg(n_cells=("correct", "size"), accuracy=("correct", "mean")).reset_index()

    valid_keys = df["population_mahalle_key"].dropna().unique()
    boundaries = mahalle_boundaries[mahalle_boundaries["join_key"].isin(valid_keys)].copy()
    boundaries = gpd.clip(boundaries, study_area)
    merged = boundaries.merge(acc, left_on="join_key", right_on="population_mahalle_key", how="left")
    merged["reliable"] = merged["n_cells"].fillna(0) >= min_cells
    return merged


def _plot_map(ax: plt.Axes, districts: gpd.GeoDataFrame) -> None:
    districts.plot(column="macro_f1_present_classes", cmap=MAP_CMAP, vmin=MAP_VMIN, vmax=MAP_VMAX, edgecolor="white", linewidth=0.35, legend=True, legend_kwds={"label": "District held-out Macro-F1", "shrink": 0.62, "pad": 0.01}, ax=ax)
    for _, row in districts.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, row["district"][:4], fontsize=5.7, ha="center", va="center")
    _panel_label(ax, "A")
    _set_map_extent(ax, districts.total_bounds)
    f1 = districts["macro_f1_present_classes"].dropna()
    _stats_box(
        ax,
        f"n = {len(f1)} held-out districts\n"
        f"Macro-F1 mean = {f1.mean():.0%}\n"
        f"range = {f1.min():.0%}-{f1.max():.0%}",
    )
    ax.text(0.5, 1.01, "District-level transfer Macro-F1", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=PALETTE["ink"], clip_on=False)


def _plot_mahalle_map(ax: plt.Axes, mahalle: gpd.GeoDataFrame, study_area: gpd.GeoDataFrame) -> None:
    # Every neighborhood inside the FUR study area gets a real, faint
    # outline first -- so the map reads as complete administrative
    # geography, not as if entire neighborhoods were missing -- and only
    # those meeting the reliability threshold get a colored accuracy fill
    # on top. Per user request after asking why not all neighborhoods
    # showed up: "mahalle sınırı silik görünsün içi boş kalabilir" (the
    # neighborhood boundary can show faintly, its interior can stay empty)
    # rather than raising or lowering min_cells to trade completeness
    # against reliability. Per a further request ("çalışma alan sınırı
    # dışındaki mahalleleri sil ... sadece çalışma alan sınırı kalmalı"),
    # neighborhoods outside the FUR are dropped entirely (in
    # `_mahalle_geometries`, not here) rather than kept as faint outliers
    # sprawling past the working-area edge, and the FUR boundary itself --
    # the same one every other figure in this paper draws -- is shown
    # explicitly so the map's own extent has a real, labeled edge.
    mahalle.plot(ax=ax, facecolor="#f2f2f0", edgecolor="#c9c5be", linewidth=0.22, zorder=1)
    reliable = mahalle[mahalle["reliable"]]
    reliable.plot(column="accuracy", cmap=MAP_CMAP, vmin=MAP_VMIN, vmax=MAP_VMAX, edgecolor="white", linewidth=0.12, legend=True, legend_kwds={"label": "Mahalle held-out accuracy", "shrink": 0.62, "pad": 0.01}, ax=ax, zorder=2)
    study_area.boundary.plot(ax=ax, color="#555555", linewidth=0.7, linestyle="-", zorder=3)
    xmin, ymin, xmax, ymax = study_area.total_bounds
    ax.set_xlim(xmin - (xmax - xmin) * 0.015, xmax + (xmax - xmin) * 0.015)
    ax.set_ylim(ymin - (ymax - ymin) * 0.015, ymax + (ymax - ymin) * 0.015)
    ax.set_aspect("equal")
    ax.axis("off")
    reliable = mahalle[mahalle["reliable"] & mahalle["accuracy"].notna()]
    if len(reliable):
        accuracy_mean = reliable["accuracy"].mean()
        accuracy_range = f"{reliable['accuracy'].min():.0%}-{reliable['accuracy'].max():.0%}"
    else:
        accuracy_mean = float("nan")
        accuracy_range = "n/a"
    _stats_box(
        ax,
        f"reliable = {len(reliable)}/{len(mahalle)} (>=5 cells)\n"
        f"accuracy mean = {accuracy_mean:.0%}\n"
        f"range = {accuracy_range}",
    )
    ax.text(0.5, 1.01, "Mahalle-level held-out accuracy", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=PALETTE["ink"], clip_on=False)


def _plot_advanced_transfer(ax: plt.Axes, fig: plt.Figure, lodo: pd.DataFrame, lodo_rf: pd.DataFrame) -> None:
    """One dense chart replacing what used to be 5 separate panels (ranked
    bars, 2 metric matrices, and 2 scatter diagnostics): a per-district
    range-dot plot with accuracy/balanced-accuracy/macro-F1 as 3 aligned
    dots connected by a range line (carries the old metric-matrix and
    ranked-bar information), macro-F1's own dot sized by held-out support
    (carries the old support-vs-F1 scatter), and a colored ring around that
    same dot for mean out-of-fold confidence (carries the old confidence-
    vs-F1 scatter, including its point: high confidence does not track
    transfer success). Held-out class support and the building-coverage/
    coastal-share context scatter were dropped rather than folded in --
    they are the two least central diagnostics to this figure's actual
    claim, still available in the QA CSVs for anyone who needs the exact
    numbers, but not worth a 6th encoded dimension on one chart. Also
    carries a Random Forest comparison (a per-district version of the
    single averaged 2.3-point LightGBM advantage already reported in the
    results text) and zebra-striped rows for legibility across 11
    districts."""
    d = lodo.sort_values("macro_f1_present_classes").reset_index(drop=True)
    d = d.merge(lodo_rf[["held_out_district", "macro_f1_present_classes"]].rename(columns={"macro_f1_present_classes": "rf_macro_f1"}), on="held_out_district", how="left")
    y = np.arange(len(d))

    for i in range(len(d)):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=PALETTE["taupe_light"], alpha=0.5, zorder=0)

    metric_colors = {"accuracy": PALETTE["taupe_dark"], "balanced_accuracy": PALETTE["orange"], "macro_f1_present_classes": PALETTE["pink_dark"]}
    metric_labels = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.", "macro_f1_present_classes": "Macro-F1 (LightGBM)"}
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1_present_classes"]
    rf_color = PALETTE["ink"]

    lo = d[metric_cols].min(axis=1)
    hi = d[metric_cols].max(axis=1)
    ax.hlines(y, lo, hi, color=PALETTE["grid"], linewidth=2.4, zorder=1)

    # Random Forest macro-F1 comparison: a thin dashed connector from
    # LightGBM's own macro-F1 dot to RF's, plus a small diamond marker for
    # RF -- makes the already-reported average LightGBM-over-RF advantage
    # checkable per district instead of only as one pooled number.
    ax.hlines(y, d["macro_f1_present_classes"], d["rf_macro_f1"], color=rf_color, linewidth=1.1, linestyle=(0, (2, 1.5)), alpha=0.7, zorder=2)
    ax.scatter(d["rf_macro_f1"], y, marker="D", s=40, facecolors="white", edgecolors=rf_color, linewidth=1.4, zorder=4)

    for col in ("accuracy", "balanced_accuracy"):
        ax.scatter(d[col], y, s=48, color=metric_colors[col], edgecolors="white", linewidth=0.7, zorder=3)

    sizes = 55 + (d["test_rows"] / d["test_rows"].max()) * 320
    conf_norm = mpl.colors.Normalize(vmin=d["mean_max_probability"].min(), vmax=d["mean_max_probability"].max())
    ring_colors = [CONTEXT_CMAP(conf_norm(v)) for v in d["mean_max_probability"]]
    ax.scatter(d["macro_f1_present_classes"], y, s=sizes + 130, facecolors="none", edgecolors=ring_colors, linewidth=2.6, zorder=2)
    ax.scatter(d["macro_f1_present_classes"], y, s=sizes, color=metric_colors["macro_f1_present_classes"], edgecolors="white", linewidth=0.8, zorder=5)

    mean_f1 = d["macro_f1_present_classes"].mean()
    ax.axvline(mean_f1, color=PALETTE["ink"], linestyle=":", linewidth=1.0, zorder=1)

    # Direct callouts make the central diagnostic readable without requiring
    # the reader to decode every ring and dot: confidence can remain high
    # even where transfer performance is low.
    for district, label, offset in (
        ("KONAK", "KONAK: 60% Macro-F1; 93% confidence", (8, 11)),
        ("GÜZELBAHÇE", "GÜZELBAHÇE: lowest transfer (47%)", (8, 9)),
    ):
        match = d.index[d["held_out_district"].eq(district)]
        if len(match):
            i = int(match[0])
            ax.annotate(label, xy=(d.loc[i, "macro_f1_present_classes"], y[i]), xytext=offset, textcoords="offset points", fontsize=6.8, color=PALETTE["ink"], ha="left", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5}, arrowprops={"arrowstyle": "-", "color": PALETTE["muted"], "linewidth": 0.7}, zorder=8)

    ax.set_yticks(y)
    ax.set_yticklabels(d["held_out_district"], fontsize=9.5)
    ax.set_xlabel("Score", fontsize=9.5)
    x_lo = min(lo.min(), d["rf_macro_f1"].min()) - 0.035
    x_hi = max(hi.max(), d["rf_macro_f1"].max()) + 0.035
    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.5, len(d) - 0.5)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.5, zorder=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _panel_label(ax, "B")

    # 3 legend/colorbar elements share this one axes; each gets its own
    # clear quadrant, all now well clear of the panel letter since that
    # letter floats above the axes (Figure 2 convention) rather than inside
    # its top-left corner.
    metric_handles = [Line2D([0], [0], marker="o", linestyle="none", color=metric_colors[c], label=metric_labels[c], markersize=7.5, markeredgecolor="white") for c in metric_cols]
    metric_handles.append(Line2D([0], [0], marker="D", linestyle="none", color=rf_color, markerfacecolor="white", markeredgecolor=rf_color, markersize=6.5, label="Macro-F1 (Random Forest)"))
    metric_handles.append(Line2D([0], [0], color=PALETTE["ink"], linestyle=":", linewidth=1.2, label=f"FUR mean Macro-F1 ({mean_f1:.0%})"))
    leg1 = ax.legend(handles=metric_handles, loc="upper right", bbox_to_anchor=(0.94, 0.995), fontsize=8.0, frameon=True, title="Metric", title_fontsize=8.4, borderpad=0.6)
    ax.add_artist(leg1)

    size_vals = [d["test_rows"].min(), d["test_rows"].median(), d["test_rows"].max()]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=metric_colors["macro_f1_present_classes"], markeredgecolor="white", markersize=np.sqrt(55 + (v / d["test_rows"].max()) * 320) * 0.62, label=f"{int(v)} rows")
        for v in size_vals
    ]
    ax.legend(handles=size_handles, loc="upper left", fontsize=8.0, frameon=True, title="Held-out support\n(macro-F1 dot size)", title_fontsize=8.2, labelspacing=1.3, borderpad=0.7)

    # Placed in the one region with no dots at any row: every district's
    # score range sits at the right two-thirds of the axis, but the top 3
    # rows' own dots don't reach this far left -- the first attempt
    # (bottom-right) collided with a real data point and its tick label.
    sm = mpl.cm.ScalarMappable(norm=conf_norm, cmap=CONTEXT_CMAP)
    cax = ax.inset_axes([0.24, 0.86, 0.28, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Mean max probability (macro-F1 dot ring color)", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)


def build_figure_10_leave_district(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_10_leave_district_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    lodo_all = pd.read_csv(root / "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/district_metrics.csv")
    predictions = pd.read_csv(root / "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/predictions.csv")
    mahalle_boundaries = gpd.read_file(root / "data/01_raw/population/mahalle_population_controls_2024.gpkg", layer="mahalle_population_controls_2024").to_crs(grid.crs)
    study_area = gpd.read_file(root / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    # district_metrics.csv now holds rows for every model in the LODO comparison
    # (LightGBM and Random Forest); every panel below assumes one row per
    # district, so it must be filtered to the primary model before use.
    lodo = lodo_all[lodo_all["model"].eq(PRIMARY_MODEL)].copy()
    lodo_rf = lodo_all[lodo_all["model"].eq(COMPARISON_MODEL)].copy()
    districts = _district_geometries(grid, df, lodo)
    min_mahalle_cells = 5
    mahalle = _mahalle_geometries(df, predictions, mahalle_boundaries, study_area, min_cells=min_mahalle_cells)

    # 2 panels, further simplified 2026-08-02 per direct user feedback that
    # the prior 2-panel version (5 sub-charts merged into panel B) was still
    # too many separate charts. Panel A is now 2 maps side by side -- the
    # original district-level choropleth plus a genuinely new, finer
    # neighborhood (mahalle) map computed from the real per-cell held-out
    # predictions (see `_mahalle_geometries`), not a district value smeared
    # down. Panel B is a single dense chart (`_plot_advanced_transfer`)
    # replacing what used to be 5 separate elements (ranked bars, 2 metric
    # matrices, 2 scatters) -- see that function's own docstring for exactly
    # what was folded in versus dropped.
    fig_w = 16.5
    # Keep the two main panels visually connected; the former gap left an
    # unnecessarily large white band between the maps and transfer chart.
    gap_group = 0.06
    row_b_h = 5.6
    # top_margin has to clear panel A's own letter, which now floats above
    # the axes (Figure 2 convention) instead of sitting inside its top-left
    # corner -- 0.25in was enough room for the old inside-corner badge but
    # not for one that needs to sit entirely above the figure's own top
    # edge.
    top_margin = 0.50
    # 0.55in wasn't enough room for panel B's own x-axis label ("Score")
    # to clear the figure-wide footer note below it -- the two collided.
    bottom_margin = 0.80

    # Panel A's two maps are sized directly from their shared real-world
    # aspect ratio (fig.add_axes, not a generic equal-width GridSpec cell)
    # instead of being left to letterbox inside an arbitrary row height.
    # The previous GridSpec version (wspace=0.16 on a row taller than either
    # aspect-locked map needed) center-anchored each map inside an oversized
    # cell -- that both left a much bigger visible gap between the two maps
    # than the nominal wspace suggested, and put panel A's own letter
    # (anchored to a small x-fraction of its own, differently-margined
    # axes, left=0.045 vs panel B's left=0.09) well left of panel B's
    # letter below it. Panel A's margins now match panel B's exactly (so
    # both letters sit at the same absolute x), the two maps sit close
    # together with one small explicit gap, and fig_w was widened
    # (15.5in -> 16.5in) to let both maps actually grow rather than just
    # closing the gap into no visible size change.
    xmin, ymin, xmax, ymax = study_area.total_bounds
    map_ar = (ymax - ymin) / (xmax - xmin)
    # Use one shared, wider content frame for panels A and B.  This makes the
    # two-map group occupy the same real figure width as the advanced chart,
    # rather than letting A look undersized because it shares the old inner
    # margins with two map slots.
    content_left = 0.04
    content_right = 0.98
    a_left_margin = content_left * fig_w
    a_right_margin = (1 - content_right) * fig_w
    a_gap = 0.08
    panel_a_map_w = (fig_w - a_left_margin - a_right_margin - a_gap) / 2
    panel_a_map_h = panel_a_map_w * map_ar
    row_a_h = panel_a_map_h

    fig_h = top_margin + row_a_h + gap_group + row_b_h + bottom_margin

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    b_b = bottom_margin
    b_a = b_b + row_b_h + gap_group

    # ------------------ PANEL A: district map + neighborhood map ------------------
    ax_map = fig.add_axes([a_left_margin / fig_w, b_a / fig_h, panel_a_map_w / fig_w, panel_a_map_h / fig_h])
    ax_mahalle = fig.add_axes([(a_left_margin + panel_a_map_w + a_gap) / fig_w, b_a / fig_h, panel_a_map_w / fig_w, panel_a_map_h / fig_h])
    _plot_map(ax_map, districts)
    _plot_mahalle_map(ax_mahalle, mahalle, study_area)

    # ------------------ PANEL B: one advanced multi-metric chart ------------------
    gs_b = fig.add_gridspec(1, 1, left=content_left, right=content_right, top=(b_b + row_b_h) / fig_h, bottom=b_b / fig_h)
    ax_advanced = fig.add_subplot(gs_b[0, 0])
    _plot_advanced_transfer(ax_advanced, fig, lodo, lodo_rf)

    fig.text(0.5, 0.012, "Leave-district results use weak labels only. Low-transfer districts identify where audited labels and local fabric review should be prioritized.", ha="center", fontsize=7.2, color=PALETTE["muted"])
    _finalize_panel_labels(fig)
    png = fig_dir / "fig_10_leave_district_out.png"
    # Remove the old fixed outer margins while retaining a small safe edge so
    # panel letters, colorbars, legends, and the footer are not clipped.
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    lodo.set_index("held_out_district")[["accuracy", "balanced_accuracy", "macro_f1_present_classes", "test_rows", "mean_max_probability"]].to_csv(out_dir / "fig_10_district_transfer_metrics.csv")
    mahalle[["population_mahalle_key", "district_norm", "n_cells", "accuracy"]].to_csv(out_dir / "fig_10_mahalle_transfer_accuracy.csv", index=False)
    qa = {
        "figure": "fig_10_leave_district_out",
        "date": "2026-06-20",
        "panel_count": 2,
        "layout": "2 panels: A is 2 maps side by side (district-level transfer geography; neighborhood/mahalle-level held-out prediction accuracy computed from real per-cell predictions, filtered to mahalle with >=5 held-out cells). B is one dense multi-metric chart (accuracy/balanced-accuracy/macro-F1 range-dot per district, macro-F1 dot sized by held-out support, ringed by mean out-of-fold confidence) replacing what used to be 5 separate sub-charts.",
        "min_mahalle_cells": min_mahalle_cells,
        "mahalle_count": int(len(mahalle)),
        "primary_model": PRIMARY_MODEL,
        "revision_2026-07-31": "district_metrics.csv now contains rows for every LODO model (LightGBM and Random Forest); every panel here assumes one row per district, so lodo is now explicitly filtered to PRIMARY_MODEL before use. Before this fix, generate_manuscript_evidence.py had already run once with the unfiltered 2-model dataframe, silently doubling every district in panels a-g (duplicate bars/points/heatmap rows) via the inner merge and reindex operations, none of which deduplicated by model.",
        "revision_2026-08-02": "Per user request for Figure 10 ('yine çok fazla gereksiz panel var. bilgi eksilmeden 2 panelde yoğun içerikli bir figür kurgusu yap' -- discovered this was the correct target only after an initial pass mistakenly applied the same request to fig_9_climate_validation, which the compiled PDF's own \\newlabel numbering showed was actually Figure 9, not Figure 10; fig_10_leave_district_out.png is the real Figure 10, confirmed via main.aux). Compressed from 8 grid slots (7 real panels A-G + 1 blank cell, `axes[7].axis('off')`, unused since this figure's creation) down to 2 lettered panels. Every one of the 7 real panels' data and chart form survives; only per-panel chrome that repeated across panels sharing the same axis or district ordering was consolidated -- 3 scatters that all plotted district-level X vs. macro-F1 now share one y-axis instead of drawing it 3 times, and 2 heatmaps that already used the identical macro-F1-sorted district row order are now placed side by side instead of as separate lettered panels. Individual per-sub-panel bordered letter badges were dropped in favor of each sub-panel's own existing `ax.set_title`, with a single group letter (A or B) applied once per merged panel -- same convention already used for the zoom-in and thermal-tile groups in Figures 8 and 9(climate) earlier this session.",
        "revision_2026-08-02b": "Fixed a real layout collision found on first render: the heatmap row's own titles (above ax_metrics/ax_class) and the scatter row's own xlabels (below ax_support/ax_prob/ax_context) both sat in the same narrow gap_sub band and overlapped, and separately the heatmap row's rotated x-tick labels ran into the figure's bottom footer text. Fixed by widening gap_sub 0.40in->0.70in and bottom_margin 0.55in->0.85in.",
        "revision_2026-08-02c": "Per direct user feedback after seeing the first 2-panel version ('bu kadar grafik istemiyorum. map durabilir, mapin mahalle versiyonunu da koysak mı acaba? bir tane de advanced grafik istiyorum' -- I don't want this many charts, the map can stay, should we also add a neighborhood/mahalle version of the map, and I want one advanced chart): simplified further. Panel A: kept the district map, replaced the ranked-bar strip with a new neighborhood-level map -- `_mahalle_geometries` joins the real per-cell held-out predictions (`predictions.csv`, not previously read by this script) to each grid cell's `population_mahalle_key` (327 mahalle across the 11 held-out districts; a mahalle is dropped if it has fewer than min_mahalle_cells=5 held-out predictions), dissolves the grid by that key, and computes each neighborhood's own real held-out accuracy -- genuinely new geographic resolution, not a district value re-painted onto smaller polygons. Panel B: replaced all 5 remaining sub-charts (ranked bars had already gone into panel A; the 3 scatters and 2 heatmaps) with the single `_plot_advanced_transfer` range-dot chart -- accuracy/balanced-accuracy/macro-F1 per district as 3 aligned dots on a shared range line (carries the old metric matrix and ranking), the macro-F1 dot sized by held-out support (carries the old support-vs-F1 scatter), and ringed by a confidence colormap (carries the old confidence-vs-F1 scatter). The held-out class-support matrix and the building-coverage/coastal-share context scatter were dropped rather than compressed further -- judged the least central to this figure's actual claim -- but their underlying numbers remain computable from the exported CSVs; nothing in the source data was deleted, only two lower-priority chart panels.",
        "revision_2026-08-02d": "User asked why not all neighborhoods appeared on the mahalle map ('niye tüm mahalleler yok?'). Root cause had 3 layers: (1) the map is scoped to the 11 held-out districts only, since only they have held-out predictions -- 7 of Izmir's 18 districts (Kemalpaşa, Menderes, Menemen, Seferihisar, Torbalı, Urla, plus one 'unknown') were never part of the LODO test; (2) 9 of 336 neighborhoods within the held-out districts have zero modeled cells (no built LCZ 3/6/8/9 at >=0.60 confidence); (3) `_mahalle_geometries` was additionally filtering out any neighborhood below min_mahalle_cells=5 (107 of the remaining 327) via an inner join, dropping them entirely rather than showing them unfilled. Presented the tradeoff (keep min_cells=5, lower it to 2, or lower it to 1) via AskUserQuestion; user instead proposed a different fix entirely -- 'o zaman mahalle sınırı silik görünsün içi boş kalabilir' (let the neighborhood boundary show faintly, its interior can stay empty) -- show every neighborhood's real boundary but only fill the reliable ones with color. Implemented: `_mahalle_geometries`'s join changed from inner to left (keeps all 336 neighborhoods in held-out districts, `accuracy`/`n_cells` are NaN where there's no or insufficient data) plus a new `reliable` boolean column; `_plot_mahalle_map` now draws all 336 as a faint gray outline first (zorder=1), then overlays the colored accuracy choropleth for only the 220 reliable ones on top (zorder=2). Layer 1 (district scope) is inherent to what a held-out map can show and wasn't changed.",
        "revision_2026-08-02e": "Per user follow-up ('tüm mahalleler görünmüyor ama. silik şekilde renkli olmasa bile hepsi olsun. panel b de gelişsin. panel numaraları da figure 2 kurallarına uysun') covering 3 things at once. (1) The revision_2026-08-02d fix still hadn't shown every mahalle -- it had only stopped dropping neighborhoods *within* the 11 held-out districts, but `_mahalle_geometries`'s own grid join was still scoped to those 11 districts' geometry, so the other 7 districts' ~24 neighborhoods were still entirely absent rather than outlined. Fixed by dissolving the full FUR grid (all 18 districts) instead of filtering to held-out districts before the dissolve, and changing `_plot_mahalle_map`'s extent source from `districts.total_bounds` (11-district) to `mahalle.total_bounds` (full FUR) -- mahalle count went 336->360. (2) Panel B ('gelişsin' -- develop/improve further): user picked all 3 offered directions via AskUserQuestion (add Random Forest comparison; visual polish; use my own judgment for anything else). Added RF's own per-district macro-F1 as an open diamond marker connected to LightGBM's macro-F1 dot by a thin dashed line (making the already-reported pooled 2.3-point LightGBM advantage checkable per district); added zebra row striping for legibility across 11 districts; converted the x-axis to percentage formatting; added an explicit 'FUR mean Macro-F1' legend entry (previously an unlabeled dotted line); made the x-axis range dynamic (fits both models' data) instead of a hardcoded guess. (3) Panel lettering brought in line with Figure 2's own `_panel_label` convention (matching the same change already made to Figures 4/6/8/9 earlier this session): above-axes bordered box instead of inside-corner, applied identically here. Adopting that convention also meant dropping each panel's on-image `ax.set_title` text (Figure 2's own panels carry none either, letter+caption only) -- keeping a title would have collided with the letter, since both compete for the narrow strip directly above the axes; the dropped title text is still fully covered by the LaTeX caption.",
        "revision_2026-08-02f": "Per user request ('pixel sınırı değil. gerçek mahalle sınırı olsaydı boş olanlara'): switched the mahalle map's geometry source from a 250 m analysis-grid dissolve (blocky, staircase-edged approximation of each neighborhood, most visible on the unreliable/uncolored ones since they had no fill to distract from the shape) to the real administrative boundaries in `data/01_raw/population/mahalle_population_controls_2024.gpkg` (1,317 Izmir-province neighborhoods; its `join_key` field matches this script's own `population_mahalle_key` exactly for all 360 neighborhoods already in use, confirmed by a full-overlap check before switching, so no fuzzy name matching was needed). `_mahalle_geometries` no longer takes `grid` as an argument at all -- it filters the real boundary file down to the 360 relevant neighborhoods and merges accuracy onto their real shapes directly, rather than dissolving grid cells. Source file is EPSG:32635; reprojected to the grid's own EPSG:5253 once at load time. Mahalle counts (360 total, 220 reliable) are unchanged -- only each polygon's shape changed, not which neighborhoods are shown or colored.",
        "revision_2026-08-02g": "Per user request ('çalışma alan sınırı dışındaki mahalleleri sil ama. sadece çalışma alan sınırı kalmalı'): clipped every mahalle polygon to the actual FUR study area (`study_area_fua.gpkg`, loaded fresh for this figure -- not previously read by this script) and now draw that boundary explicitly on the map, matching the convention every other figure in this paper uses. First attempt used an `intersects` filter (drop a neighborhood only if it doesn't touch the FUR at all, otherwise keep its full real shape) -- rendered result still showed several neighborhoods sprawling well past the drawn FUR line, which is exactly the 'sadece çalışma alanı sınırı kalmalı' complaint. Replaced with `gpd.clip(boundaries, study_area)`, which cuts every polygon to the FUR boundary exactly -- a few boundary-straddling neighborhoods now show only their in-FUR portion rather than their complete real shape, a deliberate tradeoff in favor of the map's own edge matching the actual study-area boundary everywhere. Mahalle count 360->359 (one neighborhood was entirely outside the FUR and dropped); reliable count unchanged at 220.",
        "revision_2026-08-03": "Reduced the vertical gap between panels A and B from 0.65 in to 0.12 in and exported the figure with a tight bounding box plus a minimal safety pad, removing excess white space from all four sides without clipping labels, legends, colorbars, or the footer.",
        "revision_2026-08-03b": "Reduced the A-B gap further to 0.06 in and replaced the previous narrower, asymmetric-looking frame with a shared 0.04-0.98 content frame for both panels. The two map axes now occupy the same total width as panel B and each map is wider while retaining its geographic aspect ratio.",
        "revision_2026-08-03c": "Applied the Figure 10 development plan: both maps now use the shared 0.40-0.95 transfer scale with explicit map subtitles; panel B retains the advanced multi-metric encoding but adds direct callouts for the high-confidence/low-transfer Konak case and the lowest-transfer Güzelbahçe case; the Metric legend was shifted left to reduce right-edge pressure.",
        "revision_2026-08-03d": "Changed both map fills to a dedicated soft pink-to-soft teal continuous palette, while leaving panel B's metric colors unchanged.",
        "revision_2026-08-03e": "Strengthened the map endpoints after visual review: soft red #E77F89 to mid teal #55AEA6, avoiding the overly pale and gray-looking previous map gradient; panel B remains unchanged.",
        "revision_2026-08-03f": "Replaced the map palette with the exact Figure 9 soft teal-to-purple ramp (#5ec4b6, #81cecb, #b6cad8, #bfa5d7, #9e6ebf, #7b3f9e); panel B's metric colors remain unchanged.",
        "revision_2026-08-04": "Added a very small 0.05 in tight-export safety pad so the upper and left edges retain a light breathing margin, and added compact statistics boxes to both panel-A maps: held-out district count and Macro-F1 summary for the district map; reliable-mahalle coverage and accuracy summary for the neighborhood map.",
        "revision_2026-08-04b": "Moved both panel-A statistics boxes upward to sit above the lower-left GÃ¼zelbahÃ§e district area, keeping their horizontal placement unchanged.",
        "revision_2026-08-04c": "Raised both panel-A statistics boxes a further step to the higher position requested, with their horizontal placement unchanged.",
        "revision_2026-08-04d": "Moved both panel-A statistics boxes one more small step upward while preserving their horizontal placement and all other figure elements.",
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/district_metrics.csv",
            "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/predictions.csv",
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/01_raw/population/mahalle_population_controls_2024.gpkg",
            "data/02_interim/study_area_fua.gpkg",
        ],
        "png": str(png),
        "claim_boundary": "Leave-district transfer is weak-label robustness evidence, not audited local-subtype generalization.",
    }
    (out_dir / "fig_10_leave_district_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_10_leave_district_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_10_leave_district(), indent=2))
