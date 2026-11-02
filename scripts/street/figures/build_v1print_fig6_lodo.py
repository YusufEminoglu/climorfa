"""Print-legible version of main-text Figure 6 (Stage 1 leave-district-out transfer).

Same panels, chart types, colours, data and numbers as the v1 figure built by
``scripts/build_figure_10_leave_district_multipanel.py``; only the type scale and
the layout change.  Authored at 13.0 in wide and passed through
``climorfa_style.scale_text`` so every text element prints at >= 6 pt at 190 mm.

Run (cwd = project root):
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_v1print_fig6_lodo
Writes only paper/figures/v2/main/fig_10_leave_district_out.png.

Layout changes relative to v1
  * figure 13.0 in wide (v1: 16.5 in), all panels placed in inches;
  * map colour bars drawn in their own axes beside each map (v1: geopandas legend);
  * the two map statistics boxes moved out of the map area to a one-line box under each map;
  * panel (b): the metric legend (2 columns), support legend (1 row) and confidence colour bar moved
    out of the data area into a key strip above the chart; colour-bar label wrapped onto two lines;
    the chart now spans the full width;
  * "Mahalle-level" relabelled "Neighbourhood" (manuscript wording);
  * the footer sentence of v1 is omitted (it was cropped from the manuscript version).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import geopandas as gpd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import build_figure_10_leave_district_multipanel as V1  # noqa: E402
from scripts.street.figures import climorfa_style as S  # noqa: E402

PALETTE = V1.PALETTE
OUT = ROOT / "paper/figures/v2/main/fig_10_leave_district_out.png"

# ---------------------------------------------------------------- geometry (inches)
FIG_W = S.V2_FIG_WIDTH_IN
M_LEFT = 0.10
MAP_W = 5.05
CB_GAP, CB_W = 0.10, 0.17
CB_ROOM = 1.05            # colour bar + ticks + rotated label
MAP_GAP = 0.25
STATS_H = 0.42            # one-line statistics box under each map
TITLE_H = 0.45            # map title + panel letter above maps
B_LEFT = 1.75             # room for district tick labels
B_RIGHT = 0.15
KEY_H = 1.20              # key strip (legends + colour bar) above panel (b)
B_H = 6.6
B_BOTTOM = 0.80
GAP_AB = 0.30             # clearance between the stats boxes and the key strip
TOP = 0.15


def load():
    root = ROOT
    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    lodo_all = pd.read_csv(root / "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/district_metrics.csv")
    predictions = pd.read_csv(root / "outputs/modeling/leave_district_out_v8_2026-07-27_coastline_fix/predictions.csv")
    mahalle_boundaries = gpd.read_file(root / "data/01_raw/population/mahalle_population_controls_2024.gpkg",
                                       layer="mahalle_population_controls_2024").to_crs(grid.crs)
    study_area = gpd.read_file(root / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    lodo = lodo_all[lodo_all["model"].eq(V1.PRIMARY_MODEL)].copy()
    lodo_rf = lodo_all[lodo_all["model"].eq(V1.COMPARISON_MODEL)].copy()
    districts = V1._district_geometries(grid, df, lodo)
    mahalle = V1._mahalle_geometries(df, predictions, mahalle_boundaries, study_area, min_cells=5)
    return districts, mahalle, study_area, lodo, lodo_rf


def stats_line(fig, ax, text):
    """v1 ``_stats_box`` content and style, as one line centred under the map (outside the data)."""
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.06 / fig.get_size_inches()[1], text, ha="center", va="top",
             fontsize=7.0, color=PALETTE["ink"],
             bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.28",
                   "linewidth": 0.55, "alpha": 0.92}, zorder=12)


def map_colorbar(fig, ax, label):
    bb = ax.get_position()
    h = bb.height * 0.62
    cax = fig.add_axes([bb.x1 + CB_GAP / FIG_W, bb.y0 + (bb.height - h) / 2, CB_W / FIG_W, h])
    sm = mpl.cm.ScalarMappable(cmap=V1.MAP_CMAP, norm=mpl.colors.Normalize(V1.MAP_VMIN, V1.MAP_VMAX))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label)
    cb.set_ticks(np.round(np.arange(0.4, 0.951, 0.1), 1))
    return cb


def plot_district_map(fig, ax, districts):
    districts.plot(column="macro_f1_present_classes", cmap=V1.MAP_CMAP, vmin=V1.MAP_VMIN, vmax=V1.MAP_VMAX,
                   edgecolor="white", linewidth=0.35, ax=ax)
    for _, row in districts.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, row["district"][:4], fontsize=5.7, ha="center", va="center")
    V1._panel_label(ax, "A")
    V1._set_map_extent(ax, districts.total_bounds)
    map_colorbar(fig, ax, "District held-out Macro-F1")
    f1 = districts["macro_f1_present_classes"].dropna()
    stats_line(fig, ax, f"n = {len(f1)} held-out districts;  Macro-F1 mean = {f1.mean():.0%};  "
                        f"range = {f1.min():.0%}-{f1.max():.0%}")
    ax.text(0.5, 1.01, "District-level transfer Macro-F1", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color=PALETTE["ink"], clip_on=False)


def plot_mahalle_map(fig, ax, mahalle, study_area):
    mahalle.plot(ax=ax, facecolor="#f2f2f0", edgecolor="#c9c5be", linewidth=0.22, zorder=1)
    reliable = mahalle[mahalle["reliable"]]
    reliable.plot(column="accuracy", cmap=V1.MAP_CMAP, vmin=V1.MAP_VMIN, vmax=V1.MAP_VMAX, edgecolor="white",
                  linewidth=0.12, ax=ax, zorder=2)
    study_area.boundary.plot(ax=ax, color="#555555", linewidth=0.7, linestyle="-", zorder=3)
    V1._set_map_extent(ax, study_area.total_bounds)
    map_colorbar(fig, ax, "Neighbourhood held-out accuracy")
    reliable = mahalle[mahalle["reliable"] & mahalle["accuracy"].notna()]
    acc_mean = reliable["accuracy"].mean()
    acc_rng = f"{reliable['accuracy'].min():.0%}-{reliable['accuracy'].max():.0%}"
    stats_line(fig, ax, f"reliable = {len(reliable)}/{len(mahalle)} (>=5 cells);  accuracy mean = {acc_mean:.0%};  "
                        f"range = {acc_rng}")
    ax.text(0.5, 1.01, "Neighbourhood held-out accuracy", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, fontweight="bold", color=PALETTE["ink"], clip_on=False)


def plot_transfer(fig, ax, key_rect, lodo, lodo_rf):
    """v1 ``_plot_advanced_transfer`` with the legends and colour bar moved to a key column."""
    d = lodo.sort_values("macro_f1_present_classes").reset_index(drop=True)
    d = d.merge(lodo_rf[["held_out_district", "macro_f1_present_classes"]]
                .rename(columns={"macro_f1_present_classes": "rf_macro_f1"}), on="held_out_district", how="left")
    y = np.arange(len(d))
    for i in range(len(d)):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=PALETTE["taupe_light"], alpha=0.5, zorder=0)

    metric_colors = {"accuracy": PALETTE["taupe_dark"], "balanced_accuracy": PALETTE["orange"],
                     "macro_f1_present_classes": PALETTE["pink_dark"]}
    metric_labels = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
                     "macro_f1_present_classes": "Macro-F1 (LightGBM)"}
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1_present_classes"]
    rf_color = PALETTE["ink"]
    lo = d[metric_cols].min(axis=1)
    hi = d[metric_cols].max(axis=1)
    ax.hlines(y, lo, hi, color=PALETTE["grid"], linewidth=2.4, zorder=1)
    ax.hlines(y, d["macro_f1_present_classes"], d["rf_macro_f1"], color=rf_color, linewidth=1.1,
              linestyle=(0, (2, 1.5)), alpha=0.7, zorder=2)
    ax.scatter(d["rf_macro_f1"], y, marker="D", s=40, facecolors="white", edgecolors=rf_color, linewidth=1.4, zorder=4)
    for col in ("accuracy", "balanced_accuracy"):
        ax.scatter(d[col], y, s=48, color=metric_colors[col], edgecolors="white", linewidth=0.7, zorder=3)
    sizes = 55 + (d["test_rows"] / d["test_rows"].max()) * 320
    conf_norm = mpl.colors.Normalize(vmin=d["mean_max_probability"].min(), vmax=d["mean_max_probability"].max())
    ring_colors = [V1.CONTEXT_CMAP(conf_norm(v)) for v in d["mean_max_probability"]]
    ax.scatter(d["macro_f1_present_classes"], y, s=sizes + 130, facecolors="none", edgecolors=ring_colors,
               linewidth=2.6, zorder=2)
    ax.scatter(d["macro_f1_present_classes"], y, s=sizes, color=metric_colors["macro_f1_present_classes"],
               edgecolors="white", linewidth=0.8, zorder=5)
    mean_f1 = d["macro_f1_present_classes"].mean()
    ax.axvline(mean_f1, color=PALETTE["ink"], linestyle=":", linewidth=1.0, zorder=1)

    for district, label, offset in (
        ("KONAK", "KONAK: 60% Macro-F1; 93% confidence", (8, 11)),
        ("GÜZELBAHÇE", "GÜZELBAHÇE: lowest transfer (47%)", (8, 9)),
    ):
        match = d.index[d["held_out_district"].eq(district)]
        if len(match):
            i = int(match[0])
            ax.annotate(label, xy=(d.loc[i, "macro_f1_present_classes"], y[i]), xytext=offset,
                        textcoords="offset points", fontsize=6.8, color=PALETTE["ink"], ha="left", va="bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
                        arrowprops={"arrowstyle": "-", "color": PALETTE["muted"], "linewidth": 0.7}, zorder=8)

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
    b_w_in = ax.get_position().width * FIG_W
    V1._panel_label(ax, "B", x=(M_LEFT + 0.03 - B_LEFT) / b_w_in, y=1.0 + (KEY_H - 0.36) / B_H)

    # ---- key column (outside the data area) -----------------------------------------
    kx, ky, kw, kh = key_rect  # figure fractions
    kax = fig.add_axes([kx, ky, kw, kh])
    kax.axis("off")
    metric_handles = [Line2D([0], [0], marker="o", linestyle="none", color=metric_colors[c], label=metric_labels[c],
                             markersize=7.5, markeredgecolor="white") for c in metric_cols]
    metric_handles.append(Line2D([0], [0], marker="D", linestyle="none", color=rf_color, markerfacecolor="white",
                                 markeredgecolor=rf_color, markersize=6.5, label="Macro-F1 (Random Forest)"))
    metric_handles.append(Line2D([0], [0], color=PALETTE["ink"], linestyle=":", linewidth=1.2,
                                 label=f"FUR mean Macro-F1 ({mean_f1:.0%})"))
    leg1 = kax.legend(handles=metric_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8.0, ncol=2,
                      frameon=True, title="Metric", title_fontsize=8.4, borderpad=0.6)
    kax.add_artist(leg1)
    size_vals = [d["test_rows"].min(), d["test_rows"].median(), d["test_rows"].max()]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=metric_colors["macro_f1_present_classes"],
               markeredgecolor="white", markersize=np.sqrt(55 + (v / d["test_rows"].max()) * 320) * 0.62,
               label=f"{int(v)} rows")
        for v in size_vals
    ]
    kax.legend(handles=size_handles, loc="upper left", bbox_to_anchor=(0.425, 1.0), fontsize=8.0, frameon=True, ncol=3,
               title="Held-out support\n(macro-F1 dot size)", title_fontsize=8.2, labelspacing=1.3, borderpad=0.7,
               columnspacing=1.0, handletextpad=0.3)
    sm = mpl.cm.ScalarMappable(norm=conf_norm, cmap=V1.CONTEXT_CMAP)
    cax = fig.add_axes([kx + 0.735 * kw, ky + 0.70 * kh, 0.255 * kw, 0.15 * kh])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Mean max probability\n(macro-F1 dot ring color)", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)
    cb.ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
    return kax


def build() -> Path:
    districts, mahalle, study_area, lodo, lodo_rf = load()
    xmin, ymin, xmax, ymax = study_area.total_bounds
    map_h = MAP_W * (ymax - ymin) / (xmax - xmin)
    fig_h = B_BOTTOM + B_H + KEY_H + GAP_AB + STATS_H + map_h + TITLE_H + TOP

    S.apply_style()
    fig = plt.figure(figsize=(FIG_W, fig_h))
    map_b = B_BOTTOM + B_H + KEY_H + GAP_AB + STATS_H
    ax_d = S.inch_axes(fig, M_LEFT, map_b, MAP_W, map_h)
    x2 = M_LEFT + MAP_W + CB_GAP + CB_W + CB_ROOM + MAP_GAP
    ax_m = S.inch_axes(fig, x2, map_b, MAP_W, map_h)
    plot_district_map(fig, ax_d, districts)
    plot_mahalle_map(fig, ax_m, mahalle, study_area)

    b_w = FIG_W - B_LEFT - B_RIGHT
    ax_b = S.inch_axes(fig, B_LEFT, B_BOTTOM, b_w, B_H)
    key_x = 0.80                   # right of the (b) letter
    key_rect = [key_x / FIG_W, (B_BOTTOM + B_H + 0.08) / fig_h, (FIG_W - key_x - 0.10) / FIG_W, (KEY_H - 0.08) / fig_h]
    plot_transfer(fig, ax_b, key_rect, lodo, lodo_rf)

    V1._finalize_panel_labels(fig)
    S.scale_text(fig)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=S.DPI, facecolor="white", bbox_inches=S.tight_crop_bbox(fig, pad_in=0.05, full_width=True))
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text() and t.get_visible()]
    print(f"fig {FIG_W:.2f} x {fig_h:.2f} in; min authored font {min(sizes):.2f} pt "
          f"= {min(sizes) * S.PRINT_WIDTH_IN / FIG_W:.2f} pt printed at 190 mm")
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
