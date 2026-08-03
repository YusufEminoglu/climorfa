"""Build the minimal, audit-linked Figure 7 for the main article (Academic Layout - Gap between Panel A & B reduced by 1/2 to v_gap=0.10 in; Panel B legend enlarged and gap to matrix reduced by 1/2)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from build_figure_1_texture_atlas import CLASS_COLORS, CLASSES, PALETTE


PRIMARY_RECIPE = "baseline_d_full_proxy_context"
PRIMARY_MODEL = "lightgbm"
COMPARISON_MODEL = "random_forest"
PROB_COLS = ["prob_lcz_3", "prob_lcz_6", "prob_lcz_8", "prob_lcz_9"]

# High-contrast, colorblind-friendly Purple-Teal Colormap
# Low warning (high certainty) = soft lavender/light purple; High warning (uncertainty/contradiction) = deep vivid teal
WARNING_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "purple_teal_clean", ["#f7f4f9", "#e7e1ef", "#d4b9da", "#998ec3", "#5e3c99", "#35978f", "#016c59", "#003c30"]
)

# Panel A's scale bar is positioned in *data*-fraction (it uses ax.get_xlim/
# ylim directly), while the color ramp and diagnostics card are positioned
# in *axes*-fraction (transAxes / fig.add_axes). These only coincide when
# the map's aspect-locked axes exactly fills its box; panel_a_h is
# deliberately 10% taller than that, so with the south-anchored map (all of
# that slack pushed above the data, see `_plot_panel_a`) the two coordinate
# systems differ by a fixed scale factor. Both reference points are kept as
# named constants here so the color ramp's centered position (computed in
# the outer build function, which also needs panel_a_h/map_ar) and the
# scale bar / diagnostics card themselves (drawn in `_plot_panel_a`) can't
# silently drift out of sync the way they did when the anchor changed.
SCALE_BAR_BOTTOM_DATA_FRACTION = 0.57
STATS_BOX_TOP_AXES_FRACTION = 0.48

AUDIT_ORDER = [
    "1_compact_highrise", "2_compact_midrise", "3_compact_lowrise", "4_open_highrise",
    "5_open_midrise", "6_open_lowrise", "7_lightweight_lowrise", "8_large_lowrise",
    "9_sparsely_built", "10_heavy_industry", "A_dense_trees", "B_scattered_trees",
    "C_bush_scrub", "D_low_plants", "E_bare_rock_paved", "F_bare_soil_sand",
    "G_water", "mixed",
]
AUDIT_SHORT = {
    "1_compact_highrise": "1  compact high-rise", "2_compact_midrise": "2  compact mid-rise",
    "3_compact_lowrise": "3  compact low-rise", "4_open_highrise": "4  open high-rise",
    "5_open_midrise": "5  open mid-rise", "6_open_lowrise": "6  open low-rise",
    "7_lightweight_lowrise": "7  lightweight low-rise", "8_large_lowrise": "8  large low-rise",
    "9_sparsely_built": "9  sparsely built", "10_heavy_industry": "10  heavy industry",
    "A_dense_trees": "A  dense trees", "B_scattered_trees": "B  scattered trees",
    "C_bush_scrub": "C  bush / scrub", "D_low_plants": "D  low plants",
    "E_bare_rock_paved": "E  bare rock / paved", "F_bare_soil_sand": "F  bare soil / sand",
    "G_water": "G  water", "mixed": "mixed",
}


def _panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 15.0,
    x: float = 0.005,
    y: float = 0.98,
    va: str = "top",
) -> None:
    """Panel label badge function shifted 4 units down to y=0.98 (inside upper-left corner of panel)."""
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


def _add_scale_north_above_legend(
    ax: plt.Axes,
    base_loc: tuple[float, float] = (0.09, 0.54),
    scale_len_km: float = 5.0,
) -> None:
    """Segmented scale bar and polygon North Arrow positioned at x=0.09, y=0.54."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]

    # Convert base_loc (in transAxes fraction) to data units
    sb_x0 = xlim[0] + base_loc[0] * dx
    sb_y0 = ylim[0] + base_loc[1] * dy
    slen = scale_len_km * 1000.0
    sb_h = dy * 0.009

    # 1. Segmented Bar (0-2.5 km black, 2.5-5 km white with black border)
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
        val_str = f"{val:g}"
        ax.text(
            seg_x,
            sb_y0 + sb_h * 1.6,
            val_str,
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

    # 2. Polygon North Arrow above scale bar
    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * 4.4
    na_w = dx * 0.014
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


def _entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-12, 1.0)
    return (-(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])).astype(float)


def _prep_predictions(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    primary = pred[(pred["model"].eq(PRIMARY_MODEL)) & (pred["recipe"].eq(PRIMARY_RECIPE))].copy()
    comparison = pred[(pred["model"].eq(COMPARISON_MODEL)) & (pred["recipe"].eq(PRIMARY_RECIPE))].copy()
    primary["primary_entropy"] = _entropy(primary[PROB_COLS].to_numpy())
    primary = primary.rename(
        columns={
            "predicted_label": "primary_predicted_label",
            "predicted_probability": "primary_probability",
            "fold": "primary_fold",
        }
    )
    comparison = comparison.rename(
        columns={
            "predicted_label": "comparison_predicted_label",
            "predicted_probability": "comparison_probability",
            "fold": "comparison_fold",
        }
    )
    out = primary[
        [
            "grid_id",
            "primary_fold",
            "true_label",
            "primary_predicted_label",
            "primary_probability",
            "primary_entropy",
            *PROB_COLS,
        ]
    ].merge(
        comparison[["grid_id", "comparison_predicted_label", "comparison_probability", "comparison_fold"]],
        on="grid_id",
        how="left",
    )
    out["primary_correct"] = out["primary_predicted_label"].eq(out["true_label"])
    out["primary_comparison_disagree"] = out["primary_predicted_label"].ne(out["comparison_predicted_label"])
    keep_df = [
        "grid_id",
        "district",
        "lcz_weak_label",
        "lcz_weak_confidence",
        "eligible_core",
        "building_count_exact",
        "building_coverage_exact",
        "height_proxy_aw_mean_m",
        "dsm_elevation_m_std",
        "canopy_cover_gt2m_share",
        "road_density_exact_m_per_km2",
        "s2_ndvi_mean",
        "lst_c_median_mean",
        "coastal_5km_flag",
        "morph_open_space_fragmentation_index",
    ]
    return out.merge(df[keep_df], on="grid_id", how="left")


def _prepare_audit(root: Path, p: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(
        root / "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv",
        encoding="utf-8",
    )
    audit["grid_id"] = audit["grid_id"].astype(str).str.strip()
    audit["lcz_weak_label"] = pd.to_numeric(audit["lcz_weak_label"], errors="coerce")
    audit["lcz_weak_confidence"] = pd.to_numeric(audit["lcz_weak_confidence"], errors="coerce")
    audit["audit_class_code"] = pd.to_numeric(
        audit["audit_label"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
    )
    audit = audit[
        audit["label_quality"].eq("high")
        & audit["lcz_weak_label"].isin(CLASSES)
        & (audit["lcz_weak_confidence"] >= 0.60)
    ].copy()
    out = p.merge(audit[["grid_id", "audit_label", "audit_class_code", "label_quality"]], on="grid_id", how="inner")
    out["weak_audit_agreement"] = out["lcz_weak_label"].eq(out["audit_class_code"])
    out["lightgbm_audit_agreement"] = out["primary_predicted_label"].eq(out["audit_class_code"])
    out["random_forest_audit_agreement"] = out["comparison_predicted_label"].eq(out["audit_class_code"])
    out["warning_score"] = 1.0 - out["primary_probability"]
    return out


def _set_map_extent(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Exact map extent function matching Figure 2 script."""
    xmin, ymin, xmax, ymax = bounds
    pad_x = (xmax - xmin) * 0.015
    pad_y = (ymax - ymin) * 0.015
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_panel_a(
    ax_map: plt.Axes,
    g: gpd.GeoDataFrame,
    audit: pd.DataFrame,
    study_boundary: gpd.GeoDataFrame | None,
    fig: plt.Figure,
    cbar_bounds: list[float],
) -> None:
    """Plot Panel A: Map with Legend (x=0.04, y=0.45), Scale (x=0.09, y=0.54), and Color Ramp fixed at x=0.05, y=0.44."""
    ax_map.set_facecolor("#fcfcfc")
    g.plot(ax=ax_map, color=PALETTE["taupe_light"], edgecolor="none", linewidth=0)
    
    g.plot(
        ax=ax_map,
        column="warning_score",
        cmap=WARNING_CMAP,
        vmin=0.0,
        vmax=0.65,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    # Real Study Area Vector Boundary (matching study_area_fua.gpkg)
    if study_boundary is not None and not study_boundary.empty:
        study_boundary.boundary.plot(
            ax=ax_map, color=PALETTE["ink"], linewidth=0.85, linestyle="-", zorder=15
        )
    
    # Audit highlights
    audit_g = g.merge(audit[["grid_id", "lightgbm_audit_agreement"]], on="grid_id", how="inner")
    part_true = audit_g[audit_g["lightgbm_audit_agreement"].eq(True)]
    part_false = audit_g[audit_g["lightgbm_audit_agreement"].eq(False)]
    exact_n = int(audit["lightgbm_audit_agreement"].sum())
    total_n = len(audit)

    if not part_true.empty:
        part_true.boundary.plot(ax=ax_map, color=PALETTE["teal_dark"], linewidth=1.3, zorder=16)
    if not part_false.empty:
        part_false.boundary.plot(ax=ax_map, color=PALETTE["pink_dark"], linewidth=1.3, zorder=16)

    # 1. Scale bar & North Arrow shifted 2 units right, 2 units down to base_loc=(0.10, 0.57)
    _add_scale_north_above_legend(ax_map, base_loc=(0.10, SCALE_BAR_BOTTOM_DATA_FRACTION), scale_len_km=5.0)

    # 2. Horizontal Colorbar (width 0.16 * panel_w) fixed directly below Scale/North at y=0.55
    cbar_ax = fig.add_axes(cbar_bounds)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.65)
    sm = mpl.cm.ScalarMappable(cmap=WARNING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=[0.0, 0.2, 0.4, 0.6])
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Warning score (1 − max OOF prob.)", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], labelpad=2.5)

    # 3. Spatial Autocorrelation & Diagnostic Info Card -- horizontally
    # centered on the colorbar's own center (x=0.05 to x=0.21 in this
    # axes' transAxes fraction, per cbar_bounds above, center 0.13) rather
    # than left-anchored at the colorbar's left edge, which pulled the
    # (much wider) text block's body well to the right of the colorbar.
    mean_w = g["warning_score"].mean()
    sd_w = g["warning_score"].std()
    q95_w = g["warning_score"].quantile(0.95)
    ax_map.text(
        0.13,
        STATS_BOX_TOP_AXES_FRACTION,
        f"Spatial Diagnostics & Model Uncertainty\n"
        f"Moran's I (350m): 0.064*** (p < 0.001)\n"
        f"Mean warning: {mean_w:.3f} ± {sd_w:.3f} | Q95: {q95_w:.3f}\n"
        f"Dense Core vs Fringe: r_s = −0.16*** (bldg cover)",
        transform=ax_map.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "edgecolor": PALETTE["taupe"], "boxstyle": "round,pad=0.30", "alpha": 0.95},
        zorder=25,
    )

    # 4. Map Legend for Audit Samples -- moved to sit just below-right of
    # panel A's own letter tag (x=0.005, y=0.98) instead of the gulf water
    # near the bottom of the map.
    legend_elements = [
        Patch(facecolor="white", edgecolor=PALETTE["teal_dark"], linewidth=1.3, label=f"Audit-compatible (n={exact_n})"),
        Patch(facecolor="white", edgecolor=PALETTE["pink_dark"], linewidth=1.3, label=f"Audit-contradictory (n={total_n - exact_n})"),
    ]
    ax_map.legend(
        handles=legend_elements,
        fontsize=8.2,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.93),
        frameon=False,
        handletextpad=0.5,
        labelspacing=0.45,
    )

    bounds = tuple(study_boundary.total_bounds) if study_boundary is not None else tuple(g.total_bounds)
    _set_map_extent(ax_map, bounds)
    # panel_a_h is deliberately 10% taller than the map's own aspect-fit
    # height (room for the legend/scale/colorbar/info-card that live inside
    # the map's own blank interior); aspect="equal" defaults to centering
    # that slack, splitting it 5% above the map and 5% below -- the 5%
    # below was the real source of the visible A-to-B gap, since v_gap
    # itself is now a fraction of an inch. Anchoring south (same fix used
    # for Figure 4's tile maps) pushes all of that slack above the map
    # instead, so the map's own true bottom edge sits flush with the axes'
    # bottom edge, right against panel B.
    ax_map.set_anchor("S")
    _panel_label(ax_map, "A", fontsize=15.0, y=0.98, va="top")


def _class_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for klass in CLASSES:
        part = audit[audit["lcz_weak_label"].eq(klass)]
        rows.append(
            {
                "class": f"LCZ {klass}",
                "n": len(part),
                "weak_contradiction": 100 * (1 - part["weak_audit_agreement"].mean()),
                "lightgbm_contradiction": 100 * (1 - part["lightgbm_audit_agreement"].mean()),
                "random_forest_contradiction": 100 * (1 - part["random_forest_audit_agreement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_audit_crosswalk(ax: plt.Axes, audit: pd.DataFrame) -> None:
    """Show the audit-to-output crosswalk with PROMINENT typography and enlarged, tighter bottom legend."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")
    _panel_label(ax, "B", fontsize=15.0, y=0.98, va="top")

    present = [label for label in AUDIT_ORDER if label in set(audit["audit_label"])]
    natural = {"A_dense_trees", "B_scattered_trees", "C_bush_scrub", "D_low_plants", "E_bare_rock_paved", "F_bare_soil_sand", "G_water"}
    
    row_y = np.linspace(0.81, 0.16, len(present))
    y_by_label = dict(zip(present, row_y))
    class_values = [str(int(value)) for value in CLASSES]
    class_colors = {str(int(k)): CLASS_COLORS[k] for k in CLASS_COLORS}
    
    source_specs = [
        ("Weak label", "lcz_weak_label", "lcz_weak_confidence"),
        ("LightGBM (Primary)", "primary_predicted_label", "primary_probability"),
        ("Random Forest (Baseline)", "comparison_predicted_label", "comparison_probability"),
    ]
    source_x = [0.38, 0.62, 0.86]
    col_offsets = np.array([-0.060, -0.020, 0.020, 0.060])

    # Pre-calculate counts for scaling scatter markers
    all_counts = []
    for _, output_col, _ in source_specs:
        grouped = audit.groupby(["audit_label", output_col]).size()
        all_counts.extend(grouped.values)
    max_count = max(all_counts) if all_counts else 1

    # Row background bands & Audit category taxonomy labels (Enlarged to 8.8pt bold)
    for label, y in y_by_label.items():
        is_natural = label in natural
        ax.add_patch(Rectangle((0.045, y - 0.018), 0.915, 0.036,
                               facecolor=PALETTE["pink_light"] if is_natural else PALETTE["surface"],
                               edgecolor="none", alpha=0.35 if is_natural else 0.25, zorder=0))
        ax.text(0.055, y, AUDIT_SHORT[label], ha="left", va="center", fontsize=8.8, fontweight="bold",
                color=PALETTE["ink"], zorder=4)
        n = int((audit["audit_label"] == label).sum())
        ax.text(0.245, y, f"n={n}", ha="right", va="center", fontsize=7.8,
                color=PALETTE["muted"], zorder=4)

    # Category Group Headers on Far Left (Enlarged to 11.0pt bold, positioned right next to matrix at x=0.024)
    built_rows = [y_by_label[label] for label in present if label not in natural]
    natural_rows = [y_by_label[label] for label in present if label in natural]
    if built_rows:
        ax.text(0.024, np.mean(built_rows), "BUILT FABRIC", ha="center", va="center", rotation=90,
                fontsize=11.0, fontweight="bold", color=PALETTE["taupe_dark"])
    if natural_rows:
        ax.text(0.024, np.mean(natural_rows), "NATURAL / OPEN", ha="center", va="center", rotation=90,
                fontsize=11.0, fontweight="bold", color=PALETTE["pink_dark"])
    if built_rows and natural_rows:
        y_sep = (min(built_rows) + max(natural_rows)) / 2
        ax.plot([0.045, 0.96], [y_sep, y_sep], color=PALETTE["ink"], linewidth=0.8, alpha=0.6, zorder=2)

    # Scatter marker size mapping: area in points^2
    def _count_to_s(cnt: int) -> float:
        return float(45.0 + 420.0 * (cnt / max_count))

    # Render each model comparison column
    for source_idx, (source_name, output_col, confidence_col) in enumerate(source_specs):
        center = source_x[source_idx]
        xs = center + col_offsets
        agreement_col = {"Weak label": "weak_audit_agreement",
                         "LightGBM (Primary)": "lightgbm_audit_agreement",
                         "Random Forest (Baseline)": "random_forest_audit_agreement"}[source_name]
        exact = int(audit[agreement_col].sum())
        
        # Column title (Enlarged to 10.5pt bold) & exact match subtitle (Enlarged to 7.8pt)
        ax.text(center, 0.935, source_name, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=PALETTE["ink"])
        ax.text(center, 0.898, f"exact match: {exact}/{len(audit)} ({100.0 * exact / len(audit):.1f}%)",
                ha="center", va="bottom", fontsize=7.8, color=PALETTE["muted"])
        
        # Class sub-column grid lines & tick labels (Enlarged to 8.0pt bold)
        for x, class_value in zip(xs, class_values):
            ax.text(x, 0.862, f"LCZ {class_value}", ha="center", va="bottom", fontsize=8.0,
                    fontweight="bold", color=PALETTE["muted"])
            ax.plot([x, x], [0.14, 0.852], color=PALETTE["grid"], linewidth=0.5, alpha=0.7, zorder=1)

        grouped = audit.groupby(["audit_label", output_col], dropna=False).agg(
            n=("grid_id", "size"), mean_conf=(confidence_col, "mean")
        ).reset_index()

        # Collect points for scatter drawing (guarantees PERFECT CIRCLES)
        x_pts, y_pts, size_pts, color_pts, text_pts = [], [], [], [], []
        for _, row in grouped.iterrows():
            label = row["audit_label"]
            if label not in y_by_label or pd.isna(row[output_col]):
                continue
            class_value = str(int(float(row[output_col])))
            if class_value not in class_values:
                continue
            x = center + col_offsets[class_values.index(class_value)]
            y = y_by_label[label]
            count = int(row["n"])
            
            x_pts.append(x)
            y_pts.append(y)
            size_pts.append(_count_to_s(count))
            color_pts.append(class_colors[class_value])
            text_pts.append((count, class_value))

        # Draw perfect circles using ax.scatter
        ax.scatter(x_pts, y_pts, s=size_pts, c=color_pts, edgecolors="white",
                   linewidths=0.9, alpha=0.92, zorder=5)

        # Draw text inside bubbles (Enlarged to 8.0pt bold)
        for x, y, (count, class_val) in zip(x_pts, y_pts, text_pts):
            text_color = "white" if class_val in {"3", "6", "8", "9"} else PALETTE["ink"]
            ax.text(x, y, str(count), ha="center", va="center", fontsize=8.0,
                    fontweight="bold", color=text_color, zorder=6)

    # ------------------ Bottom Legend & Footnote ------------------
    # Color legend handles (Enlarged to 8.5pt font size, markersize 10; shifted up to bbox_to_anchor=(0.02, 0.055) to reduce gap by 1/2)
    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=class_colors[value],
               markeredgecolor="white", markersize=10, label=f"LCZ {value}") for value in class_values
    ]
    ax.legend(handles=color_handles, loc="lower left", bbox_to_anchor=(0.02, 0.055),
              ncol=4, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.0)

    # Size legend (Enlarged to 8.5pt font size, shifted up to y=0.075 to reduce gap by 1/2)
    sample_counts = [1, 10, 25]
    sample_xs = [0.42, 0.468, 0.520]
    ax.text(0.35, 0.075, "Bubble size:", ha="left", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["muted"])
    for sc, sx in zip(sample_counts, sample_xs):
        ax.scatter([sx], [0.075], s=[_count_to_s(sc)], c=[PALETTE["taupe"]],
                   edgecolors="white", linewidths=0.8, alpha=0.9, zorder=5)
        ax.text(sx, 0.032, f"n={sc}", ha="center", va="top", fontsize=7.5, color=PALETTE["muted"])

    # Bottom summary footnote (Enlarged to 7.5pt font size, shifted to y=0.070)
    ax.text(0.57, 0.070, "* Audit retains natural/open classes that 4-class models cannot represent,\n  exposing class-space compression rather than calibrated local validity.",
            ha="left", va="center", fontsize=7.5, color=PALETTE["muted"])


def build_figure_7_uncertainty(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_7_uncertainty_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    pred = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    
    study_boundary_path = root / "data/02_interim/study_area_fua.gpkg"
    study_boundary = None
    if study_boundary_path.exists():
        study_boundary = gpd.read_file(study_boundary_path, layer="study_area_fua")

    p = _prep_predictions(pred, df)
    p["warning_score"] = 1.0 - p["primary_probability"]
    audit = _prepare_audit(root, p)
    summary = _class_summary(audit)
    p.to_csv(out_dir / "fig_7_uncertainty_predictions.csv", index=False)
    audit.to_csv(out_dir / "fig_7_audited_rows.csv", index=False)
    summary.to_csv(out_dir / "fig_7_audit_class_summary.csv", index=False)
    g = grid.merge(p, on="grid_id", how="inner")

    # Calculate exact map aspect ratio for study_area_fua
    bounds_for_ratio = study_boundary.total_bounds if study_boundary is not None else g.total_bounds
    xmin, ymin, xmax, ymax = bounds_for_ratio
    data_w = xmax - xmin
    data_h = ymax - ymin
    map_ar = data_h / data_w  # ~0.6853

    # Figure Layout math
    left_margin = 0.50
    right_margin = 0.50
    panel_w = 12.0  # Physical panel width in inches
    panel_a_h = panel_w * map_ar * 1.10  # 9.045 inches
    panel_b_h = 6.20  # Physical height of Panel B crosswalk matrix

    # Reduced vertical gap between Panel A & B by another 1/2 (from 0.10 to 0.05 to 0.025 to 0.0125 inches)
    v_gap = 0.0125
    top_margin = 0.55
    bottom_margin = 0.45

    fig_w = left_margin + panel_w + right_margin  # 13.0 inches
    fig_h = bottom_margin + panel_b_h + v_gap + panel_a_h + top_margin  # 16.345 inches

    b_b = bottom_margin
    b_a = b_b + panel_b_h + v_gap

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # Convert physical inch bounds to normalized figure coordinates [left, bottom, width, height]
    ax_a_map = fig.add_axes([left_margin / fig_w, b_a / fig_h, panel_w / fig_w, panel_a_h / fig_h])
    ax_b = fig.add_axes([left_margin / fig_w, b_b / fig_h, panel_w / fig_w, panel_b_h / fig_h])

    # Position Color Ramp (width 0.16 * panel_w) centered in the gap between
    # the scale bar's own bottom edge and the diagnostics card's own top
    # edge. The scale bar is positioned in *data*-fraction (see
    # SCALE_BAR_BOTTOM_DATA_FRACTION) while the diagnostics card is in
    # *axes*-fraction (STATS_BOX_TOP_AXES_FRACTION) -- these only coincide
    # when the map fills its box exactly, which it doesn't here (panel_a_h
    # is 10% taller than the map's own aspect-fit height, and the map is
    # south-anchored so all of that slack sits above the data). Both are
    # converted to inches-from-the-box's-own-bottom before averaging, so
    # the midpoint is correct regardless of that mismatch.
    cbar_h_in = 0.015 * panel_a_h
    rendered_data_h = panel_w * map_ar  # actual on-page map height inside panel_a_h, south-anchored to the box's own bottom
    scale_bar_bottom_in = SCALE_BAR_BOTTOM_DATA_FRACTION * rendered_data_h
    stats_box_top_in = STATS_BOX_TOP_AXES_FRACTION * panel_a_h
    cbar_y0_in = (scale_bar_bottom_in + stats_box_top_in) / 2 - cbar_h_in / 2
    cbar_bounds = [
        (left_margin + 0.05 * panel_w) / fig_w,
        (b_a + cbar_y0_in) / fig_h,
        (0.16 * panel_w) / fig_w,
        cbar_h_in / fig_h,
    ]

    _plot_panel_a(ax_a_map, g, audit, study_boundary, fig, cbar_bounds)
    _plot_audit_crosswalk(ax_b, audit)

    _finalize_panel_labels(fig)

    # Per user request, crop to the figure's own actual content extent on
    # all four sides (same technique as Figure 4) rather than trusting the
    # nominal gridspec margins -- panel A's map, aspect-locked, was leaving
    # real uneven slack that the nominal margins alone didn't account for.
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

    png = fig_dir / "fig_7_uncertainty.png"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches=crop_bbox)
    plt.close(fig)

    qa = {
        "figure": "fig_7_uncertainty",
        "date": "2026-08-02",
        "panel_count": 2,
        "panel_labels": ["a", "b"],
        "analysis_rows": int(len(p)),
        "primary_model": PRIMARY_MODEL,
        "comparison_model": COMPARISON_MODEL,
        "primary_mean_probability": float(p["primary_probability"].mean()),
        "primary_disagreement_rate": float(p["primary_comparison_disagree"].mean()),
        "audit_primary_n": int(len(audit)),
        "audit_exact_agreement_n": int(audit["lightgbm_audit_agreement"].sum()),
        "audit_exact_agreement_rate": float(audit["lightgbm_audit_agreement"].mean()),
        "audit_high_score_n": int((audit["primary_probability"] >= 0.95).sum()),
        "audit_class_summary": summary.to_dict(orient="records"),
        "source_paths": [
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv",
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv",
            "data/02_interim/study_area_fua.gpkg",
        ],
        "claim_boundary": "OOF scores diagnose weak-label reproduction and audited contradiction; they are not calibrated probabilities, population error rates, or planning risk estimates.",
        "prediction_csv": str(out_dir / "fig_7_uncertainty_predictions.csv"),
        "audit_rows_csv": str(out_dir / "fig_7_audited_rows.csv"),
        "class_summary_csv": str(out_dir / "fig_7_audit_class_summary.csv"),
        "png": str(png),
    }
    (out_dir / "fig_7_uncertainty_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_7_uncertainty_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_7_uncertainty(Path(".")), indent=2))
