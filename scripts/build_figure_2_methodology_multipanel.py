"""Build dense methodology and graphical-abstract panels."""
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
import seaborn as sns

from build_figure_1_texture_atlas import CLASS_COLORS, CLASSES, PALETTE


RECIPE_LABELS = {
    "baseline_a_morphology_only": "Morph.",
    "baseline_b_morphology_plus_vegetation": "+ veg.",
    "baseline_c_morphology_plus_green_blue_context": "+ green-blue",
    "baseline_d_no_green_2sfca": "Full, no 2SFCA",
    "baseline_d_green_2sfca_400m": "+ 400 m",
    "baseline_d_full_proxy_context": "+ 800 m",
    "baseline_d_green_2sfca_1200m": "+ 1200 m",
}
RECIPE_ORDER = list(RECIPE_LABELS)
FAMILY_COLORS = {
    "Morphology": PALETTE["teal"],
    "Vegetation": "#56B881",
    "Green-blue": PALETTE["orange"],
    "Network": "#61A0AF",
    "Climate response": PALETTE["pink"],
    "Gate": PALETTE["taupe"],
}


def _panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 15.0,
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


def _finalize_panel_labels(fig: plt.Figure) -> None:
    """Render bare panel letters (e.g. 'A') as lowercase-parenthesized (e.g. '(a)').

    Insets created via ax.inset_axes() live in ax.child_axes, not fig.axes,
    so they must be visited explicitly or their labels are silently skipped.
    """
    all_axes = list(fig.axes)
    for ax in fig.axes:
        all_axes.extend(ax.child_axes)
    for ax in all_axes:
        for child in ax.texts:
            text = child.get_text()
            if text in ["A", "B", "C", "D", "E"]:
                child.set_text(f"({text.lower()})")


def _style_inset(ax: plt.Axes) -> None:
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.set_zorder(20)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=5.6)


def _add_inset_backing(
    parent_ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    pad: tuple[float, float, float, float] = (0.02, 0.02, 0.065, 0.11),
) -> None:
    """Draw an opaque card behind an inset's bounds, sized to also cover its
    title and tick labels which render outside the inset's own axes box."""
    left, right, top_extra, bottom_extra = pad
    x0, y0, w, h = bounds
    rect = mpatches.FancyBboxPatch(
        (x0 - left, y0 - bottom_extra),
        w + left + right,
        h + top_extra + bottom_extra,
        transform=parent_ax.transAxes,
        boxstyle="round,pad=0.006",
        facecolor="white",
        edgecolor=PALETTE["grid"],
        linewidth=0.8,
        zorder=15,
    )
    parent_ax.add_patch(rect)


def _set_map_extent(ax: plt.Axes, bounds: np.ndarray) -> None:
    xmin, ymin, xmax, ymax = bounds
    pad_x = (xmax - xmin) * 0.015
    pad_y = (ymax - ymin) * 0.015
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_study_support(ax: plt.Axes, grid: gpd.GeoDataFrame, df: pd.DataFrame) -> None:
    g = grid.copy()
    g.plot(ax=ax, color=PALETTE["teal_light"], edgecolor="#4A9B93", linewidth=0.15)
    excluded = g[g["eligible_core"].ne(1)]
    excluded.plot(ax=ax, color=PALETTE["pink_dark"], edgecolor="none", linewidth=0)

    # Outer boundary perimeter of the Izmir FUR area with a thin grey line
    fur_boundary = g.unary_union
    gpd.GeoSeries([fur_boundary], crs=g.crs).boundary.plot(
        ax=ax, color="#555555", linewidth=0.5, linestyle="-", zorder=10
    )

    _panel_label(ax, "A")
    _set_map_extent(ax, g.total_bounds)
    n_eligible = int(g["eligible_core"].sum())
    n_excluded = int(len(g) - n_eligible)
    handles = [
        mpatches.Patch(facecolor=PALETTE["teal_light"], edgecolor="#4A9B93", linewidth=0.3, label=f"eligible-core\n(n={n_eligible:,})"),
        mpatches.Patch(facecolor=PALETTE["pink_dark"], label=f"excluded\n(n={n_excluded:,})"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.35), frameon=False, handleheight=2.0)


def _plot_family_missing(ax: plt.Axes, family_missing: pd.DataFrame) -> None:
    fm = family_missing.sort_values("columns", ascending=True).tail(5)
    labels = fm["family"].str.replace("_", " ").tolist()
    cols = fm["columns"].to_numpy()
    y_pos = np.arange(len(fm))
    ax.barh(y_pos, cols, color=PALETTE["teal"], alpha=0.85, height=0.65)
    ax.set_yticks([])
    for i, (label, n_col) in enumerate(zip(labels, cols)):
        ax.text(12.0, i, f"{label} (n={n_col})", ha="left", va="center", fontsize=7.0, fontweight="bold", color="#1A3835", zorder=10)
    ax.scatter(fm["avg_missing_share"].to_numpy() * cols, y_pos, color=PALETTE["pink"], s=20, label="missing-wt", zorder=15)
    ax.set_xlabel("Feature columns", fontsize=7.5)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7.0)
    ax.legend(fontsize=6.8, frameon=True, loc="lower right", handletextpad=0.4)
    sns.despine(ax=ax, left=True)
    _panel_label(ax, "B")


def _plot_recipe_performance(ax: plt.Axes, summary: pd.DataFrame) -> None:
    s = summary[summary["model"].eq("lightgbm") & summary["recipe"].isin(RECIPE_ORDER)].copy()
    s["recipe_label"] = s["recipe"].map(RECIPE_LABELS)
    x = np.arange(len(s))
    y = s["macro_f1_mean"].to_numpy()
    feats = s["features"].to_numpy()
    labels = s["recipe_label"].tolist()

    # Fill area under curve for visual depth
    ax.fill_between(x, y, 0.60, color=PALETTE["teal_light"], alpha=0.35, zorder=1)
    ax.plot(x, y, marker="o", color=PALETTE["teal"], linewidth=1.8, markersize=5.5, zorder=3)

    # Highlight peak performance
    max_idx = int(np.argmax(y))
    ax.scatter([max_idx], [y[max_idx]], color=PALETTE["orange"], s=45, zorder=5, edgecolor="white", linewidth=1.2)
    ax.axhline(y[max_idx], color=PALETTE["orange"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    # Annotate exact F1 score above and feature count below each point
    for xi, yi, feat in zip(x, y, feats):
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6.8,
            fontweight="bold",
            color="#1F2421",
        )
        ax.annotate(
            f"({int(feat)}f)",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -11),
            ha="center",
            fontsize=6.0,
            color=PALETTE["orange_dark"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.0)
    ax.set_ylabel("Macro-F1", fontsize=7.5)
    ax.set_xlabel("")
    ax.tick_params(axis="y", labelsize=7.0)
    ax.set_ylim(0.60, 0.88)
    sns.despine(ax=ax)
    _panel_label(ax, "C")


def _plot_2sfca(ax: plt.Axes, df: pd.DataFrame) -> None:
    fields = [
        ("green_2sfca_400m_access_log1p", "400 m", PALETTE["teal"]),
        ("green_2sfca_800m_access_log1p", "800 m", PALETTE["orange"]),
        ("green_2sfca_1200m_access_log1p", "1200 m", PALETTE["pink"]),
    ]
    for col, label, color in fields:
        s = df[col].dropna()
        sns.kdeplot(s, ax=ax, color=color, label=label, linewidth=1.8, fill=True, alpha=0.12)
        mean_val = s.mean()
        ax.axvline(mean_val, color=color, linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("log1p m2/person", fontsize=7.5)
    ax.set_ylabel("Density", fontsize=7.5)
    ax.set_xlim(0, 2.5)
    ax.tick_params(labelsize=7.0)
    ax.legend(
        fontsize=8.2,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.6, 0.6),
        ncols=3,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    sns.despine(ax=ax)
    _panel_label(ax, "D")


def _add_scale_north_above_legend(
    ax: plt.Axes,
    base_loc: tuple[float, float] = (0.02, 0.60),
    scale_len_km: float = 5.0,
) -> None:
    """Draw a classic black/white segmented scale bar (5 km) and polygon North arrow above legend.
    Inspired by p2_fig10_policy.py."""
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
            sb_y0 + sb_h * 1.5,
            val_str,
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#222222",
            zorder=25,
        )
    ax.text(
        sb_x0 + slen,
        sb_y0 + sb_h * 1.5,
        f"{int(scale_len_km)} km",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color="#222222",
        zorder=25,
    )

    # 2. Polygon North Arrow above scale bar
    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * 4.2
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
        fontsize=8.5,
        fontweight="bold",
        color="#2C3E50",
        zorder=25,
    )


def _plot_fold_map(ax: plt.Axes, grid: gpd.GeoDataFrame, assign: pd.DataFrame) -> None:
    # Faint thin FUR background area with no fill
    grid.plot(ax=ax, facecolor="none", edgecolor="#B0B0B0", linewidth=0.15, alpha=0.5, zorder=1)

    # Outer boundary perimeter of the Izmir FUR area with a thin grey line
    fur_boundary = grid.unary_union
    gpd.GeoSeries([fur_boundary], crs=grid.crs).boundary.plot(
        ax=ax, color="#555555", linewidth=0.5, linestyle="-", zorder=10
    )

    g = grid.merge(assign[["grid_id", "fold"]], on="grid_id", how="inner")
    fold_colors = [
        "#5C6370",          # Fold 1: Slate Grey
        PALETTE["orange"],  # Fold 2: Orange
        PALETTE["pink"],    # Fold 3: Pink
        "#A39E98",          # Fold 4: Muted Grey
        PALETTE["teal"],    # Fold 5: Teal
    ]
    fold_cmap = mpl.colors.ListedColormap(fold_colors)

    g.plot(
        column="fold",
        categorical=True,
        cmap=fold_cmap,
        ax=ax,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )

    fold_counts = g["fold"].value_counts().sort_index()
    handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=f"Fold {i} (n={count:,})")
        for i, (color, count) in enumerate(zip(fold_colors, fold_counts), 1)
    ]

    # Legend font size enlarged to 11.5pt
    # Per user request, legend + scale/north group moved up together (by
    # the same 0.15 axes-fraction offset) into the map's own real blank
    # area above them (the FUR polygon's northern landmass doesn't start
    # until well above this group), rather than leaving that space empty.
    group_shift_up = 0.15
    ax.legend(
        handles=handles,
        fontsize=11.5,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.35 + group_shift_up),
        frameon=False,
        handletextpad=0.6,
        labelspacing=0.55,
    )

    _set_map_extent(ax, grid.total_bounds)
    _add_scale_north_above_legend(ax, base_loc=(0.035, 0.49 + group_shift_up), scale_len_km=5.0)
    _panel_label(ax, "E", x=0.012, y=0.94, va="top")



def build_figure_2_methodology(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_2_methodology_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(root / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    family_missing = pd.read_csv(root / "outputs/diagnostics/feature_diagnostics_v8_2026-07-27_coastline_fix/family_missingness.csv")
    summary = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/summary_metrics.csv")
    assign = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/spatial_fold_assignments.csv")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})

    # Map aspect ratio dy/dx = 33750 / 49250 = 0.685279
    map_ar = 0.685279

    # Row 1: 4 panels (a, b, c, d) side by side
    panel_w = 3.60
    panel_h = panel_w * map_ar  # 2.467
    h_gap = 0.35
    v_gap = 0.50

    left_margin = 0.60
    right_margin = 0.60
    top_margin = 0.55
    bottom_margin = 0.45

    # Total width spanned by panels a-d in Row 1
    w_total = 4 * panel_w + 3 * h_gap  # 15.45 in

    # Row 2: Panel E spans w_total, height matches map_ar
    panel_e_w = w_total
    panel_e_h = panel_e_w * map_ar  # 10.588 in

    fig_w = left_margin + w_total + right_margin
    fig_h = bottom_margin + panel_e_h + v_gap + panel_h + top_margin

    # Coordinates
    b_e = bottom_margin
    b_row1 = b_e + panel_e_h + v_gap

    left_a = left_margin
    left_b = left_a + panel_w + h_gap
    left_c = left_b + panel_w + h_gap
    left_d = left_c + panel_w + h_gap

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # Convert to normalized figure coordinates [left, bottom, width, height]
    ax_a = fig.add_axes([left_a / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_b = fig.add_axes([left_b / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_c = fig.add_axes([left_c / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_d = fig.add_axes([left_d / fig_w, b_row1 / fig_h, panel_w / fig_w, panel_h / fig_h])
    ax_e = fig.add_axes([left_margin / fig_w, b_e / fig_h, panel_e_w / fig_w, panel_e_h / fig_h])

    _plot_study_support(ax_a, grid, df)
    _plot_family_missing(ax_b, family_missing)
    _plot_recipe_performance(ax_c, summary)
    _plot_2sfca(ax_d, df)
    _plot_fold_map(ax_e, grid, assign)

    _finalize_panel_labels(fig)

    # Per user request, crop to the figure's own actual content extent
    # (same technique as Figures 4/7) rather than trusting the nominal
    # left/right margins -- panel E's map shape (irregular FUR coastline)
    # doesn't reach its own box edges evenly, so the nominal margins alone
    # understated the real left/right whitespace.
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

    png = fig_dir / "fig_2_methodology_workflow.png"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches=crop_bbox)
    plt.close(fig)


    qa = {
        "figure": "fig_2_methodology_workflow",
        "date": "2026-07-27",
        "panel_count": 5,
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "data/03_processed/grid_250m_model_features_v8.csv",
            "data/03_processed/analysis_grids.gpkg",
            "outputs/diagnostics/feature_diagnostics_v8_2026-07-27_coastline_fix/family_missingness.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/summary_metrics.csv",
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/spatial_fold_assignments.csv",
        ],
        "png": str(png),
        "claim_boundary": "Methodology figure shows completed tabular/network evidence maps and charts only.",
    }
    (out_dir / "fig_2_methodology_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_2_methodology_multipanel_qa.json")}


def build_graphical_abstract(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/graphical_abstract_multipanel_2026-07-27_coastline_fix"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/summary_metrics.csv")
    climate = json.loads((root / "outputs/diagnostics/manuscript_evidence_2026-07-27_coastline_fix/climate_validation.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(14.2, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.2)
    lanes = [
        (0.5, 5.0, 2.5, 1.25, PALETTE["teal_light"], PALETTE["teal"], "Izmir FUR", "250 m grid\nsurface model\nbuilding-street fabric"),
        (3.6, 5.0, 2.5, 1.25, PALETTE["orange_light"], PALETTE["orange"], "Evidence frame", "374 fields\nmorphology + network\nLST response-only"),
        (6.7, 5.0, 2.5, 1.25, PALETTE["teal_light"], PALETTE["teal"], "Spatial tests", "5 folds\nleave-district\n2SFCA sensitivity"),
        (9.8, 5.0, 3.2, 1.25, PALETTE["pink_light"], PALETTE["pink"], "Gate", "manual audit 0/569\ndeep branches not claimed"),
    ]
    for x, y, w, h, fc, ec, title, body in lanes:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + 0.43, body, ha="center", va="center", fontsize=8, linespacing=1.15)
    for x in [3.2, 6.3, 9.4]:
        ax.annotate("", xy=(x + 0.25, 5.62), xytext=(x - 0.25, 5.62), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": PALETTE["taupe_dark"]})
    primary_f1 = summary[(summary["model"].eq("lightgbm")) & (summary["recipe"].eq("baseline_d_full_proxy_context"))]["macro_f1_mean"].iloc[0]
    facts = [
        ("Macro-F1", f"{primary_f1:.3f}", PALETTE["teal"]),
        ("LST H", f"{climate['kruskal_h']:.0f}", PALETTE["pink"]),
        ("epsilon2", f"{climate['epsilon_squared']:.3f}", PALETTE["orange"]),
        ("2SFCA", "400/800/1200 m", PALETTE["taupe"]),
    ]
    for i, (k, v, color) in enumerate(facts):
        x = 1.0 + i * 3.1
        ax.add_patch(mpl.patches.FancyBboxPatch((x, 2.45), 2.55, 1.15, boxstyle="round,pad=0.04", facecolor="white", edgecolor=color, linewidth=1.4))
        ax.text(x + 1.275, 3.18, k, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x + 1.275, 2.76, v, ha="center", va="center", fontsize=10, color=PALETTE["ink"])
    ax.text(7.1, 1.25, "Climate-sensitive urban fabric learning in Izmir: strong weak-label evidence, explicit uncertainty, and gated confirmatory claims.", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["ink"])
    ax.text(7.1, 0.55, "Completed evidence supports reproducible climate-morphology diagnostics; audited LCZ-like subtype and deep multimodal claims remain future work.", ha="center", fontsize=8.2, color=PALETTE["muted"])
    png = fig_dir / "graphical_abstract.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    qa = {"figure": "graphical_abstract", "date": "2026-07-27", "panel_count": 8, "png": str(png), "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"]}
    (out_dir / "graphical_abstract_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "graphical_abstract_qa.json")}


if __name__ == "__main__":
    print(json.dumps({"figure_2": build_figure_2_methodology(), "graphical": build_graphical_abstract()}, indent=2))
