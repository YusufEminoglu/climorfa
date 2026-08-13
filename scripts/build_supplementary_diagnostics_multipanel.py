"""Build the final supplementary diagnostic figures.

The current manuscript supplement contains only evidence-bearing outputs:
feature quality and spatial-fold diagnostics. Municipal/reference park
triangulation, weak-label confidence panels, and the duplicate audit queue
were retired so the supplement matches the manuscript claim boundary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from build_figure_1_texture_atlas import CLASSES, PALETTE


PRIMARY_RECIPE = "baseline_d_full_proxy_context"


def _panel_label(ax: plt.Axes, label: str, *args, fontsize: float = 13.5, x: float = 0.005, y: float = 1.02, va: str = "bottom", **kwargs) -> None:
    label_text = f"({label.lower()})" if len(label) == 1 and label.isalpha() else label
    ax.text(
        x, y,
        label_text,
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


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _build_missingness(fig_dir: Path, out_dir: Path, root: Path) -> dict:
    fm = pd.read_csv(root / "outputs/diagnostics/feature_diagnostics_v8_2026-07-27_coastline_fix/family_missingness.csv")
    col = pd.read_csv(root / "outputs/diagnostics/feature_diagnostics_v8_2026-07-27_coastline_fix/collinearity_groups_corr95.csv")
    
    # 16.0 x 12.0 figure gives each of the 2x2 grid panels an exact 4:3 aspect ratio (8.0 x 6.0 in)
    fig = plt.figure(figsize=(16.0, 12.0), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        left=0.080, right=0.985, top=0.940, bottom=0.055,
        wspace=0.32, hspace=0.28,
    )
    
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bot_left = fig.add_subplot(gs[1, 0])
    ax_bot_right = fig.add_subplot(gs[1, 1])

    # ----------------------------------------------------
    # Panel A (TOP LEFT): Feature Family Inventory & Quality Breakdown
    # ----------------------------------------------------
    ax = ax_top_left
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
        
    fb = fm.sort_values("columns", ascending=True).copy()
    fb_y = np.arange(len(fb))
    
    near_const = fb["near_constant_columns"].values
    has_missing = fb["columns_with_any_missing"].values
    complete = np.maximum(0, fb["columns"].values - near_const - has_missing)
    
    bar_h_b = 0.58
    color_complete = PALETTE["teal"]       # Light fresh teal #87D8CD
    color_missing = PALETTE["pink"]        # Light fresh pink #F978A5
    color_lowvar = PALETTE["orange"]       # Light warm orange #F4AB55
    
    ax.barh(fb_y, complete, height=bar_h_b, color=color_complete, edgecolor="white", linewidth=0.5, label="Complete & variable columns")
    ax.barh(fb_y, has_missing, left=complete, height=bar_h_b, color=color_missing, edgecolor="white", linewidth=0.5, label="Columns with missing values")
    ax.barh(fb_y, near_const, left=complete + has_missing, height=bar_h_b, color=color_lowvar, edgecolor="white", linewidth=0.5, label="Near-constant / low-variance columns")

    # Annotate row badges
    for idx, row in enumerate(fb.itertuples()):
        tot = row.columns
        nc = row.near_constant_columns
        ms = row.columns_with_any_missing
        label_str = f"n={tot}"
        details = []
        if ms > 0:
            details.append(f"{ms} ms")
        if nc > 0:
            details.append(f"{nc} nc")
        if details:
            label_str += f" ({', '.join(details)})"
        ax.text(tot + 1.5, idx, label_str, ha="left", va="center", fontsize=8.2, color=PALETTE["ink"], fontweight="medium")

    fb_labels = [s.replace("_", " ") for s in fb["family"]]
    ax.set_yticks(fb_y, fb_labels)
    ax.tick_params(axis="y", labelsize=9.5, colors=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=9.5, colors=PALETTE["ink"])
    ax.set_xlabel("Feature column count", fontsize=11.0, fontweight="bold", color=PALETTE["ink"])
    ax.set_title("Feature family inventory & quality breakdown", fontsize=12.5, fontweight="bold", color=PALETTE["ink"], pad=14)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.6)
    ax.set_xlim(0, max(fb["columns"]) * 1.28)
    ax.legend(fontsize=8.8, frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")
    _panel_label(ax, "A")

    # ----------------------------------------------------
    # Panel B (TOP RIGHT): Standalone Spatial Domain Composition Donut Chart
    # ----------------------------------------------------
    ax = ax_top_right
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
    
    domain_map = {
        'road_network_reach': 'Network & Mobility',
        'road_network_exact': 'Network & Mobility',
        'preliminary_network': 'Network & Mobility',
        'building_footprint_morphometrics': 'Morphology & Form',
        'building_floor_proxy': 'Morphology & Form',
        'street_facing_morphology': 'Morphology & Form',
        'preliminary_morphology': 'Morphology & Form',
        'weak_lcz_label': 'Morphology & Form',
        'osm_proxy_context': 'Context & Demographics',
        'green_space_2sfca': 'Context & Demographics',
        'coastal_context': 'Context & Demographics',
        'population_demand_allocation': 'Context & Demographics',
        'sampling_frame': 'Context & Demographics',
        'dynamic_world': 'Satellite & Land Cover',
        'worldcover': 'Satellite & Land Cover',
        'sentinel2': 'Satellite & Land Cover',
        'canopy_height': 'Elevation & Canopy',
        'dsm_5m': 'Elevation & Canopy',
        'climate_response': 'Elevation & Canopy',
    }
    fm_temp = fm.copy()
    fm_temp['domain'] = fm_temp['family'].map(domain_map)
    dom_counts = fm_temp.groupby('domain')['columns'].sum().reset_index()
    dom_counts = dom_counts.sort_values('columns', ascending=False)

    domain_colors = {
        'Network & Mobility': PALETTE["teal"],
        'Context & Demographics': PALETTE["taupe"],
        'Morphology & Form': PALETTE["pink"],
        'Satellite & Land Cover': PALETTE["orange"],
        'Elevation & Canopy': PALETTE["teal_dark"],
    }
    
    # Donut Chart on left side of Panel B
    ax_pie = ax.inset_axes([0.02, 0.04, 0.48, 0.88])
    colors_list = [domain_colors[d] for d in dom_counts['domain']]
    
    wedges, texts, autotexts = ax_pie.pie(
        dom_counts['columns'],
        labels=None,
        autopct='%1.1f%%',
        pctdistance=0.74,
        startangle=140,
        colors=colors_list,
        wedgeprops=dict(width=0.40, edgecolor='white', linewidth=2.0),
    )
    for at in autotexts:
        at.set_fontsize(9.2)
        at.set_fontweight("bold")
        at.set_color("#111111")
        
    ax_pie.text(0, 0.14, "N = 385", ha="center", va="center", fontsize=12.0, fontweight="bold", color=PALETTE["ink"])
    ax_pie.text(0, -0.12, "94.2% Clean", ha="center", va="center", fontsize=9.8, fontweight="bold", color=PALETTE["teal_dark"])

    # Domain Data Legend on right side of Panel B
    y_start = 0.78
    ax.text(0.52, 0.90, "Spatial Domain Breakdown", fontsize=11.0, fontweight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.plot([0.52, 0.96], [0.85, 0.85], color="#cccccc", linewidth=0.8, transform=ax.transAxes)
    
    for r_idx, row in enumerate(dom_counts.itertuples()):
        c_color = domain_colors[row.domain]
        pct = (row.columns / 385) * 100
        y_p = y_start - r_idx * 0.15
        
        # Color marker square
        ax.scatter(0.54, y_p, s=85, marker="s", color=c_color, edgecolors="white", linewidth=0.6, transform=ax.transAxes, zorder=5)
        # Domain name
        ax.text(0.59, y_p + 0.022, f"{row.domain}", fontsize=9.2, fontweight="bold", color=PALETTE["ink"], transform=ax.transAxes, va="center")
        # Column count badge
        ax.text(0.59, y_p - 0.025, f"{row.columns} columns ({pct:.1f}%)", fontsize=8.4, fontweight="bold", color=c_color, transform=ax.transAxes, va="center")

    ax.axis("off")
    ax.set_title("Spatial domain composition & data health", fontsize=12.5, fontweight="bold", color=PALETTE["ink"], pad=14)
    _panel_label(ax, "B")

    # ----------------------------------------------------
    # Panel C (BOTTOM LEFT): Missingness Dumbbell / Gap Plot (Mean vs Max)
    # ----------------------------------------------------
    ax = ax_bot_left
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
        
    f = fm[fm["max_missing_share"] > 0].sort_values("avg_missing_share").copy()
    y = np.arange(len(f))
    
    family_labels = [s.replace("_", " ") for s in f["family"]]
    color_mean = PALETTE["teal"]
    color_max = PALETTE["pink"]
    
    # Draw dumbbell connecting gap lines
    for idx, row in enumerate(f.itertuples()):
        m_val = row.avg_missing_share * 100
        mx_val = row.max_missing_share * 100
        ax.plot([m_val, mx_val], [idx, idx], color="#d5d0ca", linewidth=4.5, zorder=2, solid_capstyle="round")
        ax.plot([m_val, mx_val], [idx, idx], color="#a09a90", linestyle=":", linewidth=1.0, zorder=3)
    
    # Plot dots
    scatter_mean = ax.scatter(f["avg_missing_share"] * 100, y, s=100, color=color_mean, edgecolors=PALETTE["teal_dark"], linewidth=1.2, zorder=5, label="Mean missingness")
    scatter_max = ax.scatter(f["max_missing_share"] * 100, y, s=100, color=color_max, edgecolors=PALETTE["pink_dark"], linewidth=1.2, zorder=5, label="Max missingness")
    
    # Precise value callouts
    for idx, row in enumerate(f.itertuples()):
        m_val = row.avg_missing_share * 100
        mx_val = row.max_missing_share * 100
        ax.text(m_val, idx + 0.28, f"{m_val:.1f}%", ha="center", va="bottom", fontsize=8.0, color=PALETTE["teal_dark"], fontweight="bold")
        ax.text(mx_val, idx - 0.30, f"{mx_val:.1f}%", ha="center", va="top", fontsize=8.0, color=PALETTE["pink_dark"], fontweight="bold")

    ax.set_yticks(y, family_labels)
    ax.tick_params(axis="y", labelsize=9.5, colors=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=9.5, colors=PALETTE["ink"])
    ax.set_xlabel("Missingness (%)", fontsize=11.0, fontweight="bold", color=PALETTE["ink"])
    ax.set_title("Missingness gap profile (Mean vs Max %)", fontsize=12.5, fontweight="bold", color=PALETTE["ink"], pad=14)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.6)
    ax.grid(axis="y", color="#f0ece6", linestyle=":", linewidth=0.5)
    ax.set_xlim(-2, max(f["max_missing_share"] * 100) * 1.18)
    ax.legend(fontsize=8.8, frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")
    _panel_label(ax, "C")

    # ----------------------------------------------------
    # Panel D (BOTTOM RIGHT): Largest Collinearity Clusters (r >= 0.95 Lollipop)
    # ----------------------------------------------------
    ax = ax_bot_right
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
    
    cluster_names = {
        "corr95_016": "Road & intersection reach",
        "corr95_003": "Building count & perimeter",
        "corr95_002": "Building area & coverage",
        "corr95_001": "Network permeability",
        "corr95_008": "Surface model elevation",
        "corr95_038": "Road presence & continuity",
        "corr95_012": "Height proxy & max floors",
        "corr95_004": "Floor-area ratio & volume",
        "corr95_007": "Coastal distance metrics",
        "corr95_006": "Canopy volume proxies",
    }
    
    top = col.sort_values("feature_count", ascending=False).head(10).sort_values("feature_count").copy()
    top["clean_label"] = top["group_id"].map(lambda gid: f"{cluster_names.get(gid, gid)}")
    
    c_y = np.arange(len(top))
    color_cluster = PALETTE["taupe"]        # Light taupe #ACA49C
    
    # Draw Lollipop stems & nodes
    ax.hlines(c_y, 0, top["feature_count"], color="#8c867e", linewidth=2.6, zorder=2)
    ax.scatter(top["feature_count"], c_y, s=160, color=color_cluster, edgecolors=PALETTE["taupe_dark"], linewidth=1.4, zorder=5)
    
    for idx, row in enumerate(top.itertuples()):
        w = row.feature_count
        ax.text(w, idx, f"{w}", ha="center", va="center", fontsize=8.5, color="#ffffff", fontweight="bold", zorder=6)

    ax.set_yticks(c_y, top["clean_label"])
    ax.tick_params(axis="y", labelsize=9.5, pad=6, colors=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=9.5, colors=PALETTE["ink"])
    ax.set_xlabel("Feature count in cluster", fontsize=11.0, fontweight="bold", color=PALETTE["ink"])
    ax.set_title(r"Largest collinearity clusters ($r \geq 0.95$)", fontsize=12.5, fontweight="bold", color=PALETTE["ink"], pad=14)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.6)
    ax.set_xlim(0, max(top["feature_count"]) * 1.15)
    _panel_label(ax, "D")

    path = fig_dir / "fig_s2_feature_quality.png"
    _save(fig, path)
    return {"figure": path.name, "panel_count": 4, "png": str(path)}


def _build_folds(fig_dir: Path, out_dir: Path, root: Path) -> dict:
    fold_df = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/fold_metrics.csv")
    sum_df = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/summary_metrics.csv")
    pooled_df = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/pooled_class_metrics.csv")

    fig, axs = plt.subplots(1, 2, figsize=(16.5, 7.0), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.988, top=0.920, bottom=0.100, wspace=0.34)

    # ----------------------------------------------------
    # Panel A: Spatial Cross-Validation Progression Across Feature Recipes
    # ----------------------------------------------------
    ax = axs[0]
    ax.set_facecolor("#ffffff")
    
    recipes = [
        ("baseline_a_morphology_only", "Baseline A\n(Morphology, 15f)", PALETTE["taupe"], PALETTE["taupe_dark"]),
        ("baseline_b_morphology_plus_vegetation", "Baseline B\n(+Vegetation, 22f)", PALETTE["orange"], PALETTE["orange_dark"]),
        ("baseline_c_morphology_plus_green_blue_context", "Baseline C\n(+Green/Blue, 29f)", PALETTE["pink"], PALETTE["pink_dark"]),
        ("baseline_d_full_proxy_context", "Baseline D\n(Full Context, 68f)", PALETTE["teal"], PALETTE["teal_dark"]),
    ]
    
    lgb_folds = fold_df[fold_df["model"].eq("lightgbm")].copy()
    
    x_positions = np.arange(len(recipes))
    bar_width = 0.52
    
    # Draw primary recipe mean bars
    means = []
    sds = []
    for r_idx, (r_code, r_name, c_main, c_dark) in enumerate(recipes):
        vals = lgb_folds[lgb_folds["recipe"] == r_code]["macro_f1"].values
        m_val = vals.mean() if len(vals) > 0 else 0
        s_val = vals.std() if len(vals) > 0 else 0
        means.append(m_val)
        sds.append(s_val)
        
        ax.bar(r_idx, m_val, width=bar_width, color=c_main, edgecolor="white", linewidth=1.0, zorder=2, alpha=0.85)
        
        # Mean callout badge box
        ax.text(
            r_idx, m_val + 0.045,
            f"Mean F1 = {m_val:.3f}\n(±{s_val:.3f} SD)",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=c_dark,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=c_dark, linewidth=1.0, alpha=0.95),
            zorder=10,
        )

    # Connect individual spatial fold trajectories across recipes to show consistent fold-wise gain
    folds = [1, 2, 3, 4, 5]
    fold_markers = ["o", "s", "^", "D", "v"]
    fold_colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    
    for f_idx, fold_num in enumerate(folds):
        f_y = []
        for r_code, r_name, c_main, c_dark in recipes:
            val = lgb_folds[(lgb_folds["recipe"] == r_code) & (lgb_folds["fold"] == fold_num)]["macro_f1"].values
            f_y.append(val[0] if len(val) > 0 else 0)
        
        ax.plot(x_positions, f_y, linestyle="--", linewidth=1.1, color="#777777", alpha=0.7, zorder=4)
        ax.scatter(x_positions, f_y, marker=fold_markers[f_idx], s=45, color=fold_colors[f_idx], edgecolors="#ffffff", linewidth=0.8, zorder=5, label=f"Fold {fold_num}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([r[1] for r in recipes], fontsize=9.2, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9.2, colors=PALETTE["ink"])
    ax.set_ylabel("Spatial Out-of-Fold Macro F1 Score", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.set_title("Spatial cross-validation stability across feature recipes", fontsize=12.0, fontweight="bold", color=PALETTE["ink"], pad=14)
    ax.grid(axis="y", color=PALETTE["grid"], linestyle="--", linewidth=0.6)
    ax.set_ylim(0.52, 0.94)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", title="Spatial Folds", title_fontsize=8.8, ncols=2)
    _panel_label(ax, "A")

    # ----------------------------------------------------
    # Panel B: Model Benchmark Generalization Across Spatial Folds
    # ----------------------------------------------------
    ax = axs[1]
    ax.set_facecolor("#ffffff")
    
    # Model family comparison under Baseline D (excluding dummy majority to zoom in on model differences)
    bd_models = sum_df[(sum_df["recipe"] == "baseline_d_full_proxy_context") & (sum_df["model"] != "dummy_majority")].sort_values("macro_f1_mean").copy()
    
    model_labels = {
        "extra_trees": "Extra Trees",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM (Primary)",
    }
    bd_models["clean_model"] = bd_models["model"].map(model_labels)
    
    m_y = np.arange(len(bd_models))
    m_colors = [PALETTE["taupe"], PALETTE["orange"], PALETTE["pink_dark"], PALETTE["teal_dark"]]
    
    bars_m = ax.barh(m_y, bd_models["macro_f1_mean"], xerr=bd_models["macro_f1_sd"], height=0.48, color=m_colors, edgecolor="none", capsize=4, error_kw={"elinewidth": 1.4, "ecolor": "#333333"}, zorder=3)
    
    for idx, row in enumerate(bd_models.itertuples()):
        val = row.macro_f1_mean
        sd = row.macro_f1_sd
        ax.text(val + sd + 0.003, idx, f"{val:.3f} (±{sd:.3f} SD)", ha="left", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])

    ax.set_yticks(m_y, bd_models["clean_model"])
    ax.tick_params(axis="y", labelsize=9.5, colors=PALETTE["ink"])
    ax.tick_params(axis="x", labelsize=9.5, colors=PALETTE["ink"])
    ax.set_xlabel("Spatial Out-of-Fold Macro F1 Score", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.set_title("Model generalization benchmark across spatial folds", fontsize=12.0, fontweight="bold", color=PALETTE["ink"], pad=14)
    ax.grid(axis="x", color=PALETTE["grid"], linestyle="--", linewidth=0.6)
    ax.set_xlim(0.76, 0.885)
    
    _panel_label(ax, "B")

    path = fig_dir / "fig_s3_spatial_fold_diagnostics.png"
    _save(fig, path)
    return {"figure": path.name, "panel_count": 2, "png": str(path)}


def build_supplementary_diagnostics(root: Path | str = Path(".")) -> dict[str, dict]:
    root = Path(root)
    fig_dir = root / "paper/figures/supplementary"
    out_dir = root / "outputs/diagnostics/supplementary_diagnostics_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    outputs = {
        "feature_quality": _build_missingness(fig_dir, out_dir, root),
        "spatial_fold_diagnostics": _build_folds(fig_dir, out_dir, root),
    }
    qa = {
        "date": "2026-08-03",
        "figures": outputs,
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "claim_boundary": "Supplementary diagnostics support weak-label interpretation and reproducibility; they do not provide audited local-subtype validation.",
        "revision_2026-08-03": "Retired manual_audit_queue (overlapped Figure 3) and dead weak_label_confidence code per a full supplementary-figure audit against the main text; trimmed spatial_folds from 6 to 2 panels (cut the Figure 2(e) duplicate map and the pre-LightGBM-migration RF/ET-only panels); merged feature_missingness's mean/max missingness panels into one grouped bar (4 -> 3 panels).",
        "revision_2026-08-04": "Removed the municipal/reference park triangulation from the manuscript supplement at user request; after the merged S1 texture-surface-canopy atlas, feature quality is S2 and spatial-fold diagnostics is S3.",
    }
    (out_dir / "supplementary_diagnostics_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    print(json.dumps(build_supplementary_diagnostics(), indent=2))
