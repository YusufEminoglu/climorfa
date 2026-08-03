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


def _panel_label(ax: plt.Axes, label: str, *args, fontsize: float = 14.0, x: float = 0.005, y: float = 1.02, va: str = "bottom", **kwargs) -> None:
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
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.34)

    # Panel A: Mean and maximum missingness per family, combined into one
    # grouped bar (was two separate near-identical panels -- same families,
    # same sort dimension, only the aggregation statistic differed).
    ax = axs[0]
    ax.set_facecolor("#ffffff")
    f = fm.sort_values("avg_missing_share")
    y = np.arange(len(f))
    bar_h = 0.36
    ax.barh(y + bar_h / 2, f["avg_missing_share"] * 100, height=bar_h, color=PALETTE["teal_dark"], label="Mean")
    ax.barh(y - bar_h / 2, f["max_missing_share"] * 100, height=bar_h, color=PALETTE["pink_dark"], label="Maximum")
    ax.set_yticks(y, f["family"].str.replace("_", " "))
    ax.tick_params(axis="y", labelsize=8.0)
    ax.set_xlabel("Missingness (%)", fontsize=9.0, fontweight="bold")
    ax.set_title("Feature family mean vs. maximum missingness", fontsize=10.5, fontweight="bold", pad=26)
    ax.grid(axis="x", color="#e5e5e5", linestyle="--", linewidth=0.6)
    ax.legend(fontsize=8.0, frameon=True, facecolor="white", edgecolor="#cccccc")
    _panel_label(ax, "A")

    # Panel B: Feature count vs near constant
    ax = axs[1]
    ax.set_facecolor("#ffffff")
    sc = ax.scatter(fm["columns"], fm["near_constant_columns"], s=45 + fm["columns"] * 2, c=fm["avg_missing_share"] * 100, cmap="viridis", edgecolors="white", linewidth=0.8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Missingness (%)", fontsize=8.0, fontweight="bold")
    ax.set_xlabel("Family total columns", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Near-constant columns", fontsize=9.0, fontweight="bold")
    ax.set_title("Feature count vs. near-constant share", fontsize=10.5, fontweight="bold", pad=26)
    ax.grid(color="#e5e5e5", linestyle="--", linewidth=0.6)
    _panel_label(ax, "B")

    # Panel C: Collinearity groups
    ax = axs[2]
    ax.set_facecolor("#ffffff")
    top = col.sort_values("feature_count", ascending=False).head(10).sort_values("feature_count")
    ax.barh(top["group_id"], top["feature_count"], color=PALETTE["taupe_dark"])
    ax.set_xlabel("Feature count in group", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Collinearity Group ID", fontsize=9.0, fontweight="bold")
    ax.set_title(r"Largest collinearity clusters ($r \geq 0.95$)", fontsize=10.5, fontweight="bold", pad=26)
    ax.grid(axis="x", color="#e5e5e5", linestyle="--", linewidth=0.6)
    _panel_label(ax, "C")

    path = fig_dir / "fig_s2_feature_quality.png"
    _save(fig, path)
    return {"figure": path.name, "panel_count": 3, "png": str(path)}


def _build_folds(fig_dir: Path, out_dir: Path, root: Path) -> dict:
    assign = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/spatial_fold_assignments.csv")
    pred = pd.read_csv(root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/out_of_fold_predictions.csv")

    # Per 2026-08-03 user-requested audit against the main text: this used
    # to be a 6-panel figure. Panel A (spatial fold map) was cut -- it was
    # an exact duplicate of Figure 2 panel (e), same data, same map, at
    # lower resolution here. Panel B (cell counts per fold) was cut -- the
    # same counts are already stated directly in Figure 2(e)'s own legend
    # ("Fold 1, n=1,068" etc). Former panels D (macro-F1 by fold) and E
    # (test blocks by fold) were cut outright rather than updated to
    # LightGBM: they only ever compared Random Forest/Extra Trees (a
    # pre-LightGBM-migration leftover), and Figure 5's own fold-level
    # spread panel already covers macro-F1 by fold for all four models
    # more completely. What remains -- class support per fold and
    # LightGBM's own OOF-confidence spread per fold -- is not shown
    # anywhere else in the paper.
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.13, wspace=0.34)

    # Panel A: Class support per fold
    ax = axs[0]
    mat = pd.crosstab(assign["fold"], assign["lcz_weak_label"]).reindex(columns=CLASSES)
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "Cells", "shrink": 0.75})
    ax.set_title("Class support per fold", fontsize=10.5, fontweight="bold", pad=14)
    ax.set_xlabel("Weak LCZ Class", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Fold index", fontsize=9.0, fontweight="bold")
    _panel_label(ax, "A")

    # Panel B: LightGBM OOF confidence per fold
    ax = axs[1]
    ax.set_facecolor("#ffffff")
    p = pred[(pred["recipe"].eq(PRIMARY_RECIPE)) & pred["model"].eq("lightgbm")]
    sns.boxplot(data=p, x="fold", y="predicted_probability", color=PALETTE["teal_dark"], ax=ax, showfliers=False)
    ax.set_title("LightGBM prediction probability by fold", fontsize=10.5, fontweight="bold", pad=14)
    ax.set_xlabel("Fold index", fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Predicted probability", fontsize=9.0, fontweight="bold")
    ax.grid(axis="y", color="#e5e5e5", linestyle="--", linewidth=0.6)
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
