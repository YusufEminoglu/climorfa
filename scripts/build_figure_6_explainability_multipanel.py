"""Build a multi-panel model explainability and evidence-gate figure."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from build_figure_1_texture_atlas import PALETTE


PRIMARY_RECIPE = "baseline_d_full_proxy_context"
PRIMARY_MODEL = "lightgbm"
MODEL_LABELS = {"random_forest": "Random Forest", "extra_trees": "Extra Trees", "lightgbm": "LightGBM"}
MODEL_COLORS = {"random_forest": PALETTE["pink_dark"], "extra_trees": PALETTE["orange"], "lightgbm": PALETTE["teal"]}


def _family(feature: str) -> str:
    f = feature.lower()
    if f.startswith("green_2sfca"):
        return "Green access"
    if f.startswith("dsm_") or f.startswith("height_") or "floor" in f or "rise_" in f or "volume" in f:
        return "3D / Surface"
    if f.startswith("canopy") or "ndvi" in f or "trees" in f or "tree_cover" in f:
        return "Vegetation"
    if f.startswith("building") or f.startswith("morph_") or "compactness" in f:
        return "2D form"
    if f.startswith("road") or f.startswith("network") or f.startswith("street_frontage") or f.startswith("block_"):
        return "Street / network"
    if f.startswith("coast") or f.startswith("coastal") or f.startswith("osm_"):
        return "Context"
    if f.startswith("s2_") or f.startswith("dw_") or f.startswith("worldcover"):
        return "Remote semantic"
    return "Other"


FAMILY_COLORS = {
    "Remote semantic": PALETTE["pink"],
    "2D form": PALETTE["teal"],
    "3D / Surface": PALETTE["orange"],
    "Vegetation": "#4FB286",
    "Street / network": PALETTE["taupe"],
    "Green access": "#61A0AF",
    "Context": "#C07C7C",
    "Other": "#B8B8B8",
}
FAMILY_DISPLAY = {
    "Remote semantic": "Remote sem.",
    "2D form": "2D form",
    "3D / Surface": "3D / Surface",
    "Vegetation": "Vegetation",
    "Street / network": "Street/net.",
    "Green access": "Green access",
    "Context": "Context",
    "Other": "Other",
}


def _short_feature(feature: str) -> str:
    replacements = {
        "worldcover_share_built_up": "WorldCover built",
        "worldcover_share_tree_cover": "WorldCover tree",
        "worldcover_share_water": "WorldCover water",
        "dw_crops_prob_mean": "DW crops",
        "dw_built_prob_mean": "DW built",
        "dw_trees_prob_mean": "DW trees",
        "dw_water_prob_mean": "DW water",
        "s2_ndwi_mean": "Sentinel-2 NDWI",
        "s2_ndvi_mean": "Sentinel-2 NDVI",
        "s2_ndbi_mean": "Sentinel-2 NDBI",
        "building_coverage_exact": "Building coverage",
        "morph_building_perimeter_density_m_per_ha": "Building perimeter dens.",
        "floor_area_ratio_proxy": "Floor-area ratio proxy",
        "street_frontage_open_buffer_share": "Open frontage buffer",
        "coast_min_distance_m": "Coast distance",
        "canopy_volume_gt2m_proxy_m3_per_ha": "Canopy volume >2 m",
        "green_2sfca_800m_access_log1p": "2SFCA 800 m",
        "road_density_exact_m_per_km2": "Road density",
        "network_intersection_density_per_km2": "Intersection density",
        "dsm_elevation_m_std": "Surface SD",
        "height_proxy_aw_mean_m": "Height proxy",
    }
    if feature in replacements:
        return replacements[feature]
    text = feature.replace("_mean", "").replace("_exact", "").replace("_proxy", "").replace("_", " ")
    text = text.replace("per km2", "/km2").replace("gt2m", ">2m")
    return text[:34]


def _panel_label(ax: plt.Axes, label: str) -> None:
    # Positioned above the axes (not inside its top-left corner) with a
    # neutral border, matching the house convention used in Figures 2 and 4
    # rather than this script's own former in-corner, per-model-colored tag.
    ax.text(
        0.005,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
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


def _summarize_importance(by_fold: pd.DataFrame) -> pd.DataFrame:
    primary = by_fold[by_fold["recipe"].eq(PRIMARY_RECIPE) & by_fold["model"].isin(MODEL_LABELS)].copy()
    primary["family"] = primary["feature"].map(_family)
    summary = (
        primary.groupby(["model", "feature", "family"], as_index=False)["importance"]
        .agg(importance_mean="mean", importance_sd="std")
        .sort_values(["model", "importance_mean"], ascending=[True, False])
    )
    summary["feature_short"] = summary["feature"].map(_short_feature)
    return summary


def _plot_family(ax: plt.Axes, by_fold: pd.DataFrame, model: str = PRIMARY_MODEL) -> pd.DataFrame:
    primary = by_fold[by_fold["recipe"].eq(PRIMARY_RECIPE) & by_fold["model"].eq(model)].copy()
    primary["family"] = primary["feature"].map(_family)
    fam_fold = primary.groupby(["model", "fold", "family"], as_index=False)["importance"].sum()
    # Between-fold SD alongside the mean: A previously showed only a point
    # estimate per family with no robustness signal, unlike B, which has
    # always carried a fold-SD whisker per feature -- this closes that gap
    # at the family level using the same 5-fold spread B already uses.
    fam_stats = fam_fold.groupby("family")["importance"].agg(importance_mean="mean", importance_sd="std").reset_index()
    order = fam_stats.sort_values("importance_mean", ascending=True)["family"].tolist()
    fam_stats = fam_stats.set_index("family").loc[order].reset_index()
    values = fam_stats.set_index("family").loc[order, "importance_mean"]
    sds = fam_stats.set_index("family").loc[order, "importance_sd"].fillna(0)
    colors = [FAMILY_COLORS.get(f, "#B8B8B8") for f in order]
    y = list(range(len(order)))
    # Capsule bars: a thick, round-capped line from the baseline to each
    # value reads as a softer, more considered "pill" than a filled
    # rectangle with square corners -- the mark alone carries the value, so
    # no border is needed to separate it from the page.
    for yi, val, color in zip(y, values, colors):
        ax.plot([0, val], [yi, yi], color=color, linewidth=13, solid_capstyle="round", zorder=3, alpha=0.94)
    ax.errorbar(values, y, xerr=sds, fmt="none", ecolor=PALETTE["muted"], elinewidth=1.2, capsize=3.5, zorder=4)
    for yi, val, sd_val in zip(y, values, sds):
        ax.text(val + sd_val + values.max() * 0.02, yi, f"{val * 100:.1f}%", va="center", ha="left", fontsize=9.5, color=PALETTE["ink"], fontweight="bold")
    ax.set_yticks(y, [FAMILY_DISPLAY.get(x, x) for x in order])
    ax.set_ylim(-0.65, len(order) - 0.35)
    ax.set_xlim(0, (values.max() + sds.max()) * 1.2)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Summed normalized importance", fontsize=11)
    ax.set_ylabel("")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11, length=0)
    ax.tick_params(axis="x", labelsize=9.5)
    _panel_label(ax, "A")
    return fam_stats


def _plot_top(ax: plt.Axes, summary: pd.DataFrame, model: str, label: str) -> pd.DataFrame:
    top = summary[summary["model"].eq(model)].nlargest(16, "importance_mean").sort_values("importance_mean")
    colors = [FAMILY_COLORS.get(f, "#B8B8B8") for f in top["family"]]
    y = list(range(len(top)))
    val = top["importance_mean"]
    sd = top["importance_sd"].fillna(0)
    # Lollipop form (thin stem + marker + whisker) reads as more refined
    # than 16 stacked filled bars, and the fold-SD whisker rides through the
    # marker itself rather than needing its own bar-chart error caps.
    ax.hlines(y, 0, val, color=colors, linewidth=1.6, alpha=0.55, zorder=2)
    ax.errorbar(val, y, xerr=sd, fmt="none", ecolor=PALETTE["muted"], elinewidth=1.1, capsize=2.5, zorder=3)
    ax.scatter(val, y, s=48, color=colors, edgecolors="white", linewidth=1.1, zorder=4)
    # A reference line at the mean of these 16 shown features gives the
    # panel a "typical" anchor -- without it, a reader has no way to tell
    # which of the displayed features are only modestly above the pack
    # versus genuinely standing out, since all 16 are already pre-filtered
    # to be the top performers.
    mean_val = val.mean()
    ax.axvline(mean_val, color=PALETTE["muted"], linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.text(mean_val, len(top) - 0.15, f"mean of these 16: {mean_val * 100:.1f}%", color=PALETTE["muted"], fontsize=7.6, ha="left", va="bottom", rotation=0, style="italic")
    ax.set_yticks(y, top["feature_short"])
    ax.set_ylim(-0.65, len(top) - 0.35)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Mean importance", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    _panel_label(ax, label)
    # Dot color is categorical by family here, so a legend belongs (per
    # house convention, >=2 series always gets one); kept small and tucked
    # into the panel's own dead space rather than competing with the marks.
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markersize=6, markerfacecolor=c, markeredgecolor="white", label=FAMILY_DISPLAY[f])
        for f, c in FAMILY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=True, ncol=2, handletextpad=0.35, columnspacing=0.9, borderpad=0.5, labelspacing=0.35)
    return top


def build_figure_6_explainability(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    model_dir = root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix"
    fig_dir = root / "paper/figures/main"
    out_dir = root / "outputs/diagnostics/figure_6_explainability_multipanel_2026-06-20"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_fold = pd.read_csv(model_dir / "feature_importance_by_fold.csv")
    # Random Forest's impurity importances already sum to 1 within each
    # (recipe, fold); LightGBM's gain importances do not (raw split-gain
    # totals in the thousands). Normalizing both to shares within each
    # (model, recipe, fold) puts them on a comparable scale for the panels
    # below, which all read "importance" as a share, not a raw magnitude.
    by_fold["importance"] = by_fold.groupby(["model", "recipe", "fold"])["importance"].transform(lambda x: x / x.sum())

    summary = _summarize_importance(by_fold)
    summary.to_csv(out_dir / "fig_6_importance_summary_by_model.csv", index=False)

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.6, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.105,
        right=0.985,
        top=0.93,
        bottom=0.14,
        wspace=0.38,
        width_ratios=[0.85, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    family_summary = _plot_family(ax_a, by_fold)
    # Cut down to 2 panels (from 3) per user request, after confirming panel
    # C (class medians for the leading features) was the one soft spot in an
    # otherwise-necessary figure: 2 of its 12 fields (building coverage,
    # NDVI) partially overlapped Figure 4 panel B's own class-profile
    # boxplots. Family decomposition (A) and named top features with fold-SD
    # (B) are each the only place in the paper answering their question and
    # are directly cited by name/number in the results-section prose, so
    # they stay; the bridge-to-phenotype panel did not clear that same bar.
    _plot_top(ax_b, summary, PRIMARY_MODEL, "B")

    _finalize_panel_labels(fig)

    # Per user request, crop to the figure's own actual content extent
    # (same technique as Figures 2/3/4/7) with an exact 0.5 cm pad on all
    # four sides, rather than trusting the nominal figure margins.
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig_w_in, fig_h_in = fig.get_size_inches()
    pad_in = 0.5 / 2.54
    crop_bbox = mpl.transforms.Bbox.from_extents(
        max(tight_bbox.x0 - pad_in, 0.0),
        max(tight_bbox.y0 - pad_in, 0.0),
        min(tight_bbox.x1 + pad_in, fig_w_in),
        min(tight_bbox.y1 + pad_in, fig_h_in),
    )

    png = fig_dir / "fig_6_feature_importance.png"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches=crop_bbox)
    plt.close(fig)

    qa = {
        "figure": "fig_6_feature_importance",
        "date": "2026-06-20",
        "panel_count": 2,
        "panel_labels": list("AB"),
        "layout": "single row, 2 panels: A family decomposition, B top-16 features with fold SD",
        "palette_roots": ["#87D8CD", "#F978A5", "#F4AB55", "#ACA49C"],
        "source_paths": [
            "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/feature_importance_by_fold.csv",
        ],
        "importance_summary_csv": str(out_dir / "fig_6_importance_summary_by_model.csv"),
        "family_summary_csv": str(out_dir / "fig_6_family_importance.csv"),
        "primary_model": PRIMARY_MODEL,
        "revision_2026-07-31": "LightGBM promoted to primary model (panels a-f, h-k), matching Figure 5. Random Forest kept as the named comparison baseline in panel g only. LightGBM's gain-based feature_importances_ are not normalized to sum to 1 the way Random Forest's impurity importances are, so by_fold importances are now normalized to shares within each (model, recipe, fold) before any panel uses them, making a/b/c/d/e/f comparable in scale to the prior Random Forest version. Also fixed a pre-existing bug in panel g: the recipe x-axis was sorted alphabetically by seaborn's default categorical handling instead of the additive recipe order; recipe_label is now an explicit ordered Categorical.",
        "revision_2026-08-02": "First panel-necessity review against predecessor Figure 5 and successor Figure 7, cutting the figure from 12 grid cells (11 labeled panels) to 9 (A-I): removed a macro-F1-vs-recipe ablation line duplicating Figure 5 panel A, an importance-vs-fold-SD scatter redundant with the adjacent fold-stability heatmap, and a literal empty unlabeled grid cell.",
        "revision_2026-08-02b": "Second, much larger cut per explicit user request ('2-3 kaliteli panele dusurelim, panel F sonrasi olmasin'): reduced further from 9 panels to 3 (A-C, single row). Kept only family decomposition (A), named top-16 features with fold-SD error bars (B -- the error bars already carry the fold-stability signal the now-cut dedicated stability heatmap used to show), and class medians for the leading features (C, widened from top 6 to top 12, the bridge back to Figure 4's class-phenotype story). Cut entirely: the family-vs-recipe sensitivity heatmap (recipe sensitivity is Figure 5's job, not this figure's), and all four per-class texture-tile exemplars (panels F-I) -- representative/exemplar texture tiles for LCZ 3/6/8/9 already appear in Figures 1, 4, and 7, so a fourth set here was evidence-variety padding, not new information. `_select_correct_texture_examples`, `_add_correct_badge`, `_plot_recipe_family_heatmap`, `_plot_stability`, and `_plot_recipe` were deleted outright (dead code) along with their CSV outputs and now-unused `out_of_fold_predictions.csv`/`analysis_grids.gpkg`/`akilli_sehir_fua.gpkg` source reads. Figure footprint shrunk from a 3-row, ~11.6in-tall multi-panel grid to a single-row, 6.6in-tall 3-panel strip.",
        "revision_2026-08-02c": "Visual redesign per explicit user request for a more premium presentation ('inanilmaz elite bir gosterime sahip olmali'), applying the project's dataviz-skill mark conventions rather than default seaborn/matplotlib chart chrome. Panel A: default filled barh replaced with thick round-capped 'capsule' line segments (thin marks + rounded data-ends, no border needed to separate a mark from the page) plus a direct value label at each tip. Panel B: default barh-with-error-caps replaced with a Cleveland-style lollipop (thin stem + circular marker + fold-SD whisker riding through the marker), which reads as more refined than 16 stacked filled bars at this row count; added a compact family-color legend (dot color is categorical here, so per house convention >=2 series always gets one -- it was previously missing). Both panels also got top/right/left spines removed, hairline recessive gridlines, and zero-length y-tick marks, matching the quieter, considered look used elsewhere in this paper's figures. (This revision originally also touched a third panel, C, a class-medians heatmap; that panel was cut entirely in revision_2026-08-02d below, so its colormap fix no longer applies to anything in the current figure.)",
        "revision_2026-08-02d": "Cut panel C (class medians for the leading features) after user asked for a direct honesty check on whether the just-trimmed 3-panel figure was genuinely non-redundant. On inspection, C was the one soft spot: 2 of its 12 fields (building coverage, NDVI) partially overlapped Figure 4 panel B's own class-profile boxplots (different field selection rule and chart form -- importance-ranked top-12 as standardized medians vs. a fixed 7-field list as full distributions -- so not an exact duplicate, but a real partial overlap). Panels A and B cleared the bar (each is the only place in the paper answering its question, and B's exact numbers are cited by name in the results-section prose) and stayed. Figure is now 2 panels, A-B, single row.",
        "revision_2026-08-02e": "Polish per user request: (1) panel-letter tags moved from inside each axes' top-left corner (per-model-colored border) to above the axes with a neutral border, matching the house convention used in Figures 2 and 4 (`_panel_label` no longer takes a `color` argument); (2) the per-panel descriptive subtitles ('LightGBM family decomposition, primary recipe' / 'LightGBM top features with fold SD') were removed as redundant with the caption, matching the same call made on Figure 4's panels; (3) both panels temporarily got a computed cumulative-share readout below the x-axis (panel A: top-3 families' combined share; panel B: what fraction of all considered features the shown top-16 cover).",
        "revision_2026-08-02f": "Per immediate user follow-up, removed four text elements entirely rather than keep them: the top-of-figure gate note ('Current explanations are weak-label...'), both cumulative-share readouts added in revision_2026-08-02e, and the bottom sources footer. Figure is now just the two bare panels with their axes and panel tags -- no supplementary fig-level text at all. Gridspec margins retightened (top 0.86->0.93, bottom 0.27->0.14) now that no top/bottom text needs headroom.",
        "revision_2026-08-02g": "Enlarged panel B's y-tick feature-name labels (7pt -> 10pt) plus its x-tick/x-label sizes, per direct user request that they were too small; matched panel A's tick/label sizes up for consistency (value-label text 7.8->9.5pt, y-tick 11pt, x-tick 9.5pt, x-label 11pt). Then, given a 'you decide, target is a Q1 journal' steer on further improvements: (1) panel A gained a fold-SD error whisker per family (`fam_fold.groupby('family')['importance'].agg(mean, std)`), closing a robustness gap -- A previously showed a bare point estimate per family with no uncertainty signal at all, while B has always carried this at the feature level; a same-session proposal to also add a Random Forest comparison series to A was set aside as working against this figure's established 'lean, LightGBM-focused' scope (round 2 of the 2026-08-02 cuts) rather than because it lacked merit. (2) panel B gained a dashed vertical reference line at the mean of its own 16 shown values, labeled inline, so a reader can tell which of the pre-filtered top-16 are only modestly above the pack versus genuinely standing out; a same-session proposal for family-colored row-background stripes was set aside as added visual noise without new information, given B is already fairly dense (16 rows + whiskers + legend). (3) Both panels' values converted from raw fractions (0.480) to percentages (48.0%) via `ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))` and matching text formatting, for faster reading and consistency between the two panels.",
        "png": str(png),
        "claim_boundary": "Impurity-importance tree explanation only. Permutation, SHAP, audited labels, and neural branch attribution remain required before deep/local-subtype claims.",
    }
    family_summary.to_csv(out_dir / "fig_6_family_importance.csv", index=False)
    (out_dir / "fig_6_explainability_multipanel_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png), "qa": str(out_dir / "fig_6_explainability_multipanel_qa.json")}


if __name__ == "__main__":
    print(json.dumps(build_figure_6_explainability(Path(".")), indent=2))
