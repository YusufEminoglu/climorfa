"""Build the audited-validation figure: weak-label and primary-model agreement
against Mert's completed manual audit, for the exact four-class modeling
population (lcz_weak_label in {3,6,8,9}, confidence >= 0.60)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_figure_1_texture_atlas import PALETTE

CLASS_ORDER = [3, 6, 8, 9]
CLASS_COLORS = {3: PALETTE["teal"], 6: PALETTE["pink"], 8: PALETTE["orange"], 9: PALETTE["taupe"]}
AUDIT_COL_ORDER = [
    "1_compact_highrise", "2_compact_midrise", "3_compact_lowrise", "4_open_highrise",
    "5_open_midrise", "6_open_lowrise", "7_lightweight_lowrise", "8_large_lowrise",
    "9_sparsely_built", "10_heavy_industry",
    "A_dense_trees", "B_scattered_trees", "C_bush_scrub", "D_low_plants",
    "E_bare_rock_paved", "F_bare_soil_sand", "G_water", "mixed",
]
AUDIT_COL_SHORT = {
    "1_compact_highrise": "1 cmp-hi", "2_compact_midrise": "2 cmp-mid", "3_compact_lowrise": "3 cmp-lo",
    "4_open_highrise": "4 opn-hi", "5_open_midrise": "5 opn-mid", "6_open_lowrise": "6 opn-lo",
    "7_lightweight_lowrise": "7 lgt-lo", "8_large_lowrise": "8 lrg-lo", "9_sparsely_built": "9 sparse",
    "10_heavy_industry": "10 indus", "A_dense_trees": "A trees", "B_scattered_trees": "B scat.trees",
    "C_bush_scrub": "C scrub", "D_low_plants": "D plants", "E_bare_rock_paved": "E paved",
    "F_bare_soil_sand": "F soil", "G_water": "G water", "mixed": "mixed",
}
TIER_LABELS = {"primary_high": "High quality\n(primary)", "sensitivity_high_medium": "High + medium\n(sensitivity)", "all_quality": "All quality\n(reference)"}
TIER_ORDER = ["primary_high", "sensitivity_high_medium", "all_quality"]


def _panel_label(
    ax: plt.Axes,
    label: str,
    fontsize: float = 15.0,
    x: float = 0.005,
    y: float = 1.02,
    va: str = "bottom",
) -> None:
    """Panel label badge function matching Figure 2/7/9 standard."""
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
    for ax in fig.axes:
        for child in ax.texts:
            text = child.get_text()
            if re.fullmatch(r"[A-Za-z]", text):
                child.set_text(f"({text.lower()})")


def _confusion_heatmap(ax: plt.Axes, cm_path: Path, title: str, label: str, color: str) -> None:
    cm = pd.read_csv(cm_path, index_col=0)
    cm.index = [int(float(i)) for i in cm.index]
    cols = [c for c in AUDIT_COL_ORDER if c in cm.columns]
    cm = cm.reindex(index=CLASS_ORDER, columns=cols).fillna(0).astype(int)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("audit_cm", [PALETTE["taupe_light"], color])
    im = ax.imshow(cm.values, cmap=cmap, aspect="auto", vmin=0)
    
    # 1. Annotate cell counts and highlight exact-match diagonal cells
    for i, c_row in enumerate(CLASS_ORDER):
        for j, c_col in enumerate(cols):
            v = cm.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["ink"] if v < cm.values.max() * 0.6 else "white")
            
            # Check if column matches row LCZ exact class (e.g. LCZ 3 vs 3_compact_lowrise)
            is_exact = str(c_row) in c_col.split("_")[0]
            if is_exact:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor=color, linewidth=1.5, linestyle="-", zorder=10
                )
                ax.add_patch(rect)

    # 2. Draw vertical divider between Built Fabric (1-10) and Natural/Open (A-F)
    built_cols = [c for c in cols if c[0].isdigit()]
    if 0 < len(built_cols) < len(cols):
        split_idx = len(built_cols)
        ax.axvline(split_idx - 0.5, color=PALETTE["ink"], linestyle="--", linewidth=1.0, alpha=0.65, zorder=12)
        ax.text(split_idx / 2.0 - 0.5, -0.68, "BUILT FABRIC (1–10)", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=PALETTE["ink"])
        ax.text((split_idx + len(cols)) / 2.0 - 0.5, -0.68, "NATURAL / OPEN (A–F)", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=PALETTE["ink"])

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([AUDIT_COL_SHORT.get(c, c) for c in cols], rotation=55, ha="right", fontsize=7.2)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels([f"LCZ {c}" for c in CLASS_ORDER], fontsize=8.5, fontweight="bold")
    for i, c in enumerate(CLASS_ORDER):
        ax.get_yticklabels()[i].set_color(CLASS_COLORS[c])
    ax.set_xlabel("Audited ground-truth label", fontsize=9.0, fontweight="bold", labelpad=4.0)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=24)
    ax.set_ylim(len(CLASS_ORDER) - 0.5, -0.5)
    _panel_label(ax, label, x=-0.03, y=1.06)


def _agreement_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    s = summary.set_index("tier").loc[TIER_ORDER]
    x = np.arange(len(TIER_ORDER))
    width = 0.25
    series = [
        ("weak_label_agreement", "Weak label", PALETTE["taupe_dark"]),
        ("lightgbm_agreement", "LightGBM (Primary)", PALETTE["teal_dark"]),
        ("random_forest_agreement", "Random Forest (Baseline)", PALETTE["pink_dark"]),
    ]
    ax.set_facecolor("#ffffff")
    for i, (col, name, color) in enumerate(series):
        vals = 100 * s[col].to_numpy()
        bars = ax.bar(x + (i - 1) * width, vals, width=width, color=color, label=name, edgecolor="white", linewidth=0.6, zorder=3)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2, color=color, fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TIER_LABELS[t]}\n(n={int(s.loc[t, 'n'])})" for t in TIER_ORDER], fontsize=8.0)
    ax.set_ylabel("Exact-class agreement with audit (%)", fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, max(18, 100 * s[[c for c, _, _ in series]].to_numpy().max() * 1.35))
    ax.legend(fontsize=7.8, frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.grid(axis="y", color="#e5e5e5", linestyle="--", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.set_title("Agreement by audit-quality tier", fontsize=10.5, fontweight="bold", pad=24)
    _panel_label(ax, "C", x=-0.03, y=1.06)


def build_figure_11_audited_validation(root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(root)
    fig_dir = root / "paper/figures/main"
    audit_dir = root / "outputs/diagnostics/audited_model_validation_2026-07-31"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(audit_dir / "audited_validation_summary.csv")

    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    # Panel titles + letter tags (pad=24pt title offset plus the label's own
    # y=1.06 axes-fraction offset) were using up the entire nominal top
    # margin, leaving 0 cm of actual clearance above them. Rather than
    # shrinking the panels to reclaim margin, the canvas itself grows by
    # exactly 0.2 cm and the gridspec's top/bottom are recomputed from the
    # ORIGINAL absolute inch positions so the panels keep their exact prior
    # size -- only the blank margin above them grows.
    base_fig_h = 5.2
    orig_top_in = 0.91 * base_fig_h
    orig_bottom_in = 0.22 * base_fig_h
    fig_h = base_fig_h + 0.2 / 2.54
    fig = plt.figure(figsize=(15.0, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.985, top=orig_top_in / fig_h, bottom=orig_bottom_in / fig_h, wspace=0.18, width_ratios=[1.15, 1.15, 1.0])
    ax_weak = fig.add_subplot(gs[0, 0])
    ax_model = fig.add_subplot(gs[0, 1])
    ax_bars = fig.add_subplot(gs[0, 2])

    _confusion_heatmap(ax_weak, audit_dir / "confusion_weak_vs_audit_primary_high.csv", "Weak label vs. audit (n=96)", "A", PALETTE["taupe_dark"])
    _confusion_heatmap(ax_model, audit_dir / "confusion_lightgbm_vs_audit_primary_high.csv", "LightGBM prediction vs. audit (n=96)", "B", PALETTE["teal_dark"])
    _agreement_bars(ax_bars, summary)

    _finalize_panel_labels(fig)

    png = fig_dir / "fig_11_audited_validation.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)

    qa = {
        "figure": "fig_11_audited_validation",
        "date": "2026-07-31",
        "panel_count": 3,
        "source_paths": [
            str(audit_dir / "audited_validation_summary.csv"),
            str(audit_dir / "confusion_weak_vs_audit_primary_high.csv"),
            str(audit_dir / "confusion_lightgbm_vs_audit_primary_high.csv"),
        ],
        "png": str(png),
        "claim_boundary": "n=96 at the primary tier; agreement shares carry wide sampling uncertainty and are reported descriptively, not as precise population parameters.",
    }
    (audit_dir / "fig_11_audited_validation_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"png": str(png)}


if __name__ == "__main__":
    print(json.dumps(build_figure_11_audited_validation(Path(".")), indent=2))
