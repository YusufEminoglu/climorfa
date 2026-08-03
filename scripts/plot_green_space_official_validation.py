"""Render the official-IZBB versus OSM district triangulation figure."""
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TOKENS = {
    "surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430",
    "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"}
NEUTRAL = {"mid": "#7A828F", "dark": "#464C55"}
FONT = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/diagnostics/green_space_official_validation_2026-06-19/district_comparison.csv")
    parser.add_argument("--summary", default="outputs/diagnostics/green_space_official_validation_2026-06-19/summary.json")
    parser.add_argument("--out-dir", default="outputs/diagnostics/green_space_official_validation_2026-06-19")
    return parser.parse_args()


def use_theme() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "grid.linewidth": 0.8,
        "font.family": "sans-serif", "font.sans-serif": FONT,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.965, textwrap.fill(title, 95), ha="left", va="top", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.07, 0.905, textwrap.fill(subtitle, 145), ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def panel(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str, corr_text: str, label_offsets: dict[str, tuple[int, int]]) -> None:
    x = np.log1p(frame[x_col].astype(float).to_numpy())
    y = np.log1p(frame[y_col].astype(float).to_numpy())
    sns.scatterplot(x=x, y=y, ax=ax, color=BLUE["base"], edgecolor=BLUE["dark"], linewidth=0.9, s=58, alpha=0.9)
    fit = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, fit[0] * x_fit + fit[1], color=NEUTRAL["dark"], linewidth=1.0, linestyle="--")
    for (_, row), px, py in zip(frame.iterrows(), x, y):
        district = str(row["district"])
        offset = label_offsets.get(district, (6, 6))
        ax.annotate(
            district, (px, py), xytext=offset, textcoords="offset points",
            fontsize=7.2, color=TOKENS["ink"],
            ha="right" if offset[0] < 0 else "left",
            arrowprops={"arrowstyle": "-", "color": TOKENS["axis"], "linewidth": 0.65, "shrinkA": 1, "shrinkB": 3},
        )
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=TOKENS["ink"], pad=12)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=TOKENS["muted"], bbox={"facecolor": TOKENS["panel"], "edgecolor": TOKENS["axis"], "pad": 4})
    ax.tick_params(labelsize=8, colors=TOKENS["muted"], length=0)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    summary = pd.read_json(args.summary, typ="series")
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=160)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.80, wspace=0.24)
    add_header(
        fig,
        "Official IZBB and OSM green-space inventories show a similar broad district pattern",
        "Twelve districts; official Park records and OSM park/garden polygons restricted to 112 official neighbourhood names matched to FUA population keys. Axes show log1p values; dashed lines are descriptive least-squares fits, not calibration lines.",
    )
    panel(
        axes[0], frame, "official_park_sites", "osm_polygon_count", "A. Park counts",
        "Official Park sites (log1p)", "OSM park/garden polygons (log1p)",
        f"Pearson r = {summary['district_log_count_correlation']:.3f}\nSpearman rho = {summary['district_rank_count_correlation']:.3f}",
        {"Balçova": (6, 10), "Güzelbahçe": (6, 7), "Çiğli": (8, -12), "Gaziemir": (8, -2),
         "Bayraklı": (8, -14), "Karşıyaka": (6, 10), "Bornova": (8, 8), "Buca": (8, -14),
         "Karabağlar": (8, -14), "Konak": (8, -14), "Narlıdere": (8, -14), "Menemen": (8, 8)},
    )
    panel(
        axes[1], frame, "official_park_area_m2", "osm_area_m2", "B. Reported/mapped area",
        "Official reported park area, m2 (log1p)", "OSM mapped park/garden area, m2 (log1p)",
        f"Pearson r = {summary['district_log_area_correlation']:.3f}\nSpearman rho = {summary['district_rank_area_correlation']:.3f}",
        {"Bayraklı": (-10, -18), "Karşıyaka": (-8, 12), "Bornova": (10, 10), "Buca": (8, -18),
         "Çiğli": (10, 8), "Karabağlar": (10, -18), "Konak": (8, -16), "Gaziemir": (8, -16),
         "Balçova": (8, 10), "Güzelbahçe": (8, 8), "Narlıdere": (8, -16), "Menemen": (8, 8)},
    )
    fig.text(
        0.07, 0.055,
        "Source: IZBB Open Data north/south park-maintenance inventory (metadata modified 2023-04-26) and CLIMORFA OSM 2SFCA supply audit. The official inventory is un-geocoded and branch-limited; this figure supports macro triangulation, not object-level completeness or public-access validation.",
        ha="left", va="bottom", fontsize=8.2, color=TOKENS["muted"], wrap=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "district_official_osm_triangulation.png"
    svg = out_dir / "district_official_osm_triangulation.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=TOKENS["surface"])
    fig.savefig(svg, bbox_inches="tight", facecolor=TOKENS["surface"])
    plt.close(fig)
    (out_dir / "chart_map.md").write_text(
        "# Chart Map\n\n- Section: external green-space source audit\n- Question: do official and OSM inventories share district-level spatial pattern?\n- Family: Relationship; two-panel labelled scatter\n- Grain: 12 districts\n- Fields: official/OSM park counts and areas, FUA-neighbourhood-matched scope\n- Palette: single blue root plus neutral descriptive fit\n- Outputs: district_official_osm_triangulation.png and .svg\n- Guardrail: triangulation only; no object-level completeness or public-access claim\n",
        encoding="utf-8",
    )
    print(png)
    print(svg)


if __name__ == "__main__":
    main()

