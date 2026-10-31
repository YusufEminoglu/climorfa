"""Fidelity proof for climorfa_style.py (WP4.0).

Re-renders (1) Figure 9 panel (a) and (2) Figure 4 panel B using ONLY climorfa_style
(+ the original data), then compares them with crops of the stored original PNGs.

Run from the repo root:
    python -m scripts.street.figures.style_proof
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from scripts.street.figures import climorfa_style as S

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(".")
OUT = ROOT / "outputs/street/figures/style_proof"
OUT.mkdir(parents=True, exist_ok=True)
ORIG = ROOT / "paper/figures/main"


# --------------------------------------------------------------------------
def render_fig9_panel_a() -> tuple[Path, float, float]:
    df = pd.read_csv(ROOT / "data/03_processed/grid_250m_model_features_v8.csv")
    grid = gpd.read_file(ROOT / "data/03_processed/analysis_grids.gpkg", layer="grid_250m")
    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")

    P = S.PRESET_WIDE_13
    xmin, ymin, xmax, ymax = study.total_bounds
    map_ar = (ymax - ymin) / (xmax - xmin)
    panel_a_h = P.panel_w * map_ar * 0.98
    # vertical stack of the ORIGINAL figure (panel B trend row + tile strip below panel A)
    below_a = P.bottom_margin + S.TILE_STRIP["tile_h"] + S.TILE_STRIP["gap_trend_tiles"] + S.TILE_STRIP["trend_h"] + 0.05
    fig_w, fig_h = P.fig_w, below_a + panel_a_h + P.top_margin

    S.apply_style()
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    ax = S.inch_axes(fig, P.left_margin, below_a, P.panel_w, panel_a_h)
    ax.set_facecolor(S.MAP_BG)

    S.draw_land_cells(ax, grid)
    g = grid.merge(df[["grid_id", "lcz_weak_label", "lcz_weak_confidence", "lst_c_median_mean"]], on="grid_id", how="left")
    keep = g["lcz_weak_label"].isin(S.CLASSES) & (
        pd.to_numeric(g["lcz_weak_confidence"], errors="coerce") >= S.LCZ_RETAIN_CONFIDENCE)
    retained = g[keep].copy()
    vmin, vmax = S.lst_value_range(retained["lst_c_median_mean"])
    retained.plot(column="lst_c_median_mean", ax=ax, cmap=S.LST_TEAL_PURPLE_CMAP, vmin=vmin, vmax=vmax,
                  edgecolor="none", linewidth=0, zorder=2)
    S.draw_study_outline(ax, study, "ink")
    S.add_scale_north(ax, base_loc=(0.09, 0.59), scale_len_km=5.0, spec=S.SCALE_NORTH_FIG9)
    S.add_colorbar(fig, S.horizontal_cbar_bounds_in_water(P.left_margin, below_a, P.panel_w, panel_a_h, fig_w, fig_h),
                   S.LST_TEAL_PURPLE_CMAP, vmin, vmax, "Summer median Land Surface Temperature (°C)")
    S.info_card(ax, f"Izmir FUR Macro Thermal Geography\n{len(retained):,} retained 250m cells (2021–2025)\n"
                    f"Built LCZ Subset (3, 6, 8, 9)", x=0.02, y=0.49)
    S.set_map_extent(ax, tuple(study.total_bounds), exact=True)
    S.panel_label(ax, "A", fontsize=15.0, y=0.98, va="top")
    S.finalize_panel_labels(fig)
    return S.save_figure(fig, OUT / "fig9_panel_a_rerender.png"), below_a, fig_h


def render_fig4_panel_b() -> Path:
    df = pd.read_csv(ROOT / "data/03_processed/grid_250m_model_features_v8.csv")
    a = df[df["lcz_weak_label"].isin(S.CLASSES)
           & (pd.to_numeric(df["lcz_weak_confidence"], errors="coerce") >= 0.60)].copy()
    a["class"] = a["lcz_weak_label"].astype(int)
    fields = {
        "building_coverage_exact": "Coverage", "height_proxy_aw_mean_m": "Height proxy",
        "dsm_elevation_m_std": "Surface SD", "canopy_cover_gt2m_share": "Canopy >2m",
        "s2_ndvi_mean": "NDVI", "road_density_exact_m_per_km2": "Road density", "lst_c_median_mean": "LST",
    }
    rows = []
    for f, lab in fields.items():
        a[f] = pd.to_numeric(a[f], errors="coerce")
        rows.append(pd.DataFrame({"variable": lab, "value": S.scaled_1_99(a[f]), "class": a["class"]}))
    long = pd.concat(rows, ignore_index=True)

    S.apply_style()
    W, H = S.FIGSIZES["fig4_profiles"]
    fig = plt.figure(figsize=(W, H), constrained_layout=False)
    gs_b = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.650, bottom=0.488, wspace=0.08)  # fig4:430
    ax = fig.add_subplot(gs_b[0, 0:4])
    np.random.seed(0)  # the original is UNSEEDED (seaborn strip jitter); seed here only for reproducibility
    S.class_boxplot_panel(ax, long, list(fields.values()), letter="B")
    S.finalize_panel_labels(fig)
    # save only a horizontal band around panel B, at integer-pixel crop phase (like the stored PNG)
    top_px = (1 - 0.650) * H * 300 - 90
    bot_px = (1 - 0.488) * H * 300 + 84
    crop = plt.matplotlib.transforms.Bbox.from_extents(0, H - bot_px / 300.0, W, H - top_px / 300.0)
    path = OUT / "fig4_panel_b_rerender.png"
    fig.savefig(path, dpi=S.DPI, facecolor=S.FACECOLOR, bbox_inches=crop)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def mae_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def side_by_side(orig: np.ndarray, new: np.ndarray, path: Path, max_w: int = 2600) -> None:
    diff = np.abs(orig.astype(np.int16) - new.astype(np.int16)).max(axis=2)
    diff_vis = 255 - np.clip(diff * 4, 0, 255).astype(np.uint8)  # white = identical, dark = different (x4 gain)
    diff_rgb = np.stack([diff_vis] * 3, axis=2)
    h, w = orig.shape[:2]
    tiles = [Image.fromarray(x) for x in (orig, new, diff_rgb)]
    scale = min(1.0, max_w / (3 * w))
    tiles = [t.resize((int(w * scale), int(h * scale)), Image.LANCZOS) for t in tiles]
    gap = 12
    canvas = Image.new("RGB", (sum(t.width for t in tiles) + 2 * gap, tiles[0].height), (255, 0, 255))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0)); x += t.width + gap
    canvas.save(path)


def compare_fig9(rerender: Path, below_a: float, fig_h: float) -> dict:
    o = np.array(Image.open(ORIG / "fig_9_climate_validation.png").convert("RGB"))
    n = np.array(Image.open(rerender).convert("RGB"))
    res = {"orig_size": o.shape[:2][::-1], "new_size": n.shape[:2][::-1]}
    # the original and the re-render share the same top edge (figure top), compare panel A rows only
    # Compare everything above panel B's own '(b)' badge (which the single-panel re-render does not draw):
    # the trend row's top edge is below_a - 0.05 in from the bottom; the badge reaches ~0.37 in above it.
    rows = int((fig_h - (below_a - 0.05) - 0.37) * 300)
    rows = min(rows, o.shape[0], n.shape[0])
    h = min(o.shape[0], n.shape[0]); w = min(o.shape[1], n.shape[1])
    oc, nc = o[:rows, :w], n[:rows, :w]
    res["compared_region_px"] = [w, rows]
    res["mae_rgb_panel_a"] = mae_rgb(oc, nc)
    d = np.abs(oc.astype(np.int16) - nc.astype(np.int16)).max(axis=2)
    res["pct_pixels_any_channel_diff_gt_0"] = float((d > 0).mean() * 100)
    res["pct_pixels_diff_gt_16"] = float((d > 16).mean() * 100)
    res["pct_pixels_diff_gt_64"] = float((d > 64).mean() * 100)
    side_by_side(oc, nc, OUT / "compare_fig9_panel_a.png")
    return res


def compare_fig4(rerender: Path) -> dict:
    o = np.array(Image.open(ORIG / "fig_4_class_profiles.png").convert("RGB"))
    nc = np.array(Image.open(rerender).convert("RGB"))
    bh = nc.shape[0]
    w = min(o.shape[1], nc.shape[1])
    gray_o, gray_n = o.mean(axis=2), nc.mean(axis=2)
    # the stored PNG is a vertical crop of the full 15.5 in figure; locate panel B's band by search
    errs = {dy: np.abs(gray_o[dy:dy + bh, :w] - gray_n[:, :w]).mean() for dy in range(0, o.shape[0] - bh)}
    dy = min(errs, key=errs.get)
    oc = o[dy:dy + bh, :w]; nc = nc[:, :w]
    res = {"orig_size": o.shape[:2][::-1], "band_px": [w, bh], "y_offset_found_px": int(dy),
           "mae_rgb_panel_b": mae_rgb(oc, nc)}
    d = np.abs(oc.astype(np.int16) - nc.astype(np.int16)).max(axis=2)
    res["pct_pixels_any_channel_diff_gt_0"] = float((d > 0).mean() * 100)
    res["pct_pixels_diff_gt_16"] = float((d > 16).mean() * 100)
    res["pct_pixels_diff_gt_64"] = float((d > 64).mean() * 100)
    side_by_side(oc, nc, OUT / "compare_fig4_panel_b.png")
    Image.fromarray(oc).save(OUT / "fig4_panel_b_original_crop.png")
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p9, below_a, fig_h = render_fig9_panel_a()
    r9 = compare_fig9(p9, below_a, fig_h)
    p4 = render_fig4_panel_b()
    r4 = compare_fig4(p4)
    out = {"fig9_panel_a": r9, "fig4_panel_b": r4}
    (OUT / "fidelity_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
