"""CLIMORFA v2 Figure 4: city-wide street-level UTCI at 15:00 on the two design days (WP4.4).

Run from the repo root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig4_street_utci
        -> variant C (anomaly, the Fig 4 adopted by the author on 2026-10-30): paper/figures/v2/main/fig_4_street_utci.png
    ... build_fig4_street_utci --variant absolute
        -> variant A (option): paper/figures/v2/main/_options/fig_4_street_utci_absolute.png
    (add ``--qa <dir>`` to also write the QA json into <dir>; nothing else is written)

Composition language: original Fig 9 (12 in panel column in a 13.0 in sheet, inside boxed letters, ink FUA outline,
taupe_light land, white sea, Fig 9 scale bar / north arrow, compact tile strip with coloured tag + bottom read-out,
standalone horizontal colourbar).  Everything stylistic comes from ``climorfa_style`` (apply_style first); all text is
authored at the Fig 9 point sizes and then enlarged once by ``climorfa_style.scale_text`` (v2 type scale, x1.4 with a
6 pt-printed floor; author decision 2026-10-30).  Non-text furniture keeps the Fig 9 physical size.

Variants (same layout)
  A  'absolute' -> paper/figures/v2/main/_options/fig_4_street_utci_absolute.png (option)
     M1 = segment median of street-wind UTCI at label 15 (hour ending 15:00 local, i.e. 14:00-15:00), one FIXED scale
     UTCI_LO..UTCI_HI for both days, canonical LST teal-purple ramp, Broede et al. (2012) category bounds on the bar.
  C  'anomaly'  -> paper/figures/v2/main/fig_4_street_utci.png (primary, default)
     within-day anomaly dUTCI = value - that day's median over all coloured segments (segments and close-up cells
     against the same day median), one shared diverging scale +-ANOM_LIM, teal -> #F7F7F7 -> purple built from the
     LST ramp's end colours.  The map tags keep the absolute day medians, so the day difference stays explicit.
Panels: (a) IMBAT 2022-07-05, (b) CALM 2023-07-26 (same extent / scale); (c) three 1 km close-ups x two days at 2 m,
UTCI on pedestrian-space (PS) cells only.

Partial runs: segments of tiles not yet extracted are drawn pale grey under the coloured ones, and a grey
'DRAFT - partial run, n/427 tiles' note is stamped bottom-right.  Both disappear automatically once
n_tiles_at_extraction == 427 in both metric tables, so one re-run on the final files gives the final figure.
Double-anonymised: no names, no URLs in the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.collections import LineCollection

from scripts.street.figures import climorfa_style as S

ROOT = Path(__file__).resolve().parents[3]
OUT = {  # author decision 2026-10-30: variant C (anomaly) is Fig 4; A (absolute) is kept as an option
    "anomaly": ROOT / "paper/figures/v2/main/fig_4_street_utci.png",
    "absolute": ROOT / "paper/figures/v2/main/_options/fig_4_street_utci_absolute.png",
}
DEFAULT_VARIANT = "anomaly"
P = S.PALETTE

N_TILES_TOTAL = 427
BAND_COL = "utci_b15"          # label 15 = hour ending 15:00 local (14:00-15:00), methods.tex 'Metrics'
RASTER_BAND = 10               # 1-based band of utci_street/<tile>.tif: bands 06..21 -> 15:00 is band 10 ('hour_15_1500LST')
DAYS = [  # key, date, letter
    ("IMBAT", "2022-07-05", "A"),
    ("CALM", "2023-07-26", "B"),
]
TILES = [  # same three close-ups as Fig 1
    ("T_0512_4255", "Alsancak"),
    ("T_0508_4248", "Karaba\u011flar"),
    ("T_0503_4262", "\u00c7i\u011fli AOSB"),
]
DAY_TAG_COLOR = {"IMBAT": P["teal_dark"], "CALM": P["pink_dark"]}  # Fig 9 cool / hot tag colours (fig9:518)

# ---- colour scales --------------------------------------------------------------------------
# A: fixed for both days.  32 degC = lower bound of 'strong heat stress' (Broede et al. 2012), 50 degC = first round
# number above the hottest value of either day; ticks every 2 degC put the category bounds 32 / 38 / 46 on a tick.
UTCI_LO, UTCI_HI = 32.0, 50.0
CATS = [(32.0, 38.0, "strong"), (38.0, 46.0, "very strong"), (46.0, 50.0, "extreme")]  # Broede et al. 2012
VSTRONG, EXTREME = 38.0, 46.0
CMAP_ABS = S.LST_TEAL_PURPLE_CMAP
# C: +-2 degC.  The cool tail is a separate cluster at -4..-7 degC (shaded segments / cells), so +-2 and +-3 saturate
# the same cool set (segments beyond -2: 3.4 / 4.1 %, beyond -3: 3.3 / 4.0 %, IMBAT / CALM), while the hot side ends at
# P99 = +1.3 / +0.6 degC; +-2 therefore spends more of the ramp where the street texture is.
ANOM_LIM = 2.0
ANOM_MID = "#F7F7F7"
CMAP_ANOM = mpl.colors.LinearSegmentedColormap.from_list(
    "anom_teal_white_purple", [S.LST_TEAL_PURPLE_COLORS[0], ANOM_MID, S.LST_TEAL_PURPLE_COLORS[-1]])

# ---- sheet geometry (inches) : Fig 9 preset ------------------------------------------------
PR = S.PRESET_WIDE_13                    # 0.50 + 12.0 + 0.50 = 13.0 in (= S.V2_FIG_WIDTH_IN)
MAP_GAP = 0.12                           # Fig 9 tile gap
PAIR_GAP_IN, PAIR_GAP_OUT = 0.10, 0.30   # within a day pair / between places
TAG_BAND = 0.36                          # computed tag strip under each map
GAP_TAG_TILES = 0.62                     # (c) letter + place headers above the strip
GAP_TILES_CBAR = 0.50                    # category labels above the bar
CBAR_W, CBAR_H = 4.80, 0.12              # bar height = Fig 9 (0.015 x 8.05 in)
CBAR_BLOCK = 0.80                        # bar + tick labels + label (after scale_text)
SEG_LW = 0.45                            # pt, base street width (v1 tuning)
LW_SCALE = 1.3                           # print-legibility factor chosen on 1500 px downsamples (1.0 / 1.3 / 1.6 tested)
# tiles not yet run (draft only): one light grey, no casing, clearly apart from the casing tone of computed streets
PENDING_COLOR = "#ECEAE7"
# C only: a faint neutral casing under every computed segment, so streets near the day median (drawn ~#F7F7F7, the
# diverging midpoint) still show the network on taupe_light land; anomalies are drawn on top of it.
CASING_COLOR = "#D9D4CF"
CASING_EXTRA = 0.30                      # pt: casing = street width + 0.30 (0.75 pt at LW_SCALE 1.0, as reviewed)


def sgn(x: float, nd: int = 1) -> str:
    """Signed number with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", "−")
SCALE_LOC = (0.02, 0.42)    # Fig 9 logic: open gulf water west of the bay (now free: tags moved below the maps)
SCALE_KM = 10.0             # 10 km = 1.20 in, the physical length of Fig 9's 5 km bar (1.22 in)


# =============================================================================================
# data
# =============================================================================================
def load_data():
    segs = gpd.read_file(ROOT / "data/street/analysis/segments.gpkg", layer="segments",
                         columns=["seg_id", "tile_id"])
    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    land = gpd.read_file(ROOT / "data/03_processed/analysis_grids.gpkg", layer="grid_250m", columns=["grid_id"])
    land = gpd.GeoDataFrame(geometry=[shapely.union_all(land.geometry.values)], crs=land.crs)  # Fig 9 land, no seams
    tiles = gpd.read_file(ROOT / "data/street/grid/tiles_core.gpkg").set_index("tile_id")
    mets = {}
    for day, _, _ in DAYS:
        mets[day] = pd.read_parquet(ROOT / f"data/street/analysis/seg_metrics_{day}.parquet",
                                    columns=["seg_id", BAND_COL, "n_tiles_at_extraction", "partial_run"])
    return segs, study, land, tiles, mets


def day_stats(m: pd.DataFrame) -> dict:
    u = m[BAND_COL].dropna()
    med = float(u.median())
    a = u - med
    return dict(
        n=int(len(u)), min=float(u.min()), max=float(u.max()), median=med,
        p10=float(u.quantile(0.10)), p90=float(u.quantile(0.90)),
        share_ge38=float((u >= VSTRONG).mean()), share_ge46=float((u >= EXTREME).mean()),
        share_below_lo=float((u < UTCI_LO).mean()), share_above_hi=float((u > UTCI_HI).mean()),
        anom_p1=float(a.quantile(0.01)), anom_p99=float(a.quantile(0.99)),
        anom_share_below=float((a < -ANOM_LIM).mean()), anom_share_above=float((a > ANOM_LIM).mean()),
        n_tiles=int(m["n_tiles_at_extraction"].max()),
    )


def line_arrays(geoms) -> list[np.ndarray]:
    coords, idx = shapely.get_coordinates(np.asarray(geoms), return_index=True)
    cuts = np.flatnonzero(np.diff(idx)) + 1
    return np.split(coords, cuts)


def tile_layers(tid: str, day: str):
    """UTCI on PS cells (masked array or None if the tile is not finished) and the building mask; inner 1 km at 2 m."""
    bdir = ROOT / "data/street/tiles/2m"
    dsm = rasterio.open(bdir / "building_dsm" / f"{tid}.tif").read(1)
    dem = rasterio.open(bdir / "dem" / f"{tid}.tif").read(1)
    sl = slice(50, 550)                    # inner 1000 m of the 600 x 600 buffered tile (grid_spec)
    h = (dsm - dem)[sl, sl]
    bld = np.isfinite(h) & (h > 0.5)
    ps = rasterio.open(ROOT / "data/street/analysis/ps_index" / f"{tid}.tif").read(1) > 0
    p = ROOT / f"data/street/utci/{day}/utci_street/{tid}.tif"
    if not p.exists():
        return None, bld
    with rasterio.open(p) as src:
        assert src.count == 16 and "15" in (src.descriptions[RASTER_BAND - 1] or "hour_15"), src.descriptions
        u = src.read(RASTER_BAND).astype(float)
        nod = src.nodata
    valid = ps & np.isfinite(u) & (u != nod)
    return np.ma.masked_where(~valid, u), bld


# =============================================================================================
# furniture
# =============================================================================================
def add_scale_north_fig9_size(ax, base_loc, scale_len_km=5.0, spec=S.SCALE_NORTH_FIG9):
    """``climorfa_style.add_scale_north`` with the bar height, arrow size and offsets held at their PHYSICAL size in
    Fig 9 (a) instead of a fixed fraction of the axes.  Fig 9's map axes are 8.05 in tall (12 in panel, fig9:592),
    ours ~4.07 in, so the axes-fraction version would draw a half-size arrow.
    Geometry otherwise identical to fig9:114-195 (2 segments, lw 0.7, #2C3E50 arrow with white 0.9 pt edge)."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    h_in, w_in = bb.height / fig.dpi, bb.width / fig.dpi
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    ref_h = PR.panel_w * (dy / dx) * 0.98            # Fig 9 panel_a_h (fig9:592), same FUA bounds
    ref_w = ref_h * dx / dy                          # drawn width after set_aspect('equal')
    ky, kx = ref_h / h_in, ref_w / w_in              # physical-size factors (~1.98)
    sb_x0 = xlim[0] + base_loc[0] * dx
    sb_y0 = ylim[0] + base_loc[1] * dy
    slen = scale_len_km * 1000.0
    sb_h = dy * 0.009 * ky
    for i in range(2):
        seg_x = sb_x0 + i * (slen / 2.0)
        ax.add_patch(mpatches.Rectangle((seg_x, sb_y0), slen / 2.0, sb_h, facecolor="black" if i == 0 else "white",
                                        edgecolor="black", linewidth=0.7, zorder=25))
        ax.text(seg_x, sb_y0 + sb_h * spec.label_k, f"{i * scale_len_km / 2.0:g}", ha="center", va="bottom",
                fontsize=spec.label_fs, fontweight="bold", color=S.PANEL_TEXT, zorder=25)
    ax.text(sb_x0 + slen, sb_y0 + sb_h * spec.label_k, f"{int(scale_len_km)} km", ha="center", va="bottom",
            fontsize=spec.label_fs, fontweight="bold", color=S.PANEL_TEXT, zorder=25)
    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * spec.arrow_base_k
    na_w, na_h = dx * spec.arrow_w_frac * kx, dy * 0.040 * ky
    ax.add_patch(mpatches.Polygon(
        [[cx, na_base + na_h], [cx + na_w / 2.0, na_base], [cx, na_base + na_h * 0.25], [cx - na_w / 2.0, na_base]],
        facecolor=S.NORTH_ARROW, edgecolor="white", linewidth=0.9, zorder=25))
    ax.text(cx, na_base + na_h + dy * 0.006 * ky, "N", ha="center", va="bottom", fontsize=spec.north_fs,
            fontweight="bold", color=S.NORTH_ARROW, zorder=25)


def tag_box(ax, text, x, y, va="top"):
    """Computed tag of a map: Fig 9 badge idiom (7.2 pt bold ink, white, taupe edge; fig9:375-381)."""
    return ax.text(x, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=7.2, fontweight="bold",
                   color=P["ink"], linespacing=1.35, clip_on=False,
                   bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.30", "alpha": 0.95},
                   zorder=26)


def add_tile_scale(ax, x0, y0, length=200.0):
    """200 m two-segment bar (add_scale_north geometry) at the lower right, as the Fig 1 close-ups."""
    sb_h = 1000.0 * 0.012
    sx = x0 + 1000.0 - 60.0 - length
    sy = y0 + 45.0
    ax.add_patch(mpatches.Rectangle((sx - 30, sy - 25), length + 60, 175, facecolor="white", alpha=0.85,
                                    edgecolor="none", zorder=24))
    for i in range(2):
        ax.add_patch(mpatches.Rectangle((sx + i * length / 2, sy), length / 2, sb_h,
                                        facecolor="black" if i == 0 else "white", edgecolor="black", linewidth=0.5,
                                        zorder=25))
    ax.text(sx + length / 2, sy + sb_h * 1.8, f"{length:g} m", ha="center", va="bottom", fontsize=6.2,
            fontweight="bold", color=S.PANEL_TEXT, zorder=25)


# =============================================================================================
# panels
# =============================================================================================
def draw_map(ax, day, date, segs_xy, seg_ids, met, study, land, cmap, norm, st, extent, scalebar, tiles, variant,
             ref_median, lw):
    S.draw_land_cells(ax, land)                                       # taupe_light, sea = white page (fig9:241)
    u = pd.Series(met.set_index("seg_id")[BAND_COL])
    vals = u.reindex(seg_ids).to_numpy()
    have = np.isfinite(vals)
    if (~have).any():   # tiles not yet extracted: pale grey, under the coloured streets
        ax.add_collection(LineCollection([segs_xy[i] for i in np.flatnonzero(~have)], colors=PENDING_COLOR,
                                         linewidths=lw * 0.75, capstyle="round", zorder=3))
    if variant == "anomaly":
        ax.add_collection(LineCollection([segs_xy[i] for i in np.flatnonzero(have)], colors=CASING_COLOR,
                                         linewidths=lw + CASING_EXTRA, capstyle="round", zorder=3.5))
        vals = vals - st["median"]
        key = np.abs(vals[have])                                      # largest anomalies (either sign) on top
    else:
        key = vals[have]                                              # ascending: hottest streets on top
    order = np.flatnonzero(have)[np.argsort(key, kind="stable")]
    ax.add_collection(LineCollection([segs_xy[i] for i in order], array=vals[order], cmap=cmap, norm=norm,
                                     linewidths=lw, capstyle="round", zorder=4))
    S.draw_study_outline(ax, study, "ink")                            # ink lw 0.85 zorder 15 (fig9:263)
    S.set_map_extent(ax, extent, exact=True)                          # fig9:198-203
    if scalebar:
        add_scale_north_fig9_size(ax, base_loc=SCALE_LOC, scale_len_km=SCALE_KM)
        for k, (tid, _) in enumerate(TILES):                          # locator squares of (c)
            x0, y0 = float(tiles.loc[tid, "x0"]), float(tiles.loc[tid, "y0"])
            sq = mpatches.Rectangle((x0, y0), 1000, 1000, fill=False, edgecolor="black", linewidth=1.0, zorder=20)
            sq.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
            ax.add_patch(sq)
            left = tid == "T_0512_4255"   # Alsancak is on the shore: label into the gulf, clear of the FUA outline
            ax.text(x0 - 450 if left else x0 + 1000 + 450, y0 + 1000 + 250, f"c.{k + 1}",
                    ha="right" if left else "left", va="bottom", fontsize=7.2,
                    fontweight="bold", color=S.PANEL_TEXT, zorder=21,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    zero = " (\u0394 = 0)" if variant == "anomaly" else ""
    line1 = f"{day} \u00b7 median {st['median']:.1f}\u00b0C"
    pad_in = 0.30 * S.v2_fs(7.2) / 72.0                                # box pad at the final (scaled) size
    ax_w_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    tag_box(ax, line1, x=pad_in / ax_w_in, y=-0.03, va="top")   # box edge flush with the map's left       # below the map (no room in the gulf at v2 type)


def draw_tile(ax, tid, day, tiles, cmap, norm, variant, day_median, scalebar=False):
    x0, y0 = float(tiles.loc[tid, "x0"]), float(tiles.loc[tid, "y0"])
    ext = [x0, x0 + 1000.0, y0, y0 + 1000.0]
    u, bld = tile_layers(tid, day)
    ax.set_facecolor("white")
    st = dict(tile_id=tid, day=day, finished=u is not None)
    if u is not None:
        img = u - day_median if variant == "anomaly" else u
        ax.imshow(img, cmap=cmap, norm=norm, extent=ext, interpolation="nearest", zorder=2)
        c = u.compressed()
        st.update(ps_median=float(np.median(c)), ps_p10=float(np.percentile(c, 10)),
                  ps_p90=float(np.percentile(c, 90)), n_ps=int(c.size),
                  anom_median=float(np.median(c) - day_median),
                  share_below_anom=float((c - day_median < -ANOM_LIM).mean()),
                  share_above_anom=float((c - day_median > ANOM_LIM).mean()),
                  share_below_lo=float((c < UTCI_LO).mean()), share_above_hi=float((c > UTCI_HI).mean()))
    else:   # tile not finished yet: hatched placeholder
        ax.add_patch(mpatches.Rectangle((x0, y0), 1000, 1000, facecolor="white", edgecolor=P["taupe"],
                                        hatch="////", linewidth=0, zorder=1))
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=P["muted"],
                bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.20", "alpha": 0.95},
                zorder=30)
    ax.imshow(np.ma.masked_where(~bld, np.ones_like(bld, dtype=float)),
              cmap=mpl.colors.ListedColormap([P["taupe"]]), vmin=0, vmax=1, extent=ext, interpolation="nearest",
              zorder=3)
    xs = np.linspace(x0 + 1.0, x0 + 999.0, bld.shape[1])
    ys = np.linspace(y0 + 999.0, y0 + 1.0, bld.shape[0])
    ax.contour(xs, ys, bld.astype(float), levels=[0.5], colors=S.PANEL_EDGE, linewidths=0.15, zorder=4)
    ax.add_patch(mpatches.Rectangle((x0, y0), 1000.0, 1000.0, fill=False, edgecolor=P["ink"], linewidth=0.9,
                                    zorder=6))                        # fig9:511 cell outline
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.axis("off")
    S.tile_tag(ax, day, DAY_TAG_COLOR[day], x=0.04, y=0.96)          # fig9:520-532
    if u is not None:
        txt = (f"UTCI {st['ps_median']:.1f}\u00b0C" if variant == "absolute"
               else f"UTCI {st['ps_median']:.1f}\u00b0C\n\u0394 {sgn(st['anom_median'])}\u00b0C")
        S.tile_readout(ax, txt, x=0.04, y=0.04)                       # fig9:533-545
    if scalebar:
        add_tile_scale(ax, x0, y0)
    return st


def draw_colorbar(fig, bounds, variant, cmap, norm, any_below, any_above):
    cax = fig.add_axes(bounds)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    extend = "neither"
    if variant == "anomaly":   # clipped values exist: say so with end triangles (C only; A clips < 0.01 % of segments)
        extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((any_below, any_above), "neither")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend, extendfrac=0.04)
    cb.ax.tick_params(labelsize=8.0)                                                         # fig9:274
    cb.ax.grid(False)
    trans = cax.get_xaxis_transform()
    if variant == "absolute":
        cb.set_label("Street-level UTCI at 15:00 on pedestrian space (\u00b0C)", fontsize=8.5, fontweight="bold",
                     color=P["ink"], labelpad=3.0)                                           # fig9:275
        ticks = np.arange(UTCI_LO, UTCI_HI + 0.1, 2.0)
        cb.set_ticks(ticks); cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
        for thr in (VSTRONG, EXTREME):   # heat-stress category bounds (Broede et al. 2012)
            cax.plot([thr, thr], [0.0, 1.55], transform=trans, color=P["ink"], linewidth=0.8, clip_on=False,
                     zorder=5, solid_capstyle="butt")
        for lo, hi, name in CATS:
            cax.text((lo + hi) / 2.0, 1.55, name, transform=trans, ha="center", va="bottom", fontsize=7.5,
                     fontweight="bold", color=P["ink"], clip_on=False)
    else:
        cb.set_label("\u0394UTCI at 15:00 vs the day\u2019s median segment (\u00b0C)", fontsize=8.5,
                     fontweight="bold", color=P["ink"], labelpad=3.0)
        ticks = np.arange(-ANOM_LIM, ANOM_LIM + 0.01, 1.0)
        cb.set_ticks(ticks)
        cb.ax.set_xticklabels([("\u2212" if t < 0 else "+" if t > 0 else "") + f"{abs(t):g}" for t in ticks])
        cax.text(-ANOM_LIM, 1.55, "cooler than the day median", transform=trans, ha="left", va="bottom",
                 fontsize=7.5, fontweight="bold", color=P["teal_dark"], clip_on=False)
        cax.text(ANOM_LIM, 1.55, "hotter", transform=trans, ha="right", va="bottom", fontsize=7.5,
                 fontweight="bold", color=S.LST_TEAL_PURPLE_COLORS[-1], clip_on=False)
    return cb


# =============================================================================================
def build(variant: str = DEFAULT_VARIANT, data=None, lw_scale: float = LW_SCALE):
    assert variant in OUT
    S.apply_style()
    np.random.seed(20261030)
    segs, study, land, tiles, mets = data if data is not None else load_data()
    stats = {day: day_stats(mets[day]) for day, _, _ in DAYS}
    n_tiles = min(s["n_tiles"] for s in stats.values())
    partial = n_tiles < N_TILES_TOTAL or any(bool(mets[d]["partial_run"].any()) for d in mets)
    if variant == "anomaly":
        cmap, norm = CMAP_ANOM, mpl.colors.Normalize(-ANOM_LIM, ANOM_LIM)
    else:
        cmap, norm = CMAP_ABS, mpl.colors.Normalize(UTCI_LO, UTCI_HI)

    extent = tuple(study.total_bounds)
    map_ar = (extent[3] - extent[1]) / (extent[2] - extent[0])
    MAP_W = (PR.panel_w - MAP_GAP) / 2.0
    MAP_H = MAP_W * map_ar
    TILE = (PR.panel_w - 3 * PAIR_GAP_IN - 2 * PAIR_GAP_OUT) / 6.0
    FW = PR.fig_w
    FH = (PR.bottom_margin + CBAR_BLOCK + GAP_TILES_CBAR + TILE + GAP_TAG_TILES + TAG_BAND + MAP_H
          + PR.top_margin)
    fig = plt.figure(figsize=(FW, FH))
    y_map = FH - PR.top_margin - MAP_H
    y_tile = y_map - TAG_BAND - GAP_TAG_TILES - TILE
    y_cbar = y_tile - GAP_TILES_CBAR - CBAR_H

    segs_xy = line_arrays(segs.geometry.values)
    seg_ids = segs["seg_id"].to_numpy()
    map_axes = {}
    for j, (day, date, letter) in enumerate(DAYS):
        ax = S.inch_axes(fig, PR.left_margin + j * (MAP_W + MAP_GAP), y_map, MAP_W, MAP_H)
        draw_map(ax, day, date, segs_xy, seg_ids, mets[day], study, land, cmap, norm, stats[day], extent,
                 scalebar=(j == 0), tiles=tiles, variant=variant,
                 ref_median=None if j == 0 else stats[DAYS[0][0]]["median"], lw=SEG_LW * lw_scale)
        S.panel_label(ax, letter, fontsize=15.0, y=0.98, va="top")    # fig9:295
        map_axes[day] = ax

    tile_stats, tile_axes, headers = [], [], []
    x = PR.left_margin
    for k, (tid, place) in enumerate(TILES):
        x_pair = x
        for j, (day, _, _) in enumerate(DAYS):
            axt = S.inch_axes(fig, x, y_tile, TILE, TILE)
            st = draw_tile(axt, tid, day, tiles, cmap, norm, variant, stats[day]["median"],
                           scalebar=(k == len(TILES) - 1 and j == 1))
            st["place"] = place
            tile_stats.append(st)
            tile_axes.append(axt)
            x += TILE + (PAIR_GAP_IN if j == 0 else PAIR_GAP_OUT)
        # place header centred over the day pair (links to the c.k locator squares in (a))
        cx = x_pair + TILE + PAIR_GAP_IN / 2.0
        headers.append(fig.text(cx / FW, (y_tile + TILE + 0.07) / FH, f"c.{k + 1}  {place}", ha="center",
                                va="bottom", fontsize=7.5, fontweight="bold", color=P["ink"]))
    S.panel_label(tile_axes[0], "C", fontsize=15.0, x=0.0, y=1.04, va="bottom")   # fig9:407 (strip letter above)

    any_below = any(s["anom_share_below"] > 0 for s in stats.values()) or any(
        t.get("share_below_anom", 0) > 0 for t in tile_stats)
    any_above = any(s["anom_share_above"] > 0 for s in stats.values()) or any(
        t.get("share_above_anom", 0) > 0 for t in tile_stats)
    cb_left = PR.left_margin + (PR.panel_w - CBAR_W) / 2.0
    cb = draw_colorbar(fig, [cb_left / FW, y_cbar / FH, CBAR_W / FW, CBAR_H / FH], variant, cmap, norm,
                       any_below, any_above)

    draft_txt = None
    if partial:
        n_pending = int(np.sum(~np.logical_and.reduce(
            [pd.Series(seg_ids).isin(mets[d]["seg_id"]).to_numpy() for d in mets])))
        draft_txt = fig.text((FW - PR.right_margin) / FW, (PR.bottom_margin + 0.02) / FH,
                             f"DRAFT \u2014 partial run, {n_tiles}/{N_TILES_TOTAL} tiles\n"
                             f"pale grey: {n_pending:,} segments not yet run",
                             ha="right", va="bottom", fontsize=7.0, color="#8A8580", linespacing=1.25)
    S.finalize_panel_labels(fig)
    n_scaled = S.scale_text(fig)          # v2 type scale; all layout / QA below sees the final sizes

    meta = dict(variant=variant, fig_size_in=[FW, FH], map_in=[MAP_W, MAP_H], tile_in=TILE,
                utci_limits=[UTCI_LO, UTCI_HI], anom_limit=ANOM_LIM, seg_lw_pt=SEG_LW * lw_scale, lw_scale=lw_scale, partial=partial,
                n_tiles=n_tiles, n_segments_total=int(len(segs)), n_text_scaled=n_scaled, days=stats,
                tiles=tile_stats,
                n_segments_coloured={d: int(np.isfinite(mets[d].set_index("seg_id")[BAND_COL]
                                                         .reindex(seg_ids)).sum()) for d in mets})
    handles = dict(fig=fig, map_axes=map_axes, tile_axes=tile_axes, cbar=cb, draft=draft_txt, segs=segs)
    return fig, meta, handles


# =============================================================================================
# QA: overlaps, clipping, smallest printed size
# =============================================================================================
def is_letter(t) -> bool:
    """Boxed panel letters '(a)' sit on the panel frame by Fig 9 convention; exempt from the spill rule."""
    txt = t.get_text()
    return len(txt) == 3 and txt[0] == "(" and txt[2] == ")"


def qa_overlaps(fig, handles) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fbb = fig.bbox
    texts = []
    for ax in fig.axes:
        axis_txt = ([ax.xaxis.label] + list(ax.get_xticklabels())) if ax.axison else []
        for t in list(ax.texts) + axis_txt:
            if t.get_visible() and t.get_text().strip():
                bb = t.get_bbox_patch().get_window_extent(r) if t.get_bbox_patch() else t.get_window_extent(r)
                texts.append((ax, t, bb))
    for t in fig.texts:
        if t.get_text().strip():
            texts.append((None, t, t.get_window_extent(r)))
    pair_hits = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][2], texts[j][2]
            if a.overlaps(b):
                ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 1.0:
                    pair_hits.append([texts[i][1].get_text()[:30], texts[j][1].get_text()[:30], round(ov, 1)])
    # outside the sheet (clipped on save)
    clipped = [t.get_text()[:30] for (_, t, bb) in texts
               if bb.x0 < fbb.x0 - 0.5 or bb.y0 < fbb.y0 - 0.5 or bb.x1 > fbb.x1 + 0.5 or bb.y1 > fbb.y1 + 0.5]
    # tile texts must stay inside their tile; map tags must not run into the neighbouring map / tile row
    spill = []
    for ax in handles["tile_axes"]:
        abb = ax.get_window_extent(r)
        for (tax, t, bb) in texts:
            if tax is ax and not is_letter(t) and (bb.x0 < abb.x0 or bb.x1 > abb.x1 or bb.y0 < abb.y0
                                                   or bb.y1 > abb.y1):
                spill.append(t.get_text()[:30])
    for ax in handles["map_axes"].values():
        abb = ax.get_window_extent(r)
        for (tax, t, bb) in texts:
            if tax is ax and not is_letter(t) and (bb.x1 > abb.x1 + 1 or bb.x0 < abb.x0 - 3):
                spill.append("map: " + t.get_text()[:30])
    # against drawn street segments (map axes); halo'd texts (path_effects) are exempt
    tree = shapely.STRtree(handles["segs"].geometry.values)
    seg_hits = []
    for ax in handles["map_axes"].values():
        inv = ax.transData.inverted()
        for (tax, t, bb) in texts:
            if tax is not ax:
                continue
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            n = len(tree.query(shapely.box(x0, y0, x1, y1), predicate="intersects"))
            seg_hits.append([t.get_text().split("\n")[0][:30], n, bool(t.get_path_effects())])
    cbb = handles["cbar"].ax.get_window_extent(r)
    cbar_hits = [t.get_text() for (tax, t, bb) in texts if tax is handles["cbar"].ax and t.get_text() and
                 bb.overlaps(cbb) and (min(bb.y1, cbb.y1) - max(bb.y0, cbb.y0)) > 1.0]
    sizes = sorted({round(t.get_fontsize(), 2) for (_, t, _) in texts})
    k_print = S.PRINT_WIDTH_IN / fig.get_size_inches()[0]
    return dict(n_texts=len(texts), text_text_overlaps=pair_hits, clipped_outside_sheet=clipped,
                spill_out_of_panel=spill, text_vs_segments=seg_hits, cbar_text_on_bar=cbar_hits,
                author_pt_sizes=sizes, smallest_author_pt=sizes[0],
                smallest_printed_pt_190mm=round(sizes[0] * k_print, 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(OUT), default=DEFAULT_VARIANT,
                    help="anomaly (default) = Fig 4 -> fig_4_street_utci.png; absolute -> _options/fig_4_street_utci_absolute.png")
    ap.add_argument("--qa", type=Path, default=None, help="directory for QA json (nothing written otherwise)")
    ap.add_argument("--lw-scale", type=float, default=LW_SCALE, help="street linewidth factor (tests only)")
    ap.add_argument("--out", type=Path, default=None, help="override output PNG (tests only)")
    args = ap.parse_args(argv)
    data = load_data()
    out = {}
    for variant, path in [(args.variant, args.out or OUT[args.variant])]:
        fig, meta, h = build(variant, data, lw_scale=args.lw_scale)
        qa = qa_overlaps(fig, h) if args.qa else None
        path.parent.mkdir(parents=True, exist_ok=True)
        S.save_figure(fig, path, crop=None)        # Fig 9: full sheet, no bbox crop
        meta["png"] = str(path)
        out[variant] = dict(meta=meta, qa=qa)
    if args.qa:
        args.qa.mkdir(parents=True, exist_ok=True)
        (args.qa / f"fig4_qa_{args.variant}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: v["meta"][kk] for kk in ("variant", "fig_size_in", "partial", "n_tiles", "png")}
                      for k, v in out.items()}, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
