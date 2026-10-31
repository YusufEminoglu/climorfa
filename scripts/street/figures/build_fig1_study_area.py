"""CLIMORFA v2 Figure 1: study area, LCZ context, afternoon wind, three 1 km close-ups (WP4.1).

Run from the repo root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig1_study_area

Composition language: original Fig 1 (atlas) + Fig 9 map furniture.  Everything stylistic comes from
``climorfa_style`` (apply_style first).  No panel titles; axis-less maps; boxed letters (a)-(e).
Panels: (a) FUA + LCZ + 427 tiles + locator squares; (b) three afternoon wind roses + key;
(c)-(e) three 1 km close-ups (building height, Meta tree canopy, street segments).
Double-anonymised: no names, no URLs in the figure.
"""
from __future__ import annotations

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
from shapely.geometry import box as shp_box
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from scripts.street.figures import climorfa_style as S

ROOT = Path(__file__).resolve().parents[3]
OUT_PNG = ROOT / "paper/figures/v2/main/fig_1_study_area.png"
QA_DIR = ROOT / "outputs/street/figures/fig1_qa"
P = S.PALETTE

# ---------------------------------------------------------------------------------------------
# LCZ palette: original four roots kept; the other classes are blends of the same roots toward
# white or charcoal (colour contract of FIGURE_MAPPING.md section 3).  Documented in the caption.
# ---------------------------------------------------------------------------------------------


def blend(c1: str, c2: str, f: float) -> str:
    a, b = np.array(mpl.colors.to_rgb(c1)), np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex(a * (1 - f) + b * f)


GREEN = S.FAMILY_COLORS_FIG2["Vegetation"]  # '#56B881', the paper's vegetation family colour (fig2:29-36)
LCZ_COLORS = {
    2: P["teal_dark"],                      # compact mid-rise: darker teal
    3: P["teal"],                           # ORIGINAL
    4: blend(P["pink_dark"], P["ink"], 0.45),  # open high-rise: darkest pink
    5: P["pink_dark"],                      # open mid-rise
    6: P["pink"],                           # ORIGINAL
    8: P["orange"],                         # ORIGINAL
    9: P["taupe"],                          # ORIGINAL
    10: P["orange_dark"],                   # heavy industry: darker orange
    # natural classes A-G (codes 11-17) are one taupe_light category (Fig 9: non-focus land = taupe_light)
}
LCZ_NAMES = {
    2: "compact mid-rise", 3: "compact low-rise", 4: "open high-rise", 5: "open mid-rise", 6: "open low-rise",
    8: "large low-rise", 9: "sparsely built", 10: "heavy industry",
    11: "dense trees", 12: "scattered trees", 13: "bush, scrub", 14: "low plants", 15: "bare rock, paved",
    16: "bare soil, sand", 17: "water",
}
LCZ_LABEL = {k: (str(k) if k < 11 else "ABCDEFG"[k - 11]) for k in range(2, 18)}
NATURAL = list(range(11, 18))   # A-G
NATURAL_COLOR = P["taupe_light"]
STUDY_LCZ = [3, 6, 8, 9]


def lcz_dark(k: int) -> str:
    """Text/edge colour of a class tag: the original dark key for 3/6/8/9, blended toward ink otherwise."""
    if k in S.CLASS_DARK_KEY:
        return P[S.CLASS_DARK_KEY[k]]
    return blend(LCZ_COLORS[k], P["ink"], 0.55)


# Building-height ramp: palette tokens only (taupe_light -> taupe -> taupe_dark, full range): neutral "context" taupe so that
# no LCZ class colour is reused for a different meaning inside the close-ups.
BLD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "bld_height", [P["taupe_light"], P["taupe"], P["taupe_dark"]])
BLD_VMIN, BLD_VMAX = 2.0, 30.0

# Wind-speed bins (m/s) and ramp: teal root blended toward white / charcoal
SPEED_BINS = [0, 2, 4, 6, 8, np.inf]
SPEED_LABELS = ["< 2", "2 to 4", "4 to 6", "6 to 8", "\u2265 8"]
SPEED_COLORS = [blend(P["teal"], "white", 0.65), blend(P["teal"], "white", 0.30), P["teal"],
                P["teal_dark"], blend(P["teal_dark"], P["ink"], 0.55)]
NSEC, SEC_W = 36, 10.0   # METAR directions are reported in 10 degree steps: 36 sectors centred on multiples of 10
SB_LO, SB_HI = 250.0, 340.0  # sea-breeze sector used in the study (selected_design_days.json)

TILES = [  # id, short place label for the read-out
    ("T_0512_4255", "Alsancak"),
    ("T_0508_4248", "Karaba\u011flar"),
    ("T_0503_4262", "\u00c7i\u011fli AOSB"),
]
KNOT = 0.514444


# =============================================================================================
# data
# =============================================================================================
def load_lcz_5253(bounds, res=50.0):
    """Band 1 of the global LCZ raster nearest-neighbour reprojected to EPSG:5253 over ``bounds``."""
    src = rasterio.open(ROOT / "data/01_raw/gee/lcz/izmir_fua_rub_lcz_latest.tif")
    xmin, ymin, xmax, ymax = bounds
    w, h = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    tr = from_origin(xmin, ymin + h * res, res, res)
    dst = np.full((h, w), 255, dtype=np.uint8)
    reproject(source=rasterio.band(src, 1), destination=dst, dst_transform=tr, dst_crs="EPSG:5253",
              resampling=Resampling.nearest, src_nodata=None, dst_nodata=255)
    return dst, tr


def lcz_rgba(cls, inside):
    """Only the FUA interior is drawn; sea / outside stay transparent (= white page, Fig 9 rule)."""
    img = np.zeros(cls.shape + (4,))
    for k in NATURAL:
        img[(cls == k) & inside] = mpl.colors.to_rgba(NATURAL_COLOR)
    for k, c in LCZ_COLORS.items():
        img[(cls == k) & inside] = mpl.colors.to_rgba(c)
    return img


def wind_tables():
    d = pd.read_csv(ROOT / "data/street/forcing_summer/LTBJ_summer_2021_2025.csv", low_memory=False)
    loc = pd.to_datetime(d["valid"]) + pd.Timedelta(hours=3)   # UTC -> local (+3 h)
    d["date"] = loc.dt.strftime("%Y-%m-%d")
    d["hr"] = loc.dt.hour + loc.dt.minute / 60.0
    d["spd"] = pd.to_numeric(d["sknt"], errors="coerce") * KNOT
    d["vrb"] = d["drct"].astype(str).str.strip().eq("M")
    d["dir"] = pd.to_numeric(d["drct"], errors="coerce")
    aft = d[(d["hr"] >= 14.0) & (d["hr"] <= 17.0) & d["spd"].notna()].copy()   # window of 02_select_design_days.py
    cand = pd.read_csv(ROOT / "data/street/forcing_summer/candidate_days_summary.csv")
    clear = set(cand.loc[cand["sky_clear_09_18"] & cand["no_precip"], "date"])  # study's clear-sky rule
    sets = {
        "all": aft[aft["date"].isin(clear)],
        "imbat": aft[aft["date"] == "2022-07-05"],
        "calm": aft[aft["date"] == "2023-07-26"],
    }
    return sets, len(clear)


def rose_stats(df):
    """16-sector stacked frequencies (share of ALL reports, %), VRB count, sector share."""
    n = len(df)
    nv = int(df["vrb"].sum())
    dd = df[~df["vrb"] & df["dir"].notna()]
    sector = (np.round(dd["dir"].values / SEC_W).astype(int)) % NSEC
    sbin = np.digitize(dd["spd"].values, SPEED_BINS[1:-1])
    mat = np.zeros((NSEC, len(SPEED_LABELS)))
    for s_, b_ in zip(sector, sbin):
        mat[s_, b_] += 1
    in_sb = ((dd["dir"] >= SB_LO) & (dd["dir"] <= SB_HI)).sum()
    return dict(n=n, n_vrb=nv, mat=100.0 * mat / n, in_sb=int(in_sb), in_sb_pct=100.0 * in_sb / n,
                med_spd=float(df["spd"].median()), n_dir=len(dd))


def nice_rings(vmax):
    for step in (1, 2, 5, 10, 20, 25, 50):
        if vmax / step <= 2.6:
            return step, np.arange(step, vmax + 1e-9, step)
    return 50, np.array([50, 100])


# =============================================================================================
# panels
# =============================================================================================
def ring_label_angle(tot, window=20.0):
    """Angle (deg clockwise from N) for the ring labels: least bar mass within +-window deg, preferring ESE..SSE
    (100-170 deg).  ``tot`` = per-sector total (36 sectors of 10 deg)."""
    cand = np.arange(0.0, 360.0, 5.0)

    def mass(a):
        d = np.abs(((np.arange(NSEC) * SEC_W - a + 180.0) % 360.0) - 180.0)
        return float(tot[d <= window].sum())
    pref = [a for a in cand if 100.0 <= a <= 170.0]
    best_pref = min(pref, key=lambda a: (mass(a), abs(a - 135.0)))
    if mass(best_pref) == 0.0:
        return float(best_pref)
    return float(min(cand, key=lambda a: (mass(a), abs(a - 135.0))))


FIG9_AXES_W_IN = 12.0   # Fig 9 map axes width (PRESET_WIDE_13 panel_w)
# Measured calibration: with the same polygon formula the rendered navy arrow came out 35 x 80 px vs Fig 9's 39 x 88 px
# (the Fig 9 axes are height-limited and its arrow is not eroded as much by the 0.9 pt white edge); 1.10 matches it.
ARROW_CAL = 1.10


def add_scale_north_fig9_physical(ax, axes_w_in, base_loc=(0.09, 0.59), scale_len_km=10.0, spec=None):
    """Local variant of climorfa_style.add_scale_north.  Same construction (black/white halves, polygon arrow, N glyph,
    same ratios), but bar thickness, arrow size and gaps are the Fig 9 PHYSICAL sizes (they are fractions of the axes
    in the module, so they shrink with a narrower map while the text stays 7.5 / 9.5 pt)."""
    spec = spec or S.SCALE_NORTH_FIG9
    k = FIG9_AXES_W_IN / axes_w_in            # axes-fraction -> Fig 9 absolute size
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    sb_x0 = xlim[0] + base_loc[0] * dx
    sb_y0 = ylim[0] + base_loc[1] * dy
    slen = scale_len_km * 1000.0
    sb_h = dy * 0.009 * k
    half = scale_len_km / 2.0
    for i in range(2):
        seg_x = sb_x0 + i * (slen / 2.0)
        ax.add_patch(mpatches.Rectangle((seg_x, sb_y0), slen / 2.0, sb_h, facecolor="black" if i == 0 else "white",
                                        edgecolor="black", linewidth=0.7, zorder=25))
        ax.text(seg_x, sb_y0 + sb_h * spec.label_k, f"{i * half:g}", ha="center", va="bottom", fontsize=spec.label_fs,
                fontweight="bold", color=S.PANEL_TEXT, zorder=25)
    ax.text(sb_x0 + slen, sb_y0 + sb_h * spec.label_k, f"{int(scale_len_km)} km", ha="center", va="bottom",
            fontsize=spec.label_fs, fontweight="bold", color=S.PANEL_TEXT, zorder=25)
    cx = sb_x0 + slen / 2.0
    na_base = sb_y0 + sb_h * spec.arrow_base_k
    na_w, na_h = dx * spec.arrow_w_frac * k * ARROW_CAL, dy * 0.040 * k * ARROW_CAL
    ax.add_patch(mpatches.Polygon(
        [[cx, na_base + na_h], [cx + na_w / 2.0, na_base], [cx, na_base + na_h * 0.25], [cx - na_w / 2.0, na_base]],
        facecolor=S.NORTH_ARROW, edgecolor="white", linewidth=0.9, zorder=25))
    ax.text(cx, na_base + na_h + dy * 0.006 * k, "N", ha="center", va="bottom", fontsize=spec.north_fs,
            fontweight="bold", color=S.NORTH_ARROW, zorder=25)


def draw_rose(ax, st, tag, rmax_pct=None, hole=0.36):
    th = np.radians(np.arange(NSEC) * SEC_W)
    width = np.radians(SEC_W) * 0.86
    tot = st["mat"].sum(axis=1)
    peak = tot.max() if rmax_pct is None else rmax_pct
    step, rings = nice_rings(peak * 1.05)
    rlim = rings[-1] if rings[-1] >= peak * 1.02 else rings[-1] + step
    rings = np.arange(step, rlim + 1e-9, step)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rorigin(-hole * rlim)
    ax.set_ylim(0, rlim)
    ax.set_facecolor("white")
    # sea-breeze sector behind the bars (250-340 deg)
    mid, half = np.radians((SB_LO + SB_HI) / 2), np.radians((SB_HI - SB_LO + SEC_W) / 2)  # sectors 250..340 inclusive
    ax.bar(mid, rlim, width=2 * half, bottom=0, color=P["teal_light"], edgecolor="none", alpha=0.95, zorder=0.5)
    for edge in (SB_LO - SEC_W / 2, SB_HI + SEC_W / 2):
        ax.plot([np.radians(edge)] * 2, [0, rlim], color=P["teal_dark"], lw=0.7, ls=(0, (3, 2)), zorder=1.5)
    bottom = np.zeros(NSEC)
    for j in range(st["mat"].shape[1]):
        for b_ in ax.bar(th, st["mat"][:, j], width=width, bottom=bottom, color=SPEED_COLORS[j],
                         edgecolor="white", linewidth=0.3, zorder=3):
            b_.set_gid("rose_bar")
        bottom += st["mat"][:, j]
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8.5, color=P["ink"], fontweight="bold")
    ax.tick_params(axis="x", pad=-1, bottom=False)
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{r:g}%" for r in rings], fontsize=7.5, color=P["muted"])
    ax.set_rlabel_position(ring_label_angle(tot))
    ax._ring_label_angle = ax.get_rlabel_position()
    for t in ax.get_yticklabels():
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.1, alpha=0.9)); t.set_zorder(8)
    ax.tick_params(axis="y", left=False)
    ax.grid(True, color=P["grid"], linewidth=0.6)
    ax.xaxis.grid(True, color=P["grid"], linewidth=0.5)
    ax.spines["polar"].set_color(P["grid"]); ax.spines["polar"].set_linewidth(0.8); ax.spines["polar"].set_zorder(0.3)
    # central variable-wind (VRB) count, never plotted at 0 deg
    r0 = -hole * rlim
    ax.bar(0, hole * rlim, width=2 * np.pi, bottom=r0, color="white", edgecolor=P["taupe"], linewidth=0.8, zorder=4)
    ax.text(0, r0, f"VRB\n{st['n_vrb']}", ha="center", va="center", fontsize=6.2, fontweight="bold",
            color=P["ink"], zorder=6, linespacing=0.95)
    # sector label outside the rim
    ax.text(mid, rlim * 1.22, f"{SB_LO:g}\u2013{SB_HI:g}\u00b0", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=P["teal_dark"], clip_on=False)
    return rlim


def rose_badge(ax, text, x=0.0, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, fontweight="bold",   # fig9 badge size
            color=P["ink"], linespacing=1.15,
            bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.18", "alpha": 0.90},
            zorder=10, clip_on=False)


def tile_arrays(tid):
    base = ROOT / "data/street/tiles/2m"
    rd = lambda n: rasterio.open(base / n / f"{tid}.tif").read(1)
    dsm, dem, tr, lm = rd("building_dsm"), rd("dem"), rd("tree_dsm"), rd("landmask")
    sl = slice(50, 550)  # inner 1000 m of the 600 x 600 buffered tile
    h = (dsm - dem)[sl, sl]
    h = np.where(np.isfinite(h), h, 0.0)
    return h, tr[sl, sl], lm[sl, sl]


def add_tile_scale(ax, x0, y0, length=200.0):
    """Two-segment black/white scale bar (geometry of climorfa_style.add_scale_north) at the tile's upper right
    (the lower left carries the read-out box)."""
    sb_h = 1000.0 * 0.009
    sx = x0 + 1000.0 - 170.0 - length
    sy = y0 + 1000.0 - 190.0
    ax.add_patch(mpatches.Rectangle((sx - 40, sy - 22), length + 40 + 85, 112, facecolor="white", alpha=0.85,
                                    edgecolor="none", zorder=24))
    for i in range(2):
        ax.add_patch(mpatches.Rectangle((sx + i * length / 2, sy), length / 2, sb_h,
                                        facecolor="black" if i == 0 else "white", edgecolor="black", linewidth=0.7, zorder=25))
    ax.text(sx, sy + sb_h * 1.6, "0", ha="center", va="bottom",
            fontsize=S.SCALE_NORTH_FIG9.label_fs, fontweight="bold", color=S.PANEL_TEXT, zorder=25)
    ax.text(sx + length, sy + sb_h * 1.6, f"{length:g} m", ha="center", va="bottom",
            fontsize=S.SCALE_NORTH_FIG9.label_fs, fontweight="bold", color=S.PANEL_TEXT, zorder=25)


def plot_closeup(ax, tid, place, tiles_gdf, seg_gdf, lcz_arr, lcz_tr, inside_mask):
    row = tiles_gdf.set_index("tile_id").loc[tid]
    x0, y0 = float(row["x0"]), float(row["y0"])
    box = (x0, y0, x0 + 1000.0, y0 + 1000.0)
    h, tr, lm = tile_arrays(tid)
    land = lm > 0.5
    bld = (h >= BLD_VMIN) & land
    can = (tr >= 2.0) & land
    ext = [x0, x0 + 1000.0, y0, y0 + 1000.0]

    ax.set_facecolor(P["surface"])                               # tile surface (fig1:301)
    ax.imshow(np.ma.masked_where(~can, tr), cmap=S.CANOPY_CMAP_10M, vmin=S.CANOPY_VMIN, vmax=S.CANOPY_VMAX,
              alpha=S.CANOPY_ALPHA, extent=ext, zorder=2, interpolation="nearest")   # fig9:498-503
    seg = seg_gdf[seg_gdf["tile_id"] == tid].copy()
    seg["geometry"] = seg.geometry.intersection(shp_box(*box))
    seg = seg[~seg.geometry.is_empty]
    seg.plot(ax=ax, color=P["taupe_dark"], linewidth=0.7, alpha=0.82, zorder=3, capstyle="round")  # fig1:303
    ax.imshow(np.ma.masked_where(~bld, h), cmap=BLD_CMAP, vmin=BLD_VMIN, vmax=BLD_VMAX, extent=ext,
              zorder=4, interpolation="nearest")
    xs = np.linspace(x0, x0 + 1000.0, h.shape[1]); ys = np.linspace(y0 + 1000.0, y0, h.shape[0])
    ax.contour(xs + 1.0, ys - 1.0, bld.astype(float), levels=[0.5], colors=S.PANEL_EDGE, linewidths=0.15, zorder=5)   # fig9 close-up building outline
    ax.add_patch(mpatches.Rectangle((x0, y0), 1000.0, 1000.0, fill=False, edgecolor=P["ink"], linewidth=1.15, zorder=6))
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3]); ax.set_aspect("equal"); ax.axis("off")

    # LCZ of the close-up: majority of the global LCZ pixels inside the tile
    r0 = int(round(((y0 + 1000.0) - lcz_tr.f) / lcz_tr.e))
    c0 = int(round((x0 - lcz_tr.c) / lcz_tr.a))
    n = int(1000 / abs(lcz_tr.a))
    win = lcz_arr[r0:r0 + n, c0:c0 + n]
    vals, cnts = np.unique(win[(win > 0) & (win < 17)], return_counts=True)
    klass = int(vals[cnts.argmax()])
    share = float(cnts.max() / cnts.sum())
    stats = dict(tile_id=tid, place=place, lcz=klass, lcz_share=share,
                 med_h=float(np.median(h[bld])) if bld.any() else float("nan"),
                 bld_cover=float(bld.sum() / land.sum()), canopy_share=float(can.sum() / land.sum()),
                 sea_share=float((~land).mean()), n_seg=int(len(seg)),
                 seg_len_km=float(seg.length.sum() / 1000.0), lcz_hist={int(a): int(b) for a, b in zip(vals, cnts)})
    order = np.argsort(-cnts)
    tag = f"LCZ {LCZ_LABEL[klass]} ({100 * share:.0f} %)"
    if share < 0.60:   # mixed tile: the next two classes follow ('LCZ 8 (32 %); 2, 3'); full mix in the caption
        tag += "; " + ", ".join(LCZ_LABEL[int(vals[i])] for i in order[1:3])
    S.tile_tag(ax, tag, lcz_dark(klass), x=0.03, y=0.97)
    ax.texts[-1].get_bbox_patch().set_edgecolor(LCZ_COLORS[klass])
    ax.texts[-1].set_linespacing(1.25)
    add_tile_scale(ax, x0, y0)
    S.tile_readout(ax, f"{place}  {tid}\nmedian h {stats['med_h']:.0f} m, canopy {100 * stats['canopy_share']:.0f} %",
                   x=0.03, y=0.03)
    return stats


# =============================================================================================
def text_overlap_qa(fig, tol_px=1.0):
    """Pairwise overlap of all visible text boxes (incl. their bbox patches) and clipping at the figure edge."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip() or t.get_text().startswith(("Easting", "Northing")):
            continue   # geopandas axis labels of axis-off map axes are never drawn
        bb = t.get_bbox_patch().get_window_extent(r) if t.get_bbox_patch() is not None else t.get_window_extent(r)
        if bb.width < 1 or bb.height < 1:
            continue
        items.append((t.get_text().replace("\n", " / ")[:40], bb))
    W, H = fig.canvas.get_width_height()
    hits, clipped = [], []
    for i in range(len(items)):
        n, b = items[i]
        if b.x0 < -tol_px or b.y0 < -tol_px or b.x1 > W + tol_px or b.y1 > H + tol_px:
            clipped.append(n)
        for j in range(i + 1, len(items)):
            m, c = items[j]
            ox = min(b.x1, c.x1) - max(b.x0, c.x0)
            oy = min(b.y1, c.y1) - max(b.y0, c.y0)
            if ox > tol_px and oy > tol_px:
                hits.append(dict(a=n, b=m, overlap_px=[round(ox, 1), round(oy, 1)]))
    # ring labels vs bars (true wedge paths) and vs the sector-shading text, per rose
    ring = {}
    for ax in fig.axes:
        if not hasattr(ax, "_ring_label_angle"):
            continue
        bars = [p for p in ax.patches if p.get_gid() == "rose_bar"]
        sector_txt = [t for t in ax.texts if "\u00b0" in t.get_text() and t.get_text().startswith("250")]
        bad = []
        for lab in ax.get_yticklabels():
            if not lab.get_text():
                continue
            bb = lab.get_window_extent(r)
            if any(p.get_transform().transform_path(p.get_path()).intersects_bbox(bb, filled=True) for p in bars):
                bad.append(lab.get_text() + " x bar")
            for t in sector_txt:
                tb = t.get_window_extent(r)
                if min(bb.x1, tb.x1) - max(bb.x0, tb.x0) > 0 and min(bb.y1, tb.y1) - max(bb.y0, tb.y0) > 0:
                    bad.append(lab.get_text() + " x sector text")
        ring[f"rose_{len(ring)}_angle_{ax._ring_label_angle:.0f}"] = bad
    return dict(n_text=len(items), overlaps=hits, clipped=clipped, ring_label_checks=ring)


def main(variant="final"):
    S.apply_style()
    np.random.seed(20261030)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    shore = gpd.read_file(ROOT / "data/02_interim/shoreline_epsg5253.gpkg")
    tiles = gpd.read_file(ROOT / "data/street/grid/tiles_core.gpkg")
    segs = gpd.read_file(ROOT / "data/street/analysis/segments.gpkg", columns=["tile_id", "seg_id"])
    xmin, ymin, xmax, ymax = study.total_bounds
    pad = 0.015
    ext = (xmin - (xmax - xmin) * pad, ymin - (ymax - ymin) * pad, xmax + (xmax - xmin) * pad, ymax + (ymax - ymin) * pad)
    map_ar = (ext[3] - ext[1]) / (ext[2] - ext[0])

    # ---- geometry of the sheet (inches) -------------------------------------------------------
    # 13.0 in wide (Fig 9 width); all text is multiplied by 1.4 (floor 6 pt printed) by S.scale_text at the end, so the
    # layout below is planned for the SCALED text sizes.  Height is free.
    FW = S.PRESET_WIDE_13.fig_w
    L, R, TOPM, BOTM = S.PRESET_WIDE_13.left_margin, S.PRESET_WIDE_13.right_margin, 0.50, 0.08
    CONTENT_W = FW - L - R
    MAP_W = 6.80
    MAP_H = MAP_W * map_ar
    LEGCOL_X = L + MAP_W + 0.30
    LEGCOL_W = FW - R - LEGCOL_X
    GAP12, GAP23 = 0.45, 0.55   # = height of a boxed panel letter + a hair (letters sit just above their content)
    ROW2_H = 3.50
    KEY_W = 3.00
    T_GAP = 0.25
    LEGCELL_W = 2.70
    TILE = (CONTENT_W - 3 * T_GAP - LEGCELL_W) / 3.0
    FH = TOPM + MAP_H + GAP12 + ROW2_H + GAP23 + TILE + BOTM
    fig = plt.figure(figsize=(FW, FH))
    y_row1_top = FH - TOPM
    y_map0 = y_row1_top - MAP_H
    SCALE_BASE = (0.005, 0.50)   # Fig 9 used (0.09, 0.59) for a 5 km bar; the 10 km bar needs the open sea further left and lower

    # ---- (a) map: sea = white page, nothing drawn; FUA outline defines the coast (Fig 9 rule) -----
    ax_a = S.inch_axes(fig, L, y_map0, MAP_W, MAP_H)
    ax_a.set_anchor("S")
    cls, tr = load_lcz_5253(ext, 50.0)
    inside = rasterize(study.geometry, out_shape=cls.shape, transform=tr, fill=0, default_value=1).astype(bool)
    ax_a.imshow(lcz_rgba(cls, inside), extent=[ext[0], ext[2], ext[1], ext[3]], origin="upper",
                interpolation="nearest", zorder=1)
    tiles.boundary.plot(ax=ax_a, color=S.PANEL_EDGE, linewidth=0.25, alpha=0.6, zorder=9)
    S.draw_study_outline(ax_a, study, "ink")
    S.set_map_extent(ax_a, study.total_bounds, pad=pad)
    add_scale_north_fig9_physical(ax_a, MAP_W, base_loc=SCALE_BASE, scale_len_km=10.0)
    # locator squares for (c)-(e): one ink 1 km square with a white halo
    LETTERS = ["C", "D", "E"]
    trow = tiles.set_index("tile_id")
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    for (tid, _), letter in zip(TILES, LETTERS):
        x0, y0 = trow.loc[tid, "x0"], trow.loc[tid, "y0"]
        ax_a.add_patch(mpatches.Rectangle((x0, y0), 1000, 1000, fill=False, edgecolor=P["ink"], linewidth=1.15,
                                          zorder=20, path_effects=halo))
        ax_a.text(x0 + 500, y0 + 1000 + 700, letter, ha="center", va="bottom", fontsize=9, fontweight="bold",
                  color=S.PANEL_TEXT, bbox=dict(S.PANEL_LABEL_BBOX, boxstyle="round,pad=0.16"), zorder=30)
    S.panel_label(ax_a, "A")

    # LCZ legend: its own column right of the map (plenty of room for the enlarged text); stat note below it
    ax_leg = S.inch_axes(fig, LEGCOL_X, y_map0, LEGCOL_W, MAP_H)
    ax_leg.axis("off")
    area = {k: float(((cls == k) & inside).sum()) for k in range(2, 18)}
    tot_in = float(sum(area.values()))
    nat_pct = 100 * sum(area[k] for k in NATURAL) / tot_in
    handles = []
    for k in [2, 3, 4, 5, 6, 8, 9, 10]:
        pct = 100 * area[k] / tot_in
        lab = f"{LCZ_LABEL[k]}  {LCZ_NAMES[k]}  ({pct:.1f} %)" if pct >= 0.1 else f"{LCZ_LABEL[k]}  {LCZ_NAMES[k]}  (<0.1 %)"
        handles.append(Patch(facecolor=LCZ_COLORS[k], edgecolor="none", label=lab))
    handles.append(Patch(facecolor=NATURAL_COLOR, edgecolor="#A39E98", linewidth=0.4,
                         label=f"natural cover, LCZ A–G  ({nat_pct:.1f} %)"))
    kw = dict(S.LEGEND_CHIPS_NOFRAME); kw["fontsize"] = 8.0; kw.pop("handleheight")
    ax_leg.legend(handles=handles, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0,
                  handlelength=1.2, handleheight=1.0, labelspacing=0.45, **kw)
    ax_leg.legend_.set_title("Global LCZ (share of FUA area)")
    ax_leg.legend_.get_title().set_fontsize(8.5); ax_leg.legend_.get_title().set_fontweight("bold")
    ax_leg.legend_._legend_box.align = "left"
    note_leg = S.stat_note(ax_leg, f"{len(tiles)} analysis tiles (1 km)\n{len(segs):,} street segments", x=0.0, y=0.20) or ax_leg.texts[-1]
    note_leg.set_va("top")

    # ---- (b) wind roses: own row -----------------------------------------------------------------
    sets, n_clear = wind_tables()
    stats = {k: rose_stats(v) for k, v in sets.items()}
    y_row2_top = y_map0 - GAP12
    y_row2_bot = y_row2_top - ROW2_H
    ax_b = S.inch_axes(fig, L, y_row2_bot, CONTENT_W, ROW2_H)   # invisible carrier of the letter
    ax_b.axis("off"); ax_b.set_facecolor("none")
    S.panel_label(ax_b, "B")
    cell_w = (CONTENT_W - KEY_W) / 3.0
    rose_in = 2.10
    keys = ["all", "imbat", "calm"]
    names = {
        "all": f"All clear afternoons\n{n_clear} days, {stats['all']['n']:,} reports",
        "imbat": f"IMBAT design day\n2022-07-05, {stats['imbat']['n']} reports",
        "calm": f"CALM design day\n2023-07-26, {stats['calm']['n']} reports",
    }
    rose_axes = {}
    for i, key in enumerate(keys):
        x_in = L + i * cell_w + (cell_w - rose_in) / 2.0
        y_in = y_row2_bot + 0.70
        axr = fig.add_axes([x_in / FW, y_in / FH, rose_in / FW, rose_in / FH], projection="polar")
        draw_rose(axr, stats[key], key)
        st = stats[key]
        rose_badge(axr, names[key], x=-0.12, y=1.153)   # box bottom ~0.08 in above the N label
        S.tile_readout(axr, f"{SB_LO:g}–{SB_HI:g}°: {st['in_sb_pct']:.0f} %\nmedian {st['med_spd']:.1f} m/s",
                       x=-0.12, y=-0.125)
        axr.texts[-1].set_va("top")                    # box top ~0.08 in below the S label
        rose_axes[key] = axr
    # key cell (right end of the row)
    kx = L + 3 * cell_w + 0.10
    ax_k = S.inch_axes(fig, kx, y_row2_bot, KEY_W - 0.10, ROW2_H)
    ax_k.axis("off")
    kh = [Patch(facecolor=c, edgecolor="white", linewidth=0.3, label=l) for c, l in zip(SPEED_COLORS, SPEED_LABELS)]
    lg1 = ax_k.legend(handles=kh, title="Wind speed (m/s)", loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1,
                      fontsize=8.0, title_fontsize=8.5, frameon=False, handlelength=1.3, handleheight=0.9,
                      borderaxespad=0.0, labelspacing=0.3)
    lg1._legend_box.align = "left"
    lg1.get_title().set_fontweight("bold")
    ax_k.add_artist(lg1)
    kh2 = [Patch(facecolor=P["teal_light"], edgecolor=P["teal_dark"], linewidth=0.7, linestyle=(0, (3, 2)),
                 label=f"sea-breeze sector\n{SB_LO:g}–{SB_HI:g}° (wind from)"),
           Patch(facecolor="white", edgecolor=P["taupe"], linewidth=0.8, label="VRB: variable\nreports, count")]
    lg2 = ax_k.legend(handles=kh2, loc="upper left", bbox_to_anchor=(0.0, 0.50), fontsize=8.0, frameon=False,
                handlelength=1.3, handleheight=1.5, borderaxespad=0.0, labelspacing=0.5)
    note_k = ax_k.text(0.0, 0.0, "LTBJ METAR, 14–17 h local;\n10° sectors, rings in %.",
                       transform=ax_k.transAxes, fontsize=7.5, color=P["ink"], va="top", ha="left", linespacing=1.3)

    # ---- (c)-(e) close-ups -------------------------------------------------------------------
    y_row3 = BOTM
    tile_stats = []
    for i, ((tid, place), letter) in enumerate(zip(TILES, LETTERS)):
        x_in = L + i * (TILE + T_GAP)
        axt = S.inch_axes(fig, x_in, y_row3, TILE, TILE)
        tile_stats.append(plot_closeup(axt, tid, place, tiles, gpd.read_file(
            ROOT / "data/street/analysis/segments.gpkg", columns=["tile_id", "seg_id"], bbox=tuple(
                [trow.loc[tid, "x0"], trow.loc[tid, "y0"], trow.loc[tid, "x0"] + 1000, trow.loc[tid, "y0"] + 1000])),
            cls, tr, inside))
        S.panel_label(axt, letter)
    # legend cell (standalone colourbars with the module's Fig 9 font sizes + keys)
    lx = L + 3 * (TILE + T_GAP)
    cb_w = LEGCELL_W - 0.30
    for j, (cm, v0, v1, lab, y_off) in enumerate([
            (BLD_CMAP, BLD_VMIN, BLD_VMAX, "Building height (m)", TILE - 0.45),
            (S.CANOPY_CMAP_10M, S.CANOPY_VMIN, S.CANOPY_VMAX, "Tree canopy height (m)", TILE - 1.45)]):
        b = [(lx + 0.10) / FW, (y_row3 + y_off) / FH, cb_w / FW, 0.13 / FH]
        cb = S.add_colorbar(fig, b, cm, v0, v1, lab, "horizontal")
        if j == 0:
            cb.set_ticks([2, 10, 20, 30]); cb.ax.set_xticklabels(["2", "10", "20", "≥ 30"])
        else:
            cb.set_ticks([2, 5, 10, 15]); cb.ax.set_xticklabels(["2", "5", "10", "≥ 15"])
        cb.outline.set_edgecolor(P["grid"]); cb.ax.grid(False)
    ax_l = S.inch_axes(fig, lx, y_row3, LEGCELL_W, 0.80)
    ax_l.axis("off")
    kh3 = [Line2D([0], [0], color=P["taupe_dark"], lw=0.7, alpha=0.82, label="street segment"),
           Patch(facecolor="none", edgecolor=P["ink"], linewidth=1.15, label="1 km close-up tile")]
    ax_l.legend(handles=kh3, loc="upper left", fontsize=8.0, frameon=False, handlelength=1.8, borderaxespad=0.0,
                labelspacing=0.6)

    S.finalize_panel_labels(fig)
    n_scaled = S.scale_text(fig)   # v2 type scale: x1.4, floor 6 pt printed at 190 mm
    # ---- stack the notes directly under their legends (measured after the type scale) ------------
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()

    def bottom_frac(ax, artist):
        return ax.transAxes.inverted().transform((0, artist.get_window_extent(rr).y0))[1]
    note_leg.set_y(bottom_frac(ax_leg, ax_leg.get_legend()) - 0.15 / MAP_H)          # 0.15 in under the LCZ legend
    lg2.set_bbox_to_anchor((0.0, bottom_frac(ax_k, lg1) - 0.20 / ROW2_H), transform=ax_k.transAxes)
    fig.canvas.draw()
    note_k.set_y(bottom_frac(ax_k, ax_k.get_legend()) - 0.12 / ROW2_H)                 # 0.12 in under the sector key

    ovl = text_overlap_qa(fig)
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text)
             if t.get_visible() and t.get_text().strip() and not t.get_text().startswith(("Easting", "Northing"))]
    ovl["min_authored_pt"] = float(min(sizes))
    ovl["min_printed_pt_190mm"] = float(min(sizes) * (190 / 25.4) / FW)
    (QA_DIR / "text_overlap.json").write_text(json.dumps(ovl, indent=1, ensure_ascii=False), encoding="utf-8")
    print("TEXT OVERLAP:", json.dumps(ovl, ensure_ascii=False))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    S.save_figure(fig, OUT_PNG, crop=None)
    qa = dict(fig_size_in=[FW, FH], map_in=[MAP_W, MAP_H], tile_in=TILE, tiles=tile_stats,
              wind={k: {kk: (vv if not isinstance(vv, np.ndarray) else None) for kk, vv in v.items() if kk != "mat"}
                    for k, v in stats.items()},
              n_clear_days=n_clear, lcz_area_share_pct={LCZ_LABEL[k]: round(100 * area[k] / tot_in, 2) for k in area}, natural_share_pct=round(nat_pct, 2),
              n_tiles=len(tiles), n_segments=int(len(segs)))
    (QA_DIR / "fig1_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    return OUT_PNG


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
