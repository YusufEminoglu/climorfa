"""CLIMORFA v2 graphical abstract: 'Same zone, different streets' (WP4.7).

Run from the repo root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_graphical_abstract
        -> paper/figures/v2/main/graphical_abstract_v2.png
    (add ``--qa <dir>`` to also write the QA json and a 1328 px downsample into <dir>; nothing else is written)

Composition: one flat LCZ tile, the two real 1 km close-ups that carry it, three computed numbers.  Sheet 13.0 x 5.2 in
at 300 dpi (Elsevier landscape 1328 x 531 px and above).  The close-ups are drawn by ``build_fig4_street_utci.draw_tile``,
so grammar and anomaly ramp are those of Fig 4 (c); the anomaly is taken against the day median of all segments.

Type: ``climorfa_style.scale_text`` is NOT used.  Its x1.4 factor with a 10.4 pt floor is meant for v1 sizes; here every
size is set directly with a floor of MIN_PT = 11 pt on the 13 in sheet, checked by the QA.

Every number comes from the final files (h1_results.json, final_20261030/seg_metrics_*.parquet, the 2 m tiles, the LCZ
raster through the Fig 1 route).  The build stops on anything that is not the final 427-tile run.
Double-anonymised: no names, no tool names, no URLs in the figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image

from scripts.street.figures import build_fig1_study_area as F1
from scripts.street.figures import build_fig4_street_utci as F4
from scripts.street.figures import climorfa_style as S

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/figures/v2/main/graphical_abstract_v2.png"
H1_JSON = ROOT / "outputs/street/analysis/h1/results/h1_results.json"
FINAL = ROOT / "data/street/analysis/final_20261030"
P = S.PALETTE

DAY, CALM = "IMBAT", "CALM"
LCZ = 8
TILES = [("T_0512_4255", "Alsancak"), ("T_0503_4262", "\u00c7i\u011fli AOSB")]
QA_THUMB_W = 1328

# ---- sheet geometry (inches) ----------------------------------------------------------------
FW, FH = S.V2_FIG_WIDTH_IN, 5.2
ML = 0.30                         # outer margin
FLAT_W = 2.05                     # flat LCZ tile = width of the left column
GAP_ARROW = 0.34                  # left column -> close-ups
TILE = 3.30
GAP_TILE = 0.12
GAP_R = 0.32                      # close-ups -> numbers
X_T0 = ML + FLAT_W + GAP_ARROW
X_R = X_T0 + 2 * TILE + GAP_TILE + GAP_R
Y_TILE = 0.80                     # bottom of the close-ups
Y_TOP = Y_TILE + TILE             # top of the close-ups
BAR_H = 0.085                     # LCZ share bar, sits on the close-up
Y_HEAD = Y_TOP + BAR_H + 0.055    # baseline of the place header
Y_TITLE = FH - 0.15               # top of the headline

# ---- type (pt on the 13 in sheet) -----------------------------------------------------------
MIN_PT = 11.0
FS_TITLE = 25.0
FS_LCZ = 38.0
FS_LCZ_NAME = 15.0
FS_NUM = 36.0
FS_NUM_DAY = 29.0
FS_LABEL = 13.0
FS_PLACE = 14.0
FS_SMALL = 12.0
FS_KEY = 11.5


# =============================================================================================
# data
# =============================================================================================
def load_numbers() -> dict:
    res = json.loads(H1_JSON.read_text(encoding="utf-8"))["results"]
    days = {}
    for d in (DAY, CALM):
        if not str(res[d]["label"]).startswith(f"FINAL ({F4.N_TILES_TOTAL} tiles)"):
            raise SystemExit(f"{H1_JSON.name}: {d} is not the final run: {res[d]['label']!r}")
        m = pd.read_parquet(FINAL / f"seg_metrics_{d}.parquet",
                            columns=["seg_id", F4.BAND_COL, "svf_med", "n_tiles_at_extraction", "partial_run"])
        if int(m["n_tiles_at_extraction"].max()) != F4.N_TILES_TOTAL or bool(m["partial_run"].any()):
            raise SystemExit(f"seg_metrics_{d}: not the final {F4.N_TILES_TOTAL}-tile table")
        days[d] = dict(met=m, stats=F4.day_stats(m), n_segments=int(res[d]["n_segments"]),
                       r2_lcz=float(res[d]["cv"]["r2_lcz_only"]), r2_morph=float(res[d]["cv"]["r2_morphology_lgbm"]),
                       r2_lcz_ci=res[d]["cv"]["r2_lcz_only_ci95"], r2_morph_ci=res[d]["cv"]["r2_morphology_lgbm_ci95"])
    return days


def tile_facts(tid: str, tiles: gpd.GeoDataFrame, lcz_arr, lcz_tr, met: pd.DataFrame) -> dict:
    """LCZ share (the Fig 1 rule: most frequent global LCZ class of the 50 m pixels in the tile), median building height
    (the Fig 1 rule: h >= 2 m on land) and the median sky view factor of the tile's segments."""
    x0, y0 = float(tiles.loc[tid, "x0"]), float(tiles.loc[tid, "y0"])
    r0 = int(round(((y0 + 1000.0) - lcz_tr.f) / lcz_tr.e))
    c0 = int(round((x0 - lcz_tr.c) / lcz_tr.a))
    n = int(1000 / abs(lcz_tr.a))
    win = lcz_arr[r0:r0 + n, c0:c0 + n]
    vals, cnts = np.unique(win[(win > 0) & (win < 17)], return_counts=True)
    hist = {int(a): int(b) for a, b in zip(vals, cnts)}
    if int(vals[cnts.argmax()]) != LCZ:
        raise SystemExit(f"{tid}: most frequent class is not LCZ {LCZ}: {hist}")
    h, _, lm = F1.tile_arrays(tid)
    bld = (h >= F1.BLD_VMIN) & (lm > 0.5)
    seg = pyogrio.read_dataframe(FINAL / "segments.gpkg", layer="segments", read_geometry=False,
                                 columns=["seg_id", "tile_id"], where=f"tile_id = '{tid}'")
    svf = met.loc[met["seg_id"].isin(seg["seg_id"]), "svf_med"]
    return dict(tile_id=tid, lcz_share=hist[LCZ] / sum(hist.values()), lcz_hist=hist, med_h=float(np.median(h[bld])),
                med_svf=float(svf.median()), n_segments=int(len(seg)))


# =============================================================================================
# drawing helpers (all positions in inches)
# =============================================================================================
def tx(fig, x, y, s, fs, color=None, weight="normal", ha="left", va="baseline", **kw):
    return fig.text(x / FW, y / FH, s, fontsize=fs, color=color or P["ink"], fontweight=weight, ha=ha, va=va, **kw)


def rect(fig, x, y, w, h, color, **kw):
    r = mpatches.Rectangle((x, y), w, h, transform=fig.dpi_scale_trans, facecolor=color, edgecolor="none",
                           linewidth=0, **kw)
    fig.add_artist(r)
    return r


def draw_colour_key(fig, x, y, w, any_below, any_above):
    """Fig 4 anomaly bar at the GA type sizes."""
    h = 0.15
    cax = fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    sm = mpl.cm.ScalarMappable(cmap=F4.CMAP_ANOM, norm=mpl.colors.Normalize(-F4.ANOM_LIM, F4.ANOM_LIM))
    sm.set_array([])
    extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((any_below, any_above), "neither")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend, extendfrac=0.05)
    ticks = np.arange(-F4.ANOM_LIM, F4.ANOM_LIM + 0.01, 2.0)
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([("\u2212" if t < 0 else "+" if t > 0 else "") + f"{abs(t):g}" for t in ticks])
    cb.ax.tick_params(labelsize=FS_KEY, colors=P["muted"], length=2.5, width=0.7, pad=1.5)
    cb.ax.grid(False)
    tx(fig, x, y + h + 0.075, "cooler", FS_KEY, P["teal_dark"], "bold")
    tx(fig, x + w, y + h + 0.075, "hotter", FS_KEY, S.LST_TEAL_PURPLE_COLORS[-1], "bold", ha="right")
    return cb


# =============================================================================================
def build(days: dict, data: dict):
    S.apply_style()
    np.random.seed(20261030)
    tiles = data["tiles"]
    d, c = days[DAY], days[CALM]
    med_d, med_c = d["stats"]["median"], c["stats"]["median"]

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    # ---- headline ---------------------------------------------------------------------------
    tx(fig, ML, Y_TITLE, "Same zone, different streets", FS_TITLE, weight="bold", va="top")

    # ---- left: one LCZ label ----------------------------------------------------------------
    y_flat = Y_TOP - FLAT_W + BAR_H
    rect(fig, ML, y_flat, FLAT_W, FLAT_W, S.CLASS_COLORS[LCZ], zorder=2)
    fig.add_artist(mpatches.Rectangle((ML, y_flat), FLAT_W, FLAT_W, transform=fig.dpi_scale_trans, fill=False,
                                      edgecolor=P["ink"], linewidth=0.9, zorder=3))
    cx = ML + FLAT_W / 2.0
    tx(fig, cx, y_flat + FLAT_W * 0.54, f"LCZ {LCZ}", FS_LCZ, weight="bold", ha="center", va="center", zorder=4)
    tx(fig, cx, y_flat + FLAT_W * 0.30, F1.LCZ_NAMES[LCZ], FS_LCZ_NAME, ha="center", va="center", zorder=4)

    # ---- connector --------------------------------------------------------------------------
    y_arrow = y_flat + FLAT_W / 2.0
    arrow = mpatches.FancyArrowPatch((ML + FLAT_W + 0.07, y_arrow), (X_T0 - 0.07, y_arrow),
                                     transform=fig.dpi_scale_trans, arrowstyle="-|>", mutation_scale=15,
                                     color=P["taupe_dark"], linewidth=1.6, shrinkA=0, shrinkB=0)
    fig.add_artist(arrow)

    # ---- centre: the two real close-ups ------------------------------------------------------
    cmap, norm = F4.CMAP_ANOM, mpl.colors.Normalize(-F4.ANOM_LIM, F4.ANOM_LIM)
    tile_axes, tile_st = [], []
    for k, (tid, place) in enumerate(TILES):
        x = X_T0 + k * (TILE + GAP_TILE)
        ax = S.inch_axes(fig, x, Y_TILE, TILE, TILE)
        st = F4.draw_tile(ax, tid, DAY, tiles, cmap, norm, "anomaly", med_d, scalebar=(k == 1))
        for t in list(ax.texts)[:2]:        # the Fig 4 tag and read-out: their content moves to the header / caption
            t.remove()
        for t in ax.texts:                  # the 200 m scale label is the only text left
            t.set_fontsize(FS_KEY)
        tile_axes.append(ax)
        facts = tile_facts(tid, tiles, data["lcz_arr"], data["lcz_tr"], d["met"])
        st.update(place=place, **facts)
        tile_st.append(st)

        # header: place + LCZ share, then the share bar resting on the tile
        tx(fig, x, Y_HEAD, place, FS_PLACE, weight="bold")
        tx(fig, x + TILE, Y_HEAD, f"LCZ {LCZ} on {100 * st['lcz_share']:.0f} % of tile", FS_SMALL,
           P["orange_dark"], "bold", ha="right")
        rect(fig, x, Y_TOP, TILE, BAR_H, P["orange_light"], zorder=2)
        rect(fig, x, Y_TOP, TILE * st["lcz_share"], BAR_H, S.CLASS_COLORS[LCZ], zorder=3)

        # computed facts under the tile
        for j, (val, lab) in enumerate([(f"{st['med_h']:.0f} m", "median building height"),
                                        (f"{st['med_svf']:.2f}", "median sky view factor")]):
            y = Y_TILE - 0.235 - j * 0.255
            tx(fig, x, y, val, FS_SMALL, weight="bold")
            tx(fig, x + 0.60, y, lab, FS_SMALL, P["muted"])

    # ---- key under the flat tile --------------------------------------------------------------
    any_below = any(t.get("share_below_anom", 0) > 0 for t in tile_st)
    any_above = any(t.get("share_above_anom", 0) > 0 for t in tile_st)
    cb = draw_colour_key(fig, ML + 0.02, 1.12, FLAT_W - 0.04, any_below, any_above)
    tx(fig, ML, 0.73, "ΔUTCI vs day median, °C", FS_KEY)
    tx(fig, ML, 0.52, "breezy day, 15:00", FS_KEY, P["muted"])
    rect(fig, ML, 0.305, 0.16, 0.13, P["taupe"], zorder=2)
    tx(fig, ML + 0.23, 0.31, "buildings", FS_KEY, P["muted"])

    # ---- right: three computed numbers -------------------------------------------------------
    tx(fig, X_R, Y_HEAD, f"{d['n_segments']:,} segments, breezy day", FS_SMALL, P["muted"])
    y1, y2, y3 = 3.40, 2.38, 0.85
    tx(fig, X_R, y1, f"{d['r2_lcz']:.2f}", FS_NUM, P["orange_dark"], "bold")
    tx(fig, X_R, y1 - 0.30, "CV R\u00b2, LCZ label alone", FS_LABEL)
    tx(fig, X_R, y2, f"{d['r2_morph']:.2f}", FS_NUM, weight="bold")
    tx(fig, X_R, y2 - 0.30, "CV R\u00b2, street morphology", FS_LABEL)
    rect(fig, X_R, 1.60, FW - ML - X_R, 0.012, P["taupe"], zorder=2)
    x2 = X_R + 1.24
    tx(fig, X_R, y3, f"{med_d:.1f}", FS_NUM_DAY, F4.DAY_TAG_COLOR[DAY], "bold")
    tx(fig, x2, y3, f"{med_c:.1f} \u00b0C", FS_NUM_DAY, F4.DAY_TAG_COLOR[CALM], "bold")
    tx(fig, X_R, y3 - 0.27, "breezy day", FS_LABEL, F4.DAY_TAG_COLOR[DAY])
    tx(fig, x2, y3 - 0.27, "calm day", FS_LABEL, F4.DAY_TAG_COLOR[CALM])
    tx(fig, X_R, y3 - 0.54, "median street UTCI", FS_LABEL)

    meta = dict(fig_size_in=[FW, FH], tile_in=TILE, anom_limit=F4.ANOM_LIM, tiles=tile_st,
                n_segments=d["n_segments"], median_utci={DAY: med_d, CALM: med_c},
                cv_r2={k: {"lcz_only": days[k]["r2_lcz"], "lcz_only_ci95": days[k]["r2_lcz_ci"],
                           "morphology": days[k]["r2_morph"], "morphology_ci95": days[k]["r2_morph_ci"]}
                       for k in (DAY, CALM)})
    return fig, meta, dict(tile_axes=tile_axes, cbar=cb)


# =============================================================================================
# QA: overlaps, clipping, type floor, text on the maps
# =============================================================================================
def qa(fig, h) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip():
            items.append((t, t.get_window_extent(r)))
    fbb = fig.bbox
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a.overlaps(b):
                ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 1.0:
                    overlaps.append([items[i][0].get_text()[:30], items[j][0].get_text()[:30], round(ov, 1)])
    clipped = [t.get_text()[:30] for t, bb in items if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fbb.x1 or bb.y1 > fbb.y1]
    margin_px = 0.16 * fig.dpi
    near_edge = [t.get_text()[:30] for t, bb in items if bb.x0 < margin_px or bb.y0 < margin_px * 0.6
                 or bb.x1 > fbb.x1 - margin_px or bb.y1 > fbb.y1 - margin_px * 0.6]
    on_tiles = []
    for ax in h["tile_axes"]:
        abb = ax.get_window_extent(r)
        for t, bb in items:
            if t.axes is not ax and bb.overlaps(abb):
                on_tiles.append(t.get_text()[:30])
    sizes = sorted({round(t.get_fontsize(), 2) for t, _ in items})
    return dict(n_texts=len(items), text_text_overlaps=overlaps, clipped_outside_sheet=clipped, near_sheet_edge=near_edge,
                figure_text_on_tiles=on_tiles, author_pt_sizes=sizes, smallest_author_pt=sizes[0],
                below_floor=[t.get_text()[:30] for t, _ in items if t.get_fontsize() < MIN_PT],
                smallest_px_em_at_1328=round(sizes[0] / 72.0 * QA_THUMB_W / FW, 1),
                texts=sorted({t.get_text() for t, _ in items}))


def load_context() -> dict:
    tiles = gpd.read_file(ROOT / "data/street/grid/tiles_core.gpkg").set_index("tile_id")
    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    xmin, ymin, xmax, ymax = study.total_bounds
    pad = 0.015                            # the Fig 1 grid, so the shares equal the Fig 1 tags
    ext = (xmin - (xmax - xmin) * pad, ymin - (ymax - ymin) * pad, xmax + (xmax - xmin) * pad,
           ymax + (ymax - ymin) * pad)
    lcz_arr, lcz_tr = F1.load_lcz_5253(ext, 50.0)
    return dict(tiles=tiles, lcz_arr=lcz_arr, lcz_tr=lcz_tr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=None, help="directory for the QA json and 1328 px downsample")
    args = ap.parse_args(argv)
    days = load_numbers()
    fig, meta, h = build(days, load_context())
    report = qa(fig, h)
    S.save_figure(fig, OUT, crop=None, close=False)
    plt.close(fig)
    if args.qa:
        args.qa.mkdir(parents=True, exist_ok=True)
        (args.qa / "ga_qa.json").write_text(json.dumps(dict(meta=meta, qa=report), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        with Image.open(OUT) as im:
            im.convert("RGB").resize((QA_THUMB_W, round(im.height * QA_THUMB_W / im.width)), Image.LANCZOS).save(
                args.qa / "ga_1328px.png")
    print(json.dumps(dict(meta={k: v for k, v in meta.items() if k != "tiles"},
                          tiles=[{k: v for k, v in t.items() if k != "lcz_hist"} for t in meta["tiles"]],
                          qa={k: v for k, v in report.items() if k != "texts"}), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
