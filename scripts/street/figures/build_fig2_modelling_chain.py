"""CLIMORFA v2 Figure 2: modelling chain and analytical design (branching flowchart, top-down).

House language (taken from the ORIGINAL paper):
  * rcParams via climorfa_style.apply_style()  (seaborn paper x1.05 + Fig 1 rc update, DejaVu Sans)
  * node grammar of the original graphical abstract / Fig 2 lanes:
      FancyBboxPatch "round,pad=0.04", light family fill + root-colour edge lw 1.4,
      bold titles, regular body, data units = inches
  * solid dark-variant fill + white text for the primary node (graphical-abstract "LightGBM (Primary)")
  * arrows: arrowstyle "->", lw 1.8, taupe_dark (graphical abstract)
  * boxed bold panel letters (a)-(d) above the stage rows (Fig 2 _panel_label)
  * family colours of Fig 2 (FAMILY_COLORS): teal terrain/buildings, green vegetation, orange forcing,
    blue network (used here for wind), pink climate response, taupe gate/units

v2 print scale (author decision 2026-10-30): canvas is exactly 13.0 in wide; every text is authored in v1 point
sizes and then passed through climorfa_style.scale_text(fig) (x1.4, floor 10.4 pt = 6 pt at 190 mm).
Layout (all positions are 'd' = distance from the top edge, in inches) was reworked to a top-down flow so that
the enlarged type fits the 13 in canvas.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib as mpl  # noqa: E402
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.street.figures import climorfa_style as S  # noqa: E402

S.apply_style()

OUT_PNG = ROOT / "paper/figures/v2/main/fig_2_modelling_chain.png"
QA_DIR = ROOT / "outputs/street/figures/fig2_qa"

P = S.PALETTE
INK, MUTED = P["ink"], P["muted"]
ARROW = P["taupe_dark"]


def mix(c1: str, c2: str, f: float) -> str:
    a, b = np.array(mpl.colors.to_rgb(c1)), np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex(a * (1 - f) + b * f)


def family(base: str) -> dict:
    return dict(base=base, light=mix(base, "#FFFFFF", 0.80), dark=mix(base, INK, 0.42))


FAM = {
    "teal": dict(base=P["teal"], light=P["teal_light"], dark=P["teal_dark"]),
    "green": family(S.FAMILY_COLORS_FIG2["Vegetation"]),
    "orange": dict(base=P["orange"], light=P["orange_light"], dark=P["orange_dark"]),
    "blue": family(S.FAMILY_COLORS_FIG2["Network"]),
    "pink": dict(base=P["pink"], light=P["pink_light"], dark=P["pink_dark"]),
    "taupe": dict(base=P["taupe"], light=P["taupe_light"], dark=P["taupe_dark"]),
}

# ----------------------------------------------------------------------------- vertical layout (d = distance from top, in)
SRC_H, GAP_H, PRD_H, DD_H, TOP_IN = 0.56, 0.28, 0.84, 1.52, 0.74
PROD_BOTTOM = TOP_IN + SRC_H + GAP_H + PRD_H
DD_BOTTOM = TOP_IN + SRC_H + GAP_H + DD_H
HUB_D, HUB_H = PROD_BOTTOM + 0.34, 0.60
HUB_BOTTOM = HUB_D + HUB_H
ENG_D, ENG_H = HUB_BOTTOM + 0.66, 2.12
ENG_BOT = ENG_D + ENG_H
BUS_D = ENG_BOT + 0.34
ROW_D, ROW_H = BUS_D + 0.30, 2.76
ROW_MID = ROW_D + ROW_H / 2
RAIL_D = ROW_D + ROW_H + 0.26
HH_, ROW_GAP = 1.34, 0.40
H_TOP = ROW_D + ROW_H + 0.66
H_BOTTOM = H_TOP + 2 * HH_ + ROW_GAP

# ----------------------------------------------------------------------------- canvas
W = S.V2_FIG_WIDTH_IN        # 13.0 in
FH = round(ROW_D + ROW_H + 0.10, 2)  # figure height, in
                   # figure height, in (tuned below)
fig = plt.figure(figsize=(W, FH), constrained_layout=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, FH)
ax.axis("off")


def Y(d: float) -> float:
    """distance from the top -> data y"""
    return FH - d


TEXTS: list[tuple] = []   # (artist, container (x0,y0,x1,y1) or None, tag)

# authored (v1) sizes; scale_text() multiplies by 1.4 (floor 10.4 pt)
FS_HEAD = 10.5
FS_BODY = 9.3
FS_SMALL = 8.8
FS_STAGE = 10.0


def T(x, d, s, size=FS_BODY, weight="normal", color=INK, ha="left", va="center", box=None, tag="", **kw):
    t = ax.text(x, Y(d), s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=20, **kw)
    TEXTS.append((t, box, tag or s))
    return t


def cbox(x, d, w, h):
    """container tuple in data coordinates for a card with top at d"""
    return (x, Y(d + h), x + w, Y(d))


def box(x, d, w, h, fc, ec, lw=1.4, ls="-", z=5, pad=0.04):
    p = mp.FancyBboxPatch((x, Y(d + h)), w, h, boxstyle=f"round,pad={pad}", facecolor=fc, edgecolor=ec,
                          linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


HH = 0.50   # header strip height


def card(x, d, w, h, fam, header=None, ls="-", z=5, lw=1.4):
    """Lane-style card: light family fill, family-coloured edge; optional dark header strip."""
    f = FAM[fam]
    box(x, d, w, h, f["light"], f["base"] if not header else f["dark"], lw=lw, ls=ls, z=z)
    if header:
        box(x, d, w, HH, f["dark"], f["dark"], lw=lw, z=z + 1)
        ax.add_patch(mp.Rectangle((x - 0.04, Y(d + HH)), w + 0.08, HH / 2,
                                  facecolor=f["dark"], edgecolor=f["dark"], linewidth=lw, zorder=z + 1))
        T(x + w / 2, d + HH / 2 + 0.01, header, FS_HEAD, "bold", "white", "center",
          box=cbox(x, d, w, HH), tag="hdr:" + header)


def source_box(x, d, w, h, fam, lines):
    f = FAM[fam]
    box(x, d, w, h, "white", f["base"], lw=1.1, z=5)
    n = len(lines)
    for i, ln in enumerate(lines):
        T(x + w / 2, d + h / 2 - (n - 1) * 0.135 + i * 0.27, ln, FS_BODY, "normal", INK, "center", box=cbox(x, d, w, h))


def arrow(pts_d, head=True, lw=1.8, color=ARROW, z=8, ls="-"):
    """Polyline given in (x, d) with an optional '->' head on the last segment."""
    pts = [(x, Y(d)) for x, d in pts_d]
    body = pts[:-1] if head else pts
    if len(body) >= 2:
        xs, ys = zip(*body)
        ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt", solid_joinstyle="miter", ls=ls)
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=z, solid_capstyle="butt", ls=ls)
        ax.annotate("", xy=(x1, y1), xytext=(x0 + (x1 - x0) * 0.6, y0 + (y1 - y0) * 0.6),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=0, shrinkB=0,
                                    mutation_scale=15), zorder=z + 1)


def dot(x, d, r=0.058):
    ax.add_patch(mp.Circle((x, Y(d)), r, facecolor=ARROW, edgecolor="none", zorder=12))


def chip(x, d, w, h, fam, text, solid=False, size=FS_SMALL, weight="normal", lines=None):
    f = FAM[fam]
    fc = f["dark"] if solid else f["light"]
    box(x, d, w, h, fc, f["base"] if not solid else f["dark"], lw=1.0, z=6, pad=0.03)
    cb = cbox(x, d, w, h)
    if lines:
        T(x + 0.09, d + h * 0.30, lines[0], size, "normal", INK, "left", box=cb)
        T(x + 0.09, d + h * 0.74, lines[1], size - 0.4, "normal", MUTED, "left", box=cb)
    else:
        T(x + 0.09, d + h / 2, text, size, "bold" if solid else weight, "white" if solid else INK, "left", box=cb)


# ----------------------------------------------------------------------------- icons (data-y based, centre cx, cy)
def icon_dtm(cx, d):
    cy = Y(d)
    xs = np.linspace(cx - 0.27, cx + 0.27, 60)
    for k, off in enumerate((0.11, 0.0, -0.11)):
        ax.plot(xs, cy + off + 0.045 * np.sin((xs - cx) * 11 + k * 0.9), color=FAM["teal"]["dark"], lw=1.2,
                zorder=9, solid_capstyle="round")


def icon_buildings(cx, d):
    cy = Y(d)
    for dx, bw, bh in ((-0.2, 0.15, 0.22), (-0.03, 0.15, 0.40), (0.14, 0.15, 0.30)):
        ax.add_patch(mp.Rectangle((cx + dx, cy - 0.2), bw, bh, facecolor=FAM["teal"]["base"],
                                  edgecolor=FAM["teal"]["dark"], linewidth=0.9, zorder=9))


def icon_trees(cx, d):
    cy = Y(d)
    for dx, r in ((-0.15, 0.115), (0.08, 0.15)):
        ax.plot([cx + dx, cx + dx], [cy - 0.2, cy - 0.02], color=FAM["green"]["dark"], lw=1.4, zorder=9)
        ax.add_patch(mp.Circle((cx + dx, cy + 0.06), r, facecolor=FAM["green"]["base"],
                               edgecolor=FAM["green"]["dark"], linewidth=0.9, zorder=10))


def icon_sun(cx, d, r=0.085):
    cy = Y(d)
    ax.add_patch(mp.Circle((cx, cy), r, facecolor=FAM["orange"]["base"], edgecolor=FAM["orange"]["dark"],
                           linewidth=0.9, zorder=10))
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([cx + np.cos(a) * (r + 0.04), cx + np.cos(a) * (r + 0.115)],
                [cy + np.sin(a) * (r + 0.04), cy + np.sin(a) * (r + 0.115)],
                color=FAM["orange"]["dark"], lw=1.1, zorder=9, solid_capstyle="round")


def icon_grid(cx, d):
    cy = Y(d)
    s = 0.115
    for i in range(3):
        for j in range(3):
            ax.add_patch(mp.Rectangle((cx - 1.5 * s + i * s, cy - 1.5 * s + j * s), s, s, facecolor="white",
                                      edgecolor=FAM["taupe"]["dark"], linewidth=0.8, zorder=9))


def icon_lock(cx, d, col, k=1.25):
    cy = Y(d)
    ax.add_patch(mp.Arc((cx, cy + 0.07 * k), 0.2 * k, 0.26 * k, theta1=0, theta2=180, edgecolor=col, linewidth=2.2, zorder=10))
    ax.plot([cx - 0.1 * k, cx - 0.1 * k], [cy + 0.07 * k, cy - 0.01 * k], color=col, lw=2.2, zorder=10)
    ax.plot([cx + 0.1 * k, cx + 0.1 * k], [cy + 0.07 * k, cy - 0.01 * k], color=col, lw=2.2, zorder=10)
    ax.add_patch(mp.FancyBboxPatch((cx - 0.17 * k, cy - 0.2 * k), 0.34 * k, 0.21 * k,
                                   boxstyle="round,pad=0.0,rounding_size=0.04",
                                   facecolor=col, edgecolor=col, linewidth=0.8, zorder=11))
    ax.add_patch(mp.Circle((cx, cy - 0.095 * k), 0.03 * k, facecolor="white", edgecolor="none", zorder=12))


# ----------------------------------------------------------------------------- grid
X0, CW, G = 0.25, 2.85, 0.37
CX = [X0 + i * (CW + G) for i in range(4)]
CC = [x + CW / 2 for x in CX]          # column centres


# ============================================================================ stage headers
def stage_header(x, d, letter, name):
    lt = ax.text(x + 0.02, Y(d), f"({letter.lower()})", ha="left", va="center", fontsize=15, fontweight="bold",
                 color=S.PANEL_TEXT, bbox=dict(S.PANEL_LABEL_BBOX), zorder=30, clip_on=False)
    nt = T(x + 0.95, d, name, FS_STAGE, "bold", INK, "left", tag="stage:" + name)
    return lt, nt


HDR_D = {"A": 0.34, "B": (HUB_BOTTOM + ENG_D) / 2, "C": ROW_D}
STAGES = {"A": "INPUTS AND GEOMETRY", "B": "PHYSICS CHAIN", "C": "AGGREGATION"}
hdr_art = {k: stage_header(0.27, HDR_D[k], k, STAGES[k]) for k in "ABC"}
S.scale_text(fig)          # scale what exists so far so headers can be measured
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_inv = ax.transData.inverted()


def data_bbox_of(art):
    bb = art.get_window_extent(_rend)
    (x0, y0), (x1, y1) = _inv.transform((bb.x0, bb.y0)), _inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1


# place the name text right after the letter box (letter box patch extent) and measure the header end
hdr_end = {}
for k in "ABC":
    lt, nt = hdr_art[k]
    pb = lt.get_bbox_patch().get_window_extent(_rend)
    px1 = _inv.transform((pb.x1, pb.y1))[0]
    nt.set_x(px1 + 0.12)
    fig.canvas.draw()
    hdr_end[k] = data_bbox_of(nt)[2]
print("header ends:", {k: round(v, 2) for k, v in hdr_end.items()})

# ============================================================================ (a) inputs
rows = [  # col, family, source lines, product title, product sub, icon
    (0, "teal", ["5 m surface model"], "DTM v3", "bare earth, 2 m", icon_dtm),
    (1, "teal", ["Cadastral buildings", "+ storey QA"], "2 m building DSM", "3.0 m per storey", icon_buildings),
    (2, "green", ["1 m canopy map"], "2 m tree DSM", "canopy height, 2 m", icon_trees),
]
prod_bottom = PROD_BOTTOM
for ci, fam, src, title, sub, icon in rows:
    x = CX[ci]
    source_box(x, TOP_IN, CW, SRC_H, fam, src)
    arrow([(CC[ci], TOP_IN + SRC_H + 0.01), (CC[ci], TOP_IN + SRC_H + GAP_H - 0.01)], lw=1.5)
    pd_ = TOP_IN + SRC_H + GAP_H
    card(x, pd_, CW, PRD_H, fam)
    T(x + 0.12, pd_ + 0.25, title, FS_HEAD - 0.3, "bold", FAM[fam]["dark"], box=cbox(x, pd_, CW, PRD_H))
    T(x + 0.12, pd_ + PRD_H - 0.22, sub, FS_SMALL, "normal", MUTED, box=cbox(x, pd_, CW, PRD_H))
    icon(x + CW - 0.40, pd_ + PRD_H / 2)

# design days column
x = CX[3]
source_box(x, TOP_IN, CW, SRC_H, "orange", ["METAR + ERA5"])
arrow([(CC[3], TOP_IN + SRC_H + 0.01), (CC[3], TOP_IN + SRC_H + GAP_H - 0.01)], lw=1.5)
pd_ = TOP_IN + SRC_H + GAP_H
card(x, pd_, CW, DD_H, "orange")
T(x + 0.12, pd_ + 0.25, "Two design days", FS_HEAD - 0.3, "bold", FAM["orange"]["dark"], box=cbox(x, pd_, CW, DD_H))
icon_sun(x + CW - 0.32, pd_ + 0.25)
chip(x + 0.10, pd_ + 0.52, CW - 0.20, 0.42, "orange", "Breezy day, 2022-07-05")
chip(x + 0.10, pd_ + 1.00, CW - 0.20, 0.42, "orange", "Calm day, 2023-07-26")
dd_bottom = DD_BOTTOM

# hub: common grid (spans columns 1-3)
HUB_X, HUB_W = CX[0], 8.55 - CX[0]
card(HUB_X, HUB_D, HUB_W, HUB_H, "taupe")
hbox = cbox(HUB_X, HUB_D, HUB_W, HUB_H)
icon_grid(HUB_X + 0.45, HUB_D + HUB_H / 2)
T(HUB_X + 0.90, HUB_D + HUB_H / 2, "Common 2 m grid", FS_HEAD - 0.3, "bold", FAM["taupe"]["dark"], box=hbox)
T(HUB_X + 3.25, HUB_D + HUB_H / 2, "1 km tiles + 100 m buffer, 427 tiles", FS_SMALL, "normal", MUTED, box=hbox)
for ci in range(3):
    arrow([(CC[ci], prod_bottom + 0.02), (CC[ci], HUB_D - 0.03)])
hub_bottom = HUB_BOTTOM

# ============================================================================ (b) physics
ARR_D = ENG_D + 1.05     # horizontal arrows between engine cards

# W: wind screening (col 0)
card(CX[0], ENG_D, CW, ENG_H, "blue", header="Wind screening")
wb = cbox(CX[0], ENG_D, CW, ENG_H)
for k, ln in enumerate(["geometric, pedestrian level", "wake, canyon, corner terms", "trees as porous obstacles"]):
    T(CX[0] + 0.12, ENG_D + 0.76 + k * 0.27, ln, FS_SMALL, "normal", INK, box=wb)
T(CX[0] + 0.12, ENG_D + 1.62, "wind ratio r", FS_BODY + 0.2, "bold", FAM["blue"]["dark"], box=wb)
T(CX[0] + 0.12, ENG_D + 1.89, "local / open-terrain wind", FS_SMALL, "normal", MUTED, box=wb)

# Us: UTCI street wind (col 1)
card(CX[1], ENG_D, CW, ENG_H, "pink", header="UTCI, street wind")
ub = cbox(CX[1], ENG_D, CW, ENG_H)
T(CX[1] + 0.12, ENG_D + 0.76, r"from $T_{mrt}$, Ta, RH and", FS_SMALL, "normal", INK, box=ub)
T(CX[1] + 0.12, ENG_D + 1.04, r"$v_{10,\mathrm{eff}}$ = clip($U_{10}\,r$, 0.5, 17)", FS_SMALL - 0.4, "normal", INK, box=ub)
T(CX[1] + 0.12, ENG_D + 1.31, "street-level wind structure", FS_SMALL - 0.3, "normal", MUTED, box=ub)
chip(CX[1] + 0.10, ENG_D + 1.68, CW - 0.20, 0.34, "orange", "Ta, RH, U10: design day", size=FS_SMALL - 0.4)

# S: SOLWEIG-GPU (col 2)
card(CX[2], ENG_D, CW, ENG_H, "pink", header="SOLWEIG-GPU")
sb = cbox(CX[2], ENG_D, CW, ENG_H)
T(CX[2] + 0.12, ENG_D + 0.76, "shortwave + longwave fluxes", FS_SMALL, "normal", INK, box=sb)
T(CX[2] + 0.12, ENG_D + 1.02, "2 days × 16 daytime hours", FS_SMALL, "normal", INK, box=sb)
T(CX[2] + 0.12, ENG_D + 1.36, "outputs per 2 m cell", FS_SMALL, "normal", MUTED, box=sb)
T(CX[2] + 0.12, ENG_D + 1.64, r"$T_{mrt}$, sunlit fraction, SVF", FS_BODY - 0.3, "bold", FAM["pink"]["dark"], box=sb)
T(CX[2] + 0.12, ENG_D + 1.91, "(SVF: sky view factor)", FS_SMALL - 0.4, "normal", MUTED, box=sb)

# Uu: UTCI uniform wind (col 3)
card(CX[3], ENG_D, CW, ENG_H, "pink", header="UTCI, uniform wind")
ub2 = cbox(CX[3], ENG_D, CW, ENG_H)
T(CX[3] + 0.12, ENG_D + 0.78, r"$v$ = clip($U_{10}$, 0.5, 17)", FS_SMALL, "normal", INK, box=ub2)
T(CX[3] + 0.12, ENG_D + 1.07, "no street-wind structure", FS_SMALL - 0.3, "normal", MUTED, box=ub2)
chip(CX[3] + 0.10, ENG_D + 1.68, CW - 0.20, 0.34, "orange", "Ta, RH, U10: design day", size=FS_SMALL - 0.4)

# fork from the common grid (hub) to wind (col 0) and radiation (col 2)
wx = max(hdr_end["B"] + 0.28, CX[0] + 2.55)   # right part of the wind card, clear of the stage name
assert hdr_end["B"] + 0.2 < wx < CX[0] + CW - 0.1, (hdr_end["B"], wx)
arrow([(wx, hub_bottom + 0.02), (wx, ENG_D - 0.03)])
arrow([(CX[2] + 0.75, hub_bottom + 0.02), (CX[2] + 0.75, ENG_D - 0.03)])

# forcing: design days -> SOLWEIG-GPU (enters from the top, right part)
F_D = DD_BOTTOM + 0.45
sx = CX[2] + CW - 0.50
arrow([(CC[3], dd_bottom + 0.02), (CC[3], F_D), (sx, F_D), (sx, ENG_D - 0.03)])
T((CC[3] + sx) / 2 + 0.02, F_D - 0.19, "irradiance, Ta, RH, wind", FS_SMALL - 0.3, "normal", MUTED, "center", tag="rail label")

# arrows between engine cards (wind -> street UTCI ; SOLWEIG -> both UTCI)
arrow([(CX[0] + CW + 0.02, ARR_D), (CX[1] - 0.03, ARR_D)])
arrow([(CX[2] - 0.02, ARR_D), (CX[1] + CW + 0.03, ARR_D)])
arrow([(CX[2] + CW + 0.02, ARR_D), (CX[3] - 0.03, ARR_D)])

# ============================================================================ (c) aggregation

# merge bus: street UTCI, SOLWEIG bypass (Tmrt, sunlit, SVF), uniform UTCI -> pedestrian space
ps_x = max(hdr_end["C"] + 0.28, CX[0] + 2.55)
assert ps_x < CX[0] + CW - 0.1
arrow([(CC[1], ENG_BOT + 0.02), (CC[1], BUS_D)], head=False)
arrow([(CC[2], ENG_BOT + 0.02), (CC[2], BUS_D)], head=False)
arrow([(CC[3], ENG_BOT + 0.02), (CC[3], BUS_D), (ps_x, BUS_D)], head=False)
T(CC[2] + 0.12, (ENG_BOT + BUS_D) / 2, r"$T_{mrt}$, sunlit, SVF", FS_SMALL - 0.3, "normal", MUTED, "left", tag="bypass label")
dot(CC[1], BUS_D)
dot(CC[2], BUS_D)

# pedestrian space (col 0)
PS_H = 1.98
PS_D = ROW_MID - PS_H / 2
card(CX[0], PS_D, CW, PS_H, "taupe", header="Pedestrian space")
pb_ = cbox(CX[0], PS_D, CW, PS_H)
for k, ln in enumerate(["2 m land cells, not building,", "within 15 m of a road", "centreline"]):
    T(CX[0] + 0.12, PS_D + 0.78 + k * 0.27, ln, FS_SMALL, "normal", INK, box=pb_)
bx0, bx1 = CX[0] + 0.20, CX[0] + CW - 0.20
band_d = PS_D + 1.70
ax.add_patch(mp.Rectangle((bx0, Y(band_d) - 0.15), bx1 - bx0, 0.30, facecolor=FAM["taupe"]["base"], alpha=0.45,
                          edgecolor="none", zorder=7))
ax.plot([bx0, bx1], [Y(band_d), Y(band_d)], color=FAM["taupe"]["dark"], lw=2.0, zorder=9, solid_capstyle="round")
for xx in np.linspace(bx0 + 0.06, bx1 - 0.06, 14):
    for dy in (-0.09, 0.09):
        ax.add_patch(mp.Rectangle((xx - 0.025, Y(band_d) + dy - 0.025), 0.05, 0.05, facecolor="white",
                                  edgecolor=FAM["taupe"]["dark"], linewidth=0.5, zorder=9))
arrow([(ps_x, BUS_D), (ps_x, PS_D - 0.03)])

# segments (col 1)
SEG_H = 1.24
SEG_D = ROW_MID - SEG_H / 2
card(CX[1], SEG_D, CW, SEG_H, "taupe")
gb = cbox(CX[1], SEG_D, CW, SEG_H)
T(CX[1] + 0.12, SEG_D + 0.28, "74,178 street segments", FS_BODY, "bold", FAM["taupe"]["dark"], box=gb)
T(CX[1] + 0.12, SEG_D + 0.66, "split at junctions, tile edges", FS_SMALL, "normal", INK, box=gb)
T(CX[1] + 0.12, SEG_D + 0.97, "carry LCZ, LST, H/W, SVF", FS_SMALL, "normal", INK, box=gb)
arrow([(CX[0] + CW + 0.02, ROW_MID), (CX[1] - 0.03, ROW_MID)])

# metrics group (col 2)
box(CX[2], ROW_D, CW, ROW_H, FAM["pink"]["light"], FAM["pink"]["base"], lw=1.4, z=4)
mb = cbox(CX[2], ROW_D, CW, ROW_H)
T(CX[2] + 0.12, ROW_D + 0.22, "metrics, per segment and day", FS_SMALL - 0.3, "normal", MUTED, box=mb)
metrics = [("M1", "UTCI 15:00, street wind", True, 0.42),
           ("M2", "Heat-stress hours, 10–18 h", False, 0.42),
           ("M3", r"$T_{mrt}$ at 15:00", False, 0.42),
           ("M4", "Shade fraction, 10–18 h", False, 0.42),
           ("M5", "UTCI 15:00, uniform wind", False, 0.42)]
cd = ROW_D + 0.42
for m, txt, prim, hch in metrics:
    chip(CX[2] + 0.10, cd, CW - 0.20, hch, "pink", txt, solid=prim, size=FS_SMALL - 0.2)
    cd += hch + 0.06
arrow([(CX[1] + CW + 0.02, ROW_MID), (CX[2] - 0.03, ROW_MID)])

# scale ladder (col 3)
LD_H = 1.24
LD_D = ROW_MID - LD_H / 2
card(CX[3], LD_D, CW, LD_H, "taupe")
lb = cbox(CX[3], LD_D, CW, LD_H)
T(CX[3] + 0.12, LD_D + 0.27, "Scale ladder, m", FS_BODY, "bold", FAM["taupe"]["dark"], box=lb)
base_d = LD_D + 0.74
slots = [CX[3] + 0.32 + k * 0.45 for k in range(6)]
ax.plot([slots[0] - 0.14, slots[0] + 0.14], [Y(base_d) + 0.04, Y(base_d) + 0.04], color=FAM["taupe"]["dark"], lw=3.0,
        zorder=9, solid_capstyle="round")
for k, sz in zip(range(1, 5), (0.09, 0.15, 0.23, 0.32)):
    ax.add_patch(mp.Rectangle((slots[k] - sz / 2, Y(base_d)), sz, sz, facecolor=FAM["taupe"]["light"],
                              edgecolor=FAM["taupe"]["dark"], linewidth=1.0, zorder=9))
poly = np.array([[-0.17, 0.0], [0.17, 0.02], [0.2, 0.24], [0.04, 0.34], [-0.14, 0.27]]) + [slots[5] + 0.03, Y(base_d)]
ax.add_patch(mp.Polygon(poly, closed=True, facecolor=FAM["teal"]["light"], edgecolor=FAM["teal"]["dark"],
                        linewidth=1.0, zorder=9))
for xx, lab in zip(slots, ["seg.", "50", "100", "250", "500", "LCZ"]):
    T(xx, base_d + 0.27, lab, FS_SMALL - 0.4, "normal", INK, "center", box=lb)
arrow([(CX[2] + CW + 0.02, ROW_MID), (CX[3] - 0.03, ROW_MID)])

# ============================================================================ legend (top right, beside stage A)
LEG_ROWS = [
    [("teal", "terrain and buildings", 4.25), ("green", "trees", 6.65), ("orange", "weather forcing", 8.05), ("blue", "wind", 10.3)],
    [("pink", "heat exposure and stress", 4.25), ("taupe", "units and metrics", 7.0)],
]
for r, row in enumerate(LEG_ROWS):
    d = 0.14 + r * 0.36
    for fam, lab, x in row:
        ax.add_patch(mp.FancyBboxPatch((x, Y(d) - 0.075), 0.26, 0.15, boxstyle="round,pad=0.02", facecolor=FAM[fam]["light"],
                                       edgecolor=FAM[fam]["base"], linewidth=1.2, zorder=6))
        T(x + 0.38, d, lab, FS_SMALL - 0.3, "normal", INK, tag="leg:" + lab)
d = 0.14 + 0.36
arrow([(9.8, d), (10.1, d)], lw=1.5)
T(10.18, d, "data flow", FS_SMALL - 0.3, "normal", INK, tag="leg:flow")

S.finalize_panel_labels(fig)
nscaled = S.scale_text(fig)        # idempotent: scales the texts created after the header pass

# ============================================================================ QA + save
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def data_bbox(t):
    bb = t.get_window_extent(rend)
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1


issues, bbs, sizes = [], [], []
for t, cont, tag in TEXTS:
    b = data_bbox(t)
    bbs.append((tag, b))
    sizes.append(t.get_fontsize())
    if cont is not None:
        pad = 0.06
        if b[0] < cont[0] + pad or b[2] > cont[2] - pad or b[1] < cont[1] + 0.02 or b[3] > cont[3] - 0.02:
            issues.append(f"OUTSIDE container: '{tag}' text=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}) box={tuple(round(v, 2) for v in cont)}")
for i in range(len(bbs)):
    for j in range(i + 1, len(bbs)):
        a, b = bbs[i][1], bbs[j][1]
        if a[0] < b[2] - 0.01 and b[0] < a[2] - 0.01 and a[1] < b[3] - 0.01 and b[1] < a[3] - 0.01:
            issues.append(f"TEXT OVERLAP: '{bbs[i][0]}' x '{bbs[j][0]}'")
# stage letter patch vs its name
for k in "ABC":
    lt, nt = hdr_art[k]
    pb = lt.get_bbox_patch().get_window_extent(rend)
    nb = nt.get_window_extent(rend)
    if pb.x1 > nb.x0:
        issues.append(f"stage {k}: letter box overlaps name")
min_pt = float(min(sizes))
print(f"scaled artists (last pass): {nscaled}; smallest authored pt = {min_pt:.2f}; printed at 190 mm = {min_pt * 7.48 / 13.0:.2f} pt")
QA_DIR.mkdir(parents=True, exist_ok=True)
(QA_DIR / "text_qa.json").write_text(json.dumps({"fig_w": W, "fig_h": FH, "issues": issues, "n_texts": len(TEXTS),
                                                 "min_authored_pt": min_pt,
                                                 "min_printed_pt_190mm": min_pt * 7.48 / 13.0}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
print("issues:", len(issues))
for s in issues:
    print("  ", s)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
S.save_figure(fig, OUT_PNG, crop="all")
print("saved", OUT_PNG)
