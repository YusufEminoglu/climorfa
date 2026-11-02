"""CLIMORFA v2 Figure 6: H2 + H4, what drives the street-to-street differences in M1 (WP4.5).

Final run:   C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig6_drivers
             (or ``build_fig5_h1_scale --all`` for both figures)  -> paper/figures/v2/main/fig_6_drivers.png
Synthetic:   ... build_fig6_drivers --results outputs/street/SYNTHETIC/fig56/analysis --subdir synthetic_H1T
             -> outputs/street/SYNTHETIC/fig56/fig_6_drivers_SYNTHETIC_H1T.png (stamped; never paper/)

Inputs (exactly the files of the real run, see build_fig5_h1_scale.ResultSet):
  h2/<sub>/h2_results*.json                 model A TreeSHAP: feature_mean_abs_shap, family_share (+ CI), CV R2
  h4/<sub>/h4_results*.json                 Var(shade), Var(breeze), ratio (+ CI), forcing (sun az 14:30, wind ref,
                                            n reports, R), mechanism, decision
  h4/<sub>/h4_orientation_x_hw_table*.csv   orientation class x H/W class (terciles + 'no enclosing buildings', A8-2)
  h4/<sub>/seg_h4_terms_<DAY>*.parquet      per-segment m1, azimuth_deg, block (panel b)

Panels: (a) TreeSHAP family shares (capsules, 95 % block-bootstrap CI) and mean |SHAP| per feature, both days;
(b) within-day M1 anomaly by street-axis azimuth (axial, 15 deg bins, mirrored), with sun azimuth at 14:30 (A8-1) and
the 250-340 deg sea-breeze sector in the Fig 1 sector style; the band is a 2 km block-bootstrap 95 % CI of the bin
median (descriptive, computed here with the lock bootstrap settings); (c) Var(shade term) and Var(breeze term) per day
against the 4x rule, and the mean shade / breeze terms in the orientation x H/W table; (d) computed H4 card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.street.figures import build_fig5_h1_scale as F5  # noqa: E402  (shared helpers, day encoding, QA)
from scripts.street.figures.build_fig5_h1_scale import (  # noqa: E402
    CHECK, CROSS, DAYS, DAY_STYLE, MINUS, P, S, blend, card_frame, cm, ctext, num, sgn)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Feature families: Fig 2 v2 family colours (terrain/buildings teal, trees green, forcing/sun orange, wind blue,
# units/context taupe), S.FAMILY_COLORS_FIG2.  Fig 6 shows no LCZ class, so no class colour is reused for a class.
FAMILY = {
    "shade_geometry": ("Shade / geometry", P["teal"]),
    "vegetation": ("Vegetation", S.FAMILY_COLORS_FIG2["Vegetation"]),
    "orientation": ("Orientation", P["orange"]),
    "ventilation": ("Ventilation", S.FAMILY_COLORS_FIG2["Network"]),
    "context": ("Context", P["taupe"]),
}
FEATURE_LABEL = {
    "svf_med": "Sky view factor", "hw_ratio": "H/W", "bh_mean30": "Building height (30 m)",
    "bcover30": "Building cover (30 m)", "tcover15": "Tree cover (15 m)", "tht15": "Canopy height (15 m)",
    "az_sin": "Street axis, sin", "az_cos": "Street axis, cos", "ang_sun": "Angle to sun (14:30)",
    "ang_wind": "Angle to wind", "ratio_med_b15": "Wind ratio", "dist_coast_m": "Distance to coast",
    "elev_m": "Elevation", "slope_deg": "Slope",
}
SHADE_C, BREEZE_C = FAMILY["shade_geometry"][1], FAMILY["ventilation"][1]
SUN_C = P["orange_dark"]
SB_LO, SB_HI = 250.0, 340.0          # sea-breeze sector, as Fig 1 (b)
AZ_BIN = 15.0                         # axial bins of 15 deg (12 bins over 0-180)
N_BOOT_POLAR = cm.N_BOOT              # lock: 1000 resamples of 2 km blocks
ORIENT = ["aligned", "oblique", "perpendicular"]
HW = ["low", "mid", "high", "no enclosing buildings"]
HW_LABEL = {"low": "low", "mid": "mid", "high": "high", "no enclosing buildings": "no encl.\nbldgs"}
CMAP_ANOM = mpl.colors.LinearSegmentedColormap.from_list(   # = Fig 4 v2 anomaly ramp (build_fig4_street_utci)
    "anom_teal_white_purple", [S.LST_TEAL_PURPLE_COLORS[0], "#F7F7F7", S.LST_TEAL_PURPLE_COLORS[-1]])


def nice_ceil(x: float, steps=(0.25, 0.5, 1.0, 2.0, 2.5, 5.0)) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(x))
    for s_ in steps:
        if s_ * mag >= x - 1e-12:
            return float(s_ * mag)
    return float(10 * mag)


def fmt_sig(x: float) -> str:
    """Variance-style numbers: 3 significant digits, typographic minus."""
    if not np.isfinite(x):
        return "\u221e"
    a = abs(x)
    nd = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3 if a >= 0.01 else 4
    return f"{x:.{nd}f}".replace("-", MINUS)


# =============================================================================================
# data
# =============================================================================================
def load(rs: F5.ResultSet) -> dict:
    h2 = rs.json("h2", "h2_results")
    h4 = rs.json("h4", "h4_results")
    tab = pd.read_csv(rs.file("h4", "h4_orientation_x_hw_table", "csv"))
    for d in (h2, h4):
        if set(d["results"]) != {day for day, _ in DAYS}:
            raise RuntimeError(f"results for {sorted(d['results'])} only; the figure needs both design days")
    segs = {day: rs.seg_terms(day, ["seg_id", "m1", "azimuth_deg", "block"]) for day, _ in DAYS}
    return dict(h2=h2, h4=h4, tab=tab, segs=segs)


def h4_verdict(h4: dict) -> tuple[bool, dict]:
    per = {day: bool(h4["results"][day]["var_shade"] > 4.0 * h4["results"][day]["var_breeze"]) for day, _ in DAYS}
    return all(per.values()) and len(per) == 2, per


def polar_bins(df: pd.DataFrame, rng_seed: int) -> dict:
    """Median within-day anomaly per 15 deg axial azimuth bin, 2 km block-bootstrap 95 % CI."""
    d = df.dropna(subset=["m1", "azimuth_deg"]).reset_index(drop=True)
    anom = (d["m1"] - d["m1"].median()).to_numpy(float)
    b = np.minimum((d["azimuth_deg"].to_numpy(float) % 180.0) // AZ_BIN, 180 / AZ_BIN - 1).astype(int)
    nb = int(180 / AZ_BIN)
    med = np.array([np.median(anom[b == k]) if (b == k).any() else np.nan for k in range(nb)])
    n = np.bincount(b, minlength=nb)
    ub, inv, counts = cm.boot_block_counts(d["block"].to_numpy(), N_BOOT_POLAR, rng_seed)
    order = np.lexsort((anom, b))                               # rows sorted by bin, then anomaly
    bs, av = b[order], anom[order]
    starts = np.searchsorted(bs, np.arange(nb)); ends = np.searchsorted(bs, np.arange(nb), side="right")
    boots = np.full((counts.shape[0], nb), np.nan)
    blk = inv[order]
    for r in range(counts.shape[0]):
        w = counts[r][blk].astype(float)
        for k in range(nb):
            ww = w[starts[k]:ends[k]]
            if ww.sum() <= 0:
                continue
            c = np.cumsum(ww)
            boots[r, k] = av[starts[k]:ends[k]][np.searchsorted(c, 0.5 * c[-1])]   # weighted median (sorted)
    lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    centers = (np.arange(nb) + 0.5) * AZ_BIN
    return dict(centers=centers, med=med, lo=lo, hi=hi, n=n)


# =============================================================================================
# panels
# =============================================================================================
def draw_families(ax, h2):
    fams = list(FAMILY)
    share = {day: h2["results"][day]["model_A_features_only"]["family_share"] for day, _ in DAYS}
    order = sorted(fams, key=lambda f: np.mean([share[d][f]["share"] for d, _ in DAYS]))
    off = {"IMBAT": 0.19, "CALM": -0.19}
    xmax = 0.0
    xref = max(share[d][f]["ci95"][1] for d, _ in DAYS for f in fams)
    for yi, f in enumerate(order):
        col = FAMILY[f][1]
        for day, _ in DAYS:
            v, (lo, hi) = share[day][f]["share"], share[day][f]["ci95"]
            y = yi + off[day]
            ax.plot([0, v], [y, y], color=col, lw=9.0, solid_capstyle="round", zorder=3, alpha=0.94)
            if day == "CALM":                                     # open capsule = CALM (day encoding of Figs 5/6)
                ax.plot([0, v], [y, y], color="white", lw=6.0, solid_capstyle="round", zorder=3.2)
            ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="none", ecolor=P["muted"], elinewidth=1.1,
                        capsize=2.6, zorder=4)
            ax.text(hi + 0.045 * xref, y, f"{100 * v:.0f} %", va="center", ha="left", fontsize=7.4, color=P["ink"],
                    fontweight="bold", zorder=5)
            xmax = max(xmax, hi)
    ax.set_yticks(range(len(order)), [FAMILY[f][0] for f in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, xmax * 1.45)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Share of mean |SHAP|")
    _v1_fig6_axes(ax)
    return order


def _v1_fig6_axes(ax):
    """Original Fig 6 treatment: only the bottom spine, hairline x-grid, zero-length y ticks."""
    ax.grid(False)
    ax.grid(axis="x", color=P["grid"], linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_features(ax, h2):
    imp = {day: h2["results"][day]["model_A_features_only"]["feature_mean_abs_shap"] for day, _ in DAYS}
    feats = sorted(imp["IMBAT"], key=lambda f: np.mean([imp[d][f] for d, _ in DAYS]))
    for yi, f in enumerate(feats):
        col = FAMILY[cm.FAMILY_OF[f]][1]
        for day, _ in DAYS:
            yy = yi + (0.15 if day == "IMBAT" else -0.15)
            ax.plot([0, imp[day][f]], [yy, yy], color=col, lw=1.4, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.scatter([imp[day][f]], [yy], s=40, zorder=4 if day == "IMBAT" else 5,
                       facecolor=col if day == "IMBAT" else "white", edgecolor=col if day == "CALM" else "white",
                       linewidth=1.4 if day == "CALM" else 1.0)
    ax.set_yticks(range(len(feats)), [FEATURE_LABEL.get(f, f) for f in feats])
    ax.set_ylim(-0.6, len(feats) - 0.4)
    ax.set_xlim(0, max(max(v.values()) for v in imp.values()) * 1.12)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_xlabel("Mean |SHAP| (\u00b0C)")
    _v1_fig6_axes(ax)
    h = [Line2D([0], [0], marker="o", ls="none", ms=7, mfc=P["muted"], mec="white", label="IMBAT"),
         Line2D([0], [0], marker="o", ls="none", ms=7, mfc="white", mec=P["muted"], mew=1.4, label="CALM")]
    ax.legend(handles=h, loc="lower right", fontsize=6.6, frameon=True, ncol=1, handletextpad=0.35, borderpad=0.5,
              labelspacing=0.35)
    r = {day: h2["results"][day]["model_A_features_only"] for day, _ in DAYS}
    return feats


def rlabel_angle(bins: dict, ticks, h4) -> float:
    """Angle for the ring labels: the direction (5 deg steps) where the plotted curves stay farthest from every
    ring value, avoiding the N/E/S/W labels, the sun-azimuth lines and the sector label (Fig 1 ring-label logic)."""
    cand = np.arange(0.0, 360.0, 5.0)
    bad = [0, 90, 180, 270] + [h4["results"][d]["forcing"]["sun_az"] % 180 + k for d, _ in DAYS for k in (0, 180)]
    best, best_s = 135.0, -1.0
    for a in cand:
        if min(min(abs((a - b + 180) % 360 - 180) for b in bad), 99) < 14 or SB_LO - 5 <= a <= SB_HI + 5:
            continue
        sc = []
        for b in bins.values():
            th = np.r_[b["centers"], b["centers"] + 180.0, b["centers"][0] + 360.0]
            for arr in (b["med"], b["lo"], b["hi"]):
                r = np.interp(a, th, np.r_[arr, arr, arr[0]])
                sc += [abs(r - t) for t in ticks]
        s_ = min(sc)
        if s_ > best_s:
            best, best_s = float(a), s_
    return best


def rlabel_angle_rim(h4) -> float:
    """Rim label direction: south-east (135 deg) unless a sun line passes within 15 deg, then 150 / 120 deg."""
    sun = [h4["results"][d]["forcing"]["sun_az"] % 180 for d, _ in DAYS]
    for a in (135.0, 150.0, 120.0, 105.0):
        if all(abs((a - s_ + 90) % 180 - 90) > 15 for s_ in sun):
            return a
    return 135.0


def draw_polar(ax, bins: dict, h4: dict):
    allv = np.concatenate([np.r_[b["lo"], b["hi"], b["med"]] for b in bins.values()])
    L = nice_ceil(1.08 * np.nanmax(np.abs(allv)))
    hole = 0.30
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(-L, L)
    ax.set_rorigin(-L - hole * 2 * L)
    ax.set_facecolor("white")
    # sea-breeze sector, Fig 1 (b) style: teal_light wedge, dashed teal_dark edges, label outside the rim
    mid, half = np.radians((SB_LO + SB_HI) / 2), np.radians((SB_HI - SB_LO) / 2)
    ax.bar(mid, 2 * L, width=2 * half, bottom=-L, color=P["teal_light"], edgecolor="none", alpha=0.95, zorder=0.5)
    for edge in (SB_LO, SB_HI):
        ax.plot([np.radians(edge)] * 2, [-L, L], color=P["teal_dark"], lw=0.7, ls=(0, (3, 2)), zorder=1.5)
    ax.text(mid, L + 0.36 * L, f"{SB_LO:g}\u2013{SB_HI:g}\u00b0", ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=P["teal_dark"], clip_on=False)
    th_full = np.linspace(0, 2 * np.pi, 361)
    ax.plot(th_full, np.zeros_like(th_full), color=S.SPINE_GREY, ls=":", lw=1.0, zorder=2)   # day median ring
    for day, _ in DAYS:
        b, st = bins[day], DAY_STYLE[day]
        th = np.radians(np.r_[b["centers"], b["centers"] + 180.0, b["centers"][0] + 360.0])
        cl = lambda a: np.r_[a, a, a[0]]  # noqa: E731  (axial: mirror 0-180 onto 180-360, close the loop)
        ax.fill_between(th, cl(b["lo"]), cl(b["hi"]), color=st["color"], alpha=0.16, linewidth=0, zorder=2.5)
        ax.plot(th, cl(b["med"]), color=st["color"], ls=st["ls"], lw=1.7, zorder=3)
        ax.scatter(th[:-1], cl(b["med"])[:-1], s=14, facecolor=st["mfc"], edgecolor=st["color"], linewidth=1.0,
                   zorder=4)
        sun = h4["results"][day]["forcing"]["sun_az"]
        for a in (sun, sun + 180.0):                          # a street axis parallel to the sun azimuth
            ax.plot([np.radians(a)] * 2, [-L, L], color=SUN_C, lw=1.2, ls=st["ls"], zorder=3.5)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8.5, color=P["ink"], fontweight="bold")
    ax.tick_params(axis="x", pad=-1)
    ticks = [-L, 0.0, L]
    ax.set_yticks(ticks)                                     # rings: inner edge = -L, dotted = day median, rim = +L
    ax.set_yticklabels([])
    nd = 2 if L < 1 else 1
    # ring values placed where no curve can run: inside the central hole (-L) and just outside the rim (+L)
    ax.text(0.0, -L - hole * 2 * L, f"{sgn(-L, nd)}°C", ha="center", va="center", fontsize=7.5, color=P["muted"],
            zorder=9)
    a_out = np.radians(rlabel_angle_rim(h4))
    ax.text(a_out, L * 1.07, f"{sgn(L, nd)}°C", ha="left", va="center", fontsize=7.5, color=P["muted"],
            zorder=9, clip_on=False)
    ax.grid(True, color=P["grid"], linewidth=0.6)
    ax.spines["polar"].set_color(P["grid"]); ax.spines["polar"].set_linewidth(0.8)
    r0 = -L - hole * 2 * L
    ax.bar(0, hole * 2 * L, width=2 * np.pi, bottom=r0, color="white", edgecolor=P["taupe"], linewidth=0.8, zorder=4)
    return L


def polar_legend(ax_leg, h4):
    ax_leg.set_axis_off()
    sun = {day: h4["results"][day]["forcing"]["sun_az"] for day, _ in DAYS}
    h = [Line2D([0], [0], color=DAY_STYLE[d]["color"], ls=DAY_STYLE[d]["ls"], lw=1.7, marker="o", ms=4.5,
                mfc=DAY_STYLE[d]["mfc"], mec=DAY_STYLE[d]["color"], label=f"{d} \u0394M1") for d, _ in DAYS]
    h += [Line2D([0], [0], color=SUN_C, ls=DAY_STYLE[d]["ls"], lw=1.2, label=f"sun 14:30, {d} {sun[d]:.0f}\u00b0")
          for d, _ in DAYS]
    h.append(Line2D([0], [0], color=S.SPINE_GREY, ls=":", lw=1.0, label="day median (ΔM1 = 0)"))
    h.append(mpl.patches.Patch(facecolor=P["teal_light"], edgecolor=P["teal_dark"], linestyle=(0, (3, 2)),
                               linewidth=0.7, label=f"sea-breeze sector {SB_LO:g}\u2013{SB_HI:g}\u00b0"))
    ax_leg.legend(handles=h, loc="upper left", ncol=1, fontsize=6.6, frameon=False, handlelength=2.2,
                  borderaxespad=0.0, labelspacing=0.45)


def draw_var(ax, h4):
    res = h4["results"]
    xs = {("IMBAT", "S"): 0.0, ("IMBAT", "B"): 0.62, ("CALM", "S"): 1.62, ("CALM", "B"): 2.24}
    vals = []
    for day, _ in DAYS:
        r = res[day]
        for term, key, col in (("S", "var_shade", SHADE_C), ("B", "var_breeze", BREEZE_C)):
            v, (lo, hi) = r[key], r[f"{key}_ci95"]
            x = xs[(day, term)]
            vals += [lo, hi]
            ax.plot([x, x], [1e-12, v], color=col, lw=1.6, alpha=0.55, zorder=2, solid_capstyle="butt")
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none", ecolor=P["muted"], elinewidth=1.1,
                        capsize=2.6, zorder=3)
            ax.scatter([x], [v], s=58, facecolor=col if day == "IMBAT" else "white",
                       edgecolor="white" if day == "IMBAT" else col, linewidth=1.0 if day == "IMBAT" else 1.5,
                       zorder=4)
        rule = 4.0 * r["var_breeze"]                           # lock: shade-dominated iff Var(S) > 4 Var(B)
        xs_ = xs[(day, "S")]
        ax.plot([xs_ - 0.24, xs_ + 0.24], [rule, rule], color=P["ink"], lw=1.1, ls=(0, (3, 2)), zorder=5)
        vals.append(rule)
    vmin, vmax = min(v for v in vals if v > 0), max(vals)
    lo_e, hi_e = np.floor(np.log10(vmin)) - 0.15, np.ceil(np.log10(vmax)) + 0.55
    ax.set_yscale("log")
    ax.set_ylim(10 ** lo_e, 10 ** hi_e)
    for day, _ in DAYS:
        r = res[day]
        ratio = r["ratio_var_shade_over_breeze"]
        xm = (xs[(day, "S")] + xs[(day, "B")]) / 2
        ax.text(xm, 10 ** (hi_e - 0.12), f"ratio {fmt_sig(ratio)}", ha="center", va="top", fontsize=7.2,
                fontweight="bold", color=P["ink"])
    # dashed tick explained once, next to the first one
    ax.text(xs[("IMBAT", "S")] + 0.27, 4.0 * res["IMBAT"]["var_breeze"], "4 \u00d7 Var(B)", ha="left",
            va="center", fontsize=6.4, color=P["ink"], style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85), zorder=6)
    ax.set_xticks(list(xs.values()), ["S", "B", "S", "B"])
    ax.set_xlim(-0.45, 2.69)
    for day, _ in DAYS:
        xm = (xs[(day, "S")] + xs[(day, "B")]) / 2
        ax.annotate(day, xy=(xm, 0.0), xycoords=ax.get_xaxis_transform(), xytext=(0, -27),
                    textcoords="offset points", ha="center", va="top", fontsize=7.5, fontweight="bold",
                    color=P["ink"])
    ax.set_ylabel("Variance across segments (\u00b0C\u00b2)")
    ax.grid(False)
    ax.grid(axis="y", which="major", color=P["grid"], linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(P["taupe"]); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, colors=P["ink"])
    ax.tick_params(axis="y", which="both", left=True, colors=P["muted"])


def draw_tables(fig, cells: dict, tab: pd.DataFrame, cbar_rect):
    """cells[(day, term)] = axes.  Mean shade / breeze term per orientation x H/W class, diverging Fig 4 ramp,
    one symmetric scale for both terms and days so magnitudes compare; cells with n < 5 hatched."""
    vals = np.r_[tab.get("mean_shade", pd.Series(dtype=float)).to_numpy(float),
                 tab.get("mean_breeze", pd.Series(dtype=float)).to_numpy(float)]
    L = nice_ceil(np.nanmax(np.abs(vals)))
    norm = mpl.colors.Normalize(-L, L)
    for (day, term), ax in cells.items():
        col = "mean_shade" if term == "S" else "mean_breeze"
        t = tab[tab["day"] == day].set_index(["orient", "hw_tercile"])
        ax.set_xlim(0, 4.25); ax.set_ylim(0, 3)
        ax.invert_yaxis()
        ax.set_axis_off()
        for i, oc in enumerate(ORIENT):
            for j, hw in enumerate(HW):
                x0 = j + (0.25 if j == 3 else 0.0)            # 'no enclosing buildings' set apart (A8-2)
                n = int(t.loc[(oc, hw), "n"]) if (oc, hw) in t.index else 0
                v = t.loc[(oc, hw), col] if (oc, hw) in t.index and col in t.columns else np.nan
                if n < 5 or not np.isfinite(v):
                    ax.add_patch(mpl.patches.Rectangle((x0, i), 1, 1, facecolor=P["taupe_light"], edgecolor="white",
                                                       linewidth=1.2, hatch="////", zorder=1))
                    ax.text(x0 + 0.5, i + 0.5, "n < 5", ha="center", va="center", fontsize=6.0, color=P["muted"])
                    continue
                rgba = CMAP_ANOM(norm(v))
                ax.add_patch(mpl.patches.Rectangle((x0, i), 1, 1, facecolor=rgba, edgecolor="white", linewidth=1.2,
                                                   zorder=1))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(x0 + 0.5, i + 0.5, sgn(v, 2), ha="center", va="center", fontsize=6.0,
                        color="white" if lum < 0.45 else P["ink"], zorder=3)
    cax = fig.add_axes(cbar_rect)
    sm = mpl.cm.ScalarMappable(cmap=CMAP_ANOM, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=7.5, length=0)
    cb.set_ticks([-L, 0, L])
    cb.ax.set_yticklabels([sgn(-L, 2 if L < 1 else 1), "0", sgn(L, 2 if L < 1 else 1)])
    cb.set_label("Mean term vs city reference (\u00b0C)", fontsize=7.5, color=P["ink"], labelpad=4)
    cb.outline.set_edgecolor(P["grid"])
    fig._fig6_cax = cax
    return L


def draw_card(ax, h4, ok, per):
    """Wide computed card: one row per day plus the IMBAT - CALM contrast (lock H4), columns = H4 quantities;
    verdict line from the lock rule."""
    card_frame(ax)
    res = h4["results"]
    cols = [(0.025, "left", ""), (0.215, "center", "Var(S)"), (0.345, "center", "Var(B)"),
            (0.490, "center", "Var(S) / Var(B)"), (0.640, "center", "ratio > 4"),
            (0.840, "center", ""), (0.840, "center", "wind ref. 14–17 h")]
    for x, ha, lab in cols:
        if lab:
            ctext(ax, x, 0.895, lab, ha=ha, size=7.2)
    ys = {"IMBAT": 0.700, "CALM": 0.500, "contrast": 0.300}
    dv, dc = 0.035, 0.060

    def val(x, v, ci, y):
        ctext(ax, x, y + dv, fmt_sig(v), ha="center")
        ctext(ax, x, y - dc, f"[{fmt_sig(ci[0])}, {fmt_sig(ci[1])}]", ha="center", size=6.6, weight="normal",
              color=P["muted"])

    for day, _ in DAYS:
        r, y = res[day], ys[day]
        fo = r["forcing"]
        ctext(ax, cols[0][0], y, day, ha="left")
        val(cols[1][0], r["var_shade"], r["var_shade_ci95"], y)
        val(cols[2][0], r["var_breeze"], r["var_breeze_ci95"], y)
        val(cols[3][0], r["ratio_var_shade_over_breeze"], r["ratio_ci95"], y)
        ctext(ax, cols[4][0], y + dv, CHECK if per[day] else CROSS, ha="center")
        ctext(ax, cols[6][0], y + dv, f"{fo['wd_mean_15_17']:.0f}°", ha="center", weight="normal")
        ctext(ax, cols[6][0], y - dc, f"n = {fo['wd_ref_n_reports']} reports, R = {fo['wd_ref_R']:.2f}",
              ha="center", size=6.6, weight="normal", color=P["muted"])
    c = h4.get("contrast_IMBAT_minus_CALM")
    if c:
        y = ys["contrast"]
        ctext(ax, cols[0][0], y, "IMBAT − CALM", ha="left")
        for (x, _, _), k in zip(cols[1:4], ("dVar_shade", "dVar_breeze", "dRatio")):
            val(x, c[k]["value"], c[k]["ci95"], y)
        ctext(ax, cols[5][0], y + dv, f"paired, n = {c['n_common']:,} segments", ha="center", weight="normal",
              size=7.0, color=P["muted"])
    ax.plot([0.025, 0.975], [0.165, 0.165], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    ctext(ax, 0.025, 0.083, "H4, lock rule Var(S) > 4 × Var(B) on both days:")
    ctext(ax, 0.975, 0.083, "shade-dominated" if ok else "not shade-dominated", ha="right")


# =============================================================================================
def build(rs: F5.ResultSet):
    S.apply_style()
    np.random.seed(20261030)
    d = load(rs)
    h2, h4, tab, segs = d["h2"], d["h4"], d["tab"], d["segs"]
    ok, per = h4_verdict(h4)
    if (h4["decision"]["H4"] == "SUPPORTED") != ok:
        raise RuntimeError(f"H4 verdict mismatch: figure rule {ok} vs results JSON {h4['decision']['H4']}")
    bins = {day: polar_bins(segs[day], cm.SEED + 6) for day, _ in DAYS}

    FW = S.V2_FIG_WIDTH_IN
    T, A_H, A_XL, GAP = 0.42, 3.90, 0.80, 0.42
    C_H, C_XL, BOT = 3.80, 0.62, 0.12
    FH = T + A_H + A_XL + GAP + C_H + C_XL + BOT
    fig = plt.figure(figsize=(FW, FH))
    ya = FH - T - A_H
    axF = S.inch_axes(fig, 1.95, ya, 1.85, A_H)
    axS = S.inch_axes(fig, 6.20, ya, 2.55, A_H)
    fam_order = draw_families(axF, h2)
    feats = draw_features(axS, h2)
    S.panel_label(axF, "A", fontsize=15.0, x=-0.95)
    PS = 3.05
    axP = fig.add_axes([9.40 / FW, (ya + A_H - PS - 0.12) / FH, PS / FW, PS / FH], projection="polar")
    L_polar = draw_polar(axP, bins, h4)
    axPL = S.inch_axes(fig, 9.25, ya - 0.78, 3.6, 0.95)
    polar_legend(axPL, h4)
    S.panel_label(axP, "B", fontsize=15.0, x=-0.09, y=(A_H + 0.03 - (A_H - PS - 0.12) - 0.0) / PS - 0.0)

    yc = ya - A_XL - GAP - C_H
    axV = S.inch_axes(fig, 1.05, yc, 2.45, C_H)
    draw_var(axV, h4)
    S.panel_label(axV, "C", fontsize=15.0)
    cw, ch = 0.52, 0.46
    tw, th = 4.25 * cw, 3 * ch
    x_tab0 = 5.95
    cells, gap_tab, gap_day = {}, 0.30, 0.30
    for i, (day, _) in enumerate(DAYS):
        yrow = yc + C_H - 0.62 - th - i * (th + gap_day)
        for j, term in enumerate(("S", "B")):
            cells[(day, term)] = S.inch_axes(fig, x_tab0 + j * (tw + gap_tab), yrow, tw, th)
    draw_ax = cells[("IMBAT", "S")]
    L_tab = draw_tables(fig, cells, tab, [(x_tab0 + 2 * tw + gap_tab + 0.40) / FW, (yc + 0.25) / FH, 0.13 / FW,
                                         (C_H - 0.85) / FH])
    # table furniture: term headers, H/W column headers, orientation row labels, day labels (computed-free text)
    for j, (term, lab) in enumerate((("S", "shade term S"), ("B", "breeze term B"))):
        ax = cells[("IMBAT", term)]
        ax.text(4.25 / 2, -0.95, lab, ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                color=P["teal_dark"] if term == "S" else blend(BREEZE_C, P["ink"], 0.42), transform=ax.transData, clip_on=False)
        for jj, hw in enumerate(HW):
            x0 = jj + (0.25 if jj == 3 else 0.0)
            ax.text(x0 + 0.5, -0.1, HW_LABEL[hw], ha="center", va="bottom", fontsize=6.0, color=P["muted"],
                    linespacing=0.95, clip_on=False)
    for day, _ in DAYS:
        ax = cells[(day, "S")]
        for i, oc in enumerate(ORIENT):
            ax.text(-0.12, i + 0.5, oc, ha="right", va="center", fontsize=6.6, color=P["ink"], clip_on=False)
        ax.text(-2.85, 1.5, day, ha="center", va="center", rotation=90, fontsize=7.2, fontweight="bold",
                color=P["ink"], clip_on=False)
    cells[("IMBAT", "S")].text(-0.12, -0.1, "H/W class", ha="right", va="bottom", fontsize=6.6,
                               color=P["muted"], clip_on=False)

    label = h4["results"]["IMBAT"]["label"]
    arts = F5.stamp(fig, rs, label)
    meta = dict(verdict_rule=ok, verdict_json=h4["decision"]["H4"], per_day=per, label=label,
                family_order=fam_order, feature_order=feats, polar_limit_C=L_polar, table_limit_C=L_tab,
                polar_bins={day: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in b.items()}
                            for day, b in bins.items()},
                sun_az={day: h4["results"][day]["forcing"]["sun_az"] for day, _ in DAYS})
    panels = {"a": [axF, axS], "b": [axP, axPL], "c": [axV] + list(cells.values()) + [fig._fig6_cax]}
    return fig, meta, panels, [], arts, draw_ax


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    F5.common_args(ap)
    a = ap.parse_args(argv)
    rs = F5.ResultSet(a.results, a.subdir, a.allow_partial)
    fig, meta, panels, cards, _, _ = build(rs)
    png = rs.out_png("fig_6_drivers", a.out)
    qa = F5.finish(fig, rs, png, panels, cards, dict(meta=meta), rs.qa_dir(a.qa))
    print(json.dumps({k: qa[k] for k in ("png", "fig_size_in", "smallest_author_pt", "smallest_printed_pt_190mm",
                                         "text_text_overlaps", "text_text_touching", "clipped_outside_sheet",
                                         "spill_out_of_container")},
                     indent=1, ensure_ascii=False))
    print("H4 verdict (figure rule / JSON):", meta["verdict_rule"], meta["verdict_json"])
    return qa


if __name__ == "__main__":
    main()
