"""CLIMORFA v2 Figure 3: design-day forcing and model behaviour (WP4.3).

One command (re-run unchanged on the final data; the DRAFT note disappears at 427/427 tiles):

    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig3_forcing_model
        -> paper/figures/v2/main/fig_3_forcing_model.png
    options: --qa <dir>   QA json (overlap / clipping / sizes / all plotted numbers) + 100 % panel crops
             --out <png>  test output

No field data and no hypothesis (H1-H5) quantity is read or shown.  Every number printed in the figure is computed here.

Inputs
  (a) data/street/forcing_summer/met_DESIGN_<DAY>_<date>_hourly.txt   the met files of the SOLWEIG run
      (scripts/street/solweig/run_solweig_citywide.py DESIGN_DAYS); the CALM _v2 file used by the UTCI step differs only in
      Wd (asserted); era5_hourly_<DAY>_<date>.csv (hour-ending GHI, asserted equal to the met-file Kdn);
      pvlib Ineichen clear-sky GHI at the interval midpoint (h - 30 min), 38.42 N 27.14 E, 20 m, Europe/Istanbul (UTC+3),
      exactly as scripts/street/forcing_summer/04_build_forcing_and_compare.py; vrb_flags.csv (lock A7).
  (b) data/street/solweig/<DAY>/{tmrt,shadow}/<tile>.tif, data/street/utci/<DAY>/utci_street/<tile>.tif (16 bands =
      labels 06-21, band = label - 5, band_hour_table.csv), data/street/analysis/ps_index/<tile>.tif (PS > 0), for the
      tiles of data/street/analysis/sens/subset_40_tiles.csv finished on BOTH days (UTCI status OK in
      logs/wp/WP2.3_progress.csv and all six rasters present).  Sunlit = shadow >= 50 (% sunlit, WP2.3 convention),
      shaded = shadow < 50.  Per-hour medians and quartiles over the pooled PS cells from 0.01 degC histograms.
  (c, d) data/street/analysis/seg_metrics_<DAY>.parquet joined to segments.gpkg (hw_ratio); frames built like
      scripts/street/stats/common.build_frame (finite utci_b15).  H/W classes = the H4 code's classes (terciles of the
      non-NaN hw_ratio pooled over both day frames, h4_shade_breeze.hw_class) + 'no enclosing buildings' (A8-2).
      SVF quintiles of svf_med over the pooled segment set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "street" / "stats"))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.street.figures import build_fig5_h1_scale as F5  # noqa: E402  (day encoding, QA, card idioms)
from scripts.street.figures.build_fig5_h1_scale import DAY_STYLE, MINUS, P, S, card_frame, ctext, kfmt, sgn  # noqa: E402

DATA = ROOT / "data" / "street"
FORC = DATA / "forcing_summer"
AN = DATA / "analysis"
OUT_PNG = ROOT / "paper" / "figures" / "v2" / "main" / "fig_3_forcing_model.png"
N_TILES_TOTAL = 427
DAYS = [("IMBAT", "2022-07-05"), ("CALM", "2023-07-26")]
HOURS = np.arange(6, 22)                       # hour labels 06-21 (hour ending at h:00 local, UTC+3)
NOON_LABEL = 13                                # 'noon' of WP1.4 / 04_build_forcing_and_compare.py (12:00-13:00)
LAT, LON, ALT, TZ = 38.42, 27.14, 20.0, "Europe/Istanbul"
SUNLIT_MIN = 50                                # shadow raster = sunlit fraction x 100; WP2.3 sunlit/shaded split
MIN_EXPOSURE_SHARE = 0.01                      # plot an exposure class at an hour only if >= 1 % of PS cells
UTCI_BOUNDS = [(38.0, "very strong"), (46.0, "extreme")]   # UTCI assessment scale lower bounds (Broede et al. 2012)
V_FLOOR = 0.5                                  # clip lower bound of v10,eff (methods)
BIN = 0.01
EDGES = np.arange(-30.0, 110.0 + BIN, BIN)

SUN_C = P["orange_dark"]                       # sunlit: Fig 6 sun colour
SHADE_C = P["teal_dark"]                       # shaded: Fig 6 shade family (teal), dark tone for lines
EXPO = {"sunlit": SUN_C, "shaded": SHADE_C}
WIND_C = S.FAMILY_COLORS_FIG2["Network"]       # Fig 2 v2 'wind' family
WIND_DARK = F5.blend(WIND_C, P["ink"], 0.42)
HW = ["low", "mid", "high", "no enclosing buildings"]
SB_LO, SB_HI = 250.0, 340.0                    # sea-breeze sector of the IMBAT rule


# =============================================================================================== data: (a) forcing
def met_file(day: str, date: str) -> Path:
    """The met file of the SOLWEIG run (read from the run script, so a change there is followed)."""
    src = (ROOT / "scripts" / "street" / "solweig" / "run_solweig_citywide.py").read_text(encoding="utf-8")
    name = f"met_DESIGN_{day}_{date}_hourly.txt"
    if f"data/street/forcing_summer/{name}" not in src:
        raise RuntimeError(f"{name} is not the met file named in run_solweig_citywide.py")
    return FORC / name


def load_forcing() -> dict:
    import pvlib
    loc = pvlib.location.Location(LAT, LON, tz=TZ, altitude=ALT)
    flags = pd.read_csv(FORC / "vrb_flags.csv")
    out = {}
    for day, date in DAYS:
        mf = met_file(day, date)
        met = pd.read_csv(mf, sep=r"\s+")
        v2 = FORC / f"met_DESIGN_{day}_{date}_hourly_v2.txt"
        if v2.exists():                                       # UTCI step reads v2: identical except Wd (WP2.3b)
            m2 = pd.read_csv(v2, sep=r"\s+")
            cols = [c for c in met.columns if c != "Wd"]
            if not np.allclose(met[cols].to_numpy(float), m2[cols].to_numpy(float)):
                raise RuntimeError(f"{v2.name} differs from {mf.name} outside Wd")
        era = pd.read_csv(FORC / f"era5_hourly_{day}_{date}.csv")
        if not np.allclose(era["ghi_wm2"].clip(lower=0).to_numpy(), met["Kdn"].clip(lower=0).to_numpy(), atol=0.011):
            raise RuntimeError(f"{day}: ERA5 GHI and met-file Kdn differ")
        grid = pd.date_range(pd.Timestamp(f"{date} 00:00", tz=TZ), periods=24, freq="1h")
        cs = loc.get_clearsky(grid - pd.Timedelta(minutes=30), model="ineichen")["ghi"].to_numpy()
        ghi = era["ghi_wm2"].clip(lower=0).to_numpy(float)
        f = flags[flags["day"] == day].set_index("it")["vrb"].astype(bool)
        idx = met["it"].to_numpy(int)
        sel = np.isin(idx, HOURS)
        d = dict(ta=met["Td"].to_numpy(float)[sel], rh=met["RH"].to_numpy(float)[sel],
                 u10=met["Wind"].to_numpy(float)[sel], ghi=ghi[sel], cs=cs[sel],
                 vrb=np.array([bool(f.get(h, False)) for h in HOURS]), met_file=mf.name)
        d["noon_ratio"] = float(ghi[NOON_LABEL] / cs[NOON_LABEL])
        d["noon_ghi"], d["noon_cs"] = float(ghi[NOON_LABEL]), float(cs[NOON_LABEL])
        d["ratio_by_label"] = {int(h): round(float(ghi[h] / cs[h]), 4) for h in range(9, 18)}
        for k in ("ta", "ghi", "rh", "u10"):
            i = int(np.argmax(d[k]))
            d[f"peak_{k}"], d[f"peak_{k}_label"] = float(d[k][i]), int(HOURS[i])
        d["min_rh"] = float(d["rh"].min())
        d["vrb_labels"] = [int(h) for h in HOURS[d["vrb"]]]
        i15 = int(np.where(HOURS == 15)[0][0])
        d["ta15"], d["rh15"], d["u10_15"] = float(d["ta"][i15]), float(d["rh"][i15]), float(d["u10"][i15])
        out[day] = d
    return out


# =============================================================================================== data: (b) rasters
def finished_tiles(tiles: list[str]) -> list[str]:
    prog = pd.read_csv(ROOT / "logs" / "wp" / "WP2.3_progress.csv")
    ok = prog[prog["status"].astype(str).str.upper() == "OK"].groupby("tile")["day"].apply(set)
    done = []
    for t in tiles:
        if not ({"IMBAT", "CALM"} <= ok.get(t, set())):
            continue
        paths = [DATA / "solweig" / d / v / f"{t}.tif" for d, _ in DAYS for v in ("tmrt", "shadow")] + \
                [DATA / "utci" / d / "utci_street" / f"{t}.tif" for d, _ in DAYS] + [AN / "ps_index" / f"{t}.tif"]
        if all(p.exists() for p in paths):
            done.append(t)
    return done


def _read(p: Path):
    import rasterio
    with rasterio.open(p) as r:
        a = r.read()
        nd = r.nodata
    a = a.astype(np.float32) if a.dtype != np.float32 else a
    if nd is not None:
        a[a == nd] = np.nan
    return a


def raster_pass(tiles: list[str]) -> dict:
    """Histograms of Tmrt and UTCI_street per day x exposure x hour label over PS cells."""
    H = {(d, v, e): np.zeros((len(HOURS), len(EDGES) - 1), np.int64)
         for d, _ in DAYS for v in ("tmrt", "utci") for e in EXPO}
    ncell = {(d, e): np.zeros(len(HOURS), np.int64) for d, _ in DAYS for e in EXPO}
    nps = {d: np.zeros(len(HOURS), np.int64) for d, _ in DAYS}
    for t in tiles:
        ps = _read(AN / "ps_index" / f"{t}.tif")[0]
        ps = np.nan_to_num(ps, nan=0.0) > 0
        for day, _ in DAYS:
            tm = _read(DATA / "solweig" / day / "tmrt" / f"{t}.tif")
            sh = _read(DATA / "solweig" / day / "shadow" / f"{t}.tif")
            ut = _read(DATA / "utci" / day / "utci_street" / f"{t}.tif")
            if not (tm.shape[0] == sh.shape[0] == ut.shape[0] == len(HOURS)):
                raise RuntimeError(f"{t} {day}: expected {len(HOURS)} bands")
            for b in range(len(HOURS)):
                ok = ps & np.isfinite(tm[b]) & np.isfinite(sh[b]) & np.isfinite(ut[b])
                nps[day][b] += int(ok.sum())
                for e, m in (("sunlit", sh[b] >= SUNLIT_MIN), ("shaded", sh[b] < SUNLIT_MIN)):
                    mm = ok & m
                    ncell[(day, e)][b] += int(mm.sum())
                    for v, arr in (("tmrt", tm[b]), ("utci", ut[b])):
                        H[(day, v, e)][b] += np.histogram(arr[mm], bins=EDGES)[0]
    return dict(H=H, ncell=ncell, nps=nps)


def hist_q(h: np.ndarray, qs=(0.25, 0.5, 0.75)) -> np.ndarray:
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.full(len(qs), np.nan)
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    return np.array([centers[np.searchsorted(c, q * c[-1])] for q in qs])


def diurnal_stats(rp: dict) -> dict:
    out = {}
    for day, _ in DAYS:
        for v in ("tmrt", "utci"):
            for e in EXPO:
                q = np.array([hist_q(rp["H"][(day, v, e)][b]) for b in range(len(HOURS))])
                share = rp["ncell"][(day, e)] / np.maximum(rp["nps"][day], 1)
                q[share < MIN_EXPOSURE_SHARE] = np.nan
                out[(day, v, e)] = dict(q25=q[:, 0], med=q[:, 1], q75=q[:, 2], share=share)
    return out


# =============================================================================================== data: (c, d) segments
def load_segment_frames() -> dict:
    import pyogrio
    import h4_shade_breeze as H4
    seg = pyogrio.read_dataframe(AN / "segments.gpkg", layer="segments", columns=["seg_id", "hw_ratio"],
                                 read_geometry=False)
    fr, n_tiles, partial = {}, {}, {}
    for day, _ in DAYS:
        m = pd.read_parquet(AN / f"seg_metrics_{day}.parquet",
                            columns=["seg_id", "utci_b15", "utci_unif_b15", "ratio_med_b15", "v10eff_b15", "svf_med",
                                     "partial_run", "n_tiles_at_extraction"])
        n_tiles[day] = int(m["n_tiles_at_extraction"].max())
        partial[day] = bool(m["partial_run"].any())
        f = seg.merge(m, on="seg_id", how="inner")
        fr[day] = f[np.isfinite(f["utci_b15"])].reset_index(drop=True)
    hw_all = pd.concat([f["hw_ratio"] for f in fr.values()]).dropna()
    edges = (float(hw_all.quantile(1 / 3)), float(hw_all.quantile(2 / 3)))
    svf_all = pd.concat([f[["seg_id", "svf_med"]] for f in fr.values()]).drop_duplicates("seg_id")["svf_med"].dropna()
    q_edges = np.quantile(svf_all, [0.2, 0.4, 0.6, 0.8])
    for day, f in fr.items():
        f["hw_class"] = H4.hw_class(f["hw_ratio"], edges).to_numpy()
        f["svf_q"] = np.where(f["svf_med"].notna(), np.searchsorted(q_edges, f["svf_med"].to_numpy(), side="right"), -1)
        f["d_street_unif"] = f["utci_b15"] - f["utci_unif_b15"]
    if set(H4.HW_CLASSES) != set(HW):
        raise RuntimeError(f"H/W classes of the H4 code changed: {H4.HW_CLASSES}")
    return dict(fr=fr, hw_edges=edges, svf_edges=q_edges.tolist(), n_tiles=n_tiles, partial=partial)


def segment_numbers(sd: dict) -> dict:
    fr = sd["fr"]
    out = dict(hw={}, svf={})
    for day, f in fr.items():
        g = f.groupby("hw_class")
        out["hw"][day] = {k: dict(n=int((f["hw_class"] == k).sum()),
                                  r_med=float(g.get_group(k)["ratio_med_b15"].median()),
                                  r_q25=float(g.get_group(k)["ratio_med_b15"].quantile(0.25)),
                                  r_q75=float(g.get_group(k)["ratio_med_b15"].quantile(0.75)),
                                  v_med=float(g.get_group(k)["v10eff_b15"].median())) for k in HW}
        rows = {}
        for q in range(5):
            s = f[f["svf_q"] == q]
            rows[q] = dict(n=int(len(s)), svf_med=float(s["svf_med"].median()),
                           d_med=float(s["d_street_unif"].median()),
                           d_q25=float(s["d_street_unif"].quantile(0.25)), d_q75=float(s["d_street_unif"].quantile(0.75)),
                           m1_street=float(s["utci_b15"].median()), m1_unif=float(s["utci_unif_b15"].median()))
        out["svf"][day] = rows
        out["svf"][day]["enclosed_minus_open"] = dict(
            street=rows[0]["m1_street"] - rows[4]["m1_street"], uniform=rows[0]["m1_unif"] - rows[4]["m1_unif"])
    return out


# =============================================================================================== drawing helpers
def trend_style(ax, ylabel=None):
    """Fig 9 trend sub-row axes: taupe bottom/left spines, dashed y-grid, ink ticks (S.trend_axes_style, first=True)."""
    S.trend_axes_style(ax, first=True)
    ax.tick_params(axis="x", labelsize=9.24, colors=P["muted"], length=0)
    ax.tick_params(axis="y", labelsize=9.24, colors=P["muted"])
    if ylabel:
        ax.set_ylabel(ylabel)


def hour_axis(ax, labels=True):
    ax.set_xlim(HOURS[0] - 0.5, HOURS[-1] + 0.5)
    ax.set_xticks([6, 9, 12, 15, 18, 21])
    if labels:
        ax.set_xticklabels(["06", "09", "12", "15", "18", "21"])
    else:
        ax.set_xticklabels([])
    ax.grid(True, axis="x", color=P["grid"], linewidth=0.5, alpha=0.6, linestyle="--")


def day_line(ax, x, y, day, lw=1.6, ms=4.2, color=None, z=3, **kw):
    st = DAY_STYLE[day]
    c = color or st["color"]
    ax.plot(x, y, color=c, ls=st["ls"], lw=lw, zorder=z, solid_capstyle="round", **kw)
    ax.scatter(x, y, s=ms ** 2, facecolor=c if day == "IMBAT" else "white", edgecolor=c, linewidth=1.0, zorder=z + 0.1)


def note(ax, x, y, s, ha="left", va="top", size=7.2):
    """Computed read-out box (Fig 5 facet-tag / Fig 9 card idiom, small)."""
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, fontweight="bold", color=P["ink"],
                   linespacing=1.25, zorder=12,
                   bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.28", "alpha": 0.95})


def fmt_label(h: int) -> str:
    return f"{h:02d}"


# =============================================================================================== panels
def draw_forcing(axs: dict, fo: dict):
    for day, _ in DAYS:
        d = fo[day]
        day_line(axs["ta"], HOURS, d["ta"], day)
        day_line(axs["rh"], HOURS, d["rh"], day)
        day_line(axs["u10"], HOURS, d["u10"], day)
        day_line(axs["ghi"], HOURS, d["ghi"], day)
        axs["ghi"].plot(HOURS, d["cs"], color=DAY_STYLE[day]["color"], ls=(0, (1, 1.6)), lw=1.3, zorder=2.5)
    # VRB-flagged hours (lock A7): light bands drawn only behind that day's own series (from 0 to just above its
    # daily U10 maximum), labelled with the day, so a day without VRB hours (IMBAT) is not implied
    for day, _ in DAYS:
        hs = fo[day]["vrb_labels"]
        if not hs:
            continue
        top = float(np.ceil(fo[day]["u10"].max())) + 0.6
        band_c = F5.blend(DAY_STYLE[day]["color"], "white", 0.80)
        runs = [[hs[0]]]                                      # one band + label per run of consecutive flagged hours
        for h in hs[1:]:
            if h == runs[-1][-1] + 1:
                runs[-1].append(h)
            else:
                runs.append([h])
        for r_ in runs:
            axs["u10"].fill_between([r_[0] - 0.5, r_[-1] + 0.5], 0, top, facecolor=band_c, edgecolor="none",
                                    zorder=0.6)
            axs["u10"].text((r_[0] + r_[-1]) / 2, top + 0.15, f"VRB ({day})", ha="center", va="bottom",
                            fontsize=6.6, color=P["muted"], fontweight="bold", zorder=6)
    axs["u10"].axhline(V_FLOOR, color=P["muted"], ls=(0, (4, 3)), lw=1.0, zorder=1)
    axs["u10"].text(HOURS[-1] + 0.35, V_FLOOR + 0.25, f"{V_FLOOR:g} m s\u207b\u00b9 floor of v\u2081\u2080,eff",
                    ha="right", va="bottom", fontsize=6.6, color=P["muted"], style="italic", zorder=6)
    ylab = {"ta": "Air temperature (\u00b0C)", "rh": "Relative humidity (%)", "u10": "U\u2081\u2080 (m s\u207b\u00b9)",
            "ghi": "GHI (W m\u207b\u00b2)"}
    for k, ax in axs.items():
        trend_style(ax, ylab[k])
        hour_axis(ax, labels=k in ("rh", "u10"))
    ta_all = np.r_[fo["IMBAT"]["ta"], fo["CALM"]["ta"]]
    axs["ta"].set_ylim(np.floor(ta_all.min() / 2) * 2 - 6, np.ceil(ta_all.max() / 2) * 2 + 2)
    axs["ghi"].set_ylim(0, 1380)
    axs["ghi"].set_yticks([0, 250, 500, 750, 1000])
    rh_all = np.r_[fo["IMBAT"]["rh"], fo["CALM"]["rh"]]
    axs["rh"].set_ylim(0, max(60, np.ceil(rh_all.max() / 10) * 10 + 5))
    u_all = np.r_[fo["IMBAT"]["u10"], fo["CALM"]["u10"]]
    axs["u10"].set_ylim(0, np.ceil(u_all.max() / 2) * 2 + 1.5)
    axs["u10"].set_yticks(np.arange(0, np.ceil(u_all.max() / 2) * 2 + 0.1, 2.5))
    for k in ("rh", "u10"):
        axs[k].set_xlabel("Hour label (local, hour ending)")
    # computed notes
    a, b = fo["IMBAT"], fo["CALM"]


def forcing_legend(ax):
    h = F5.day_legend_handles()
    for hh, (day, date) in zip(h, DAYS):
        hh.set_label(f"{day} {date}")
        hh.set_markersize(4.2)
    h.append(Line2D([0], [0], color=P["muted"], ls=(0, (1, 1.6)), lw=1.3, label="Ineichen clear sky"))
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.035), ncol=3, fontsize=6.6, frameon=False,
              handlelength=2.0, columnspacing=1.0, borderaxespad=0.0, handletextpad=0.5)


def draw_diurnal(ax_t, ax_u, ds: dict):
    for ax, v in ((ax_t, "tmrt"), (ax_u, "utci")):
        for day, _ in DAYS:
            for e, col in EXPO.items():
                s = ds[(day, v, e)]
                ax.fill_between(HOURS, s["q25"], s["q75"], color=col, alpha=0.13 if day == "IMBAT" else 0.09,
                                linewidth=0, zorder=1.5)
                day_line(ax, HOURS, s["med"], day, color=col, lw=1.6, ms=4.0)
        trend_style(ax)
        hour_axis(ax, labels=(v == "utci"))
    ax_t.set_ylabel("T$_{mrt}$ (\u00b0C)")
    ax_u.set_ylabel("UTCI, street wind (\u00b0C)")
    ax_u.set_xlabel("Hour label (local, hour ending)")
    for ax, v in ((ax_t, "tmrt"), (ax_u, "utci")):
        vals = np.concatenate([np.r_[ds[(d, v, e)]["q25"], ds[(d, v, e)]["q75"]] for d, _ in DAYS for e in EXPO])
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        ax.set_ylim(np.floor(lo / 5) * 5 - 2, np.ceil(hi / 5) * 5 + (14 if v == "tmrt" else 3))
    for yv, lab in UTCI_BOUNDS:
        ax_u.axhline(yv, color=P["muted"], ls=(0, (4, 3)), lw=0.9, zorder=1)
        ax_u.text(HOURS[0] - 0.35, yv + 0.3, f"{yv:.0f}\u00b0C", ha="left", va="bottom", fontsize=6.6,
                  color=P["muted"], style="italic", zorder=6,
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))


def diurnal_legend(ax):
    h = [Line2D([0], [0], color=SUN_C, lw=1.6, label="sunlit"),
         Line2D([0], [0], color=SHADE_C, lw=1.6, label="shaded"),
         Line2D([0], [0], color=P["muted"], lw=1.6, ls="-", marker="o", ms=4.0, mfc=P["muted"], label="IMBAT"),
         Line2D([0], [0], color=P["muted"], lw=1.6, ls=DAY_STYLE["CALM"]["ls"], marker="o", ms=4.0, mfc="white",
                label="CALM"),
         mpl.patches.Patch(facecolor=P["muted"], alpha=0.18, edgecolor="none", label="IQR")]
    ax.legend(handles=h, loc="upper left", ncol=5, fontsize=6.6, frameon=True, facecolor="white", edgecolor="none",
              borderpad=0.3, handlelength=2.0, columnspacing=0.9, handletextpad=0.45, borderaxespad=0.35)


def draw_ratio(ax, sd: dict, sn: dict, fo_imbat: dict):
    f = sd["fr"]["IMBAT"]
    rng = np.random.default_rng(20261030)
    lo, hi = 0.0, 1.62
    box_data, pos = [], []
    for i, k in enumerate(HW):
        x = i + (0.25 if i == 3 else 0.0)
        vals = f.loc[f["hw_class"] == k, "ratio_med_b15"].dropna().to_numpy(float)
        face = WIND_C if i < 3 else P["taupe_light"]
        edge = WIND_DARK if i < 3 else P["taupe_dark"]
        # violin clipped to the observed range (cut = 0): F5.half_violin evaluates the KDE only on [min, max] of vals
        F5.half_violin(ax, x - 0.02, vals, face, edge, min(lo, vals.min()), max(hi, vals.max()))
        # rain = 120 sampled segments that always include the class minimum and maximum, so the violin ends sit on
        # visible points (the tails beyond the 1.5 IQR whiskers are real segments, not KDE spread)
        sub = vals if len(vals) <= 120 else np.r_[rng.choice(vals, 118, replace=False), vals.min(), vals.max()]
        ax.scatter(x + 0.20 + rng.uniform(0, 0.20, len(sub)), sub, s=4.8, color=WIND_C if i < 3 else P["taupe"],
                   alpha=0.30, edgecolors="none", zorder=1.8)
        box_data.append(vals); pos.append(x + 0.085)
    bp = ax.boxplot(box_data, positions=pos, widths=0.11, whis=1.5, showfliers=False, patch_artist=True,
                    manage_ticks=False, zorder=3,
                    medianprops=dict(color=S.PANEL_EDGE, linewidth=1.1, solid_capstyle="butt"),
                    whiskerprops=dict(color=S.PANEL_EDGE, linewidth=0.8),
                    capprops=dict(color=S.PANEL_EDGE, linewidth=0.8),
                    boxprops=dict(edgecolor=S.PANEL_EDGE, linewidth=0.8))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(WIND_C if i < 3 else P["taupe_light"])
    S.apply_box_gradient(ax)
    ax.axvline(2.625, color=P["grid"], lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.axhline(1.0, color=S.SPINE_GREY, ls=":", lw=1.0, zorder=1.5)            # r = 1: open-terrain wind
    e0, e1 = sd["hw_edges"]
    n = sn["hw"]["IMBAT"]
    labs = [f"low\n< {e0:.2f}\n{kfmt(n['low']['n'])}", f"mid\n{e0:.2f}\u2013{e1:.2f}\n{kfmt(n['mid']['n'])}",
            f"high\n> {e1:.2f}\n{kfmt(n['high']['n'])}",
            f"no encl.\nbldgs\n{kfmt(n['no enclosing buildings']['n'])}"]
    ax.set_xticks([0, 1, 2, 3.25])
    ax.set_xticklabels(labs, fontsize=8.2, linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, 3.25 + 0.6)
    ax.set_ylim(lo, 1.95)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_ylabel("Wind ratio r\n(segment median)", linespacing=1.15)
    ax.grid(True, axis="y", color=P["grid"], linewidth=0.7)
    ax.grid(False, axis="x")
    for sp in ax.spines.values():                                            # v1 Fig 4 / v2 Fig 5: pale full frame
        sp.set_color(P["grid"]); sp.set_linewidth(1.0)
    ax.text(-0.5, 1.0 + 0.02, "r = 1", ha="left", va="bottom", fontsize=6.6, color=P["muted"],
            style="italic", zorder=6)


def draw_vcard(ax, sn: dict):
    card_frame(ax)
    xl, xd = 0.06, {"IMBAT": 0.555, "CALM": 0.855}
    y = 0.905
    ctext(ax, xl, y, "median v\u2081\u2080,eff", size=7.6)
    y -= 0.085
    ctext(ax, xl, y, "label 15, m s\u207b\u00b9", size=7.0, weight="normal", color=P["muted"])
    y -= 0.115
    for day, _ in DAYS:
        ctext(ax, xd[day], y, day, ha="center", size=7.2)
    y -= 0.115
    ax.plot([xl, 1 - xl + 0.02], [y + 0.055, y + 0.055], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    lab = {"low": "low H/W", "mid": "mid H/W", "high": "high H/W", "no enclosing buildings": "no encl."}
    for k in HW:
        ctext(ax, xl, y, lab[k], size=7.2, weight="normal")
        for day, _ in DAYS:
            ctext(ax, xd[day], y, f"{sn['hw'][day][k]['v_med']:.1f}", ha="center", size=7.2)
        y -= 0.105
    ax.plot([xl, 1 - xl + 0.02], [y + 0.05, y + 0.05], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    y -= 0.03
    ctext(ax, xl, y, "U\u2081\u2080", size=7.2, weight="normal")
    for day, _ in DAYS:
        ctext(ax, xd[day], y, f"{sn['u10_15'][day]:.1f}", ha="center", size=7.2, weight="normal")


def draw_svf(ax, sd: dict, sn: dict):
    off = {"IMBAT": -0.17, "CALM": 0.17}
    allw = []
    for day, _ in DAYS:
        f = sd["fr"][day]
        data = [f.loc[f["svf_q"] == q, "d_street_unif"].dropna().to_numpy(float) for q in range(5)]
        st = DAY_STYLE[day]
        bp = ax.boxplot(data, positions=np.arange(5) + off[day], widths=0.28, whis=1.5, showfliers=False,
                        patch_artist=True, manage_ticks=False, zorder=3,
                        medianprops=dict(color=S.PANEL_EDGE if day == "IMBAT" else st["color"], linewidth=1.2,
                                         solid_capstyle="butt"),
                        whiskerprops=dict(color=st["color"], linewidth=0.9, linestyle=st["ls"] if day == "CALM" else "-"),
                        capprops=dict(color=st["color"], linewidth=0.9),
                        boxprops=dict(edgecolor=st["color"], linewidth=1.0, linestyle=st["ls"] if day == "CALM" else "-"))
        for p_ in bp["boxes"]:
            p_.set_facecolor(st["color"] if day == "IMBAT" else "white")
        for w in bp["whiskers"]:
            allw.append(w.get_ydata())
    S.apply_box_gradient(ax)
    ax.axhline(0.0, color=S.SPINE_GREY, ls=":", lw=1.0, zorder=1.5)
    allw = np.concatenate(allw)
    lo, hi = float(np.min(allw)), float(np.max(allw))
    span = hi - lo
    ax.set_ylim(lo - 0.06 * span, hi + 0.62 * span)
    svf = sn["svf"]["IMBAT"]
    labs = [f"Q{q + 1}\n{svf[q]['svf_med']:.2f}" for q in range(5)]
    labs[0] = f"Q1 enclosed\n{svf[0]['svf_med']:.2f}"
    labs[4] = f"Q5 open\n{svf[4]['svf_med']:.2f}"
    ax.set_xticks(range(5))
    ax.set_xticklabels(labs, fontsize=8.2, linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylabel("UTCI street \u2212 uniform\nwind (\u00b0C)", linespacing=1.15)
    step = 0.5 if span < 4 else 1.0
    ax.set_yticks(np.arange(np.ceil((lo - 0.06 * span) / step) * step, hi + 1e-9 + step, step))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, _: "0" if abs(y) < 1e-9 else (f"{y:+.1f}".replace("-", MINUS))))
    ax.grid(True, axis="y", color=P["grid"], linewidth=0.7)
    ax.grid(False, axis="x")
    for sp in ax.spines.values():
        sp.set_color(P["grid"]); sp.set_linewidth(1.0)
    h = F5.day_legend_handles()
    for hh in h:
        hh.set_marker("s"); hh.set_markersize(6.5)
    ax.legend(handles=h, loc="upper right", ncol=1, fontsize=6.6, frameon=True, facecolor="white", edgecolor="none",
              borderpad=0.3, handlelength=2.0, handletextpad=0.5, borderaxespad=0.4)
    ax.text(0.5, -0.255, "sky view factor quintile (median SVF)", transform=ax.transAxes, ha="center", va="top",
            fontsize=10.08, color=P["ink"])


# =============================================================================================== build
def build(qa_dir: Path | None = None):
    S.apply_style()
    np.random.seed(20261030)
    import common as cm
    fo = load_forcing()
    sub = pd.read_csv(AN / "sens" / "subset_40_tiles.csv")["tile_id"].tolist()
    tiles = finished_tiles(sub)
    if not tiles:
        raise RuntimeError("no subset tile finished on both days")
    rp = raster_pass(tiles)
    ds = diurnal_stats(rp)
    sd = load_segment_frames()
    sn = segment_numbers(sd)
    fsum = {day: cm.forcing_summary(day) for day, _ in DAYS}
    sn["u10_15"] = {day: fsum[day]["u10"] for day, _ in DAYS}

    n_tiles_ext = min(sd["n_tiles"].values())
    prog = pd.read_csv(ROOT / "logs" / "wp" / "WP2.3_progress.csv")
    okp = prog[prog["status"].astype(str).str.upper() == "OK"].groupby("tile")["day"].apply(set)
    n_utci_both = int(sum({"IMBAT", "CALM"} <= s_ for s_ in okp))
    n_tiles = min(n_tiles_ext, n_utci_both)
    partial = n_tiles < N_TILES_TOTAL or any(sd["partial"].values())

    # ------------------------------------------------------------------------------------- layout (inches)
    FW = S.V2_FIG_WIDTH_IN
    BOT, XL2, H2, GAP2 = 0.30, 1.02, 2.75, 0.62
    XL1, H1, G1, T = 0.70, 2.05, 0.20, 0.62
    FH = BOT + XL2 + H2 + GAP2 + XL1 + H1 + G1 + H1 + T
    fig = plt.figure(figsize=(FW, FH))
    y1b = BOT + XL2 + H2 + GAP2 + XL1          # lower sub-row of row 1
    y1t = y1b + H1 + G1                        # upper sub-row
    L, wa, ga = 1.02, 2.45, 0.95
    axs = {"ta": S.inch_axes(fig, L, y1t, wa, H1), "ghi": S.inch_axes(fig, L + wa + ga, y1t, wa, H1),
           "rh": S.inch_axes(fig, L, y1b, wa, H1), "u10": S.inch_axes(fig, L + wa + ga, y1b, wa, H1)}
    draw_forcing(axs, fo)
    forcing_legend(axs["ta"])
    S.panel_label(axs["ta"], "A", fontsize=15.0, x=-0.30, y=1.13)

    xb = L + 2 * wa + ga + 1.02
    wb = FW - 0.22 - xb
    ax_t = S.inch_axes(fig, xb, y1t, wb, H1)
    ax_u = S.inch_axes(fig, xb, y1b, wb, H1)
    draw_diurnal(ax_t, ax_u, ds)
    diurnal_legend(ax_t)
    S.panel_label(ax_t, "B", fontsize=15.0, x=-0.17, y=1.13)

    y2 = BOT + XL2
    wc = 5.45
    ax_c = S.inch_axes(fig, L, y2, wc, H2)
    draw_ratio(ax_c, sd, sn, fsum["IMBAT"])
    ax_c.text(0.5, -0.255, "H/W class, n segments", transform=ax_c.transAxes, ha="center", va="top",
              fontsize=10.08, color=P["ink"])
    S.panel_label(ax_c, "C", fontsize=15.0, x=-0.17, y=1.03)
    xd = L + wc + 1.12
    ax_d = S.inch_axes(fig, xd, y2, FW - 0.22 - xd, H2)
    draw_svf(ax_d, sd, sn)
    S.panel_label(ax_d, "D", fontsize=15.0, x=-0.17, y=1.03)

    stamp = None
    if partial:
        stamp = fig.text((FW - 0.22) / FW, 0.06 / FH, f"DRAFT \u2014 partial run, {n_tiles}/{N_TILES_TOTAL} tiles",
                         ha="right", va="bottom", fontsize=7.0, color="#8A8580", zorder=1000)
        stamp._qa_exempt = False
    panels = {"a": list(axs.values()), "b": [ax_t, ax_u], "c": [ax_c], "d": [ax_d]}

    # ------------------------------------------------------------------------------------- numbers for caption/report
    nums = dict(
        forcing={day: {k: v for k, v in fo[day].items() if not isinstance(v, np.ndarray)} for day, _ in DAYS},
        forcing_series={day: {k: [round(float(x), 2) for x in fo[day][k]] for k in ("ta", "rh", "u10", "ghi", "cs")}
                        for day, _ in DAYS},
        diurnal={f"{day}|{v}|{e}": dict(
            med=[None if not np.isfinite(x) else round(float(x), 2) for x in ds[(day, v, e)]["med"]],
            q25=[None if not np.isfinite(x) else round(float(x), 2) for x in ds[(day, v, e)]["q25"]],
            q75=[None if not np.isfinite(x) else round(float(x), 2) for x in ds[(day, v, e)]["q75"]],
            share=[round(float(x), 4) for x in ds[(day, v, e)]["share"]],
            peak_med=round(float(np.nanmax(ds[(day, v, e)]["med"])), 2),
            peak_label=int(HOURS[int(np.nanargmax(ds[(day, v, e)]["med"]))]))
            for day, _ in DAYS for v in ("tmrt", "utci") for e in EXPO},
        n_ps_cells={day: [int(x) for x in rp["nps"][day]] for day, _ in DAYS},
        tiles_used=tiles, n_tiles_used=len(tiles), n_subset=len(sub),
        hw_edges=sd["hw_edges"], svf_q_edges=sd["svf_edges"], segments=sn,
        n_segments={day: int(len(sd["fr"][day])) for day, _ in DAYS},
        wind_ref={day: dict(wd=fsum[day]["wd_mean_15_17"], R=fsum[day]["wd_ref_R"], n=fsum[day]["wd_ref_n_reports"])
                  for day, _ in DAYS},
        n_tiles_extracted=n_tiles_ext, n_tiles_utci_both=n_utci_both, n_tiles_stamp=n_tiles, partial=partial,
        fig_size_in=[FW, round(FH, 3)])
    return fig, panels, [], nums


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qa", type=Path, default=None, help="folder for QA json + 100 %% crops")
    ap.add_argument("--out", type=Path, default=None, help="override output PNG (tests)")
    a = ap.parse_args(argv)
    fig, panels, cards, nums = build(a.qa)
    png = Path(a.out) if a.out else OUT_PNG
    S.finalize_panel_labels(fig)
    n_scaled = S.scale_text(fig)
    qa = F5.qa_texts(fig, card_axes=cards, contain_axes=[])
    qa.update(n_text_scaled=n_scaled, fig_size_in=[round(float(v), 3) for v in fig.get_size_inches()], png=str(png))
    png.parent.mkdir(parents=True, exist_ok=True)
    S.save_figure(fig, png, crop=None, close=False)
    if a.qa is not None:
        a.qa.mkdir(parents=True, exist_ok=True)
        qa["crops"] = F5.save_crops(fig, png, panels, a.qa / "crops")
        (a.qa / "fig3_qa.json").write_text(json.dumps(dict(qa=qa, numbers=nums), indent=1, ensure_ascii=False,
                                                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                                           encoding="utf-8")
    plt.close(fig)
    print(json.dumps({k: qa[k] for k in ("png", "fig_size_in", "smallest_author_pt", "smallest_printed_pt_190mm",
                                         "text_text_overlaps", "text_text_touching", "clipped_outside_sheet",
                                         "spill_out_of_container")}, indent=1, ensure_ascii=False))
    return qa, nums


if __name__ == "__main__":
    main()
