"""CLIMORFA v2 Figure 7: H3, surface temperature vs pedestrian heat stress (WP4.6).

Final run (one command re-renders Fig 7 AND Fig 8 from the final result files):

    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig7_h3_surface --all
        Fig 7 reads outputs/street/analysis/h3   (run_h3.py default --out-dir)
        Fig 8 reads outputs/street/analysis/h5   (evaluate_h5.py default) + data/street/h5/S_exact (validation tiles)
        -> paper/figures/v2/main/fig_7_h3_surface.png and fig_8_h5_targeting.png, ONLY when the results are real and
           complete (FINAL, 427 tiles, both days, no PARTIAL).  Otherwise nothing is written to paper/ (see below).

Synthetic test (planted effects; stamped; can never reach paper/):

    ... build_fig7_h3_surface --results outputs/street/SYNTHETIC/fig78/h3/synthetic_PARADOX
        -> outputs/street/SYNTHETIC/fig78/fig_7_h3_surface_SYNTHETIC_PARADOX.png (+ qa/, crops/)

Inputs (exactly the files scripts/street/wsl/h3/run_h3.py writes into its --out-dir):
  h3_summary.json                      is_synthetic, per day: data_status, n_tiles, n_segments, overall rho, paradox,
                                       decoupled_citywide_share, moran_summary
  h3_spearman_<DAY>.csv                scope, lcz_label, metric_ped, rho, ci95_lower/upper, n_segments, lcz_source
  h3_compact_paradox_<DAY>.json        the lock rule on LCZ 2/3 (rho, CI, decision, diagnostics per class)
  h3_decoupled_lcz_<DAY>.csv           shares of the two decoupled types per (pooled) LCZ class + ALL
  h3_decoupled_audit_<DAY>.csv         the same per manual-audit class (optional: absent if the audit did not load)
  h3_moran_250m_cells_<DAY>.parquet    per 250 m cell: g250_id, tile_id, lcz_primary_major, lst_med, m1 (written as
                                       m1_x / m1_y by the Moran merge), n_segments_in_cell, moran columns
  h3_moran_250m_summary_<DAY>.json     global bivariate Moran's I, cells analysed
Base layers (not results): FUA outline, 250 m land grid, buildings_fua / roads_fua, 2 m tree_dsm tiles.

Panels: (a) IMBAT bivariate map at 250 m: tercile of cell LST (hue, the Fig 9 LST ramp teal -> purple) x tercile of
cell M1 (darkness toward the v2 thermal purple), 3x3 key; decoupled cells (top LST / bottom M1 and vice versa) outlined.
(b) Spearman rho(LST, M1) and rho(LST, M3) with 2 km block-bootstrap 95 % CIs, all segments, the LCZ 2+3 rule set and
each LCZ class (LCZ colours; IMBAT filled / solid, CALM open / dashed).  (c) decoupled-segment shares by LCZ and by
manual-audit class.  (d) rule-selected example 250 m cells in the Fig 9 compact-tile language.  (e) computed card.
Every number is read from the result files; the card's verdicts are re-derived from the numbers with the lock rule and
cross-checked against the JSON decisions (a mismatch stops the build).  Rule 6: a SYNTHETIC results path (or
is_synthetic in the summary) forces the SYNTHETIC watermark and the SYNTHETIC output folder.
Double-anonymised: no names, no software or plugin names in the figure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.street.figures import build_fig5_h1_scale as F5  # noqa: E402  (shared card / QA / stamp helpers)
from scripts.street.figures.build_fig5_h1_scale import (  # noqa: E402
    CHECK, CROSS, DAY_STYLE, MINUS, P, S, blend, card_frame, ctext, kfmt, num)
from scripts.street.figures.build_fig1_study_area import LCZ_COLORS, LCZ_LABEL, NATURAL, NATURAL_COLOR, lcz_dark  # noqa: E402
from scripts.street.figures.build_fig4_street_utci import add_scale_north_fig9_size, tag_box  # noqa: E402
from scripts.street.figures.build_fig6_drivers import nice_ceil  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from shapely.geometry import box as shp_box  # noqa: E402

DEFAULT_RESULTS = ROOT / "outputs" / "street" / "analysis" / "h3"
OUT_PAPER = ROOT / "paper" / "figures" / "v2" / "main"
OUT_SYN = ROOT / "outputs" / "street" / "SYNTHETIC" / "fig78"
N_TILES_TOTAL = 427
DAYS = [("IMBAT", "2022-07-05"), ("CALM", "2023-07-26")]
MAP_DAY = "IMBAT"            # (a) and (d) show the sea-breeze design day (fixed before any result; caption says so)
POOL_BUILT, POOL_NAT = 101, 102
TILE_MIN_SEGMENTS = 3        # (d) a cell is eligible only if its M1 summarises >= 3 segments
N_TILE_CLASSES = 4           # (d) up to 4 LCZ classes x 2 decoupled types = 8 tiles (the Fig 9 strip)
AUDIT_MIN_N = 30             # (c) audit classes with fewer segments are counted in a note (lock: < 30 -> pooled)

# ---- bivariate palette: hue = LST tercile on the Fig 9 LST ramp (teal cool -> purple hot), darkness = M1 tercile
#      toward the v2 thermal purple darkened 40 % to ink.  Light = low M1, dark = high M1.
PLUM = blend(S.LST_TEAL_PURPLE_COLORS[-1], P["ink"], 0.40)
_HUES = [mpl.colors.to_hex(S.LST_TEAL_PURPLE_CMAP(t)) for t in (0.0, 0.5, 1.0)]


def bivar_color(lt: int, mt: int) -> str:
    """lt, mt in {1, 2, 3} (bottom, middle, top tercile)."""
    h = _HUES[lt - 1]
    return [blend(h, "white", 0.72), blend(h, "white", 0.22), blend(h, PLUM, 0.50)][mt - 1]


HOT_COOL = bivar_color(3, 1)     # hot surface, cool street (decoupled type 1)
COOL_HOT = bivar_color(1, 3)     # cool surface, hot street (decoupled type 2)
HOT_TAG, COOL_TAG = P["pink_dark"], P["teal_dark"]   # Fig 9 'Hot' / 'Cool' tag colours (fig9:518)
DEC_EDGE = P["ink"]


def lcz_code(label: str):
    """'LCZ 2: Compact midrise' -> 2, 'LCZ 11' -> 11, 'LCZ A: ...' -> 11, pooled -> 101 / 102."""
    s = str(label)
    if "Other built" in s:
        return POOL_BUILT
    if "Other natural" in s:
        return POOL_NAT
    m = re.match(r"LCZ\s+([0-9]+|[A-G])\b", s)
    if not m:
        return None
    v = m.group(1)
    return int(v) if v.isdigit() else 11 + "ABCDEFG".index(v)


def audit_code(label: str):
    """'2_compact_midrise' -> 2, 'A_dense_trees' -> 11, 'mixed' -> None."""
    t = str(label).split("_")[0]
    if t.isdigit():
        return int(t)
    if len(t) == 1 and t in "ABCDEFG":
        return 11 + "ABCDEFG".index(t)
    return None


def kdark(k) -> str:
    """Dark text / edge colour of a class; classes without a Fig 1 colour (absent from the FUA map, e.g. 1, 7) fall
    back to taupe_dark."""
    k = int(k)
    return lcz_dark(k) if k in LCZ_COLORS else P["taupe_dark"]


def cls_label(k) -> str:
    if k == POOL_BUILT:
        return "other built"
    if k == POOL_NAT:
        return "other natural"
    return LCZ_LABEL.get(int(k), str(k))


def cls_color(k) -> str:
    if k in (POOL_BUILT, POOL_NAT):
        return P["taupe"] if k == POOL_BUILT else NATURAL_COLOR
    return NATURAL_COLOR if k in NATURAL else LCZ_COLORS.get(int(k), P["taupe"])


def cls_edge(k) -> str:
    if k in (POOL_NAT,) or k in NATURAL:
        return P["taupe_dark"]
    if k == POOL_BUILT:
        return P["taupe_dark"]
    return kdark(k)


def cls_line(k) -> str:
    """CI line colour: the class colour, except pale classes (natural, LCZ 3) which take their dark edge."""
    if k in NATURAL or k in (POOL_NAT, POOL_BUILT):
        return P["taupe_dark"]
    c = LCZ_COLORS.get(int(k), P["taupe"])
    r, g, b = mpl.colors.to_rgb(c)
    return kdark(k) if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.70 else c


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f} %"


# =============================================================================================
# result location / status
# =============================================================================================
class H3Results:
    def __init__(self, results: Path, allow_partial: bool):
        root = Path(results).resolve()
        if not (root / "h3_summary.json").exists() and (root / "results" / "h3_summary.json").exists():
            root = root / "results"
        if not (root / "h3_summary.json").exists():
            raise FileNotFoundError(f"{root}: no h3_summary.json (run scripts/street/wsl/h3/run_h3.py first)")
        self.root = root
        self.used: dict[str, str] = {}
        self.summary = self._json("h3_summary.json")
        path_syn = "SYNTHETIC" in str(root).upper()
        flag_syn = bool(self.summary.get("is_synthetic", False))
        if flag_syn and not path_syn:
            raise RuntimeError(f"rule 6: {root} holds is_synthetic results but is not a SYNTHETIC folder")
        if path_syn and not flag_syn:
            raise RuntimeError(f"{root}: SYNTHETIC folder but is_synthetic = False in h3_summary.json")
        self.synthetic = flag_syn
        days = self.summary.get("days_analyzed", [])
        if set(days) != {d for d, _ in DAYS}:
            raise RuntimeError(f"H3 results for {days} only; Figure 7 needs both design days")
        rbd = self.summary["results_by_day"]
        self.status = {d: rbd[d]["data_status"] for d, _ in DAYS}
        self.n_tiles = {d: int(rbd[d]["n_tiles"]) for d, _ in DAYS}
        self.final = (not self.synthetic and all(s.startswith("FINAL") for s in self.status.values())
                      and all(n == N_TILES_TOTAL for n in self.n_tiles.values()))
        self.partial = not self.synthetic and not self.final
        if self.partial and not allow_partial:
            raise SystemExit(f"H3 results are not final ({self.status}); nothing written. "
                             "--allow-partial renders a stamped DRAFT into outputs/street/SYNTHETIC/fig78/.")
        sub = root.name.replace("synthetic_", "")
        self.tag = "_" + (root.parent.name if sub in ("test", "results") else sub)

    def _json(self, name: str) -> dict:
        p = self.root / name
        self.used[name] = str(p)
        return json.loads(p.read_text(encoding="utf-8"))

    def csv(self, name: str, required: bool = True) -> pd.DataFrame | None:
        p = self.root / name
        if not p.exists():
            if required:
                raise FileNotFoundError(p)
            return None
        self.used[name] = str(p)
        return pd.read_csv(p)

    def parquet(self, name: str) -> pd.DataFrame:
        p = self.root / name
        self.used[name] = str(p)
        df = pd.read_parquet(p)
        if "is_synthetic" in df.columns and bool(df["is_synthetic"].astype(bool).any()) != self.synthetic:
            raise RuntimeError(f"rule 6: is_synthetic column of {p} disagrees with the summary")
        return df

    def out_png(self, stem: str, out: Path | None) -> Path:
        if out is not None:
            out = Path(out)
            if self.synthetic and "SYNTHETIC" not in str(out.resolve()).upper():
                raise RuntimeError("rule 6: synthetic figure must be written under a SYNTHETIC folder")
            if not self.final and OUT_PAPER.resolve() in out.resolve().parents:
                raise RuntimeError("only final, complete results may be written to paper/")
            return out
        if self.final:
            return OUT_PAPER / f"{stem}.png"
        if self.synthetic:
            return OUT_SYN / f"{stem}_SYNTHETIC{self.tag}.png"
        return OUT_SYN / f"{stem}_PARTIAL.png"

    def qa_dir(self, qa: Path | None) -> Path | None:
        if qa is not None:
            return Path(qa)
        return None if self.final else OUT_SYN / "qa"


def stamp(fig, rs, label: str) -> list:
    """Fig 5/6 stamp: SYNTHETIC watermark + footer, or PARTIAL draft stamp (exempt from overlap QA)."""
    arts = []
    if rs.synthetic:
        arts.append(fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", rotation=28, fontsize=150,
                             fontweight="bold", color="#BDBDBD", alpha=0.16, zorder=1000))
        arts.append(fig.text(0.995, 0.004, f"SYNTHETIC planted-effect test data (scenario {rs.tag.strip('_')}), "
                             "not observations", ha="right", va="bottom", fontsize=7.0, color="#8A8580", zorder=1000))
    elif rs.partial:
        arts.append(fig.text(0.5, 0.5, "PARTIAL", ha="center", va="center", rotation=28, fontsize=150,
                             fontweight="bold", color="#9A9A9A", alpha=0.20, zorder=1000))
        arts.append(fig.text(0.995, 0.004, "DRAFT \u2014 " + label, ha="right", va="bottom", fontsize=7.0,
                             color="#8A8580", zorder=1000))
    for a in arts:
        a._qa_exempt = True
    return arts


def finish(fig, rs, png: Path, panels: dict, card_axes, contain_axes, qa_extra: dict, qa_dir: Path | None) -> dict:
    """letters -> (a); v2 type scale; QA after scaling; save (full 13.0 in sheet, no crop); 100 % crops."""
    S.finalize_panel_labels(fig)
    n_scaled = S.scale_text(fig)
    qa = F5.qa_texts(fig, card_axes=card_axes, contain_axes=contain_axes)
    fw, fh = fig.get_size_inches()
    qa.update(n_text_scaled=n_scaled, fig_size_in=[round(float(fw), 3), round(float(fh), 3)], fh_ok=bool(fh <= 13.5),
              png=str(png), inputs=rs.used, synthetic=rs.synthetic, partial=rs.partial, final=rs.final, **qa_extra)
    png.parent.mkdir(parents=True, exist_ok=True)
    S.save_figure(fig, png, crop=None, close=False)
    if qa_dir is not None:
        qa["crops"] = F5.save_crops(fig, png, panels, qa_dir.parent / "crops" if qa_dir.name == "qa" else qa_dir / "crops")
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / f"{png.stem}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    plt.close(fig)
    return qa


def base_layers():
    study = gpd.read_file(ROOT / "data/02_interim/study_area_fua.gpkg", layer="study_area_fua")
    land = gpd.read_file(ROOT / "data/03_processed/analysis_grids.gpkg", layer="grid_250m", columns=["grid_id"])
    land = gpd.GeoDataFrame(geometry=[shapely.union_all(land.geometry.values)], crs=land.crs)
    return study, land


def map_frame(fig, x_in, y_in, w_in, study):
    extent = tuple(study.total_bounds)
    ar = (extent[3] - extent[1]) / (extent[2] - extent[0])
    ax = S.inch_axes(fig, x_in, y_in, w_in, w_in * ar)
    return ax, extent, w_in * ar


# =============================================================================================
# data
# =============================================================================================
def tercile(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The tercile rule of decoupling.identify_decoupled_segments (<= q1 -> 1, <= q2 -> 2, else 3)."""
    q1, q2 = float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))
    return np.where(v <= q1, 1, np.where(v <= q2, 2, 3)), q1, q2


def load(rs: H3Results) -> dict:
    d = dict(spear={}, par={}, dec_lcz={}, dec_aud={}, moran={})
    for day, _ in DAYS:
        sp = rs.csv(f"h3_spearman_{day}.csv")
        sp = sp[sp["lcz_source"] == "lcz_primary"].copy()
        sp["code"] = [None if sc == "overall" else lcz_code(lab) for sc, lab in zip(sp["scope"], sp["lcz_label"])]
        sp["metric"] = np.where(sp["metric_ped"].str.startswith("M1"), "M1", "M3")
        d["spear"][day] = sp
        d["par"][day] = rs._json(f"h3_compact_paradox_{day}.json")
        dl = rs.csv(f"h3_decoupled_lcz_{day}.csv")
        dl["code"] = [None if g == "ALL" else int(float(g)) if str(g).replace(".", "").isdigit() else lcz_code(lab)
                      for g, lab in zip(dl["group"], dl["lcz_label"])]
        d["dec_lcz"][day] = dl
        d["dec_aud"][day] = rs.csv(f"h3_decoupled_audit_{day}.csv", required=False)
        d["moran"][day] = rs._json(f"h3_moran_250m_summary_{day}.json")
        for stt in (sp["data_status"].iloc[0], dl["data_status"].iloc[0], d["par"][day]["data_status"]):
            if stt != rs.status[day]:
                raise RuntimeError(f"{day}: data_status mismatch between files ({stt!r} vs {rs.status[day]!r})")
    cells = rs.parquet(f"h3_moran_250m_cells_{MAP_DAY}.parquet")
    m1col = next(c for c in ("m1", "m1_y", "m1_x") if c in cells.columns)
    cells = cells.rename(columns={m1col: "m1_cell"})
    ok = np.isfinite(cells["lst_med"].to_numpy(float)) & np.isfinite(cells["m1_cell"].to_numpy(float))
    cells = cells[ok].copy()
    n_an = int(d["moran"][MAP_DAY].get("n_grid_cells_analyzed", -1))
    if len(cells) != n_an:
        raise RuntimeError(f"(a) cell set ({len(cells)}) != Moran cells analysed ({n_an})")
    cells["lt"], lq1, lq2 = tercile(cells["lst_med"].to_numpy(float))
    cells["mt"], mq1, mq2 = tercile(cells["m1_cell"].to_numpy(float))
    xy = cells["g250_id"].str.split("_", expand=True).astype(int) * 250
    cells["x0"], cells["y0"] = xy[0].to_numpy(), xy[1].to_numpy()
    d["cells"] = cells
    d["cuts"] = dict(lst=(lq1, lq2), m1=(mq1, mq2))
    return d


def crosscheck(rs: H3Results, d: dict) -> dict:
    """Lock rule re-derived from the numbers and compared with every place the result files state it."""
    out = {}
    rbd = rs.summary["results_by_day"]
    for day, _ in DAYS:
        pj = d["par"][day]
        rho, lo, hi = pj["rho"], pj["ci95_lower"], pj["ci95_upper"]
        rule = bool(np.isfinite(rho) and np.isfinite(hi) and rho < 0 and hi < 0)
        js = bool(pj["compact_paradox_observed"])
        if rule != js or (pj["decision"] == "SUPPORTED") != rule or bool(rbd[day]["compact_paradox"]["compact_paradox_observed"]) != rule:
            raise RuntimeError(f"H3 compact-paradox mismatch on {day}: rule {rule}, JSON {js}, decision {pj['decision']}")
        sp = d["spear"][day]
        for k in (2, 3):
            row = sp[(sp["code"] == k) & (sp["metric"] == "M1")]
            diag = pj.get(f"diagnostic_{k}", {})
            if len(row) and np.isfinite(diag.get("rho", np.nan)) and abs(float(row["rho"].iloc[0]) - diag["rho"]) > 1e-9:
                raise RuntimeError(f"{day}: forest LCZ {k} rho {float(row['rho'].iloc[0])} != paradox diagnostic {diag['rho']}")
        for met, key in (("M1", "spearman_overall_m1"), ("M3", "spearman_overall_m3")):
            row = sp[(sp["scope"] == "overall") & (sp["metric"] == met)].iloc[0]
            if abs(float(row["rho"]) - rbd[day][key]["rho"]) > 1e-9:
                raise RuntimeError(f"{day}: overall {met} rho differs between CSV and summary")
        dl = d["dec_lcz"][day]
        if abs(float(dl.loc[dl["group"] == "ALL", "share_decoupled"].iloc[0]) - rbd[day]["decoupled_citywide_share"]) > 1e-12:
            raise RuntimeError(f"{day}: decoupled share differs between CSV and summary")
        out[day] = rule
    return out


def pick_tiles(cells: pd.DataFrame) -> list[dict]:
    """(d) rule: eligible cells have a valid LCZ (A6) and >= TILE_MIN_SEGMENTS segments.  For each class, the 'hot
    surface / cool street' example is the (LST T3, M1 T1) cell with the largest LST-rank minus M1-rank (percentile
    ranks over all mapped cells), and 'cool surface / hot street' the (T1, T3) cell with the largest M1-rank minus
    LST-rank.  Classes holding both types are ranked by their number of decoupled cells; the top N_TILE_CLASSES are
    shown in class order."""
    c = cells.copy()
    c["rl"] = c["lst_med"].rank(pct=True)
    c["rm"] = c["m1_cell"].rank(pct=True)
    c["k"] = pd.to_numeric(c["lcz_primary_major"], errors="coerce")
    nseg = c["n_segments_in_cell"] if "n_segments_in_cell" in c.columns else pd.Series(99, index=c.index)
    el = c[(c["k"].fillna(0) > 0) & (nseg >= TILE_MIN_SEGMENTS)]
    cand = []
    for k, g in el.groupby("k"):
        a = g[(g["lt"] == 3) & (g["mt"] == 1)]
        b = g[(g["lt"] == 1) & (g["mt"] == 3)]
        if len(a) and len(b):
            cand.append((len(a) + len(b), int(k), a.loc[(a["rl"] - a["rm"]).idxmax()], b.loc[(b["rm"] - b["rl"]).idxmax()]))
    cand.sort(key=lambda t: (-t[0], t[1]))
    sel = sorted(cand[:N_TILE_CLASSES], key=lambda t: t[1])
    out = []
    for n, k, a, b in sel:
        for kind, r in (("hot_cool", a), ("cool_hot", b)):
            out.append(dict(klass=k, kind=kind, g250_id=r["g250_id"], tile_id=r.get("tile_id"), x0=float(r["x0"]),
                            y0=float(r["y0"]), lst=float(r["lst_med"]), m1=float(r["m1_cell"]), lt=int(r["lt"]),
                            mt=int(r["mt"]), n_decoupled_in_class=int(n),
                            n_seg=int(r["n_segments_in_cell"]) if "n_segments_in_cell" in r else None))
    return out


# =============================================================================================
# panels
# =============================================================================================
def draw_map(ax, cells, study, land, extent):
    S.draw_land_cells(ax, land)
    x0 = np.floor(extent[0] / 250) * 250
    y1 = np.ceil(extent[3] / 250) * 250
    nx = int(np.ceil((extent[2] - x0) / 250))
    ny = int(np.ceil((y1 - extent[1]) / 250))
    img = np.zeros((ny, nx, 4))
    ci = ((cells["x0"].to_numpy() - x0) // 250).astype(int)
    ri = ((y1 - cells["y0"].to_numpy() - 250) // 250).astype(int)
    for (lt, mt), g in cells.groupby(["lt", "mt"]):
        rgba = mpl.colors.to_rgba(bivar_color(int(lt), int(mt)))
        m = cells.index.isin(g.index)
        img[ri[m], ci[m]] = rgba
    ax.imshow(img, extent=[x0, x0 + nx * 250, y1 - ny * 250, y1], interpolation="nearest", zorder=3)
    for (lt, mt) in ((3, 1), (1, 3)):
        g = cells[(cells["lt"] == lt) & (cells["mt"] == mt)]
        if len(g):
            u = shapely.union_all([shp_box(a, b, a + 250, b + 250) for a, b in zip(g["x0"], g["y0"])])
            gpd.GeoSeries([u]).boundary.plot(ax=ax, color=DEC_EDGE, linewidth=0.55, zorder=4)
    S.draw_study_outline(ax, study, "ink")
    S.set_map_extent(ax, extent, exact=True)


def draw_key(ax, cuts):
    """3x3 key: x = LST tercile, y = M1 tercile (tercile bounds as tick labels); decoupled corners outlined."""
    for lt in (1, 2, 3):
        for mt in (1, 2, 3):
            ax.add_patch(mpatches.Rectangle((lt - 1, mt - 1), 1, 1, facecolor=bivar_color(lt, mt), edgecolor="white",
                                            linewidth=0.8, zorder=2))
    for lt, mt in ((3, 1), (1, 3)):
        ax.add_patch(mpatches.Rectangle((lt - 1 + 0.05, mt - 1 + 0.05), 0.90, 0.90, fill=False, edgecolor=DEC_EDGE,
                                        linewidth=0.9, zorder=3))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal")
    ax.set_xticks([1, 2]); ax.set_yticks([1, 2])
    ax.set_xticklabels([f"{v:.1f}" for v in cuts["lst"]], fontsize=6.6)
    for t_, ha in zip(ax.get_xticklabels(), ("right", "left")):   # staggered off the two inner edges
        t_.set_ha(ha)
    ax.set_yticklabels([f"{v:.1f}" for v in cuts["m1"]], fontsize=6.6)
    ax.tick_params(length=2.5, pad=1.5, colors=P["ink"], bottom=True, left=True)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("LST (°C) →", fontsize=6.8, fontweight="bold", labelpad=2)
    ax.set_ylabel("M1 (°C) →", fontsize=6.8, fontweight="bold", labelpad=2)


def draw_forest(axes, d):
    """axes: {'M1': ax, 'M3': ax}; rows top-down: LCZ 2+3 rule set (M1 only), all segments, classes."""
    sp0 = d["spear"]["IMBAT"]
    codes = [c for c in sp0["code"].dropna().unique()]
    built = sorted(int(c) for c in codes if int(c) < 11)
    nat = sorted(int(c) for c in codes if 11 <= int(c) < 100)
    pooled = sorted(int(c) for c in codes if int(c) >= 100)
    rows = ["rule", "all"] + built + nat + pooled
    ypos = {r: i for i, r in enumerate(rows)}
    off = {"IMBAT": -0.17, "CALM": 0.17}
    rule_col = blend(LCZ_COLORS[2], LCZ_COLORS[3], 0.5)
    nlab = {}
    for met, ax in axes.items():
        ax.add_patch(mpatches.Rectangle((-1.1, -0.5), 2.2, 1.0, facecolor=P["taupe_light"], edgecolor="none", zorder=0))
        ax.axvline(0.0, color=S.SPINE_GREY, ls=":", lw=1.0, zorder=1)
        ax.axhline(1.5, color=P["grid"], lw=1.0, zorder=1)
        for day, _ in DAYS:
            st = DAY_STYLE[day]
            sp = d["spear"][day]
            sp = sp[sp["metric"] == met]
            for r in rows:
                if r == "rule":
                    if met != "M1":
                        continue
                    pj = d["par"][day]
                    rho, lo, hi, n = pj["rho"], pj["ci95_lower"], pj["ci95_upper"], pj["n_segments"]
                    col, edge, line = rule_col, lcz_dark(2), lcz_dark(2)
                elif r == "all":
                    q = sp[sp["scope"] == "overall"].iloc[0]
                    rho, lo, hi, n = q["rho"], q["ci95_lower"], q["ci95_upper"], q["n_segments"]
                    col, edge, line = st["color"], st["color"], st["color"]
                else:
                    q = sp[sp["code"] == r]
                    if not len(q):
                        continue
                    q = q.iloc[0]
                    rho, lo, hi, n = q["rho"], q["ci95_lower"], q["ci95_upper"], q["n_segments"]
                    col, edge, line = cls_color(r), cls_edge(r), cls_line(r)
                if day == "IMBAT":
                    nlab[r] = int(n)
                y = ypos[r] + off[day]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [y, y], color=line, lw=1.5, ls=st["ls"], solid_capstyle="butt", zorder=3)
                ax.scatter([rho], [y], s=40, marker="D" if r == "rule" else "o",
                           facecolor=col if day == "IMBAT" else "white", edgecolor=edge, linewidth=1.1, zorder=4)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticks([-0.5, 0.5], minor=True)
        ax.grid(True, axis="x", which="minor", color=P["grid"], lw=0.7)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-9 else f"{v:g}".replace("-", MINUS)))
        ax.grid(True, axis="x", color=P["grid"], lw=0.7)
        ax.grid(False, axis="y")
        for s_ in ax.spines.values():
            s_.set_color(P["grid"]); s_.set_linewidth(1.0)
        ax.tick_params(axis="x", length=0, labelsize=8.2, colors=P["muted"])
        ax.set_xlabel(f"ρ(LST, {met})", fontsize=8.6, color=P["ink"], labelpad=3)
    ax1 = axes["M1"]
    ax1.set_yticks(range(len(rows)))
    labs = ["LCZ 2 + 3", "all segments"] + [f"LCZ {cls_label(k)}" if k < 100 else cls_label(k) for k in rows[2:]]
    ax1.set_yticklabels(labs, fontsize=8.2)
    ax1.tick_params(axis="y", length=0, pad=3, colors=P["ink"])
    ax1.get_yticklabels()[0].set_fontweight("bold")
    ax3 = axes["M3"]
    ax3.set_yticks(range(len(rows)))
    ax3.tick_params(axis="y", left=False, labelleft=False)
    for r in rows:
        if r in nlab:
            ax3.text(1.04, ypos[r], kfmt(nlab[r]), transform=ax3.get_yaxis_transform(), ha="left", va="center",
                     fontsize=6.6, color=P["muted"], clip_on=False)
    ax3.text(1.04, 1.0, "n", transform=ax3.transAxes, ha="left", va="bottom", fontsize=6.6, color=P["muted"],
             fontweight="bold", clip_on=False)
    ax3.text(-0.04, 0.0, "← lock: ρ and CI < 0", ha="center", va="center", fontsize=6.6,
             color=P["muted"], style="italic", zorder=5)
    return rows


def sub_badge(ax, text):
    """Facet tag of Fig 5 (7.2 pt bold ink, white, taupe edge), inside the upper right."""
    ax.text(0.995, 0.955, text, transform=ax.transAxes, ha="right", va="top", fontsize=7.2, fontweight="bold",
            color=P["ink"], zorder=10,
            bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.25", "alpha": 0.95})


def draw_shares(ax, dfs: dict, groups: list, labels: list):
    """Stacked compact bars: cool surface / hot street (bottom) and hot surface / cool street (top);
    IMBAT filled, CALM hatched open (the filled / open day code of Figs 5/6)."""
    w, off = 0.36, {"IMBAT": -0.20, "CALM": 0.20}
    ymax = 0.0
    for day, _ in DAYS:
        df = dfs[day]
        for i, g in enumerate(groups):
            row = df[g(df)]
            if not len(row):
                continue
            row = row.iloc[0]
            a, b = float(row["share_cool_surf_hot_ped"]), float(row["share_hot_surf_cool_ped"])
            x = i + off[day]
            for y0, h, c in ((0.0, a, COOL_HOT), (a, b, HOT_COOL)):
                edge = blend(c, P["ink"], 0.35)
                if day == "IMBAT":
                    ax.bar(x, 100 * h, w, bottom=100 * y0, color=c, edgecolor=edge, linewidth=0.6, zorder=3)
                else:
                    ax.bar(x, 100 * h, w, bottom=100 * y0, facecolor="white", edgecolor=edge, linewidth=0.6,
                           hatch="//////", zorder=3)
            ymax = max(ymax, 100 * (a + b))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=8.2, color=P["ink"])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.grid(True, axis="y", color=P["grid"], lw=0.7)
    ax.grid(False, axis="x")
    for s_ in ax.spines.values():
        s_.set_color(P["grid"]); s_.set_linewidth(1.0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=8.2, colors=P["muted"], length=0)
    return ymax


def draw_tile(ax, t):
    """Fig 9 compact tile (fig9:461-545): canopy raster, taupe_dark roads, LCZ-coloured buildings, ink cell outline;
    top-left tag = cell LST, bottom-left tag = cell M1; the tag colour is the cell's tercile (Hot pink_dark /
    Cool teal_dark, the Fig 9 tag colours)."""
    x0, y0 = t["x0"], t["y0"]
    geom = shp_box(x0, y0, x0 + 250, y0 + 250)
    bounds = (x0, y0, x0 + 250, y0 + 250)
    ax.set_facecolor(S.TILE_BG_FIG9)
    tid = t.get("tile_id")
    p = ROOT / "data/street/tiles/2m/tree_dsm" / f"{tid}.tif"
    if tid and p.exists():
        with rasterio.open(p) as src:
            win = from_bounds(*bounds, transform=src.transform)
            arr = src.read(1, window=win)
            wb = rasterio.windows.bounds(win, src.transform)
        m = np.ma.masked_where(~np.isfinite(arr) | (arr < S.CANOPY_VMIN) | (arr > 100), arr)
        if m.count():
            ax.imshow(m, cmap=S.CANOPY_CMAP_10M, vmin=S.CANOPY_VMIN, vmax=S.CANOPY_VMAX, alpha=S.CANOPY_ALPHA,
                      extent=[wb[0], wb[2], wb[1], wb[3]], zorder=2, interpolation="nearest")
    roads = S.clip_to_cell(S.safe_read_layer(ROOT, "roads_fua", bounds), geom)
    blds = S.clip_to_cell(S.safe_read_layer(ROOT, "buildings_fua", bounds), geom)
    if not roads.empty:
        roads.plot(ax=ax, color=P["taupe_dark"], linewidth=0.45, alpha=0.75, zorder=3)
    if not blds.empty:
        blds.plot(ax=ax, facecolor=cls_color(t["klass"]), edgecolor=P["ink"], linewidth=0.15, alpha=0.92, zorder=4)
    gpd.GeoSeries([geom]).boundary.plot(ax=ax, color=P["ink"], linewidth=0.9, zorder=5)
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect("equal"); ax.axis("off")
    lc = HOT_TAG if t["lt"] == 3 else COOL_TAG
    mc = HOT_TAG if t["mt"] == 3 else COOL_TAG
    S.tile_tag(ax, f"LST {t['lst']:.1f}°C", lc, x=0.05, y=0.95)
    S.tile_tag(ax, f"M1 {t['m1']:.1f}°C", mc, x=0.05, y=0.05)
    ax.texts[-1].set_va("bottom")
    return dict(n_buildings=int(len(blds)), n_roads=int(len(roads)))


def draw_card(ax, rs, d, verdict, h_in: float = 3.06):
    """Row spacing is authored for a 3.06 in card and stretched by at most 45 % when the card is taller."""
    card_frame(ax)
    u = 3.06 / h_in * min(h_in / 3.06, 1.45)
    xl, xd = 0.035, {"IMBAT": 0.600, "CALM": 0.850}
    y = 1 - 0.065 * u
    for day, _ in DAYS:
        ctext(ax, xd[day], y, day, ha="center")
    rbd = rs.summary["results_by_day"]
    dy_row, dy_ci = 0.122 * u, 0.050 * u
    y -= 0.105 * u
    rows = [("ρ(LST, M1), all", lambda day: (rbd[day]["spearman_overall_m1"]["rho"],
                                                   rbd[day]["spearman_overall_m1"]["ci95"]), False),
            ("ρ(LST, M3), all", lambda day: (rbd[day]["spearman_overall_m3"]["rho"],
                                                   rbd[day]["spearman_overall_m3"]["ci95"]), False),
            ("ρ(LST, M1), LCZ 2 + 3", lambda day: (d["par"][day]["rho"],
                                                         [d["par"][day]["ci95_lower"], d["par"][day]["ci95_upper"]]), True)]
    for lab, f, rule in rows:
        ctext(ax, xl, y, lab)
        if rule:
            ctext(ax, xl, y - dy_ci, "lock: ρ < 0 and CI < 0", size=7.0, weight="normal", color=P["muted"])
        for day, _ in DAYS:
            v, ci = f(day)
            mark = ("  " + (CHECK if verdict[day] else CROSS)) if rule else ""
            ctext(ax, xd[day], y, f"{num(v, 2)}{mark}", ha="center")
            ctext(ax, xd[day], y - dy_ci, F5.ci_txt(ci), size=7.0, weight="normal", color=P["muted"], ha="center")
        y -= dy_row
    ctext(ax, xl, y, "decoupled segments")
    ctext(ax, xl, y - dy_ci, "LST / M1 terciles opposite", size=7.0, weight="normal", color=P["muted"])
    for day, _ in DAYS:
        ctext(ax, xd[day], y, pct(rbd[day]["decoupled_citywide_share"]), ha="center")
    y -= dy_row
    ctext(ax, xl, y, "Moran’s I (250 m)")
    ctext(ax, xl, y - dy_ci, "bivariate, LST vs M1", size=7.0, weight="normal", color=P["muted"])
    for day, _ in DAYS:
        m = d["moran"][day]
        ctext(ax, xd[day], y, f"{num(m['global_bivariate_moran_I'], 2)}", ha="center")
        ctext(ax, xd[day], y - dy_ci, f"p = {m['global_bivariate_moran_p_sim']:.3f}", size=7.0, weight="normal",
              color=P["muted"], ha="center")
    y -= 0.108 * u
    nseg = {rbd[day]["n_segments"] for day, _ in DAYS}
    ncel = {d["moran"][day]["n_grid_cells_analyzed"] for day, _ in DAYS}
    seg_t = f"{next(iter(nseg)):,}" if len(nseg) == 1 else " / ".join(f"{rbd[day]['n_segments']:,}" for day, _ in DAYS)
    cel_t = f"{next(iter(ncel)):,}" if len(ncel) == 1 else " / ".join(
        f"{d['moran'][day]['n_grid_cells_analyzed']:,}" for day, _ in DAYS)
    ctext(ax, xl, y, f"n = {seg_t} segments, {cel_t} cells of 250 m", size=7.0, weight="normal", color=P["muted"])
    y -= 0.055 * u
    ax.plot([xl, 1 - xl], [y, y], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    y -= 0.062 * u
    obs = [day for day, _ in DAYS if verdict[day]]
    txt = ("observed on both days" if len(obs) == 2 else f"observed on {obs[0]} only" if len(obs) == 1
           else "not observed")
    ctext(ax, xl, y, "Compact paradox, lock rule:")
    ctext(ax, 1 - xl, y - 0.068 * u, txt, ha="right")
    return txt


# =============================================================================================
def build(rs: H3Results):
    S.apply_style()
    np.random.seed(20261030)
    d = load(rs)
    verdict = crosscheck(rs, d)
    study, land = base_layers()
    tiles = pick_tiles(d["cells"])

    FW = S.V2_FIG_WIDTH_IN
    L, R, T = 0.50, 0.50, 0.15
    MAP_W = 6.85
    extent = tuple(study.total_bounds)
    ar = (extent[3] - extent[1]) / (extent[2] - extent[0])
    MAP_H = MAP_W * ar
    KEY = 0.98
    TAG_BAND = 1.52
    GAP_AC = 0.62
    C_ROW, C_GAP = 1.05, 0.66
    C_H = 2 * C_ROW + C_GAP
    C_XL = 0.36
    GAP_CD = 0.55
    TILE_GAP = S.TILE_STRIP["tile_gap"]
    n_t = max(len(tiles), 1)
    TILE = (FW - L - R - 7 * TILE_GAP) / 8
    D_HDR, D_LEG, BOT = 0.34, 0.42, 0.22
    FH = T + MAP_H + TAG_BAND + GAP_AC + C_H + C_XL + GAP_CD + D_HDR + TILE + D_LEG + BOT
    fig = plt.figure(figsize=(FW, FH))

    # (a) map, scale bar / north arrow in the gulf (Fig 4 position), key + computed tag under the map
    y_map = FH - T - MAP_H
    axA = S.inch_axes(fig, L, y_map, MAP_W, MAP_H)
    draw_map(axA, d["cells"], study, land, extent)
    add_scale_north_fig9_size(axA, base_loc=(0.02, 0.42), scale_len_km=10.0)
    S.panel_label(axA, "A", fontsize=15.0, y=0.98, va="top")
    axK = S.inch_axes(fig, L + 0.42, y_map - 0.14 - KEY, KEY, KEY)
    draw_key(axK, d["cuts"])
    cells = d["cells"]
    n_c = len(cells)
    s31 = float(((cells["lt"] == 3) & (cells["mt"] == 1)).mean())
    s13 = float(((cells["lt"] == 1) & (cells["mt"] == 3)).mean())
    date = dict(DAYS)[MAP_DAY]
    tag_box(axA, f"{MAP_DAY} · {date} · {n_c:,} cells of 250 m with LST and M1\n"
                 f"outlined = decoupled: hot surface / cool street {pct(s31)},\n"
                 f"cool surface / hot street {pct(s13)} of the cells",
            x=1.92 / MAP_W, y=-0.035, va="top")

    # (b) forest
    xb0 = L + MAP_W + 1.55
    fgap = 0.30
    fw = (FW - R - 0.42 - xb0 - fgap) / 2
    yb = y_map + 0.40
    hb = MAP_H - 0.40 - 0.42
    axB = {"M1": S.inch_axes(fig, xb0, yb, fw, hb), "M3": S.inch_axes(fig, xb0 + fw + fgap, yb, fw, hb)}
    rows = draw_forest(axB, d)
    axB["M3"].legend(handles=F5.day_legend_handles(), loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
                     **S.LEGEND_LINES_SMALL)
    S.panel_label(axB["M1"], "B", fontsize=15.0, x=-0.70, y=1.01)

    # (c) shares by LCZ (top) and by manual-audit class (bottom)
    yc = y_map - TAG_BAND - GAP_AC - C_H
    CW = MAP_W - 0.55
    dl = d["dec_lcz"]
    codes = [c for c in dl["IMBAT"]["code"].dropna().unique()]
    order = sorted(int(c) for c in codes if int(c) < 11) + sorted(int(c) for c in codes if 11 <= int(c) < 100) + \
        sorted(int(c) for c in codes if int(c) >= 100)
    axC1 = S.inch_axes(fig, L + 0.55, yc + C_ROW + C_GAP, CW, C_ROW)
    groups = [lambda df: df["group"] == "ALL"] + [(lambda k: (lambda df: df["code"] == k))(k) for k in order]
    labels = ["all"] + [{POOL_BUILT: "oth. b.", POOL_NAT: "oth. n."}.get(k, cls_label(k)) for k in order]
    ym1 = draw_shares(axC1, dl, groups, labels)
    sub_badge(axC1, "by LCZ class")
    S.panel_label(axC1, "C", fontsize=15.0, x=-0.085, y=1.18)
    da = d["dec_aud"]
    aud_meta = dict(present=all(v is not None for v in da.values()))
    axC2 = S.inch_axes(fig, L + 0.55, yc, CW, C_ROW)
    ym2 = 0.0
    if aud_meta["present"] and int(da["IMBAT"].loc[da["IMBAT"]["audit_label"] == "ALL_AUDITED", "n_segments"].iloc[0]) > 0:
        a0 = da["IMBAT"]
        cls_rows = a0[~a0["audit_label"].str.startswith("ALL_")]
        keep = cls_rows[cls_rows["n_segments"] >= AUDIT_MIN_N]["audit_label"].tolist()
        keep = sorted(keep, key=lambda s_: (audit_code(s_) is None, audit_code(s_) or 999))
        small = cls_rows[cls_rows["n_segments"] < AUDIT_MIN_N]
        g2 = [lambda df: df["audit_label"] == "ALL_AUDITED"] + \
            [(lambda s_: (lambda df: df["audit_label"] == s_))(s_) for s_ in keep]
        lab2 = ["all"] + [cls_label(audit_code(s_)) if audit_code(s_) else s_ for s_ in keep]
        ym2 = draw_shares(axC2, da, g2, lab2)
        n_aud = int(a0.loc[a0["audit_label"] == "ALL_AUDITED", "n_segments"].iloc[0])
        sub_badge(axC2, f"by manual-audit class · {n_aud:,} segments in audited cells")
        if len(small):
            axC2.text(1.0, -0.36, f"{len(small)} audit classes with < {AUDIT_MIN_N} segments not shown",
                      transform=axC2.transAxes, ha="right", va="top", fontsize=6.6, color=P["muted"])
        aud_meta.update(shown=keep, classes_not_shown=small["audit_label"].tolist(), n_audited=n_aud)
    else:
        axC2.set_axis_off()
        axC2.text(0.5, 0.5, "no manual-audit cells in this run", ha="center", va="center", fontsize=7.6,
                  color=P["muted"], transform=axC2.transAxes)
        aud_meta["shown"] = []
    ytop = np.ceil(max(ym1, ym2, 1.0) * 1.42 / 10.0) * 10.0     # nearest 10: nice_ceil jumped to 100 and hit the label
    for ax in (axC1, axC2):
        if ax.axison:
            ax.set_ylim(0, ytop)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    fig.text((L + 0.02) / FW, (yc + C_H / 2) / FH, "Decoupled segments (%)", rotation=90, ha="left", va="center",
             fontsize=9.24, color=P["ink"])
    hc = [mpatches.Patch(facecolor=HOT_COOL, edgecolor=blend(HOT_COOL, P["ink"], 0.35), label="hot surface, cool street"),
          mpatches.Patch(facecolor=COOL_HOT, edgecolor=blend(COOL_HOT, P["ink"], 0.35), label="cool surface, hot street"),
          mpatches.Patch(facecolor="#9A9A9A", edgecolor="#6A6A6A", label="IMBAT"),
          mpatches.Patch(facecolor="white", edgecolor="#6A6A6A", hatch="//////", label="CALM")]
    axC1.legend(handles=hc, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=6.8, frameon=False,
                handlelength=1.4, columnspacing=0.9, borderaxespad=0.15)

    # (e) card
    xe = L + MAP_W + 0.45
    e_top = yb - 1.02                                   # below the forest's x labels
    e_h = e_top - (yc - 0.05)
    axE = S.inch_axes(fig, xe, yc - 0.05, FW - R - xe, e_h)
    verdict_txt = draw_card(axE, rs, d, verdict, h_in=e_h)
    S.panel_label(axE, "E", fontsize=15.0, y=1.02)

    # (d) tile strip (Fig 9 compact tiles)
    yd = BOT + D_LEG
    x_strip0 = L + (FW - L - R - (n_t * TILE + (n_t - 1) * TILE_GAP)) / 2
    tile_axes, tstats = [], []
    for i, t in enumerate(tiles):
        ax = S.inch_axes(fig, x_strip0 + i * (TILE + TILE_GAP), yd, TILE, TILE)
        tstats.append({**t, **draw_tile(ax, t)})
        tile_axes.append(ax)
        if i % 2 == 0:
            k = t["klass"]
            cx = x_strip0 + i * (TILE + TILE_GAP) + TILE + TILE_GAP / 2
            fig.text(cx / FW, (yd + TILE + 0.07) / FH, f"d.{i // 2 + 1}  LCZ {cls_label(k)}", ha="center",
                     va="bottom", fontsize=7.5, fontweight="bold", color=kdark(k))
    axL = None
    if tile_axes:
        S.panel_label(tile_axes[0], "D", fontsize=15.0, x=0.0, y=1.25, va="bottom")
        axL = S.inch_axes(fig, L, yd - 0.40, 3.10, 0.30)
        axL.set_axis_off()
        S.tile_tag(axL, "Hot: top tercile", HOT_TAG, x=0.0, y=0.80)
        S.tile_tag(axL, "Cool: bottom tercile", COOL_TAG, x=0.44, y=0.80)
        fig.text((L + 3.05) / FW, (yd - 0.40 + 0.30 * 0.80) / FH,
                 "of the 250 m cells · left of each pair: hot surface, cool street; right: cool surface, hot street",
                 ha="left", va="top", fontsize=6.8, color=P["muted"])
    label = rs.status["IMBAT"]
    stamp(fig, rs, label)
    meta = dict(verdict_rule=verdict, verdict_text=verdict_txt, status=rs.status, n_tiles=rs.n_tiles,
                cuts=d["cuts"], n_cells=n_c, share_cells_hot_cool=s31, share_cells_cool_hot=s13,
                forest_rows=[str(r) for r in rows], tiles=tstats, audit=aud_meta, fh_in=FH)
    panels = {"a": [axA, axK], "b": list(axB.values()), "c": [axC1, axC2], "d": tile_axes + ([axL] if axL else []),
              "e": [axE]}
    return fig, meta, panels, [axE], tile_axes


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="folder with the run_h3.py outputs (default outputs/street/analysis/h3)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="render non-final real results as a stamped DRAFT into outputs/street/SYNTHETIC/fig78/")
    ap.add_argument("--out", type=Path, default=None, help="override output PNG (tests)")
    ap.add_argument("--qa", type=Path, default=None, help="QA json + 100 %% crops folder")
    ap.add_argument("--all", action="store_true", help="also render Figure 8 from its default (final) results")
    a = ap.parse_args(argv)
    rs = H3Results(a.results, a.allow_partial)
    fig, meta, panels, cards, tile_axes = build(rs)
    png = rs.out_png("fig_7_h3_surface", a.out)
    qa = finish(fig, rs, png, panels, cards, tile_axes, dict(meta=meta), rs.qa_dir(a.qa))
    print(json.dumps({k: qa[k] for k in ("png", "fig_size_in", "smallest_author_pt", "smallest_printed_pt_190mm",
                                         "text_text_overlaps", "text_text_touching", "clipped_outside_sheet",
                                         "spill_out_of_container")}, indent=1, ensure_ascii=False))
    print("H3 compact paradox (rule, per day):", meta["verdict_rule"], "->", meta["verdict_text"])
    if a.all:
        from scripts.street.figures import build_fig8_h5_targeting as f8
        f8.main(["--allow-partial"] if a.allow_partial else [])
    return qa


if __name__ == "__main__":
    main()
