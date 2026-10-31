"""CLIMORFA v2 Figure 8: H5, street- vs zone-targeted tree planting (WP4.6).

Final run:   C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig8_h5_targeting
             (or ``build_fig7_h3_surface --all`` for both figures)  -> paper/figures/v2/main/fig_8_h5_targeting.png,
             ONLY when the H5 results are real and complete (run_status 'COMPLETE (427 tiles)', not partial, not
             synthetic).  Otherwise nothing is written to paper/.
Synthetic:   ... build_fig8_h5_targeting --results outputs/street/SYNTHETIC/fig78/h5/synthetic_H5T
             -> outputs/street/SYNTHETIC/fig78/fig_8_h5_targeting_SYNTHETIC_H5T.png (stamped; never paper/)

Inputs (exactly what the H5 code writes; SYNTHETIC_ prefix in synthetic runs):
  <results>/h5_evaluation_report.json        evaluate_h5.py: tree_budget_N, scenario_S, scenario_Z_20_draws,
                                             targeting_ratio_S_vs_Z (median, ci_95), jaccard_overlap_priority_sets,
                                             decision, run_status, is_partial, is_synthetic, pedestrian weight column
  <results>/h5_evaluation_summary.csv        evaluate_h5.py: one row per Z draw (delta_weighted_m2_z / _s, ratio)
  <scen>/scenarios_summary.json              select_scenarios.py: target LCZ, N, S length, Jaccard
  <scen>/scenario_S_segments.parquet         select_scenarios.py: S segments with geometry
  <scen>/scenario_Z_draws.parquet            select_scenarios.py: (draw_id, seg_id, n_trees) per Z draw
  <h5data>/S_exact/tree_dsm/<tile>.tif       plant_trees.py: tree DSM with the planted crowns (S, validation tiles)
  <h5data>/S_exact/utci/<DAY>/utci_street/<tile>  run_scenario_solweig.py: UTCI after planting (16 bands, 06-21)
  baseline UTCI                              data/street/utci/<DAY>/utci_street/<tile>.tif (WP2.3)
  <scen> = <results> if it holds the scenario tables (evaluate_h5.py copies them there), else data/street/h5/selection;
  <h5data> = <results>/h5data if present, else data/street/h5.  Segment geometry for Z: <scen>/segments.gpkg if present,
  else data/street/analysis/segments.gpkg.

Panels: (a) priority sets on the FUA map: S segments (purple) over the Z segments coloured by how many of the 20
draws picked them; (b) weighted heat-stress reduction Delta(M2 x w) of S (diamond) against the 20 Z draws (dots,
median tick, 2.5-97.5 % range) and the 1.5 x Z-median line of the lock; (c) before / after / difference close-up of a
300 m window of one S tile at 2 m (UTCI at 15:00, IMBAT, pedestrian-space cells), planted crowns outlined. S runs exactly
only on the 40 validation tiles (A10 ruling 5), so the tile is the validation tile with the largest pedestrian-weighted
reduction in very-strong-heat-stress hours (M2_38 per PS cell, bands 10-18, UTCI >= 38 degC, IMBAT) and the window the
300 m square (50 m steps) with the largest such sum; (d) computed card.
The card's verdict is re-derived from the per-draw table with the lock rule, its verdict status from the additive and
exact validation sums with the 15 % rule, and both are cross-checked against the report (a mismatch stops the build).
No software or plugin is named or depicted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.street.figures import build_fig5_h1_scale as F5  # noqa: E402
from scripts.street.figures import build_fig7_h3_surface as F7  # noqa: E402
from scripts.street.figures.build_fig5_h1_scale import CHECK, CROSS, MINUS, P, S, blend, card_frame, ctext, num  # noqa: E402
from scripts.street.figures.build_fig1_study_area import LCZ_LABEL  # noqa: E402
from scripts.street.figures.build_fig4_street_utci import add_scale_north_fig9_size, line_arrays, tag_box  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DEFAULT_RESULTS = ROOT / "outputs" / "street" / "analysis" / "h5"
REAL_SCEN = ROOT / "data" / "street" / "h5" / "selection"
REAL_H5DATA = ROOT / "data" / "street" / "h5"
OUT_PAPER = F7.OUT_PAPER
OUT_SYN = F7.OUT_SYN
N_TILES_TOTAL = 427
DAY = "IMBAT"                      # H5 selects and scores on IMBAT M2 (lock 2 H5); the close-up shows that day
BAND15 = 10                        # 1-based band of the 16-band UTCI files: bands 06..21 -> label 15 is band 10
M2_BANDS = list(range(5, 14))      # 1-based bands of labels 10..18 (M2 window, lock 1)
UTCI_VSTRONG = 38.0                # A10 ruling 1: M2_38, very strong heat stress
WIN_PX, WIN_STEP = 150, 25         # 300 m window, 50 m steps (2 m pixels)
S_COLOR = S.LST_TEAL_PURPLE_COLORS[-1]                        # v2 thermal purple: the hottest streets, targeted
Z_CMAP = mpl.colors.LinearSegmentedColormap.from_list(       # neutral taupe -> ink ramp for draw frequency
    "z_freq", [blend(P["taupe"], "white", 0.15), P["taupe_dark"], P["ink"]])
CMAP_ABS = S.LST_TEAL_PURPLE_CMAP                              # Fig 4 variant A ramp for absolute UTCI
CMAP_D = mpl.colors.LinearSegmentedColormap.from_list(       # = Fig 4 v2 anomaly ramp
    "anom_teal_white_purple", [S.LST_TEAL_PURPLE_COLORS[0], "#F7F7F7", S.LST_TEAL_PURPLE_COLORS[-1]])
TILES_2M = ROOT / "data/street/tiles/2m"


# =============================================================================================
class H5Results:
    def __init__(self, results: Path, allow_partial: bool):
        root = Path(results).resolve()
        if (root / "SYNTHETIC_h5_evaluation_report.json").exists():
            self.prefix = "SYNTHETIC_"
        elif (root / "h5_evaluation_report.json").exists():
            self.prefix = ""
        else:
            raise FileNotFoundError(f"{root}: no h5_evaluation_report.json (run scripts/street/h5/evaluate_h5.py)")
        self.root = root
        self.used: dict[str, str] = {}
        self.report = self._json(root / f"{self.prefix}h5_evaluation_report.json")
        path_syn = "SYNTHETIC" in str(root).upper()
        flag_syn = bool(self.report.get("is_synthetic", False)) or self.prefix == "SYNTHETIC_"
        if flag_syn and not path_syn:
            raise RuntimeError(f"rule 6: {root} holds synthetic H5 results but is not a SYNTHETIC folder")
        if path_syn and not flag_syn:
            raise RuntimeError(f"{root}: SYNTHETIC folder but the report is not synthetic")
        self.synthetic = flag_syn
        self.scen = root if (root / f"{self.prefix}scenario_S_segments.parquet").exists() else REAL_SCEN
        self.h5data = root / "h5data" if (root / "h5data").is_dir() else REAL_H5DATA
        self.summary = self._json(self.scen / f"{self.prefix}scenarios_summary.json")
        rs_ = self.report.get("run_status", "")
        self.final = (not self.synthetic and rs_ == f"COMPLETE ({N_TILES_TOTAL} tiles)"
                      and not self.report.get("is_partial", True) and not self.summary.get("partial_run", True)
                      and int(self.summary.get("n_tiles_at_extraction", 0)) == N_TILES_TOTAL)
        self.partial = not self.synthetic and not self.final
        if self.partial and not allow_partial:
            raise SystemExit(f"H5 results are not final (run_status {rs_!r}); nothing written. "
                             "--allow-partial renders a stamped DRAFT into outputs/street/SYNTHETIC/fig78/.")
        self.status = rs_
        sub = root.name.replace("synthetic_", "")
        self.tag = "_" + sub

    def _json(self, p: Path) -> dict:
        self.used[p.name] = str(p)
        return json.loads(Path(p).read_text(encoding="utf-8"))

    def table(self, name: str, where: Path | None = None) -> pd.DataFrame:
        p = (where or self.scen) / f"{self.prefix}{name}"
        self.used[p.name] = str(p)
        df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
        if "is_synthetic" in df.columns and bool(df["is_synthetic"].astype(bool).any()) != self.synthetic:
            raise RuntimeError(f"rule 6: is_synthetic of {p} disagrees with the report")
        return df

    def segments_path(self) -> Path:
        p = self.scen / f"{self.prefix}segments.gpkg"
        return p if p.exists() else ROOT / "data/street/analysis/segments.gpkg"

    def grid_path(self) -> Path:
        p = self.scen / f"{self.prefix}grid_250.gpkg"
        return p if p.exists() else ROOT / "data/street/analysis/grid_250.gpkg"

    def baseline_utci(self, tid: str) -> Path:
        b = self.h5data / "baseline_utci"
        base = b if b.is_dir() else ROOT / "data/street/utci"
        return base / DAY / "utci_street" / f"{tid}.tif"

    def scen_utci(self, tid: str) -> Path:
        return self.h5data / "S_exact" / "utci" / DAY / "utci_street" / f"{tid}.tif"

    def scen_tree(self, tid: str) -> Path:
        return self.h5data / "S_exact" / "tree_dsm" / f"{tid}.tif"

    def ps_index(self, tid: str) -> Path:
        b = self.h5data / "ps_index"
        return (b if b.is_dir() else ROOT / "data/street/analysis/ps_index") / f"{tid}.tif"

    def validation_tiles(self) -> list[str]:
        """The tiles with an exact S run (A10 ruling 5); selection summary and report must agree."""
        a = self.summary.get("s_exact_validation", {}).get("tiles")
        b = self.report.get("validation", {}).get("tiles")
        if not a or sorted(a) != sorted(b or []):
            raise RuntimeError("validation tiles differ between scenarios_summary.json and the H5 report")
        return sorted(a)

    out_png = F7.H3Results.out_png
    qa_dir = F7.H3Results.qa_dir


# =============================================================================================
def load(rs: H5Results) -> dict:
    rep = rs.report
    per = rs.table("h5_evaluation_summary.csv", where=rs.root)
    s_tab = gpd.read_parquet(rs.scen / f"{rs.prefix}scenario_S_segments.parquet")
    rs.used["scenario_S_segments"] = str(rs.scen / f"{rs.prefix}scenario_S_segments.parquet")
    z_tab = rs.table("scenario_Z_draws.parquet")
    segp = rs.segments_path()
    rs.used["segments (Z geometry)"] = str(segp)
    segs = gpd.read_file(segp, layer="segments", columns=["seg_id"])
    freq = z_tab.groupby("seg_id")["draw_id"].nunique()
    zg = segs[segs["seg_id"].isin(freq.index)].copy()
    zg["freq"] = zg["seg_id"].map(freq).astype(int)
    if len(zg) != len(freq):
        raise RuntimeError(f"{len(freq) - len(zg)} Z segments have no geometry in {segp}")
    return dict(rep=rep, per=per, s=s_tab, z=z_tab, zg=zg, n_draws=int(z_tab["draw_id"].nunique()))


def crosscheck(rs: H5Results, d: dict) -> tuple[bool, dict]:
    rep, per = d["rep"], d["per"]
    r = per["targeting_ratio_S_vs_Z"].to_numpy(float)
    med = float(np.median(r))
    lo, hi = float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
    t = rep["targeting_ratio_S_vs_Z"]
    if abs(round(med, 3) - t["median"]) > 0.0015 or abs(round(lo, 3) - t["ci_95"][0]) > 0.0015 or \
            abs(round(hi, 3) - t["ci_95"][1]) > 0.0015:
        raise RuntimeError(f"H5 ratio mismatch: per-draw table {med:.3f} [{lo:.3f}, {hi:.3f}] vs report {t}")
    rule = bool(med >= t.get("threshold", 1.5))
    if rule != bool(rep["decision"]["threshold_condition_met"]):
        raise RuntimeError(f"H5 verdict mismatch: rule {rule} vs report {rep['decision']}")
    if rs.final and rep["decision"]["status"] != ("H5 SUPPORTED" if rule else "H5 NOT SUPPORTED"):
        raise RuntimeError(f"H5 verdict mismatch: rule {rule} vs report status {rep['decision']['status']}")
    s_val = per["delta_weighted_m2_s"].to_numpy(float)
    if np.ptp(s_val) > 1e-6 * max(1.0, abs(s_val[0])) or abs(s_val[0] - rep["scenario_S"]["delta_weighted_m2"]) > 0.02:
        raise RuntimeError("S outcome differs between the per-draw table and the report")
    val = rep["validation"]
    rel = abs(val["additive"] - val["exact"]) / abs(val["exact"]) if val["exact"] else float("inf")
    status = "descriptive only" if rel > val["rel_tol"] else "test"
    if status != rep["verdict_status"] or status != rep["decision"]["verdict_status"]:
        raise RuntimeError(f"H5 verdict status mismatch: rule {status} (rel diff {rel:.4f}) vs report "
                           f"{rep['verdict_status']}")
    a11 = rep["a11"]["days"][DAY]
    for name in ("cell_h38", "mean_utci_b15"):
        r_ = float(np.median(per[f"ratio_{name}"].to_numpy(float)))
        t_ = a11["companions"][name]["ratio_S_vs_Z"]["median"]
        if (t_ is None) != (not np.isfinite(r_)) or (t_ is not None and abs(round(r_, 3) - t_) > 0.0015):
            raise RuntimeError(f"A11 companion {name}: per-draw median {r_:.3f} vs report {t_}")
    if d["n_draws"] != rep["scenario_Z_20_draws"]["n_draws"] or len(per) != d["n_draws"]:
        raise RuntimeError("number of Z draws differs between tables")
    if int(d["s"]["n_trees"].sum()) != int(rep["tree_budget_N"]):
        raise RuntimeError("tree budget N differs between the S table and the report")
    jz = rep["jaccard_overlap_priority_sets"]["jaccard_segment_count"]
    js = d["s"]["seg_id"]
    jac = []
    for _, g in d["z"].groupby("draw_id"):
        a, b = set(js), set(g["seg_id"])
        jac.append(len(a & b) / len(a | b))
    if abs(round(float(np.median(jac)), 4) - jz["median"]) > 0.00015:
        raise RuntimeError(f"Jaccard mismatch: recomputed {np.median(jac):.4f} vs report {jz['median']}")
    return rule, dict(median=med, ci=[lo, hi], jaccard_recomputed=float(np.median(jac)), rel_diff=rel, status=status)


# ---------------------------------------------------------------------------------------------
def tile_arrays(rs: H5Results, tid: str):
    """Everything panel (c) needs for one S tile, inner 1 km at 2 m, or None if a raster is missing."""
    paths = dict(base=rs.baseline_utci(tid), scen=rs.scen_utci(tid), tree1=rs.scen_tree(tid),
                 tree0=TILES_2M / "tree_dsm" / f"{tid}.tif", ps=rs.ps_index(tid),
                 dsm=TILES_2M / "building_dsm" / f"{tid}.tif", dem=TILES_2M / "dem" / f"{tid}.tif")
    if not all(p.exists() for p in paths.values()):
        return None
    sl = slice(50, 550)
    with rasterio.open(paths["base"]) as a, rasterio.open(paths["scen"]) as b:
        assert a.count == 16 and b.count == 16
        bands = sorted(set(M2_BANDS + [BAND15]))
        u0 = {k: a.read(k).astype(float) for k in bands}
        u1 = {k: b.read(k).astype(float) for k in bands}
        nod0, nod1 = a.nodata, b.nodata
        tr = a.transform
    ps = rasterio.open(paths["ps"]).read(1) > 0
    h = (rasterio.open(paths["dsm"]).read(1) - rasterio.open(paths["dem"]).read(1))[sl, sl]
    bld = np.isfinite(h) & (h > 0.5)
    t0 = rasterio.open(paths["tree0"]).read(1)[sl, sl]
    t1 = rasterio.open(paths["tree1"]).read(1)[sl, sl]
    valid = ps & np.isfinite(u0[BAND15]) & (u0[BAND15] != nod0) & np.isfinite(u1[BAND15]) & (u1[BAND15] != nod1)
    m2_0 = sum((u0[k] >= UTCI_VSTRONG).astype(int) for k in M2_BANDS)
    m2_1 = sum((u1[k] >= UTCI_VSTRONG).astype(int) for k in M2_BANDS)
    return dict(tid=tid, u0=u0[BAND15], u1=u1[BAND15], valid=valid, bld=bld, crowns=t1 > t0 + 0.01,
                dm2=np.where(valid, m2_1 - m2_0, 0), x0=tr.c, y1=tr.f)


def pixel_weights(arr: dict, weights: dict, med_w: float) -> np.ndarray:
    xs = arr["x0"] + 1.0 + 2.0 * np.arange(500)
    ys = arr["y1"] - 1.0 - 2.0 * np.arange(500)
    gx, gy = (xs // 250).astype(int), (ys // 250).astype(int)
    w = np.empty((500, 500))
    for j, yy in enumerate(np.unique(gy)):
        for xx in np.unique(gx):
            w[np.ix_(gy == yy, gx == xx)] = weights.get(f"{xx}_{yy}", med_w)
    return w


def pick_tile(rs: H5Results, d: dict) -> tuple[dict | None, dict]:
    from scripts.street.h5.evaluate_h5 import load_grid_weights   # the analysis's own weight builder (A4)
    gp = rs.grid_path()
    rs.used["grid_250 (weights)"] = str(gp)
    weights, pop_col = load_grid_weights(gp, pop_col=d["rep"].get("pedestrian_weight_pop_column"))
    med_w = float(np.median(list(weights.values())))
    tiles = rs.validation_tiles()                         # S runs exactly only on these tiles (A10 ruling 5)
    stray = sorted(set(tiles) - set(d["s"]["tile_id"]))
    if stray:
        raise RuntimeError(f"validation tiles outside the S tile set: {stray[:5]}")
    best, scores, missing = None, {}, []
    for tid in tiles:
        a = tile_arrays(rs, tid)
        if a is None:
            missing.append(tid)
            continue
        w = pixel_weights(a, weights, med_w)
        red = -a["dm2"] * w                                   # weighted reduction of very-strong-heat-stress hours
        scores[tid] = float(red.sum())
        if best is None or scores[tid] > scores[best["tid"]]:
            a["red"] = red
            best = a
    if rs.final and missing:
        raise RuntimeError(f"final run but {len(missing)} S tiles lack scenario rasters: {missing[:5]}")
    info = dict(n_s_tiles=int(d["s"]["tile_id"].nunique()), n_validation_tiles=len(tiles), n_scored=len(scores),
                missing=missing, pop_col=pop_col,
                tile=best["tid"] if best else None, score=scores.get(best["tid"]) if best else None)
    if best is not None:
        csum = np.cumsum(np.cumsum(np.pad(best["red"], ((1, 0), (1, 0))), 0), 1)
        bw, br, bc = -np.inf, 0, 0
        for r in range(0, 500 - WIN_PX + 1, WIN_STEP):
            for c in range(0, 500 - WIN_PX + 1, WIN_STEP):
                v = csum[r + WIN_PX, c + WIN_PX] - csum[r, c + WIN_PX] - csum[r + WIN_PX, c] + csum[r, c]
                if v > bw:
                    bw, br, bc = v, r, c
        best["win"] = (br, bc)
        info["window_rc"] = [int(br), int(bc)]
        info["window_score"] = float(bw)
    return best, info


# =============================================================================================
def draw_map(ax, d, study, land, extent, tile_xy):
    S.draw_land_cells(ax, land)
    zg = d["zg"].sort_values("freq")
    xy = line_arrays(zg.geometry.values)
    norm = mpl.colors.Normalize(1, d["n_draws"])
    ax.add_collection(LineCollection(xy, array=zg["freq"].to_numpy(float), cmap=Z_CMAP, norm=norm, linewidths=0.9,
                                     capstyle="round", zorder=3))
    sxy = line_arrays(d["s"].geometry.values)
    ax.add_collection(LineCollection(sxy, colors="white", linewidths=2.3, capstyle="round", zorder=4))
    ax.add_collection(LineCollection(sxy, colors=S_COLOR, linewidths=1.3, capstyle="round", zorder=5))
    S.draw_study_outline(ax, study, "ink")
    S.set_map_extent(ax, extent, exact=True)
    if tile_xy is not None:
        x0, y0 = tile_xy
        sq = mpatches.Rectangle((x0, y0), 1000, 1000, fill=False, edgecolor="black", linewidth=1.0, zorder=20)
        sq.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
        ax.add_patch(sq)
        ax.text(x0 + 1000 + 450, y0 + 1000 + 250, "c", ha="left", va="bottom", fontsize=7.2, fontweight="bold",
                color=S.PANEL_TEXT, zorder=21, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    return norm


def sci_scale(v: float) -> tuple[float, int]:
    if v <= 0 or not np.isfinite(v):
        return 1.0, 0
    k = int(np.floor(np.log10(v) / 3) * 3)
    return 10.0 ** k, k


def draw_outcome(ax, d, chk):
    per = d["per"]
    z = per["delta_weighted_m2_z"].to_numpy(float)
    s_ = float(per["delta_weighted_m2_s"].iloc[0])
    f, k = sci_scale(max(s_, z.max()))
    z, s_ = z / f, s_ / f
    zm = float(np.median(z))
    lo, hi = np.percentile(z, [2.5, 97.5])
    rng = np.random.default_rng(20261030)
    yz, ys = 0.0, 1.0
    ax.scatter(z, yz + rng.uniform(-0.16, 0.16, len(z)), s=26, facecolor=P["taupe"], edgecolor=P["taupe_dark"],
               linewidth=0.7, zorder=3)
    ax.plot([lo, hi], [yz - 0.30, yz - 0.30], color=P["taupe_dark"], lw=1.4, zorder=3, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [yz - 0.36, yz - 0.24], color=P["taupe_dark"], lw=1.2, zorder=3)
    ax.plot([zm, zm], [yz - 0.26, yz + 0.26], color=P["ink"], lw=2.0, zorder=4, solid_capstyle="butt")
    ax.scatter([s_], [ys], s=70, marker="D", facecolor=S_COLOR, edgecolor=blend(S_COLOR, P["ink"], 0.4),
               linewidth=1.0, zorder=5)
    thr = 1.5 * zm
    ax.axvline(thr, color=P["muted"], ls=(0, (4, 3)), lw=1.1, zorder=2)
    ax.text(thr - 0.01 * thr, 1.62, "1.5 \u00d7 Z median (lock)", ha="right", va="bottom", fontsize=6.6, color=P["muted"],
            style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85), zorder=6)
    ax.text(zm, yz + 0.30, "median", ha="center", va="bottom", fontsize=6.6, color=P["ink"], zorder=6)
    ax.text((lo + hi) / 2, yz - 0.42, "95 % of draws", ha="center", va="top", fontsize=6.6, color=P["taupe_dark"],
            zorder=6)
    xmax = max(s_, thr, hi) * 1.12
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.95, 1.95)
    ax.set_yticks([yz, ys])
    ax.set_yticklabels([f"Z, {len(z)} draws", "S"], fontsize=8.2)
    ax.tick_params(axis="y", length=0, colors=P["ink"])
    ax.tick_params(axis="x", length=0, labelsize=8.2, colors=P["muted"])
    ax.grid(True, axis="x", color=P["grid"], lw=0.7)
    ax.grid(False, axis="y")
    for s__ in ax.spines.values():
        s__.set_color(P["grid"]); s__.set_linewidth(1.0)
    unit = "" if k == 0 else f" (\u00d710$^{{{k}}}$)"
    ax.set_xlabel(f"Weighted reduction \u0394(M2_38 \u00d7 w){unit}", fontsize=8.6, color=P["ink"], labelpad=3)
    return dict(scale_exp=k, s=s_ * f, z_median=zm * f, z_range=[lo * f, hi * f], line=thr * f)


def draw_close(axs, a, cax_abs, cax_d, fig):
    r, c = a["win"]
    sl = (slice(r, r + WIN_PX), slice(c, c + WIN_PX))
    u0 = np.where(a["valid"], a["u0"], np.nan)[sl]
    u1 = np.where(a["valid"], a["u1"], np.nan)[sl]
    du = u1 - u0
    bld = a["bld"][sl]
    crowns = a["crowns"][sl]
    v = np.concatenate([u0[np.isfinite(u0)], u1[np.isfinite(u1)]])
    lo, hi = np.floor(np.percentile(v, 2)), np.ceil(np.percentile(v, 98))
    if hi - lo < 2:
        hi = lo + 2
    dd = du[np.isfinite(du)]
    dl = max(1.0, float(np.ceil(np.percentile(np.abs(dd), 99)))) if dd.size else 1.0
    x0 = a["x0"] + 2.0 * c
    y1 = a["y1"] - 2.0 * r
    ext = [x0, x0 + 2.0 * WIN_PX, y1 - 2.0 * WIN_PX, y1]
    norm_a = mpl.colors.Normalize(lo, hi)
    norm_d = mpl.colors.Normalize(-dl, dl)
    xs = np.linspace(ext[0] + 1, ext[1] - 1, WIN_PX)
    ys = np.linspace(ext[3] - 1, ext[2] + 1, WIN_PX)
    stats = {}
    for ax, img, cmap, norm, tag in ((axs[0], u0, CMAP_ABS, norm_a, "before"), (axs[1], u1, CMAP_ABS, norm_a, "after"),
                                     (axs[2], du, CMAP_D, norm_d, "after \u2212 before")):
        ax.set_facecolor("white")
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, norm=norm, extent=ext, interpolation="nearest", zorder=2)
        ax.imshow(np.ma.masked_where(~bld, np.ones_like(bld, float)), cmap=mpl.colors.ListedColormap([P["taupe"]]),
                  vmin=0, vmax=1, extent=ext, interpolation="nearest", zorder=3)
        ax.contour(xs, ys, bld.astype(float), levels=[0.5], colors=S.PANEL_EDGE, linewidths=0.15, zorder=4)
        if tag != "before" and crowns.any():
            ax.contour(xs, ys, crowns.astype(float), levels=[0.5], colors=P["ink"], linewidths=0.55, zorder=5)
        elif crowns.any():
            ax.contour(xs, ys, crowns.astype(float), levels=[0.5], colors=P["ink"], linewidths=0.55,
                       linestyles=[(0, (1.5, 1.2))], zorder=5)
        ax.add_patch(mpatches.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False,
                                        edgecolor=P["ink"], linewidth=0.9, zorder=6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.axis("off")
        S.tile_tag(ax, f"{DAY} \u00b7 {tag}", P["taupe_dark"], x=0.04, y=0.96)
    f0, f1 = u0[np.isfinite(u0)], u1[np.isfinite(u1)]
    cooled = dd[dd < -0.05]
    dm2 = a["dm2"][sl][np.isfinite(u0)]
    stats.update(window_rc=[int(r), int(c)], utci_before_median=float(np.median(f0)), utci_after_median=float(np.median(f1)),
                 n_ps=int(f0.size), share_cooled=float(cooled.size / max(dd.size, 1)),
                 dutci_cooled_median=float(np.median(cooled)) if cooled.size else 0.0,
                 dm2_mean_ps=float(dm2.mean()) if dm2.size else 0.0, abs_range=[float(lo), float(hi)], d_lim=dl,
                 n_crown_px=int(crowns.sum()))
    S.tile_readout(axs[0], f"UTCI {stats['utci_before_median']:.1f}\u00b0C", x=0.04, y=0.04)
    S.tile_readout(axs[1], f"UTCI {stats['utci_after_median']:.1f}\u00b0C", x=0.04, y=0.04)
    S.tile_readout(axs[2], f"\u0394UTCI {F5.sgn(stats['dutci_cooled_median'])}\u00b0C on cooled cells\n"
                           f"\u0394M2_38 {F5.sgn(stats['dm2_mean_ps'], 2)} h per PS cell", x=0.04, y=0.04)
    add_win_scale(axs[0], ext)
    for cax, cmap, norm, lab, ticks in ((cax_abs, CMAP_ABS, norm_a, "UTCI at 15:00, PS cells (\u00b0C)", None),
                                        (cax_d, CMAP_D, norm_d, "\u0394UTCI after \u2212 before (\u00b0C)", None)):
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=8.0)
        cb.ax.grid(False)
        cb.set_label(lab, fontsize=8.5, fontweight="bold", color=P["ink"], labelpad=3.0)
        cb.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
        signed = cmap is CMAP_D
        cb.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v_, _, signed=signed: "0" if abs(v_) < 1e-9 else
            (f"{v_:+g}" if signed else f"{v_:g}").replace("-", MINUS)))
    return stats


def add_win_scale(ax, ext, length=100.0):
    w = ext[1] - ext[0]
    sb_h = w * 0.012
    sx = ext[1] - 0.06 * w - length
    sy = ext[2] + 0.05 * w
    ax.add_patch(mpatches.Rectangle((sx - 0.03 * w, sy - 0.025 * w), length + 0.06 * w, 0.17 * w, facecolor="white",
                                    alpha=0.85, edgecolor="none", zorder=24))
    for i in range(2):
        ax.add_patch(mpatches.Rectangle((sx + i * length / 2, sy), length / 2, sb_h,
                                        facecolor="black" if i == 0 else "white", edgecolor="black", linewidth=0.5,
                                        zorder=25))
    ax.text(sx + length / 2, sy + sb_h * 1.8, f"{length:g} m", ha="center", va="bottom", fontsize=6.2,
            fontweight="bold", color=S.PANEL_TEXT, zorder=25)


def draw_card(ax, rs, d, verdict, chk, h_in):
    card_frame(ax)
    rep, summ = d["rep"], rs.summary
    u = 1.0 / h_in                                         # spacing authored in inches
    xl, xv = 0.035, 0.965
    y = 1 - 0.24 * u
    step, sub = 0.37 * u, 0.165 * u
    t = rep["targeting_ratio_S_vs_Z"]
    jz = rep["jaccard_overlap_priority_sets"]["jaccard_segment_count"]
    jl = rep["jaccard_overlap_priority_sets"]["jaccard_length_weighted"]
    zc = summ.get("scenario_Z", {}).get("target_lcz")
    sS = summ.get("scenario_S", {})
    a11 = rep["a11"]["days"][DAY]

    def comp(name):
        v = a11["companions"][name]["ratio_S_vs_Z"]["median"]
        return "n/a" if v is None else num(v, 2)

    rows = [
        ("tree budget N", f"{int(rep['tree_budget_N']):,} trees",
         f"8 m spacing \u00b7 S = {sS.get('n_segments', len(d['s'])):,} segments, {sS.get('planted_length_m', 0) / 1000:,.1f} km"),
        ("Z target class", f"LCZ {LCZ_LABEL.get(int(zc), zc) if zc is not None else '?'}",
         f"{d['n_draws']} length-weighted draws (A5)"),
        ("S / Z, median of draws", f"{num(t['median'], 2)}  " + (CHECK if verdict else CROSS),
         f"95 % of draws {F5.ci_txt(t['ci_95'])} \u00b7 lock: \u2265 {t.get('threshold', 1.5):g}"),
        ("Jaccard S \u2229 Z", f"{num(jz['median'], 2)}",
         f"95 % {F5.ci_txt(jz['ci_95'])} \u00b7 by length {num(jl['median'], 2)}"),
        ("additive vs exact S", f"{num(chk['rel_diff'] * 100, 1)} %",
         f"{len(rep['validation']['tiles'])} tiles \u00b7 limit {rep['validation']['rel_tol'] * 100:g} % (A10) \u00b7 "
         f"{chk['status']}"),
        ("\u0394M2_38 = 0 under P (A11)", f"{num(a11['share_dm2_38_zero_in_P']['all_segments'] * 100, 0)} %",
         f"companion S / Z: cell-hours {comp('cell_h38')} \u00b7 mean UTCI {comp('mean_utci_b15')}"),
    ]
    for lab, val, note in rows:
        ctext(ax, xl, y, lab)
        ctext(ax, xv, y, val, ha="right")
        ctext(ax, xl, y - sub, note, size=7.0, weight="normal", color=P["muted"])
        y -= step
    w = rep.get("pedestrian_weight_pop_column", "?")
    ctext(ax, xl, y + 0.06 * u, f"w = 400 m reach \u00d7 {'residents (A4)' if w == 'pop_res' else w}, per 250 m cell",
          size=7.0, weight="normal", color=P["muted"])
    y -= 0.14 * u
    ax.plot([xl, 1 - xl], [y, y], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    y -= 0.24 * u
    ctext(ax, xl, y, "H5, S \u2265 1.5 \u00d7 Z:")
    word = "supported" if verdict else "not supported"
    ctext(ax, xv, y, word if chk["status"] == "test" else f"{word}, descriptive only", ha="right")


# =============================================================================================
def build(rs: H5Results):
    S.apply_style()
    np.random.seed(20261030)
    d = load(rs)
    verdict, chk = crosscheck(rs, d)
    study, land = F7.base_layers()
    best, tinfo = pick_tile(rs, d)

    FW = S.V2_FIG_WIDTH_IN
    L, R, T = 0.50, 0.50, 0.15
    MAP_W = 7.05
    extent = tuple(study.total_bounds)
    ar = (extent[3] - extent[1]) / (extent[2] - extent[0])
    MAP_H = MAP_W * ar
    TAG_BAND = 1.10
    GAP_MC = 0.72
    C_HDR = 0.36
    TILE = 2.85
    TGAP = 0.12
    BOT = 0.30
    FH = T + MAP_H + TAG_BAND + GAP_MC + C_HDR + TILE + BOT
    fig = plt.figure(figsize=(FW, FH))

    # (a) map
    y_map = FH - T - MAP_H
    axA = S.inch_axes(fig, L, y_map, MAP_W, MAP_H)
    tile_xy = None
    if best is not None:
        tile_xy = (best["x0"], best["y1"] - 1000.0)
    norm = draw_map(axA, d, study, land, extent, tile_xy)
    add_scale_north_fig9_size(axA, base_loc=(0.02, 0.42), scale_len_km=10.0)
    S.panel_label(axA, "A", fontsize=15.0, y=0.98, va="top")
    # legend under the map (S line, Z draw-frequency bar), computed tag to its right
    axLg = S.inch_axes(fig, L, y_map - 0.40, 2.10, 0.28)
    axLg.set_axis_off(); axLg.set_xlim(0, 1); axLg.set_ylim(0, 1)
    axLg.plot([0.01, 0.13], [0.5, 0.5], color="white", lw=3.2, solid_capstyle="round")
    axLg.plot([0.01, 0.13], [0.5, 0.5], color=S_COLOR, lw=1.8, solid_capstyle="round")
    axLg.text(0.17, 0.5, "S, street-targeted", ha="left", va="center", fontsize=7.4, fontweight="bold",
              color=P["ink"])
    cZ = S.inch_axes(fig, L + 0.04, y_map - 1.00, 1.90, 0.11)
    sm = mpl.cm.ScalarMappable(cmap=Z_CMAP, norm=norm)
    cb = fig.colorbar(sm, cax=cZ, orientation="horizontal")
    cb.set_ticks(sorted({1, int(round(d["n_draws"] / 2)), d["n_draws"]}))
    cb.ax.tick_params(labelsize=8.0)
    cb.ax.grid(False)
    cZ.text(0.0, 1.5, "Z, draws per segment", transform=cZ.transAxes, ha="left", va="bottom",
            fontsize=7.4, fontweight="bold", color=P["ink"])
    jz = d["rep"]["jaccard_overlap_priority_sets"]["jaccard_segment_count"]
    sS = rs.summary.get("scenario_S", {})
    zc = rs.summary.get("scenario_Z", {}).get("target_lcz")
    zlab = LCZ_LABEL.get(int(zc), zc) if zc is not None else "?"
    tag_box(axA, f"{DAY} \u00b7 S: hottest 10 % of segment length by M2_38,\n"
                 f"{sS.get('n_segments', len(d['s'])):,} segments \u00b7 Z: same budget N in LCZ {zlab},\n"
                 f"{d['zg'].shape[0]:,} segments in \u2265 1 of {d['n_draws']} draws\n"
                 f"Jaccard S \u2229 Z: median {num(jz['median'], 2)} {F5.ci_txt(jz['ci_95'])}",
            x=2.40 / MAP_W, y=-0.035, va="top")

    # (b) outcome strip
    xb = L + MAP_W + 1.62
    bw = FW - R - xb
    hb = 1.85
    yb = FH - T - 0.42 - hb
    axB = S.inch_axes(fig, xb, yb, bw, hb)
    ostats = draw_outcome(axB, d, chk)
    S.panel_label(axB, "B", fontsize=15.0, x=-1.62 / bw, y=1.02)

    # (d) card
    xd = L + MAP_W + 0.45
    d_top = yb - 0.85
    d_bot = y_map - TAG_BAND - 0.28                       # 6 card rows: reaches into the gap above panel (c)
    axD = S.inch_axes(fig, xd, d_bot, FW - R - xd, d_top - d_bot)
    draw_card(axD, rs, d, verdict, chk, d_top - d_bot)
    S.panel_label(axD, "D", fontsize=15.0, y=1.02)

    # (c) close-up: three tiles, colour bars stacked to their right
    y_tile = BOT
    axC = [S.inch_axes(fig, L + i * (TILE + TGAP), y_tile, TILE, TILE) for i in range(3)]
    xcb = L + 3 * TILE + 2 * TGAP + 0.42
    cbw = FW - R - xcb - 0.10
    cax_abs = S.inch_axes(fig, xcb, y_tile + 0.66 * TILE, cbw, 0.12)
    cax_d = S.inch_axes(fig, xcb, y_tile + 0.22 * TILE, cbw, 0.12)
    cstats = {}
    if best is not None:
        cstats = draw_close(axC, best, cax_abs, cax_d, fig)
        fig.text((L + 1.5 * TILE + TGAP) / FW, (y_tile + TILE + 0.07) / FH,
                 f"{best['tid']} \u00b7 300 m window \u00b7 15:00 \u00b7 planted crowns outlined "
                 f"(dotted: before planting)", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                 color=P["ink"])
    else:
        for ax in axC + [cax_abs, cax_d]:
            ax.set_axis_off()
        axC[1].text(0.5, 0.5, "no S tile with scenario rasters yet", ha="center", va="center", fontsize=7.6,
                    color=P["muted"], transform=axC[1].transAxes)
    S.panel_label(axC[0], "C", fontsize=15.0, x=0.0, y=1.11, va="bottom")
    F7.stamp(fig, rs, rs.status)
    meta = dict(verdict_rule=verdict, recomputed=chk, status=rs.status, tile=tinfo, close=cstats, outcome=ostats,
                n_z_segments=int(d["zg"].shape[0]), n_s_segments=int(len(d["s"])), fh_in=FH)
    panels = {"a": [axA, axLg, cZ], "b": [axB], "c": axC + [cax_abs, cax_d], "d": [axD]}
    return fig, meta, panels, [axD], axC


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="folder with the evaluate_h5.py outputs (default outputs/street/analysis/h5)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="render non-final real results as a stamped DRAFT into outputs/street/SYNTHETIC/fig78/")
    ap.add_argument("--out", type=Path, default=None, help="override output PNG (tests)")
    ap.add_argument("--qa", type=Path, default=None, help="QA json + 100 %% crops folder")
    a = ap.parse_args(argv)
    rs = H5Results(a.results, a.allow_partial)
    fig, meta, panels, cards, tiles = build(rs)
    png = rs.out_png("fig_8_h5_targeting", a.out)
    qa = F7.finish(fig, rs, png, panels, cards, tiles, dict(meta=meta), rs.qa_dir(a.qa))
    print(json.dumps({k: qa[k] for k in ("png", "fig_size_in", "smallest_author_pt", "smallest_printed_pt_190mm",
                                         "text_text_overlaps", "text_text_touching", "clipped_outside_sheet",
                                         "spill_out_of_container")}, indent=1, ensure_ascii=False))
    print("H5 (rule / report):", meta["verdict_rule"], meta["recomputed"], "tile:", meta["tile"].get("tile"))
    return qa


if __name__ == "__main__":
    main()
