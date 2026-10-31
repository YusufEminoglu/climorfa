"""CLIMORFA v2 Figure 5: H1, how much street-level heat stress the LCZ label resolves (WP4.5).

Final run (one command re-renders Fig 5 AND Fig 6 from the final result files of scripts/street/stats/run_all.py):

    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.build_fig5_h1_scale --all
        (= --results outputs/street/analysis, sub-folder 'results' of h1/, h2/, h4/)
        -> paper/figures/v2/main/fig_5_h1_scale.png  and  fig_6_drivers.png

Synthetic test (planted effects; the figure is stamped and can never reach paper/):

    ... build_fig5_h1_scale --all --results outputs/street/SYNTHETIC/fig56/analysis --subdir synthetic_H1T
        -> outputs/street/SYNTHETIC/fig56/fig_5_h1_scale_SYNTHETIC_H1T.png (+ qa/, crops/)

Inputs (exactly the files the real run writes; names from common.fname):
  h1/<sub>/h1_results[.json|_PARTIAL.json|_SYNTHETIC.json]      LMM shares, CV R2, eta2 ladder, decision
  h4/<sub>/seg_h4_terms_<DAY>[...].parquet                       per-segment frame of the run (m1, lcz, lcz_valid)
The segment set of (a) is checked against the H1 LMM segment count (A6: LCZ no-data excluded).

Panels: (a) raincloud of the within-day anomaly of M1 (UTCI_street at label 15) by LCZ class, IMBAT | CALM facets,
classes with >= 30 segments only (lock 1; pooled classes and A6 no-data are counted in a note); (b) scale ladder of
eta2 (segment, 50, 100, 250, 500 m) plus the two H1 decision quantities with their lock limits; (c) computed card.
Every number in the figure is read from the result files; nothing is typed in.  Rule 6: a results path containing
'SYNTHETIC' (or a SYNTHETIC label in the JSON) forces the SYNTHETIC watermark and the SYNTHETIC output folder.
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
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common as cm  # noqa: E402  (lock constants: MIN_CLASS_N, pooled codes, block bootstrap, day dates)
from scripts.street.figures import climorfa_style as S  # noqa: E402
from scripts.street.figures.build_fig1_study_area import LCZ_COLORS, LCZ_LABEL, NATURAL, NATURAL_COLOR, lcz_dark  # noqa: E402

P = S.PALETTE
DEFAULT_RESULTS = ROOT / "outputs" / "street" / "analysis"
OUT_PAPER = ROOT / "paper" / "figures" / "v2" / "main"
OUT_SYN = ROOT / "outputs" / "street" / "SYNTHETIC" / "fig56"
DAYS = [("IMBAT", cm.DAY_DATES["IMBAT"]), ("CALM", cm.DAY_DATES["CALM"])]

# Day encoding (Figs 5 and 6): one neutral hue family, IMBAT solid / filled, CALM dashed / open.  Neutral greys from the
# original Fig 2 fold palette (S.FOLD_NEUTRALS) so no LCZ class colour is reused for a day.
DAY_STYLE = {
    "IMBAT": dict(color=S.FOLD_NEUTRALS[0], ls="-", mfc=S.FOLD_NEUTRALS[0], marker="o"),
    "CALM": dict(color=S.FOLD_NEUTRALS[1], ls=(0, (4, 2)), mfc="white", marker="o"),
}
MINUS = "\u2212"
CHECK, CROSS = "\u2713", "\u2717"


# =============================================================================================
# shared helpers (also used by build_fig6_drivers.py)
# =============================================================================================
def sgn(x: float, nd: int = 1) -> str:
    if round(float(x), nd) == 0:                       # no '-0.00'
        return f"{0:.{nd}f}"
    return f"{x:+.{nd}f}".replace("-", MINUS)


def num(x: float, nd: int = 2) -> str:
    if round(float(x), nd) == 0:                       # no '-0.00'
        return f"{0:.{nd}f}"
    return f"{x:.{nd}f}".replace("-", MINUS)


def ci_txt(ci, nd: int = 2) -> str:
    return f"[{num(ci[0], nd)}, {num(ci[1], nd)}]"


def kfmt(n: int) -> str:
    return f"{n:,}" if n < 10000 else f"{n / 1000:.0f}k"


def blend(c1: str, c2: str, f: float) -> str:
    a, b = np.array(mpl.colors.to_rgb(c1)), np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex(a * (1 - f) + b * f)


class ResultSet:
    """Locates the result files of one run.  ``results`` is the analysis root holding h1/, h2/, h4/ (each with a
    sub-folder such as 'results' or 'synthetic_H1T'), or one flat folder holding all files."""

    def __init__(self, results: Path, subdir: str | None, allow_partial: bool):
        self.root = Path(results).resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.synthetic = "SYNTHETIC" in str(self.root).upper() or "SYNTHETIC" in str(subdir or "").upper()
        self.subdir = subdir
        self.allow_partial = allow_partial
        self.partial = False
        self.used: dict[str, str] = {}

    def folder(self, h: str) -> Path:
        hd = self.root / h
        if not hd.is_dir():
            return self.root                                   # flat layout
        if self.subdir:
            return hd / self.subdir
        if (hd / "results").is_dir() and not self.synthetic:
            return hd / "results"                              # lock primary run
        cands = sorted(d for d in hd.iterdir() if d.is_dir() and any(d.glob(f"{h}_results*")))
        if len(cands) != 1:
            raise FileNotFoundError(f"{hd}: give --subdir, candidates {[c.name for c in cands]}")
        return cands[0]

    def file(self, h: str, stem: str, ext: str) -> Path:
        d = self.folder(h)
        if self.synthetic:
            names = [f"{stem}_SYNTHETIC.{ext}"]
        else:
            names = [f"{stem}.{ext}"] + ([f"{stem}_PARTIAL.{ext}"] if self.allow_partial else [])
        for nm in names:
            p = d / nm
            if p.exists():
                if nm.endswith(f"_PARTIAL.{ext}"):
                    self.partial = True
                self.used[f"{h}/{stem}"] = str(p)
                return p
        hint = "" if self.synthetic or self.allow_partial or not (d / f"{stem}_PARTIAL.{ext}").exists() else \
            " (only a _PARTIAL file exists: the final run is not finished; --allow-partial renders a stamped draft)"
        raise FileNotFoundError(f"{d}: none of {names}{hint}")

    def json(self, h: str, stem: str) -> dict:
        p = self.file(h, stem, "json")
        d = json.loads(p.read_text(encoding="utf-8"))
        labels = {r.get("label", "") for r in d.get("results", {}).values()}
        if any("SYNTHETIC" in lab for lab in labels) and not self.synthetic:
            raise RuntimeError(f"rule 6: {p} carries a SYNTHETIC label but its path is not a SYNTHETIC folder")
        if self.synthetic and not all("SYNTHETIC" in lab for lab in labels):
            raise RuntimeError(f"{p}: SYNTHETIC path but non-synthetic result labels {labels}")
        return d

    def seg_terms(self, day: str, cols: list[str]) -> pd.DataFrame:
        p = self.file("h4", f"seg_h4_terms_{day}", "parquet")
        df = pd.read_parquet(p, columns=cols + (["is_synthetic"] if self.synthetic else []))
        if self.synthetic:
            assert df["is_synthetic"].astype(bool).all(), "rule 6: synthetic folder with non-synthetic rows"
        return df

    @property
    def tag(self) -> str:
        """'_H1T' for subdir 'synthetic_H1T' (synthetic output file names)."""
        sub = self.subdir or self.folder("h1").name
        return sub.replace("synthetic", "") if sub.startswith("synthetic") else f"_{sub}"

    def out_png(self, stem: str, out: Path | None) -> Path:
        if out is not None:
            out = Path(out)
            if self.synthetic and "SYNTHETIC" not in str(out.resolve()).upper():
                raise RuntimeError("rule 6: synthetic figure must be written under a SYNTHETIC folder")
            return out
        if self.synthetic:
            return OUT_SYN / f"{stem}_SYNTHETIC{self.tag}.png"
        if self.subdir not in (None, "results"):
            raise RuntimeError(f"sub-folder '{self.subdir}' is not the primary run: pass --out explicitly")
        return OUT_PAPER / f"{stem}.png"

    def qa_dir(self, qa: Path | None) -> Path | None:
        if qa is not None:
            return Path(qa)
        return OUT_SYN / "qa" if self.synthetic else None


def stamp(fig: plt.Figure, rs: ResultSet, label: str) -> list:
    """SYNTHETIC watermark (rule 6) or PARTIAL draft stamp.  Returns the artists (exempt from overlap QA)."""
    arts = []
    if rs.synthetic:
        arts.append(fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", rotation=28, fontsize=150,
                             fontweight="bold", color="#BDBDBD", alpha=0.16, zorder=1000))
        arts.append(fig.text(0.995, 0.004, f"SYNTHETIC planted-effect test data (scenario {rs.tag.strip('_')}), "
                             "not observations",
                             ha="right", va="bottom", fontsize=7.0, color="#8A8580", zorder=1000))
    elif rs.partial:
        arts.append(fig.text(0.5, 0.5, "PARTIAL", ha="center", va="center", rotation=28, fontsize=150,
                             fontweight="bold", color="#9A9A9A", alpha=0.20, zorder=1000))
        arts.append(fig.text(0.995, 0.004, "DRAFT \u2014 " + label, ha="right", va="bottom", fontsize=7.0,
                             color="#8A8580", zorder=1000))
    for a in arts:
        a._qa_exempt = True
    return arts


def card_frame(ax: plt.Axes) -> mpatches.FancyBboxPatch:
    """Fig 9 read-out card idiom (white, taupe edge, round pad, alpha 0.95) as a frame filling the axes."""
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fr = mpatches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.025",
                                 transform=ax.transAxes, facecolor="white", edgecolor=P["taupe"], linewidth=1.0,
                                 alpha=0.95, zorder=1, clip_on=False)
    ax.add_patch(fr)
    return fr


def ctext(ax, x, y, s, size=8.2, weight="bold", color=None, ha="left", va="center", **kw):
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size, fontweight=weight,
                   color=color or P["ink"], zorder=5, **kw)


def day_legend_handles():
    return [Line2D([0], [0], color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"], ms=5.5,
                   mfc=st["mfc"], mec=st["color"], mew=1.2, label=day) for day, st in DAY_STYLE.items()]


# ---------------------------------------------------------------------------------------------- QA
def _text_bbox(t, r):
    bp = t.get_bbox_patch()
    return bp.get_window_extent(r) if bp is not None else t.get_window_extent(r)


def collect_texts(fig):
    """(owner, text, bbox) for every visible, non-empty text after drawing (ticks inside the view only)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        items = list(ax.texts)
        if ax.axison:
            items += [ax.xaxis.label, ax.yaxis.label]
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for tk in axis.get_major_ticks():
                    if tk.label1.get_visible() and lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                        items.append(tk.label1)
        if ax.name == "polar":
            items += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
        leg = ax.get_legend()
        if leg is not None:
            items += list(leg.get_texts()) + ([leg.get_title()] if leg.get_title().get_text() else [])
        for t in items:
            if t.get_visible() and t.get_text().strip():
                out.append((ax, t, _text_bbox(t, r)))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append((None, t, _text_bbox(t, r)))
    seen, uniq = set(), []
    for o in out:
        if id(o[1]) not in seen:
            seen.add(id(o[1]))
            uniq.append(o)
    return uniq


def qa_texts(fig, card_axes=(), contain_axes=()) -> dict:
    """Text/text overlap (> 1 px^2 at the figure dpi), text outside the sheet, card text outside its card,
    smallest authored and printed size (190 mm).  Watermark / stamp texts are exempt from overlaps."""
    fig.set_dpi(S.DPI)                                   # measure at the saved resolution (300 dpi pixels)
    texts = collect_texts(fig)
    fbb = fig.bbox
    live = [x for x in texts if not getattr(x[1], "_qa_exempt", False)]
    hits, tight = [], []
    pad = 0.6 / 72 * S.DPI                               # < 0.6 pt horizontal gap on one line = 'touching'
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i][2], live[j][2]
            if a.overlaps(b):
                ov = (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 1.0:
                    hits.append([live[i][1].get_text()[:40], live[j][1].get_text()[:40], round(float(ov), 1)])
                    continue
            yov = min(a.y1, b.y1) - max(a.y0, b.y0)
            gap = max(a.x0, b.x0) - min(a.x1, b.x1)
            if yov > 0.5 * min(a.height, b.height) and -1.0 < gap < pad:
                tight.append([live[i][1].get_text()[:40], live[j][1].get_text()[:40], round(float(gap), 1)])
    clipped = [t.get_text()[:40] for (_, t, bb) in live
               if bb.x0 < fbb.x0 - 0.5 or bb.y0 < fbb.y0 - 0.5 or bb.x1 > fbb.x1 + 0.5 or bb.y1 > fbb.y1 + 0.5]
    r = fig.canvas.get_renderer()
    spill = []
    for ax in list(card_axes) + list(contain_axes):
        abb = ax.get_window_extent(r)
        for (o, t, bb) in live:
            if o is ax and not _is_letter(t) and (bb.x0 < abb.x0 - 0.5 or bb.x1 > abb.x1 + 0.5 or
                                                  bb.y0 < abb.y0 - 0.5 or bb.y1 > abb.y1 + 0.5):
                spill.append(t.get_text()[:40])
    sizes = sorted({round(float(t.get_fontsize()), 2) for (_, t, _) in live})
    k = S.PRINT_WIDTH_IN / fig.get_size_inches()[0]
    return dict(n_texts=len(live), text_text_overlaps=hits, text_text_touching=tight, clipped_outside_sheet=clipped,
                spill_out_of_container=spill, author_pt_sizes=sizes, smallest_author_pt=sizes[0],
                smallest_printed_pt_190mm=round(sizes[0] * k, 2))


def _is_letter(t) -> bool:
    s = t.get_text()
    return len(s) == 3 and s[0] == "(" and s[2] == ")"


def save_crops(fig, png: Path, panels: dict, crop_dir: Path) -> list[str]:
    """100 % crops (native 300 dpi pixels) of each panel = union of its axes' tight boxes, from the saved PNG."""
    from PIL import Image
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    im = Image.open(png)
    W, H = im.size
    k = S.DPI / fig.dpi
    crop_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, axes in panels.items():
        bbs = [ax.get_tightbbox(r) for ax in axes]
        x0 = min(b.x0 for b in bbs) * k - 12; x1 = max(b.x1 for b in bbs) * k + 12
        y0 = max(b.y1 for b in bbs) * k + 12; y1 = min(b.y0 for b in bbs) * k - 12
        box = (max(int(x0), 0), max(int(H - y0), 0), min(int(x1), W), min(int(H - y1), H))
        p = crop_dir / f"{png.stem}__{name}.png"
        im.crop(box).save(p)
        out.append(str(p))
    return out


def finish(fig, rs: ResultSet, png: Path, panels: dict, card_axes, qa_extra: dict, qa_dir: Path | None,
           contain_axes=()) -> dict:
    """letters -> (a); v2 type scale; QA after scaling; save (full 13.0 in sheet, no crop); crops."""
    S.finalize_panel_labels(fig)
    n_scaled = S.scale_text(fig)
    qa = qa_texts(fig, card_axes=card_axes, contain_axes=contain_axes)
    qa.update(n_text_scaled=n_scaled, fig_size_in=[round(float(v), 3) for v in fig.get_size_inches()],
              png=str(png), inputs=rs.used, synthetic=rs.synthetic, partial=rs.partial, **qa_extra)
    png.parent.mkdir(parents=True, exist_ok=True)
    S.save_figure(fig, png, crop=None, close=False)
    if qa_dir is not None:
        qa["crops"] = save_crops(fig, png, panels, qa_dir.parent / "crops" if qa_dir.name == "qa" else qa_dir / "crops")
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / f"{png.stem}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    plt.close(fig)
    return qa


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                    help="analysis root with h1/ h2/ h4/ (default outputs/street/analysis) or one flat folder")
    ap.add_argument("--subdir", default=None, help="run sub-folder inside h1/ h2/ h4/ (default 'results')")
    ap.add_argument("--allow-partial", action="store_true",
                    help="render _PARTIAL result files as a stamped DRAFT (never the default)")
    ap.add_argument("--out", type=Path, default=None, help="override output PNG (tests)")
    ap.add_argument("--qa", type=Path, default=None, help="QA json + 100 %% crops folder (default for SYNTHETIC: "
                                                          "outputs/street/SYNTHETIC/fig56/qa)")
    ap.add_argument("--all", action="store_true", help="render Figure 5 and Figure 6 from the same results")


# =============================================================================================
# Figure 5 data
# =============================================================================================
def lcz_color(k: int) -> str:
    return NATURAL_COLOR if k in NATURAL else LCZ_COLORS.get(k, P["taupe"])


def lcz_edge(k: int) -> str:
    return P["taupe_dark"] if k in NATURAL else lcz_dark(k)


def rain_color(k: int) -> str:
    return P["taupe"] if k in NATURAL else LCZ_COLORS.get(k, P["taupe"])


def load_fig5(rs: ResultSet) -> dict:
    h1 = rs.json("h1", "h1_results")
    res = h1["results"]
    if set(res) != {d for d, _ in DAYS}:
        raise RuntimeError(f"H1 results for {sorted(res)} only; the figure needs both design days")
    segs = {}
    for day, _ in DAYS:
        df = rs.seg_terms(day, ["seg_id", "m1", "lcz", "lcz_valid"])
        med = float(df["m1"].median())                     # the day median over all segments (Fig 4 anomaly)
        df["anom"] = df["m1"] - med
        segs[day] = dict(df=df, median=med)
    return dict(h1=h1, res=res, segs=segs)


def h1_verdict(res: dict) -> tuple[bool, dict]:
    """Lock section 2 H1 rule with A7 item 2, re-evaluated from the numbers (cross-checked against the JSON)."""
    per = {}
    for day, _ in DAYS:
        rsh = res[day]["lmm"]["shares"]["resid_share"]
        cv = res[day]["cv"]["r2_lcz_only"]
        per[day] = dict(c1=rsh > 0.5, c2=cv < 0.30)
    ok = all(v["c1"] and v["c2"] for v in per.values()) and len(per) == 2
    return ok, per


# =============================================================================================
# Figure 5 panels
# =============================================================================================
def half_violin(ax, x, vals, color, edge, lo, hi, width=0.36, side=-1):
    from scipy.stats import gaussian_kde
    v = vals[np.isfinite(vals)]
    if len(v) > 5000:
        v = np.random.default_rng(20261030).choice(v, 5000, replace=False)
    a, b = max(np.min(vals), lo), min(np.max(vals), hi)
    grid = np.linspace(a, b, 220)
    dens = gaussian_kde(v)(grid)
    w = width * dens / dens.max()                         # each class scaled to its own peak (density_norm='width')
    ax.fill_betweenx(grid, x, x + side * w, facecolor=color, edgecolor=edge, linewidth=0.6, alpha=0.92, zorder=2)


def draw_raincloud(ax, day, date, seg, classes, lo, hi, top, show_ylabel, rng):
    df = seg["df"]
    v = df[df["lcz_valid"].astype(bool)]
    ns = {}
    box_data, pos, cols = [], [], []
    for i, k in enumerate(classes):
        vals = v.loc[v["lcz"] == k, "anom"].to_numpy(float)
        ns[k] = int(len(vals))
        if len(vals) == 0:
            continue
        half_violin(ax, i - 0.02, vals, lcz_color(k), lcz_edge(k), lo, hi)
        # rain: Fig 4 strip grammar (120 per group, size 2.2 -> s ~ 4.8, alpha 0.18)
        sub = vals if len(vals) <= 120 else rng.choice(vals, 120, replace=False)
        ax.scatter(i + 0.20 + rng.uniform(0, 0.20, len(sub)), sub, s=4.8, color=rain_color(k), alpha=0.24,
                   edgecolors="none", zorder=1.8, rasterized=False)
        box_data.append(vals); pos.append(i + 0.085); cols.append(lcz_color(k))
    bp = ax.boxplot(box_data, positions=pos, widths=0.11, whis=1.5, showfliers=False, patch_artist=True,
                    manage_ticks=False, zorder=3,
                    medianprops=dict(color=S.PANEL_EDGE, linewidth=1.1, solid_capstyle="butt"),
                    whiskerprops=dict(color=S.PANEL_EDGE, linewidth=0.8),
                    capprops=dict(color=S.PANEL_EDGE, linewidth=0.8),
                    boxprops=dict(edgecolor=S.PANEL_EDGE, linewidth=0.8))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    ax.axhline(0.0, color=S.SPINE_GREY, linestyle=":", linewidth=1.0, zorder=1.5)   # Fig 9 'FUR Med.' baseline
    ax.set_xlim(-0.55, len(classes) - 0.45)
    ax.set_ylim(lo, top)
    S.apply_box_gradient(ax)                                 # Fig 4 box sheen (fig4:119-145)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{LCZ_LABEL.get(k, k)}\n{kfmt(ns[k])}" for k in classes], fontsize=8.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(True, axis="y", color=P["grid"], linewidth=0.7)
    ax.grid(False, axis="x")
    for sp in ax.spines.values():                             # v1 Fig 4: pale full frame
        sp.set_color(P["grid"]); sp.set_linewidth(1.0)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))   # integer-friendly steps
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda y, _: "0" if abs(y) < 1e-9 else (f"{y:+g}".replace("-", MINUS))))
    if show_ylabel:
        ax.set_ylabel("\u0394M1 vs day median (\u00b0C)")
    else:
        ax.tick_params(axis="y", labelleft=False)
    n = int(len(df))
    ax.text(0.012, 0.975, f"{day} \u00b7 {date} \u00b7 day median {seg['median']:.1f}\u00b0C \u00b7 n = {n:,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2, fontweight="bold", color=P["ink"],
            bbox={"facecolor": "white", "edgecolor": P["taupe"], "boxstyle": "round,pad=0.25", "alpha": 0.95},
            zorder=10)
    return ns


LADDER_KEYS = [("segment", "segment"), ("50", "50 m"), ("100", "100 m"), ("250", "250 m"), ("500", "500 m")]
X_LMM, X_CV = 5.95, 7.62


def draw_ladder(ax, res):
    off = {"IMBAT": -0.11, "CALM": 0.11}
    for day, _ in DAYS:
        st = DAY_STYLE[day]
        lad = res[day]["ladder_eta2"]
        keys = [k for k, _ in LADDER_KEYS if k in lad]
        xs = np.array([[k for k, _ in LADDER_KEYS].index(k) for k in keys], float) + off[day]
        ys = np.array([lad[k]["eta2"] for k in keys])
        lo = np.array([lad[k]["ci95"][0] for k in keys]); hi = np.array([lad[k]["ci95"][1] for k in keys])
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.6, zorder=3, solid_capstyle="round")
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none", ecolor=st["color"], elinewidth=1.1, capsize=2.5,
                    zorder=4)
        ax.scatter(xs, ys, s=34, marker="o", facecolor=st["mfc"], edgecolor=st["color"], linewidth=1.2, zorder=5)
        # decision quantities (not joined to the ladder)
        lm = res[day]["lmm"]
        y1 = 1.0 - lm["shares"]["resid_share"]
        c1 = [1.0 - lm["ci95"]["resid_share"][1], 1.0 - lm["ci95"]["resid_share"][0]]
        cv = res[day]["cv"]
        y2, c2 = cv["r2_lcz_only"], cv["r2_lcz_only_ci95"]
        for x, y, c in ((X_LMM, y1, c1), (X_CV, y2, c2)):
            ax.errorbar([x + off[day]], [y], yerr=[[y - c[0]], [c[1] - y]], fmt="none", ecolor=st["color"],
                        elinewidth=1.1, capsize=2.5, zorder=4)
            ax.scatter([x + off[day]], [y], s=46, marker="D", facecolor=st["mfc"], edgecolor=st["color"],
                       linewidth=1.2, zorder=5)
    # lock limits, drawn only over the quantity they apply to (thin dashed, Fig 6 reference-line idiom)
    for x, y, lab in ((X_LMM, 0.50, "H1 needs < 0.50"), (X_CV, 0.30, "H1 needs < 0.30")):
        ax.plot([x - 0.48, x + 0.48], [y, y], color=P["muted"], ls=(0, (4, 3)), lw=1.1, zorder=2)
        ax.text(x, y + 0.025, lab, ha="center", va="bottom", fontsize=6.6, color=P["muted"], style="italic", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.axvline(5.15, color=P["grid"], lw=1.0, ls=(0, (1, 2)), zorder=1)
    ax.set_xticks(list(range(len(LADDER_KEYS))) + [X_LMM, X_CV])
    ax.set_xticklabels([lab for _, lab in LADDER_KEYS] + ["1 \u2212 within share\n(LMM)", "LCZ-only\nCV R\u00b2"])
    ax.set_xlim(-0.5, X_CV + 0.62)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Share of M1 variance explained")
    ax.text(2.0, -0.215, "\u03b7\u00b2 of M1 by LCZ class, per unit", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.2, color=P["muted"])
    ax.text((X_LMM + X_CV) / 2, -0.215, "H1 decision quantities", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.2, color=P["muted"])
    S.trend_axes_style(ax, first=True)                        # Fig 9 trend sub-row: taupe bottom/left, dashed y-grid
    ax.tick_params(axis="x", labelsize=9.24, colors=P["muted"], length=0)
    ax.tick_params(axis="y", labelsize=9.24, colors=P["muted"])
    ax.legend(handles=day_legend_handles(), loc="upper left", ncol=2, **S.LEGEND_LINES_SMALL)


def draw_card(ax, res, ok_all, per):
    card_frame(ax)
    xl, xd = 0.035, {"IMBAT": 0.615, "CALM": 0.855}
    y = 0.925
    for day, _ in DAYS:
        ctext(ax, xd[day], y, day, ha="center")
    rows = [
        ("within district \u00d7 LCZ", lambda r: r["lmm"]["shares"]["resid_share"],
         lambda r: r["lmm"]["ci95"]["resid_share"], "c1", "> 0.50"),
        ("LCZ-only CV R\u00b2", lambda r: r["cv"]["r2_lcz_only"], lambda r: r["cv"]["r2_lcz_only_ci95"], "c2", "< 0.30"),
        ("morphology CV R\u00b2", lambda r: r["cv"]["r2_morphology_lgbm"],
         lambda r: r["cv"]["r2_morphology_lgbm_ci95"], None, ""),
    ]
    dy_row, dy_ci = 0.172, 0.070
    y -= 0.150
    for lab, fv, fc, key, rule in rows:
        ctext(ax, xl, y, lab)
        if rule:
            ctext(ax, xl, y - dy_ci, f"lock: {rule}", size=7.0, weight="normal", color=P["muted"])
        for day, _ in DAYS:
            r = res[day]
            mark = "" if key is None else ("  " + (CHECK if per[day][key] else CROSS))
            v_ = fv(r)
            nd_ = 3 if round(v_, 2) in (0.50, 0.30) else 2     # never print a value that rounds onto its limit
            ctext(ax, xd[day], y, f"{num(v_, nd_)}{mark}", ha="center")
            ctext(ax, xd[day], y - dy_ci, ci_txt(fc(r)), size=7.0, weight="normal", color=P["muted"], ha="center")
        y -= dy_row
    ctext(ax, xl, y, "variance estimator")
    for day, _ in DAYS:
        m = res[day]["lmm"]["point"]["method"]
        ctext(ax, xd[day], y, "REML" if m.startswith("mixedlm") else "ANOVA (A7)", ha="center", weight="normal")
    y -= 0.095
    ax.plot([xl, 1 - xl], [y, y], transform=ax.transAxes, color=P["grid"], lw=1.0, zorder=4)
    y -= 0.090
    ctext(ax, xl, y, "H1, lock rule on both days:")
    ctext(ax, 1 - xl, y, "supported" if ok_all else "not supported", ha="right")


def build(rs: ResultSet):
    S.apply_style()
    np.random.seed(20261030)
    rng = np.random.default_rng(20261030)
    d = load_fig5(rs)
    res, segs = d["res"], d["segs"]
    ok_all, per = h1_verdict(res)
    json_verdict = d["h1"]["decision"]["H1"]
    if (json_verdict == "SUPPORTED") != ok_all:
        raise RuntimeError(f"H1 verdict mismatch: figure rule {ok_all} vs results JSON {json_verdict}")

    # classes shown: valid LCZ, not pooled (lock 1: < 30 segments pooled into 101/102, reported separately)
    pooled = {cm.POOL_BUILT, cm.POOL_NAT}
    classes = sorted({int(k) for day, _ in DAYS for k in segs[day]["df"].loc[segs[day]["df"]["lcz_valid"].astype(bool), "lcz"].unique()} - pooled)
    excl = {}
    for day, _ in DAYS:
        df = segs[day]["df"]
        valid = df["lcz_valid"].astype(bool)
        excl[day] = dict(pooled=int((valid & df["lcz"].isin(pooled)).sum()), a6=int((~valid).sum()),
                         lmm_n=int(res[day]["lmm_n_segments"]), valid=int(valid.sum()))
        if excl[day]["valid"] != excl[day]["lmm_n"]:
            raise RuntimeError(f"{day}: segment set of (a) ({excl[day]['valid']}) != H1 LMM n ({excl[day]['lmm_n']})")
    allv = np.concatenate([segs[day]["df"].loc[segs[day]["df"]["lcz_valid"].astype(bool), "anom"].to_numpy(float)
                           for day, _ in DAYS])
    lo, hi = np.nanmin(allv), np.nanmax(allv)                   # full range: the cool shaded tail is a finding
    span = hi - lo
    lo, hi = lo - 0.04 * span, hi + 0.04 * span
    top = hi + 0.26 * (hi - lo)                                 # headroom for the facet tag

    FW = S.V2_FIG_WIDTH_IN
    L, R, T = 1.02, 0.22, 0.52
    A_H, A_XL, GAP_AB = 3.55, 1.20, 0.50
    B_H, B_XL, BOT = 3.05, 0.95, 0.30
    FH = T + A_H + A_XL + GAP_AB + B_H + B_XL + BOT
    fig = plt.figure(figsize=(FW, FH))
    fgap = 0.16
    fw = (FW - L - R - fgap) / 2
    ya = FH - T - A_H
    axA = [S.inch_axes(fig, L + j * (fw + fgap), ya, fw, A_H) for j in range(2)]
    ns = {}
    for j, (day, date) in enumerate(DAYS):
        ns[day] = draw_raincloud(axA[j], day, date, segs[day], classes, lo, hi, top, j == 0, rng)
    axA[1].sharey(axA[0])
    fig.text((L + fw + fgap / 2) / FW, (ya - 0.80) / FH, "LCZ class (number of segments)", ha="center", va="bottom",
             fontsize=10.08, color=P["ink"])
    e = excl["IMBAT"]
    fig.text((FW - R) / FW, (ya - 1.13) / FH,
             f"not shown: pooled classes (< {cm.MIN_CLASS_N} segments) n = {e['pooled']:,}; LCZ no-data (A6) n = {e['a6']:,}",
             ha="right", va="bottom", fontsize=6.6, color=P["muted"])
    S.panel_label(axA[0], "A", fontsize=15.0)

    yb = BOT + B_XL
    B_W = 6.65
    axB = S.inch_axes(fig, L, yb, B_W, B_H)
    draw_ladder(axB, res)
    S.panel_label(axB, "B", fontsize=15.0)
    cx = L + B_W + 0.55
    axC = S.inch_axes(fig, cx, yb + 0.05, FW - R - cx, B_H - 0.05)
    draw_card(axC, res, ok_all, per)
    S.panel_label(axC, "C", fontsize=15.0, y=1.025)

    label = res["IMBAT"]["label"]
    arts = stamp(fig, rs, label)
    meta = dict(classes_shown=classes, n_per_class=ns, excluded=excl, ylim=[float(lo), float(top)],
                verdict_rule=ok_all, verdict_json=json_verdict, per_day=per, label=label,
                day_median={d_: segs[d_]["median"] for d_, _ in DAYS})
    panels = {"a": axA, "b": [axB], "c": [axC]}
    return fig, meta, panels, [axC], arts


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common_args(ap)
    a = ap.parse_args(argv)
    if a.all and a.out is not None:
        ap.error("--out cannot be combined with --all (each figure has its own fixed output)")
    rs = ResultSet(a.results, a.subdir, a.allow_partial)
    fig, meta, panels, cards, _ = build(rs)
    png = rs.out_png("fig_5_h1_scale", a.out if not a.all else None)
    qa = finish(fig, rs, png, panels, cards, dict(meta=meta), rs.qa_dir(a.qa))
    print(json.dumps({k: qa[k] for k in ("png", "fig_size_in", "smallest_author_pt", "smallest_printed_pt_190mm",
                                         "text_text_overlaps", "text_text_touching", "clipped_outside_sheet",
                                         "spill_out_of_container")},
                     indent=1, ensure_ascii=False))
    print("H1 verdict (figure rule / JSON):", meta["verdict_rule"], meta["verdict_json"])
    if a.all:
        from scripts.street.figures import build_fig6_drivers as f6
        f6.main([x for x in (argv if argv is not None else sys.argv[1:]) if x != "--all"])
    return qa


if __name__ == "__main__":
    main()
