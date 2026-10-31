"""WP3.0 QA. Distinguishes checks that are by construction (descriptive) from INDEPENDENT checks the method could fail.

INDEPENDENT checks (can fail):
  I1 network length conservation vs GEOS clipping of the raw roads to each tile (different algorithm from the custom splitter)
  I2 PS mask vs exact polygon rasterisation (buffer(15 m) of the road lines, rasterio.features.rasterize), 15 tiles
  I3 nearest-segment assignment vs exact shapely distances, random PS cells
  I4 UTCI consistency: UTCI recomputed with pythermalcomfort from segment-median Tmrt_b15, v10eff_b15 and the met row it==15
     must reproduce the segment-median utci_b15 (would fail if band index / hour were misaligned)
  I5 sign tests on physical expectations (SVF vs H/W, shade vs SVF, ...) on the PARTIAL dynamic table
  I6 250 m aggregates of bcover30 / tcover15 vs the old vector-footprint / canopy features of the 250 m grid
Descriptive (by construction): PS share per tile, segment length distribution, overlays, M1 by LCZ range check.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import *  # noqa

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
from scipy import stats

RNG = np.random.default_rng(20261030)
Q: dict = {}


def tile_ps(tid):
    with rasterio.open(PS_DIR / f"{tid}.tif") as s:
        return s.read(1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segs = gpd.read_file(SEGMENTS_GPKG)
    pre = gpd.read_file(PRE_PS_GPKG, layer="segments_pre_ps")
    tiles = gpd.read_file(TILES_CORE)
    st = pd.read_csv(OUT_DIR / "tile_ps_stats.csv")

    # ---------------- descriptive: PS share, lengths
    Q["n_segments_final"] = int(len(segs))
    Q["ps_share_of_land"] = {k: float(v) for k, v in st["ps_share_of_land"].describe().items()}
    Q["tiles_with_zero_ps"] = int((st["ps_cells"] == 0).sum())
    Q["segment_length_m_percentiles"] = {str(p): float(np.percentile(segs.length_m, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    Q["segments_lt20m"] = int((segs.length_m < 20).sum())
    Q["segments_gt500m"] = int((segs.length_m > 500).sum())
    Q["n_ps_percentiles"] = {str(p): float(np.percentile(segs.n_ps, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
    Q["dropped_segments_lt25_ps_total"] = int(st["n_small_segments"].sum())
    Q["cells_ps_total"] = int(st["ps_cells"].sum())
    Q["cells_foreign_dropped_total"] = int(st["foreign_dropped"].sum())
    Q["cells_small_seg_dropped_total"] = int(st["small_seg_dropped_cells"].sum())
    Q["road_class_counts_final"] = {k: int(v) for k, v in segs.road_class.value_counts().items()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(st["ps_share_of_land"].dropna(), bins=40, color="#4477aa")
    ax[0].set(xlabel="PS cells / land cells (inner 1 km tile)", ylabel="tiles", title="PS share of land per tile (427 tiles)")
    ax[1].hist(np.clip(segs.length_m, 0, 600), bins=60, color="#228833")
    ax[1].axvline(20, color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="segment length (m, clipped at 600)", ylabel="segments", title=f"Segment length (n={len(segs)})")
    ax[2].hist(np.log10(segs.n_ps), bins=60, color="#aa3377")
    ax[2].axvline(np.log10(25), color="k", ls="--", lw=0.8)
    ax[2].set(xlabel="log10 PS cells per segment", ylabel="segments", title="PS cells per segment (min 25)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qa_ps_share_and_lengths.png", dpi=130)
    plt.close(fig)

    # ---------------- overlays
    for tid in ("T_0512_4255", "T_0487_4248", "T_0484_4242"):
        overlay(tid, segs, pre)

    # ---------------- I1 length conservation
    raw = gpd.read_file(ROADS_GPKG, layer="roads_fua")
    raw = raw[raw.geometry.notna()]
    parts = shapely.get_parts(raw.geometry.values)
    tree = shapely.STRtree(parts)
    pre_len = pre.groupby("tile_id")["length_m"].sum()
    rows = []
    for _, r in tiles.iterrows():
        idx = tree.query(r.geometry, predicate="intersects")
        clipped = shapely.intersection(parts[idx], r.geometry)
        L = float(shapely.length(clipped).sum())
        rows.append((r.tile_id, L, float(pre_len.get(r.tile_id, 0.0))))
    d = pd.DataFrame(rows, columns=["tile_id", "raw_clipped_m", "segments_m"])
    d["diff_m"] = d.segments_m - d.raw_clipped_m
    d["rel"] = d["diff_m"] / d.raw_clipped_m.where(d.raw_clipped_m > 0)
    d.to_csv(OUT_DIR / "qa_I1_length_conservation.csv", index=False)
    Q["I1_total_raw_clipped_km"] = float(d.raw_clipped_m.sum() / 1000)
    Q["I1_total_segments_km"] = float(d.segments_m.sum() / 1000)
    Q["I1_max_abs_diff_m_per_tile"] = float(d.diff_m.abs().max())
    Q["I1_n_tiles_abs_diff_gt_1m"] = int((d.diff_m.abs() > 1.0).sum())
    Q["I1_worst_tile"] = d.loc[d.diff_m.abs().idxmax(), "tile_id"]

    # ---------------- I2 exact PS mask, I3 nearest assignment
    pre_tree = shapely.STRtree(pre.geometry.values)
    cand_tiles = st.loc[st.ps_cells > 0, "tile_id"].tolist()
    chosen = list(RNG.choice(cand_tiles, 15, replace=False))
    mask_rows = []
    agree = []
    gaps = []
    for tid in chosen:
        x0, y0 = tile_xy(tid)
        tf = from_origin(x0, y0 + TILE_M, RES, RES)
        box = shapely.box(x0 - 20, y0 - 20, x0 + TILE_M + 20, y0 + TILE_M + 20)
        gi = pre_tree.query(box, predicate="intersects")
        buf = shapely.union_all(shapely.buffer(pre.geometry.values[gi], PS_BUFFER_M))
        exact = features.rasterize([(buf, 1)], out_shape=(INNER_PX, INNER_PX), transform=tf, fill=0, all_touched=False).astype(bool)
        la = {}
        for lyr in ("dem", "building_dsm", "landmask"):
            with rasterio.open(TILE_DIR / lyr / f"{tid}.tif") as s:
                la[lyr] = s.read(1)[BUF_PX:BUF_PX + INNER_PX, BUF_PX:BUF_PX + INNER_PX]
        okc = (la["landmask"] == 1) & ((la["building_dsm"] - la["dem"]) <= 0.5) & (la["building_dsm"] != -9999) & (la["dem"] != -9999)
        exact &= okc
        ps = tile_ps(tid) > 0
        s_row = st[st.tile_id == tid].iloc[0]
        mask_rows.append({"tile_id": tid, "exact_cells": int(exact.sum()), "ps_cells": int(ps.sum()),
                          "in_exact_not_ps": int((exact & ~ps).sum()), "in_ps_not_exact": int((ps & ~exact).sum()),
                          "recorded_foreign_plus_small_dropped": int(s_row.foreign_dropped + s_row.small_seg_dropped_cells),
                          "ps_cells_15m_kd": int(s_row.ps_cells_15m)})
        # I3
        rr, cc = np.nonzero(ps)
        pick = RNG.choice(len(rr), min(1500, len(rr)), replace=False)
        px = x0 + RES / 2 + RES * cc[pick]
        py = y0 + TILE_M - RES / 2 - RES * rr[pick]
        pts = shapely.points(px, py)
        assigned = tile_ps(tid)[rr[pick], cc[pick]]
        qi, ti = pre_tree.query(pts, predicate="dwithin", distance=PS_BUFFER_M + 2.0)
        dist = shapely.distance(pts[qi], pre.geometry.values[ti])
        df = pd.DataFrame({"q": qi, "seg": pre.seg_id.values[ti], "d": dist})
        best = df.loc[df.groupby("q")["d"].idxmin()].set_index("q")
        da = df[df.seg.values == assigned[df.q.values]].groupby("q")["d"].min()
        for q in range(len(pts)):
            if q in best.index and q in da.index:
                agree.append(best.loc[q, "seg"] == assigned[q])
                gaps.append(da.loc[q] - best.loc[q, "d"])
            elif q in best.index:
                agree.append(False)
                gaps.append(np.nan)
    m = pd.DataFrame(mask_rows)
    m["iou"] = (m.exact_cells - m.in_exact_not_ps) / (m.exact_cells + m.in_ps_not_exact)
    m.to_csv(OUT_DIR / "qa_I2_ps_mask_vs_exact.csv", index=False)
    Q["I2_tiles_checked"] = len(m)
    Q["I2_sum_exact_cells"] = int(m.exact_cells.sum())
    Q["I2_sum_ps_cells"] = int(m.ps_cells.sum())
    Q["I2_sum_in_exact_not_ps"] = int(m.in_exact_not_ps.sum())
    Q["I2_sum_in_ps_not_exact"] = int(m.in_ps_not_exact.sum())
    Q["I2_sum_recorded_dropped"] = int(m.recorded_foreign_plus_small_dropped.sum())
    Q["I2_iou_min"] = float(m.iou.min())
    Q["I2_iou_mean"] = float(m.iou.mean())
    gaps = np.array(gaps, dtype=float)
    Q["I3_cells_checked"] = int(len(agree))
    Q["I3_assigned_is_exact_nearest_share"] = float(np.mean(agree))
    Q["I3_gap_m_median"] = float(np.nanmedian(gaps))
    Q["I3_gap_m_p99"] = float(np.nanpercentile(gaps, 99))
    Q["I3_gap_m_max"] = float(np.nanmax(gaps))

    # ---------------- dynamic table checks (PARTIAL)
    from pythermalcomfort.models import utci
    met = pd.read_csv(MET_FILES["IMBAT"], sep=r"\s+")
    dyn = {}
    for day in DAYS:
        t = pd.read_parquet(ANALYSIS_DIR / f"seg_metrics_{day}.parquet")
        dyn[day] = t
        ntiles = int(t.n_tiles_at_extraction.iloc[0])
        lab = f"PARTIAL ({ntiles} tiles)" if t.partial_run.iloc[0] else f"COMPLETE ({ntiles} tiles)"
        Q[f"dynamic_{day}_label"] = lab
        Q[f"dynamic_{day}_n_segments"] = int(len(t))
        # I4
        mm = pd.read_csv(MET_FILES[day], sep=r"\s+")
        r15 = mm[mm["it"] == 15].iloc[0]
        rec = utci(tdb=float(r15["Td"]), tr=t.tmrt_b15.values, v=t.v10eff_b15.values, rh=float(r15["RH"]),
                   limit_inputs=False, round_output=False).utci
        diff = rec - t.utci_b15.values
        Q[f"I4_{day}_utci_recompute_minus_segment_median_degC"] = {
            "median_abs": float(np.nanmedian(np.abs(diff))), "p99_abs": float(np.nanpercentile(np.abs(diff), 99)),
            "max_abs": float(np.nanmax(np.abs(diff))), "n": int(len(diff))}
        # I5 sign tests
        j = t.merge(segs.drop(columns="geometry"), on="seg_id")
        sign_tests = []
        for a, b, exp in (("svf_med", "hw_ratio", "-"), ("shade_frac_10_18", "svf_med", "-"), ("shade_frac_10_18", "tcover15", "+"),
                          ("tmrt_b15", "shade_frac_10_18", "-"), ("utci_b15", "shade_frac_10_18", "-"),
                          ("tmrt_b15", "svf_med", "+"), ("shade_frac_10_18", "hw_ratio", "+"), ("v10eff_b15", "bcover30", "-")):
            ok = j[[a, b]].dropna()
            rho, p = stats.spearmanr(ok[a], ok[b])
            sign_tests.append({"x": a, "y": b, "expected_sign": exp, "rho": float(rho), "n": int(len(ok)),
                               "passed": bool((rho < 0) == (exp == "-"))})
        Q[f"I5_{day}_sign_tests_{lab.replace(' ', '_')}"] = sign_tests
        # M1 by LCZ (range sanity only)
        g = j.groupby("lcz_primary")["utci_b15"].agg(n="count", min="min", p25=lambda s: s.quantile(.25), median="median",
                                                      p75=lambda s: s.quantile(.75), max="max").reset_index()
        g.insert(0, "label", lab)
        g.insert(0, "day", day)
        g.to_csv(OUT_DIR / f"qa_m1_by_lcz_{day}_PARTIAL.csv", index=False)
    # static plausibility on ALL segments
    stat_tests = []
    for a, b, exp in (("bcover30", "hw_ratio", "+"), ("bh_mean30", "hw_ratio", "+"), ("lst_med", "tcover15", "-"),
                      ("lst_med", "elev_m", "-"), ("elev_m", "slope_deg", "+"), ("bcover30", "tcover15", "-")):
        ok = segs[[a, b]].dropna()
        rho, p = stats.spearmanr(ok[a], ok[b])
        stat_tests.append({"x": a, "y": b, "expected_sign": exp, "rho": float(rho), "n": int(len(ok)),
                           "passed": bool((rho < 0) == (exp == "-"))})
    Q["I5_static_sign_tests_all_segments"] = stat_tests
    Q["lcz_primary_vs_weak250_exact_agreement"] = float((segs.lcz_primary == segs.lcz_weak250).mean())

    # ---------------- I6 250 m aggregates vs old features
    segs["w"] = segs.length_m
    agg = segs.groupby("g250_id").apply(lambda d: pd.Series({
        "bcover30": np.average(d.bcover30, weights=d.w), "tcover15": np.average(d.tcover15, weights=d.w), "nseg": len(d)}),
        include_groups=False).reset_index()
    kx = agg.g250_id.str.split("_").str[0].astype(int)
    ky = agg.g250_id.str.split("_").str[1].astype(int)
    w = gpd.read_file(WEAK_GPKG)[["grid_id", "geometry"]]
    b = w.geometry.bounds
    w["g250_id"] = (np.round(b.minx / 250).astype(int).astype(str) + "_" + np.round(b.miny / 250).astype(int).astype(str))
    mor = pd.read_csv(ROOT / "data/03_processed/grid_250m_building_morphometrics.csv")[["grid_id", "morph_building_area_share"]]
    can = pd.read_csv(ROOT / "data/03_processed/grid_250m_canopy_height_features.csv")[["grid_id", "canopy_area_gt2m_m2"]]
    j = agg.merge(w[["grid_id", "g250_id"]], on="g250_id").merge(mor, on="grid_id").merge(can, on="grid_id")
    j = j[j.nseg >= 3]
    r1, _ = stats.spearmanr(j.bcover30, j.morph_building_area_share)
    r2, _ = stats.spearmanr(j.tcover15, (j.canopy_area_gt2m_m2 / 62500.0).fillna(0))
    Q["I6_cells_nseg_ge3"] = int(len(j))
    Q["I6_spearman_bcover30_vs_vector_footprint_area_share"] = float(r1)
    Q["I6_spearman_tcover15_vs_old_canopy_area_gt2m_over_cell_area"] = float(r2)
    Q["I6_note"] = "old canopy layer has 83% mean cover >2 m (evidently includes buildings), so the tree-cover comparison is a weak, non-diagnostic reference; the building comparison uses independent vector footprints"

    (OUT_DIR / "qa_summary.json").write_text(json.dumps(Q, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(Q, indent=2, ensure_ascii=False))


def overlay(tid, segs, pre):
    x0, y0 = tile_xy(tid)
    ps = tile_ps(tid)
    with rasterio.open(TILE_DIR / "building_dsm" / f"{tid}.tif") as s:
        b = s.read(1)[BUF_PX:BUF_PX + INNER_PX, BUF_PX:BUF_PX + INNER_PX]
    with rasterio.open(TILE_DIR / "dem" / f"{tid}.tif") as s:
        d = s.read(1)[BUF_PX:BUF_PX + INNER_PX, BUF_PX:BUF_PX + INNER_PX]
    bld = (b - d) > 0.5
    rgb = np.ones((INNER_PX, INNER_PX, 3))
    rgb[bld] = (0.35, 0.35, 0.35)
    cmap = plt.get_cmap("tab20")
    m = ps > 0
    col = np.array([cmap(i % 20)[:3] for i in range(20)])
    rgb[m] = col[(ps[m] % 20)]
    ext = (x0, x0 + TILE_M, y0, y0 + TILE_M)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
    sub = pre[pre.tile_id == tid]
    kept = set(segs[segs.tile_id == tid].seg_id)
    for a, zoom in zip(ax, (None, (x0 + 350, x0 + 650, y0 + 350, y0 + 650))):
        a.imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
        for g, sid in zip(sub.geometry.values, sub.seg_id.values):
            xy = shapely.get_coordinates(g)
            a.plot(xy[:, 0], xy[:, 1], color="k" if sid in kept else "red", lw=0.7 if zoom is None else 1.4)
            if zoom is not None:
                mx, my = shapely.get_coordinates(shapely.line_interpolate_point(g, 0.5, normalized=True))[0]
                if zoom[0] < mx < zoom[1] and zoom[2] < my < zoom[3]:
                    a.text(mx, my, str(sid), fontsize=7, color="k", bbox=dict(fc="white", alpha=0.6, lw=0, pad=0.5))
        a.set_xlim(*(zoom[:2] if zoom else ext[:2]))
        a.set_ylim(*(zoom[2:] if zoom else ext[2:]))
        a.set_aspect("equal")
    ax[0].set_title(f"{tid}: PS cells coloured by nearest segment (white = not PS, grey = building); black = kept segments, red = dropped (<25 PS)")
    ax[1].set_title("300 m zoom with seg_id")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"overlay_{tid}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    lower_priority()
    main()
