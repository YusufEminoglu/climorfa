"""Rule-based choice of the two 1 km tiles of the graphical abstract (WP4.7).

Run from the repo root:
    C:/Users/YE/miniforge3/envs/climorfa-street/python.exe -m scripts.street.figures.select_ga_tiles [--out <dir>]
    -> <dir>/ga_candidates.csv (eligible tiles), <dir>/ga_pairs.csv (all same-class pairs, best first)

RULE
  Eligible tile: one of the 427 final tiles; dominant global LCZ class (the Fig 1 pixel share: most frequent class of the
  50 m pixels in the tile, values 1-16, water and no data out of the denominator) is 3 or 8, the two classes with the most
  segments; that class covers >= 70 % of the pixels; >= 50 street segments.
  Pair: two eligible tiles of the same class.
  Contrast of a pair: mean of four absolute differences, each divided by that metric's standard deviation over all eligible
  tiles: median building height, median segment svf_med, building cover, share of pedestrian-space cells more than 2 degC
  below the day median (breezy day, 15:00).  The smallest of the four (min_z) is reported next to it, so that a pair carried
  by one metric alone is visible.
  Guards against a share that does not describe a built 1 km tile: land >= 95 % of the tile (a share on 76 pixels of a tile
  that is mostly sea is not a 1 km tile), building cover >= 10 % (a built class), and >= 70 % of the tile's segments carry the
  same primary label (lcz_primary, the label behind the R2 values).  Pairs that fail a guard stay in the output, flagged.
  Strong pair: every one of the four differences is at least 1 SD (min_z >= 1).  Cigli AOSB stays if it has a strong partner
  that passes the guards; otherwise the pair with the highest contrast among strong pairs that pass the guards is chosen.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

import rasterio
from rasterio.warp import Resampling, reproject
from shapely.geometry import box as shp_box

from scripts.street.figures import build_fig1_study_area as F1
from scripts.street.figures import build_fig4_street_utci as F4
from scripts.street.figures import build_graphical_abstract as G

ROOT = G.ROOT
MAHALLE = ROOT / "data/01_raw/population/mahalle_population_controls_2024.gpkg"
CLASSES = (3, 8)
MIN_SHARE, MIN_SEG = 0.70, 50
METRICS = ["med_h", "med_svf", "bld_cover", "below2"]
MIN_LAND, MIN_COVER, MIN_SEG_LABEL, STRONG_Z = 0.95, 0.10, 0.70, 1.0
KEEP = "T_0503_4262"                 # Cigli AOSB, already in Figs 1 and 4


def lcz_shares(tiles: gpd.GeoDataFrame, lcz_arr, lcz_tr) -> pd.DataFrame:
    """Fig 1 window and denominator; water share (class 17) kept apart so that it can be reported."""
    n = int(1000 / abs(lcz_tr.a))
    rows = []
    for tid, t in tiles.iterrows():
        r0 = int(round(((float(t["y0"]) + 1000.0) - lcz_tr.f) / lcz_tr.e))
        c0 = int(round((float(t["x0"]) - lcz_tr.c) / lcz_tr.a))
        win = lcz_arr[r0:r0 + n, c0:c0 + n]
        vals, cnts = np.unique(win[(win > 0) & (win < 17)], return_counts=True)
        if not len(vals):
            continue
        k = int(vals[cnts.argmax()])
        rows.append(dict(tile_id=tid, lcz=k, lcz_share=float(cnts.max() / cnts.sum()), n_px=int(cnts.sum()),
                         n_water=int((win == 17).sum()), second=int(vals[np.argsort(-cnts)[1]]) if len(vals) > 1 else 0,
                         second_share=float(np.sort(cnts)[-2] / cnts.sum()) if len(vals) > 1 else 0.0))
    return pd.DataFrame(rows).set_index("tile_id")


def filtered_share(tid: str, klass: int, tiles, lcz_arr, lcz_tr) -> float:
    """Share of ``klass`` on band 2 (LCZ_Filter) of the same raster, same grid and window: robustness of the label."""
    n = int(1000 / abs(lcz_tr.a))
    t = tiles.loc[tid]
    r0 = int(round(((float(t["y0"]) + 1000.0) - lcz_tr.f) / lcz_tr.e))
    c0 = int(round((float(t["x0"]) - lcz_tr.c) / lcz_tr.a))
    win = lcz_arr[r0:r0 + n, c0:c0 + n]
    win = win[(win > 0) & (win < 17)]
    return float((win == klass).sum() / win.size)


def load_band2(shape, tr) -> np.ndarray:
    dst = np.full(shape, 255, dtype=np.uint8)
    with rasterio.open(ROOT / "data/01_raw/gee/lcz/izmir_fua_rub_lcz_latest.tif") as src:
        reproject(source=rasterio.band(src, 2), destination=dst, dst_transform=tr, dst_crs="EPSG:5253",
                  resampling=Resampling.nearest, src_nodata=None, dst_nodata=255)
    return dst


def neighbourhoods(tid: str, tiles, mah: gpd.GeoDataFrame) -> list:
    """Neighbourhoods of the project's population layer inside the tile, largest overlap first, with their area share."""
    t = tiles.loc[tid]
    box = shp_box(float(t["x0"]), float(t["y0"]), float(t["x0"]) + 1000.0, float(t["y0"]) + 1000.0)
    m = mah[mah.intersects(box)].copy()
    m["a"] = m.geometry.intersection(box).area / 1e6
    return [(r["district_name"], r["mahalle_name"], round(float(r["a"]), 3))
            for _, r in m.sort_values("a", ascending=False).iterrows() if r["a"] > 0.01]


def tile_metrics(tid: str, med: dict) -> dict:
    h, _, lm = F1.tile_arrays(tid)
    land = lm > 0.5
    bld = (h >= F1.BLD_VMIN) & land
    out = dict(med_h=float(np.median(h[bld])) if bld.any() else np.nan, bld_cover=float(bld.sum() / land.sum()),
               land_share=float(land.mean()))
    for day, tag in ((G.DAY, ""), (G.CALM, "_calm")):
        u, _ = F4.tile_layers(tid, day)
        a = u.compressed() - med[day]
        out.update({f"below2{tag}": float((a < -2).mean()), f"above2{tag}": float((a > 2).mean()),
                    f"anom_med{tag}": float(np.median(a)), f"anom_p10{tag}": float(np.percentile(a, 10)),
                    f"anom_p90{tag}": float(np.percentile(a, 90)), f"n_ps{tag}": int(a.size)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/street/figures/ga_selection")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    ctx = G.load_context()
    days = G.load_numbers()
    tiles = ctx["tiles"]
    if len(tiles) != F4.N_TILES_TOTAL:
        raise SystemExit(f"{len(tiles)} tiles, expected {F4.N_TILES_TOTAL}")
    t = lcz_shares(tiles, ctx["lcz_arr"], ctx["lcz_tr"])

    seg = pyogrio.read_dataframe(G.FINAL / "segments.gpkg", layer="segments", read_geometry=False,
                                 columns=["seg_id", "tile_id", "district", "lcz_primary", "length_m"])
    seg = seg.merge(days[G.DAY]["met"][["seg_id", "svf_med", F4.BAND_COL]], on="seg_id", how="left")
    print("segments by label:", seg["lcz_primary"].value_counts().head(4).to_dict())
    grp = seg.groupby("tile_id")
    t = t.join(pd.DataFrame(dict(
        n_seg=grp.size(), med_svf=grp["svf_med"].median(),
        district=grp["district"].agg(lambda s: s.value_counts().index[0]),
        n_districts=grp["district"].nunique(),
        district_share=grp["district"].agg(lambda s: s.value_counts(normalize=True).iloc[0]),
        seg_label=grp["lcz_primary"].agg(lambda s: int(s.value_counts().index[0])),
        seg_label_share=grp["lcz_primary"].agg(lambda s: s.value_counts(normalize=True).iloc[0]))))
    print(f"tiles {len(t)}, with segments {int(t['n_seg'].notna().sum())}")

    e = t[t["lcz"].isin(CLASSES) & (t["lcz_share"] >= MIN_SHARE) & (t["n_seg"] >= MIN_SEG)].copy()
    print("eligible by class:", e["lcz"].value_counts().to_dict())
    med = {d: days[d]["stats"]["median"] for d in (G.DAY, G.CALM)}
    e = e.join(pd.DataFrame({tid: tile_metrics(tid, med) for tid in e.index}).T)

    band2 = load_band2(ctx["lcz_arr"].shape, ctx["lcz_tr"])
    e["lcz_share_band2"] = [filtered_share(tid, int(e.at[tid, "lcz"]), tiles, band2, ctx["lcz_tr"]) for tid in e.index]
    e["guards_ok"] = (e["land_share"] >= MIN_LAND) & (e["bld_cover"] >= MIN_COVER) & (e["seg_label_share"] >= MIN_SEG_LABEL)
    e.to_csv(args.out / "ga_candidates.csv", encoding="utf-8-sig")

    sd = e[METRICS].std(ddof=1)
    rows = []
    for a, b in itertools.combinations(e.index, 2):
        if e.at[a, "lcz"] != e.at[b, "lcz"]:
            continue
        z = {m: abs(e.at[a, m] - e.at[b, m]) / sd[m] for m in METRICS}
        rows.append(dict(a=a, b=b, lcz=int(e.at[a, "lcz"]), score=float(np.mean(list(z.values()))),
                         min_z=float(min(z.values())), guards_ok=bool(e.at[a, "guards_ok"] and e.at[b, "guards_ok"]),
                         keeps_cigli=KEEP in (a, b), **{f"z_{m}": v for m, v in z.items()}))
    p = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    p["strong"] = p["min_z"] >= STRONG_Z
    p.to_csv(args.out / "ga_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"sd over {len(e)} eligible tiles:", sd.round(3).to_dict())
    print(f"pairs {len(p)}; pass guards {int(p['guards_ok'].sum())}; strong {int(p['strong'].sum())}; "
          f"both {int((p['guards_ok'] & p['strong']).sum())}; with Cigli strong {int((p['keeps_cigli'] & p['strong']).sum())}")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    ok = p[p["guards_ok"]]
    for name, q in (("top pairs, no guards", p.head(8)), ("top pairs, guards", ok.head(8)),
                    ("top LCZ 8 pairs, guards", ok[ok["lcz"] == 8].head(8)),
                    ("top pairs with Cigli, no guards", p[p["keeps_cigli"]].head(5)),
                    ("top pairs with Cigli, guards", ok[ok["keeps_cigli"]].head(5))):
        print(); print("==", name)
        print(q.round(2).to_string())

    strong = ok[ok["strong"]]
    keep = strong[strong["keeps_cigli"]]
    print()
    print("Cigli kept:", not keep.empty, "; best min_z of a Cigli pair:", round(float(ok[ok["keeps_cigli"]]["min_z"].max()), 2))
    print("chosen:", (keep if not keep.empty else strong).iloc[0][["a", "b", "lcz", "score", "min_z"]].to_dict())

    mah = gpd.read_file(MAHALLE).to_crs(tiles.crs)
    cols = ["lcz", "lcz_share", "lcz_share_band2", "second", "second_share", "n_px", "n_seg", "seg_label_share", "district",
            "district_share", "med_h", "med_svf", "bld_cover", "below2", "above2", "anom_med", "anom_p10", "anom_p90",
            "below2_calm", "above2_calm", "land_share"]
    top = list(dict.fromkeys(list(ok.head(6)[["a", "b"]].to_numpy().ravel()) + [KEEP]))
    print(); print("== tiles of the top pairs"); print(e.loc[top, cols].round(3).T.to_string())
    for tid in top:
        print(tid, neighbourhoods(tid, tiles, mah))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
