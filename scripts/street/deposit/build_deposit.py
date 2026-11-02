"""Build the public derived-data deposit for the street-scale analysis (no geometry, no source-layer attributes).

What goes out: per-segment derived statistics (morphometrics summarised in buffers, modelled UTCI/Tmrt/wind ratio,
LCZ labels, LST, residents weight and H5 outcomes). What stays in: line geometry, coordinates, the municipal road-class
attribute and every grid/tile identifier that encodes a location. Location codes are replaced by order-preserving
codes (rank in the sorted original list), so the registered spatial-block folds and block bootstrap, which sort block
names, are reproduced exactly from the deposit (verified by verify_deposit.py).

Usage (repo root):  python scripts/street/deposit/build_deposit.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "street" / "stats"))
import common as cm  # noqa: E402

FINAL = ROOT / "data/street/analysis/final_20261030"
H5 = ROOT / "outputs/street/analysis/h5/h5_seg_outcomes.parquet"
OUT = ROOT / "output/deposit/climorfa_street_derived_v1"

SEG_KEEP = ["seg_id", "district", "lcz_primary", "lcz_weak250", "length_m", "n_ps", "azimuth_deg", "hw_ratio",
            "hw_w_mean_m", "hw_valid_share", "bh_mean30", "bcover30", "tcover15", "tht15", "dist_coast_m", "elev_m",
            "slope_deg", "lst_med"]


def rank_code(s: pd.Series, prefix: str) -> pd.Series:
    """Order-preserving anonymous code: rank of each value in the sorted list of unique values."""
    u = sorted(s.dropna().astype(str).unique())
    width = len(str(len(u)))
    m = {v: f"{prefix}{i:0{width}d}" for i, v in enumerate(u)}
    return s.astype(str).map(m)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = cm.load_segments(FINAL / "segments.gpkg")  # adds x_mid, y_mid from the line geometry
    block = (np.floor(seg["x_mid"] / cm.BLOCK_M).astype(int).astype(str) + "_"
             + np.floor(seg["y_mid"] / cm.BLOCK_M).astype(int).astype(str))
    d = seg[SEG_KEEP].copy()
    d["is_motorway"] = (seg["road_class"] == "OTOYOL").astype(int)
    d["block_2km"] = rank_code(block, "B")
    d["tile"] = rank_code(seg["tile_id"], "T")
    for c, p in (("g050_id", "C050_"), ("g100_id", "C100_"), ("g250_id", "C250_"), ("g500_id", "C500_")):
        d[c.replace("g", "cell_").replace("_id", "m")] = rank_code(seg[c], p)
    d.to_csv(OUT / "segments_derived.csv", index=False)

    g250 = dict(zip(seg["g250_id"].astype(str), d["cell_250m"]))
    tile = dict(zip(seg["tile_id"].astype(str), d["tile"]))
    for day in cm.DAYS:
        m = pd.read_parquet(FINAL / f"seg_metrics_{day}.parquet")
        m.to_csv(OUT / f"seg_metrics_{day}.csv", index=False)
    h = pd.read_parquet(H5)
    assert not h["is_synthetic"].astype(bool).any()
    h = h.drop(columns=["is_synthetic"])
    h["tile_id"] = h["tile_id"].astype(str).map(tile)
    h["g250_id"] = h["g250_id"].astype(str).map(g250)
    h = h.rename(columns={"tile_id": "tile", "g250_id": "cell_250m"})
    assert h["tile"].notna().all() and h["cell_250m"].notna().all()
    h.to_csv(OUT / "h5_segment_outcomes.csv", index=False)

    files = sorted(p for p in OUT.glob("*.csv"))
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), encoding="utf-8")
    for p in files:
        print(p.name, pd.read_csv(p, nrows=0).shape[1], "columns", f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
