"""WP2.3b non-circular checks. Subcommands:
  backup   snapshot BEFORE state: sha256 manifest of all UTCI rasters, copy of CALM utci_street rasters and of
           seg_metrics_CALM.parquet, choice of the 2 check tiles (largest CALM utci_street files among extracted tiles)
  compare  AFTER state vs snapshot: hashes (IMBAT street/uniform and CALM uniform must be identical), band-wise change of
           CALM utci_street (non-VRB bands must be bit-identical), median/p5/p95 of the change in VRB bands for the check
           tiles, and an independent per-cell recomputation of two VRB hours with pythermalcomfort.
  imbat    recompute IMBAT of the 2 check tiles with the NEW code into outputs/.../wp23b/imbat_recompute and compare
           the raster hash with the original files (proves the new code path reproduces IMBAT bit-for-bit).
Outputs only under outputs/street/utci/wp23b/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/street/utci/wp23b"
UTCI = ROOT / "data/street/utci"
PY = r"C:\Users\YE\miniforge3\envs\climorfa-street\python.exe"
NODATA = -9999.0


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def manifest() -> pd.DataFrame:
    rows = []
    for day in ("IMBAT", "CALM"):
        for var in ("utci_street", "utci_uniform"):
            for p in sorted((UTCI / day / var).glob("*.tif")):
                rows.append(dict(day=day, var=var, tile=p.stem, sha256=sha(p), size=p.stat().st_size))
    return pd.DataFrame(rows)


def backup():
    OUT.mkdir(parents=True, exist_ok=True)
    m = manifest()
    m.to_csv(OUT / "hash_before.csv", index=False)
    dst = OUT / "before_CALM_utci_street"
    dst.mkdir(exist_ok=True)
    for p in (UTCI / "CALM/utci_street").glob("*.tif"):
        shutil.copy2(p, dst / p.name)
    shutil.copy2(ROOT / "data/street/analysis/seg_metrics_CALM.parquet", OUT / "before_seg_metrics_CALM.parquet")
    shutil.copy2(ROOT / "data/street/analysis/seg_metrics_IMBAT.parquet", OUT / "before_seg_metrics_IMBAT.parquet")
    done = pd.read_csv(ROOT / "data/street/analysis/WP3.0_tiles_done.csv")
    ext = sorted(done["tile_id"].unique())
    sz = m[(m.day == "CALM") & (m["var"] == "utci_street") & (m.tile.isin(ext))].sort_values("size", ascending=False)
    chk = sz["tile"].head(2).tolist()
    json.dump(dict(check_tiles=chk, n_tiles_before=int(m[(m.day == "CALM") & (m["var"] == "utci_street")].shape[0]),
                   n_extracted_before=len(ext)), open(OUT / "check_tiles.json", "w"))
    print("backup done; manifest rows", len(m), "check tiles", chk)


def read_tif(p):
    import rasterio
    with rasterio.open(p) as s:
        return s.read().astype(np.float64)


def compare():
    chk = json.load(open(OUT / "check_tiles.json"))["check_tiles"]
    before = pd.read_csv(OUT / "hash_before.csv")
    after = manifest()
    mm = before.merge(after, on=["day", "var", "tile"], how="outer", suffixes=("_b", "_a"), indicator=True)
    print("tiles only before/after:", (mm["_merge"] == "left_only").sum(), "/", (mm["_merge"] == "right_only").sum())
    both = mm[mm["_merge"] == "both"].copy()
    both["same"] = both["sha256_b"] == both["sha256_a"]
    summ = both.groupby(["day", "var"])["same"].agg(["sum", "count"])
    print("raster hash identical (n identical / n both-present):\n", summ.to_string())
    # tiles that existed before for CALM street
    tiles = sorted(both[(both.day == "CALM") & (both["var"] == "utci_street")]["tile"])
    flags = pd.read_csv(ROOT / "data/street/forcing_summer/vrb_flags.csv")
    vrb_it = sorted(flags[(flags.day == "CALM") & flags.vrb & (flags.it >= 6) & (flags.it <= 21)]["it"].tolist())
    print("CALM VRB-flagged hours it:", vrb_it, "-> raster bands", [i - 5 for i in vrb_it])
    # band-wise bit-identity on all before-existing CALM tiles
    changed = {it: 0 for it in range(6, 22)}
    ident = {it: 0 for it in range(6, 22)}
    nan_mismatch = 0
    chg_rows = []
    for t in tiles:
        b = read_tif(OUT / "before_CALM_utci_street" / f"{t}.tif")
        a = read_tif(UTCI / "CALM/utci_street" / f"{t}.tif")
        for it in range(6, 22):
            bi, ai = b[it - 6], a[it - 6]
            if np.array_equal(bi, ai):
                ident[it] += 1
            else:
                changed[it] += 1
            nan_mismatch += int(((bi == NODATA) != (ai == NODATA)).sum())
            if t in chk and it in vrb_it:
                v = (bi != NODATA) & (ai != NODATA) & np.isfinite(bi) & np.isfinite(ai)
                d = (ai - bi)[v]
                chg_rows.append(dict(tile=t, hour_it=it, raster_band=it - 5, n_cells=int(v.sum()),
                                     median=np.median(d), p5=np.percentile(d, 5), p95=np.percentile(d, 95),
                                     mean=d.mean(), min=d.min(), max=d.max(), frac_changed=float((d != 0).mean()),
                                     before_median_utci=np.median(bi[v]), after_median_utci=np.median(ai[v])))
    tab = pd.DataFrame(dict(it=list(changed), n_tiles_identical=[ident[i] for i in changed],
                            n_tiles_changed=[changed[i] for i in changed], vrb_flagged=[i in vrb_it for i in changed]))
    print(f"CALM utci_street band-wise comparison over {len(tiles)} tiles (valid/nodata mask mismatches: {nan_mismatch}):")
    print(tab.to_string(index=False))
    tab.to_csv(OUT / "calm_band_identity.csv", index=False)
    pd.DataFrame(chg_rows).to_csv(OUT / "calm_vrb_change_checktiles.csv", index=False)
    print("UTCI_street change (after - before, degC) on check tiles, VRB hours:")
    print(pd.DataFrame(chg_rows).round(3).to_string(index=False))
    both[["day", "var", "tile", "same"]].to_csv(OUT / "hash_compare.csv", index=False)

    # independent per-cell recomputation for check tile 0, VRB hours 14,15,16 (and a non-VRB hour 13)
    import rasterio
    from pythermalcomfort.models import utci
    from scripts.street.utci.vrb_utils import MET_V2
    met = pd.read_csv(MET_V2["CALM"], sep=r"\s+").set_index("it")
    t = chk[0]
    with rasterio.open(ROOT / f"data/street/wind/ratio16/{t}.tif") as s:
        r16 = s.read().astype(np.float32)
    with rasterio.open(ROOT / f"data/street/solweig/CALM/tmrt/{t}.tif") as s:
        tm = s.read().astype(np.float64)
    a = read_tif(UTCI / "CALM/utci_street" / f"{t}.tif")
    rng = np.random.default_rng(1)
    out = []
    for it in (14, 15, 16, 13):
        valid = np.argwhere((a[it - 6] != NODATA) & np.isfinite(tm[it - 6]) & (tm[it - 6] != NODATA))
        pick = valid[rng.choice(len(valid), 3000, replace=False)]
        rr, cc = pick[:, 0], pick[:, 1]
        r_cell = r16[:, rr, cc]                                # (16, n)
        if it in vrb_it:
            ratio = r_cell.astype(np.float64).mean(axis=0)     # independent float64 mean of the 16 sectors
        else:
            wd = met.loc[it, "Wd"]
            s_ = (wd % 360.0) / 22.5
            k0 = int(np.floor(s_)) % 16
            f = s_ - np.floor(s_)
            ratio = (1 - f) * r_cell[k0].astype(np.float64) + f * r_cell[(k0 + 1) % 16].astype(np.float64)
        v = np.clip(met.loc[it, "Wind"] * ratio, 0.5, 17.0)
        ref = utci(tdb=float(met.loc[it, "Td"]), tr=tm[it - 6][rr, cc], v=v, rh=float(met.loc[it, "RH"]),
                   limit_inputs=False, round_output=False).utci
        diff = np.abs(ref - a[it - 6][rr, cc])
        out.append(dict(tile=t, hour_it=it, vrb=it in vrb_it, n=3000, max_abs_diff=float(diff.max()),
                        mean_abs_diff=float(diff.mean())))
    od = pd.DataFrame(out)
    print("independent per-cell recomputation vs written raster (float32 storage => expect ~1e-5):")
    print(od.to_string(index=False))
    od.to_csv(OUT / "independent_cell_recompute.csv", index=False)


def imbat():
    chk = json.load(open(OUT / "check_tiles.json"))["check_tiles"]
    dst = OUT / "imbat_recompute"
    if dst.exists():
        shutil.rmtree(dst)
    subprocess.run([PY, str(ROOT / "scripts/street/utci/compute_street_utci.py"), "--force-day", "IMBAT", "--tiles", *chk,
                    "--out-dir", str(dst)], cwd=str(ROOT), check=True)
    rows = []
    for t in chk:
        for var in ("utci_street", "utci_uniform"):
            p0 = UTCI / "IMBAT" / var / f"{t}.tif"
            p1 = dst / "IMBAT" / var / f"{t}.tif"
            a, b = read_tif(p0), read_tif(p1)
            rows.append(dict(tile=t, var=var, sha_orig=sha(p0)[:16], sha_recomputed=sha(p1)[:16],
                             file_hash_equal=sha(p0) == sha(p1), pixels_equal=bool(np.array_equal(a, b))))
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    d.to_csv(OUT / "imbat_recompute_check.csv", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"backup": backup, "compare": compare, "imbat": imbat}[sys.argv[1]]()
