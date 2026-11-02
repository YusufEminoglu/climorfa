"""Reproduce registered H1 numbers from the public derived-data deposit alone (no geometry).

Rebuilds the analysis frames with the registered helpers, takes the spatial block from the deposit's block_2km code
instead of segment coordinates, assigns the registered folds and recomputes the LCZ-only and morphology CV R^2 with
the registered block-bootstrap interval for the LCZ-only model. Expected: 0.0300 [IMBAT], 0.0109 [CALM] and
0.7057 / 0.6271 (h1_results.json).

Usage:  python scripts/street/deposit/verify_deposit.py [deposit_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "street" / "stats"))
import common as cm  # noqa: E402

DEP = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output/deposit/climorfa_street_derived_v1"


def frames_from_deposit(dep: Path) -> dict[str, pd.DataFrame]:
    seg = pd.read_csv(dep / "segments_derived.csv")
    seg["x_mid"] = 0.0
    seg["y_mid"] = 0.0
    frames = {}
    for day in cm.DAYS:
        met = pd.read_csv(dep / f"seg_metrics_{day}.csv")
        f = cm.build_frame(seg, met, day)
        f["block"] = f["block_2km"]
        f.attrs["forcing"] = cm.forcing_summary(day)
        frames[day] = f
    cm.assign_shared_folds(frames)
    return frames


def main():
    frames = frames_from_deposit(DEP)
    out = {}
    for day, df in frames.items():
        y = df["m1"].to_numpy(float)
        vf = df["lcz_valid"].to_numpy(float)
        p_lcz = np.nan_to_num(cm.lcz_only_oof(df))
        p_mor = cm.lgb_oof(df, cm.FEATURES, n_jobs=cm.MAX_PROCS)
        ub, inv, counts = cm.boot_block_counts(df["block"].to_numpy(), cm.N_BOOT, cm.SEED)
        bs = np.array([cm.wr2(y, p_lcz, counts[r][inv] * vf) for r in range(counts.shape[0])])
        out[day] = dict(n=len(df), n_blocks=len(ub), cv_r2_lcz_only=round(cm.wr2(y, p_lcz, vf), 4),
                        ci95_lcz_only=[round(float(v), 3) for v in cm.ci95(bs)],
                        cv_r2_morphology=round(cm.wr2(y, p_mor, vf), 4))
        print(day, out[day], flush=True)
    (DEP / "verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
