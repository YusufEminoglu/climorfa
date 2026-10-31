"""WP2.3b: seg_metrics_CALM.parquet before vs after the VRB correction (snapshot in outputs/street/utci/wp23b/)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
W = R / "outputs/street/utci/wp23b"
A = R / "data/street/analysis"
b = pd.read_parquet(W / "before_seg_metrics_CALM.parquet").sort_values("seg_id").reset_index(drop=True)
a = pd.read_parquet(A / "seg_metrics_CALM.parquet").sort_values("seg_id").reset_index(drop=True)
print("rows before/after", len(b), len(a), "| same seg_id set:", set(b.seg_id) == set(a.seg_id), "| duplicated seg_id after:", int(a.seg_id.duplicated().sum()))
assert list(b.columns) == list(a.columns)
rows = []
for c in b.columns:
    if c == "seg_id":
        continue
    x, y = b[c].to_numpy(), a[c].to_numpy()
    same = (x == y) | (pd.isna(x) & pd.isna(y))
    d = (y.astype(float) - x.astype(float)) if np.issubdtype(x.dtype, np.number) and x.dtype != bool else None
    rows.append(dict(col=c, n_changed=int((~same).sum()), frac_changed=round(float((~same).mean()), 4),
                     median_change=None if d is None else float(np.nanmedian(d[~same])) if (~same).any() else 0.0,
                     p5=None if d is None else float(np.nanpercentile(d[~same], 5)) if (~same).any() else 0.0,
                     p95=None if d is None else float(np.nanpercentile(d[~same], 95)) if (~same).any() else 0.0))
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(W / "seg_metrics_CALM_before_after.csv", index=False)
print("partial_run", a.partial_run.unique(), "n_tiles", a.n_tiles_at_extraction.unique())
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
print("IMBAT parquet sha256 equal to pre-change snapshot:", h(W / "before_seg_metrics_IMBAT.parquet") == h(A / "seg_metrics_IMBAT.parquet"))
