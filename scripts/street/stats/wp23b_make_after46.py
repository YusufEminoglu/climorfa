"""WP2.3b: 46-tile AFTER table for a like-for-like before/after comparison. The live CALM table has since grown to more tiles
(the re-runner extracted new tiles with the corrected code). Keep exactly the segments of the pre-change 46-tile snapshot,
labelled n_tiles_at_extraction = 46 (a segment belongs to exactly one extraction tile; asserted: no duplicate seg_id)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas")
W = R / "outputs/street/utci/wp23b"
b = pd.read_parquet(W / "before_seg_metrics_CALM.parquet")
a = pd.read_parquet(R / "data/street/analysis/seg_metrics_CALM.parquet")
assert not a["seg_id"].duplicated().any()
sub = a[a["seg_id"].isin(set(b["seg_id"]))].copy().sort_values("seg_id").reset_index(drop=True)
assert len(sub) == len(b) and (sub["seg_id"].to_numpy() == b.sort_values("seg_id")["seg_id"].to_numpy()).all()
sub["partial_run"] = True
sub["n_tiles_at_extraction"] = np.int64(int(b["n_tiles_at_extraction"].max()))
sub.to_parquet(W / "after46_seg_metrics_CALM.parquet", index=False)
print("after46 rows", len(sub), "live rows", len(a), "n_tiles", sub["n_tiles_at_extraction"].unique())
# unchanged columns must equal the snapshot (sanity)
bb = b.sort_values("seg_id").reset_index(drop=True)
same_cols = [c for c in b.columns if c != "seg_id" and ((bb[c] == sub[c]) | (bb[c].isna() & sub[c].isna())).all()]
print("columns identical to before-snapshot:", len(same_cols), "of", len(b.columns) - 1)
