"""Post-hoc agreement measures for the Stage 1 manual audit (descriptive, not part of any lock).

Reads the frozen primary-tier rows (label quality high, weak label in LCZ 3/6/8/9) and reports exact agreement together with
three coarser readings: agreement among units whose audited class lies inside the four-class label
space, built versus natural agreement, and agreement on the compactness family (LCZ 1-3 compact, 4-6 open, 7-10 other built).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ROWS = ROOT / "outputs/diagnostics/audited_model_validation_2026-07-31/audited_population_rows.csv"
OUT = ROOT / "outputs/street/analysis/posthoc/audit.json"


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def code(label: str) -> str:
    return str(label).split("_")[0]


def family(c: str) -> str:
    if c.isdigit():
        k = int(c)
        return "compact" if k <= 3 else ("open" if k <= 6 else "other_built")
    return "natural"


def measure(df: pd.DataFrame, pred_col: str) -> dict:
    weak = df[pred_col].astype(float).astype(int).astype(str)
    aud = df["audit_label"].map(code)
    exact = (weak == aud)
    inscope = aud.isin(["3", "6", "8", "9"])
    built = aud.str.isdigit()
    fam = weak.map(family) == aud.map(family)
    out = {}
    for name, mask, hit in [("exact", np.ones(len(df), bool), exact), ("within_four_class_scope", inscope, exact),
                            ("built_vs_natural", np.ones(len(df), bool), built), ("compactness_family", np.ones(len(df), bool), fam)]:
        n = int(np.sum(mask)); k = int(np.sum(hit & mask))
        out[name] = dict(k=k, n=n, pct=round(100 * k / n, 1) if n else None, wilson95=wilson(k, n))
    per_class = {}
    for c in ["3", "6", "8", "9"]:
        m = weak == c
        per_class[f"LCZ {c}"] = dict(n=int(m.sum()), audited_built=int((built & m).sum()), exact=int((exact & m).sum()),
                                     audited_compact=int(((aud.map(family) == "compact") & m).sum()))
    out["per_weak_class"] = per_class
    from sklearn.metrics import cohen_kappa_score
    out["kappa"] = {
        "exact_classes": round(float(cohen_kappa_score(weak, aud)), 3),
        "compactness_family": round(float(cohen_kappa_score(weak.map(family), aud.map(family))), 3),
        "built_vs_natural": None,   # the weak label is built for every unit, so kappa is undefined
        "note": "Mappings defined after the results were seen (post hoc).",
    }
    return out


def main() -> None:
    d = pd.read_csv(ROWS)
    prim = d[d["label_quality"] == "high"].copy()
    res = dict(source=str(ROWS.relative_to(ROOT)), n_rows=int(len(d)), n_primary=int(len(prim)),
               reviewer_count=int(d["audit_reviewer"].nunique()), source_layers=sorted(d["audit_source_layers"].unique().tolist()),
               weak_label=measure(prim, "lcz_weak_label"), lightgbm=measure(prim, "lightgbm_predicted_label"),
               note="Unweighted sample proportions; design weights were not applied. Post hoc, descriptive.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
