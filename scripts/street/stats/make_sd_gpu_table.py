"""CLIMORFA Street — Supplementary S-D table of the four model-run variants on the 40-tile subset, from the run_all
--gpu-sens outputs (no number typed by hand).

    python -m scripts.street.stats.make_sd_gpu_table
Reads outputs/street/analysis/sens_gpu/gpu_sens_{headlines,paired_m1_m3}.csv and gpu_sens_comparison.json; writes
paper/manuscript_v2/src/tables/sd/tab_sd_subset.tex. Every variant is compared with the primary run cut to the same tiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
A = ROOT / "outputs/street/analysis/sens_gpu"
OUT = ROOT / "paper/manuscript_v2/src/tables/sd/tab_sd_subset.tex"
LABEL = {"radsens": "RADSENS (shortwave $\\times$0.88)", "h26": "Storey height 2.6~m", "h34": "Storey height 3.4~m",
         "landcover": "Land cover (A9)"}


def sgn(x: float, nd: int) -> str:
    return f"{x:+.{nd}f}".replace("-", "$-$")


def main():
    h = pd.read_csv(A / "gpu_sens_headlines.csv")
    p = pd.read_csv(A / "gpu_sens_paired_m1_m3.csv")
    verdict = json.loads((A / "gpu_sens_comparison.json").read_text(encoding="utf-8"))["h1_verdict_on_40_tiles"]
    get = lambda v, d, m, c="value": h.query("variant == @v and day == @d and metric == @m")[c].iloc[0]  # noqa: E731
    pm = lambda v, d, m, c: p.query("variant == @v and day == @d and metric == @m")[c].iloc[0]  # noqa: E731
    rows = []
    for i, d in enumerate(("IMBAT", "CALM")):
        v0 = "radsens"
        rows.append(f"{'Primary, same 40 tiles' if i == 0 else ''} & {d} & {get(v0, d, 'h1_resid_share', 'primary40'):.3f} & "
                    f"{get(v0, d, 'h1_cv_r2_lcz_only', 'primary40'):.3f} & {get(v0, d, 'h1_cv_r2_morphology', 'primary40'):.3f} & "
                    f"{get(v0, d, 'h2_share_shade_geometry', 'primary40'):.2f} & {get(v0, d, 'h2_share_ventilation', 'primary40'):.2f} "
                    f"& -- & -- & -- \\\\")
    rows.append("\\addlinespace[2pt]")
    for v in LABEL:
        if verdict[v] != "SUPPORTED":
            raise RuntimeError(f"{v}: H1 {verdict[v]}; the table caption would be wrong")
        for i, d in enumerate(("IMBAT", "CALM")):
            rows.append(f"{LABEL[v] if i == 0 else ''} & {d} & {get(v, d, 'h1_resid_share'):.3f} & {get(v, d, 'h1_cv_r2_lcz_only'):.3f} & "
                        f"{get(v, d, 'h1_cv_r2_morphology'):.3f} & {get(v, d, 'h2_share_shade_geometry'):.2f} & "
                        f"{get(v, d, 'h2_share_ventilation'):.2f} & {sgn(pm(v, d, 'M1', 'mean_delta'), 2)} "
                        f"[{sgn(pm(v, d, 'M1', 'p05'), 2)}, {sgn(pm(v, d, 'M1', 'p95'), 2)}] & "
                        f"{sgn(pm(v, d, 'M3', 'mean_delta'), 2)} & {sgn(pm(v, d, 'M2_38', 'mean_delta'), 2)} \\\\")
        rows.append("\\addlinespace[2pt]")
    rows.pop()
    n = int(p["n_segments"].iloc[0])
    tex = ("\\begin{table}[t]\n\\centering\\footnotesize\n"
           f"\\caption{{H1 and H2 headline values under the model-run variants, on the 40-tile subset ({n:,} segments). Each variant "
           "is compared with the primary run cut to the same tiles, not with the 427-tile result. H1 is supported under the lock "
           "rule in the primary subset and in all four variants. Shade and Vent.\\ are the family shares of mean $|$SHAP$|$. "
           "$\\Delta$ columns are paired per-segment means, variant minus primary; the brackets give the 5th and 95th percentile of "
           "$\\Delta$M1 across segments.}\n\\label{tab:sd_subset}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lcccccccrr}\n"
           "\\toprule\nVariant & Day & Within share & LCZ-only CV $R^2$ & Morphology CV $R^2$ & Shade & Vent. & "
           "$\\Delta$M1 (\\textdegree{}C) & $\\Delta$M3 (\\textdegree{}C) & $\\Delta\\mathrm{M2}_{38}$ (h) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    OUT.write_text(tex, encoding="utf-8", newline="\n")
    print(tex)


if __name__ == "__main__":
    main()
