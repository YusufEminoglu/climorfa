"""CLIMORFA Street — Supplementary S-D table of the H5 variants, from the evaluation reports (no number typed by hand).

    python -m scripts.street.h5.make_sd_h5_table
Reads outputs/street/analysis/h5/h5_evaluation_report.json, h5_pop_all/h5_evaluation_report.json and
h5/h5_decomposition.json; writes paper/manuscript_v2/src/tables/sd/tab_sd_h5.tex.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("C:/Users/YE/PyCharmMiscProject/izmir_lcz_deep_morphology_mert_yavas").resolve()
A = ROOT / "outputs/street/analysis"
OUT = ROOT / "paper/manuscript_v2/src/tables/sd/tab_sd_h5.tex"


def load(rel: str) -> dict:
    return json.loads((A / rel).read_text(encoding="utf-8"))


def ratio(x: dict) -> str:
    lo, hi = x["ci_95"]
    return f"{x['median']:.2f} [{lo:.2f}, {hi:.2f}]"


def main():
    r, pa, dc = load("h5/h5_evaluation_report.json"), load("h5_pop_all/h5_evaluation_report.json"), load("h5/h5_decomposition.json")
    rows = []

    def add(label, n, rep, var, key):
        for i, d in enumerate(("IMBAT", "CALM")):
            x = rep["variants"][var]["days"][d]["targeting_ratio_S_vs_Z"]
            rel = rep["validation"]["days"][d][key]["rel_diff"] * 100
            rows.append(f"{label if i == 0 else ''} & {f'{n:,}' if i == 0 else ''} & {d} & {ratio(x)} & {rel:.2f}\\% \\\\")
        rows.append("\\addlinespace[2pt]")

    add("Primary: $\\mathrm{M2}_{38}$, residents", r["variants"]["M2_38"]["tree_budget_N"], r, "M2_38", "dm2_38")
    add("Locked $\\mathrm{M2}_{32}$, residents", r["variants"]["M2_32"]["tree_budget_N"], r, "M2_32", "dm2_32")
    add("$\\mathrm{M2}_{38}$, all-building population", pa["variants"]["M2_38"]["tree_budget_N"], pa, "M2_38", "dm2_38")
    a11 = r["a11"]["days"]
    for name, key in (("Cell-based companion (A11)", "cell_h38"), ("Continuous companion (A11)", "mean_utci_b15")):
        for i, d in enumerate(("IMBAT", "CALM")):
            rows.append(f"{name if i == 0 else ''} & & {d} & {ratio(a11[d]['companions'][key]['ratio_S_vs_Z'])} & -- \\\\")
        rows.append("\\addlinespace[2pt]")
    for name, key in (("No pedestrian weight (descriptive)", "ratio_unweighted_sum_dm2"),
                      ("Weighted by segment length (descriptive)", "ratio_length_weighted")):
        for i, d in enumerate(("IMBAT", "CALM")):
            rows.append(f"{name if i == 0 else ''} & & {d} & {dc['days'][d][key]:.2f} & -- \\\\")
        rows.append("\\addlinespace[2pt]")
    while rows[-1].startswith("\\addlinespace"):
        rows.pop()

    tex = ("\\begin{table}[t]\n\\centering\\footnotesize\n"
           "\\caption{H5 under its variants (427 tiles). S/Z is the median of the 20 per-draw ratios, with 95\\% of the draws "
           "in brackets; the lock rule requires at least 1.5 on IMBAT. The last column is the difference between the additive "
           "and the exact estimate on the 40 validation tiles (limit 15\\%, amendment A10), for the primary S segments. The "
           "last two blocks are a descriptive decomposition written after the verdict; they are not in the lock.}\n"
           "\\label{tab:sd_h5}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrccc}\n\\toprule\n"
           "Variant & $N$ & Day & S/Z & Additive vs exact \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    OUT.write_text(tex, encoding="utf-8", newline="\n")
    print(tex)


if __name__ == "__main__":
    main()
