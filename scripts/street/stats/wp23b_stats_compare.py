"""WP2.3b: read-only comparison of H1/H4 PARTIAL results before / after the VRB correction (key numbers straight from the JSONs).

    python wp23b_stats_compare.py [official_results_dir_suffix]    (default: '' = outputs/street/analysis/{h1,h4}/results)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
A = Path(r"C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas\outputs\street\analysis")
suffixes = {"before (46 tiles, old code+tables)": "_wp23b_before", "after (46 tiles, new)": "_wp23b_after46",
            "official run_all (current tables)": sys.argv[1] if len(sys.argv) > 1 else ""}


def f(x, lo=None, hi=None, nd=3):
    if x is None:
        return "NA"
    return f"{x:.{nd}f}" if lo is None else f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


rows = []
for lab, suf in suffixes.items():
    p1 = A / "h1" / f"results{suf}" / "h1_results_PARTIAL.json"
    p4 = A / "h4" / f"results{suf}" / "h4_results_PARTIAL.json"
    if not p1.exists() or not p4.exists():
        rows.append(dict(run=lab, note="missing"))
        continue
    h1, h4 = json.load(open(p1)), json.load(open(p4))
    for day in ("CALM", "IMBAT"):
        r, d = h1["results"][day], h1["decision"][day]
        s, ci, c = r["lmm"]["shares"], r["lmm"]["ci95"], r["cv"]
        q = h4["results"][day] if "results" in h4 else h4[day]
        rows.append(dict(
            run=lab, day=day, label=q.get("label"), n_seg=r["n_segments"],
            H1_resid_share=f(s["resid_share"], *ci["resid_share"]), H1_cvR2_LCZ=f(c["r2_lcz_only"], *c["r2_lcz_only_ci95"]),
            H1_eta2_segment=f(r["ladder_eta2"]["segment"]["eta2"], *r["ladder_eta2"]["segment"]["ci95"]),
            H1_lmm=f"{r['lmm']['point']['method']} conv={r['lmm']['point']['converged']} bnd={r['lmm']['point'].get('boundary')}",
            H1_day_pass=d["day_pass"],
            H4_varShade=f(q["var_shade"], *q["var_shade_ci95"]), H4_varBreeze=f(q["var_breeze"], *q["var_breeze_ci95"], nd=4),
            H4_ratio=f(q["ratio_var_shade_over_breeze"], *q["ratio_ci95"], nd=1), H4_ratio_gt_4=q["ratio_gt_4"],
            H4_DiD_shade=f(h4["orientation_hw_interaction_DiD"][day]["shade_term"]["did"], *h4["orientation_hw_interaction_DiD"][day]["shade_term"]["ci95"]),
            H4_DiD_breeze=f(h4["orientation_hw_interaction_DiD"][day]["breeze_term"]["did"], *h4["orientation_hw_interaction_DiD"][day]["breeze_term"]["ci95"], nd=4),
            v_ref=f(q["reference"]["v_ref"]), Wd_ref=f(q["forcing"]["wd_mean_15_17"], nd=1),
            R=f(q["forcing"].get("wd_ref_R"))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
for day in ("CALM", "IMBAT"):
    print(f"\n===== {day} =====")
    print(df[df["day"] == day].drop(columns=["day"]).T.to_string(header=False))
print("\ndecisions:", {lab: (json.load(open(A / 'h1' / f'results{suf}' / 'h1_results_PARTIAL.json'))['decision']['H1'],
                           json.load(open(A / 'h4' / f'results{suf}' / 'h4_results_PARTIAL.json')).get('decision', {}))
                       for lab, suf in suffixes.items() if (A / 'h1' / f'results{suf}' / 'h1_results_PARTIAL.json').exists()})
