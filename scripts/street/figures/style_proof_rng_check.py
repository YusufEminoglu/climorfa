"""Diagnostic: is the residual difference in Fig 4 panel B only seaborn's unseeded stripplot jitter?
Renders panel B with (i) the ORIGINAL helper and (ii) climorfa_style, under identical np.random seeds,
and with different seeds, and compares each with the stored original PNG band."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scripts.street.figures import climorfa_style as S
from scripts.street.figures import style_proof as P
import build_figure_4_morphology_multipanel as f4

Image.MAX_IMAGE_PIXELS = None
OUT = P.OUT

def data():
    df = pd.read_csv("data/03_processed/grid_250m_model_features_v8.csv")
    a = f4._analysis_subset(df)
    return a

def render(kind: str, seed: int, a) -> np.ndarray:
    S.apply_style()
    fig = plt.figure(figsize=S.FIGSIZES["fig4_profiles"], constrained_layout=False)
    gs_b = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.650, bottom=0.488, wspace=0.08)
    ax = fig.add_subplot(gs_b[0, 0:4])
    np.random.seed(seed)
    if kind == "orig":
        f4._plot_profile_boxplots(ax, a)
    else:
        fields = list(f4.PROFILE_FIELDS)
        sel = ["building_coverage_exact","height_proxy_aw_mean_m","dsm_elevation_m_std","canopy_cover_gt2m_share","s2_ndvi_mean","road_density_exact_m_per_km2","lst_c_median_mean"]
        rows = [pd.DataFrame({"variable": f4.PROFILE_FIELDS[f], "value": S.scaled_1_99(a[f]), "class": a["class"]}) for f in sel]
        S.class_boxplot_panel(ax, pd.concat(rows, ignore_index=True), [f4.PROFILE_FIELDS[f] for f in sel], letter="B")
    f4._finalize_panel_labels(fig)
    p = OUT / f"_tmp_{kind}_{seed}.png"
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    arr = np.array(Image.open(p).convert("RGB")); p.unlink()
    H = arr.shape[0]
    t = int(round((1 - 0.650) * H)) - 90; b = int(round((1 - 0.488) * H)) + 84
    return arr[t:b]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = data()
    o_s0, m_s0 = render("orig", 0, a), render("mine", 0, a)
    o_s1 = render("orig", 1, a)
    res = {
        "mine_vs_orig_helper_same_seed_MAE": P.mae_rgb(o_s0, m_s0),
        "mine_vs_orig_helper_same_seed_max_abs_diff": int(np.abs(o_s0.astype(int) - m_s0.astype(int)).max()),
        "orig_helper_seed0_vs_seed1_MAE (pure jitter noise floor)": P.mae_rgb(o_s0, o_s1),
    }
    # stored PNG band for reference
    o = np.array(Image.open(P.ORIG / "fig_4_class_profiles.png").convert("RGB"))
    w = min(o.shape[1], o_s0.shape[1]); bh = o_s0.shape[0]
    g_o, g_n = o.mean(axis=2), o_s0.mean(axis=2)
    errs = {dy: np.abs(g_o[dy:dy+bh, :w] - g_n[:, :w]).mean() for dy in range(870, 910)}
    dy = min(errs, key=errs.get)
    res["stored_png_offset_px"] = dy
    res["stored_vs_orig_helper_seed0_MAE"] = P.mae_rgb(o[dy:dy+bh, :w], o_s0[:, :w])
    res["stored_vs_mine_seed0_MAE"] = P.mae_rgb(o[dy:dy+bh, :w], m_s0[:, :w])
    print(json.dumps(res, indent=2))
    (OUT / "fig4_rng_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
