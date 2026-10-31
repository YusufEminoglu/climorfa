"""Diagnostic: the stored Fig 4 PNG was written with bbox_inches=<fractional crop>, so its content sits at a
sub-pixel vertical phase relative to a plain render.  Sweep that phase with the climorfa_style render."""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from PIL import Image
from scripts.street.figures import climorfa_style as S
from scripts.street.figures import style_proof as P
import build_figure_4_morphology_multipanel as f4
Image.MAX_IMAGE_PIXELS = None

def render(phase_px: float, a) -> np.ndarray:
    S.apply_style()
    W, Hh = S.FIGSIZES["fig4_profiles"]
    fig = plt.figure(figsize=(W, Hh), constrained_layout=False)
    gs_b = fig.add_gridspec(1, 4, left=0.035, right=0.988, top=0.650, bottom=0.488, wspace=0.08)
    ax = fig.add_subplot(gs_b[0, 0:4])
    np.random.seed(0)
    sel = ["building_coverage_exact","height_proxy_aw_mean_m","dsm_elevation_m_std","canopy_cover_gt2m_share","s2_ndvi_mean","road_density_exact_m_per_km2","lst_c_median_mean"]
    rows = [pd.DataFrame({"variable": f4.PROFILE_FIELDS[f], "value": S.scaled_1_99(a[f]), "class": a["class"]}) for f in sel]
    S.class_boxplot_panel(ax, pd.concat(rows, ignore_index=True), [f4.PROFILE_FIELDS[f] for f in sel], letter="B")
    S.finalize_panel_labels(fig)
    # crop window: integer-pixel band, shifted by a fractional pixel phase
    top_px = (1 - 0.650) * Hh * 300 - 90
    bot_px = (1 - 0.488) * Hh * 300 + 84
    hi = Hh - top_px / 300.0 + phase_px / 300.0   # inches from the figure bottom (y grows upward)
    lo = Hh - bot_px / 300.0 + phase_px / 300.0
    p = P.OUT / "_tmp_phase.png"
    fig.savefig(p, dpi=300, facecolor="white", bbox_inches=mpl.transforms.Bbox.from_extents(0, lo, W, hi))
    plt.close(fig)
    arr = np.array(Image.open(p).convert("RGB")); p.unlink()
    return arr

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv("data/03_processed/grid_250m_model_features_v8.csv")
    a = f4._analysis_subset(df)
    o = np.array(Image.open(P.ORIG / "fig_4_class_profiles.png").convert("RGB"))
    out = []
    for ph in [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]:
        n = render(ph, a)
        bh, w = n.shape[0], min(n.shape[1], o.shape[1])
        gray_o, gray_n = o.mean(axis=2), n.mean(axis=2)
        best = min(((np.abs(gray_o[dy:dy+bh, :w] - gray_n[:, :w]).mean(), dy) for dy in range(878, 900)))
        dy = best[1]
        d = np.abs(o[dy:dy+bh, :w].astype(int) - n[:, :w].astype(int)).max(axis=2)
        out.append({"phase_px": ph, "band_h": bh, "dy": dy, "MAE": P.mae_rgb(o[dy:dy+bh,:w], n[:,:w]),
                    "pct_gt64": float((d > 64).mean() * 100), "pct_gt16": float((d > 16).mean() * 100)})
        print(out[-1])
    (P.OUT / "fig4_phase_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
