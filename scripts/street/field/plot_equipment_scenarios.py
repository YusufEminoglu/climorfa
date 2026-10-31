"""
CLIMORFA Street — Equipment Scenarios Visualization (WP0.5-A3).
Authoritative specification: docs/climorfa_street/wp/WP0.5A3_equipment.md

Generates publication-quality watermarked diagnostic figures comparing:
1. 95% Confidence Interval Half-Width (Overall and Max-Type) vs 1.5 K threshold across scenarios S0 to S4.
2. Statistical Power for detecting Sunlit vs. Shaded exposure bias contrast vs 80% threshold.
3. Measurement vs. Structural Variance Decomposition in compact canyons (Type A/B).
4. Operational Trade-off: Stop Duration (equilibration) vs. Feasible Visits and Stops.

Output:
outputs/street/field/SYNTHETIC_fig_equipment_scenarios.png
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def add_synthetic_watermark(ax, text="SYNTHETIC PLACEHOLDER — NOT FIELD DATA", fontsize=13, alpha=0.18):
    """Adds a prominent semi-transparent diagonal red watermark across the axis."""
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color="#d62728",
        alpha=alpha,
        ha="center",
        va="center",
        rotation=28,
        weight="heavy",
        zorder=100,
        bbox=dict(boxstyle="square,pad=0.3", edgecolor="none", facecolor="none"),
    )


def plot_equipment_scenarios(
    csv_path: str = "outputs/street/field/SYNTHETIC_equipment_scenarios.csv",
    out_png: str = "outputs/street/field/SYNTHETIC_fig_equipment_scenarios.png",
):
    """
    Renders 4-panel comparison of equipment scenarios S0–S4.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    sns.set_theme(style="ticks", font_scale=1.05)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    palette = {
        "S0": "#7f7f7f",  # Gray
        "S1": "#1f77b4",  # Blue
        "S2": "#2ca02c",  # Green
        "S3": "#ff7f0e",  # Orange
        "S4": "#9467bd",  # Purple
    }

    scenarios_order = ["S0", "S1", "S2", "S3", "S4"]
    sc_labels = {
        "S0": "S0: Kestrel 5400\n(Relative Co-loc)",
        "S1": "S1: Kestrel 5400\n(+ Isothermal Cal)",
        "S2": "S2: 38mm Grey\n(+ Class-A Pt100)",
        "S3": "S3: 150mm Black\n(+ Class-A Pt100)",
        "S4": "S4: 6-Dir Reference\n(Net Radiometer)",
    }

    # -------------------------------------------------------------
    # Panel A: 95% CI Half-Width (Planned 16 pts x 3 v/d vs 1.5 K target)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    df_planned = df[df["design_name"] == "Planned (16 pts x 3 v/d x 2 d)"].set_index("scenario_id").reindex(scenarios_order)

    x = np.arange(len(scenarios_order))
    width = 0.35

    rects1 = ax_a.bar(x - width / 2, df_planned["rmse_ci_half_width_overall_k"], width, label="Overall Campaign CI Half-Width", color="#3b528b", alpha=0.9, edgecolor="black", lw=1)
    rects2 = ax_a.bar(x + width / 2, df_planned["rmse_ci_half_width_max_type_k"], width, label="Max-Type CI Half-Width", color="#5dc863", alpha=0.9, edgecolor="black", lw=1)

    # 1.5 K Target line
    ax_a.axhline(1.50, color="#d62728", linestyle="--", lw=2.2, label="§4 Target Threshold (≤ 1.5 K)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([sc_labels[s] for s in scenarios_order], fontsize=9.5)
    ax_a.set_ylabel("95% Confidence Interval Half-Width (K)", fontweight="bold")
    ax_a.set_title("(A) RMSE 95% Confidence Interval Half-Width (Planned Design: 16 pts × 3 visits × 2 days)", fontsize=11.5, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_a.set_ylim(0, 6.5)

    # Value labels
    for r in rects1:
        h = r.get_height()
        status = "PASS" if h <= 1.50 else "FAIL"
        ax_a.text(r.get_x() + r.get_width() / 2.0, h + 0.10, f"±{h:.2f}K\n({status})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#3b528b")
    for r in rects2:
        h = r.get_height()
        status = "PASS" if h <= 1.50 else "FAIL"
        ax_a.text(r.get_x() + r.get_width() / 2.0, h + 0.10, f"±{h:.2f}K\n({status})", ha="center", va="bottom", fontsize=8.0, color="#2c7233")

    add_synthetic_watermark(ax_a, fontsize=12)

    # -------------------------------------------------------------
    # Panel B: Statistical Power for Sunlit vs Shaded Contrast
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    df_extended = df[df["design_name"] == "Extended (24 pts x 4 v/d x 2 d)"].set_index("scenario_id").reindex(scenarios_order)

    pwr_planned = df_planned["power_exposure_contrast_pct"]
    pwr_extended = df_extended["power_exposure_contrast_pct"]

    rects_p1 = ax_b.bar(x - width / 2, pwr_planned, width, label="Planned Design (16 pts × 3 v/d, 96 stops)", color="#21918c", alpha=0.9, edgecolor="black", lw=1)
    rects_p2 = ax_b.bar(x + width / 2, pwr_extended, width, label="Extended Design (24 pts × 4 v/d, 192 stops)", color="#fde725", alpha=0.9, edgecolor="black", lw=1)

    # Check for S3 feasible
    df_s3_f = df[(df["scenario_id"] == "S3") & (df["design_name"].str.contains("Feasible-16"))]
    if len(df_s3_f) > 0:
        pwr_s3_f = df_s3_f["power_exposure_contrast_pct"].values[0]
        # Overlay a marker or hatch for S3 feasible
        ax_b.scatter([3 - width / 2], [pwr_s3_f], color="#d62728", s=140, zorder=10, marker="X", label=f"S3 Feasible (1 v/d, 32 stops): {pwr_s3_f:.1f}%")
        ax_b.annotate(
            f"S3 Feasible\n{pwr_s3_f:.1f}% (FAIL)",
            xy=(3 - width / 2, pwr_s3_f),
            xytext=(3 - width / 2 - 0.35, pwr_s3_f - 18),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1, headwidth=5),
            fontsize=9.0,
            fontweight="bold",
            color="#d62728",
        )

    ax_b.axhline(80.0, color="#d62728", linestyle="--", lw=2.2, label="Standard Power Threshold (80%)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([sc_labels[s] for s in scenarios_order], fontsize=9.5)
    ax_b.set_ylabel("Statistical Power (α = 0.05, %)", fontweight="bold")
    ax_b.set_title("(B) Statistical Power to Detect Exposure Bias Contrast (Sunlit vs. Shaded)", fontsize=11.5, fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="lower right", frameon=True, fontsize=9.0)
    ax_b.set_ylim(40, 105)

    for r in rects_p1:
        h = r.get_height()
        ax_b.text(r.get_x() + r.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects_p2:
        h = r.get_height()
        ax_b.text(r.get_x() + r.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    add_synthetic_watermark(ax_b, fontsize=12)

    # -------------------------------------------------------------
    # Panel C: Measurement vs. Structural Variance Decomposition in Canyons (Type A)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    meas_sq_A = [df_planned.loc[s, "sigma_meas_type_A_k"] ** 2 for s in scenarios_order]
    struct_sq_A = [df_planned.loc[s, "sigma_struct_type_A_k"] ** 2 for s in scenarios_order]

    target_sq_A = 4.90 ** 2  # 24.01 K^2

    bar_width = 0.55
    b_meas = ax_c.bar(x, meas_sq_A, bar_width, label="Measurement Variance (σ_meas²: instrument noise + offset)", color="#e41a1c", alpha=0.85, edgecolor="black", lw=1)
    b_struct = ax_c.bar(x, struct_sq_A, bar_width, bottom=meas_sq_A, label="Model Structural Remainder (σ_struct²)", color="#377eb8", alpha=0.85, edgecolor="black", lw=1)

    ax_c.axhline(target_sq_A, color="black", linestyle="--", lw=2.0, label="Literature Composite Target (Target² = 24.01 K²)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([sc_labels[s] for s in scenarios_order], fontsize=9.5)
    ax_c.set_ylabel("Variance Contribution (K²)", fontweight="bold")
    ax_c.set_title("(C) Error Variance Decomposition in Compact Canyons (Type A, Target RMSE = 4.90 °C)", fontsize=11.5, fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_c.set_ylim(0, 36)

    # Annotate S0 dominance vs S1-S4 model separation
    ax_c.annotate(
        "S0: Instrument Error Dominance\n(σ_meas² > Target² -> σ_struct = 0 K)\nCannot evaluate model!",
        xy=(0, meas_sq_A[0]),
        xytext=(0.15, 30.5),
        arrowprops=dict(facecolor="#e41a1c", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        color="#e41a1c",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffeeee", ec="#e41a1c", lw=1),
    )

    ax_c.annotate(
        "S2: Clean Model Separation\nσ_meas = 0.77 K (3% var)\nσ_struct = 4.84 K (97% var)",
        xy=(2, meas_sq_A[2] + struct_sq_A[2]),
        xytext=(1.8, 12.0),
        arrowprops=dict(facecolor="#377eb8", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        color="#377eb8",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec="#377eb8", lw=1),
    )

    add_synthetic_watermark(ax_c, fontsize=12)

    # -------------------------------------------------------------
    # Panel D: Operational Feasibility (Equilibration vs Daily Sampling Capacity)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]

    stop_durations = [10, 10, 10, 25, 10]  # minutes
    feasible_stops_16pts = [96, 96, 96, 32, 96]  # total stops over 2 days (3 vs 1 v/d)
    feasible_vpd_16pts = [3, 3, 3, 1, 3]

    color_list = ["#4daf4a" if s != "S3" else "#e41a1c" for s in scenarios_order]
    rects_d = ax_d.bar(x, feasible_stops_16pts, bar_width, color=color_list, alpha=0.85, edgecolor="black", lw=1)

    ax_d.axhline(96, color="#4daf4a", linestyle=":", lw=1.8, label="Planned Sampling Target (96 stops = 3 visits/day)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([sc_labels[s] for s in scenarios_order], fontsize=9.5)
    ax_d.set_ylabel("Feasible Campaign Stops in 2 Days (10:00–16:00)", fontweight="bold")
    ax_d.set_title("(D) Operational Feasibility: Sensor Time Constant vs. Daily Sampling Stops", fontsize=11.5, fontweight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.set_ylim(0, 120)

    for i, r in enumerate(rects_d):
        h = r.get_height()
        dur = stop_durations[i]
        vpd = feasible_vpd_16pts[i]
        txt = f"{h} stops\n({vpd} v/d, {dur}m stop)"
        col = "#2c7233" if h >= 96 else "#d62728"
        ax_d.text(r.get_x() + r.get_width() / 2.0, h + 2.0, txt, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=col)

    ax_d.annotate(
        "S3 Severe Operational Bottleneck:\n20 min equilibration (ISO 7726)\n-> Only 1 visit/day fits 10:00-16:00\nLoss of 67% sample size!",
        xy=(3, 32),
        xytext=(2.6, 75),
        arrowprops=dict(facecolor="#e41a1c", shrink=0.08, width=1.2, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        color="#e41a1c",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffeeee", ec="#e41a1c", lw=1),
    )

    ax_d.legend(loc="upper right", frameon=True, fontsize=9.5)
    add_synthetic_watermark(ax_d, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved watermarked equipment scenario figure to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Equipment Scenarios")
    parser.add_argument("--csv", default="outputs/street/field/SYNTHETIC_equipment_scenarios.csv")
    parser.add_argument("--out", default="outputs/street/field/SYNTHETIC_fig_equipment_scenarios.png")
    args = parser.parse_args()

    plot_equipment_scenarios(csv_path=args.csv, out_png=args.out)
