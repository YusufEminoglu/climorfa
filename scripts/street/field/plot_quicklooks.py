"""
CLIMORFA Street — Quicklook Figures for WP0.5 (v1.1).
Generates publication-quality diagnostic quicklook figures with prominent synthetic watermarking:
1. SYNTHETIC_fig_model_vs_obs_scatter.png:
   Multi-panel scatter plot of Observed vs Modeled Tmrt and UTCI, broken down by morphology type and exposure,
   with 1:1 line, linear regression, and key metrics annotations.
2. SYNTHETIC_fig_power_analysis.png:
   Power analysis curves showing RMSE 95% CI half-width and exposure bias detection power
   across experimental designs, comparing with vs. without co-location correction and highlighting planned campaign.
3. SYNTHETIC_fig_globe_amplification.png:
   ISO 7726 Tmrt amplification factor (dTmrt / dTg) vs. wind speed for 25.4 mm (1-inch Kestrel), 38 mm, and 150 mm globes.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scripts.street.integrity.synthetic import watermark


def plot_scatter(
    stops_csv: str = r"data/street/field/SYNTHETIC_processed_stops.csv",
    out_png: str = r"outputs/street/field/SYNTHETIC_fig_model_vs_obs_scatter.png",
):
    df = pd.read_csv(stops_csv)

    sns.set_theme(style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    type_colors = {
        "A": "#1f77b4",  # Blue (canyon E-W)
        "B": "#ff7f0e",  # Orange (canyon N-S)
        "C": "#2ca02c",  # Green (trees)
        "D": "#d62728",  # Red (open space)
    }
    exp_markers = {
        "sunlit": "o",
        "shaded": "s",
    }

    # ---------------- Panel 1: Tmrt Scatter ----------------
    ax1 = axes[0]
    min_tmrt = min(df["tmrt_obs"].min(), df["tmrt_model"].min()) - 3
    max_tmrt = max(df["tmrt_obs"].max(), df["tmrt_model"].max()) + 3

    ax1.plot([min_tmrt, max_tmrt], [min_tmrt, max_tmrt], "k--", lw=1.5, alpha=0.7, label="1:1 Line")

    for mtype in sorted(df["morphology_type"].unique()):
        for exp in ["sunlit", "shaded"]:
            sub = df[(df["morphology_type"] == mtype) & (df["exposure_actual"] == exp)]
            if len(sub) == 0:
                continue
            ax1.scatter(
                sub["tmrt_model"],
                sub["tmrt_obs"],
                color=type_colors[mtype],
                marker=exp_markers[exp],
                s=65,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.5,
                label=f"Type {mtype} ({exp})",
            )

    rmse_tmrt = np.sqrt(np.mean((df["tmrt_model"] - df["tmrt_obs"]) ** 2))
    bias_tmrt = np.mean(df["tmrt_model"] - df["tmrt_obs"])
    r_tmrt = np.corrcoef(df["tmrt_model"], df["tmrt_obs"])[0, 1]

    text_tmrt = (
        f"Overall Tmrt (n = {len(df)}):\n"
        f"RMSE = {rmse_tmrt:.2f} °C\n"
        f"Bias (Mod - Obs) = {bias_tmrt:.2f} °C\n"
        f"Pearson r = {r_tmrt:.3f}"
    )
    ax1.text(
        0.05, 0.95, text_tmrt,
        transform=ax1.transAxes,
        fontsize=10.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    ax1.set_xlabel("SOLWEIG-GPU Modeled $T_{mrt}$ (°C)", weight="bold")
    ax1.set_ylabel("Observed $T_{mrt}$ (ISO 7726) (°C)", weight="bold")
    ax1.set_title("Mean Radiant Temperature ($T_{mrt}$)", weight="bold", pad=10)
    ax1.set_xlim(min_tmrt, max_tmrt)
    ax1.set_ylim(min_tmrt, max_tmrt)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ---------------- Panel 2: UTCI Scatter ----------------
    ax2 = axes[1]
    df_clean_u = df.dropna(subset=["utci_obs", "utci_model"])
    min_utci = min(df_clean_u["utci_obs"].min(), df_clean_u["utci_model"].min()) - 2
    max_utci = max(df_clean_u["utci_obs"].max(), df_clean_u["utci_model"].max()) + 2

    ax2.plot([min_utci, max_utci], [min_utci, max_utci], "k--", lw=1.5, alpha=0.7, label="1:1 Line")

    for mtype in sorted(df_clean_u["morphology_type"].unique()):
        for exp in ["sunlit", "shaded"]:
            sub = df_clean_u[(df_clean_u["morphology_type"] == mtype) & (df_clean_u["exposure_actual"] == exp)]
            if len(sub) == 0:
                continue
            ax2.scatter(
                sub["utci_model"],
                sub["utci_obs"],
                color=type_colors[mtype],
                marker=exp_markers[exp],
                s=65,
                alpha=0.85,
                edgecolors="k",
                linewidths=0.5,
            )

    rmse_utci = np.sqrt(np.mean((df_clean_u["utci_model"] - df_clean_u["utci_obs"]) ** 2))
    bias_utci = np.mean(df_clean_u["utci_model"] - df_clean_u["utci_obs"])
    r_utci = np.corrcoef(df_clean_u["utci_model"], df_clean_u["utci_obs"])[0, 1]

    text_utci = (
        f"Overall UTCI (n = {len(df_clean_u)}):\n"
        f"RMSE = {rmse_utci:.2f} °C\n"
        f"Bias (Mod - Obs) = {bias_utci:.2f} °C\n"
        f"Pearson r = {r_utci:.3f}"
    )
    ax2.text(
        0.05, 0.95, text_utci,
        transform=ax2.transAxes,
        fontsize=10.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    ax2.set_xlabel("SOLWEIG-GPU Modeled UTCI (°C)", weight="bold")
    ax2.set_ylabel("Observed UTCI (°C)", weight="bold")
    ax2.set_title("Universal Thermal Climate Index (UTCI)", weight="bold", pad=10)
    ax2.set_xlim(min_utci, max_utci)
    ax2.set_ylim(min_utci, max_utci)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Common legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Apply synthetic watermark
    watermark(fig)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot quicklook: {out_png}")


def plot_power_analysis(
    summary_csv: str = r"outputs/street/field/SYNTHETIC_power_analysis_summary.csv",
    out_png: str = r"outputs/street/field/SYNTHETIC_fig_power_analysis.png",
):
    df = pd.read_csv(summary_csv)

    sns.set_theme(style="ticks", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.0))

    # ---------------- Panel A: CI Half-width vs Total Stops ----------------
    ax1 = axes[0]
    # Filter with vs without co-location correction
    df_coloc = df[df["with_coloc_correction"] == True]
    df_nocoloc = df[df["with_coloc_correction"] == False]

    palette_coloc = {1: "#1f77b4", 2: "#d62728"}
    palette_nocoloc = {1: "#a6cee3", 2: "#fb9a99"}

    for d, grp in df_coloc.groupby("n_days"):
        grp = grp.sort_values("total_stops")
        ax1.plot(
            grp["total_stops"],
            grp["overall_rmse_ci_half_width"],
            marker="o",
            lw=2.2,
            color=palette_coloc[d],
            label=f"{d} Day{'s' if d > 1 else ''} (with Co-location)",
        )

    for d, grp in df_nocoloc.groupby("n_days"):
        grp = grp.sort_values("total_stops")
        ax1.plot(
            grp["total_stops"],
            grp["overall_rmse_ci_half_width"],
            marker="^",
            lw=1.5,
            linestyle="--",
            color=palette_nocoloc[d],
            label=f"{d} Day{'s' if d > 1 else ''} (Uncorrected)",
        )

    # Target threshold <= 1.5 C
    ax1.axhline(1.50, color="black", linestyle=":", lw=1.6, label=r"Target Precision ($\leq$1.50 °C)")

    # Highlight planned campaign design (96 stops, 2 days, 16 points, with co-location)
    planned = df_coloc[(df_coloc["n_points"] == 16) & (df_coloc["visits_per_day"] == 3) & (df_coloc["n_days"] == 2)].iloc[0]
    ax1.scatter(
        [planned["total_stops"]],
        [planned["overall_rmse_ci_half_width"]],
        color="gold",
        edgecolors="black",
        s=150,
        zorder=5,
        label=f"Planned Design ({int(planned['n_points'])} pts, {int(planned['visits_per_day'])} v/d, {int(planned['n_days'])} d)",
    )
    ax1.annotate(
        f"Planned (96 stops)\nCI HW = {planned['overall_rmse_ci_half_width']:.2f} °C",
        xy=(planned["total_stops"], planned["overall_rmse_ci_half_width"]),
        xytext=(planned["total_stops"] + 12, planned["overall_rmse_ci_half_width"] + 0.35),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
    )

    ax1.set_xlabel("Total Campaign Point-Visits (Stops)", weight="bold")
    ax1.set_ylabel("Overall RMSE 95% CI Half-Width (°C)", weight="bold")
    ax1.set_title("RMSE Estimation Precision vs Campaign Scale", weight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9.0, frameon=True)

    # ---------------- Panel B: Detection Power vs Total Stops ----------------
    ax2 = axes[1]

    for d, grp in df_coloc.groupby("n_days"):
        grp = grp.sort_values("total_stops")
        ax2.plot(
            grp["total_stops"],
            grp["power_bias_detection"] * 100.0,
            marker="s",
            lw=2.2,
            color=palette_coloc[d],
            label=f"{d} Day{'s' if d > 1 else ''} (with Co-location)",
        )

    for d, grp in df_nocoloc.groupby("n_days"):
        grp = grp.sort_values("total_stops")
        ax2.plot(
            grp["total_stops"],
            grp["power_bias_detection"] * 100.0,
            marker="v",
            lw=1.5,
            linestyle="--",
            color=palette_nocoloc[d],
            label=f"{d} Day{'s' if d > 1 else ''} (Uncorrected)",
        )

    ax2.axhline(80.0, color="black", linestyle=":", lw=1.6, label="Standard 80% Power")

    ax2.scatter(
        [planned["total_stops"]],
        [planned["power_bias_detection"] * 100.0],
        color="gold",
        edgecolors="black",
        s=150,
        zorder=5,
    )
    ax2.annotate(
        f"Power = {planned['power_bias_detection'] * 100:.1f}%\n(planned design)",
        xy=(planned["total_stops"], planned["power_bias_detection"] * 100.0),
        xytext=(planned["total_stops"] - 50, planned["power_bias_detection"] * 100.0 - 15),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
    )

    ax2.set_xlabel("Total Campaign Point-Visits (Stops)", weight="bold")
    ax2.set_ylabel("Exposure Bias Detection Power (%)", weight="bold")
    ax2.set_title("Statistical Power to Detect Sunlit/Shaded Bias Contrast", weight="bold", pad=10)
    ax2.set_ylim(25, 102)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9.0, frameon=True)

    plt.tight_layout()

    # Apply synthetic watermark
    watermark(fig)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved power analysis quicklook: {out_png}")


def plot_globe_amplification(
    out_png: str = r"outputs/street/field/SYNTHETIC_fig_globe_amplification.png",
):
    """
    Panel of ISO 7726 Tmrt amplification factor (dTmrt / dTg) vs wind speed
    for 25.4 mm (1-inch Kestrel 5400), 38 mm (table tennis / grey globe), and 150 mm (standard copper globe).
    Authoritative specification: docs/climorfa_street/field_error_model.md §3.5.
    """
    sns.set_theme(style="ticks", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 6.0))

    sigma = 5.670374419e-8
    eps = 0.95
    t_k = 295.15  # ~22 °C representative ambient temperature

    v_grid = np.linspace(0.1, 5.0, 200)

    globe_configs = [
        {"d_mm": 25.4, "label": "1-inch / 25.4 mm (Kestrel 5400 HST)", "color": "#d62728", "lw": 2.5, "ls": "-"},
        {"d_mm": 38.0, "label": "38 mm (Thorsson et al. 2007)", "color": "#2ca02c", "lw": 2.2, "ls": "--"},
        {"d_mm": 150.0, "label": "150 mm (ISO 7726 Standard Globe)", "color": "#1f77b4", "lw": 2.2, "ls": "-."},
    ]

    for g in globe_configs:
        d_m = g["d_mm"] / 1000.0
        h_c = 6.3 * (v_grid ** 0.6) / (d_m ** 0.4)
        amp = 1.0 + h_c / (4.0 * eps * sigma * (t_k ** 3))
        ax.plot(v_grid, amp, label=g["label"], color=g["color"], lw=g["lw"], linestyle=g["ls"])

    # Annotate key literature reference points (§3.5)
    # 25.4 mm at 2.6 m/s (mean campaign wind): dTmrt/dTg ~ 9.8
    h_c_26 = 6.3 * (2.6 ** 0.6) / ((0.0254) ** 0.4)
    amp_26 = 1.0 + h_c_26 / (4.0 * eps * sigma * (t_k ** 3))
    ax.scatter([2.6], [amp_26], color="#d62728", s=80, zorder=5)
    ax.annotate(
        f"25.4 mm @ 2.6 m/s\n∂Tmrt/∂Tg = {amp_26:.1f}x\n(WP0.5 root cause)",
        xy=(2.6, amp_26),
        xytext=(2.6 + 0.3, amp_26 - 1.2),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
    )

    # 150 mm at 2.0 m/s: ~4.7
    h_c_150_20 = 6.3 * (2.0 ** 0.6) / ((0.150) ** 0.4)
    amp_150_20 = 1.0 + h_c_150_20 / (4.0 * eps * sigma * (t_k ** 3))
    ax.scatter([2.0], [amp_150_20], color="#1f77b4", s=80, zorder=5)
    ax.annotate(
        f"150 mm @ 2.0 m/s\n∂Tmrt/∂Tg = {amp_150_20:.1f}x",
        xy=(2.0, amp_150_20),
        xytext=(2.0 + 0.3, amp_150_20 - 1.2),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1, headwidth=6),
        fontsize=9.5,
        fontweight="bold",
        color="#1f77b4",
    )

    ax.set_xlabel("Wind Speed at Sensor Height ($v$, m/s)", weight="bold")
    ax.set_ylabel(r"Globe $T_{mrt}$ Error Amplification Factor $\left(\frac{\partial T_{mrt}}{\partial T_g}\right)$", weight="bold")
    ax.set_title("ISO 7726 $T_{mrt}$ Sensitivity to Globe Temperature Error vs. Wind Speed", weight="bold", pad=12)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(1.0, 15.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)

    plt.tight_layout()

    # Apply synthetic watermark
    watermark(fig)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved globe amplification quicklook: {out_png}")


def main():
    plot_scatter()
    plot_power_analysis()
    plot_globe_amplification()
    print("All diagnostic quicklook figures generated successfully!")


if __name__ == "__main__":
    main()
