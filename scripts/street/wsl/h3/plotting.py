"""Draft visualization routines for WP3.3 (H3: surface vs pedestrian heat stress).

Pre-analysis lock & rules:
- Draft figures only (final figures are a later WP).
- Every figure carries 'PARTIAL (n tiles)' (or 'SYNTHETIC ...') prominently.
- If synthetic, applies integrity watermark per Rule 6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.street.integrity.synthetic import watermark


def plot_lst_vs_m1_scatter(
    df: pd.DataFrame,
    day: str,
    label_str: str,
    out_path: Path,
    is_synthetic: bool = False,
) -> None:
    """Draft Figure 1: Scatter / hexbin of LST vs M1 with terciles and decoupled highlighting."""
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)

    valid = df[np.isfinite(df["lst_med"]) & np.isfinite(df["m1"])].copy()
    if len(valid) == 0:
        plt.close(fig)
        return

    # Base scatter / density
    if len(valid) > 2000:
        hb = ax.hexbin(valid["lst_med"], valid["m1"], gridsize=40, cmap="Blues", mincnt=1, alpha=0.6)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Segment count")
    else:
        ax.scatter(valid["lst_med"], valid["m1"], c="cornflowerblue", alpha=0.4, s=15, edgecolors="none", label="Segments")

    # Tercile lines if present
    terciles = valid.attrs.get("terciles")
    if terciles:
        ax.axvline(terciles["lst_q33"], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(terciles["lst_q67"], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(terciles["m1_q33"], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(terciles["m1_q67"], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    # Highlight decoupled
    dec1 = valid[valid.get("decoupled_hot_surf_cool_ped", False) == True]
    dec2 = valid[valid.get("decoupled_cool_surf_hot_ped", False) == True]

    if len(dec1) > 0:
        ax.scatter(dec1["lst_med"], dec1["m1"], c="#d95f02", s=22, alpha=0.7, edgecolors="none",
                   label=f"Decoupled: Hot Surf / Cool Ped (n={len(dec1)})")
    if len(dec2) > 0:
        ax.scatter(dec2["lst_med"], dec2["m1"], c="#7570b3", s=22, alpha=0.7, edgecolors="none",
                   label=f"Decoupled: Cool Surf / Hot Ped (n={len(dec2)})")

    # Highlight LCZ 2/3 (compact mid/lowrise)
    lcz23 = valid[valid["lcz_primary"].isin([2, 3, "2", "3"])]
    if len(lcz23) > 0:
        ax.scatter(lcz23["lst_med"], lcz23["m1"], facecolors="none", edgecolors="red", s=30, alpha=0.5,
                   linewidth=0.8, label=f"LCZ 2/3 Compact (n={len(lcz23)})")

    ax.set_xlabel("Landsat Summer LST (°C)")
    ax.set_ylabel("M1: Pedestrian UTCI Band 15 (°C)")
    ax.set_title(f"H3: Surface LST vs Pedestrian UTCI (M1) — {day}\n[{label_str}]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    if is_synthetic:
        watermark(fig)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_spearman_forest(
    spearman_df: pd.DataFrame,
    day: str,
    label_str: str,
    out_path: Path,
    is_synthetic: bool = False,
) -> None:
    """Draft Figure 2: Forest plot of Spearman rho with 95% block-bootstrap CIs across LCZs."""
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)

    df_plot = spearman_df.dropna(subset=["rho"]).copy()
    if len(df_plot) == 0:
        plt.close(fig)
        return

    # Separate metrics
    m1_df = df_plot[df_plot["metric_ped"].str.contains("M1")].copy()
    m3_df = df_plot[df_plot["metric_ped"].str.contains("M3")].copy()

    labels = m1_df["lcz_label"].tolist()
    y_pos = np.arange(len(labels))

    ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)

    # Plot M1
    if len(m1_df) > 0:
        err_m1_lo = m1_df["rho"] - m1_df["ci95_lower"]
        err_m1_hi = m1_df["ci95_upper"] - m1_df["rho"]
        err_m1 = np.vstack([err_m1_lo.clip(lower=0), err_m1_hi.clip(lower=0)])
        ax.errorbar(
            m1_df["rho"], y_pos - 0.15, xerr=err_m1, fmt="o", color="#e41a1c",
            label="rho(LST, M1 UTCI)", capsize=3, markersize=5
        )

    # Plot M3
    if len(m3_df) > 0:
        err_m3_lo = m3_df["rho"] - m3_df["ci95_lower"]
        err_m3_hi = m3_df["ci95_upper"] - m3_df["rho"]
        err_m3 = np.vstack([err_m3_lo.clip(lower=0), err_m3_hi.clip(lower=0)])
        ax.errorbar(
            m3_df["rho"], y_pos + 0.15, xerr=err_m3, fmt="s", color="#377eb8",
            label="rho(LST, M3 Tmrt)", capsize=3, markersize=5
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman Rank Correlation (rho) [95% Block-Bootstrap CI]")
    ax.set_title(f"H3: Spearman Correlation by LCZ Class — {day}\n[{label_str}]", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)

    if is_synthetic:
        watermark(fig)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_moran_bivariate_map(
    grid_gdf: gpd.GeoDataFrame,
    moran_summary: dict,
    day: str,
    label_str: str,
    out_path: Path,
    is_synthetic: bool = False,
) -> None:
    """Draft Figure 3: Bivariate Local Moran 250 m spatial cluster map."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    color_map = {
        "HH": "#e41a1c",               # High LST, High M1 (Red)
        "LL": "#377eb8",               # Low LST, Low M1 (Blue)
        "HL": "#ff7f00",               # High LST, Low M1 (Orange - decoupled)
        "LH": "#4daf4a",               # Low LST, High M1 (Green - decoupled)
        "Not Significant": "#d9d9d9",  # Light grey
    }

    # Plot base / not significant
    grid_gdf.plot(color="#f0f0f0", edgecolor="#bdbdbd", linewidth=0.2, ax=ax)

    if "cluster_label" in grid_gdf.columns:
        for ctype, color in color_map.items():
            sub = grid_gdf[grid_gdf["cluster_label"] == ctype]
            if len(sub) > 0:
                sub.plot(color=color, edgecolor="#525252", linewidth=0.3, ax=ax, label=f"{ctype} (n={len(sub)})")

    # Legend patches
    patches = [
        mpatches.Patch(color=color, label=f"{ctype} (n={moran_summary.get('cluster_counts', {}).get(ctype, 0)})")
        for ctype, color in color_map.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)

    glob_I = moran_summary.get("global_bivariate_moran_I", float("nan"))
    glob_p = moran_summary.get("global_bivariate_moran_p_sim", float("nan"))
    ax.set_title(
        f"H3: Bivariate Local Moran (LST vs M1) at 250 m — {day}\n"
        f"Global Bivariate Moran's I = {glob_I:.3f} (p = {glob_p:.3f}) [{label_str}]",
        fontsize=10, fontweight="bold",
    )
    ax.set_axis_off()

    if is_synthetic:
        watermark(fig)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_decoupled_shares(
    df_lcz_shares: pd.DataFrame,
    df_audit_shares: pd.DataFrame,
    day: str,
    label_str: str,
    out_path: Path,
    is_synthetic: bool = False,
) -> None:
    """Draft Figure 4: Decoupled segment shares across LCZ classes and audit classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    # Panel A: By LCZ
    lcz_sub = df_lcz_shares[df_lcz_shares["group"] != "ALL"].copy()
    if len(lcz_sub) > 0:
        y1 = np.arange(len(lcz_sub))
        ax1.barh(y1, lcz_sub["share_hot_surf_cool_ped"] * 100, color="#d95f02", label="Hot Surf / Cool Ped (T3/T1)", alpha=0.85)
        ax1.barh(y1, lcz_sub["share_cool_surf_hot_ped"] * 100, left=lcz_sub["share_hot_surf_cool_ped"] * 100,
                 color="#7570b3", label="Cool Surf / Hot Ped (T1/T3)", alpha=0.85)
        ax1.set_yticks(y1)
        ax1.set_yticklabels(lcz_sub["lcz_label"], fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Decoupled Segments Share (%)")
        ax1.set_title(f"A. Decoupled Shares by LCZ Class — {day}", fontsize=10, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Panel B: By Audit Class
    audit_sub = df_audit_shares[
        (~df_audit_shares["audit_label"].str.startswith("ALL"))
        & (df_audit_shares["label_quality_subset"] == "ALL")
    ].copy()
    if len(audit_sub) > 0:
        y2 = np.arange(len(audit_sub))
        ax2.barh(y2, audit_sub["share_hot_surf_cool_ped"] * 100, color="#d95f02", label="Hot Surf / Cool Ped (T3/T1)", alpha=0.85)
        ax2.barh(y2, audit_sub["share_cool_surf_hot_ped"] * 100, left=audit_sub["share_hot_surf_cool_ped"] * 100,
                 color="#7570b3", label="Cool Surf / Hot Ped (T1/T3)", alpha=0.85)
        ax2.set_yticks(y2)
        ax2.set_yticklabels(audit_sub["audit_label"], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel("Decoupled Segments Share (%)")
        ax2.set_title(f"B. Decoupled Shares by Manual Audit Class — {day}", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(f"H3 Decoupled Segment Shares [{label_str}]", fontsize=11, fontweight="bold")

    if is_synthetic:
        watermark(fig)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
