"""WP0.3 Step 3 & 4: Compare 02Geo3D comfort functions against pythermalcomfort reference.

Computes:
- UTCI: pythermalcomfort.models.utci vs plugin comfort.js utci
- PET: pythermalcomfort.models.pet_steady vs plugin comfort.js pet

Parameter Mapping Documentation:
1. UTCI:
   - Both implementations evaluate the Broede et al. (2012) 6th-order polynomial.
   - Reference: pythermalcomfort.models.utci(tdb, tr, v, rh)
   - Plugin: utci(tdb, tr, v, rh)
   - Validity range in pythermalcomfort: tdb in [-50, 50], (tr - tdb) in [-30, 70], v in [0.5, 17], pa in [0, 5] kPa.
2. PET:
   - Both implementations solve the Munich Energy-balance Model for Individuals (MEMI; Walther & Goestchel 2018).
   - In comfort.js:
     met = 80 W (metabolic activity heat in Watts, added to basal metabolism metab_male ~86.8 W).
     clo = 0.9.
     Internal person constants: age = 35, sex = 'male', weight = 75 kg, height = 1.8 m, posture = sitting (feff=0.725, hc=2.67+6.5*v^0.67), p_atm = 1013.25 hPa.
   - In pythermalcomfort.models.pet_steady:
     met parameter is in [met], multiplied internally by met_factor = 58.2 W/met to get Watts: _met = met * 58.2.
     Therefore, to match comfort.js met=80 W exactly, met = 80.0 / 58.2 (~1.37457 met).
     Person parameters set to: clo=0.9, age=35, sex='male', weight=75.0, height=1.8, position='sitting', p_atm=1013.25, wme=0.

Outputs:
- Console table of metrics (Rule 5: exact reproducible numbers)
- outputs/street/crosscheck/comfort_comparison_metrics.json
- outputs/street/crosscheck/comfort_comparison_summary.csv
- outputs/street/crosscheck/utci_pet_crosscheck.png (2 hexbin panels, 1:1 line)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pythermalcomfort.models as ptm


def get_pet_category(pet_val: float) -> str:
    """Classify PET into standardized thermal sensation categories (Matzarakis et al. 1999)."""
    if not np.isfinite(pet_val):
        return "Unknown"
    categories = [
        (4.0, "Extreme cold stress"),
        (8.0, "Strong cold stress"),
        (13.0, "Moderate cold stress"),
        (18.0, "Slight cold stress"),
        (23.0, "Comfortable"),
        (29.0, "Slight heat stress"),
        (35.0, "Moderate heat stress"),
        (41.0, "Strong heat stress"),
        (float("inf"), "Extreme heat stress"),
    ]
    for max_val, label in categories:
        if pet_val < max_val:
            return label
    return "Extreme heat stress"


def compute_metrics(
    y_plugin: np.ndarray,
    y_ref: np.ndarray,
    cat_plugin: list[str] | np.ndarray,
    cat_ref: list[str] | np.ndarray,
) -> dict[str, float]:
    """Compute statistical agreement metrics between plugin and reference."""
    mask = np.isfinite(y_plugin) & np.isfinite(y_ref)
    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_abs_diff": np.nan,
            "pearson_r": np.nan,
            "pct_within_0_1k": np.nan,
            "pct_within_0_5k": np.nan,
            "cat_agreement_pct": np.nan,
        }

    yp = y_plugin[mask]
    yr = y_ref[mask]
    diff = yp - yr
    abs_diff = np.abs(diff)

    bias = float(np.mean(diff))
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(abs_diff))

    if np.std(yp) > 1e-12 and np.std(yr) > 1e-12:
        r = float(np.corrcoef(yp, yr)[0, 1])
    else:
        r = 1.0

    p_01 = float(np.mean(abs_diff <= 0.1) * 100.0)
    p_05 = float(np.mean(abs_diff <= 0.5) * 100.0)

    # Category agreement
    cp = np.array([str(c).strip().lower() for c in np.array(cat_plugin)[mask]])
    cr = np.array([str(c).strip().lower() for c in np.array(cat_ref)[mask]])
    cat_agree = float(np.mean(cp == cr) * 100.0)

    return {
        "n": n_valid,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "max_abs_diff": max_abs,
        "pearson_r": r,
        "pct_within_0_1k": p_01,
        "pct_within_0_5k": p_05,
        "cat_agreement_pct": cat_agree,
    }


def main():
    csv_path = Path(
        "outputs/street/crosscheck/comfort_plugin_predictions_SYNTHETIC.csv"
    )
    out_dir = Path("outputs/street/crosscheck")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"Loaded {n_total} rows.")

    tdb = df["tdb"].values
    tr = df["tr"].values
    v = df["v"].values
    rh = df["rh"].values

    print("\nComputing pythermalcomfort UTCI reference values...")
    # Exact unlimited polynomial for full numerical evaluation
    res_utci_unlim = ptm.utci(
        tdb, tr, v, rh, limit_inputs=False, round_output=False
    )
    # Standard validity-checked UTCI
    res_utci_lim = ptm.utci(
        tdb, tr, v, rh, limit_inputs=True, round_output=False
    )

    df["utci_ref_unlim"] = res_utci_unlim.utci
    df["utci_ref_lim"] = res_utci_lim.utci
    df["utci_in_valid_range"] = ~np.isnan(res_utci_lim.utci)
    df["utci_category_ref"] = [
        c.title() for c in res_utci_unlim.stress_category
    ]

    print("Computing pythermalcomfort PET reference values (pet_steady)...")
    res_pet = ptm.pet_steady(
        tdb,
        tr,
        v,
        rh,
        met=80.0 / 58.2,
        clo=0.9,
        age=35,
        sex="male",
        weight=75.0,
        height=1.8,
        position="sitting",
        p_atm=1013.25,
        wme=0,
    )
    df["pet_ref"] = res_pet.pet
    df["pet_category_ref"] = [get_pet_category(p) for p in res_pet.pet]

    # Save augmented predictions
    augmented_csv = (
        out_dir / "comfort_crosscheck_full_evaluation_SYNTHETIC.csv"
    )
    df.to_csv(augmented_csv, index=False, encoding="utf-8")
    print(f"Saved augmented evaluation dataset to {augmented_csv}")

    # Subsets to evaluate
    subsets = {
        "All Samples (N=3000)": np.ones(n_total, dtype=bool),
        "Full-Domain LHS (N=2000)": df["sample_set"] == "full_domain_lhs",
        "Inside UTCI Validity Range (N=2865)": df["utci_in_valid_range"],
        "Dedicated Izmir Summer LHS (N=1000)": df["sample_set"]
        == "izmir_summer_lhs",
        "All Points in Izmir Summer Domain (N=1160)": df[
            "in_izmir_summer_domain"
        ],
    }

    summary_rows = []

    print("\n" + "=" * 115)
    print(
        f"{'Variable & Domain':<44} | {'N':>5} | {'Bias (K)':>9} | {'MAE (K)':>8} | {'RMSE (K)':>9} | {'MaxDiff(K)':>10} | {'r':>8} | {'<=0.1K%':>7} | {'<=0.5K%':>7} | {'CatAgre%':>8}"
    )
    print("-" * 115)

    all_metrics = {}

    for var_name, col_plugin, col_ref, cat_col_plugin, cat_col_ref in [
        (
            "UTCI",
            "utci_plugin",
            "utci_ref_unlim",
            "utci_category_plugin",
            "utci_category_ref",
        ),
        (
            "PET",
            "pet_plugin",
            "pet_ref",
            "pet_category_plugin",
            "pet_category_ref",
        ),
    ]:
        all_metrics[var_name] = {}
        for subset_name, mask in subsets.items():
            # If UTCI and inside validity range, we can also use col_ref = 'utci_ref_lim'
            m = compute_metrics(
                df.loc[mask, col_plugin].values,
                df.loc[mask, col_ref].values,
                df.loc[mask, cat_col_plugin].values,
                df.loc[mask, cat_col_ref].values,
            )
            all_metrics[var_name][subset_name] = m

            row_title = f"{var_name} [{subset_name}]"
            print(
                f"{row_title:<44} | {m['n']:>5} | {m['bias']:>9.4f} | {m['mae']:>8.4f} | {m['rmse']:>9.4f} | {m['max_abs_diff']:>10.4f} | {m['pearson_r']:>8.6f} | {m['pct_within_0_1k']:>7.2f} | {m['pct_within_0_5k']:>7.2f} | {m['cat_agreement_pct']:>8.2f}"
            )

            summary_rows.append(
                {
                    "Variable": var_name,
                    "Subset": subset_name,
                    "N": m["n"],
                    "Bias_K": round(m["bias"], 6),
                    "MAE_K": round(m["mae"], 6),
                    "RMSE_K": round(m["rmse"], 6),
                    "Max_Abs_Diff_K": round(m["max_abs_diff"], 6),
                    "Pearson_r": round(m["pearson_r"], 8),
                    "Pct_Within_0_1K": round(m["pct_within_0_1k"], 2),
                    "Pct_Within_0_5K": round(m["pct_within_0_5k"], 2),
                    "Category_Agreement_Pct": round(
                        m["cat_agreement_pct"], 2
                    ),
                }
            )

    print("=" * 115)

    # Save JSON metrics
    metrics_json = out_dir / "comfort_comparison_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics JSON to {metrics_json}")

    # Save summary table CSV
    summary_csv = out_dir / "comfort_comparison_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"Saved summary CSV to {summary_csv}")

    # Plot 2-panel hexbin figure
    print("\nGenerating 2-panel hexbin comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=300)

    # Panel A: UTCI
    ax1 = axes[0]
    x_utci = df["utci_ref_unlim"].values
    y_utci = df["utci_plugin"].values
    m_utci = all_metrics["UTCI"]["All Samples (N=3000)"]

    hb1 = ax1.hexbin(
        x_utci,
        y_utci,
        gridsize=45,
        cmap="viridis",
        mincnt=1,
        bins="log",
        edgecolors="none",
    )
    cb1 = fig.colorbar(hb1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("Log10(Count)", fontsize=10)

    lims_utci = [
        min(np.min(x_utci), np.min(y_utci)) - 2,
        max(np.max(x_utci), np.max(y_utci)) + 2,
    ]
    ax1.plot(
        lims_utci,
        lims_utci,
        "r--",
        linewidth=1.2,
        label="1:1 Reference Line (y = x)",
    )
    ax1.set_xlim(lims_utci)
    ax1.set_ylim(lims_utci)
    ax1.set_title(
        "(a) UTCI: 02Geo3D comfort.js vs pythermalcomfort",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.set_xlabel("pythermalcomfort UTCI [°C]", fontsize=10.5)
    ax1.set_ylabel("02Geo3D comfort.js UTCI [°C]", fontsize=10.5)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    stats_text_utci = (
        f"N = {m_utci['n']:,}\n"
        f"Bias = {m_utci['bias']:+.2e} K\n"
        f"MAE = {m_utci['mae']:.2e} K\n"
        f"RMSE = {m_utci['rmse']:.2e} K\n"
        f"Max |Diff| = {m_utci['max_abs_diff']:.2e} K\n"
        f"Pearson r = {m_utci['pearson_r']:.6f}\n"
        f"|Diff| ≤ 0.1 K: {m_utci['pct_within_0_1k']:.1f}%\n"
        f"Stress Cat. Agreement: {m_utci['cat_agreement_pct']:.1f}%"
    )
    ax1.text(
        0.96,
        0.05,
        stats_text_utci,
        transform=ax1.transAxes,
        fontsize=9.0,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    # Panel B: PET
    ax2 = axes[1]
    x_pet = df["pet_ref"].values
    y_pet = df["pet_plugin"].values
    m_pet = all_metrics["PET"]["All Samples (N=3000)"]
    m_pet_valid = all_metrics["PET"]["Inside UTCI Validity Range (N=2865)"]

    hb2 = ax2.hexbin(
        x_pet,
        y_pet,
        gridsize=45,
        cmap="plasma",
        mincnt=1,
        bins="log",
        edgecolors="none",
    )
    cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Log10(Count)", fontsize=10)

    lims_pet = [
        min(np.min(x_pet), np.min(y_pet)) - 2,
        max(np.max(x_pet), np.max(y_pet)) + 2,
    ]
    ax2.plot(
        lims_pet,
        lims_pet,
        "r--",
        linewidth=1.2,
        label="1:1 Reference Line (y = x)",
    )
    ax2.set_xlim(lims_pet)
    ax2.set_ylim(lims_pet)
    ax2.set_title(
        "(b) PET: 02Geo3D comfort.js vs pythermalcomfort (MEMI)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.set_xlabel("pythermalcomfort PET [°C]", fontsize=10.5)
    ax2.set_ylabel("02Geo3D comfort.js PET [°C]", fontsize=10.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    stats_text_pet = (
        f"All Samples (N={m_pet['n']:,}):\n"
        f"  Bias = {m_pet['bias']:+.3f} K | MAE = {m_pet['mae']:.3f} K\n"
        f"  RMSE = {m_pet['rmse']:.3f} K | Max |Diff| = {m_pet['max_abs_diff']:.2f} K\n"
        f"  r = {m_pet['pearson_r']:.6f} | |Diff| ≤ 0.5 K: {m_pet['pct_within_0_5k']:.1f}%\n"
        f"Inside Validity Range (N={m_pet_valid['n']:,}):\n"
        f"  Bias = {m_pet_valid['bias']:+.3f} K | MAE = {m_pet_valid['mae']:.3f} K\n"
        f"  RMSE = {m_pet_valid['rmse']:.3f} K | Max |Diff| = {m_pet_valid['max_abs_diff']:.2f} K\n"
        f"  r = {m_pet_valid['pearson_r']:.6f} | |Diff| ≤ 0.5 K: {m_pet_valid['pct_within_0_5k']:.1f}%\n"
        f"Stress Cat. Agreement: {m_pet['cat_agreement_pct']:.1f}%"
    )
    ax2.text(
        0.96,
        0.05,
        stats_text_pet,
        transform=ax2.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    plt.tight_layout()
    fig_path = out_dir / "utci_pet_crosscheck.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-check figure to {fig_path}")


if __name__ == "__main__":
    main()
