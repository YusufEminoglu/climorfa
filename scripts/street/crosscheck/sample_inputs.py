"""Generate Latin Hypercube input samples for thermal comfort cross-checking.

WP0.3 Step 1:
Draws N=2000 input tuples with fixed seed (20261029) via Latin Hypercube Sampling:
- Ta: 0..45 C
- Tmrt - Ta: -5..+30 K
- v10: 0.5..12 m/s
- RH: 10..90 %

Also generates a dedicated Izmir summer Latin Hypercube subset:
- Ta: 28..42 C
- Tmrt - Ta: 0..30 K
- v10: 0.5..8 m/s
- RH: 30..70 %

All synthetic data carries is_synthetic=True and 'SYNTHETIC' in the filename per _RULES.md Rule 6.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc


def generate_samples(
    n_full: int = 2000,
    n_izmir: int = 1000,
    seed: int = 20261029,
    out_dir: Path = Path("outputs/street/crosscheck"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full domain LHS bounds
    # [Ta, delta_Tmrt, v, RH]
    full_lower = np.array([0.0, -5.0, 0.5, 10.0])
    full_upper = np.array([45.0, 30.0, 12.0, 90.0])

    sampler_full = qmc.LatinHypercube(d=4, seed=seed)
    lhs_full = sampler_full.random(n=n_full)
    scaled_full = qmc.scale(lhs_full, full_lower, full_upper)

    df_full = pd.DataFrame(
        {
            "sample_id": np.arange(n_full),
            "sample_set": "full_domain_lhs",
            "is_synthetic": True,
            "tdb": np.round(scaled_full[:, 0], 4),
            "delta_tmrt": np.round(scaled_full[:, 1], 4),
            "tr": np.round(scaled_full[:, 0] + scaled_full[:, 1], 4),
            "v": np.round(scaled_full[:, 2], 4),
            "rh": np.round(scaled_full[:, 3], 4),
        }
    )

    # Check which samples in df_full fall inside the Izmir summer bounds
    izmir_mask_full = (
        (df_full["tdb"] >= 28.0)
        & (df_full["tdb"] <= 42.0)
        & (df_full["delta_tmrt"] >= 0.0)
        & (df_full["delta_tmrt"] <= 30.0)
        & (df_full["v"] >= 0.5)
        & (df_full["v"] <= 8.0)
        & (df_full["rh"] >= 30.0)
        & (df_full["rh"] <= 70.0)
    )
    df_full["in_izmir_summer_domain"] = izmir_mask_full

    # Dedicated Izmir summer domain LHS bounds
    izmir_lower = np.array([28.0, 0.0, 0.5, 30.0])
    izmir_upper = np.array([42.0, 30.0, 8.0, 70.0])

    sampler_izmir = qmc.LatinHypercube(d=4, seed=seed + 1)
    lhs_izmir = sampler_izmir.random(n=n_izmir)
    scaled_izmir = qmc.scale(lhs_izmir, izmir_lower, izmir_upper)

    df_izmir = pd.DataFrame(
        {
            "sample_id": np.arange(n_full, n_full + n_izmir),
            "sample_set": "izmir_summer_lhs",
            "is_synthetic": True,
            "tdb": np.round(scaled_izmir[:, 0], 4),
            "delta_tmrt": np.round(scaled_izmir[:, 1], 4),
            "tr": np.round(scaled_izmir[:, 0] + scaled_izmir[:, 1], 4),
            "v": np.round(scaled_izmir[:, 2], 4),
            "rh": np.round(scaled_izmir[:, 3], 4),
            "in_izmir_summer_domain": True,
        }
    )

    # Combined master dataframe
    df_combined = pd.concat([df_full, df_izmir], ignore_index=True)

    # Save CSVs
    combined_csv = out_dir / "comfort_inputs_SYNTHETIC.csv"
    full_csv = out_dir / "comfort_inputs_full_lhs_SYNTHETIC.csv"
    izmir_csv = out_dir / "comfort_inputs_izmir_summer_SYNTHETIC.csv"

    df_combined.to_csv(combined_csv, index=False, encoding="utf-8")
    df_full.to_csv(full_csv, index=False, encoding="utf-8")
    df_izmir.to_csv(izmir_csv, index=False, encoding="utf-8")

    print(f"Generated {len(df_full)} full-domain LHS samples (seed={seed}).")
    print(
        f"  - Points within Izmir summer bounds: {izmir_mask_full.sum()} ({izmir_mask_full.mean():.1%})"
    )
    print(
        f"Generated {len(df_izmir)} dedicated Izmir summer LHS samples (seed={seed + 1})."
    )
    print(f"Saved master dataset: {combined_csv} (total rows: {len(df_combined)})")
    print(f"Saved separate full-domain dataset: {full_csv}")
    print(f"Saved separate Izmir summer dataset: {izmir_csv}")

    return df_combined, df_full, df_izmir


if __name__ == "__main__":
    generate_samples()
