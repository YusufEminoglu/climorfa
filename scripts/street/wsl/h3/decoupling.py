"""Decoupled segment identification and shares by LCZ and audit class for H3.

Pre-analysis lock (§2 H3):
- 'Decoupled' segments = top LST tercile and bottom M1 tercile (and vice versa).
- Report their share by LCZ and by audit class (data from the manual LCZ audit).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd

from scripts.street.wsl.h3.bootstrap import LCZ_NAMES, lcz_nodata_report, mask_lcz_nodata, pool_lcz_series


def identify_decoupled_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Compute citywide terciles of LST and M1, and label decoupled segments.

    Decoupled segments:
    - Type 1 (hot_surf_cool_ped): top LST tercile (T3) AND bottom M1 tercile (T1)
    - Type 2 (cool_surf_hot_ped): bottom LST tercile (T1) AND top M1 tercile (T3)
    """
    res = df.copy()

    # Drop NaNs for quantile computation
    valid = np.isfinite(res["lst_med"]) & np.isfinite(res["m1"])
    lst_vals = res.loc[valid, "lst_med"]
    m1_vals = res.loc[valid, "m1"]

    # Compute tercile boundaries
    lst_q1 = float(np.quantile(lst_vals, 1.0 / 3.0))
    lst_q2 = float(np.quantile(lst_vals, 2.0 / 3.0))

    m1_q1 = float(np.quantile(m1_vals, 1.0 / 3.0))
    m1_q2 = float(np.quantile(m1_vals, 2.0 / 3.0))

    # Assign tercile labels (1: bottom, 2: mid, 3: top)
    res["lst_tercile"] = np.nan
    res["m1_tercile"] = np.nan

    res.loc[valid & (res["lst_med"] <= lst_q1), "lst_tercile"] = 1
    res.loc[valid & (res["lst_med"] > lst_q1) & (res["lst_med"] <= lst_q2), "lst_tercile"] = 2
    res.loc[valid & (res["lst_med"] > lst_q2), "lst_tercile"] = 3

    res.loc[valid & (res["m1"] <= m1_q1), "m1_tercile"] = 1
    res.loc[valid & (res["m1"] > m1_q1) & (res["m1"] <= m1_q2), "m1_tercile"] = 2
    res.loc[valid & (res["m1"] > m1_q2), "m1_tercile"] = 3

    # Flags
    res["decoupled_hot_surf_cool_ped"] = (res["lst_tercile"] == 3) & (res["m1_tercile"] == 1)
    res["decoupled_cool_surf_hot_ped"] = (res["lst_tercile"] == 1) & (res["m1_tercile"] == 3)
    res["is_decoupled"] = res["decoupled_hot_surf_cool_ped"] | res["decoupled_cool_surf_hot_ped"]

    # Category code
    # Concordant Hot: (3, 3); Concordant Cool: (1, 1); Neutral: has 2; Decoupled 1: (3, 1); Decoupled 2: (1, 3)
    def _cat(row):
        if pd.isna(row["lst_tercile"]) or pd.isna(row["m1_tercile"]):
            return "Invalid"
        lt, mt = int(row["lst_tercile"]), int(row["m1_tercile"])
        if lt == 3 and mt == 1:
            return "Decoupled (Hot Surf / Cool Ped)"
        if lt == 1 and mt == 3:
            return "Decoupled (Cool Surf / Hot Ped)"
        if lt == 3 and mt == 3:
            return "Concordant Hot (T3/T3)"
        if lt == 1 and mt == 1:
            return "Concordant Cool (T1/T1)"
        return "Intermediate (T2)"

    res["coupling_category"] = res.apply(_cat, axis=1)

    # Store tercile thresholds in attrs
    res.attrs["terciles"] = {
        "lst_q33": lst_q1,
        "lst_q67": lst_q2,
        "m1_q33": m1_q1,
        "m1_q67": m1_q2,
    }

    return res


def compute_decoupled_shares_by_lcz(
    df_decoupled: pd.DataFrame,
    lcz_col: str = "lcz_primary",
    pool: bool = True,
) -> pd.DataFrame:
    """Compute count and share of decoupled segments by LCZ class.

    Amendment A6: segments with LCZ 0 (raster no-data) or a missing label are excluded from every by-LCZ row. The 'ALL'
    row (citywide, no LCZ used) keeps them; its n_lcz_nodata_excluded column reports how many there are.
    """
    work = df_decoupled.copy()
    nodata = lcz_nodata_report(work, (lcz_col,))
    if pool:
        work["lcz_eval"] = pool_lcz_series(work[lcz_col])
    else:
        work["lcz_eval"] = mask_lcz_nodata(work[lcz_col])

    rows = []

    # Overall row
    n_tot = len(work)
    n_dec = int(work["is_decoupled"].sum())
    n_t3_t1 = int(work["decoupled_hot_surf_cool_ped"].sum())
    n_t1_t3 = int(work["decoupled_cool_surf_hot_ped"].sum())

    rows.append({
        "group": "ALL",
        "lcz_label": "ALL (Citywide)",
        "n_segments": n_tot,
        "n_decoupled": n_dec,
        "share_decoupled": n_dec / n_tot if n_tot > 0 else 0.0,
        "n_hot_surf_cool_ped": n_t3_t1,
        "share_hot_surf_cool_ped": n_t3_t1 / n_tot if n_tot > 0 else 0.0,
        "n_cool_surf_hot_ped": n_t1_t3,
        "share_cool_surf_hot_ped": n_t1_t3 / n_tot if n_tot > 0 else 0.0,
        "n_lcz_nodata_excluded": nodata.get(lcz_col, {}).get("n_excluded_from_by_lcz", 0),
    })

    # Grouped rows
    for c in sorted(work["lcz_eval"].dropna().unique(), key=lambda v: str(v)):
        sub = work[work["lcz_eval"] == c]
        n_c = len(sub)
        n_d = int(sub["is_decoupled"].sum())
        n_1 = int(sub["decoupled_hot_surf_cool_ped"].sum())
        n_2 = int(sub["decoupled_cool_surf_hot_ped"].sum())
        rows.append({
            "group": str(c),
            "lcz_label": LCZ_NAMES.get(c, f"LCZ {c}"),
            "n_segments": n_c,
            "n_decoupled": n_d,
            "share_decoupled": n_d / n_c if n_c > 0 else 0.0,
            "n_hot_surf_cool_ped": n_1,
            "share_hot_surf_cool_ped": n_1 / n_c if n_c > 0 else 0.0,
            "n_cool_surf_hot_ped": n_2,
            "share_cool_surf_hot_ped": n_2 / n_c if n_c > 0 else 0.0,
        })

    out = pd.DataFrame(rows)
    out.attrs["lcz_nodata_excluded"] = nodata
    return out


def load_manual_audit_table(
    audit_csv: Union[str, Path] = "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv",
    audit_gpkg: Union[str, Path] = "data/04_training_labels/audit_sample_v1.gpkg",
) -> pd.DataFrame:
    """Load the authoritative 569-point manual LCZ audit dataset and compute g250_id."""
    audit_csv = Path(audit_csv)
    audit_gpkg = Path(audit_gpkg)

    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")
    if not audit_gpkg.exists():
        raise FileNotFoundError(f"Audit GPKG not found: {audit_gpkg}")

    df_csv = pd.read_csv(audit_csv)
    gdf_poly = gpd.read_file(audit_gpkg)

    # Compute g250_id from polygon bounds: f"{x0//250}_{y0//250}"
    b = gdf_poly.geometry.bounds
    gdf_poly["g250_id"] = (
        (b["minx"].astype(int) // 250).astype(str) + "_" + (b["miny"].astype(int) // 250).astype(str)
    )

    merged = gdf_poly[["grid_id", "g250_id", "geometry"]].merge(
        df_csv, on="grid_id", how="inner"
    )

    return merged


def link_segments_to_audit(
    df_segments: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join audit labels onto segments using g250_id."""
    audit_sub = audit_df[["g250_id", "audit_label", "label_quality", "grid_id"]].rename(
        columns={"grid_id": "audit_grid_id"}
    )
    # Segments have g250_id
    res = df_segments.merge(audit_sub, on="g250_id", how="left")
    return res


def compute_decoupled_shares_by_audit(
    df_decoupled_with_audit: pd.DataFrame,
    min_segments: int = 5,
) -> pd.DataFrame:
    """Compute count and share of decoupled segments by manual audit label."""
    # Filter to segments that fall in audited grid cells
    audited = df_decoupled_with_audit[df_decoupled_with_audit["audit_label"].notna()].copy()

    rows = []

    # Overall audited row
    n_tot = len(audited)
    n_dec = int(audited["is_decoupled"].sum())
    n_t3_t1 = int(audited["decoupled_hot_surf_cool_ped"].sum())
    n_t1_t3 = int(audited["decoupled_cool_surf_hot_ped"].sum())

    rows.append({
        "audit_label": "ALL_AUDITED",
        "label_quality_subset": "ALL",
        "n_segments": n_tot,
        "n_decoupled": n_dec,
        "share_decoupled": n_dec / n_tot if n_tot > 0 else 0.0,
        "n_hot_surf_cool_ped": n_t3_t1,
        "share_hot_surf_cool_ped": n_t3_t1 / n_tot if n_tot > 0 else 0.0,
        "n_cool_surf_hot_ped": n_t1_t3,
        "share_cool_surf_hot_ped": n_t1_t3 / n_tot if n_tot > 0 else 0.0,
    })

    # Group by audit_label
    labels = sorted(audited["audit_label"].unique())
    for lbl in labels:
        sub = audited[audited["audit_label"] == lbl]
        n_c = len(sub)
        n_d = int(sub["is_decoupled"].sum())
        n_1 = int(sub["decoupled_hot_surf_cool_ped"].sum())
        n_2 = int(sub["decoupled_cool_surf_hot_ped"].sum())

        rows.append({
            "audit_label": lbl,
            "label_quality_subset": "ALL",
            "n_segments": n_c,
            "n_decoupled": n_d,
            "share_decoupled": n_d / n_c if n_c > 0 else 0.0,
            "n_hot_surf_cool_ped": n_1,
            "share_hot_surf_cool_ped": n_1 / n_c if n_c > 0 else 0.0,
            "n_cool_surf_hot_ped": n_2,
            "share_cool_surf_hot_ped": n_2 / n_c if n_c > 0 else 0.0,
        })

    # Also compute for high quality audit labels only
    high_q = audited[audited["label_quality"] == "high"]
    if len(high_q) > 0:
        n_hq_tot = len(high_q)
        n_hq_dec = int(high_q["is_decoupled"].sum())
        rows.append({
            "audit_label": "ALL_AUDITED_HIGH_QUALITY",
            "label_quality_subset": "high_only",
            "n_segments": n_hq_tot,
            "n_decoupled": n_hq_dec,
            "share_decoupled": n_hq_dec / n_hq_tot if n_hq_tot > 0 else 0.0,
            "n_hot_surf_cool_ped": int(high_q["decoupled_hot_surf_cool_ped"].sum()),
            "share_hot_surf_cool_ped": int(high_q["decoupled_hot_surf_cool_ped"].sum()) / n_hq_tot,
            "n_cool_surf_hot_ped": int(high_q["decoupled_cool_surf_hot_ped"].sum()),
            "share_cool_surf_hot_ped": int(high_q["decoupled_cool_surf_hot_ped"].sum()) / n_hq_tot,
        })

    return pd.DataFrame(rows)
