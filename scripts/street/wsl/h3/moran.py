"""Bivariate Local Moran analysis (LST vs M1) at 250 m for H3.

Pre-analysis lock (§2 H3):
- Landsat summer LST: median per 250 m cell.
- Bivariate local Moran (LST vs M1) at 250 m.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
from esda.moran import Moran_BV, Moran_Local_BV

SEED = 20261030
PERMUTATIONS = 999


def aggregate_m1_to_grid250(
    segments_with_metrics: pd.DataFrame,
    grid_250_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate segment M1 (utci_b15) to 250 m grid cells using PS-weighted averaging."""
    gdf = grid_250_gdf.copy()

    # Check if m1 already present in grid_250
    if "m1" in gdf.columns and gdf["m1"].notna().sum() > 0:
        return gdf

    # Aggregate from segments
    valid_segs = segments_with_metrics[
        segments_with_metrics["g250_id"].notna()
        & np.isfinite(segments_with_metrics["m1"])
    ].copy()

    if len(valid_segs) == 0:
        gdf["m1"] = np.nan
        gdf["n_segments_in_cell"] = 0
        return gdf

    # PS weight: n_ps_valid if present, else n_ps, else 1
    if "n_ps_valid" in valid_segs.columns:
        valid_segs["weight"] = valid_segs["n_ps_valid"].fillna(1).clip(lower=1)
    elif "n_ps" in valid_segs.columns:
        valid_segs["weight"] = valid_segs["n_ps"].fillna(1).clip(lower=1)
    else:
        valid_segs["weight"] = 1.0

    valid_segs["weighted_m1"] = valid_segs["m1"] * valid_segs["weight"]

    agg = valid_segs.groupby("g250_id").agg(
        sum_weighted_m1=("weighted_m1", "sum"),
        sum_weight=("weight", "sum"),
        n_segments_in_cell=("m1", "count"),
        m1_median=("m1", "median"),
    ).reset_index()

    agg["m1"] = agg["sum_weighted_m1"] / agg["sum_weight"]

    # Merge into grid_250
    merged = gdf.merge(agg[["g250_id", "m1", "m1_median", "n_segments_in_cell"]], on="g250_id", how="left")
    return merged


def run_bivariate_moran_250m(
    grid_250_gdf: gpd.GeoDataFrame,
    lst_col: str = "lst_med",
    m1_col: str = "m1",
    permutations: int = PERMUTATIONS,
    seed: int = SEED,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Compute Global Bivariate Moran's I and Bivariate Local Moran (LST vs M1) on 250 m grid.

    Returns:
        (gdf_annotated, summary_dict)
    """
    # Filter to cells with both valid LST and valid M1
    valid_mask = (
        np.isfinite(grid_250_gdf[lst_col])
        & np.isfinite(grid_250_gdf[m1_col])
        & grid_250_gdf.geometry.is_valid
        & ~grid_250_gdf.geometry.is_empty
    )

    gdf_valid = grid_250_gdf[valid_mask].copy().reset_index(drop=True)
    n_valid = len(gdf_valid)

    if n_valid < 10:
        summary = {
            "error": "Insufficient valid grid cells for spatial Moran analysis",
            "n_valid": n_valid,
            "global_moran_I": float("nan"),
            "global_moran_p": float("nan"),
        }
        return grid_250_gdf, summary

    x = gdf_valid[lst_col].to_numpy(float)
    y = gdf_valid[m1_col].to_numpy(float)

    # Spatial weights: Queen contiguity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = libpysal.weights.Queen.from_dataframe(gdf_valid, use_index=False)
        w.silence_warnings = True
        w.transform = "r"

    # Global Bivariate Moran
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glob_bv = Moran_BV(x, y, w, permutations=permutations)

    # Local Bivariate Moran
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_bv = Moran_Local_BV(x, y, w, permutations=permutations, seed=seed)

    gdf_valid["moran_bv_I"] = loc_bv.Is
    gdf_valid["moran_bv_q"] = loc_bv.q
    gdf_valid["moran_bv_p_sim"] = loc_bv.p_sim
    gdf_valid["moran_bv_sig"] = loc_bv.p_sim < 0.05

    # Quadrant mapping in PySAL Moran_Local_BV:
    # 1: High x, High Wy -> High LST, High M1 (HH)
    # 2: Low x, High Wy  -> Low LST, High M1 (LH)
    # 3: Low x, Low Wy   -> Low LST, Low M1 (LL)
    # 4: High x, Low Wy  -> High LST, Low M1 (HL)
    quad_code_to_name = {
        1: "HH",
        2: "LH",
        3: "LL",
        4: "HL",
    }
    gdf_valid["quadrant_raw"] = gdf_valid["moran_bv_q"].map(quad_code_to_name)

    def _cluster_type(row):
        if not row["moran_bv_sig"]:
            return "Not Significant"
        return row["quadrant_raw"]

    gdf_valid["cluster_label"] = gdf_valid.apply(_cluster_type, axis=1)

    # Cluster counts and shares
    cluster_counts = gdf_valid["cluster_label"].value_counts().to_dict()
    quad_raw_counts = gdf_valid["quadrant_raw"].value_counts().to_dict()

    summary = {
        "n_grid_cells_total": len(grid_250_gdf),
        "n_grid_cells_analyzed": n_valid,
        "global_bivariate_moran_I": float(glob_bv.I),
        "global_bivariate_moran_p_sim": float(glob_bv.p_sim),
        "global_bivariate_moran_z_sim": float(glob_bv.z_sim) if hasattr(glob_bv, "z_sim") else float("nan"),
        "global_bivariate_moran_EI_sim": float(getattr(glob_bv, "EI_sim", float("nan"))),
        "significant_clusters_count": int(gdf_valid["moran_bv_sig"].sum()),
        "significant_clusters_share": float(gdf_valid["moran_bv_sig"].mean()),
        "cluster_counts": cluster_counts,
        "quadrant_raw_counts": quad_raw_counts,
        "hh_count": int(cluster_counts.get("HH", 0)),
        "ll_count": int(cluster_counts.get("LL", 0)),
        "hl_count": int(cluster_counts.get("HL", 0)),
        "lh_count": int(cluster_counts.get("LH", 0)),
        "not_sig_count": int(cluster_counts.get("Not Significant", 0)),
        "hh_share": float(cluster_counts.get("HH", 0) / n_valid),
        "ll_share": float(cluster_counts.get("LL", 0) / n_valid),
        "hl_share": float(cluster_counts.get("HL", 0) / n_valid),
        "lh_share": float(cluster_counts.get("LH", 0) / n_valid),
        "not_sig_share": float(cluster_counts.get("Not Significant", 0) / n_valid),
    }

    # Merge results back into original grid
    res_gdf = grid_250_gdf.merge(
        gdf_valid[["g250_id", "m1", "moran_bv_I", "moran_bv_q", "moran_bv_p_sim", "moran_bv_sig", "quadrant_raw", "cluster_label"]],
        on="g250_id",
        how="left",
    )

    return res_gdf, summary
