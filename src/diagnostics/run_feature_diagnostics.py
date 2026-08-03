"""Run diagnostics for the 250 m CLIMORFA model feature matrix.

The script is intentionally dependency-light: pandas and numpy only. It writes
machine-readable CSV/JSON outputs plus a compact Markdown handoff report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/03_processed/grid_250m_model_features_v8.csv"
DEFAULT_OUT_DIR = "outputs/diagnostics/feature_diagnostics_v8_2026-06-18"
DEFAULT_REPORT = "docs/methodology/feature_diagnostics_v8_2026-06-18.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-md", default=DEFAULT_REPORT)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--corr-review-threshold", type=float, default=0.90)
    parser.add_argument("--min-corr-periods", type=int, default=200)
    return parser.parse_args()


def infer_family(column: str) -> str:
    if column in {
        "grid_id",
        "cell_m",
        "cell_area_m2",
        "fua_area_m2",
        "fua_ratio",
        "eligible_core",
        "row_id",
        "col_id",
        "district",
        "built_intensity_bin",
        "industrial_port_flag",
    }:
        return "sampling_frame"
    if column.startswith("lcz_"):
        return "weak_lcz_label"
    if column.startswith("lst_"):
        return "climate_response"
    if column.startswith("dsm_"):
        return "dsm_5m"
    if column.startswith("dw_"):
        return "dynamic_world"
    if column.startswith("s2_"):
        return "sentinel2"
    if column.startswith("worldcover_"):
        return "worldcover"
    if column.startswith("coast_") or column.startswith("coastal_"):
        return "coastal_context"
    if column.startswith("canopy_"):
        return "canopy_height"
    if column.startswith("osm_"):
        return "osm_proxy_context"
    if column.startswith("green_2sfca_"):
        return "green_space_2sfca"
    if column.startswith("population_") or column == "residential_floor_area_proxy_m2":
        return "population_demand_allocation"
    if column.startswith(("building_count_prelim", "building_area_prelim", "building_coverage_prelim")):
        return "preliminary_morphology"
    if column.startswith("morph_"):
        return "building_footprint_morphometrics"
    if column.startswith("street_frontage_") or column.startswith("block_permeability_"):
        return "street_facing_morphology"
    if column.startswith("road_") and "prelim" in column:
        return "preliminary_network"
    if column.startswith("network_reach_"):
        return "road_network_reach"
    if column.startswith("road_") or column.startswith("network_"):
        return "road_network_exact"
    if column.startswith(
        (
            "building_",
            "floor_",
            "height_proxy_",
            "underground_",
            "lowrise_",
            "midrise_",
            "upper_midrise_",
            "highrise_",
            "type_",
            "dominant_yapitip",
            "built_volume_",
        )
    ):
        return "building_floor_proxy"
    return "other"


def infer_role(column: str) -> str:
    if column == "grid_id":
        return "id"
    if column in {"row_id", "col_id"}:
        return "spatial_index_do_not_predict"
    if column == "district":
        return "admin_context_validate_by_leave_district_out"
    if column in {"cell_m", "cell_area_m2", "fua_area_m2", "fua_ratio", "eligible_core"}:
        return "grid_frame_or_sample_filter"
    if column.startswith("lcz_"):
        return "label_or_label_quality_do_not_predict_lcz"
    if column.startswith("lst_"):
        return "climate_outcome_or_validation"
    if (
        column.endswith("_valid_px")
        or column.endswith("_valid_pixels")
        or column.endswith("_valid_buildings")
        or "obs_count" in column
        or "uncertainty_valid" in column
        or column.endswith("_buffer_area_m2")
        or column.endswith("_fua_area_m2")
        or column.endswith("_fua_coverage_share")
    ):
        return "coverage_or_quality_control"
    if column in {"built_intensity_bin", "industrial_port_flag"}:
        return "sampling_or_context_flag"
    if column in {
        "population_mahalle_key",
        "population_2024_dasymetric",
        "population_allocation_method",
        "residential_floor_area_proxy_m2",
        "green_2sfca_network_available_flag",
        "green_2sfca_network_snap_distance_m",
    }:
        return "2sfca_demand_or_quality_control"
    if column.startswith("green_2sfca_") and not column.endswith("_access_log1p"):
        return "2sfca_descriptive_or_redundant"
    return "candidate_predictor"


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f_value) or math.isinf(f_value):
        return None
    return f_value


def markdown_table(rows: list[dict[str, object]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    if not shown:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in shown:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if abs(value) >= 1000:
                    values.append(f"{value:,.2f}")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] * len(columns)) + " |")
    return "\n".join(lines)


def get_quantiles(series: pd.Series) -> dict[str, float | None]:
    values = series.dropna()
    if values.empty:
        return {key: None for key in ["min", "p01", "p05", "median", "mean", "p95", "p99", "max"]}
    quantiles = values.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "min": safe_float(values.min()),
        "p01": safe_float(quantiles.loc[0.01]),
        "p05": safe_float(quantiles.loc[0.05]),
        "median": safe_float(quantiles.loc[0.5]),
        "mean": safe_float(values.mean()),
        "p95": safe_float(quantiles.loc[0.95]),
        "p99": safe_float(quantiles.loc[0.99]),
        "max": safe_float(values.max()),
    }


def build_column_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_count = len(df)
    for column in df.columns:
        raw = df[column]
        numeric = clean_numeric(raw)
        numeric_non_null = int(numeric.notna().sum())
        non_null = int(raw.notna().sum())
        missing_count = row_count - non_null
        unique_count = int(raw.nunique(dropna=True))
        numeric_share = numeric_non_null / row_count if row_count else 0.0
        is_numeric = numeric_share >= 0.95 or (numeric_non_null == non_null and numeric_non_null > 0)
        values_for_zero = numeric.dropna()
        zero_share = None
        if is_numeric and len(values_for_zero) > 0:
            zero_share = safe_float((values_for_zero == 0).mean())
        q = get_quantiles(numeric) if is_numeric else {key: None for key in ["min", "p01", "p05", "median", "mean", "p95", "p99", "max"]}
        top_share = None
        if non_null:
            top_share = safe_float(raw.value_counts(dropna=True).iloc[0] / non_null)
        rows.append(
            {
                "column": column,
                "family": infer_family(column),
                "role": infer_role(column),
                "source_dtype": str(raw.dtype),
                "is_numeric": bool(is_numeric),
                "row_count": row_count,
                "non_null": non_null,
                "missing_count": missing_count,
                "missing_share": safe_float(missing_count / row_count if row_count else 0.0),
                "unique_count": unique_count,
                "top_value_share": top_share,
                "zero_share": zero_share,
                "constant_flag": bool(unique_count <= 1),
                "near_constant_flag": bool((top_share is not None and top_share >= 0.99) or unique_count <= 1),
                **q,
            }
        )
    return pd.DataFrame(rows)


def summarize_family_missingness(column_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in column_diag.groupby("family", sort=True):
        rows.append(
            {
                "family": family,
                "columns": int(len(group)),
                "avg_missing_share": float(group["missing_share"].mean()),
                "max_missing_share": float(group["missing_share"].max()),
                "columns_with_any_missing": int((group["missing_count"] > 0).sum()),
                "near_constant_columns": int(group["near_constant_flag"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_missing_share", "avg_missing_share"], ascending=False)


def categorical_value_counts(df: pd.DataFrame, column_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in column_diag.to_dict("records"):
        column = str(record["column"])
        if bool(record["is_numeric"]) and int(record["unique_count"]) > 20:
            continue
        counts = df[column].value_counts(dropna=False).head(30)
        for value, count in counts.items():
            rows.append(
                {
                    "column": column,
                    "family": record["family"],
                    "value": "<NA>" if pd.isna(value) else str(value),
                    "count": int(count),
                    "share": float(count / len(df)) if len(df) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def candidate_numeric_columns(column_diag: pd.DataFrame) -> list[str]:
    excluded_roles = {
        "id",
        "spatial_index_do_not_predict",
        "grid_frame_or_sample_filter",
        "label_or_label_quality_do_not_predict_lcz",
        "climate_outcome_or_validation",
        "coverage_or_quality_control",
        "2sfca_demand_or_quality_control",
        "2sfca_descriptive_or_redundant",
    }
    rows = column_diag[
        (column_diag["is_numeric"])
        & (~column_diag["role"].isin(excluded_roles))
        & (~column_diag["constant_flag"])
        & (column_diag["non_null"] >= 200)
        & (column_diag["missing_share"] < 0.80)
    ]
    return rows["column"].astype(str).tolist()


def build_correlation_pairs(
    df: pd.DataFrame,
    column_diag: pd.DataFrame,
    threshold: float,
    review_threshold: float,
    min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = candidate_numeric_columns(column_diag)
    if len(columns) < 2:
        return pd.DataFrame(), pd.DataFrame()
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="pearson", min_periods=min_periods)
    families = dict(zip(column_diag["column"], column_diag["family"]))
    roles = dict(zip(column_diag["column"], column_diag["role"]))
    rows = []
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if abs_value >= review_threshold:
                rows.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "corr": float(value),
                        "abs_corr": abs_value,
                        "family_a": families.get(col_a, ""),
                        "family_b": families.get(col_b, ""),
                        "role_a": roles.get(col_a, ""),
                        "role_b": roles.get(col_b, ""),
                        "action": "collapse_or_ablate" if abs_value >= threshold else "review",
                    }
                )
    pairs = pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame()
    return pairs, corr


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def retention_score(column: str) -> int:
    priority_overrides = {
        "building_coverage_exact": 20,
        "building_density_exact_per_km2": 20,
        "floor_area_ratio_proxy": 20,
        "height_proxy_aw_mean_m": 18,
        "canopy_volume_gt2m_proxy_m3_per_ha": 18,
        "canopy_height_p95_m": 18,
        "coast_min_distance_m": 18,
        "road_density_exact_m_per_km2": 18,
        "network_intersection_density_per_km2": 18,
        "road_orientation_entropy_norm": 17,
        "network_reach_250m_road_density_m_per_km2": 17,
        "network_reach_400m_road_density_m_per_km2": 18,
        "network_reach_800m_road_density_m_per_km2": 17,
        "network_reach_400m_intersection_density_per_km2": 18,
        "network_reach_400m_orientation_entropy_norm": 17,
        "network_reach_250m_road_excl_motorway_density_m_per_km2": 15,
        "network_reach_400m_road_excl_motorway_density_m_per_km2": 16,
        "network_reach_800m_road_excl_motorway_density_m_per_km2": 15,
        "morph_building_perimeter_density_m_per_ha": 18,
        "morph_building_area_gini": 17,
        "morph_building_compactness_aw_mean": 18,
        "morph_open_space_fragmentation_index": 17,
        "street_frontage_continuity_proxy": 18,
        "street_frontage_open_buffer_share": 17,
        "street_frontage_edge_m_per_road_km": 17,
        "block_permeability_network_proxy": -8,
        "block_permeability_street_open_network_proxy": -10,
        "road_density_prelim_m_per_km2": 16,
    }
    score = priority_overrides.get(column, 0)
    good_tokens = ["share", "ratio", "density", "coverage", "per_ha", "mean", "median", "aw_mean", "p95", "exact"]
    weak_tokens = ["valid", "count", "area_m2", "flag", "max", "std", "raw", "prelim"]
    for token in good_tokens:
        if token in column:
            score += 2
    for token in weak_tokens:
        if token in column:
            score -= 1
    if column.startswith(("osm_share_", "worldcover_share_", "dw_")):
        score += 1
    if column.endswith("_flag"):
        score -= 2
    return score


def build_collinearity_groups(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    collapse_pairs = pairs[pairs["abs_corr"] >= threshold]
    features = sorted(set(collapse_pairs["feature_a"]).union(set(collapse_pairs["feature_b"])))
    uf = UnionFind(features)
    for row in collapse_pairs.to_dict("records"):
        uf.union(str(row["feature_a"]), str(row["feature_b"]))
    grouped: dict[str, list[str]] = defaultdict(list)
    for feature in features:
        grouped[uf.find(feature)].append(feature)
    rows = []
    for index, features_in_group in enumerate(grouped.values(), start=1):
        if len(features_in_group) < 2:
            continue
        ranked = sorted(features_in_group, key=lambda item: (-retention_score(item), item))
        rows.append(
            {
                "group_id": f"corr95_{index:03d}",
                "feature_count": len(features_in_group),
                "features": "; ".join(sorted(features_in_group)),
                "retain_first_review": "; ".join(ranked[:3]),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_count", ascending=False) if rows else pd.DataFrame()


def build_leakage_screen(column_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in column_diag.to_dict("records"):
        column = str(record["column"])
        role = str(record["role"])
        family = str(record["family"])
        issue = None
        recommendation = None
        if role == "label_or_label_quality_do_not_predict_lcz":
            issue = "Weak label, confidence, or label-quality field."
            recommendation = "Exclude from LCZ/audit-label predictors; use as target, weight, or audit metadata only."
        elif role == "climate_outcome_or_validation":
            issue = "Climate response or validation field."
            recommendation = "Exclude from LCZ-like typology predictors; use for thermal-response modelling or post-hoc validation."
        elif role == "coverage_or_quality_control":
            issue = "Pixel/building coverage or quality field."
            recommendation = "Use for filtering, weighting, or QA; avoid as a morphology signal unless explicitly justified."
        elif role == "spatial_index_do_not_predict":
            issue = "Spatial index can encode location directly."
            recommendation = "Use for spatial CV blocks and diagnostics, not as a model predictor."
        elif role == "admin_context_validate_by_leave_district_out":
            issue = "Administrative context can overfit district identity."
            recommendation = "Do not use in baseline predictors; if tested, report leave-district-out sensitivity."
        elif column == "built_intensity_bin":
            issue = "Sampling stratum derived from preliminary built intensity."
            recommendation = "Use for audit/sample balance; do not include in final predictors unless documented as a coarse baseline."
        elif family in {"dynamic_world", "worldcover", "sentinel2"}:
            issue = "Remote-sensing predictor may partially overlap the weak LCZ source logic."
            recommendation = "Keep in raster/RS branch but ablate against morphology-only models to quantify semantic leakage risk."
        elif family == "osm_proxy_context":
            issue = "Proxy functional context, not official zoning."
            recommendation = "Use only with conservative wording and run sensitivity with/without OSM context."
        if issue is not None:
            rows.append(
                {
                    "column": column,
                    "family": family,
                    "role": role,
                    "issue": issue,
                    "recommendation": recommendation,
                }
            )
    return pd.DataFrame(rows)


def build_feature_shortlist(column_diag: pd.DataFrame, collinearity_groups: pd.DataFrame) -> pd.DataFrame:
    known_available = {
        "building_coverage_exact": ("tabular_morphology_core", "Retain", "Exact footprint coverage; stronger than preliminary coverage."),
        "building_density_exact_per_km2": ("tabular_morphology_core", "Retain", "Building count intensity normalized by area."),
        "floor_area_ratio_proxy": ("tabular_morphology_core", "Retain", "Central floor-count/FAR proxy; keep wording as proxy."),
        "height_proxy_aw_mean_m": ("tabular_morphology_core", "Retain", "Area-weighted floor-height proxy."),
        "height_proxy_max_m": ("tabular_morphology_core", "Review", "Useful vertical-intensity extreme; sensitive to outliers."),
        "lowrise_1_2_area_share": ("tabular_morphology_core", "Retain", "Low-rise composition."),
        "midrise_3_5_area_share": ("tabular_morphology_core", "Retain", "Mid-rise composition."),
        "upper_midrise_6_10_area_share": ("tabular_morphology_core", "Retain", "Upper-mid-rise composition."),
        "highrise_11_plus_area_share": ("tabular_morphology_core", "Retain", "High-rise composition."),
        "morph_building_perimeter_density_m_per_ha": ("footprint_morphometrics", "Retain", "Footprint edge intensity and frontage-like roughness proxy."),
        "morph_building_area_gini": ("footprint_morphometrics", "Retain", "Building-size heterogeneity."),
        "morph_building_area_cv": ("footprint_morphometrics", "Review", "Alternative building-size heterogeneity measure."),
        "morph_building_compactness_aw_mean": ("footprint_morphometrics", "Retain", "Area-weighted footprint compactness."),
        "morph_open_space_patch_density_per_ha": ("footprint_morphometrics", "Review", "Open-space fragmentation count normalized by cell area."),
        "morph_open_space_fragmentation_index": ("footprint_morphometrics", "Retain", "Open-space fragmentation proxy combining patch count and largest-patch dominance."),
        "street_frontage_continuity_proxy": ("street_facing_morphology", "Retain", "Road-buffer building-edge continuity proxy; not a cadastral frontage measure."),
        "street_frontage_open_buffer_share": ("street_facing_morphology", "Retain", "Road-adjacent openness proxy; do not interpret it as block or pedestrian permeability."),
        "street_frontage_edge_m_per_road_km": ("street_facing_morphology", "Review", "Building-edge intensity normalized by road length; likely correlated with continuity."),
        "block_permeability_network_proxy": ("street_facing_morphology", "Experimental", "Visual QA selected motorway/interchange conditions; exclude from first baselines pending topology and road-class refinement."),
        "green_2sfca_800m_access_log1p": ("green_space_2sfca", "Retain", "Primary competition-adjusted green-space accessibility context; log1p controls the rural upper tail."),
        "green_2sfca_400m_access_log1p": ("green_space_2sfca", "Sensitivity", "Short-walk 2SFCA sensitivity."),
        "green_2sfca_1200m_access_log1p": ("green_space_2sfca", "Sensitivity", "Broader-walk 2SFCA sensitivity."),
        "block_permeability_street_open_network_proxy": ("street_facing_morphology", "Exclude", "Failed semantic visual QA on 2026-06-18; maximum scores represented motorway/interchange landscapes rather than permeable urban blocks."),
        "dsm_elevation_m_mean": ("dsm_branch", "Review", "Surface/elevation context; not nDSM."),
        "dsm_elevation_m_std": ("dsm_branch", "Retain", "Surface roughness candidate."),
        "dsm_elevation_m_range": ("dsm_branch", "Retain", "Surface relief candidate; check terrain confounding."),
        "canopy_cover_gt2m_share": ("vegetation_branch", "Retain", "Core canopy-cover metric."),
        "canopy_height_p95_m": ("vegetation_branch", "Retain", "Tall-canopy structure."),
        "canopy_volume_gt2m_proxy_m3_per_ha": ("vegetation_branch", "Retain", "Planning-friendly vegetation volume proxy."),
        "s2_ndvi_mean": ("raster_rs_branch", "Retain", "Spectral greenness."),
        "s2_ndbi_mean": ("raster_rs_branch", "Retain", "Built-up spectral signal; ablate for LCZ weak-label leakage."),
        "s2_ndwi_mean": ("raster_rs_branch", "Retain", "Water/moisture signal."),
        "dw_built_prob_mean": ("raster_rs_branch", "Review", "Strong LCZ-overlap risk; keep branch ablation."),
        "dw_trees_prob_mean": ("raster_rs_branch", "Review", "Vegetation semantic signal; compare with canopy metrics."),
        "dw_water_prob_mean": ("raster_rs_branch", "Review", "Water semantic signal; compare with coast/WorldCover."),
        "worldcover_share_built_up": ("raster_rs_branch", "Review", "Built land-cover share; ablate against morphology-only."),
        "worldcover_share_tree_cover": ("raster_rs_branch", "Review", "Tree-cover share; compare with canopy."),
        "worldcover_share_water": ("raster_rs_branch", "Review", "Water share; compare with coastal context."),
        "coast_min_distance_m": ("coastal_context", "Retain", "Izmir-specific coastal climate gradient."),
        "coastal_2km_flag": ("coastal_context", "Review", "Coarse coastal band; may duplicate continuous distance."),
        "coastal_5km_flag": ("coastal_context", "Review", "Coarse coastal band; may duplicate continuous distance."),
        "road_density_exact_m_per_km2": ("network_context_current", "Retain", "Exact grid-intersected road length density."),
        "network_intersection_density_per_km2": ("network_context_current", "Retain", "Endpoint-derived intersection-node density proxy."),
        "network_dead_end_density_per_km2": ("network_context_current", "Review", "Endpoint-derived dead-end density proxy; sensitive to road segmentation."),
        "network_mean_endpoint_degree": ("network_context_current", "Review", "Endpoint-derived connectivity proxy; validate against topology assumptions."),
        "road_orientation_entropy_norm": ("network_context_current", "Retain", "Directional order/disorder of road segments within the cell."),
        "road_orientation_dominant_bin_share": ("network_context_current", "Review", "Complements orientation entropy; likely redundant in first baselines."),
        "road_presence_flag": ("network_context_current", "Review", "Useful for sparse/unbuilt cells; may duplicate road length."),
        "road_density_prelim_m_per_km2": ("network_context_current", "Superseded", "Keep for comparison only; exact road density is now available."),
        "network_reach_250m_road_density_m_per_km2": ("network_reach_context", "Retain", "250 m centroid-buffer road-density context."),
        "network_reach_400m_road_density_m_per_km2": ("network_reach_context", "Retain", "400 m centroid-buffer road-density context."),
        "network_reach_800m_road_density_m_per_km2": ("network_reach_context", "Review", "800 m broader context; useful for sensitivity but may smooth local form."),
        "network_reach_400m_intersection_density_per_km2": ("network_reach_context", "Retain", "Mid-scale endpoint-derived intersection-density context."),
        "network_reach_400m_dead_end_density_per_km2": ("network_reach_context", "Review", "Mid-scale endpoint-derived dead-end context; validate segmentation effects."),
        "network_reach_400m_orientation_entropy_norm": ("network_reach_context", "Retain", "Mid-scale directional order/disorder context."),
        "network_reach_250m_road_excl_motorway_density_m_per_km2": ("network_reach_motorway_sensitivity", "Sensitivity", "250 m road density after excluding features explicitly coded YOLTIP=OTOYOL."),
        "network_reach_400m_road_excl_motorway_density_m_per_km2": ("network_reach_motorway_sensitivity", "Sensitivity", "400 m road density after excluding features explicitly coded YOLTIP=OTOYOL."),
        "network_reach_800m_road_excl_motorway_density_m_per_km2": ("network_reach_motorway_sensitivity", "Sensitivity", "800 m road density after excluding features explicitly coded YOLTIP=OTOYOL."),
        "network_reach_250m_motorway_share_of_road_length": ("network_reach_motorway_sensitivity", "QA", "Explicit motorway share for interpreting local road-density inflation."),
        "network_reach_400m_fua_coverage_share": ("network_reach_context_qa", "QA", "Edge/coverage QA field; use for filtering or weighting, not morphology signal."),
        "osm_share_industrial_port": ("osm_proxy_context", "Review", "Functional proxy; not official zoning."),
        "osm_share_green_open": ("osm_proxy_context", "Review", "Proxy context; compare with vegetation/land-cover metrics."),
        "osm_share_blue_water": ("osm_proxy_context", "Review", "Proxy context; compare with water/coast metrics."),
        "osm_special_infrastructure_flag": ("osm_proxy_context", "Review", "Useful exclusion/context flag; run sensitivity without it."),
    }
    available = set(column_diag["column"].astype(str))
    grouped_features = set()
    if not collinearity_groups.empty:
        for features in collinearity_groups["features"].astype(str):
            grouped_features.update(item.strip() for item in features.split(";"))
    rows = []
    for feature, (branch, decision, rationale) in known_available.items():
        rows.append(
            {
                "feature": feature,
                "branch": branch,
                "decision": decision if feature in available else "Missing",
                "available": feature in available,
                "in_corr95_group": feature in grouped_features,
                "rationale": rationale,
            }
        )
    remaining_tasks = [
        ("cadastral_frontage_measurement", "tabular_morphology_optional", "Optional", False, False, "The road-buffer frontage proxy passed visual QA; cadastral frontage remains optional."),
        ("block_permeability_topology_refinement", "tabular_morphology_to_build", "Build", False, False, "Current block proxy failed semantic visual QA; add road classes and planar/route-aware topology before reuse."),
        ("green_space_2sfca_external_validation", "network_context_validation", "Partial", False, False, "IZBB 2023 north/south maintenance inventory triangulation is complete; district patterns are consistent after FUA-neighbourhood matching, but coordinates and current public-access status remain unavailable."),
    ]
    rows.extend(
        {
            "feature": feature,
            "branch": branch,
            "decision": decision,
            "available": available_flag,
            "in_corr95_group": corr_flag,
            "rationale": rationale,
        }
        for feature, branch, decision, available_flag, corr_flag, rationale in remaining_tasks
    )
    return pd.DataFrame(rows)


def format_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def build_report(
    input_csv: Path,
    df: pd.DataFrame,
    column_diag: pd.DataFrame,
    family_missingness: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    collinearity_groups: pd.DataFrame,
    leakage_screen: pd.DataFrame,
    feature_shortlist: pd.DataFrame,
    out_dir: Path,
) -> str:
    row_count = len(df)
    column_count = len(df.columns)
    numeric_count = int(column_diag["is_numeric"].sum())
    candidate_count = len(candidate_numeric_columns(column_diag))
    rows_with_any_missing = int(df.isna().any(axis=1).sum())
    rows_complete = row_count - rows_with_any_missing
    built_cells = None
    unbuilt_cells = None
    floor_structural_missing = None
    if "building_count_exact" in df.columns:
        building_count = pd.to_numeric(df["building_count_exact"], errors="coerce")
        built_cells = int((building_count > 0).sum())
        unbuilt_cells = int((building_count <= 0).sum())
    if "floor_count_aw_mean" in df.columns:
        floor_structural_missing = int(df["floor_count_aw_mean"].isna().sum())
    road_cells = None
    no_road_cells = None
    node_cells = None
    no_node_cells = None
    street_frontage_road_cells = None
    street_frontage_building_cells = None
    street_frontage_continuity_cells = None
    if "road_length_exact_m" in df.columns:
        road_length = pd.to_numeric(df["road_length_exact_m"], errors="coerce").fillna(0)
        road_cells = int((road_length > 0).sum())
        no_road_cells = int((road_length <= 0).sum())
    if "network_endpoint_node_count" in df.columns:
        node_count = pd.to_numeric(df["network_endpoint_node_count"], errors="coerce").fillna(0)
        node_cells = int((node_count > 0).sum())
        no_node_cells = int((node_count <= 0).sum())
    if "street_frontage_road_length_m" in df.columns:
        street_road_length = pd.to_numeric(df["street_frontage_road_length_m"], errors="coerce").fillna(0)
        street_frontage_road_cells = int((street_road_length > 0).sum())
    if "street_frontage_building_count" in df.columns:
        street_building_count = pd.to_numeric(df["street_frontage_building_count"], errors="coerce").fillna(0)
        street_frontage_building_cells = int((street_building_count > 0).sum())
    if "street_frontage_continuity_proxy" in df.columns:
        street_frontage_continuity_cells = int(pd.to_numeric(df["street_frontage_continuity_proxy"], errors="coerce").notna().sum())

    missing_rows = family_missingness.copy()
    missing_rows["avg_missing_share"] = missing_rows["avg_missing_share"].map(format_pct)
    missing_rows["max_missing_share"] = missing_rows["max_missing_share"].map(format_pct)
    missing_table_rows = missing_rows.to_dict("records")

    top_missing = (
        column_diag[column_diag["missing_count"] > 0]
        .sort_values(["missing_share", "missing_count"], ascending=False)
        .head(15)
        .copy()
    )
    top_missing["missing_share"] = top_missing["missing_share"].map(format_pct)
    top_missing_rows = top_missing[
        ["column", "family", "missing_count", "missing_share", "role"]
    ].to_dict("records")

    high_corr_count = int((high_corr_pairs["abs_corr"] >= 0.95).sum()) if not high_corr_pairs.empty else 0
    review_corr_count = int((high_corr_pairs["abs_corr"] >= 0.90).sum()) if not high_corr_pairs.empty else 0
    top_corr_rows = (
        high_corr_pairs.head(20)[["feature_a", "feature_b", "corr", "family_a", "family_b", "action"]].to_dict("records")
        if not high_corr_pairs.empty
        else []
    )

    leakage_summary = (
        leakage_screen.groupby(["family", "issue"]).size().reset_index(name="columns").sort_values("columns", ascending=False)
        if not leakage_screen.empty
        else pd.DataFrame(columns=["family", "issue", "columns"])
    )

    shortlist_rows = feature_shortlist[
        ["feature", "branch", "decision", "available", "in_corr95_group", "rationale"]
    ].to_dict("records")

    lines = [
        f"# Feature Diagnostics for {input_csv.stem}",
        "",
        "Date: 2026-06-18",
        "",
        "## Scope",
        "",
        f"Input table: `{input_csv.as_posix()}`",
        "",
        "Purpose: diagnose missingness, grouped collinearity, leakage risk, and the first predictor shortlist before baseline models and CLIMORFA branch training.",
        "",
        "This report is a modelling gate. It does not replace the manual audit or the later spatial cross-validation design.",
        "",
        "## Matrix Check",
        "",
        f"- Rows: {row_count:,}",
        f"- Columns: {column_count:,}",
        f"- Numeric-like columns: {numeric_count:,}",
        f"- Candidate numeric predictor columns after target/QC exclusions: {candidate_count:,}",
        f"- Rows complete across all raw columns: {rows_complete:,} / {row_count:,}",
        f"- Rows with at least one missing raw value: {rows_with_any_missing:,} / {row_count:,}",
        f"- Cells with at least one exact building footprint: {built_cells:,}" if built_cells is not None else "",
        f"- Cells without exact building footprints: {unbuilt_cells:,}" if unbuilt_cells is not None else "",
        f"- `floor_count_aw_mean` missing cells: {floor_structural_missing:,}" if floor_structural_missing is not None else "",
        f"- Cells with exact road length: {road_cells:,}" if road_cells is not None else "",
        f"- Cells without exact road length: {no_road_cells:,}" if no_road_cells is not None else "",
        f"- Cells with endpoint-derived network nodes: {node_cells:,}" if node_cells is not None else "",
        f"- Cells without endpoint-derived network nodes: {no_node_cells:,}" if no_node_cells is not None else "",
        f"- Cells with road-buffer street-frontage context: {street_frontage_road_cells:,}" if street_frontage_road_cells is not None else "",
        f"- Cells with buildings in the street-frontage buffer: {street_frontage_building_cells:,}" if street_frontage_building_cells is not None else "",
        f"- Cells with defined street-frontage continuity proxy: {street_frontage_continuity_cells:,}" if street_frontage_continuity_cells is not None else "",
        "",
        "## Missingness by Feature Family",
        "",
        markdown_table(
            missing_table_rows,
            [
                "family",
                "columns",
                "avg_missing_share",
                "max_missing_share",
                "columns_with_any_missing",
                "near_constant_columns",
            ],
        ),
        "",
        "Top missing columns:",
        "",
        markdown_table(top_missing_rows, ["column", "family", "missing_count", "missing_share", "role"], max_rows=15),
        "",
        "Interpretation:",
        "",
        "- The large building-floor missingness is mostly structural: the same 9,988 cells have no exact building footprint and therefore no area-weighted floor-count statistic. This should be handled with a `has_building` flag plus documented zero/NA strategy, not treated as failed joins.",
        "- Building-footprint morphometric missingness is likewise mostly structural: heterogeneity and compactness fields are undefined in cells without buildings or with too few buildings for variance-style metrics.",
        "- Road/network missingness is also partly structural: orientation entropy is undefined in cells without road length, and mean endpoint degree/dead-end share are undefined in cells without endpoint-derived nodes.",
        "- Street-facing morphology missingness is structural: continuity and road-buffer open-share fields are defined only where the cell contains road length.",
        "- Canopy-height variables have limited NoData gaps and should remain usable with explicit missing handling.",
        "- Weak LCZ gaps should not be imputed into labels. They define where weak-label training is unavailable or needs audit/manual handling.",
        "- `valid_px`, observation-count, and uncertainty-count fields are QA fields, not ordinary morphology predictors.",
        "",
        "## Grouped Collinearity Screen",
        "",
        f"- High-correlation pairs at `abs(r) >= 0.95`: {high_corr_count:,}",
        f"- Review-correlation pairs at `abs(r) >= 0.90`: {review_corr_count:,}",
        f"- Correlation groups at `abs(r) >= 0.95`: {len(collinearity_groups):,}",
        "",
        "Top high-correlation pairs:",
        "",
        markdown_table(top_corr_rows, ["feature_a", "feature_b", "corr", "family_a", "family_b", "action"], max_rows=20),
        "",
        "Action:",
        "",
        "- Treat this as grouped-collinearity screening rather than final VIF. Full-matrix VIF is not appropriate yet because several feature families are intentionally compositional or duplicated for QA.",
        "- For first baselines, choose one representative from each correlation group or run family-level ablations.",
        f"- Detailed groups are in `{(out_dir / 'collinearity_groups_corr95.csv').as_posix()}`.",
        "",
        "## Leakage and Role Screen",
        "",
        markdown_table(leakage_summary.to_dict("records"), ["family", "issue", "columns"], max_rows=20),
        "",
        "Critical exclusions for the first LCZ-like morphology classifier:",
        "",
        "- Exclude all `lcz_*` columns from predictors when LCZ weak label or manual audit label is the target.",
        "- Exclude all `lst_*` fields from LCZ-like typology predictors; use them as climate response, validation, or a secondary supervised target.",
        "- Exclude `row_id`, `col_id`, and ordinary administrative identifiers from predictors. Use them for spatial blocking and leave-district-out validation.",
        "- Use `*_valid_px`, `*_valid_pixels`, observation-count, and uncertainty-count fields as filters/weights/QA only.",
        "- Keep Sentinel/Dynamic World/WorldCover variables only with explicit ablations, because they may overlap with the information used by weak LCZ products.",
        "- Keep OSM variables as proxy functional-context indicators only, and report sensitivity with/without OSM context.",
        "",
        f"Detailed leakage screen: `{(out_dir / 'leakage_screen.csv').as_posix()}`.",
        "",
        "## First Feature-Family Shortlist",
        "",
        markdown_table(shortlist_rows, ["feature", "branch", "decision", "available", "in_corr95_group", "rationale"], max_rows=40),
        "",
        "Recommended first baseline sets:",
        "",
        "1. Morphology-only baseline: exact building coverage/density, floor-count/FAR/height proxies, low/mid/high-rise area shares, surface roughness/range.",
        "2. Morphology + vegetation: add canopy cover, canopy p95, canopy volume per hectare, and NDVI.",
        "3. Morphology + coast/green-blue: add coast distance, water/green shares, NDWI, and selected land-cover shares.",
        "4. Full proxy context: add exact grid road/network metrics, 800 m green-space 2SFCA log1p, OSM proxy context, and remote-sensing semantic probabilities, reported with ablation.",
        "",
        "## Next Work",
        "",
        "1. Exclude the experimental block-permeability proxies from first baselines and refine them with road classes and planar/route-aware topology.",
        "2. Compare 400 m, 800 m, and 1200 m green-space 2SFCA thresholds under identical spatial block CV folds.",
        "3. Treat the IZBB 2023 inventory comparison as partial external triangulation; obtain a current geocoded public-access inventory before neighbourhood-equity claims.",
        "4. Update baseline modelling code to consume the feature-set manifest and separate predictors, labels, validation outcomes, QA fields, and spatial block fields.",
        "5. After Mert fills the manual audit sheet, run audit QA before treating labels as ground truth.",
        "6. Start classical baselines with spatial block CV before training CLIMORFA branches.",
        "",
        "## Rebuild",
        "",
        "```powershell",
        'python src\\diagnostics\\run_feature_diagnostics.py',
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report_md)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    column_diag = build_column_diagnostics(df)
    family_missingness = summarize_family_missingness(column_diag)
    categorical_counts = categorical_value_counts(df, column_diag)
    high_corr_pairs, corr_matrix = build_correlation_pairs(
        df,
        column_diag,
        threshold=args.corr_threshold,
        review_threshold=args.corr_review_threshold,
        min_periods=args.min_corr_periods,
    )
    collinearity_groups = build_collinearity_groups(high_corr_pairs, args.corr_threshold)
    leakage_screen = build_leakage_screen(column_diag)
    feature_shortlist = build_feature_shortlist(column_diag, collinearity_groups)

    column_diag.to_csv(out_dir / "column_diagnostics.csv", index=False)
    family_missingness.to_csv(out_dir / "family_missingness.csv", index=False)
    categorical_counts.to_csv(out_dir / "categorical_value_counts.csv", index=False)
    high_corr_pairs.to_csv(out_dir / "high_correlation_pairs.csv", index=False)
    if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
        corr_matrix.to_csv(out_dir / "candidate_predictor_correlation_matrix.csv")
    collinearity_groups.to_csv(out_dir / "collinearity_groups_corr95.csv", index=False)
    leakage_screen.to_csv(out_dir / "leakage_screen.csv", index=False)
    feature_shortlist.to_csv(out_dir / "feature_shortlist.csv", index=False)

    summary = {
        "input_csv": input_csv.as_posix(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_like_columns": int(column_diag["is_numeric"].sum()),
        "candidate_numeric_predictor_columns": int(len(candidate_numeric_columns(column_diag))),
        "columns_with_any_missing": int((column_diag["missing_count"] > 0).sum()),
        "high_corr_pairs_abs_ge_0_95": int((high_corr_pairs["abs_corr"] >= args.corr_threshold).sum()) if not high_corr_pairs.empty else 0,
        "review_corr_pairs_abs_ge_0_90": int((high_corr_pairs["abs_corr"] >= args.corr_review_threshold).sum()) if not high_corr_pairs.empty else 0,
        "collinearity_groups_abs_ge_0_95": int(len(collinearity_groups)),
        "leakage_screen_columns": int(len(leakage_screen)),
        "outputs": {
            "out_dir": out_dir.as_posix(),
            "report_md": report_path.as_posix(),
        },
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = build_report(
        input_csv=input_csv,
        df=df,
        column_diag=column_diag,
        family_missingness=family_missingness,
        high_corr_pairs=high_corr_pairs,
        collinearity_groups=collinearity_groups,
        leakage_screen=leakage_screen,
        feature_shortlist=feature_shortlist,
        out_dir=out_dir,
    )
    report_path.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

