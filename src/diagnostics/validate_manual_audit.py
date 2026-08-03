"""Validate Mert's manual LCZ audit sheet and emit a readiness gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import ogr


ogr.UseExceptions()

AUDIT_FIELDS = [
    "audit_label",
    "label_quality",
    "dominant_fabric_note",
    "dsm_note",
    "vegetation_note",
    "network_note",
    "audit_source_layers",
    "audit_reviewer",
    "audit_date",
]

FIXED_FIELDS = [
    "grid_id",
    "district",
    "built_intensity_bin",
    "building_count_prelim",
    "building_coverage_prelim",
    "road_density_prelim_m_per_km2",
    "lcz_weak_label",
    "lcz_weak_confidence",
    "lcz_probability_mean",
    "lcz_mixed_flag",
    "sample_stratum",
    "sample_role",
]

ALLOWED_QUALITY = {"high", "medium", "low", "reject_edge"}
WEAK_TO_AUDIT = {
    1: "1_compact_highrise",
    2: "2_compact_midrise",
    3: "3_compact_lowrise",
    4: "4_open_highrise",
    5: "5_open_midrise",
    6: "6_open_lowrise",
    7: "7_lightweight_lowrise",
    8: "8_large_lowrise",
    9: "9_sparsely_built",
    10: "10_heavy_industry",
    11: "A_dense_trees",
    12: "B_scattered_trees",
    13: "C_bush_scrub",
    14: "D_low_plants",
    15: "E_bare_rock_paved",
    16: "F_bare_soil_sand",
    17: "G_water",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", default="data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv")
    parser.add_argument("--audit-gpkg", default="data/04_training_labels/audit_sample_v1.gpkg")
    parser.add_argument("--audit-layer", default="audit_sample_v1")
    parser.add_argument("--codebook-csv", default="data/04_training_labels/audit_label_codebook.csv")
    parser.add_argument("--out-dir", default="outputs/diagnostics/manual_audit_qa_2026-06-19")
    return parser.parse_args()


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def read_gpkg_attributes(path: str, layer_name: str) -> pd.DataFrame:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        available = [ds.GetLayerByIndex(index).GetName() for index in range(ds.GetLayerCount())]
        if len(available) == 1:
            layer = ds.GetLayerByIndex(0)
        else:
            raise RuntimeError(f"Layer {layer_name} not found; available={available}")
    rows = []
    definition = layer.GetLayerDefn()
    fields = [definition.GetFieldDefn(index).GetName() for index in range(definition.GetFieldCount())]
    layer.ResetReading()
    for feature in layer:
        rows.append({field: feature.GetField(field) for field in fields})
    layer.ResetReading()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv, dtype={"grid_id": str})
    codebook = pd.read_csv(args.codebook_csv)
    source = read_gpkg_attributes(args.audit_gpkg, args.audit_layer)
    source["grid_id"] = source["grid_id"].astype(str)

    allowed_labels = set(codebook["audit_label"].astype(str))
    missing_columns = [field for field in [*FIXED_FIELDS, *AUDIT_FIELDS] if field not in audit.columns]
    issues: list[dict] = []

    def add_issue(grid_id: str, field: str, severity: str, issue: str, value: object = "") -> None:
        issues.append(
            {
                "grid_id": grid_id,
                "field": field,
                "severity": severity,
                "issue": issue,
                "value": "" if pd.isna(value) else value,
            }
        )

    if missing_columns:
        for field in missing_columns:
            add_issue("", field, "error", "required_column_missing")
    if audit["grid_id"].duplicated().any():
        for grid_id in audit.loc[audit["grid_id"].duplicated(keep=False), "grid_id"]:
            add_issue(grid_id, "grid_id", "error", "duplicate_grid_id", grid_id)

    csv_ids = set(audit["grid_id"])
    source_ids = set(source["grid_id"])
    for grid_id in sorted(csv_ids - source_ids):
        add_issue(grid_id, "grid_id", "error", "grid_id_not_in_source_gpkg", grid_id)
    for grid_id in sorted(source_ids - csv_ids):
        add_issue(grid_id, "grid_id", "error", "source_grid_id_missing_from_csv", grid_id)

    if not missing_columns:
        audit_indexed = audit.set_index("grid_id")
        source_indexed = source.set_index("grid_id")
        comparable_ids = sorted(csv_ids & source_ids)
        numeric_fixed_fields = {
            "building_count_prelim",
            "building_coverage_prelim",
            "road_density_prelim_m_per_km2",
            "lcz_weak_label",
            "lcz_weak_confidence",
            "lcz_probability_mean",
            "lcz_mixed_flag",
        }
        for field in FIXED_FIELDS[1:]:
            if field not in source_indexed.columns:
                add_issue("", field, "warning", "fixed_field_not_available_in_source_gpkg")
                continue
            if field in numeric_fixed_fields:
                left_numeric = pd.to_numeric(audit_indexed.loc[comparable_ids, field], errors="coerce")
                right_numeric = pd.to_numeric(source_indexed.loc[comparable_ids, field], errors="coerce")
                equal = np.isclose(left_numeric, right_numeric, rtol=1e-9, atol=1e-9, equal_nan=True)
                changed = pd.Series(~equal, index=comparable_ids)
            else:
                left = normalized_text(audit_indexed.loc[comparable_ids, field])
                right = normalized_text(source_indexed.loc[comparable_ids, field])
                changed = left != right
            for grid_id in changed[changed].index:
                add_issue(grid_id, field, "error", "fixed_source_field_changed", audit_indexed.loc[grid_id, field])

        label = normalized_text(audit["audit_label"])
        quality = normalized_text(audit["label_quality"])
        completed = label.ne("")
        any_audit_content = audit[AUDIT_FIELDS].apply(normalized_text).ne("").any(axis=1)

        for index, row in audit.iterrows():
            grid_id = str(row["grid_id"])
            row_label = label.iloc[index]
            row_quality = quality.iloc[index]
            if any_audit_content.iloc[index] and not completed.iloc[index]:
                add_issue(grid_id, "audit_label", "error", "partial_row_without_audit_label")
            if completed.iloc[index] and row_label not in allowed_labels:
                add_issue(grid_id, "audit_label", "error", "invalid_audit_label", row_label)
            if completed.iloc[index] and row_quality not in ALLOWED_QUALITY:
                add_issue(grid_id, "label_quality", "error", "invalid_or_missing_label_quality", row_quality)
            if completed.iloc[index]:
                for required_field in ("dominant_fabric_note", "audit_source_layers", "audit_reviewer", "audit_date"):
                    if not str(row.get(required_field, "") if not pd.isna(row.get(required_field, "")) else "").strip():
                        add_issue(grid_id, required_field, "error", "required_audit_field_missing")
                date_value = str(row.get("audit_date", "") if not pd.isna(row.get("audit_date", "")) else "").strip()
                if date_value and pd.isna(pd.to_datetime(date_value, format="%Y-%m-%d", errors="coerce")):
                    add_issue(grid_id, "audit_date", "error", "date_not_iso_yyyy_mm_dd", date_value)
                if row_label == "edge_insufficient" and row_quality != "reject_edge":
                    add_issue(grid_id, "label_quality", "error", "edge_insufficient_requires_reject_edge", row_quality)
                if row_quality == "reject_edge" and row_label != "edge_insufficient":
                    add_issue(grid_id, "audit_label", "warning", "reject_edge_quality_with_non_edge_label", row_label)
                if row_label in {"mixed", "uncertain"} and row_quality == "high":
                    add_issue(grid_id, "label_quality", "warning", "ambiguous_label_marked_high_quality", row_quality)
    else:
        completed = pd.Series(False, index=audit.index)
        label = pd.Series("", index=audit.index)
        quality = pd.Series("", index=audit.index)

    issues_df = pd.DataFrame(issues, columns=["grid_id", "field", "severity", "issue", "value"])
    issues_df.to_csv(out_dir / "audit_issues.csv", index=False)

    completed_count = int(completed.sum())
    error_count = int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0
    warning_count = int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0
    if completed_count == 0:
        status = "not_started"
    elif completed_count < len(audit):
        status = "partial_with_errors" if error_count else "partial"
    else:
        status = "complete_with_errors" if error_count else "ready"

    valid_completed = completed & label.isin(allowed_labels) & quality.isin(ALLOWED_QUALITY)
    agreement_rows = audit.loc[valid_completed, ["grid_id", "lcz_weak_label", "audit_label", "label_quality"]].copy()
    agreement_rows["weak_label_mapped"] = pd.to_numeric(agreement_rows["lcz_weak_label"], errors="coerce").map(WEAK_TO_AUDIT)
    agreement_rows["exact_agreement"] = agreement_rows["weak_label_mapped"] == agreement_rows["audit_label"]
    agreement_rows.to_csv(out_dir / "weak_vs_audit_rows.csv", index=False)
    if not agreement_rows.empty:
        pd.crosstab(agreement_rows["audit_label"], agreement_rows["weak_label_mapped"], dropna=False).to_csv(
            out_dir / "weak_vs_audit_confusion.csv"
        )

    summary = {
        "schema_version": "climorfa.manual_audit_qa.v1",
        "status": status,
        "audit_csv": args.audit_csv,
        "audit_source_gpkg": args.audit_gpkg,
        "rows": len(audit),
        "completed_rows": completed_count,
        "completion_share": completed_count / len(audit) if len(audit) else 0.0,
        "valid_completed_rows": int(valid_completed.sum()),
        "high_quality_rows": int((quality == "high").sum()),
        "medium_quality_rows": int((quality == "medium").sum()),
        "low_quality_rows": int((quality == "low").sum()),
        "reject_edge_rows": int((quality == "reject_edge").sum()),
        "errors": error_count,
        "warnings": warning_count,
        "exact_weak_audit_agreement_share": (
            float(agreement_rows["exact_agreement"].mean()) if not agreement_rows.empty else None
        ),
        "ready_for_audited_model_validation": status == "ready" and int((quality == "high").sum()) > 0,
        "next_action": (
            "Mert must complete audit_label, label_quality, required notes/source/reviewer/date fields."
            if completed_count == 0
            else "Resolve audit_issues.csv errors and complete remaining rows."
            if status != "ready"
            else "Run audited-label model validation using high quality as primary and high+medium as sensitivity."
        ),
    }
    (out_dir / "audit_qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Manual Audit QA Gate",
        "",
        "Date: 2026-06-19",
        "",
        f"- Status: `{status}`",
        f"- Rows: {len(audit):,}",
        f"- Completed: {completed_count:,} ({summary['completion_share']:.1%})",
        f"- Errors: {error_count:,}",
        f"- Warnings: {warning_count:,}",
        f"- Ready for audited model validation: `{str(summary['ready_for_audited_model_validation']).lower()}`",
        "",
        "## Decision",
        "",
        summary["next_action"],
        "",
        "The weak-label baseline outputs remain exploratory until this gate passes.",
        "",
        "## Outputs",
        "",
        "- `audit_issues.csv`",
        "- `audit_qa_summary.json`",
        "- `weak_vs_audit_rows.csv`",
        "- `weak_vs_audit_confusion.csv` when valid completed rows exist",
    ]
    (out_dir / "manual_audit_qa_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
