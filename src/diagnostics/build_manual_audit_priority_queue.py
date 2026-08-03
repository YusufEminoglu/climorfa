"""Prioritize the existing 569-cell manual audit without changing its sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from osgeo import ogr


ogr.UseExceptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", default="data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv")
    parser.add_argument("--audit-gpkg", default="data/04_training_labels/audit_sample_v1.gpkg")
    parser.add_argument("--audit-layer", default="audit_sample_v1")
    parser.add_argument("--model-csv", default="data/03_processed/grid_250m_model_features_v8.csv")
    parser.add_argument("--oof-predictions", default="outputs/modeling/classical_baselines_v8_2026-06-18/out_of_fold_predictions.csv")
    parser.add_argument("--out-csv", default="data/04_training_labels/audit_sample_v1_priority_queue.csv")
    parser.add_argument("--out-gpkg", default="data/04_training_labels/audit_sample_v1_priority_queue.gpkg")
    parser.add_argument("--out-layer", default="audit_sample_v1_priority_queue")
    parser.add_argument("--out-manifest", default="data/04_training_labels/audit_sample_v1_priority_queue_manifest.json")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if pd.notna(result) else default


def load_source_layer(path: str, requested_layer: str) -> tuple[ogr.DataSource, ogr.Layer]:
    ds = ogr.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    layer = ds.GetLayerByName(requested_layer)
    if layer is None and ds.GetLayerCount() == 1:
        layer = ds.GetLayerByIndex(0)
    if layer is None:
        raise RuntimeError(f"Could not find {requested_layer} in {path}")
    return ds, layer


def build_priority_rows(audit: pd.DataFrame, model: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    context_fields = [
        "grid_id",
        "coastal_2km_flag",
        "osm_special_infrastructure_flag",
        "green_2sfca_network_available_flag",
        "green_2sfca_800m_access_log1p",
    ]
    context = model[context_fields].copy()

    selected = predictions[
        (predictions["recipe"] == "baseline_d_full_proxy_context")
        & predictions["model"].isin(["random_forest", "extra_trees"])
    ][["grid_id", "model", "predicted_label", "predicted_probability"]].copy()
    wide = selected.pivot(index="grid_id", columns="model", values=["predicted_label", "predicted_probability"])
    wide.columns = [f"{model_name}_{metric}" for metric, model_name in wide.columns]
    wide = wide.reset_index()

    result = audit.merge(context, on="grid_id", how="left", validate="one_to_one")
    result = result.merge(wide, on="grid_id", how="left", validate="one_to_one")
    scores = []
    reasons_list = []
    for _, row in result.iterrows():
        score = 0.0
        reasons: list[str] = []
        confidence = min(max(safe_float(row.get("lcz_weak_confidence")), 0.0), 1.0)
        uncertainty_points = 2.5 * (1.0 - confidence)
        score += uncertainty_points
        if uncertainty_points >= 0.75:
            reasons.append("low_weak_label_confidence")
        if int(safe_float(row.get("lcz_mixed_flag"))) == 1:
            score += 3.0
            reasons.append("weak_label_mixed")
        if int(safe_float(row.get("osm_special_infrastructure_flag"))) == 1:
            score += 2.0
            reasons.append("special_infrastructure_context")
        if int(safe_float(row.get("coastal_2km_flag"))) == 1:
            score += 1.0
            reasons.append("coastal_2km")

        rf_prediction = row.get("random_forest_predicted_label")
        et_prediction = row.get("extra_trees_predicted_label")
        model_available = pd.notna(rf_prediction) and pd.notna(et_prediction)
        if model_available:
            if int(rf_prediction) != int(et_prediction):
                score += 2.0
                reasons.append("rf_extra_trees_disagree")
            weak_label = int(safe_float(row.get("lcz_weak_label"), -1))
            if int(rf_prediction) != weak_label or int(et_prediction) != weak_label:
                score += 1.5
                reasons.append("model_disagrees_with_weak_label")
            minimum_probability = min(
                safe_float(row.get("random_forest_predicted_probability")),
                safe_float(row.get("extra_trees_predicted_probability")),
            )
            score += max(1.0 - minimum_probability, 0.0)
            if minimum_probability < 0.60:
                reasons.append("low_model_confidence")
        else:
            reasons.append("model_gate_not_available")

        scores.append(score)
        reasons_list.append(";".join(reasons))

    result["audit_priority_score"] = scores
    result["audit_priority_reasons"] = reasons_list
    result = result.sort_values(
        ["audit_priority_score", "lcz_weak_confidence", "grid_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    result["audit_priority_rank"] = result.index + 1
    result["audit_priority_pass"] = pd.cut(
        result["audit_priority_rank"],
        bins=[0, 150, 350, len(result)],
        labels=["pass_1_first_150", "pass_2_next_200", "pass_3_remaining"],
        include_lowest=True,
    ).astype(str)
    result["model_gate_available_flag"] = (
        result["random_forest_predicted_label"].notna()
        & result["extra_trees_predicted_label"].notna()
    ).astype(int)
    result["rf_extra_trees_agree_flag"] = (
        result["model_gate_available_flag"].eq(1)
        & result["random_forest_predicted_label"].eq(result["extra_trees_predicted_label"])
    ).astype(int)
    return result


def create_field(name: str, sample: Any) -> ogr.FieldDefn:
    if name in {"audit_priority_reasons", "audit_priority_pass"}:
        field = ogr.FieldDefn(name, ogr.OFTString)
        field.SetWidth(254)
        return field
    if name.endswith("_flag") or name.endswith("_rank") or name.endswith("_label"):
        return ogr.FieldDefn(name, ogr.OFTInteger64)
    return ogr.FieldDefn(name, ogr.OFTReal)


def write_gpkg(
    source_layer: ogr.Layer,
    priority: pd.DataFrame,
    out_path: Path,
    out_layer_name: str,
) -> None:
    added_fields = [
        "audit_priority_rank",
        "audit_priority_score",
        "audit_priority_pass",
        "audit_priority_reasons",
        "coastal_2km_flag",
        "osm_special_infrastructure_flag",
        "green_2sfca_network_available_flag",
        "green_2sfca_800m_access_log1p",
        "model_gate_available_flag",
        "random_forest_predicted_label",
        "random_forest_predicted_probability",
        "extra_trees_predicted_label",
        "extra_trees_predicted_probability",
        "rf_extra_trees_agree_flag",
    ]
    records = priority.set_index("grid_id").to_dict("index")
    driver = ogr.GetDriverByName("GPKG")
    if out_path.exists():
        driver.DeleteDataSource(str(out_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = driver.CreateDataSource(str(out_path))
    out_layer = out_ds.CreateLayer(
        out_layer_name,
        srs=source_layer.GetSpatialRef(),
        geom_type=source_layer.GetGeomType(),
    )
    source_defn = source_layer.GetLayerDefn()
    source_fields = []
    for index in range(source_defn.GetFieldCount()):
        field = source_defn.GetFieldDefn(index)
        source_fields.append(field.GetName())
        out_layer.CreateField(field)
    for field_name in added_fields:
        sample = next((row[field_name] for row in records.values() if pd.notna(row.get(field_name))), None)
        out_layer.CreateField(create_field(field_name, sample))

    out_defn = out_layer.GetLayerDefn()
    source_layer.ResetReading()
    for source_feature in source_layer:
        grid_id = str(source_feature.GetField("grid_id"))
        if grid_id not in records:
            continue
        output = ogr.Feature(out_defn)
        for field_name in source_fields:
            value = source_feature.GetField(field_name)
            if value is not None:
                output.SetField(field_name, value)
        for field_name in added_fields:
            value = records[grid_id].get(field_name)
            if pd.notna(value):
                output.SetField(field_name, value)
        geom = source_feature.GetGeometryRef()
        if geom is not None:
            output.SetGeometry(geom.Clone())
        out_layer.CreateFeature(output)
        output = None
    out_layer.SyncToDisk()
    out_ds = None
    source_layer.ResetReading()


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit_csv, dtype={"grid_id": str})
    model = pd.read_csv(args.model_csv, dtype={"grid_id": str})
    predictions = pd.read_csv(args.oof_predictions, dtype={"grid_id": str})
    priority = build_priority_rows(audit, model, predictions)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    priority.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    source_ds, source_layer = load_source_layer(args.audit_gpkg, args.audit_layer)
    write_gpkg(source_layer, priority, Path(args.out_gpkg), args.out_layer)
    source_ds = None

    first_pass = priority[priority["audit_priority_pass"] == "pass_1_first_150"]
    manifest = {
        "schema_version": "climorfa.manual_audit_priority.v1",
        "purpose": "review order only; does not change the spatially balanced audit sample or statistical inclusion weights",
        "rows": len(priority),
        "passes": priority["audit_priority_pass"].value_counts().to_dict(),
        "model_gate_available_rows": int(priority["model_gate_available_flag"].sum()),
        "model_disagreement_rows": int(
            ((priority["model_gate_available_flag"] == 1) & (priority["rf_extra_trees_agree_flag"] == 0)).sum()
        ),
        "first_pass": {
            "rows": len(first_pass),
            "mixed_rows": int((pd.to_numeric(first_pass["lcz_mixed_flag"], errors="coerce") == 1).sum()),
            "special_infrastructure_rows": int((pd.to_numeric(first_pass["osm_special_infrastructure_flag"], errors="coerce") == 1).sum()),
            "coastal_rows": int((pd.to_numeric(first_pass["coastal_2km_flag"], errors="coerce") == 1).sum()),
            "model_disagreement_rows": int(
                ((first_pass["model_gate_available_flag"] == 1) & (first_pass["rf_extra_trees_agree_flag"] == 0)).sum()
            ),
        },
        "score": {
            "weak_label_uncertainty": "2.5 * (1 - lcz_weak_confidence)",
            "mixed_flag": 3.0,
            "special_infrastructure": 2.0,
            "coastal_2km": 1.0,
            "rf_extra_trees_disagreement": 2.0,
            "either_model_disagrees_with_weak_label": 1.5,
            "model_uncertainty": "1 - min(Random Forest probability, Extra Trees probability)",
        },
        "guardrails": [
            "Do not edit the priority queue as the authoritative audit sheet; write labels to audit_sample_v1_manual_audit_sheet.csv.",
            "Model disagreement is available only for the four-class weak-label diagnostic subset.",
            "Priority affects review order only and must not replace the original stratified sample design or validation weights.",
        ],
        "outputs": {"csv": args.out_csv, "gpkg": args.out_gpkg, "layer": args.out_layer},
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
