"""Audited-label validation for the four-class weak-label modeling population.

Restricts the completed manual audit (data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv)
to exactly the population the classical baselines were trained and evaluated
on (lcz_weak_label in {3,6,8,9}, lcz_weak_confidence >= 0.60), then compares
both the weak label itself and each model's out-of-fold prediction against
the audited ground truth. Quality tiers follow the audit QA gate's own
pre-specified analysis plan: label_quality == 'high' is primary, 'high' or
'medium' is a sensitivity check. This is intentionally decided before
looking at the agreement numbers, to avoid post-hoc quality-threshold
selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
MODEL_LABELS = {"lightgbm": "LightGBM", "random_forest": "Random Forest", "xgboost": "XGBoost", "extra_trees": "Extra Trees"}
PRIMARY_RECIPE = "baseline_d_full_proxy_context"
CLASSES = [3, 6, 8, 9]


def _agreement(labels: pd.Series, audit_labels: pd.Series) -> pd.Series:
    mapped = pd.to_numeric(labels, errors="coerce").map(WEAK_TO_AUDIT)
    return mapped == audit_labels


def main(root: Path | str = Path(".")) -> dict:
    root = Path(root)
    out_dir = root / "outputs/diagnostics/audited_model_validation_2026-07-31"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(root / "data/04_training_labels/audit_sample_v1_manual_audit_sheet.csv", dtype={"grid_id": str})
    audit = audit[audit["audit_label"].notna()].copy()

    model_dir = root / "outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix"
    pred = pd.read_csv(model_dir / "out_of_fold_predictions.csv", dtype={"grid_id": str})
    pred = pred[pred["recipe"] == PRIMARY_RECIPE]

    pop = audit[audit["lcz_weak_label"].isin(CLASSES) & (audit["lcz_weak_confidence"] >= 0.60)].copy()
    pop["weak_agreement"] = _agreement(pop["lcz_weak_label"], pop["audit_label"])

    for model_key in MODEL_LABELS:
        model_pred = pred[pred["model"] == model_key].set_index("grid_id")["predicted_label"]
        pop[f"{model_key}_predicted_label"] = pop["grid_id"].map(model_pred)
        pop[f"{model_key}_agreement"] = _agreement(pop[f"{model_key}_predicted_label"], pop["audit_label"])

    pop.to_csv(out_dir / "audited_population_rows.csv", index=False)

    tiers = {"primary_high": pop["label_quality"] == "high", "sensitivity_high_medium": pop["label_quality"].isin(["high", "medium"]), "all_quality": pd.Series(True, index=pop.index)}

    summary_rows = []
    confusion_frames = {}
    for tier_name, mask in tiers.items():
        tier = pop[mask]
        row = {
            "tier": tier_name,
            "n": int(len(tier)),
            "weak_label_agreement": float(tier["weak_agreement"].mean()) if len(tier) else None,
        }
        for model_key, model_label in MODEL_LABELS.items():
            row[f"{model_key}_agreement"] = float(tier[f"{model_key}_agreement"].mean()) if len(tier) else None
        summary_rows.append(row)

        if tier_name == "primary_high":
            weak_ct = pd.crosstab(tier["lcz_weak_label"], tier["audit_label"])
            weak_ct.to_csv(out_dir / "confusion_weak_vs_audit_primary_high.csv")
            confusion_frames["weak"] = weak_ct
            for model_key in MODEL_LABELS:
                model_ct = pd.crosstab(tier[f"{model_key}_predicted_label"], tier["audit_label"])
                model_ct.to_csv(out_dir / f"confusion_{model_key}_vs_audit_primary_high.csv")
                confusion_frames[model_key] = model_ct

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "audited_validation_summary.csv", index=False)

    # Whole-sample (all 569 audited rows, every LCZ class, unrestricted by
    # confidence) context number, from the project's own audit QA gate --
    # broader but less relevant than the tiers above, since most of that
    # sample is easier land-cover classes rather than the hard built-subtype
    # discrimination this paper's model is trained on.
    whole_sample_agreement = None
    qa_path = root / "outputs/diagnostics/manual_audit_qa_2026-06-19/audit_qa_summary.json"
    if qa_path.exists():
        whole_sample_agreement = json.loads(qa_path.read_text(encoding="utf-8")).get("exact_weak_audit_agreement_share")

    manifest = {
        "schema_version": "climorfa.audited_model_validation.v1",
        "date": "2026-07-31",
        "population": "lcz_weak_label in {3,6,8,9} and lcz_weak_confidence >= 0.60 (the exact classical-baselines modeling population)",
        "n_population": int(len(pop)),
        "primary_model": "lightgbm",
        "comparison_model": "random_forest",
        "quality_tiers": {"primary_high": "label_quality == 'high' (pre-specified primary analysis)", "sensitivity_high_medium": "label_quality in {'high','medium'} (pre-specified sensitivity check)", "all_quality": "no quality filter, for reference only"},
        "whole_sample_weak_audit_agreement_all_569_all_classes": whole_sample_agreement,
        "summary_csv": str(out_dir / "audited_validation_summary.csv"),
        "row_level_csv": str(out_dir / "audited_population_rows.csv"),
        "claim_boundary": "n=96 (primary tier) is a small, real, representative-for-its-stratum audited sample; agreement shares are informative but have wide sampling uncertainty at this n and should not be over-read as precise population parameters.",
    }
    (out_dir / "audited_model_validation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(summary_df.to_string(index=False))
    return manifest


if __name__ == "__main__":
    main(Path("."))
