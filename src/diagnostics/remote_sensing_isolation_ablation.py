"""Remote-sensing-free vs remote-sensing-only LightGBM isolation ablation.

Recipe D (the full 68-predictor recipe) mixes non-remote-sensing predictors
(morphology, network, street interface, coastal geometry, OSM proxy context,
2SFCA) with Sentinel-2, Dynamic World, and WorldCover fields drawn from the
same general Earth-observation logic used to construct the global LCZ weak
label being predicted. This isolates whether the full model's macro-F1 gain
over morphology-only Recipe A reflects genuine non-remote-sensing
discriminative signal, or mostly recovers the weak label's own generation
logic through those correlated inputs.

Reuses the exact 5 spatial folds already saved for the primary classical
baselines run (outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/
spatial_fold_assignments.csv) and the same LightGBM hyperparameters as
src/modeling/run_classical_baselines_v8.py, so results are directly
comparable to the paper's other recipe-level macro-F1 numbers.

Two LightGBM recipes are built from configs/climorfa_feature_sets_v1.json,
partitioning every one of Recipe D's 68 fields into exactly one of two
disjoint groups (verified programmatically below: 49 + 19 = 68, no overlap,
no field omitted):

- remote_sensing_only: remote_spectral_v1 (Sentinel-2 NDBI/NDVI/NDWI) +
  remote_semantic_ablation_v1 (Dynamic World probabilities + WorldCover
  shares) = 19 fields.
- remote_sensing_free: morphology_core_v1 + vegetation_structure_v1 (minus
  s2_ndvi_mean) + coast_green_blue_v1 (minus its Sentinel-2/Dynamic
  World/WorldCover fields, keeping only coast_min_distance_m,
  coastal_2km_flag, coastal_5km_flag) + osm_proxy_context_ablation_v1 +
  network_exact_v1 + green_space_2sfca_v1 + street_facing_morphology_v1 =
  49 fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

DATA = Path("data/03_processed/grid_250m_model_features_v8.csv")
MANIFEST = Path("configs/climorfa_feature_sets_v1.json")
FOLDS_CSV = Path("outputs/modeling/classical_baselines_v8_2026-07-27_coastline_fix/spatial_fold_assignments.csv")
CLASSES = [3, 6, 8, 9]


def build_recipes(manifest: dict) -> tuple[list[str], list[str]]:
    fs = manifest["feature_sets"]
    remote_sensing_only = list(dict.fromkeys(fs["remote_spectral_v1"] + fs["remote_semantic_ablation_v1"]))
    remote_flagged = set(fs["remote_spectral_v1"]) | set(fs["remote_semantic_ablation_v1"])
    coast_green_blue_kept = [f for f in fs["coast_green_blue_v1"] if f not in remote_flagged]
    vegetation_kept = [f for f in fs["vegetation_structure_v1"] if f not in remote_flagged]
    remote_sensing_free = list(dict.fromkeys(
        fs["morphology_core_v1"]
        + vegetation_kept
        + coast_green_blue_kept
        + fs["osm_proxy_context_ablation_v1"]
        + fs["network_exact_v1"]
        + fs["green_space_2sfca_v1"]
        + fs["street_facing_morphology_v1"]
    ))

    full_fields: set[str] = set()
    for feature_set in manifest["baseline_recipes"]["baseline_d_full_proxy_context"]:
        full_fields.update(fs[feature_set])
    assert set(remote_sensing_only) | set(remote_sensing_free) == full_fields
    assert set(remote_sensing_only).isdisjoint(remote_sensing_free)
    assert len(remote_sensing_only) + len(remote_sensing_free) == len(full_fields) == 68
    return remote_sensing_only, remote_sensing_free


def main(root: Path | str = Path(".")) -> dict:
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    remote_sensing_only, remote_sensing_free = build_recipes(manifest)

    df = pd.read_csv(root / DATA)
    folds = pd.read_csv(root / FOLDS_CSV)[["grid_id", "fold"]]
    analysis = df.merge(folds, on="grid_id", how="inner")
    y = analysis["lcz_weak_label"].astype(int).to_numpy()

    fold_rows = []
    for recipe_name, features in [("remote_sensing_only", remote_sensing_only), ("remote_sensing_free", remote_sensing_free)]:
        x = analysis[features].apply(pd.to_numeric, errors="coerce")
        for fold in sorted(analysis["fold"].unique()):
            train_index = analysis["fold"].to_numpy() != fold
            test_index = analysis["fold"].to_numpy() == fold
            seed = 42 + int(fold)
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMClassifier(
                    n_estimators=300, max_depth=-1, learning_rate=0.1,
                    class_weight="balanced", n_jobs=-1, random_state=seed,
                    verbosity=-1, importance_type="gain",
                )),
            ])
            pipeline.fit(x.loc[train_index], y[train_index])
            predicted = pipeline.predict(x.loc[test_index])
            macro_f1 = f1_score(y[test_index], predicted, labels=CLASSES, average="macro", zero_division=0)
            fold_rows.append({"recipe": recipe_name, "fold": int(fold), "features": len(features), "macro_f1": macro_f1})

    out_dir = root / "outputs/diagnostics/remote_sensing_isolation_ablation_2026-08-13"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "leakage_isolation_fold_metrics.csv", index=False)
    summary_df = fold_df.groupby("recipe", as_index=False).agg(
        folds=("fold", "count"), features=("features", "first"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
    )
    summary_df.to_csv(out_dir / "leakage_isolation_summary.csv", index=False)

    manifest_out = {
        "schema_version": "climorfa.remote_sensing_isolation_ablation.v1",
        "date": "2026-08-13",
        "purpose": "Isolate remote-sensing-derived (Sentinel-2/Dynamic World/WorldCover) predictor contribution from non-remote-sensing predictors within Recipe D.",
        "folds_source": str(FOLDS_CSV),
        "model": "lightgbm",
        "hyperparameters": {"n_estimators": 300, "max_depth": -1, "learning_rate": 0.1, "class_weight": "balanced", "num_leaves": "library default (31)"},
        "n_remote_sensing_only": len(remote_sensing_only),
        "n_remote_sensing_free": len(remote_sensing_free),
        "remote_sensing_only_fields": remote_sensing_only,
        "remote_sensing_free_fields": remote_sensing_free,
        "summary_csv": str(out_dir / "leakage_isolation_summary.csv"),
        "fold_metrics_csv": str(out_dir / "leakage_isolation_fold_metrics.csv"),
    }
    (out_dir / "leakage_isolation_manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_df.to_string(index=False))
    return manifest_out


if __name__ == "__main__":
    main(Path("."))
