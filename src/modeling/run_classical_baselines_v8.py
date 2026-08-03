"""Run leakage-aware first classical baselines on the CLIMORFA v8 matrix.

This is a weak-label diagnostic, not a final manuscript result. It restricts
the target to sufficiently represented built LCZ classes, uses confidence >=
0.60, and evaluates all recipes on identical stratified spatial block folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/03_processed/grid_250m_model_features_v8.csv")
    parser.add_argument("--feature-manifest", default="configs/climorfa_feature_sets_v1.json")
    parser.add_argument("--out-dir", default="outputs/modeling/classical_baselines_v8_2026-06-18")
    parser.add_argument("--confidence-min", type=float, default=0.60)
    parser.add_argument("--min-class-count", type=int, default=100)
    parser.add_argument("--block-cells", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=300)
    return parser.parse_args()


def ordered_recipe_features(manifest: dict, recipe_name: str) -> list[str]:
    result: list[str] = []
    for feature_set in manifest["baseline_recipes"][recipe_name]:
        for field in manifest["feature_sets"][feature_set]:
            if field not in result:
                result.append(field)
    return result


def model_pipeline(model_name: str, seed: int, trees: int) -> Pipeline:
    if model_name == "dummy_majority":
        estimator = DummyClassifier(strategy="most_frequent")
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    elif model_name == "xgboost":
        # XGBClassifier has no class_weight="balanced" option for multiclass;
        # balance is applied instead via per-sample weights at fit time (see
        # the xgboost branch in main()'s training loop).
        estimator = XGBClassifier(
            n_estimators=trees,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            eval_metric="mlogloss",
        )
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=trees,
            max_depth=-1,
            learning_rate=0.1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
            # "gain" (total split gain per feature) is the closest LightGBM
            # analog to sklearn's impurity-based feature_importances_ used
            # for Random Forest; unlike sklearn's, it is not normalized to
            # sum to 1, so downstream plotting normalizes it explicitly.
            importance_type="gain",
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    manifest = json.loads(Path(args.feature_manifest).read_text(encoding="utf-8"))

    target = pd.to_numeric(df["lcz_weak_label"], errors="coerce")
    confidence = pd.to_numeric(df["lcz_weak_confidence"], errors="coerce").fillna(0)
    built_mask = target.between(1, 10, inclusive="both")
    eligible_counts = target[built_mask & (confidence >= args.confidence_min)].value_counts()
    retained_classes = sorted(int(value) for value, count in eligible_counts.items() if count >= args.min_class_count)
    analysis = df[built_mask & target.isin(retained_classes) & (confidence >= args.confidence_min)].copy()
    analysis["target"] = pd.to_numeric(analysis["lcz_weak_label"], errors="raise").astype(int)
    analysis["spatial_block"] = (
        (pd.to_numeric(analysis["row_id"], errors="raise").astype(int) // args.block_cells).astype(str)
        + "_"
        + (pd.to_numeric(analysis["col_id"], errors="raise").astype(int) // args.block_cells).astype(str)
    )

    y = analysis["target"].to_numpy()
    groups = analysis["spatial_block"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.random_seed)
    split_indices = list(splitter.split(np.zeros(len(analysis)), y, groups))
    fold_assignment = np.full(len(analysis), -1, dtype=int)
    for fold, (_, test_index) in enumerate(split_indices, start=1):
        fold_assignment[test_index] = fold
    if np.any(fold_assignment < 0):
        raise RuntimeError("Spatial fold assignment is incomplete")

    assignment_columns = ["grid_id", "row_id", "col_id", "district", "lcz_weak_label", "lcz_weak_confidence"]
    assignments = analysis[assignment_columns].copy()
    assignments["spatial_block"] = groups
    assignments["fold"] = fold_assignment
    assignments.to_csv(out_dir / "spatial_fold_assignments.csv", index=False)

    recipes = list(manifest["baseline_recipes"])
    recipe_features = {name: ordered_recipe_features(manifest, name) for name in recipes}
    full_features = recipe_features["baseline_d_full_proxy_context"]
    primary_2sfca = "green_2sfca_800m_access_log1p"
    recipe_features["baseline_d_no_green_2sfca"] = [field for field in full_features if field != primary_2sfca]
    for threshold in (400, 1200):
        replacement = f"green_2sfca_{threshold}m_access_log1p"
        recipe_features[f"baseline_d_green_2sfca_{threshold}m"] = [
            replacement if field == primary_2sfca else field for field in full_features
        ]
    missing = {name: [field for field in fields if field not in analysis.columns] for name, fields in recipe_features.items()}
    missing = {name: fields for name, fields in missing.items() if fields}
    if missing:
        raise RuntimeError(f"Missing recipe fields: {missing}")

    fold_metrics: list[dict] = []
    prediction_rows: list[dict] = []
    importance_rows: list[dict] = []
    classes = np.array(retained_classes, dtype=int)
    # XGBoost (unlike sklearn's RF/ET/dummy or LightGBM) requires contiguous
    # 0..n-1 integer labels rather than arbitrary class codes, so its labels
    # are encoded before fit and decoded after predict; `classes` is already
    # sorted, so encoded index i always corresponds to `classes[i]`.
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    jobs = [("dummy_majority", "baseline_a_morphology_only")]
    jobs.extend(("random_forest", recipe) for recipe in recipe_features)
    jobs.extend(("extra_trees", recipe) for recipe in recipe_features)
    jobs.extend(("xgboost", recipe) for recipe in recipe_features)
    jobs.extend(("lightgbm", recipe) for recipe in recipe_features)
    for model_name, recipe_name in jobs:
        features = recipe_features[recipe_name]
        x = analysis[features].apply(pd.to_numeric, errors="coerce")
        for fold, (train_index, test_index) in enumerate(split_indices, start=1):
            pipeline = model_pipeline(model_name, args.random_seed + fold, args.rf_trees)
            fit_kwargs = {}
            y_train = y[train_index]
            if model_name == "xgboost":
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
                y_train = np.array([class_to_idx[v] for v in y_train])
            pipeline.fit(x.iloc[train_index], y_train, **fit_kwargs)
            predicted = pipeline.predict(x.iloc[test_index])
            if model_name == "xgboost":
                predicted = np.array([idx_to_class[int(v)] for v in predicted])
            predicted = predicted.astype(int)
            probabilities = pipeline.predict_proba(x.iloc[test_index])
            if model_name == "xgboost":
                model_classes = classes
            else:
                model_classes = pipeline.named_steps["model"].classes_.astype(int)
            probability_by_class = np.zeros((len(test_index), len(classes)), dtype=float)
            for source_index, class_value in enumerate(model_classes):
                target_index = int(np.where(classes == class_value)[0][0])
                probability_by_class[:, target_index] = probabilities[:, source_index]

            true_values = y[test_index]
            fold_metrics.append(
                {
                    "model": model_name,
                    "recipe": recipe_name,
                    "fold": fold,
                    "train_rows": len(train_index),
                    "test_rows": len(test_index),
                    "test_blocks": len(set(groups[test_index])),
                    "features": len(features),
                    "accuracy": accuracy_score(true_values, predicted),
                    "balanced_accuracy": balanced_accuracy_score(true_values, predicted),
                    "macro_f1": f1_score(true_values, predicted, labels=classes, average="macro", zero_division=0),
                    "weighted_f1": f1_score(true_values, predicted, labels=classes, average="weighted", zero_division=0),
                }
            )
            for local_index, row_index in enumerate(test_index):
                row = {
                    "grid_id": analysis.iloc[row_index]["grid_id"],
                    "model": model_name,
                    "recipe": recipe_name,
                    "fold": fold,
                    "true_label": int(true_values[local_index]),
                    "predicted_label": int(predicted[local_index]),
                    "predicted_probability": float(np.max(probability_by_class[local_index])),
                }
                for class_index, class_value in enumerate(classes):
                    row[f"prob_lcz_{class_value}"] = float(probability_by_class[local_index, class_index])
                prediction_rows.append(row)

            if model_name in ("random_forest", "lightgbm"):
                importances = pipeline.named_steps["model"].feature_importances_
                for feature, importance in zip(features, importances):
                    importance_rows.append(
                        {
                            "model": model_name,
                            "recipe": recipe_name,
                            "fold": fold,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
            print(f"completed={model_name}/{recipe_name}/fold_{fold}", flush=True)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(out_dir / "out_of_fold_predictions.csv", index=False)
    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(out_dir / "feature_importance_by_fold.csv", index=False)

    summary = (
        fold_metrics_df.groupby(["model", "recipe"], as_index=False)
        .agg(
            folds=("fold", "count"),
            features=("features", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_sd=("weighted_f1", "std"),
        )
        .sort_values("macro_f1_mean")
    )
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    reference_recipe = "baseline_d_no_green_2sfca"
    delta_rows: list[dict] = []
    for model_name in ("random_forest", "extra_trees"):
        model_fold = fold_metrics_df[fold_metrics_df["model"] == model_name].pivot(
            index="fold",
            columns="recipe",
            values=["macro_f1", "balanced_accuracy"],
        )
        for comparison_recipe in (
            "baseline_d_green_2sfca_400m",
            "baseline_d_full_proxy_context",
            "baseline_d_green_2sfca_1200m",
        ):
            for fold in model_fold.index:
                delta_rows.append(
                    {
                        "model": model_name,
                        "fold": int(fold),
                        "reference_recipe": reference_recipe,
                        "comparison_recipe": comparison_recipe,
                        "macro_f1_delta": float(
                            model_fold.loc[fold, ("macro_f1", comparison_recipe)]
                            - model_fold.loc[fold, ("macro_f1", reference_recipe)]
                        ),
                        "balanced_accuracy_delta": float(
                            model_fold.loc[fold, ("balanced_accuracy", comparison_recipe)]
                            - model_fold.loc[fold, ("balanced_accuracy", reference_recipe)]
                        ),
                    }
                )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "2sfca_fold_deltas.csv", index=False)
    delta_summary = (
        delta_df.groupby(["model", "reference_recipe", "comparison_recipe"], as_index=False)
        .agg(
            folds=("fold", "count"),
            macro_f1_delta_mean=("macro_f1_delta", "mean"),
            macro_f1_delta_sd=("macro_f1_delta", "std"),
            macro_f1_delta_min=("macro_f1_delta", "min"),
            macro_f1_delta_max=("macro_f1_delta", "max"),
            macro_f1_fold_wins=("macro_f1_delta", lambda values: int((values > 0).sum())),
            balanced_accuracy_delta_mean=("balanced_accuracy_delta", "mean"),
            balanced_accuracy_delta_sd=("balanced_accuracy_delta", "std"),
        )
        .sort_values("macro_f1_delta_mean")
    )
    delta_summary.to_csv(out_dir / "2sfca_delta_summary.csv", index=False)

    pooled_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for (model_name, recipe_name), part in predictions_df.groupby(["model", "recipe"]):
        true_values = part["true_label"].to_numpy(dtype=int)
        predicted = part["predicted_label"].to_numpy(dtype=int)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_values,
            predicted,
            labels=classes,
            zero_division=0,
        )
        for index, class_value in enumerate(classes):
            pooled_rows.append(
                {
                    "model": model_name,
                    "recipe": recipe_name,
                    "class": int(class_value),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
        matrix = confusion_matrix(true_values, predicted, labels=classes)
        for true_index, true_class in enumerate(classes):
            for pred_index, pred_class in enumerate(classes):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "recipe": recipe_name,
                        "true_class": int(true_class),
                        "predicted_class": int(pred_class),
                        "count": int(matrix[true_index, pred_index]),
                    }
                )
    pd.DataFrame(pooled_rows).to_csv(out_dir / "pooled_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "pooled_confusion_matrix_long.csv", index=False)

    importance_summary = (
        importance_df.groupby(["model", "recipe", "feature"], as_index=False)["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_sd"})
        .sort_values(["model", "recipe", "importance_mean"], ascending=[True, True, False])
    )
    importance_summary.to_csv(out_dir / "feature_importance_summary.csv", index=False)

    run_manifest = {
        "schema_version": "climorfa.classical_baselines.v1",
        "input_csv": args.input_csv,
        "feature_manifest": args.feature_manifest,
        "target": "lcz_weak_label",
        "claim_status": "exploratory weak-label diagnostic; not a manuscript result before manual audit",
        "filters": {
            "built_lcz_range": [1, 10],
            "confidence_min": args.confidence_min,
            "min_class_count": args.min_class_count,
            "retained_classes": retained_classes,
            "analysis_rows": len(analysis),
        },
        "validation": {
            "method": "StratifiedGroupKFold",
            "folds": args.folds,
            "spatial_block_cells": args.block_cells,
            "spatial_block_nominal_m": args.block_cells * 250,
            "random_seed": args.random_seed,
            "preprocessing_fit_inside_each_training_fold": True,
        },
        "models": {
            "dummy_majority": {"strategy": "most_frequent"},
            "random_forest": {
                "trees": args.rf_trees,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
            "extra_trees": {
                "trees": args.rf_trees,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced",
            },
            "xgboost": {
                "trees": args.rf_trees,
                "max_depth": 6,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "class_balance": "per-sample weights via compute_sample_weight('balanced')",
            },
            "lightgbm": {
                "trees": args.rf_trees,
                "max_depth": -1,
                "learning_rate": 0.1,
                "class_weight": "balanced",
            },
        },
        "recipes": recipe_features,
        "software": {
            "python": "Python 3.12",
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "outputs": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
