"""First Localized Predictive Congestion Model milestone -- OFFLINE RESEARCH ONLY.

Runs Phases 1-12 of the milestone against the frozen predictive_dataset_campaign_v1
CSV (2,508,480 rows, commit 1574654) and writes a single report JSON plus an
exported model artifact under data/localized_predictive_model_v1/.

NOTHING TRAINED HERE IS WIRED INTO recommendation scoring, exit ranking,
guidance, signage, LiveRuntime, or operator workflow. This script only reads
predictive_dataset's own CSV/report/scenario-metadata files and writes new
files under data/localized_predictive_model_v1/ and docs/architecture/ -- it
never imports or modifies anything under live_system/, building_state/,
recommendation/, guidance/, dynamic_signage/, or ai_registry/'s live-facing
modules.

Usage: python scripts/train_localized_predictive_model_v1.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.baselines import build_baselines
from predictive_model.calibration import evaluate_calibration_methods
from predictive_model.dataset_loader import DatasetRequirement, load_dataset, select_horizon
from predictive_model.error_analysis import build_error_analysis
from predictive_model.feature_importance import builtin_feature_importance, permutation_importance_report
from predictive_model.feature_prep import build_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight, tune_threshold
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.model_export import ModelMetadata, export_model
from predictive_model.sanity_checks import (
    feature_family_ablation_report,
    label_shuffle_test,
    leakage_correlation_recheck,
)
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.tree_models import build_tree_models, library_availability_report

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v1"
CSV_PATH = CAMPAIGN_DIR / "candidate_dataset_v1.csv"
REPORT_PATH = CAMPAIGN_DIR / "campaign_v1_report.json"
SCENARIO_METADATA_PATH = CAMPAIGN_DIR / "scenario_metadata.json"

OUTPUT_DIR = REPO_ROOT / "data" / "localized_predictive_model_v1"

SEED = 20260726
PRIMARY_HORIZON = 20.0
ALL_HORIZONS = (10.0, 20.0, 30.0, 60.0)


def _fresh_model(name: str, seed: int):
    builders = {**build_baselines(seed=seed), **build_tree_models(seed=seed)}
    return builders[name]


def _is_trivial_baseline(name: str) -> bool:
    return name in ("majority_class", "always_negative", "random")


def _prepare_horizon_split(dataset, split, horizon: float):

    horizon_frame = select_horizon(dataset, horizon)
    train_df, val_df, test_df = apply_split(horizon_frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    train_trainable = trainable_rows(train_df)
    val_trainable = trainable_rows(val_df)
    test_trainable = trainable_rows(test_df)

    train_feat = build_feature_matrix(train_trainable)
    val_feat = build_feature_matrix(val_trainable)
    test_feat = build_feature_matrix(test_trainable)

    return horizon_frame, (train_trainable, val_trainable, test_trainable), (train_feat, val_feat, test_feat)


def _run_primary_horizon(horizon: float, horizon_frame, trainable_frames, feats, dataset, split, scenario_metadata) -> Dict[str, Any]:

    train_trainable, val_trainable, test_trainable = trainable_frames
    train_feat, val_feat, test_feat = feats

    class_weight_map = compute_class_weight_map(train_feat.y)
    sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)

    model_results: Dict[str, Any] = {}
    stash: Dict[str, Any] = {}

    all_models = {**build_baselines(seed=SEED), **build_tree_models(seed=SEED)}

    for name, model in all_models.items():

        t_start = time.time()
        trivial = _is_trivial_baseline(name)
        fit_weight = None if trivial else sample_weight

        model.fit(train_feat.X, train_feat.y, sample_weight=fit_weight)

        val_prob = model.predict_proba(val_feat.X)
        test_prob = model.predict_proba(test_feat.X)

        if trivial:
            threshold, _ = 0.5, None
        else:
            threshold, _ = tune_threshold(val_feat.y, val_prob, metric="f1")

        val_metrics = compute_metrics(val_feat.y, val_prob, threshold=threshold)
        test_metrics = compute_metrics(test_feat.y, test_prob, threshold=threshold)
        test_by_type = metrics_by_group(test_feat.y, test_prob, test_feat.candidate_types, threshold=threshold)

        model_results[name] = {
            "threshold": threshold,
            "class_weight_map": class_weight_map if not trivial else None,
            "val_metrics": val_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "test_metrics_by_candidate_type": test_by_type,
            "train_seconds": time.time() - t_start,
        }
        stash[name] = {"model": model, "val_prob": val_prob, "test_prob": test_prob}

        print(f"    [{horizon}s] {name}: test ROC-AUC={test_metrics.roc_auc} PR-AUC={test_metrics.pr_auc} "
              f"({time.time() - t_start:.1f}s)")

    scored_names = [name for name in model_results if model_results[name]["test_metrics"]["pr_auc"] is not None]
    best_name = max(scored_names, key=lambda name: model_results[name]["test_metrics"]["pr_auc"])
    best = stash[best_name]
    best_threshold = model_results[best_name]["threshold"]

    print(f"    [{horizon}s] BEST MODEL: {best_name}")

    # Phase 8 -- feature importance
    importance: Dict[str, Any] = {}
    if hasattr(best["model"], "feature_importances_"):
        importance["builtin"] = builtin_feature_importance(best["model"], train_feat.feature_names)
    else:
        importance["builtin"] = None
        importance["builtin_note"] = f"{best_name} exposes no feature_importances_ (see feature_importance.py docstring)."
    importance["permutation"] = permutation_importance_report(
        best["model"], test_feat.X, test_feat.y, test_feat.feature_names, seed=SEED,
    )

    # Phase 9 -- error analysis
    test_pred = (best["test_prob"] >= best_threshold).astype(int)
    error_analysis = build_error_analysis(
        test_trainable, test_feat.y, test_pred, best["test_prob"], scenario_metadata, horizon_frame,
    )

    # Phase 10 -- calibration
    calibration = evaluate_calibration_methods(val_feat.y, best["val_prob"], test_feat.y, best["test_prob"])

    # Phase 11 -- scientific sanity checks
    def _train_fn(X: np.ndarray, y: np.ndarray):
        fresh = _fresh_model(best_name, SEED)
        if _is_trivial_baseline(best_name):
            fresh.fit(X, y)
        else:
            cw = compute_class_weight_map(y)
            fresh.fit(X, y, sample_weight=sample_weights_from_class_weight(y, cw))
        return fresh

    feature_names = train_feat.feature_names
    families = {
        "demand_signal": [n for n in feature_names if n in ("candidate_queue_length", "candidate_approaching_count")],
        "global_and_adjacent_context": [
            n for n in feature_names
            if n in ("total_active_occupant_count", "candidate_adjacent_zone_occupancy", "candidate_adjacent_zone_occupancy_missing")
        ],
        "structural": [
            n for n in feature_names
            if n.startswith("candidate_type=") or n in ("candidate_capacity", "candidate_walking_distance", "candidate_traversable")
        ],
        "derived_congestion_level": [n for n in feature_names if n.startswith("candidate_congestion_level=")],
    }

    ablation = feature_family_ablation_report(
        _train_fn, train_feat.X, train_feat.y, val_feat.X, val_feat.y, feature_names, families,
    )
    leakage_recheck = leakage_correlation_recheck(train_feat.X, train_feat.y, feature_names)
    shuffle_test = label_shuffle_test(_train_fn, train_feat.X, train_feat.y, val_feat.X, val_feat.y, seed=SEED)

    # Phase 12 -- export the best model
    production_readiness, rationale = _assess_production_readiness(
        model_results[best_name]["test_metrics"], ablation, leakage_recheck, shuffle_test,
    )

    metadata = ModelMetadata(
        model_name=best_name,
        model_library=type(best["model"]).__name__,
        dataset_schema_version=dataset.manifest.schema_version,
        dataset_campaign_version=dataset.manifest.campaign_version,
        dataset_feature_version=dataset.manifest.feature_version,
        dataset_target_version=dataset.manifest.target_version,
        prediction_horizon_seconds=horizon,
        feature_names=feature_names,
        train_scenario_count=len(split.train_scenario_ids),
        val_scenario_count=len(split.val_scenario_ids),
        test_scenario_count=len(split.test_scenario_ids),
        decision_threshold=best_threshold,
        class_weight_strategy="sklearn 'balanced' class weighting via sample_weight (no oversampling)",
        validation_metrics=model_results[best_name]["val_metrics"],
        test_metrics=model_results[best_name]["test_metrics"],
        production_readiness=production_readiness,
        production_readiness_rationale=rationale,
    )
    export_paths = export_model(best["model"], metadata, str(OUTPUT_DIR))

    return {
        "models": model_results,
        "best_model": best_name,
        "feature_importance": importance,
        "error_analysis": error_analysis,
        "calibration": calibration.to_dict(),
        "sanity_checks": {
            "feature_family_ablation": ablation,
            "leakage_recheck": leakage_recheck,
            "label_shuffle_test": shuffle_test,
        },
        "export": export_paths,
        "production_readiness": production_readiness,
        "production_readiness_rationale": rationale,
    }


def _run_secondary_horizon(horizon: float, best_model_name: str, feats) -> Dict[str, Any]:

    train_feat, val_feat, test_feat = feats

    model = _fresh_model(best_model_name, SEED)

    if _is_trivial_baseline(best_model_name):
        model.fit(train_feat.X, train_feat.y)
    else:
        class_weight_map = compute_class_weight_map(train_feat.y)
        sample_weight = sample_weights_from_class_weight(train_feat.y, class_weight_map)
        model.fit(train_feat.X, train_feat.y, sample_weight=sample_weight)

    val_prob = model.predict_proba(val_feat.X)
    test_prob = model.predict_proba(test_feat.X)

    threshold, _ = (0.5, None) if _is_trivial_baseline(best_model_name) else tune_threshold(val_feat.y, val_prob, metric="f1")

    val_metrics = compute_metrics(val_feat.y, val_prob, threshold=threshold)
    test_metrics = compute_metrics(test_feat.y, test_prob, threshold=threshold)
    test_by_type = metrics_by_group(test_feat.y, test_prob, test_feat.candidate_types, threshold=threshold)

    print(f"    [{horizon}s] {best_model_name} (horizon-robustness only): "
          f"test ROC-AUC={test_metrics.roc_auc} PR-AUC={test_metrics.pr_auc}")

    return {
        "model": best_model_name,
        "threshold": threshold,
        "val_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "test_metrics_by_candidate_type": test_by_type,
    }


def _assess_production_readiness(test_metrics, ablation, leakage_recheck, shuffle_test):

    reasons = []

    pr_auc = test_metrics["pr_auc"]
    positive_rate = test_metrics["positive_rate"]

    if leakage_recheck["flagged_for_leakage_review"]:
        return "NOT_READY", f"Leakage review flagged features: {leakage_recheck['flagged_for_leakage_review']}."

    if not shuffle_test["near_chance"]:
        return "NOT_READY", (
            f"Label-shuffle test did not collapse to chance (ROC-AUC="
            f"{shuffle_test['shuffled_label_roc_auc_on_real_val_labels']:.3f}) -- possible leakage channel."
        )

    if pr_auc is None or pr_auc < 2 * positive_rate:
        reasons.append(
            f"PR-AUC ({pr_auc}) is not meaningfully above the positive-rate baseline ({positive_rate:.3f})."
        )

    # Building-topology / total-lockout / stair-1 coverage gaps are
    # disclosed, structural, dataset-level limitations (see
    # docs/architecture/predictive_dataset_campaign_v1.md Sections 10/13)
    # -- a single-building dataset is never "READY" for controlled live
    # integration regardless of how strong its metrics are, since it has
    # not been shown to generalize beyond one fixed topology.
    reasons.append(
        "Dataset covers exactly one building topology (2 doors, 2 exits, 1 stair) with no single-exit "
        "representation and no total-lockout rows -- generalization beyond this fixed topology is unverified."
    )

    if reasons:
        return "PROMISING_BUT_NEEDS_MORE_DATA", " ".join(reasons)

    return "READY", "All sanity checks passed and metrics clear baselines by a wide margin."


def main() -> None:

    overall_start = time.time()

    print("Loading dataset...")
    dataset = load_dataset(str(CSV_PATH), str(REPORT_PATH), requirement=DatasetRequirement())

    with open(SCENARIO_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
        scenario_metadata = json.load(metadata_file)

    print(f"Loaded {len(dataset.frame)} rows, {dataset.frame['scenario_id'].nunique()} scenarios, "
          f"horizons={dataset.available_horizons}")

    all_scenario_ids = dataset.frame["scenario_id"].unique().tolist()
    split = split_scenarios(all_scenario_ids, seed=SEED)
    print(f"Scenario split: train={len(split.train_scenario_ids)} "
          f"val={len(split.val_scenario_ids)} test={len(split.test_scenario_ids)}")

    report: Dict[str, Any] = {
        "dataset_manifest": {
            "schema_version": dataset.manifest.schema_version,
            "campaign_version": dataset.manifest.campaign_version,
            "feature_version": dataset.manifest.feature_version,
            "target_version": dataset.manifest.target_version,
            "csv_path": str(CSV_PATH),
            "row_count": int(len(dataset.frame)),
            "scenario_count": int(dataset.frame["scenario_id"].nunique()),
        },
        "scenario_split": split.to_dict(),
        "library_availability": library_availability_report(),
        "horizons": {},
    }

    horizon_order = (PRIMARY_HORIZON,) + tuple(h for h in ALL_HORIZONS if h != PRIMARY_HORIZON)

    best_model_name_from_primary = None

    for horizon in horizon_order:

        print(f"\n=== Horizon {horizon}s ===")
        horizon_frame, trainable_frames, feats = _prepare_horizon_split(dataset, split, horizon)

        train_feat, val_feat, test_feat = feats
        print(f"    trainable rows: train={len(train_feat.y)} val={len(val_feat.y)} test={len(test_feat.y)} "
              f"(positive rate train={train_feat.y.mean():.3f})")

        if horizon == PRIMARY_HORIZON:
            result = _run_primary_horizon(horizon, horizon_frame, trainable_frames, feats, dataset, split, scenario_metadata)
            best_model_name_from_primary = result["best_model"]
        else:
            result = _run_secondary_horizon(horizon, best_model_name_from_primary, feats)

        report["horizons"][str(horizon)] = result

    # Phase 7 -- horizon robustness summary
    horizon_pr_auc = {
        horizon: report["horizons"][str(horizon)]["test_metrics"]["pr_auc"]
        if "test_metrics" in report["horizons"][str(horizon)]
        else report["horizons"][str(horizon)]["models"][report["horizons"][str(horizon)]["best_model"]]["test_metrics"]["pr_auc"]
        for horizon in ALL_HORIZONS
    }
    best_horizon = max(horizon_pr_auc, key=lambda h: (horizon_pr_auc[h] if horizon_pr_auc[h] is not None else -1))
    report["horizon_robustness"] = {
        "pr_auc_by_horizon": horizon_pr_auc,
        "best_horizon_by_pr_auc": best_horizon,
        "primary_horizon_confirmed_best": best_horizon == PRIMARY_HORIZON,
    }

    report["total_wall_seconds"] = time.time() - overall_start

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Report written to {report_path}")


if __name__ == "__main__":
    main()
