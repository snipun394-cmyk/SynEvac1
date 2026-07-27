"""Localized Predictive Model V2.2 milestone -- OFFLINE RESEARCH ONLY.

Trains/evaluates Model V2.2 against the full-scale V2.1 experimental
schema (V2's own 9 fields + the 3 V2.1 candidate-local/structural
fields: candidate_recent_flow_rate, candidate_congestion_trend,
candidate_alternative_route_count), reproduced at V2's original 2,500-
scenario scale (data/predictive_dataset_campaign_v2_2/, gitignored,
regenerable via scripts/run_predictive_dataset_campaign_v2_2_fullscale.py).

Reuses almost all of predictive_model/* unchanged from the V2 script
(same scenario-split convention, same model zoo, same imbalance/
calibration strategy, same sanity checks, same topology-holdout/
operational-slices machinery built for V2/V2.1) -- the only structural
difference is feature_prep_v2_1.build_experimental_feature_matrix
instead of feature_prep.build_feature_matrix. Every deliberate
methodology choice already justified in scripts/
train_localized_predictive_model_v2.py's own docstring (RandomForest
max_depth/n_jobs, memory watchdog, chunked loading) is reused verbatim
here for the same resource reasons.

NOTHING TRAINED HERE IS WIRED INTO recommendation scoring, exit
ranking, guidance, signage, LiveRuntime, or operator workflow.

Usage: python scripts/train_localized_predictive_model_v2_2.py
"""

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictive_model.baselines import build_baselines
from predictive_model.calibration import IsotonicCalibrator, evaluate_calibration_methods
from predictive_model.error_analysis import build_error_analysis
from predictive_model.feature_importance import builtin_feature_importance, permutation_importance_report
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix, trainable_rows
from predictive_model.imbalance import compute_class_weight_map, sample_weights_from_class_weight, tune_threshold
from predictive_model.metrics import compute_metrics, metrics_by_group
from predictive_model.model_export import ModelMetadata, export_model
from predictive_model.model_export_v2 import ExtendedModelMetadataV2, export_calibrator, write_metadata_v2
from predictive_model.operational_slices import build_operational_slice_report
from predictive_model.sanity_checks import (
    feature_family_ablation_report,
    label_shuffle_test,
    leakage_correlation_recheck,
)
from predictive_model.scenario_split import apply_split, assert_no_scenario_overlap, split_scenarios
from predictive_model.topology_holdout import (
    apply_topology_holdout,
    assert_no_holdout_overlap,
    build_topology_holdout_splits,
)
from predictive_model.training_size_study import training_size_study
from predictive_model.tree_models import build_tree_models, library_availability_report

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "predictive_dataset_campaign_v2_2"
CSV_PATH = DATA_DIR / "candidate_dataset_experimental.csv"
SCENARIO_METADATA_PATH = DATA_DIR / "scenario_metadata.json"

OUTPUT_DIR = REPO_ROOT / "data" / "localized_predictive_model_v2_2"

SEED = 20260726
PRIMARY_HORIZON = 20.0

N_JOBS = 2
RANDOM_FOREST_MAX_DEPTH = 20
MIN_AVAILABLE_MEMORY_BYTES = 300_000_000
WATCHDOG_CRITICAL_BYTES = 180_000_000
WATCHDOG_POLL_SECONDS = 4.0

_watchdog_log_path = None

_COMPACT_DTYPES = {
    "observation_time": "float32",
    "total_active_occupant_count": "int32",
    "candidate_capacity": "float32",
    "candidate_walking_distance": "float32",
    "candidate_traversable": "bool",
    "candidate_adjacent_zone_occupancy": "float32",
    "candidate_queue_length": "float32",
    "candidate_approaching_count": "float32",
    "candidate_recent_flow_rate": "float32",
    "candidate_alternative_route_count": "int32",
    "currently_congested": "bool",
    "had_any_activity_in_window": "bool",
}


def _mem_log(label: str) -> Dict[str, float]:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / 1e6
    available_mb = vm.available / 1e6
    print(f"    [mem] {label}: available={available_mb:.0f}MB ({vm.percent:.0f}% used), process_rss={rss_mb:.0f}MB", flush=True)
    if vm.available < MIN_AVAILABLE_MEMORY_BYTES:
        raise MemoryError(f"Available system memory critically low ({available_mb:.0f}MB) at step {label!r}.")
    return {"label": label, "available_mb": available_mb, "process_rss_mb": rss_mb, "percent_used": vm.percent}


def _memory_watchdog() -> None:
    while True:
        try:
            vm = psutil.virtual_memory()
            with open(_watchdog_log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.time():.1f} available_mb={vm.available/1e6:.0f} percent={vm.percent:.0f}\n")
            if vm.available < WATCHDOG_CRITICAL_BYTES:
                with open(_watchdog_log_path, "a", encoding="utf-8") as f:
                    f.write(f"CRITICAL: available memory {vm.available/1e6:.0f}MB < floor -- hard-exiting.\n")
                os._exit(137)
        except Exception:
            pass
        time.sleep(WATCHDOG_POLL_SECONDS)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _load_dataset_chunked(csv_path: Path, chunksize: int = 250_000) -> pd.DataFrame:

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.astype({k: v for k, v in _COMPACT_DTYPES.items() if k in chunk.columns})
        chunks.append(chunk)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    for column in ("scenario_id", "candidate_id", "candidate_type", "candidate_congestion_level", "candidate_congestion_trend"):
        if column in frame.columns:
            frame[column] = frame[column].astype("category")

    return frame


def _fresh_model(name: str, seed: int = SEED):
    builders = {
        **build_baselines(seed=seed),
        **build_tree_models(seed=seed, n_jobs=N_JOBS, random_forest_max_depth=RANDOM_FOREST_MAX_DEPTH),
    }
    return builders[name]


def _is_trivial_baseline(name: str) -> bool:
    return name in ("majority_class", "always_negative", "random")


def _fit_with_weight(model, X, y, name):
    if _is_trivial_baseline(name):
        model.fit(X, y)
        return model
    class_weight_map = compute_class_weight_map(y)
    sample_weight = sample_weights_from_class_weight(y, class_weight_map)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def main() -> None:

    global _watchdog_log_path

    overall_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _watchdog_log_path = str(OUTPUT_DIR / "memory_watchdog.log")
    with open(_watchdog_log_path, "w", encoding="utf-8") as f:
        f.write(f"watchdog started {time.time():.1f}\n")
    threading.Thread(target=_memory_watchdog, daemon=True).start()

    memory_log = [_mem_log("start")]

    print("Loading scenario_metadata.json...")
    with open(SCENARIO_METADATA_PATH, "r", encoding="utf-8") as f:
        scenario_metadata = json.load(f)

    print("Loading dataset (chunked)...")
    t0 = time.time()
    frame = _load_dataset_chunked(CSV_PATH)
    load_seconds = time.time() - t0
    print(f"    Loaded {len(frame)} rows, {frame['scenario_id'].nunique()} scenarios in {load_seconds:.1f}s")
    memory_log.append(_mem_log("after dataset load"))

    all_scenario_ids = frame["scenario_id"].astype(str).unique().tolist()
    split = split_scenarios(all_scenario_ids, seed=SEED)
    print(f"Scenario split: train={len(split.train_scenario_ids)} val={len(split.val_scenario_ids)} "
          f"test={len(split.test_scenario_ids)}")

    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    train_trainable = trainable_rows(train_df)
    val_trainable = trainable_rows(val_df)
    test_trainable = trainable_rows(test_df)

    train_feat = build_experimental_feature_matrix(train_trainable)
    val_feat = build_experimental_feature_matrix(val_trainable)
    test_feat = build_experimental_feature_matrix(test_trainable)

    print(f"Trainable rows: train={len(train_feat.y)} val={len(val_feat.y)} test={len(test_feat.y)} "
          f"(train positive rate={train_feat.y.mean():.4f})")
    memory_log.append(_mem_log("after feature matrices"))

    report: Dict[str, Any] = {
        "dataset_manifest": {
            "csv_path": str(CSV_PATH),
            "horizon_seconds": PRIMARY_HORIZON,
            "row_count": int(len(frame)),
            "scenario_count": int(frame["scenario_id"].nunique()),
            "load_seconds": load_seconds,
            "feature_schema": "V2 (9 fields) + V2.1 experimental (3 fields) = 12 fields",
        },
        "scenario_split": split.to_dict(),
        "library_availability": library_availability_report(),
        "n_jobs_used": N_JOBS,
        "random_forest_max_depth_used": RANDOM_FOREST_MAX_DEPTH,
        "trainable_row_counts": {"train": int(len(train_feat.y)), "val": int(len(val_feat.y)), "test": int(len(test_feat.y))},
    }

    # =====================================================
    # Model candidates + primary 20s evaluation
    # =====================================================

    model_results: Dict[str, Any] = {}
    stash: Dict[str, Any] = {}

    all_models = {
        **build_baselines(seed=SEED),
        **build_tree_models(seed=SEED, n_jobs=N_JOBS, random_forest_max_depth=RANDOM_FOREST_MAX_DEPTH),
    }

    for name, model in all_models.items():

        t_start = time.time()
        trivial = _is_trivial_baseline(name)

        _fit_with_weight(model, train_feat.X, train_feat.y, name)

        val_prob = model.predict_proba(val_feat.X)
        test_prob = model.predict_proba(test_feat.X)

        threshold = 0.5 if trivial else tune_threshold(val_feat.y, val_prob, metric="f1")[0]

        val_metrics = compute_metrics(val_feat.y, val_prob, threshold=threshold)
        test_metrics = compute_metrics(test_feat.y, test_prob, threshold=threshold)
        test_by_type = metrics_by_group(test_feat.y, test_prob, test_feat.candidate_types, threshold=threshold)

        model_results[name] = {
            "threshold": threshold,
            "val_metrics": val_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "test_metrics_by_candidate_type": test_by_type,
            "train_seconds": time.time() - t_start,
        }
        stash[name] = {"model": model, "val_prob": val_prob, "test_prob": test_prob}

        print(f"    {name}: test ROC-AUC={test_metrics.roc_auc} PR-AUC={test_metrics.pr_auc} "
              f"({time.time() - t_start:.1f}s)")
        memory_log.append(_mem_log(f"after fitting {name}"))

    scored_names = [name for name in model_results if model_results[name]["test_metrics"]["pr_auc"] is not None]
    best_name = max(scored_names, key=lambda name: model_results[name]["test_metrics"]["pr_auc"])
    best = stash[best_name]
    best_threshold = model_results[best_name]["threshold"]
    print(f"BEST MODEL: {best_name}")

    for name in list(stash.keys()):
        if name != best_name:
            stash[name]["model"] = None
    gc.collect()
    memory_log.append(_mem_log("after freeing non-winning models"))

    report["models"] = model_results
    report["best_model"] = best_name

    # =====================================================
    # Feature importance
    # =====================================================

    importance: Dict[str, Any] = {}
    if hasattr(best["model"], "feature_importances_"):
        importance["builtin"] = builtin_feature_importance(best["model"], train_feat.feature_names)
    else:
        importance["builtin"] = None
        importance["builtin_note"] = f"{best_name} exposes no feature_importances_."
    importance["permutation"] = permutation_importance_report(
        best["model"], test_feat.X, test_feat.y, test_feat.feature_names, seed=SEED,
    )
    report["feature_importance"] = importance
    print("Feature importance computed.")
    memory_log.append(_mem_log("after feature importance"))

    # =====================================================
    # Error analysis + operational slices
    # =====================================================

    test_pred = (best["test_prob"] >= best_threshold).astype(int)
    report["error_analysis"] = build_error_analysis(
        test_trainable, test_feat.y, test_pred, best["test_prob"], scenario_metadata, frame,
    )
    print("Error analysis computed.")

    report["operational_slices"] = build_operational_slice_report(
        test_trainable, test_feat.y, best["test_prob"], best_threshold, scenario_metadata, frame,
    )
    print("Operational slices computed.")
    memory_log.append(_mem_log("after operational slices"))

    # =====================================================
    # Calibration
    # =====================================================

    calibration = evaluate_calibration_methods(val_feat.y, best["val_prob"], test_feat.y, best["test_prob"])
    report["calibration"] = calibration.to_dict()
    print(f"Calibration: recommended={calibration.recommended_method}")

    isotonic = IsotonicCalibrator().fit(best["val_prob"], val_feat.y)

    # =====================================================
    # Ablation + leakage/sanity
    # =====================================================

    def _train_fn(X: np.ndarray, y: np.ndarray):
        fresh = _fresh_model(best_name)
        return _fit_with_weight(fresh, X, y, best_name)

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
        "v2_1_flow_and_trend": [
            n for n in feature_names
            if n == "candidate_recent_flow_rate" or n.startswith("candidate_congestion_trend=")
        ],
        "v2_1_alternative_route_structure": [n for n in feature_names if n == "candidate_alternative_route_count"],
    }

    ablation = feature_family_ablation_report(
        _train_fn, train_feat.X, train_feat.y, val_feat.X, val_feat.y, feature_names, families,
    )
    report["ablation"] = ablation
    print("Ablation computed.")
    memory_log.append(_mem_log("after ablation"))

    leakage_recheck = leakage_correlation_recheck(train_feat.X, train_feat.y, feature_names)
    shuffle_test = label_shuffle_test(_train_fn, train_feat.X, train_feat.y, val_feat.X, val_feat.y, seed=SEED)
    report["sanity_checks"] = {
        "feature_family_ablation_families": families,
        "leakage_recheck": leakage_recheck,
        "label_shuffle_test": shuffle_test,
    }
    print(f"Leakage recheck flagged: {leakage_recheck['flagged_for_leakage_review']}")
    print(f"Label shuffle test near chance: {shuffle_test['near_chance']} (ROC-AUC={shuffle_test['shuffled_label_roc_auc_on_real_val_labels']:.3f})")
    memory_log.append(_mem_log("after leakage/sanity checks"))

    # =====================================================
    # Training-size study
    # =====================================================

    def _model_factory():
        return _fresh_model(best_name)

    size_study = training_size_study(
        train_trainable, val_feat, _model_factory, seed=SEED, feature_builder=build_experimental_feature_matrix,
    )
    report["training_size_study"] = size_study
    for key, result in size_study.items():
        print(f"    size_study[{key}]: scenarios={result['train_scenario_count']} rows={result['train_row_count']} "
              f"val_pr_auc={result['val_pr_auc']}")
    memory_log.append(_mem_log("after training-size study"))

    # =====================================================
    # Leave-one-topology-family-out
    # =====================================================

    holdout_splits = build_topology_holdout_splits(scenario_metadata)
    topology_results: Dict[str, Any] = {}

    for holdout in holdout_splits:

        t_start = time.time()
        holdout_train_df, holdout_test_df = apply_topology_holdout(frame, holdout)
        assert_no_holdout_overlap(holdout, holdout_train_df, holdout_test_df)

        holdout_train_trainable = trainable_rows(holdout_train_df)
        holdout_test_trainable = trainable_rows(holdout_test_df)

        holdout_train_feat = build_experimental_feature_matrix(holdout_train_trainable)
        holdout_test_feat = build_experimental_feature_matrix(holdout_test_trainable)

        holdout_model = _fresh_model(best_name)
        _fit_with_weight(holdout_model, holdout_train_feat.X, holdout_train_feat.y, best_name)

        holdout_prob = holdout_model.predict_proba(holdout_test_feat.X)
        holdout_metrics = compute_metrics(holdout_test_feat.y, holdout_prob, threshold=best_threshold)
        holdout_by_type = metrics_by_group(
            holdout_test_feat.y, holdout_prob, holdout_test_feat.candidate_types, threshold=best_threshold,
        )
        holdout_ece = evaluate_calibration_methods(
            holdout_test_feat.y, holdout_prob, holdout_test_feat.y, holdout_prob,
        ).before["expected_calibration_error"]

        topology_results[holdout.held_out_family] = {
            "train_scenario_count": len(holdout.train_scenario_ids),
            "test_scenario_count": len(holdout.test_scenario_ids),
            "test_row_count": int(len(holdout_test_feat.y)),
            "test_positive_rate": float(holdout_test_feat.y.mean()),
            "test_metrics": holdout_metrics.to_dict(),
            "test_metrics_by_candidate_type": holdout_by_type,
            "expected_calibration_error_raw": holdout_ece,
            "train_seconds": time.time() - t_start,
        }

        print(f"    topology_holdout[{holdout.held_out_family}]: ROC-AUC={holdout_metrics.roc_auc} "
              f"PR-AUC={holdout_metrics.pr_auc} ({time.time() - t_start:.1f}s)")

        del holdout_model, holdout_train_feat, holdout_test_feat, holdout_train_trainable, holdout_test_trainable
        gc.collect()
        memory_log.append(_mem_log(f"after topology holdout [{holdout.held_out_family}]"))

    report["topology_holdout"] = topology_results

    # =====================================================
    # Export
    # =====================================================

    production_readiness, rationale = _assess_production_readiness(
        model_results[best_name]["test_metrics"], leakage_recheck, shuffle_test, topology_results,
    )

    metadata = ModelMetadata(
        model_name=best_name,
        model_library=type(best["model"]).__name__,
        dataset_schema_version="2.1-experimental",
        dataset_campaign_version="predictive_dataset_campaign_v2_2",
        dataset_feature_version="2.1",
        dataset_target_version="v1-congestion-threshold-2-horizon-window",
        prediction_horizon_seconds=PRIMARY_HORIZON,
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
    calibrator_path = export_calibrator(isotonic, str(OUTPUT_DIR))

    try:
        best_params = best["model"].underlying.get_params()
    except Exception:
        best_params = {}
    clean_params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in best_params.items()}

    metadata_v2_1 = ExtendedModelMetadataV2(
        model_version="localized_predictive_model_v2_2",
        dataset_campaign_version="predictive_dataset_campaign_v2_2",
        dataset_schema_version="2.1-experimental",
        dataset_feature_version="2.1",
        dataset_target_version="v1-congestion-threshold-2-horizon-window",
        prediction_horizon_seconds=PRIMARY_HORIZON,
        training_seed=SEED,
        split_strategy="scenario-level 70/15/15, deterministic seed, no row-level leakage",
        algorithm=best_name,
        algorithm_library=type(best["model"]).__name__,
        hyperparameters=clean_params,
        calibration_method="isotonic (fit on validation split only)",
        calibration_artifact="calibrator.joblib",
        feature_order=feature_names,
        candidate_types_supported=("Door", "Exit", "Stair"),
        decision_threshold=best_threshold,
        known_limitations=(
            "candidate_recent_flow_rate has a full live source only for Exit today (evacuation_progress.ExitFlow); "
            "Door/Stair use a new, smaller live mechanism (live_occupants.history.zone_transitions) not yet validated live.",
            "multi_exit_wide remains the hardest topology family to generalize to -- see topology_holdout results.",
            "This milestone loaded only the 20s-horizon slice -- no 10s/30s/60s horizon-robustness sweep.",
        ),
        git_commit=_git_commit(),
    )
    metadata_v2_1_path = write_metadata_v2(metadata_v2_1, str(OUTPUT_DIR))

    report["export"] = {**export_paths, "calibrator_path": calibrator_path, "metadata_v2_2_path": metadata_v2_1_path}
    report["production_readiness"] = production_readiness
    report["production_readiness_rationale"] = rationale

    # =====================================================
    # Inference performance
    # =====================================================

    single_row = test_feat.X[:1]
    n_timing_repeats = 200
    t0 = time.time()
    for _ in range(n_timing_repeats):
        best["model"].predict_proba(single_row)
    single_row_seconds = (time.time() - t0) / n_timing_repeats

    t0 = time.time()
    best["model"].predict_proba(test_feat.X)
    batch_seconds = time.time() - t0

    report["inference_performance"] = {
        "single_row_latency_seconds": single_row_seconds,
        "batch_size": int(len(test_feat.X)),
        "batch_seconds": batch_seconds,
        "candidates_per_second_batch": float(len(test_feat.X) / batch_seconds) if batch_seconds > 0 else None,
        "note": "CPU-only measurement; feature-extraction cost measured separately.",
    }

    report["memory_log"] = memory_log
    report["total_wall_seconds"] = time.time() - overall_start
    report["git_commit"] = _git_commit()

    report_path = OUTPUT_DIR / "training_report_v2_2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Report written to {report_path}")


def _assess_production_readiness(test_metrics, leakage_recheck, shuffle_test, topology_results):

    reasons = []
    pr_auc = test_metrics["pr_auc"]
    positive_rate = test_metrics["positive_rate"]

    if leakage_recheck["flagged_for_leakage_review"]:
        return "NOT_READY", f"Leakage review flagged features: {leakage_recheck['flagged_for_leakage_review']}."

    if not shuffle_test["near_chance"]:
        return "NOT_READY", (
            f"Label-shuffle test did not collapse to chance (ROC-AUC="
            f"{shuffle_test['shuffled_label_roc_auc_on_real_val_labels']:.3f})."
        )

    if pr_auc is None or pr_auc < 2 * positive_rate:
        reasons.append(f"PR-AUC ({pr_auc}) is not meaningfully above the positive-rate baseline ({positive_rate:.3f}).")

    topology_pr_aucs = [
        result["test_metrics"]["pr_auc"] for result in topology_results.values()
        if result["test_metrics"]["pr_auc"] is not None
    ]
    if topology_pr_aucs and min(topology_pr_aucs) < 0.5 * pr_auc:
        weakest = min(topology_results.items(), key=lambda kv: (kv[1]["test_metrics"]["pr_auc"] or 0))
        reasons.append(
            f"Leave-one-topology-out PR-AUC still collapses on at least one held-out family "
            f"({weakest[0]}: {weakest[1]['test_metrics']['pr_auc']}) relative to the normal-split PR-AUC ({pr_auc})."
        )

    if reasons:
        return "PROMISING_BUT_NEEDS_MORE_DATA", " ".join(reasons)

    return "READY_FOR_SHADOW_MODE_LIVE_VALIDATION", "All sanity checks passed, metrics clear baselines, and topology-holdout generalization did not collapse."


if __name__ == "__main__":
    main()
