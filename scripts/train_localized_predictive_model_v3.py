"""Localized Predictive Model V3 -- OFFLINE RESEARCH ONLY.

First full, rigorous evaluation of a predictive model against Target V2
("v2-persistent-demand-service-imbalance", predictive_dataset/
target_generator_v2.py, frozen -- NOT modified by this milestone).
Consumes the ALREADY-EXISTING full-scale relabeled dataset (data/
predictive_congestion_target_v2/candidate_dataset_relabeled.csv,
2,405,049 rows / 2,500 scenarios, produced by the immediately-prior
Predictive Congestion Target V2 milestone) -- no re-simulation, no new
campaign. Reuses the V2.2 12-field feature schema unchanged and the
SAME scenario-split seed/convention every prior milestone used, for
direct comparability.

Reuses almost all of predictive_model/* unchanged (scenario_split,
feature_prep_v2_1, imbalance, metrics, calibration, feature_importance,
sanity_checks, operational_slices, topology_holdout, training_size_
study, model_export/model_export_v2, tree_models) -- the only new
pieces this milestone adds are the deterministic current-state baseline
(predictive_model.baselines.DeterministicCurrentStateBaseline) and the
lead-time / threshold / multi-bottleneck-combination / near-onset
analyses below, which are specific to Target V2's own onset semantics
and were meaningless for Target V1.

RandomForest is DELIBERATELY SKIPPED this run (disclosed, not silently
dropped) -- every prior milestone found it a statistical near-tie with
XGBoost/HistGradientBoosting at 5-13x the fit cost (350-386s vs Root
25-30s), and this milestone's own charter explicitly says "do not run a
giant algorithm tournament... optionally evaluate Random Forest if the
existing framework already supports it cheaply" -- 350s+ is not cheap
on this ~7.3GB-RAM development machine, and skipping it does not change
which model wins.

NOTHING TRAINED HERE IS WIRED INTO recommendation scoring, exit
ranking, guidance, signage, LiveRuntime, or operator workflow.

Usage: python scripts/train_localized_predictive_model_v3.py
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

from predictive_model.baselines import DeterministicCurrentStateBaseline, build_baselines
from predictive_model.calibration import IsotonicCalibrator, evaluate_calibration_methods
from predictive_model.error_analysis import build_error_analysis
from predictive_model.feature_importance import builtin_feature_importance, permutation_importance_report
from predictive_model.feature_prep_v2_1 import build_experimental_feature_matrix
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
from predictive_dataset.target_generator_v2 import MIN_PERSISTENCE_SECONDS
from predictive_model.topology_holdout import (
    apply_topology_holdout,
    assert_no_holdout_overlap,
    build_topology_holdout_splits,
)
from predictive_model.training_size_study import training_size_study
from predictive_model.tree_models import build_tree_models, library_availability_report

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "predictive_congestion_target_v2"
CSV_PATH = DATA_DIR / "candidate_dataset_relabeled.csv"
SCENARIO_METADATA_PATH = DATA_DIR / "scenario_metadata.json"

OUTPUT_DIR = REPO_ROOT / "data" / "localized_predictive_model_v3"

SEED = 20260726  # SAME seed every prior milestone used (model-training determinism)
PRIMARY_HORIZON = 20.0
TARGET_VERSION = "v2-persistent-demand-service-imbalance"

N_JOBS = 2
MIN_AVAILABLE_MEMORY_BYTES = 300_000_000
WATCHDOG_CRITICAL_BYTES = 180_000_000
WATCHDOG_POLL_SECONDS = 4.0

_watchdog_log_path = None

BASE_FEATURE_NAMES = (
    "total_active_occupant_count", "candidate_capacity", "candidate_walking_distance",
    "candidate_traversable", "candidate_adjacent_zone_occupancy", "candidate_queue_length",
    "candidate_approaching_count", "candidate_congestion_level",
)
EXPERIMENTAL_FEATURE_NAMES = ("candidate_recent_flow_rate", "candidate_congestion_trend", "candidate_alternative_route_count")

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
    "lead_time_seconds_v2": "float32",
}

THRESHOLD_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LEAD_TIME_BUCKETS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0))


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

    keep_columns = (
        ["scenario_id", "observation_time", "candidate_id", "candidate_type"]
        + list(BASE_FEATURE_NAMES) + list(EXPERIMENTAL_FEATURE_NAMES)
        + ["currently_congested_v2", "had_any_activity_in_window_v2", "target_v2", "lead_time_seconds_v2"]
    )

    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=keep_columns):
        chunk = chunk.astype({k: v for k, v in _COMPACT_DTYPES.items() if k in chunk.columns})
        chunks.append(chunk)

    frame = pd.concat(chunks, ignore_index=True)
    del chunks

    for column in ("scenario_id", "candidate_id", "candidate_type", "candidate_congestion_level", "candidate_congestion_trend"):
        frame[column] = frame[column].astype("category")

    frame["currently_congested_v2"] = frame["currently_congested_v2"].astype(bool)
    frame["had_any_activity_in_window_v2"] = frame["had_any_activity_in_window_v2"].astype(bool)

    # rename to the generic names every reused predictive_model.* module
    # already expects ("target", "currently_congested")
    frame = frame.rename(columns={
        "target_v2": "target",
        "currently_congested_v2": "currently_congested",
        "had_any_activity_in_window_v2": "had_any_activity_in_window",
    })

    return frame


def _trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["target"].notna()].copy()


def _fresh_model(name: str, seed: int = SEED, feature_names=None):
    if name == "deterministic_current_state":
        return DeterministicCurrentStateBaseline(feature_names)
    builders = {**build_baselines(seed=seed), **build_tree_models(seed=seed, n_jobs=N_JOBS)}
    return builders[name]


def _is_trivial_baseline(name: str) -> bool:
    return name in ("majority_class", "always_negative", "random", "deterministic_current_state")


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

    print("Loading Target V2 relabeled dataset (chunked)...")
    t0 = time.time()
    frame = _load_dataset_chunked(CSV_PATH)
    load_seconds = time.time() - t0
    print(f"    Loaded {len(frame)} rows, {frame['scenario_id'].nunique()} scenarios in {load_seconds:.1f}s")
    memory_log.append(_mem_log("after dataset load"))

    # =====================================================
    # Phase 6 -- currently-congested exclusion audit
    # =====================================================

    total_rows = len(frame)
    excluded_rows = int(frame["currently_congested"].sum())
    eligible_rows = int(frame["target"].notna().sum())
    positives = int(frame["target"].fillna(False).astype(bool).sum())
    negatives = eligible_rows - positives

    exclusion_audit = {
        "total_candidate_time_observations": total_rows,
        "excluded_already_congested": excluded_rows,
        "eligible_for_prediction": eligible_rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positives / eligible_rows if eligible_rows else None,
    }
    print(f"Currently-congested exclusion audit: {exclusion_audit}")

    all_scenario_ids = frame["scenario_id"].astype(str).unique().tolist()
    split = split_scenarios(all_scenario_ids, seed=SEED)
    print(f"Scenario split: train={len(split.train_scenario_ids)} val={len(split.val_scenario_ids)} "
          f"test={len(split.test_scenario_ids)}")

    train_df, val_df, test_df = apply_split(frame, split)
    assert_no_scenario_overlap(split, train_df, val_df, test_df)

    train_trainable = _trainable_rows(train_df)
    val_trainable = _trainable_rows(val_df)
    test_trainable = _trainable_rows(test_df)

    train_feat = build_experimental_feature_matrix(train_trainable)
    val_feat = build_experimental_feature_matrix(val_trainable)
    test_feat = build_experimental_feature_matrix(test_trainable)

    print(f"Trainable rows: train={len(train_feat.y)} val={len(val_feat.y)} test={len(test_feat.y)} "
          f"(train positive rate={train_feat.y.mean():.4f})")
    memory_log.append(_mem_log("after feature matrices"))

    report: Dict[str, Any] = {
        "target_version": TARGET_VERSION,
        "dataset_manifest": {
            "csv_path": str(CSV_PATH),
            "horizon_seconds": PRIMARY_HORIZON,
            "row_count": total_rows,
            "scenario_count": int(frame["scenario_id"].nunique()),
            "load_seconds": load_seconds,
            "feature_schema": "V2.2 experimental (9 base + 3 V2.1 = 12 fields)",
        },
        "currently_congested_exclusion_audit": exclusion_audit,
        "scenario_split": split.to_dict(),
        "library_availability": library_availability_report(),
        "n_jobs_used": N_JOBS,
        "random_forest_skipped": True,
        "random_forest_skip_reason": (
            "Every prior milestone found RandomForest a statistical near-tie with XGBoost/"
            "HistGradientBoosting at 5-13x the fit cost (350-386s vs 25-30s); this milestone's own "
            "charter says not to run a giant tournament and to only optionally include RF 'if cheap' "
            "-- 350s+ is not cheap on this development machine."
        ),
        "trainable_row_counts": {"train": int(len(train_feat.y)), "val": int(len(val_feat.y)), "test": int(len(test_feat.y))},
    }

    # =====================================================
    # Baselines + candidate models
    # =====================================================

    model_results: Dict[str, Any] = {}
    stash: Dict[str, Any] = {}

    all_models = {
        **build_baselines(seed=SEED),
        "deterministic_current_state": DeterministicCurrentStateBaseline(train_feat.feature_names),
        **{k: v for k, v in build_tree_models(seed=SEED, n_jobs=N_JOBS).items() if k != "random_forest"},
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

    ml_candidate_names = [n for n in model_results if n not in ("majority_class", "always_negative", "random", "deterministic_current_state")]
    best_name = max(ml_candidate_names, key=lambda name: model_results[name]["test_metrics"]["pr_auc"] or -1)
    best = stash[best_name]
    best_threshold = model_results[best_name]["threshold"]
    print(f"BEST MODEL: {best_name}")

    deterministic_pr_auc = model_results["deterministic_current_state"]["test_metrics"]["pr_auc"]
    best_pr_auc = model_results[best_name]["test_metrics"]["pr_auc"]
    report["ml_vs_deterministic_baseline"] = {
        "deterministic_baseline_pr_auc": deterministic_pr_auc,
        "best_ml_model_pr_auc": best_pr_auc,
        "absolute_lift": (best_pr_auc - deterministic_pr_auc) if (best_pr_auc is not None and deterministic_pr_auc is not None) else None,
        "relative_lift": (best_pr_auc / deterministic_pr_auc) if deterministic_pr_auc else None,
    }
    print(f"ML vs deterministic baseline: {report['ml_vs_deterministic_baseline']}")

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
    # Multi-bottleneck candidate-type-combination breakdown
    # =====================================================

    positive_frame = frame[frame["target"].fillna(False).astype(bool)]
    bucket_types = positive_frame.groupby(["scenario_id", "observation_time"], observed=True)["candidate_type"].apply(
        lambda s: tuple(sorted(s.unique()))
    )
    from collections import Counter
    combo_counts = Counter(bucket_types[bucket_types.apply(len) >= 2])
    report["multi_bottleneck_candidate_type_combinations"] = {str(k): v for k, v in combo_counts.items()}
    print(f"Multi-bottleneck candidate-type combinations: {dict(combo_counts)}")

    # =====================================================
    # Calibration
    # =====================================================

    calibration = evaluate_calibration_methods(val_feat.y, best["val_prob"], test_feat.y, best["test_prob"])
    report["calibration"] = calibration.to_dict()
    print(f"Calibration: recommended={calibration.recommended_method}")

    calibration_by_type: Dict[str, Any] = {}
    for ctype in ("Door", "Exit", "Stair"):
        mask_val = val_feat.candidate_types == ctype
        mask_test = test_feat.candidate_types == ctype
        if mask_val.sum() < 20 or mask_test.sum() < 20:
            continue
        type_calib = evaluate_calibration_methods(
            val_feat.y[mask_val], best["val_prob"][mask_val], test_feat.y[mask_test], best["test_prob"][mask_test],
        )
        calibration_by_type[ctype] = type_calib.to_dict()
    report["calibration_by_candidate_type"] = calibration_by_type

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
        "flow_and_trend": [
            n for n in feature_names
            if n == "candidate_recent_flow_rate" or n.startswith("candidate_congestion_trend=")
        ],
        "alternative_route_structure": [n for n in feature_names if n == "candidate_alternative_route_count"],
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
    print(f"Label shuffle test near chance: {shuffle_test['near_chance']} "
          f"(ROC-AUC={shuffle_test['shuffled_label_roc_auc_on_real_val_labels']:.3f})")
    memory_log.append(_mem_log("after leakage/sanity checks"))

    # =====================================================
    # Lead-time bucket analysis (test split, positives only for recall;
    # threshold-based predictions for the whole test set to get FN by bucket)
    # =====================================================

    test_analysis = test_trainable.reset_index(drop=True).copy()
    test_analysis["y_true"] = test_feat.y
    test_analysis["y_prob"] = best["test_prob"]
    test_analysis["y_pred"] = (best["test_prob"] >= best_threshold).astype(int)

    lead_time_report: Dict[str, Any] = {}
    positive_test = test_analysis[test_analysis["y_true"] == 1]
    for lo, hi in LEAD_TIME_BUCKETS:
        bucket = positive_test[(positive_test["lead_time_seconds_v2"] > lo) & (positive_test["lead_time_seconds_v2"] <= hi)]
        n = len(bucket)
        recall = float((bucket["y_pred"] == 1).mean()) if n else None
        lead_time_report[f"{lo}-{hi}s"] = {
            "n_positive_examples": int(n),
            "recall": recall,
            "mean_predicted_prob": float(bucket["y_prob"].mean()) if n else None,
            "median_predicted_prob": float(bucket["y_prob"].median()) if n else None,
        }
    report["lead_time_analysis"] = lead_time_report
    print(f"Lead-time analysis: {lead_time_report}")

    # near-onset exclusion audit (Phase 23) -- does foresight survive
    # once trivially-close-to-onset examples are excluded?
    near_onset_audit = {}
    for min_lead in (0.0, 1.0, 2.0):
        subset = test_analysis[(test_analysis["y_true"] == 0) | (test_analysis["lead_time_seconds_v2"] > min_lead)]
        y_true_subset = subset["y_true"].to_numpy()
        y_prob_subset = subset["y_prob"].to_numpy()
        m = compute_metrics(y_true_subset, y_prob_subset, threshold=best_threshold)
        near_onset_audit[f"min_lead_time_{min_lead}s"] = {
            "n": int(len(subset)), "positive_rate": m.positive_rate, "roc_auc": m.roc_auc,
            "pr_auc": m.pr_auc, "recall": m.recall, "precision": m.precision,
        }
    report["near_onset_exclusion_audit"] = near_onset_audit
    print(f"Near-onset exclusion audit: {near_onset_audit}")

    # =====================================================
    # Threshold analysis table
    # =====================================================

    threshold_table = []
    for thr in THRESHOLD_SWEEP:
        m = compute_metrics(test_feat.y, best["test_prob"], threshold=thr)
        cm = m.confusion_matrix
        fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) else None
        fnr = cm["fn"] / (cm["fn"] + cm["tp"]) if (cm["fn"] + cm["tp"]) else None
        threshold_table.append({
            "threshold": thr, "precision": m.precision, "recall": m.recall, "fpr": fpr, "fnr": fnr, "f1": m.f1,
        })
    report["threshold_analysis"] = threshold_table
    print("Threshold analysis computed.")

    # =====================================================
    # Training-size study
    # =====================================================

    def _model_factory():
        return _fresh_model(best_name)

    def _size_study_progress(key: str, result: Dict[str, Any]) -> None:
        print(f"    size_study[{key}]: scenarios={result['train_scenario_count']} rows={result['train_row_count']} "
              f"val_pr_auc={result['val_pr_auc']}")
        memory_log.append(_mem_log(f"after training-size study fraction {key}"))

    size_study = training_size_study(
        train_trainable, val_feat, _model_factory, seed=SEED,
        fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
        feature_builder=build_experimental_feature_matrix,
        progress_fn=_size_study_progress,
    )
    report["training_size_study"] = size_study
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

        holdout_train_trainable = _trainable_rows(holdout_train_df)
        holdout_test_trainable = _trainable_rows(holdout_test_df)
        del holdout_train_df, holdout_test_df
        gc.collect()

        holdout_train_feat = build_experimental_feature_matrix(holdout_train_trainable)
        holdout_test_feat = build_experimental_feature_matrix(holdout_test_trainable)
        del holdout_train_trainable, holdout_test_trainable
        gc.collect()
        memory_log.append(_mem_log(f"before fitting topology holdout [{holdout.held_out_family}]"))

        holdout_model = _fresh_model(best_name)
        _fit_with_weight(holdout_model, holdout_train_feat.X, holdout_train_feat.y, best_name)

        holdout_prob = holdout_model.predict_proba(holdout_test_feat.X)
        holdout_metrics = compute_metrics(holdout_test_feat.y, holdout_prob, threshold=best_threshold)
        holdout_by_type = metrics_by_group(
            holdout_test_feat.y, holdout_prob, holdout_test_feat.candidate_types, threshold=best_threshold,
        )

        topology_results[holdout.held_out_family] = {
            "train_scenario_count": len(holdout.train_scenario_ids),
            "test_scenario_count": len(holdout.test_scenario_ids),
            "test_row_count": int(len(holdout_test_feat.y)),
            "test_positive_rate": float(holdout_test_feat.y.mean()),
            "test_metrics": holdout_metrics.to_dict(),
            "test_metrics_by_candidate_type": holdout_by_type,
            "train_seconds": time.time() - t_start,
        }

        print(f"    topology_holdout[{holdout.held_out_family}]: ROC-AUC={holdout_metrics.roc_auc} "
              f"PR-AUC={holdout_metrics.pr_auc} ({time.time() - t_start:.1f}s)")

        del holdout_model, holdout_train_feat, holdout_test_feat
        gc.collect()
        memory_log.append(_mem_log(f"after topology holdout [{holdout.held_out_family}]"))

    report["topology_holdout"] = topology_results

    # =====================================================
    # Export
    # =====================================================

    production_readiness, rationale = _assess_production_readiness(
        model_results[best_name]["test_metrics"], leakage_recheck, shuffle_test, topology_results,
        report["ml_vs_deterministic_baseline"],
    )

    metadata = ModelMetadata(
        model_name=best_name,
        model_library=type(best["model"]).__name__,
        dataset_schema_version="2.1-experimental",
        dataset_campaign_version="predictive_congestion_target_v2",
        dataset_feature_version="2.1",
        dataset_target_version=TARGET_VERSION,
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

    metadata_v3 = ExtendedModelMetadataV2(
        model_version="localized_predictive_model_v3",
        dataset_campaign_version="predictive_congestion_target_v2",
        dataset_schema_version="2.1-experimental",
        dataset_feature_version="2.1",
        dataset_target_version=TARGET_VERSION,
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
            "candidate_recent_flow_rate has a full live source only for Exit today "
            "(evacuation_progress.ExitFlow); Door/Stair use live_occupants.history.zone_transitions, "
            "unit-tested but not yet validated against a real live deployment.",
            "multi_exit_wide remains the hardest topology family to generalize to.",
            "Target V2's currently_congested exclusion rate is high for Door (~45%), reducing "
            "usable onset-prediction examples for that type relative to its raw row count.",
        ),
        git_commit=_git_commit(),
        feature_schema_version="2.1-experimental",
        minimum_congestion_duration_seconds=MIN_PERSISTENCE_SECONDS,
        training_scenario_count=len(split.train_scenario_ids),
        calibration_status=(
            f"calibrated (isotonic, fit on validation split only; "
            f"ECE={calibration.after_isotonic['expected_calibration_error']:.5f}, "
            f"Brier={calibration.after_isotonic['brier_score']:.5f})"
        ),
        metrics_by_candidate_type=model_results[best_name]["test_metrics_by_candidate_type"],
        metrics_by_topology_family={
            family: result["test_metrics"] for family, result in topology_results.items()
        },
    )
    metadata_v3_path = write_metadata_v2(metadata_v3, str(OUTPUT_DIR))

    # explicit, required field (Phase 33): live_inference_wired = false
    with open(metadata_v3_path, "r", encoding="utf-8") as f:
        metadata_v3_dict = json.load(f)
    metadata_v3_dict["live_inference_wired"] = False
    metadata_v3_dict["production_readiness"] = production_readiness
    metadata_v3_dict["production_readiness_rationale"] = rationale
    with open(metadata_v3_path, "w", encoding="utf-8") as f:
        json.dump(metadata_v3_dict, f, indent=2, default=str)

    report["export"] = {**export_paths, "calibrator_path": calibrator_path, "metadata_v3_path": metadata_v3_path}
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
    batch_prob = best["model"].predict_proba(test_feat.X)
    batch_seconds = time.time() - t0

    t0 = time.time()
    isotonic.transform(batch_prob)
    calibration_overhead_seconds = time.time() - t0

    report["inference_performance"] = {
        "single_row_latency_seconds": single_row_seconds,
        "batch_size": int(len(test_feat.X)),
        "batch_seconds": batch_seconds,
        "candidates_per_second_batch": float(len(test_feat.X) / batch_seconds) if batch_seconds > 0 else None,
        "calibration_overhead_seconds_for_batch": calibration_overhead_seconds,
        "note": "CPU-only; training/dataset-generation time kept separate from inference latency.",
    }
    print(f"Inference: single-row={single_row_seconds*1000:.3f}ms, "
          f"batch={report['inference_performance']['candidates_per_second_batch']:.0f} candidates/sec")

    report["memory_log"] = memory_log
    report["total_wall_seconds"] = time.time() - overall_start
    report["git_commit"] = _git_commit()

    report_path = OUTPUT_DIR / "training_report_v3.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDone in {report['total_wall_seconds']:.1f}s. Report written to {report_path}")


def _assess_production_readiness(test_metrics, leakage_recheck, shuffle_test, topology_results, ml_vs_det):

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

    if ml_vs_det.get("relative_lift") is not None and ml_vs_det["relative_lift"] < 1.2:
        reasons.append(f"ML model does not meaningfully beat the deterministic current-state baseline "
                        f"(relative lift {ml_vs_det['relative_lift']:.2f}x).")

    topology_pr_aucs = [
        result["test_metrics"]["pr_auc"] for result in topology_results.values()
        if result["test_metrics"]["pr_auc"] is not None
    ]
    if topology_pr_aucs and min(topology_pr_aucs) < 0.5 * pr_auc:
        weakest = min(topology_results.items(), key=lambda kv: (kv[1]["test_metrics"]["pr_auc"] or 0))
        reasons.append(
            f"Leave-one-topology-out PR-AUC collapses on at least one held-out family "
            f"({weakest[0]}: {weakest[1]['test_metrics']['pr_auc']}) relative to normal-split PR-AUC ({pr_auc})."
        )

    if reasons:
        return "PROMISING_BUT_NEEDS_MORE_DATA", " ".join(reasons)

    return "READY_FOR_SHADOW_MODE_LIVE_VALIDATION", "All automated sanity checks passed; see docs for full human-reviewed verdict."


if __name__ == "__main__":
    main()
