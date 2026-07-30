"""Predictive Model Development & Benchmark Campaign milestone -- the
one-shot orchestration script (same "reusable package + scripts/
driver" split as scripts/model_v4_generalization_evaluation.py uses for
predictive_model/). Loads the cached 11,997-scenario Pipeline B
(ai_registry) training campaign, trains every model_benchmark.algorithms
candidate for both bottleneck_occurrence (classification) and
evacuation_time (regression), runs a deterministic grid search per
algorithm, evaluates every model via prediction_evaluation's own metric
functions, slices robustness by condition, computes feature importance,
builds a leaderboard, and registers the winning model(s) into a
ModelRegistry (in-memory + persisted artifact, never wired into any
runtime startup path -- Shadow-Mode's existing wiring is untouched).

Does NOT modify simulation, feature extraction, dataset generation,
Recommendation, or Guidance.

Usage: python scripts/run_model_benchmark_campaign.py
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION
from ai_registry.metadata import Deployability, ModelMetadata, save_live_model
from ai_registry.registry import ModelRegistry
from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.split import apply_split, make_split

from prediction_evaluation.classification_metrics import compute_classification_metrics
from prediction_evaluation.regression_metrics import compute_regression_metrics

from model_benchmark.algorithms import (
    CLASSIFICATION_ALGORITHMS, CLASSIFICATION_PARAM_GRIDS, EXCLUDED_ALGORITHMS,
    REGRESSION_ALGORITHMS, REGRESSION_PARAM_GRIDS, display_name,
)
from model_benchmark.feature_importance import (
    correlation_analysis, native_importance, permutation_importance_classification,
    permutation_importance_regression,
)
from model_benchmark.robustness import (
    build_condition_tags, robustness_report_classification, robustness_report_regression,
)
from model_benchmark.search import grid_search_classification, grid_search_regression

SCRATCH = Path(r"C:\Users\riddh\AppData\Local\Temp\claude\C--Users-riddh-Desktop-SynEvac1\3cfe9653-d222-4b7b-ad2c-e86698320fce\scratchpad")
CACHE_PATH = SCRATCH / "benchmark_dataset_cache.joblib"
CAMPAIGN_DIR = SCRATCH / "benchmark_campaign"
OUTPUT_DIR = SCRATCH / "benchmark_results"
MODEL_ARTIFACT_DIR = SCRATCH / "benchmark_registered_models"

SPLIT_SEED = 20260730  # matches the campaign's own master_seed -- one documented seed for the
                        # whole milestone, reused (not re-chosen per split) for a single, traceable
                        # source of "why this exact train/val/test partition."
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 3
PERMUTATION_REPEATS = 3

DATASET_IDENTIFIER = "model-benchmark-campaign-v1-11997-scenarios"


class _CachedDataset:
    """Minimal TrainingDataset-shaped view over the cached, already-
    live-compatible-converted campaign rows -- lets BottleneckModel.
    build_table()/EvacuationTimeModel.build_table() be reused completely
    unmodified rather than reimplementing their label-engineering logic."""

    def __init__(self, scenario_ids, feature_rows, ground_truth_rows, outcome_rows):
        self.scenario_ids = scenario_ids
        self._feature_rows = feature_rows
        self._ground_truth_rows = ground_truth_rows
        self._outcome_rows = outcome_rows

    def scenario_feature_rows(self):
        return [dict(r) for r in self._feature_rows]

    def ground_truth_rows(self):
        return [dict(r) if r else None for r in self._ground_truth_rows]

    def outcome_rows(self):
        return [dict(r) for r in self._outcome_rows]


def _jsonable(obj):

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "value") and hasattr(type(obj), "__members__"):  # Enum
        return obj.value
    return obj


def _positive_proba(model, X_rows):

    try:
        proba = model.predict_proba(X_rows)
        classes = list(model.label_encoder.classes_)
        positive_index = classes.index(True) if True in classes else 1
        return tuple(float(v) for v in proba[:, positive_index])
    except (AttributeError, TypeError):
        return ()


def _single_row_latency_ms(model, X_rows, n_samples=50):

    sample = X_rows[:min(n_samples, len(X_rows))]
    if not sample:
        return None

    start = time.perf_counter()
    for row in sample:
        model.predict([row])
    elapsed = time.perf_counter() - start

    return (elapsed / len(sample)) * 1000


# =====================================================


def run_classification_algorithm(algorithm, Xb_train, yb_train, groups_b_train, Xb_val, yb_val, Xb_test, yb_test, tags_test):

    print(f"[classification/{algorithm}] starting", flush=True)

    def factory(params):
        return BottleneckModel(config={"algorithm": algorithm, "algorithm_kwargs": params}, target="occurrence")

    grid = CLASSIFICATION_PARAM_GRIDS[algorithm]

    search_result = None
    best_params = {}

    if grid:
        search_result = grid_search_classification(factory, Xb_train, yb_train, groups_b_train, grid, n_folds=CV_FOLDS)
        best_params = search_result.best_params

    model = factory(best_params)
    process = psutil.Process()
    mem_before = process.memory_info().rss
    fit_start = time.perf_counter()
    model.fit(Xb_train, yb_train)
    train_time = time.perf_counter() - fit_start
    train_mem_mb = (process.memory_info().rss - mem_before) / 1e6

    y_pred_test = model.predict(Xb_test)
    y_proba_test = _positive_proba(model, Xb_test)

    batch_start = time.perf_counter()
    model.predict(Xb_test)
    batch_elapsed = time.perf_counter() - batch_start
    batch_latency_ms_per_row = (batch_elapsed / len(Xb_test)) * 1000 if Xb_test else None
    single_row_latency_ms = _single_row_latency_ms(model, Xb_test)

    test_metrics = compute_classification_metrics([bool(v) for v in yb_test], [bool(v) for v in y_pred_test], y_proba_test)

    y_pred_val = model.predict(Xb_val) if Xb_val else []
    y_proba_val = _positive_proba(model, Xb_val) if Xb_val else ()
    val_metrics = compute_classification_metrics([bool(v) for v in yb_val], [bool(v) for v in y_pred_val], y_proba_val) if Xb_val else None

    robustness = robustness_report_classification(
        [bool(v) for v in yb_test], [bool(v) for v in y_pred_test], y_proba_test, tags_test,
    )

    native_imp = native_importance(model)
    perm_imp = permutation_importance_classification(
        model, Xb_test, [bool(v) for v in yb_test], CANONICAL_LIVE_FEATURE_NAMES, n_repeats=PERMUTATION_REPEATS,
    )

    print(f"[classification/{algorithm}] done: f1={test_metrics.f1}, train_time={train_time:.2f}s", flush=True)

    return {
        "algorithm": algorithm, "display_name": display_name(algorithm), "target": "bottleneck_occurrence",
        "best_params": best_params,
        "search_trials": [_jsonable(t) for t in search_result.trials] if search_result else None,
        "train_time_seconds": train_time, "train_memory_delta_mb": train_mem_mb,
        "batch_inference_latency_ms_per_row": batch_latency_ms_per_row,
        "single_row_inference_latency_ms": single_row_latency_ms,
        "test_metrics": _jsonable(test_metrics), "val_metrics": _jsonable(val_metrics),
        "robustness": [_jsonable(r) for r in robustness],
        "native_importance": native_imp,
        "permutation_importance": perm_imp,
        "n_train": len(Xb_train), "n_val": len(Xb_val), "n_test": len(Xb_test),
    }, model


def run_regression_algorithm(algorithm, Xe_train, ye_train, groups_e_train, Xe_val, ye_val, Xe_test, ye_test, tags_test):

    print(f"[regression/{algorithm}] starting", flush=True)

    def factory(params):
        return EvacuationTimeModel(config={"algorithm": algorithm, "algorithm_kwargs": params})

    grid = REGRESSION_PARAM_GRIDS[algorithm]

    search_result = None
    best_params = {}

    if grid:
        search_result = grid_search_regression(factory, Xe_train, ye_train, groups_e_train, grid, n_folds=CV_FOLDS)
        best_params = search_result.best_params

    model = factory(best_params)
    process = psutil.Process()
    mem_before = process.memory_info().rss
    fit_start = time.perf_counter()
    model.fit(Xe_train, ye_train)
    train_time = time.perf_counter() - fit_start
    train_mem_mb = (process.memory_info().rss - mem_before) / 1e6

    y_pred_test = model.predict(Xe_test)

    batch_start = time.perf_counter()
    model.predict(Xe_test)
    batch_elapsed = time.perf_counter() - batch_start
    batch_latency_ms_per_row = (batch_elapsed / len(Xe_test)) * 1000 if Xe_test else None
    single_row_latency_ms = _single_row_latency_ms(model, Xe_test)

    test_metrics = compute_regression_metrics(ye_test, y_pred_test)

    y_pred_val = model.predict(Xe_val) if Xe_val else []
    val_metrics = compute_regression_metrics(ye_val, y_pred_val) if Xe_val else None

    robustness = robustness_report_regression(ye_test, y_pred_test, tags_test)

    native_imp = native_importance(model)
    perm_imp = permutation_importance_regression(
        model, Xe_test, ye_test, CANONICAL_LIVE_FEATURE_NAMES, n_repeats=PERMUTATION_REPEATS,
    )

    print(f"[regression/{algorithm}] done: mae={test_metrics.mae}, train_time={train_time:.2f}s", flush=True)

    return {
        "algorithm": algorithm, "display_name": display_name(algorithm), "target": "evacuation_time",
        "best_params": best_params,
        "search_trials": [_jsonable(t) for t in search_result.trials] if search_result else None,
        "train_time_seconds": train_time, "train_memory_delta_mb": train_mem_mb,
        "batch_inference_latency_ms_per_row": batch_latency_ms_per_row,
        "single_row_inference_latency_ms": single_row_latency_ms,
        "test_metrics": _jsonable(test_metrics), "val_metrics": _jsonable(val_metrics),
        "robustness": [_jsonable(r) for r in robustness],
        "native_importance": native_imp,
        "permutation_importance": perm_imp,
        "n_train": len(Xe_train), "n_val": len(Xe_val), "n_test": len(Xe_test),
    }, model


# =====================================================


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading cached dataset...", flush=True)
    cache = joblib.load(CACHE_PATH)

    live_dataset = _CachedDataset(
        cache["scenario_ids"], cache["live_feature_rows"], cache["ground_truth_rows"], cache["outcome_rows"],
    )
    legacy_rows = cache["legacy_feature_rows"]
    gt_rows = cache["ground_truth_rows"]

    X_bottleneck, y_bottleneck, extra_b = BottleneckModel.build_table(live_dataset, target="occurrence")
    X_evac, y_evac, extra_e = EvacuationTimeModel.build_table(live_dataset)

    groups_b = extra_b.get("groups") or list(live_dataset.scenario_ids)
    groups_e = list(live_dataset.scenario_ids)

    split_b = make_split(len(X_bottleneck), groups=groups_b, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=SPLIT_SEED)
    split_e = make_split(len(X_evac), groups=groups_e, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=SPLIT_SEED)

    Xb_train, Xb_val, Xb_test = apply_split(X_bottleneck, split_b)
    yb_train, yb_val, yb_test = apply_split(list(y_bottleneck), split_b)
    Xe_train, Xe_val, Xe_test = apply_split(X_evac, split_e)
    ye_train, ye_val, ye_test = apply_split(list(y_evac), split_e)

    groups_b_train = [groups_b[i] for i in split_b.train_indices]
    groups_e_train = [groups_e[i] for i in split_e.train_indices]

    tags_test_b = [build_condition_tags(legacy_rows[i], gt_rows[i]) for i in split_b.test_indices]
    tags_test_e = [build_condition_tags(legacy_rows[i], gt_rows[i]) for i in split_e.test_indices]

    split_documentation = {
        "split_seed": SPLIT_SEED, "test_size": TEST_SIZE, "val_size": VAL_SIZE,
        "split_method": "ai_training.split.make_split -- GroupShuffleSplit grouped by scenario_id, "
                        "deterministic for a fixed random_state",
        "bottleneck_occurrence": {
            "n_rows_total": len(X_bottleneck),
            "n_train": len(split_b.train_indices), "n_val": len(split_b.val_indices), "n_test": len(split_b.test_indices),
        },
        "evacuation_time": {
            "n_rows_total": len(X_evac),
            "n_train": len(split_e.train_indices), "n_val": len(split_e.val_indices), "n_test": len(split_e.test_indices),
        },
    }
    with open(OUTPUT_DIR / "split_documentation.json", "w", encoding="utf-8") as f:
        json.dump(split_documentation, f, indent=2)
    print(json.dumps(split_documentation, indent=2), flush=True)

    classification_results = []
    classification_models = {}

    for algorithm in CLASSIFICATION_ALGORITHMS:

        try:
            result, model = run_classification_algorithm(
                algorithm, Xb_train, yb_train, groups_b_train, Xb_val, yb_val, Xb_test, yb_test, tags_test_b,
            )
            classification_results.append(result)
            classification_models[algorithm] = model

        except Exception as exc:  # noqa: BLE001 -- one failing algorithm must not lose the rest
            print(f"[classification/{algorithm}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            classification_results.append({"algorithm": algorithm, "target": "bottleneck_occurrence", "error": str(exc)})

        with open(OUTPUT_DIR / "classification_results.json", "w", encoding="utf-8") as f:
            json.dump(classification_results, f, indent=2, default=str)

    regression_results = []
    regression_models = {}

    for algorithm in REGRESSION_ALGORITHMS:

        try:
            result, model = run_regression_algorithm(
                algorithm, Xe_train, ye_train, groups_e_train, Xe_val, ye_val, Xe_test, ye_test, tags_test_e,
            )
            regression_results.append(result)
            regression_models[algorithm] = model

        except Exception as exc:  # noqa: BLE001
            print(f"[regression/{algorithm}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            regression_results.append({"algorithm": algorithm, "target": "evacuation_time", "error": str(exc)})

        with open(OUTPUT_DIR / "regression_results.json", "w", encoding="utf-8") as f:
            json.dump(regression_results, f, indent=2, default=str)

    # -- correlation analysis (once, over the canonical feature set, train rows) --
    pairs, numeric_cols = correlation_analysis(Xb_train, threshold=0.8)
    correlation_report = {
        "numeric_columns_considered": list(numeric_cols),
        "highly_correlated_pairs": [_jsonable(p) for p in pairs],
        "threshold": 0.8,
    }
    with open(OUTPUT_DIR / "correlation_report.json", "w", encoding="utf-8") as f:
        json.dump(correlation_report, f, indent=2)

    # -- persist fitted model objects for later phases (registration, horizon eval) --
    joblib.dump({"classification": classification_models, "regression": regression_models},
                OUTPUT_DIR / "fitted_models.joblib")

    print("BENCHMARK CAMPAIGN COMPLETE", flush=True)
    print(f"classification models: {list(classification_models.keys())}", flush=True)
    print(f"regression models: {list(regression_models.keys())}", flush=True)


if __name__ == "__main__":
    main()
