import io
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import joblib

from ai_training.dataset import TrainingDataset
from ai_training.evaluation import (
    evaluate_classification,
    evaluate_multioutput_regression,
    evaluate_regression,
)
from ai_training.experiment import MODEL_REGISTRY
from ai_training.split import apply_split, make_split


# =====================================================
# Phase 3 -- Model Benchmarking. Trains the SAME ai_training model
# type (evacuation_time/bottleneck/exit_usage/smoke_prediction) with
# several different classical-ML algorithms (random_forest/
# gradient_boosting/xgboost) against the exact same TrainingDataset and
# the exact same deterministic train/test split, so every entry below
# is comparable apples-to-apples. Never modifies ai_training -- this
# is purely an outside-in timing/sizing harness built on top of its
# already-public build_table()/fit()/predict()/evaluate_*() API.
# =====================================================


@dataclass(frozen=True)
class BenchmarkResult:

    label: str
    algorithm: str
    metrics: Dict[str, Any]
    training_time_seconds: float
    prediction_time_seconds: float
    model_size_bytes: int
    train_size: int
    test_size: int


# =====================================================


def benchmark_algorithms(
    dataset: TrainingDataset,
    model_name: str,
    algorithms: Sequence[str],
    *,
    algorithm_labels: Optional[Dict[str, str]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 0,
) -> List[BenchmarkResult]:

    if model_name not in MODEL_REGISTRY:

        raise ValueError(
            f"Unknown model_name {model_name!r} -- choose one of {sorted(MODEL_REGISTRY)}"
        )

    model_cls, task = MODEL_REGISTRY[model_name]
    model_kwargs = dict(model_kwargs or {})
    algorithm_labels = algorithm_labels or {}

    if model_name == "bottleneck":

        target = model_kwargs.get("target", "occurrence")
        X_rows, y, extra = model_cls.build_table(dataset, target=target)

    else:

        X_rows, y, extra = model_cls.build_table(dataset)

    groups = extra.get("groups")
    output_names = extra.get("output_names")

    split = make_split(len(X_rows), groups=groups, test_size=test_size, random_state=random_state)
    X_train, _val, X_test = apply_split(X_rows, split)
    y_train, _val, y_test = apply_split(list(y), split)

    results: List[BenchmarkResult] = []

    for algorithm in algorithms:

        config = {"algorithm": algorithm}

        if model_name == "bottleneck":
            model = model_cls(config=config, target=model_kwargs.get("target", "occurrence"))
        else:
            model = model_cls(config=config)

        start = time.perf_counter()

        if output_names is not None:
            model.fit(X_train, y_train, output_names=output_names)
        else:
            model.fit(X_train, y_train)

        training_time_seconds = time.perf_counter() - start

        start = time.perf_counter()
        model.predict(X_test)
        prediction_time_seconds = time.perf_counter() - start

        model_size_bytes = _serialized_size_bytes(model)

        if output_names is not None:
            metrics = evaluate_multioutput_regression(model, X_test, y_test, output_names)
        elif task == "classification":
            metrics = evaluate_classification(model, X_test, y_test)
        else:
            metrics = evaluate_regression(model, X_test, y_test)

        results.append(
            BenchmarkResult(
                label=algorithm_labels.get(algorithm, algorithm),
                algorithm=algorithm,
                metrics=metrics,
                training_time_seconds=training_time_seconds,
                prediction_time_seconds=prediction_time_seconds,
                model_size_bytes=model_size_bytes,
                train_size=len(X_train),
                test_size=len(X_test),
            )
        )

    return results


# =====================================================


def _serialized_size_bytes(model) -> int:

    buffer = io.BytesIO()
    joblib.dump(model.to_bundle(), buffer)

    return buffer.tell()


# =====================================================


def to_table_rows(results: Sequence[BenchmarkResult]) -> List[Dict[str, Any]]:

    # Flattens every BenchmarkResult into one plain, JSON/CSV-friendly
    # dict per algorithm -- the "Benchmark table" Phase 3/5 asks for.

    rows = []

    for result in results:

        row: Dict[str, Any] = {
            "label": result.label,
            "algorithm": result.algorithm,
            "training_time_seconds": result.training_time_seconds,
            "prediction_time_seconds": result.prediction_time_seconds,
            "model_size_bytes": result.model_size_bytes,
            "train_size": result.train_size,
            "test_size": result.test_size,
        }
        row.update(flatten_metrics(result.metrics))
        rows.append(row)

    return rows


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:

    if "macro_average" in metrics:  # multioutput_regression_metrics()'s own shape
        return {f"macro_{key}": value for key, value in metrics["macro_average"].items()}

    return dict(metrics)


# =====================================================


def best_result(results: Sequence[BenchmarkResult], metric_key: str, *, minimize: bool) -> BenchmarkResult:

    def _value(result: BenchmarkResult) -> float:

        flattened = flatten_metrics(result.metrics)

        return flattened[metric_key]

    return min(results, key=_value) if minimize else max(results, key=_value)
