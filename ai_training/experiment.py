import datetime
import json
import os

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ai_training.dataset import TrainingDataset
from ai_training.evaluation import (
    evaluate_classification,
    evaluate_multioutput_regression,
    evaluate_regression,
)
from ai_training.models import BottleneckModel, EvacuationTimeModel, ExitUsageModel, SmokePredictionModel
from ai_training.split import apply_split, make_split


# =====================================================
# Phase 4 -- Experiment Manager. MODEL_REGISTRY is the one place a
# model_name string (as would appear in an experiment config file)
# maps to its model class and task kind -- ExperimentRunner.run() never
# hardcodes a specific model beyond this table.
# =====================================================

MODEL_REGISTRY = {
    "evacuation_time": (EvacuationTimeModel, "regression"),
    "bottleneck": (BottleneckModel, "classification"),
    "exit_usage": (ExitUsageModel, "multioutput_regression"),
    "smoke_prediction": (SmokePredictionModel, "classification"),
}


@dataclass
class ExperimentConfig:

    # e.g. Experiment 001: name="exp-001", model_name="evacuation_time",
    # algorithm="random_forest", feature_set="scenario_features".
    # feature_set is informational only (recorded into the saved
    # manifest/model metadata) -- the actual feature source for each
    # model_name is always exactly what that model's own build_table()
    # derives (see each models/*.py module), never a generic mixer, so
    # that Phase 1's "no data leakage" guarantee can never be
    # accidentally defeated by an experiment config.

    name: str
    model_name: str
    algorithm: str = "random_forest"
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    feature_set: str = "scenario_features"
    test_size: float = 0.2
    val_size: float = 0.0
    random_state: int = 0


# =====================================================


@dataclass
class ExperimentResult:

    config: ExperimentConfig
    metrics: Dict[str, Any]
    train_size: int
    test_size_actual: int
    model: Any
    generated_at: str


# =====================================================


class ExperimentRunner:

    # Ties dataset -> model.build_table() -> split -> fit -> evaluate
    # into one call, driven entirely by an ExperimentConfig -- "Run
    # Experiment 001", "Run Experiment 002" (Phase 4) is just invoking
    # this same method with a different config.

    def run(self, dataset: TrainingDataset, config: ExperimentConfig) -> ExperimentResult:

        if config.model_name not in MODEL_REGISTRY:

            raise ValueError(
                f"Unknown model_name {config.model_name!r} -- choose one of "
                f"{sorted(MODEL_REGISTRY)}"
            )

        model_cls, task = MODEL_REGISTRY[config.model_name]
        model_config = {"algorithm": config.algorithm, "algorithm_kwargs": config.algorithm_kwargs}

        if config.model_name == "bottleneck":

            target = config.model_kwargs.get("target", "occurrence")
            model = model_cls(config=model_config, target=target)
            X_rows, y, extra = model_cls.build_table(dataset, target=target)

        else:

            model = model_cls(config=model_config)
            X_rows, y, extra = model_cls.build_table(dataset)

        groups = extra.get("groups")
        output_names = extra.get("output_names")

        split = make_split(
            len(X_rows), groups=groups, test_size=config.test_size,
            val_size=config.val_size, random_state=config.random_state,
        )

        X_train, _X_val, X_test = apply_split(X_rows, split)
        y_train, _y_val, y_test = apply_split(list(y), split)

        if output_names is not None:

            model.fit(X_train, y_train, output_names=output_names)
            metrics = evaluate_multioutput_regression(model, X_test, y_test, output_names)

        elif task == "classification":

            model.fit(X_train, y_train)
            metrics = evaluate_classification(model, X_test, y_test)

        else:

            model.fit(X_train, y_train)
            metrics = evaluate_regression(model, X_test, y_test)

        model.metadata.update({
            "experiment_name": config.name,
            "feature_set": config.feature_set,
            "task": task,
        })

        return ExperimentResult(
            config=config,
            metrics=metrics,
            train_size=len(X_train),
            test_size_actual=len(X_test),
            model=model,
            generated_at=_utc_now_iso(),
        )

    # =====================================================
    # Phase 5 -- persist an experiment's full result: the model bundle
    # (estimator + preprocessing pipeline + feature schema, via
    # BaseModel.save()) plus metrics/config/metadata as a sibling
    # manifest.json, so a saved experiment is exactly reproducible and
    # inspectable without unpickling the model first.
    # =====================================================

    def save_result(self, result: ExperimentResult, directory: str) -> Dict[str, str]:

        os.makedirs(directory, exist_ok=True)

        model_path = os.path.join(directory, "model.joblib")
        manifest_path = os.path.join(directory, "manifest.json")

        result.model.save(model_path)

        manifest = {
            "config": _config_to_dict(result.config),
            "metrics": result.metrics,
            "train_size": result.train_size,
            "test_size_actual": result.test_size_actual,
            "generated_at": result.generated_at,
            "model_metadata": result.model.metadata,
        }

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        return {"model": model_path, "manifest": manifest_path}

    # =====================================================

    def load_result(self, directory: str) -> ExperimentResult:

        manifest_path = os.path.join(directory, "manifest.json")
        model_path = os.path.join(directory, "model.joblib")

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        config = ExperimentConfig(**manifest["config"])
        model_cls, _task = MODEL_REGISTRY[config.model_name]
        model = model_cls.load(model_path)

        return ExperimentResult(
            config=config,
            metrics=manifest["metrics"],
            train_size=manifest["train_size"],
            test_size_actual=manifest["test_size_actual"],
            model=model,
            generated_at=manifest["generated_at"],
        )


# =====================================================


def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:

    return {
        "name": config.name,
        "model_name": config.model_name,
        "algorithm": config.algorithm,
        "algorithm_kwargs": config.algorithm_kwargs,
        "model_kwargs": config.model_kwargs,
        "feature_set": config.feature_set,
        "test_size": config.test_size,
        "val_size": config.val_size,
        "random_state": config.random_state,
    }


def _utc_now_iso() -> str:

    return datetime.datetime.now(datetime.timezone.utc).isoformat()
