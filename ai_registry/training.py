import datetime
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.building import Building

from ai_training.dataset import TrainingDataset
import ai_training as at
from ai_training.evaluation import evaluate_classification, evaluate_regression
from ai_training.metrics import classification_metrics, regression_metrics
from ai_training.models.bottleneck_model import BottleneckModel
from ai_training.models.evacuation_time_model import EvacuationTimeModel
from ai_training.split import apply_split, make_split

from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION

from ai_registry.baselines import bottleneck_occurrence_baselines, class_balance, evacuation_time_baselines
from ai_registry.campaign import build_live_compatible_dataset
from ai_registry.metadata import Deployability, ModelMetadata, save_live_model
from ai_registry.uncertainty import probability_calibration_report, regression_ensemble_uncertainty


class ModelStatus:

    # Phase 7/9's own required, honest self-classification -- a model
    # never gets called PRODUCTION_CANDIDATE merely for existing; it
    # must actually beat its own trivial baseline on held-out data.

    PRODUCTION_CANDIDATE = "PRODUCTION_CANDIDATE"
    EXPERIMENTAL = "EXPERIMENTAL"


def evacuation_time_model_status(test_metrics: Dict[str, float], baseline_metrics: Dict[str, Dict[str, float]]) -> str:

    best_baseline_mae = min(entry["mae"] for entry in baseline_metrics.values())

    if test_metrics["r2"] > 0.0 and test_metrics["mae"] < best_baseline_mae:
        return ModelStatus.PRODUCTION_CANDIDATE

    return ModelStatus.EXPERIMENTAL


def bottleneck_model_status(test_metrics: Dict[str, Any], baseline_metrics: Dict[str, Dict[str, Any]]) -> str:

    best_baseline_f1 = max(entry["f1"] for entry in baseline_metrics.values())

    if test_metrics["f1"] > best_baseline_f1 and (test_metrics.get("roc_auc") or 0.0) > 0.5:
        return ModelStatus.PRODUCTION_CANDIDATE

    return ModelStatus.EXPERIMENTAL


MISSING_DATA_POLICY = (
    "None (never a fabricated zero) for any field whose *_available/*_observed companion flag "
    "is False; coverage counts always accompany condition counts. See "
    "docs/architecture/ai_live_feature_parity.md §8 for the full policy."
)


# =====================================================
# Phase 3's reproducible training pipeline, and Phase 5's scenario-level
# split. `EvacuationTimeModel`/`BottleneckModel` (ai_training/models/)
# are reused completely unmodified -- this module only supplies THEIR
# `build_table()` with a live-compatible TrainingDataset instead of the
# legacy one, and adds the metadata/baseline/uncertainty reporting
# Phase 2/6/9 ask for around that unchanged core.
# =====================================================


def generate_training_campaign(output_dir: str, building: Building, definition, *, count: int, master_seed: int) -> str:

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from PyQt6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)

    from designer.campaign.campaign_worker import CampaignConfig, CampaignWorker

    config = CampaignConfig(
        name="live-ai-training-campaign", building=building, definition=definition,
        definition_id="def-live-ai-training", count=count, master_seed=master_seed,
        output_directory=output_dir, max_attempts=10, dt=10.0,
    )

    summary = CampaignWorker(config).execute()

    if summary.accepted == 0:

        raise RuntimeError(
            f"Campaign generation produced 0 accepted scenarios out of {summary.total_requested} "
            f"requested -- {summary.rejection_explanation or 'no diagnostics available'}."
        )

    return summary.output_directory, summary


# =====================================================


@dataclass
class ScenarioSplitCounts:

    train: int
    validation: int
    test: int


@dataclass
class TrainedModelResult:

    model: Any
    metadata: ModelMetadata
    baseline_metrics: Dict[str, Any]
    scenario_split_counts: ScenarioSplitCounts
    extra_evaluation: Dict[str, Any] = field(default_factory=dict)


# =====================================================


def train_evacuation_time_model(
    live_dataset: TrainingDataset,
    *,
    training_seed: int,
    dataset_identifier: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> TrainedModelResult:

    X_rows, y, _extra = EvacuationTimeModel.build_table(live_dataset)
    groups = list(live_dataset.scenario_ids)

    split = make_split(len(X_rows), groups=groups, test_size=test_size, val_size=val_size, random_state=training_seed)

    X_train, X_val, X_test = apply_split(X_rows, split)
    y_train, y_val, y_test = apply_split(list(y), split)

    model = EvacuationTimeModel(config={"algorithm": "random_forest"})
    model.fit(X_train, y_train)

    train_metrics = regression_metrics(y_train, model.predict(X_train))
    validation_metrics = evaluate_regression(model, X_val, y_val) if X_val else {}
    test_metrics = evaluate_regression(model, X_test, y_test)

    baseline_metrics = evacuation_time_baselines(y_train, y_test)

    uncertainty = regression_ensemble_uncertainty(model, X_test)

    error_breakdown = _evacuation_time_error_breakdown(X_test, y_test, model.predict(X_test))

    model_status = evacuation_time_model_status(test_metrics, baseline_metrics)

    version_stamp = _version_stamp()

    metadata = ModelMetadata(
        model_id=f"EvacuationTimeModel_LiveCompatible-{version_stamp}",
        model_type="evacuation_time",
        model_version=version_stamp,
        training_timestamp=_utc_now_iso(),
        training_dataset_identifier=dataset_identifier,
        training_seed=training_seed,
        feature_schema_version=SCHEMA_VERSION,
        ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
        prediction_target="total_evacuation_time (seconds, from simulation/incident start)",
        model_deployability=Deployability.LIVE_COMPATIBLE,
        training_metrics=train_metrics,
        validation_metrics={**validation_metrics, "test": test_metrics},
        missing_data_policy=MISSING_DATA_POLICY,
    )

    return TrainedModelResult(
        model=model,
        metadata=metadata,
        baseline_metrics=baseline_metrics,
        scenario_split_counts=ScenarioSplitCounts(
            train=len(split.train_indices), validation=len(split.val_indices), test=len(split.test_indices),
        ),
        extra_evaluation={
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "model_status": model_status,
            "uncertainty_available": uncertainty is not None,
            "uncertainty_sample": uncertainty[:10] if uncertainty else None,
            "error_breakdown_by_occupancy": error_breakdown,
            "hazard_severity_breakdown": "not applicable -- hazard severity is excluded from the "
                                          "canonical live-compatible schema (SIMULATION_ONLY, see "
                                          "docs/architecture/ai_live_feature_parity.md §3)",
            "building_complexity_breakdown": "not applicable -- this campaign trains against one "
                                              "fixed Building; no building-complexity variation exists "
                                              "to break performance down by",
        },
    )


# =====================================================


def train_bottleneck_occurrence_model(
    live_dataset: TrainingDataset,
    *,
    training_seed: int,
    dataset_identifier: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> TrainedModelResult:

    X_rows, y, extra = BottleneckModel.build_table(live_dataset, target="occurrence")
    groups = extra.get("groups") or list(live_dataset.scenario_ids)

    split = make_split(len(X_rows), groups=groups, test_size=test_size, val_size=val_size, random_state=training_seed)

    X_train, X_val, X_test = apply_split(X_rows, split)
    y_train, y_val, y_test = apply_split(list(y), split)

    model = BottleneckModel(config={"algorithm": "random_forest"}, target="occurrence")
    model.fit(X_train, y_train)

    train_metrics = classification_metrics(y_train, model.predict(X_train), model.predict_proba(X_train))
    validation_metrics = evaluate_classification(model, X_val, y_val) if X_val else {}
    test_metrics = evaluate_classification(model, X_test, y_test)

    baseline_metrics = bottleneck_occurrence_baselines(y_train, y_test)

    train_balance = class_balance(y_train)
    test_balance = class_balance(y_test)

    y_proba_test = model.predict_proba(X_test)
    positive_class_index = list(model.label_encoder.classes_).index(True) if True in list(model.label_encoder.classes_) else 1
    calibration = probability_calibration_report(
        [bool(value) for value in y_test], y_proba_test[:, positive_class_index],
    )

    confusion = _confusion_matrix(y_test, model.predict(X_test))

    pr_auc = _pr_auc(y_test, y_proba_test[:, positive_class_index])

    model_status = bottleneck_model_status(test_metrics, baseline_metrics)

    version_stamp = _version_stamp()

    metadata = ModelMetadata(
        model_id=f"BottleneckOccurrenceModel_LiveCompatible-{version_stamp}",
        model_type="bottleneck_occurrence",
        model_version=version_stamp,
        training_timestamp=_utc_now_iso(),
        training_dataset_identifier=dataset_identifier,
        training_seed=training_seed,
        feature_schema_version=SCHEMA_VERSION,
        ordered_feature_names=CANONICAL_LIVE_FEATURE_NAMES,
        prediction_target="bottleneck_occurrence (bool(GroundTruth.doors_that_became_bottlenecks))",
        model_deployability=Deployability.LIVE_COMPATIBLE,
        training_metrics=train_metrics,
        validation_metrics={**validation_metrics, "test": test_metrics},
        missing_data_policy=MISSING_DATA_POLICY,
    )

    return TrainedModelResult(
        model=model,
        metadata=metadata,
        baseline_metrics=baseline_metrics,
        scenario_split_counts=ScenarioSplitCounts(
            train=len(split.train_indices), validation=len(split.val_indices), test=len(split.test_indices),
        ),
        extra_evaluation={
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": {**test_metrics, "pr_auc": pr_auc},
            "model_status": model_status,
            "train_class_balance": train_balance,
            "test_class_balance": test_balance,
            "confusion_matrix": confusion,
            "probability_calibration": calibration,
        },
    )


def _pr_auc(y_true, y_proba_positive) -> Optional[float]:

    from sklearn.metrics import average_precision_score

    y_true_bool = [bool(v) for v in y_true]

    if len(set(y_true_bool)) < 2:
        return None

    return float(average_precision_score(y_true_bool, y_proba_positive))


# =====================================================


def _evacuation_time_error_breakdown(X_test, y_test, y_pred) -> Dict[str, Any]:

    import numpy as np

    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_test_arr - y_pred_arr)

    occupancy = np.array([row.get("total_occupant_count") or 0 for row in X_test])

    buckets = [("low_0_5", occupancy <= 5), ("medium_6_15", (occupancy > 5) & (occupancy <= 15)), ("high_16_plus", occupancy > 15)]

    by_bucket = {}
    for name, mask in buckets:

        if mask.sum() == 0:
            by_bucket[name] = {"count": 0}
            continue

        by_bucket[name] = {
            "count": int(mask.sum()),
            "mae": float(errors[mask].mean()),
        }

    worst_indices = np.argsort(errors)[::-1][:5]

    return {
        "by_occupancy_range": by_bucket,
        "worst_case_errors": [
            {"absolute_error": float(errors[i]), "y_true": float(y_test_arr[i]), "y_pred": float(y_pred_arr[i])}
            for i in worst_indices
        ],
        "mean_absolute_error": float(errors.mean()),
        "max_absolute_error": float(errors.max()) if len(errors) else None,
    }


def _confusion_matrix(y_true, y_pred) -> Dict[str, int]:

    y_true = [bool(v) for v in y_true]
    y_pred = [bool(v) for v in y_pred]

    return {
        "true_positive": sum(1 for t, p in zip(y_true, y_pred) if t and p),
        "true_negative": sum(1 for t, p in zip(y_true, y_pred) if not t and not p),
        "false_positive": sum(1 for t, p in zip(y_true, y_pred) if not t and p),
        "false_negative": sum(1 for t, p in zip(y_true, y_pred) if t and not p),
    }


def _utc_now_iso() -> str:

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _version_stamp() -> str:

    return datetime.datetime.now(datetime.timezone.utc).strftime("v%Y%m%d%H%M%S")
