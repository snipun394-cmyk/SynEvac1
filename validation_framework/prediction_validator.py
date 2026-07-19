from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_training.dataset import TrainingDataset
from ai_training.experiment import ExperimentConfig, ExperimentRunner, MODEL_REGISTRY
from ai_training.split import apply_split, make_split

from ai_explainability.comparison import (
    compare_bottleneck_location_to_decision_policy,
    summarize_comparisons,
)

from validation_framework.statistics import BootstrapResult, SensitivityResult, bootstrap_confidence_interval, feature_sensitivity


# =====================================================
# Phase 2 -- AI Evaluation. Every function below trains/evaluates
# through ai_training.experiment.ExperimentRunner (fit -> split ->
# evaluate, exactly as ai_training itself already does it) -- no
# hand-rolled fit/predict loop, no new model, no new split logic.
# y_true/y_pred are re-derived on the SAME held-out split
# (make_split/apply_split with the same config the runner used) purely
# so figures.py has arrays to plot; this is a deterministic
# recomputation, not a second independent split.
# =====================================================


@dataclass(frozen=True)
class PredictionValidationResult:

    name: str
    metrics: Dict[str, Any]
    bootstrap_ci: Dict[str, BootstrapResult]
    model: Any
    y_true: Sequence[Any]
    y_pred: Sequence[Any]
    y_proba: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "metrics": self.metrics,
            "bootstrap_ci": {key: value.to_dict() for key, value in self.bootstrap_ci.items()},
        }


def _held_out_rows(dataset: TrainingDataset, model_cls, config: ExperimentConfig, **build_kwargs):

    X_rows, y, extra = model_cls.build_table(dataset, **build_kwargs)
    groups = extra.get("groups")

    split = make_split(
        len(X_rows), groups=groups, test_size=config.test_size,
        val_size=config.val_size, random_state=config.random_state,
    )

    _X_train, _X_val, X_test = apply_split(X_rows, split)
    _y_train, _y_val, y_test = apply_split(list(y), split)

    return X_test, y_test, extra


def validate_evacuation_time_model(
    dataset: TrainingDataset, config: Optional[ExperimentConfig] = None,
) -> PredictionValidationResult:

    config = config or ExperimentConfig(name="validate-evacuation-time", model_name="evacuation_time")
    result = ExperimentRunner().run(dataset, config)

    model_cls, _task = MODEL_REGISTRY["evacuation_time"]
    X_test, y_test, _extra = _held_out_rows(dataset, model_cls, config)
    y_pred = list(result.model.predict(X_test))

    absolute_errors = [abs(float(a) - float(b)) for a, b in zip(y_test, y_pred)]

    return PredictionValidationResult(
        name="evacuation_time", metrics=result.metrics,
        bootstrap_ci={"mae": bootstrap_confidence_interval(absolute_errors)},
        model=result.model, y_true=y_test, y_pred=y_pred,
    )


def validate_bottleneck_model(
    dataset: TrainingDataset, *, target: str = "occurrence", config: Optional[ExperimentConfig] = None,
) -> PredictionValidationResult:

    config = config or ExperimentConfig(
        name=f"validate-bottleneck-{target}", model_name="bottleneck", model_kwargs={"target": target},
    )
    result = ExperimentRunner().run(dataset, config)

    model_cls, _task = MODEL_REGISTRY["bottleneck"]
    X_test, y_test, _extra = _held_out_rows(dataset, model_cls, config, target=target)
    y_pred = list(result.model.predict(X_test))

    correct = [1.0 if a == b else 0.0 for a, b in zip(y_test, y_pred)]

    y_proba = None
    try:
        y_proba = result.model.predict_proba(X_test)
    except (AttributeError, TypeError):
        y_proba = None

    return PredictionValidationResult(
        name=f"bottleneck_{target}", metrics=result.metrics,
        bootstrap_ci={"accuracy": bootstrap_confidence_interval(correct)},
        model=result.model, y_true=y_test, y_pred=y_pred, y_proba=y_proba,
    )


def validate_exit_usage_model(
    dataset: TrainingDataset, config: Optional[ExperimentConfig] = None,
) -> PredictionValidationResult:

    config = config or ExperimentConfig(name="validate-exit-usage", model_name="exit_usage")
    result = ExperimentRunner().run(dataset, config)

    model_cls, _task = MODEL_REGISTRY["exit_usage"]
    X_test, y_test, extra = _held_out_rows(dataset, model_cls, config)
    output_names = extra["output_names"]

    predictions = result.model.predict(X_test)
    y_pred = [[row[name] for name in output_names] for row in predictions]

    row_mae = [
        sum(abs(float(a) - float(b)) for a, b in zip(true_row, pred_row)) / max(1, len(true_row))
        for true_row, pred_row in zip(y_test, y_pred)
    ]

    return PredictionValidationResult(
        name="exit_usage", metrics=result.metrics,
        bootstrap_ci={"mae": bootstrap_confidence_interval(row_mae)},
        model=result.model, y_true=y_test, y_pred=y_pred,
    )


def validate_smoke_prediction_model(
    dataset: TrainingDataset, config: Optional[ExperimentConfig] = None,
) -> PredictionValidationResult:

    config = config or ExperimentConfig(name="validate-smoke-prediction", model_name="smoke_prediction")
    result = ExperimentRunner().run(dataset, config)

    model_cls, _task = MODEL_REGISTRY["smoke_prediction"]
    X_test, y_test, _extra = _held_out_rows(dataset, model_cls, config)
    y_pred = list(result.model.predict(X_test))

    correct = [1.0 if a == b else 0.0 for a, b in zip(y_test, y_pred)]

    y_proba = None
    try:
        y_proba = result.model.predict_proba(X_test)
    except (AttributeError, TypeError):
        y_proba = None

    return PredictionValidationResult(
        name="smoke_prediction", metrics=result.metrics,
        bootstrap_ci={"accuracy": bootstrap_confidence_interval(correct)},
        model=result.model, y_true=y_test, y_pred=y_pred, y_proba=y_proba,
    )


def validate_recommendation_accuracy(
    dataset: TrainingDataset, config: Optional[ExperimentConfig] = None,
) -> Dict[str, Any]:

    # "Recommendation Accuracy" is measured as agreement between the AI
    # bottleneck-location model's own prediction and Decision Policy's
    # independent, rule-based risk flagging for that same location --
    # ai_explainability.comparison already implements exactly this
    # comparison, reused here rather than reimplemented.

    config = config or ExperimentConfig(
        name="validate-recommendation-accuracy", model_name="bottleneck", model_kwargs={"target": "location"},
    )
    result = ExperimentRunner().run(dataset, config)

    records = compare_bottleneck_location_to_decision_policy(result.model, dataset)

    return summarize_comparisons(records)


def validate_firefighter_intelligence_consistency(
    dataset: TrainingDataset, *, sample_limit: int = 50,
) -> Dict[str, Any]:

    # Monotonicity half of "Firefighter Intelligence Consistency": does
    # firefighter_deployment_priority's rank order actually correlate
    # with the independently-computed zone_risk_scores it should be
    # driven by? Purely from already-exported ground_truth/decision_
    # policy dicts -- no live Building/Scenario needed. The determinism
    # half (does re-running generate_policy() on the same GroundTruth
    # twice produce the same priority order) requires live Building/
    # Scenario objects that exported campaign artifacts do not retain,
    # so it lives in recommendation_validator.py instead, where those
    # objects are already in hand from a fresh simulation run.

    correlations: List[SensitivityResult] = []
    samples_checked = 0

    for record in list(dataset)[:sample_limit]:

        if record.ground_truth is None or record.decision_policy is None:
            continue

        risk_by_zone = {
            entry.get("zone_id"): entry.get("risk_score")
            for entry in record.ground_truth.get("zone_risk_scores", [])
            if entry.get("risk_score") is not None
        }

        priority_entries = [
            entry for entry in record.decision_policy.get("firefighter_deployment_priority", [])
            if entry.get("target_type") == "zone"
        ]

        risks: List[float] = []
        inverted_ranks: List[float] = []

        for entry in priority_entries:

            zone_id = entry.get("target_id")

            if zone_id in risk_by_zone:
                risks.append(float(risk_by_zone[zone_id]))
                # Negated so a higher value means "more urgent" (rank 1
                # is the most urgent) -- a positive correlation with risk
                # is then the expected, healthy relationship.
                inverted_ranks.append(-float(entry.get("rank", 0)))

        samples_checked += 1

        if len(risks) >= 2 and len(set(risks)) >= 2:
            correlations.append(feature_sensitivity(risks, inverted_ranks))

    valid_r_values = [c.pearson_r for c in correlations if c.pearson_r is not None]

    return {
        "samples_checked": samples_checked,
        "samples_with_correlation": len(correlations),
        "mean_risk_priority_correlation": (
            sum(valid_r_values) / len(valid_r_values) if valid_r_values else None
        ),
        "monotonic": (
            (sum(valid_r_values) / len(valid_r_values)) > 0.0 if valid_r_values else None
        ),
    }
