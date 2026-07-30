from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from prediction_evaluation.classification_metrics import ClassificationMetrics, compute_classification_metrics
from prediction_evaluation.models import MatchedEvaluation
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs
from prediction_evaluation.regression_metrics import RegressionMetrics, compute_regression_metrics


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 8 --
# statistical summaries. Mean/median/95% CI/worst-case/best-case/
# standard deviation are ALREADY every field on regression_metrics.
# RegressionMetrics (Phase 4's own computation, reused here verbatim,
# never recomputed a second way) -- this module's own job is purely the
# GROUPING Phase 8 additionally asks for (per-building, per-scenario),
# which condition_analysis.py's own per-tag grouping does not cover
# (scenario_id/building_id are dedicated PredictionRecord fields, not
# free-form context_tags).
# =====================================================


@dataclass(frozen=True)
class GroupStatistics:

    group_key: str
    evaluation_count: int
    classification: ClassificationMetrics
    regression: RegressionMetrics


def _group_and_compute(evaluations: Sequence[MatchedEvaluation], key_fn) -> Mapping[str, GroupStatistics]:

    grouped: Dict[str, list] = {}

    for evaluation in evaluations:

        key = key_fn(evaluation)

        if key is None:
            continue

        grouped.setdefault(key, []).append(evaluation)

    results = {}

    for key, group_evaluations in grouped.items():

        y_true_cls, y_pred_cls, y_proba_cls = bottleneck_classification_pairs(group_evaluations)
        y_true_reg, y_pred_reg = evacuation_time_regression_pairs(group_evaluations)

        results[key] = GroupStatistics(
            group_key=key, evaluation_count=len(group_evaluations),
            classification=compute_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls),
            regression=compute_regression_metrics(y_true_reg, y_pred_reg),
        )

    return results


def per_scenario_statistics(evaluations: Sequence[MatchedEvaluation]) -> Mapping[str, GroupStatistics]:

    return _group_and_compute(evaluations, lambda e: e.prediction.scenario_id)


def per_building_statistics(evaluations: Sequence[MatchedEvaluation]) -> Mapping[str, GroupStatistics]:

    return _group_and_compute(evaluations, lambda e: e.prediction.building_id)


def overall_statistics(evaluations: Sequence[MatchedEvaluation]) -> GroupStatistics:

    y_true_cls, y_pred_cls, y_proba_cls = bottleneck_classification_pairs(evaluations)
    y_true_reg, y_pred_reg = evacuation_time_regression_pairs(evaluations)

    return GroupStatistics(
        group_key="overall", evaluation_count=len(evaluations),
        classification=compute_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls),
        regression=compute_regression_metrics(y_true_reg, y_pred_reg),
    )
