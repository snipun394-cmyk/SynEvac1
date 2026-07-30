from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from prediction_evaluation.classification_metrics import ClassificationMetrics, compute_classification_metrics
from prediction_evaluation.models import MatchedEvaluation
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs
from prediction_evaluation.regression_metrics import RegressionMetrics, compute_regression_metrics


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 6 --
# performance under different building conditions. This package defines
# no closed vocabulary of conditions -- PredictionRecord.context_tags is
# free-form key/value metadata a caller attaches when recording a
# prediction (e.g. from a Scenario Generator definition, or a live
# session's own known configuration). The well-known keys below are
# documented SUGGESTIONS matching Phase 6's own named categories, never
# enforced or validated -- analyze_by_condition() groups by whichever
# key a caller actually asks for.
# =====================================================


# Suggested tag keys/values -- purely documentation, never referenced by
# code below (analyze_by_condition() accepts any key a caller supplies).
SUGGESTED_TAG_KEYS = (
    "occupancy_level",       # "low" | "medium" | "high"
    "floor_mode",            # "single_floor" | "multi_floor"
    "exit_status",           # "all_open" | "blocked"
    "congestion_level",      # "none" | "moderate" | "heavy"
    "fire_origin",           # a zone_id, or "none"
    "building_layout",       # a topology/family name
)


@dataclass(frozen=True)
class ConditionGroupResult:

    tag_key: str
    tag_value: str
    evaluation_count: int
    classification: ClassificationMetrics
    regression: RegressionMetrics


def analyze_by_condition(
    evaluations: Sequence[MatchedEvaluation], tag_key: str,
) -> Mapping[str, ConditionGroupResult]:

    grouped: Dict[str, list] = {}

    for evaluation in evaluations:

        value = evaluation.prediction.context_tags.get(tag_key)

        if value is None:
            continue

        grouped.setdefault(value, []).append(evaluation)

    results = {}

    for value, group_evaluations in grouped.items():

        y_true_cls, y_pred_cls, y_proba_cls = bottleneck_classification_pairs(group_evaluations)
        y_true_reg, y_pred_reg = evacuation_time_regression_pairs(group_evaluations)

        results[value] = ConditionGroupResult(
            tag_key=tag_key, tag_value=value, evaluation_count=len(group_evaluations),
            classification=compute_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls),
            regression=compute_regression_metrics(y_true_reg, y_pred_reg),
        )

    return results
