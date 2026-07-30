from dataclasses import dataclass
from typing import Mapping, Sequence

from prediction_evaluation.classification_metrics import ClassificationMetrics, compute_classification_metrics
from prediction_evaluation.models import MatchedEvaluation
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs
from prediction_evaluation.regression_metrics import RegressionMetrics, compute_regression_metrics


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 9 -- model
# comparison. `evaluations_by_model` is keyed by whatever label a caller
# used (typically PredictionRecord.model_id, but never enforced -- a
# caller comparing two experimental checkpoints of the same registered
# model_id can key by model_version instead). Every model's own
# MatchedEvaluation sequence should already share the SAME underlying
# scenarios (a caller's own responsibility -- this module never verifies
# scenario identity, since that is defined by however the caller chose
# to run the comparison, e.g. re-running prediction against the SAME
# recorded ground-truth timeline for two different models).
# =====================================================


@dataclass(frozen=True)
class ModelResult:

    model_label: str
    evaluation_count: int
    classification: ClassificationMetrics
    regression: RegressionMetrics


@dataclass(frozen=True)
class ModelComparisonReport:

    results_by_model: Mapping[str, ModelResult]

    def better_classifier(self) -> "str | None":

        # Ranks by F1 (a single, disclosed, balanced criterion) among
        # models that actually produced classification metrics -- None
        # if fewer than 2 models have a comparable F1 score (never an
        # arbitrary pick).
        candidates = [
            (label, result.classification.f1) for label, result in self.results_by_model.items()
            if result.classification.f1 is not None
        ]

        if len(candidates) < 2:
            return None

        return max(candidates, key=lambda pair: pair[1])[0]

    def better_regressor(self) -> "str | None":

        # Ranks by MAE ascending (lower is better).
        candidates = [
            (label, result.regression.mae) for label, result in self.results_by_model.items()
            if result.regression.mae is not None
        ]

        if len(candidates) < 2:
            return None

        return min(candidates, key=lambda pair: pair[1])[0]


def compare_models(evaluations_by_model: Mapping[str, Sequence[MatchedEvaluation]]) -> ModelComparisonReport:

    results = {}

    for label, evaluations in evaluations_by_model.items():

        y_true_cls, y_pred_cls, y_proba_cls = bottleneck_classification_pairs(evaluations)
        y_true_reg, y_pred_reg = evacuation_time_regression_pairs(evaluations)

        results[label] = ModelResult(
            model_label=label, evaluation_count=len(evaluations),
            classification=compute_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls),
            regression=compute_regression_metrics(y_true_reg, y_pred_reg),
        )

    return ModelComparisonReport(results_by_model=results)
