from typing import List, Sequence, Tuple

from prediction_evaluation.models import MatchedEvaluation


# =====================================================
# Prediction vs Reality Evaluation Framework milestone -- the ONE place
# a (predicted, actual) pair is pulled out of a MatchedEvaluation's own
# generic .prediction.payload/.ground_truth. Duck-typed against
# live_system.live_ai_gateway.LiveAIPredictionSnapshot's own real shape
# (never hard-imported -- same "no forced import" discipline as
# evacuation_recommendation.evidence.ai_bottleneck_probability()) --
# these are the TWO model outputs that genuinely exist in this codebase
# today (Shadow-Mode Predictive AI Integration milestone's own trace:
# bottleneck_occurrence classification, evacuation_time regression). A
# future model type would need its own extractor here, mirroring this
# exact pattern -- classification_metrics.py/regression_metrics.py
# themselves are already fully generic and need no change to support it.
#
# Every extractor below silently DROPS an evaluation where either the
# predicted or the actual value is unavailable (None) -- Phase 4's own
# metrics must never be computed against a fabricated pair. This is not
# hidden: metrics.py's own result types report how many evaluations
# actually contributed vs. how many were skipped.
# =====================================================


def bottleneck_classification_pairs(
    evaluations: Sequence[MatchedEvaluation],
) -> Tuple[List[bool], List[bool], List[float]]:

    # Returns (y_true, y_pred, y_proba) -- y_proba is ALWAYS populated
    # alongside y_pred (BottleneckOccurrencePrediction always carries
    # both together), never a separate, shorter list.

    y_true: List[bool] = []
    y_pred: List[bool] = []
    y_proba: List[float] = []

    for evaluation in evaluations:

        actual = evaluation.ground_truth.congestion_detected

        if actual is None:
            continue

        bottleneck = getattr(evaluation.prediction.payload, "bottleneck", None)

        if bottleneck is None:
            continue

        predicted = getattr(bottleneck, "predicted_occurrence", None)
        probability = getattr(bottleneck, "probability", None)

        if predicted is None or probability is None:
            continue

        y_true.append(bool(actual))
        y_pred.append(bool(predicted))
        y_proba.append(float(probability))

    return y_true, y_pred, y_proba


def evacuation_time_regression_pairs(
    evaluations: Sequence[MatchedEvaluation],
) -> Tuple[List[float], List[float]]:

    # Returns (y_true, y_pred).

    y_true: List[float] = []
    y_pred: List[float] = []

    for evaluation in evaluations:

        actual = evaluation.ground_truth.evacuation_time_seconds

        if actual is None:
            continue

        evacuation_time = getattr(evaluation.prediction.payload, "evacuation_time_experimental", None)

        if evacuation_time is None:
            continue

        predicted = getattr(evacuation_time, "predicted_seconds", None)

        if predicted is None:
            continue

        y_true.append(float(actual))
        y_pred.append(float(predicted))

    return y_true, y_pred
