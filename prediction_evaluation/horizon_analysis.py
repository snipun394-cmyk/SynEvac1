from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from prediction_evaluation.classification_metrics import ClassificationMetrics, compute_classification_metrics
from prediction_evaluation.models import MatchedEvaluation
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs
from prediction_evaluation.regression_metrics import RegressionMetrics, compute_regression_metrics


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 5 -- how
# accuracy changes as prediction horizon increases. Groups
# MatchedEvaluations by prediction.prediction_horizon_seconds (the
# horizon a prediction was ACTUALLY made for -- never re-derived from
# match_time_delta_seconds, which is match slack, a different concept)
# into Phase 5's own named buckets, then reuses classification_metrics.py/
# regression_metrics.py unchanged for each bucket -- no separate
# per-horizon metric logic exists anywhere in this package.
# =====================================================


DEFAULT_HORIZON_BUCKETS_SECONDS: Tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 60.0)


@dataclass(frozen=True)
class HorizonBucketResult:

    horizon_seconds: float
    evaluation_count: int
    classification: ClassificationMetrics
    regression: RegressionMetrics


def analyze_by_horizon(
    evaluations: Sequence[MatchedEvaluation],
    horizon_buckets_seconds: Sequence[float] = DEFAULT_HORIZON_BUCKETS_SECONDS,
) -> Mapping[float, HorizonBucketResult]:

    grouped: Dict[float, list] = {bucket: [] for bucket in horizon_buckets_seconds}

    for evaluation in evaluations:

        bucket = _nearest_bucket(evaluation.prediction.prediction_horizon_seconds, horizon_buckets_seconds)

        if bucket is not None:
            grouped[bucket].append(evaluation)

    results = {}

    for bucket, bucket_evaluations in grouped.items():

        y_true_cls, y_pred_cls, y_proba_cls = bottleneck_classification_pairs(bucket_evaluations)
        y_true_reg, y_pred_reg = evacuation_time_regression_pairs(bucket_evaluations)

        results[bucket] = HorizonBucketResult(
            horizon_seconds=bucket,
            evaluation_count=len(bucket_evaluations),
            classification=compute_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls),
            regression=compute_regression_metrics(y_true_reg, y_pred_reg),
        )

    return results


def _nearest_bucket(horizon_seconds: float, buckets: Sequence[float], tolerance: float = 1e-6):

    # Exact-match-only by default (a prediction's own horizon should
    # already be one of the configured buckets, e.g. 20.0) -- but
    # honestly falls back to the nearest bucket within a tiny floating-
    # point tolerance rather than silently dropping a prediction whose
    # horizon is 20.0000001 due to accumulated arithmetic.

    for bucket in buckets:

        if abs(bucket - horizon_seconds) <= tolerance:
            return bucket

    return None
