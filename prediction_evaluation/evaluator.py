from dataclasses import dataclass
from typing import Mapping, Sequence

from prediction_evaluation.horizon_analysis import DEFAULT_HORIZON_BUCKETS_SECONDS, HorizonBucketResult, analyze_by_horizon
from prediction_evaluation.models import MatchedEvaluation, PredictionRecord
from prediction_evaluation.statistics import GroupStatistics, overall_statistics, per_building_statistics, per_scenario_statistics
from prediction_evaluation.timeline import DEFAULT_TOLERANCE_SECONDS, GroundTruthTimeline, match_predictions, unmatched_predictions


# =====================================================
# Prediction vs Reality Evaluation Framework milestone -- the single,
# convenience entry point tying every phase together into one report,
# mirroring live_stair_perception-style milestones' own "one function a
# caller actually uses, everything else independently testable" shape.
# Never required -- every module this imports is independently usable
# (a caller doing per-model comparison, for instance, uses comparison.py
# directly against several separately-matched evaluation sequences).
# =====================================================


@dataclass(frozen=True)
class EvaluationReport:

    overall: GroupStatistics
    by_horizon: Mapping[float, HorizonBucketResult]
    by_scenario: Mapping[str, GroupStatistics]
    by_building: Mapping[str, GroupStatistics]

    matched_evaluations: Sequence[MatchedEvaluation]
    unmatched_predictions: Sequence[PredictionRecord]


def evaluate(
    predictions: Sequence[PredictionRecord],
    timeline: GroundTruthTimeline,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    horizon_buckets_seconds: Sequence[float] = DEFAULT_HORIZON_BUCKETS_SECONDS,
) -> EvaluationReport:

    matched = match_predictions(predictions, timeline, tolerance_seconds)
    unmatched = unmatched_predictions(predictions, timeline, tolerance_seconds)

    return EvaluationReport(
        overall=overall_statistics(matched),
        by_horizon=analyze_by_horizon(matched, horizon_buckets_seconds),
        by_scenario=per_scenario_statistics(matched),
        by_building=per_building_statistics(matched),
        matched_evaluations=matched,
        unmatched_predictions=unmatched,
    )
