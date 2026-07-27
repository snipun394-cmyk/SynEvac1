from typing import Any, Dict, Tuple

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.simulation_extractor import (
    _current_approaching_count,
    _current_queue_length,
    extract_simulation_candidate_features,
)

# =====================================================
# Localized Predictive Model V2.1 milestone, Phase 11 -- EXPERIMENTAL
# feature extension. Does NOT modify predictive_dataset/schema.py or
# simulation_extractor.py (V1/V2's frozen 9-field schema stays exactly
# as it is); this is an ADDITIVE module producing 3 new candidate-local/
# structural fields on top of the existing ones, for one targeted
# hypothesis-test campaign only (docs/architecture/
# localized_predictive_model_v2_exit_generalization_investigation.md).
#
# Each new field is designed against that investigation's own evidence:
#   - candidate_recent_flow_rate: addresses the "no throughput/rate
#     signal" gap (Phase 4/5) -- live analog already exists,
#     evacuation_progress.ExitFlow.recent_flow_per_minute.
#   - candidate_congestion_trend: addresses the "no rate-of-change"
#     gap -- live analog already exists for every candidate type,
#     crowd_intelligence.trends.TrendTracker (AssetApproachMetrics.trend).
#   - candidate_alternative_route_count: addresses the structural-
#     context gap (Phase 6/7) -- live analog already exists for Exit,
#     evacuation_recommendation.ranking's alternative_exit_ids concept,
#     generalized here to every candidate type via the same
#     CandidateIdentity.zone_ids the frozen schema already resolves.
#
# LEAKAGE DISCIPLINE (same as simulation_extractor.py itself): every
# read here is gated by an "already realized as of `time`" test --
# recent_flow_rate only counts COMPLETED crossings at-or-before `time`;
# congestion_trend only compares `time` against `time - trend_window`
# (strictly the past); alternative_route_count is static structural
# data with zero occupancy/time dependence at all. None of these ever
# inspect (time, time+horizon].
# =====================================================

FLOW_WINDOW_SECONDS = 60.0  # matches evacuation_progress's own recent-flow window
TREND_WINDOW_SECONDS = 30.0  # matches crowd_intelligence.trends.TrendTracker's default
TREND_STABLE_MARGIN = 1.0  # demand-count units; smaller than this change is "STABLE"


def extract_experimental_candidate_features(
    candidate: CandidateIdentity,
    edge,
    time: float,
    *,
    building,
    movement_result,
    occupancy_snapshot,
    alternative_route_counts: Dict[str, int],
) -> Dict[str, Any]:
    """The frozen V2 9-field feature dict, PLUS the 3 experimental fields.
    `alternative_route_counts` is precomputed once per Building via
    `build_alternative_route_counts()` below -- never recomputed per row
    (it is occupancy/time-independent, so doing so would be wasted work,
    not a correctness issue)."""

    base = extract_simulation_candidate_features(
        candidate, edge, time,
        building=building, movement_result=movement_result, occupancy_snapshot=occupancy_snapshot,
    )

    base["candidate_recent_flow_rate"] = _recent_flow_rate(movement_result, candidate.candidate_id, time)
    base["candidate_congestion_trend"] = _congestion_trend(movement_result, candidate.candidate_id, time)
    base["candidate_alternative_route_count"] = alternative_route_counts.get(candidate.candidate_id, 0)

    return base


# =====================================================
# candidate_recent_flow_rate -- count of occupants who COMPLETED
# crossing this candidate's edge during (time - FLOW_WINDOW_SECONDS,
# time]. A pure "already happened" count, the same discipline
# simulation_extractor._current_queue_length/_current_approaching_count
# already use -- never fabricated, never dependent on (time, time+horizon].
# =====================================================


def _recent_flow_rate(movement_result, candidate_id: str, time: float, window: float = FLOW_WINDOW_SECONDS) -> int:

    window_start = time - window
    count = 0

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            if window_start < step.end_time <= time:
                count += 1

    return count


# =====================================================
# candidate_congestion_trend -- RISING/STABLE/FALLING/UNKNOWN, comparing
# a demand proxy (queue_length + approaching_count, the SAME two-signal
# combination candidate_congestion_level is already derived from) at
# `time` against the SAME candidate at `time - TREND_WINDOW_SECONDS`.
# UNKNOWN when there is no earlier observation to compare against
# (time < TREND_WINDOW_SECONDS) -- never fabricated as STABLE.
# =====================================================


def _demand_proxy(movement_result, candidate_id: str, time: float) -> int:
    return (
        _current_queue_length(movement_result, candidate_id, time)
        + _current_approaching_count(movement_result, candidate_id, time)
    )


def _congestion_trend(movement_result, candidate_id: str, time: float, window: float = TREND_WINDOW_SECONDS) -> str:

    if time < window:
        return "UNKNOWN"

    earlier = _demand_proxy(movement_result, candidate_id, time - window)
    now = _demand_proxy(movement_result, candidate_id, time)

    delta = now - earlier

    if delta > TREND_STABLE_MARGIN:
        return "RISING"
    if delta < -TREND_STABLE_MARGIN:
        return "FALLING"
    return "STABLE"


# =====================================================
# candidate_alternative_route_count -- purely structural: for a given
# Building, how many OTHER candidates (any Door/Exit/Stair) share at
# least one zone_id with this one. Computed ONCE per Building (never
# per-row) from predictive_dataset.candidate.enumerate_candidates()'s
# own already-resolved zone_ids -- zero occupancy/time dependence, so
# zero leakage risk by construction.
# =====================================================


def build_alternative_route_counts(candidates: Tuple[CandidateIdentity, ...]) -> Dict[str, int]:

    counts: Dict[str, int] = {}

    for candidate in candidates:

        own_zones = set(candidate.zone_ids)
        if not own_zones:
            counts[candidate.candidate_id] = 0
            continue

        alternatives = 0
        for other in candidates:
            if other.candidate_id == candidate.candidate_id:
                continue
            if own_zones & set(other.zone_ids):
                alternatives += 1

        counts[candidate.candidate_id] = alternatives

    return counts


EXPERIMENTAL_FEATURE_NAMES: Tuple[str, ...] = (
    "candidate_recent_flow_rate",
    "candidate_congestion_trend",
    "candidate_alternative_route_count",
)
