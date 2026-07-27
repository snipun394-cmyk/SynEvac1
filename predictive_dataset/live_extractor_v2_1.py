from typing import Any, Dict, Optional, Sequence

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.live_extractor import extract_live_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts

# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 3/4 -- THE LIVE
# EXTRACTOR for the 3 V2.1 experimental fields. Deliberately thin, same
# discipline as predictive_dataset/live_extractor.py itself (Phase 12 of
# the original data-foundation milestone): reuses whatever the live
# system ALREADY computes rather than building a second intelligence
# engine. Each field's live source is DIFFERENT and disclosed --
#
#   candidate_congestion_trend: crowd_intelligence.models.
#     AssetApproachMetrics.trend, already computed for every Door/Exit/
#     Stair asset by crowd_intelligence.trends.TrendTracker. Zero new
#     code needed beyond reading the field.
#
#   candidate_alternative_route_count: purely structural (same function
#     simulation_extractor_v2_1.py uses -- see its own docstring for why
#     zero occupancy dependence means zero sim/live divergence risk).
#
#   candidate_recent_flow_rate: the one field V2.1 flagged as Exit-only.
#     - Exit: reuses evacuation_progress.models.ExitFlow.recent_flow_
#       per_minute directly (an already-existing, already-tested live
#       signal -- the SAME 60s window predictive_dataset.simulation_
#       extractor_v2_1.FLOW_WINDOW_SECONDS uses, so units already
#       match: a per-minute rate over a 60s window IS a 60s count).
#     - Door/Stair: NEW, but built from data ALREADY being recorded for
#       an unrelated purpose (live_occupants.history.OccupantHistory.
#       zone_transitions, maintained by LiveOccupantManager on every
#       zone change, regardless of whether anything predictive ever
#       reads it) -- counts ZoneTransitionRecords whose (from_zone_id,
#       to_zone_id) pair matches this candidate's own two zones
#       (CandidateIdentity.zone_ids) within the trailing window. This is
#       a thin QUERY over already-persisted state, not a new tracking
#       subsystem: no new engine, no new event bus, no new storage.
#
# `occupants` is whatever LiveOccupantManager.all_occupants() already
# returns -- passed in by the caller, exactly like `occupancy_facts` in
# live_extractor.py, since this module must not construct a
# LiveOccupantManager of its own (that module's own documented rule,
# reused here verbatim).
# =====================================================

FLOW_WINDOW_SECONDS = 60.0  # MUST match simulation_extractor_v2_1.FLOW_WINDOW_SECONDS


def extract_live_experimental_candidate_features(
    candidate: CandidateIdentity,
    edge,
    time: float,
    *,
    building,
    crowd_snapshot,
    occupancy_facts=None,
    alternative_route_counts: Dict[str, int],
    evacuation_snapshot=None,
    occupants: Optional[Sequence] = None,
) -> Dict[str, Any]:

    base = extract_live_candidate_features(
        candidate, edge, building=building, crowd_snapshot=crowd_snapshot, occupancy_facts=occupancy_facts,
    )

    base["candidate_congestion_trend"] = _live_congestion_trend(candidate, crowd_snapshot)
    base["candidate_alternative_route_count"] = alternative_route_counts.get(candidate.candidate_id, 0)
    base["candidate_recent_flow_rate"] = _live_recent_flow_rate(
        candidate, time, evacuation_snapshot=evacuation_snapshot, occupants=occupants,
    )

    return base


# =====================================================


def _live_congestion_trend(candidate: CandidateIdentity, crowd_snapshot) -> Optional[str]:

    if crowd_snapshot is None:
        return None

    metrics = None
    if candidate.candidate_type == "Door":
        metrics = crowd_snapshot.door(candidate.candidate_id)
    elif candidate.candidate_type == "Exit":
        metrics = crowd_snapshot.exit(candidate.candidate_id)
    elif candidate.candidate_type == "Stair":
        metrics = crowd_snapshot.stair(candidate.candidate_id)

    if metrics is None:
        return None

    # TrendDirection.UNKNOWN is itself a real, honest answer (crowd_
    # intelligence's own "not enough bounded history yet" state) -- the
    # SAME "UNKNOWN" value simulation_extractor_v2_1's own trend
    # computation returns before its 30s window has elapsed. Reported
    # as the string, never coerced to None (unlike queue_length/
    # approaching_count, which DO become None on position_available=
    # False -- trend is not gated by that flag).
    return metrics.trend.name


# =====================================================


def _live_recent_flow_rate(
    candidate: CandidateIdentity, time: float, *, evacuation_snapshot, occupants: Optional[Sequence],
) -> Optional[int]:

    if candidate.candidate_type == "Exit":
        return _exit_flow_rate(candidate, evacuation_snapshot)

    return _door_or_stair_flow_rate(candidate, time, occupants)


def _exit_flow_rate(candidate: CandidateIdentity, evacuation_snapshot) -> Optional[int]:

    if evacuation_snapshot is None:
        return None

    exit_flow = evacuation_snapshot.exit(candidate.candidate_id)
    if exit_flow is None:
        return None

    # recent_flow_per_minute = recent_count / (flow_window_seconds / 60.0)
    # (evacuation_progress/engine.py's own formula) -- numerically
    # IDENTICAL to a raw "count in the last 60s" only when
    # EvacuationProgressConfig.flow_window_seconds is left at its own
    # default (60.0, matching this module's FLOW_WINDOW_SECONDS exactly).
    # EvacuationProgressSnapshot does not carry its own config, so this
    # extractor cannot verify that at call time -- a disclosed,
    # deployment-level assumption, not silently guessed.
    return exit_flow.recent_flow_per_minute


def _door_or_stair_flow_rate(
    candidate: CandidateIdentity, time: float, occupants: Optional[Sequence],
) -> Optional[int]:

    if occupants is None:
        return None

    own_zones = set(candidate.zone_ids)
    if len(own_zones) < 2:
        # A Door/Stair candidate should always touch 2 real zones; if it
        # doesn't (malformed candidate), there is no honest zone-pair to
        # match against -- None, never a fabricated 0.
        return None

    window_start = time - FLOW_WINDOW_SECONDS
    count = 0

    for occupant in occupants:
        for record in occupant.history.zone_transitions:

            if record.timestamp <= window_start or record.timestamp > time:
                continue

            transitioned_zones = {record.from_zone_id, record.to_zone_id}
            if own_zones <= transitioned_zones:
                count += 1

    return count
