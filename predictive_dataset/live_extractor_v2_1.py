from typing import Any, Dict, Optional, Sequence

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.live_extractor import extract_live_candidate_features
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts

from stair_flow.compute import compute_stair_flow_snapshot
from stair_flow.models import StairFlowSnapshot

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
#   candidate_recent_flow_rate:
#     - Exit: reuses evacuation_progress.models.ExitFlow.recent_flow_
#       per_minute directly (an already-existing, already-tested live
#       signal -- the SAME 60s window predictive_dataset.simulation_
#       extractor_v2_1.FLOW_WINDOW_SECONDS uses, so units already
#       match: a per-minute rate over a 60s window IS a 60s count).
#     - Stair: Live Stair Flow & Movement Direction Intelligence /
#       Stair Predictive-Feature Live Parity milestone -- genuine parity
#       proved (docs/architecture/stair_predictive_feature_live_parity.md)
#       between the frozen simulation semantics (count of occupants who
#       COMPLETED crossing this edge, i.e. step.end_time, during the
#       trailing 60s window -- see simulation_extractor_v2_1._recent_
#       flow_rate) and stair_flow.models.StairFlowMetrics.exits (a
#       StairTransitionRecord where THIS stair is from_stair_id -- the
#       moment an occupant physically leaves/completes traversing it,
#       the live analog of a completed edge crossing, NOT
#       StairFlowMetrics.entries, which is the moment they BEGIN
#       traversing -- see that doc's own Phase 1 finding). Used ONLY
#       when a caller supplies `stair_flow_snapshot` (built via
#       build_stair_flow_snapshot_for_prediction() below, at the SAME
#       FLOW_WINDOW_SECONDS=60.0 the frozen feature requires --
#       deliberately independent of whatever window a live
#       CrowdIntelligenceEngine happens to be operationally configured
#       with, per that milestone's own "do not silently change the
#       operational 60-second Crowd Intelligence default" instruction).
#       UNKNOWN (None) vs. a genuinely observed 0 stays distinct
#       automatically -- StairFlowMetrics.exits already encodes exactly
#       that distinction, never fabricated here.
#     - Door, or Stair when no `stair_flow_snapshot` is supplied
#       (backward-compatible default): unchanged, the ORIGINAL live
#       mechanism -- built from data ALREADY being recorded for an
#       unrelated purpose (live_occupants.history.OccupantHistory.
#       zone_transitions, maintained by LiveOccupantManager on every
#       zone change, regardless of whether anything predictive ever
#       reads it) -- counts ZoneTransitionRecords whose (from_zone_id,
#       to_zone_id) pair matches this candidate's own two zones
#       (CandidateIdentity.zone_ids) within the trailing window. Door
#       has no per-asset transition evidence analogous to Stair's
#       current_stair_id, so it keeps this proxy permanently -- this
#       milestone's own explicit "do not change Door behavior" scope.
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
    stair_flow_snapshot: Optional[StairFlowSnapshot] = None,
) -> Dict[str, Any]:

    base = extract_live_candidate_features(
        candidate, edge, building=building, crowd_snapshot=crowd_snapshot, occupancy_facts=occupancy_facts,
    )

    base["candidate_congestion_trend"] = _live_congestion_trend(candidate, crowd_snapshot)
    base["candidate_alternative_route_count"] = alternative_route_counts.get(candidate.candidate_id, 0)
    base["candidate_recent_flow_rate"] = _live_recent_flow_rate(
        candidate, time, evacuation_snapshot=evacuation_snapshot, occupants=occupants,
        stair_flow_snapshot=stair_flow_snapshot,
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
    stair_flow_snapshot: Optional[StairFlowSnapshot] = None,
) -> Optional[int]:

    if candidate.candidate_type == "Exit":
        return _exit_flow_rate(candidate, evacuation_snapshot)

    if candidate.candidate_type == "Stair" and stair_flow_snapshot is not None:
        return _stair_flow_rate(candidate, stair_flow_snapshot)

    return _door_or_stair_flow_rate(candidate, time, occupants)


def _stair_flow_rate(candidate: CandidateIdentity, stair_flow_snapshot: StairFlowSnapshot) -> Optional[int]:

    # Stair Predictive-Feature Live Parity milestone -- `.exits` is the
    # proved live counterpart to the frozen simulation semantics (a
    # COMPLETED crossing count -- see this module's own docstring and
    # docs/architecture/stair_predictive_feature_live_parity.md), never
    # `.entries` (which measures traversal START, not completion).
    # Already None (UNKNOWN) vs. a genuine 0 exactly as StairFlowMetrics
    # itself defines that distinction -- nothing coerced here.

    return stair_flow_snapshot.for_stair(candidate.candidate_id).exits


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


# =====================================================
# Stair Predictive-Feature Live Parity milestone -- the ONE recommended
# way a caller builds `stair_flow_snapshot` for
# extract_live_experimental_candidate_features()/extract_live_v4_
# candidate_features() above. Mirrors build_alternative_route_counts()'s
# own "compute ONCE per tick, pass in" convention (predictive_dataset.
# simulation_extractor_v2_1.build_alternative_route_counts) -- never
# recomputed per candidate.
#
# Deliberately computed at FLOW_WINDOW_SECONDS (60.0) EXPLICITLY, never
# by reading whatever window a live CrowdIntelligenceEngine happens to
# be operationally configured with (CrowdIntelligenceEngine.
# stair_flow_window_seconds defaults to the same 60.0, but a deployment
# is free to configure it differently for Command Center purposes) --
# this keeps the predictive feature's window requirement independent of,
# and never silently coupled to, crowd_intelligence's own operational
# default (Phase 5's own explicit instruction).
# =====================================================


def build_stair_flow_snapshot_for_prediction(
    stairs: Sequence[object], occupants: Optional[Sequence], building, time: float,
    observable_assets=None, camera_coverage=None,
) -> StairFlowSnapshot:

    # `observable_assets` (an observable_assets.models.ObservableAssetSnapshot,
    # the SAME one a caller already builds for BuildingState/
    # CrowdIntelligenceEngine -- see stair_flow.compute.compute_stair_
    # flow_snapshot()'s own docstring) is what lets a genuinely CONFIRMED
    # zero (a calibrated camera covers this stair right now, and truly
    # sees nobody) surface as `0`, not `None` -- without it, a stair with
    # no recent window evidence at all can only ever honestly report
    # UNKNOWN (Phase 6's own "must not fabricate a number when Stair
    # observation is unavailable"). Optional, since a caller with no
    # observable-asset computation configured still gets an honest
    # UNKNOWN rather than an error.
    return compute_stair_flow_snapshot(
        stairs, occupants or (), building, time, window_seconds=FLOW_WINDOW_SECONDS,
        observable_assets=observable_assets, camera_coverage=camera_coverage,
    )
