import dataclasses
from typing import Dict, Mapping, Optional, Sequence

from observable_assets.models import ObservableAssetSnapshot, ObservationStatus

from stair_flow.direction import derive_direction
from stair_flow.events import extract_stair_flow_events
from stair_flow.models import StairFlowEventType, StairFlowMetrics, StairFlowSnapshot, TrafficDirection


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone, Phase 4/5
# -- the ONE aggregation entry point. A pure function (mirrors
# observable_assets.facts.compute_asset_occupancy_snapshot()'s own "does
# NOT decide who counts, does NOT re-scan raw occupants" convention):
# `occupants` is whatever live_occupants.manager.LiveOccupantManager.
# all_occupants() already returns (TEMPORARILY_LOST/EXITED occupants
# INCLUDED -- see docs/architecture/stair_flow_intelligence.md Sec
# "Phase 6" on why excluding them would silently drop genuine, recent
# transition evidence for someone who entered a stairwell and then
# immediately lost camera coverage, the single most physically common
# real scenario this package exists to measure).
#
# Phase 5 -- window convention: 60.0 seconds is not an arbitrary choice.
# It is the SAME FLOW_WINDOW_SECONDS predictive_dataset.simulation_
# extractor_v2_1/live_extractor_v2_1 already establish ("matches
# evacuation_progress's own recent-flow window"), and the SAME
# evacuation_progress.models.EvacuationProgressConfig.flow_window_seconds
# default. Reused here verbatim rather than re-invented, so a rate
# computed by this package is unit-comparable with both without any
# conversion.
# =====================================================


DEFAULT_WINDOW_SECONDS = 60.0


def compute_stair_flow_snapshot(
    stairs: Sequence[object],
    occupants: Sequence[object],
    building,
    timestamp: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    observable_assets: Optional[ObservableAssetSnapshot] = None,
    observed_occupant_counts: Optional[Mapping[str, Optional[int]]] = None,
    camera_coverage: Optional[object] = None,
) -> StairFlowSnapshot:

    # stairs: models.staircase.Staircase-shaped objects (`.id`,
    # `.from_floor_id`, `.to_floor_id`) -- exactly crowd_intelligence.
    # engine.CrowdIntelligenceEngine's own `self._stairs`, never
    # re-derived from `building` here (that indexing is that caller's
    # own established job).
    # observable_assets: the SAME ObservableAssetSnapshot already flowing
    # into CrowdIntelligenceEngine.compute() -- used only for the
    # OBSERVED/UNKNOWN gate (Phase 9), reusing observable_assets' own
    # already-established camera-coverage-derived truth rather than
    # computing a second, competing one.
    # observed_occupant_counts: stair_id -> Optional[int], the SAME
    # numbers crowd_intelligence.models.AssetApproachMetrics.
    # observed_occupant_count already computed for each stair this cycle
    # (see crowd_intelligence.engine.CrowdIntelligenceEngine._observed_
    # occupant_count()) -- passed through, never recomputed independently
    # here. When None, this function derives it itself from
    # `observable_assets` directly (the same gate, for callers that do
    # not already have crowd_intelligence's own numbers on hand).
    # camera_coverage: OPTIONAL camera_coverage.models.CameraCoverageSnapshot
    # -- used only to enrich `provenance` text with WHICH camera(s)
    # currently cover a stair (Phase 1's own "inspect camera_coverage"
    # finding); never a second, competing OBSERVED/UNKNOWN determination
    # -- observable_assets remains the single gate.

    stairs_by_id = {stair.id: stair for stair in stairs}
    window_start = timestamp - window_seconds

    events_by_stair: Dict[str, list] = {stair_id: [] for stair_id in stairs_by_id}

    for occupant in occupants:

        for event in extract_stair_flow_events(occupant, window_start, timestamp):

            if event.stair_id not in stairs_by_id:
                # Defensive -- an id this snapshot's own `stairs` list
                # never registered (e.g. a Staircase deleted since this
                # history was recorded). Never crashes, never counted.
                continue

            if event.event_type == StairFlowEventType.ENTERED_STAIR:

                direction = derive_direction(building, stairs_by_id[event.stair_id], event.floor_id)
                event = dataclasses.replace(event, direction=direction)

            events_by_stair[event.stair_id].append(event)

    by_stair = {}
    all_events = []

    for stair_id in stairs_by_id:

        events = tuple(sorted(events_by_stair[stair_id], key=lambda e: e.timestamp))
        all_events.extend(events)

        observed_count = None
        if observed_occupant_counts is not None:
            observed_count = observed_occupant_counts.get(stair_id)

        by_stair[stair_id] = _build_metrics(
            stair_id, events, timestamp, window_seconds, observable_assets, observed_count, camera_coverage,
        )

    return StairFlowSnapshot(
        timestamp=timestamp, window_seconds=window_seconds, by_stair=by_stair,
        events=tuple(sorted(all_events, key=lambda e: e.timestamp)),
    )


# =====================================================


def _build_metrics(
    stair_id, events, timestamp, window_seconds, observable_assets, observed_count, camera_coverage,
) -> StairFlowMetrics:

    status = ObservationStatus.UNKNOWN

    if observable_assets is not None:
        status = observable_assets.observation_for(stair_id).status

    if observed_count is None and observable_assets is not None and status == ObservationStatus.OBSERVED:
        observed_count = observable_assets.observation_for(stair_id).occupant_count

    entry_events = [e for e in events if e.event_type == StairFlowEventType.ENTERED_STAIR]
    exit_events = [e for e in events if e.event_type == StairFlowEventType.EXITED_STAIR]

    entries_count = len(entry_events)
    exits_count = len(exit_events)

    has_window_evidence = entries_count > 0 or exits_count > 0
    currently_observed = status == ObservationStatus.OBSERVED

    # Phase 9 -- mirrors evacuation_progress.engine.EvacuationProgressEngine.
    # _compute_exit_flow()'s own ExitFlow.recent_flow_per_minute gate
    # exactly: report real numbers (including an honest 0) whenever
    # EITHER genuine window evidence exists (a nonzero count is itself
    # proof observation happened at the time it was recorded, regardless
    # of this cycle's own camera status) OR the stair is confirmed
    # observed RIGHT NOW (so a 0 is a confirmed, current reading, not a
    # coverage gap). Otherwise -- no evidence in the window AND no
    # current observation -- every flow field stays honestly None.
    if has_window_evidence or currently_observed:

        entries = entries_count
        exits = exits_count

        minutes = window_seconds / 60.0
        entry_rate = entries / minutes
        exit_rate = exits / minutes

        net_flow = entries - exits

        upward = sum(1 for e in entry_events if e.direction == TrafficDirection.UP)
        downward = sum(1 for e in entry_events if e.direction == TrafficDirection.DOWN)
        unknown_direction = entries - upward - downward

        upward_rate = upward / minutes
        downward_rate = downward / minutes

        provenance = _provenance_text(status, camera_coverage, stair_id)

    else:

        entries = exits = net_flow = None
        upward = downward = unknown_direction = None
        entry_rate = exit_rate = upward_rate = downward_rate = None

        provenance = "no current camera observation and no recent transition evidence in window"

    return StairFlowMetrics(
        stair_id=stair_id, status=status, observed_occupant_count=observed_count, window_seconds=window_seconds,
        entries=entries, exits=exits, entry_rate_per_minute=entry_rate, exit_rate_per_minute=exit_rate,
        net_flow=net_flow, upward_count=upward, downward_count=downward, unknown_direction_count=unknown_direction,
        upward_rate_per_minute=upward_rate, downward_rate_per_minute=downward_rate,
        timestamp=timestamp, provenance=provenance,
    )


# =====================================================


def _provenance_text(status, camera_coverage, stair_id) -> str:

    if camera_coverage is not None:

        camera_ids = camera_coverage.cameras_observing(stair_id)

        if camera_ids:
            return f"observed via camera(s): {', '.join(camera_ids)}"

    if status == ObservationStatus.OBSERVED:
        return "asset currently observed (calibrated camera covers its floor)"

    return "counts derived from recent transition history evidence; no current camera observation"
