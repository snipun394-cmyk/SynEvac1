from typing import Dict, Optional

from live_system.event_bus import EventBus

from crowd_intelligence.models import TrendDirection
from crowd_intelligence.trends import TrendConfig, TrendTracker

from evacuation_progress.ledger import EvacuationLedger
from evacuation_progress.models import (
    EvacuationProgressConfig, EvacuationProgressSnapshot, EvacuationProgressTrend,
    ExitFlow, ObservabilitySummary, ZoneClearance, ZoneClearanceStatus,
)


# =====================================================
# crowd_intelligence.trends.TrendTracker (reused directly -- Phase 10's
# own "do not duplicate" extended to its trend infrastructure too)
# reports RISING/STABLE/FALLING/UNKNOWN (crowd_intelligence.models.
# TrendDirection). This package's OWN vocabulary (evacuation_progress.
# models.EvacuationProgressTrend: IMPROVING/STABLE/SLOWING/STALLED/
# UNKNOWN) is a plain-string set, deliberately different (5 members, not
# 4) because "no change" means something DIFFERENT depending on what is
# being measured -- a STABLE exit flow rate is healthy (steadily
# processing people), but a STABLE zone-clearance fraction while
# occupants remain is a genuine problem (STALLED). The two small mapping
# functions below are the ONLY place that distinction is made, so it is
# never duplicated or drifted between _compute_zone_clearance() and
# _compute_exit_flow().
# =====================================================


def _map_zone_trend(raw_trend, clearance_fraction: Optional[float]) -> str:

    if raw_trend == TrendDirection.RISING:
        return EvacuationProgressTrend.IMPROVING

    if raw_trend == TrendDirection.FALLING:
        return EvacuationProgressTrend.SLOWING

    if raw_trend == TrendDirection.STABLE:

        if clearance_fraction is not None and clearance_fraction >= 1.0:
            return EvacuationProgressTrend.STABLE  # nothing left to clear -- stability is the correct, healthy reading

        return EvacuationProgressTrend.STALLED  # no change while occupants still remain

    return EvacuationProgressTrend.UNKNOWN


def _map_exit_trend(raw_trend) -> str:

    if raw_trend == TrendDirection.RISING:
        return EvacuationProgressTrend.IMPROVING

    if raw_trend == TrendDirection.FALLING:
        return EvacuationProgressTrend.SLOWING

    if raw_trend == TrendDirection.STABLE:
        # A steady flow rate is healthy on its own -- Phase 5's own
        # "queue HIGH, throughput HIGH, trend STABLE -> heavily loaded
        # but functioning" worked example. Whether it is a PROBLEM at
        # all is decided separately, by flow_active/queue_candidate_
        # count (see low_flow_exit_ids in compute()), never by trend
        # alone.
        return EvacuationProgressTrend.STABLE

    return EvacuationProgressTrend.UNKNOWN


def _map_overall_trend(raw_trend, progress_fraction: Optional[float]) -> str:

    if raw_trend == TrendDirection.RISING:
        return EvacuationProgressTrend.IMPROVING

    if raw_trend == TrendDirection.FALLING:
        return EvacuationProgressTrend.SLOWING

    if raw_trend == TrendDirection.STABLE:

        if progress_fraction is not None and progress_fraction >= 1.0:
            return EvacuationProgressTrend.STABLE

        return EvacuationProgressTrend.STALLED

    return EvacuationProgressTrend.UNKNOWN


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 2/11 -- the ONE object LiveRuntime owns (mirrors crowd_
# intelligence.engine.CrowdIntelligenceEngine's own "exactly one shared
# instance per live session" role). compute(time, building_state,
# crowd_snapshot) is the single entry point.
#
# Never makes an evacuation decision, never broadcasts, never executes
# a control -- purely a deterministic reporting layer, exactly the same
# discipline crowd_intelligence.engine.CrowdIntelligenceEngine already
# establishes one milestone over.
# =====================================================


class EvacuationProgressEngine:

    def __init__(
        self,
        building,
        live_occupant_manager,
        event_bus: EventBus,
        config: Optional[EvacuationProgressConfig] = None,
        trend_config: Optional[TrendConfig] = None,
    ):

        self.building = building
        self.live_occupant_manager = live_occupant_manager
        self.config = config if config is not None else EvacuationProgressConfig()

        self._ledger = EvacuationLedger(event_bus, building)
        self._trend_tracker = TrendTracker(trend_config)

        self._zones = [zone for floor in building.ordered_floors() for zone in floor.zones]
        self._exits = [exit_obj for floor in building.ordered_floors() for exit_obj in floor.exits]

        # Phase 6's own bounded-history requirement, applied to
        # current_active_count itself (used only to detect "the most
        # recent genuine decrease," never to recompute clearance_fraction
        # a second way) -- a plain dict, not a TrendTracker instance,
        # since "when did this last go down" needs the raw values, not a
        # RISING/STABLE/FALLING label.
        self._last_active_count_by_zone: Dict[str, int] = {}
        self._last_decrease_time_by_zone: Dict[str, float] = {}

    # =====================================================

    def compute(self, time: float, building_state, crowd_snapshot=None) -> EvacuationProgressSnapshot:

        # Canonical Live Occupancy Source of Truth milestone -- the
        # SAME per-cycle canonical grouping crowd_intelligence/live_
        # perception/emergency_response all read (docs/architecture/
        # canonical_live_occupancy.md), replacing this engine's own
        # previously-independent "filter active_occupants() by
        # current_zone_id" loop. known_active/active_by_zone below are
        # now both read straight off `facts`, never recomputed.
        facts = self.live_occupant_manager.canonical_occupancy(time)

        known_active = facts.total_observed_count
        known_exited_ids = self._ledger.known_exited_occupant_ids()
        known_exited = len(known_exited_ids)
        known_total_observed = len(self._ledger.ever_seen_ids())

        progress_fraction = (known_exited / known_total_observed) if known_total_observed > 0 else None
        raw_overall_trend = self._trend_tracker.observe("overall_progress", time, progress_fraction)
        overall_trend = _map_overall_trend(raw_overall_trend, progress_fraction)

        active_by_zone: Dict[str, int] = {
            zone_id: len(occupant_ids) for zone_id, occupant_ids in facts.occupant_ids_by_zone.items()
        }

        camera_covered_zone_ids = self._camera_covered_zone_ids(building_state)

        zones = {
            zone.id: self._compute_zone_clearance(zone.id, active_by_zone.get(zone.id, 0), camera_covered_zone_ids, time)
            for zone in self._zones
        }

        exits = {
            exit_obj.id: self._compute_exit_flow(exit_obj.id, crowd_snapshot, time)
            for exit_obj in self._exits
        }

        stalled_zone_ids = tuple(sorted(
            zone_id for zone_id, clearance in zones.items() if clearance.status == ZoneClearanceStatus.STALLED
        ))
        low_flow_exit_ids = tuple(sorted(
            exit_id for exit_id, flow in exits.items()
            if flow.queue_candidate_count > 0 and not flow.flow_active
        ))

        observability = self._compute_observability(known_total_observed, crowd_snapshot, camera_covered_zone_ids)

        return EvacuationProgressSnapshot(
            timestamp=time,
            known_active_occupants=known_active,
            known_exited_occupants=known_exited,
            known_total_observed_occupants=known_total_observed,
            evacuation_progress_fraction=progress_fraction,
            zones=zones,
            exits=exits,
            overall_progress_trend=overall_trend,
            stalled_zone_ids=stalled_zone_ids,
            low_flow_exit_ids=low_flow_exit_ids,
            observability=observability,
        )

    # =====================================================

    def _camera_covered_zone_ids(self, building_state) -> set:

        # Phase 7's own critical distinction -- observability comes from
        # whether an ACTIVE camera is currently assigned to this zone
        # (building_state.camera_observations, already-computed asset
        # status), never from crowd_intelligence's own position_coverage_
        # fraction (which is honestly None for a zone with zero
        # occupants -- useless for answering "is this zone even watched
        # right now").

        if building_state is None:
            return set()

        covered = set()

        for asset in building_state.camera_observations.values():

            if not asset.status.active:
                continue

            covered.update(asset.status.zone_ids)

        return covered

    # =====================================================

    def _compute_zone_clearance(self, zone_id, current_active_count, camera_covered_zone_ids, time) -> ZoneClearance:

        baseline = len(self._ledger.ever_seen_in_zone(zone_id))
        cleared_count = max(baseline - current_active_count, 0)
        clearance_fraction = (cleared_count / baseline) if baseline > 0 else None

        observable = zone_id in camera_covered_zone_ids

        raw_trend = self._trend_tracker.observe(f"zone_clearance:{zone_id}", time, clearance_fraction)
        trend = _map_zone_trend(raw_trend, clearance_fraction)

        previous_count = self._last_active_count_by_zone.get(zone_id)
        if previous_count is not None and current_active_count < previous_count:
            self._last_decrease_time_by_zone[zone_id] = time
        self._last_active_count_by_zone[zone_id] = current_active_count

        status = self._classify_zone_status(
            baseline=baseline, current_active_count=current_active_count,
            clearance_fraction=clearance_fraction, observable=observable, trend=trend,
        )

        return ZoneClearance(
            zone_id=zone_id,
            baseline_observed_count=baseline,
            current_active_count=current_active_count,
            cleared_count=cleared_count,
            clearance_fraction=clearance_fraction,
            status=status,
            trend=trend,
            observable=observable,
            last_decrease_time=self._last_decrease_time_by_zone.get(zone_id),
        )

    # =====================================================

    def _classify_zone_status(self, *, baseline, current_active_count, clearance_fraction, observable, trend) -> str:

        # Phase 4/7's own decision ladder -- observability is checked
        # FIRST and unconditionally: a zone with no current camera
        # coverage is UNKNOWN regardless of what the (untrustworthy)
        # occupant count happens to read, never labeled CLEAR merely
        # because nobody is currently being watched there.

        if not observable:
            return ZoneClearanceStatus.UNKNOWN

        if current_active_count == 0:
            # Confirmed by real, active coverage -- whether baseline is
            # 0 (nobody ever tracked here) or >0 (people were here, now
            # genuinely gone), both are an honest OBSERVED_CLEAR.
            return ZoneClearanceStatus.OBSERVED_CLEAR

        if baseline == 0:
            # Should not be reachable (current_active_count > 0 implies
            # baseline >= that count, since baseline only grows via the
            # same OCCUPANT_CREATED/ZONE_CHANGED events that populate
            # active_by_zone), but never crash on an unexpected 0/0.
            return ZoneClearanceStatus.UNKNOWN

        if clearance_fraction is not None and clearance_fraction >= self.config.nearly_clear_fraction:
            return ZoneClearanceStatus.NEARLY_CLEAR

        if trend == EvacuationProgressTrend.STALLED:
            # `trend` here is already the MAPPED EvacuationProgressTrend
            # value (see _map_zone_trend()) -- a "no change" zone-
            # clearance reading (occupants remaining, TrendTracker's own
            # STABLE classification) is mapped to STALLED specifically
            # because "unchanging while people remain" is the zone-
            # clearance concept this status name describes. This bug
            # (comparing against the pre-mapping EvacuationProgressTrend.
            # STABLE, which this field can never equal once mapped) was
            # found and fixed during the Live Emergency Response & Rescue
            # Priority Intelligence milestone's own end-to-end testing --
            # no prior test exercised the real trend -> status pipeline
            # for a genuinely stalled zone with live-computed trend data.
            return ZoneClearanceStatus.STALLED

        if trend == EvacuationProgressTrend.IMPROVING:
            return ZoneClearanceStatus.CLEARING

        return ZoneClearanceStatus.OCCUPIED

    # =====================================================

    def _compute_exit_flow(self, exit_id, crowd_snapshot, time) -> ExitFlow:

        since_time = time - self.config.flow_window_seconds
        recent_count = self._ledger.recent_exit_count(exit_id, since_time)
        recent_flow_per_minute = recent_count / (self.config.flow_window_seconds / 60.0)

        unique_exited_count = len(self._ledger.exited_records_for_exit(exit_id))

        raw_trend = self._trend_tracker.observe(f"exit_flow:{exit_id}", time, float(recent_count))
        trend = _map_exit_trend(raw_trend)

        queue_candidate_count = 0
        approaching_count = 0
        congestion_level = None
        crowd_position_available = False

        if crowd_snapshot is not None:

            crowd_exit = crowd_snapshot.exit(exit_id)

            if crowd_exit is not None:

                queue_candidate_count = crowd_exit.queue_candidate_count
                approaching_count = crowd_exit.approaching_count
                congestion_level = crowd_exit.congestion_level.name if crowd_exit.congestion_level is not None else None
                crowd_position_available = crowd_exit.position_available

        return ExitFlow(
            exit_id=exit_id,
            unique_exited_count=unique_exited_count,
            recent_flow_per_minute=recent_flow_per_minute if recent_count > 0 or crowd_position_available else None,
            queue_candidate_count=queue_candidate_count,
            approaching_count=approaching_count,
            congestion_level=congestion_level,
            flow_active=recent_count > 0,
            trend=trend,
            position_available=crowd_position_available or self._ledger.any_exit_position_available(exit_id),
        )

    # =====================================================

    def _compute_observability(self, known_total_observed, crowd_snapshot, camera_covered_zone_ids) -> ObservabilitySummary:

        calibrated_occupant_count = 0
        position_coverage_fraction = None

        if crowd_snapshot is not None:

            summary = crowd_snapshot.building_summary
            calibrated_occupant_count = summary.calibrated_occupant_count
            position_coverage_fraction = summary.position_coverage_fraction

        zones_with = tuple(sorted(zone.id for zone in self._zones if zone.id in camera_covered_zone_ids))
        zones_without = tuple(sorted(zone.id for zone in self._zones if zone.id not in camera_covered_zone_ids))

        return ObservabilitySummary(
            total_observed_occupants=known_total_observed,
            calibrated_occupant_count=calibrated_occupant_count,
            position_coverage_fraction=position_coverage_fraction,
            zones_with_camera_coverage=zones_with,
            zones_without_camera_coverage=zones_without,
        )
