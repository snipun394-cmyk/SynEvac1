from typing import Dict, List, Mapping, Optional

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanState

from hazard.severity import HazardSeverity

from crowd_intelligence.models import IntensityLevel

from evacuation_progress.models import ZoneClearanceStatus

from emergency_response.models import (
    EmergencyResponseSnapshot, FloorResponseSummary, OccupantAssistanceSignal, ResponsePriorityLevel,
    ResponsePriorityThresholds, ResponseReason, ResponseWeights, ZoneResponsePriority,
)


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 3/6/14 -- the ONE object LiveRuntime owns (mirrors crowd_
# intelligence.engine.CrowdIntelligenceEngine/evacuation_progress.
# engine.EvacuationProgressEngine's own "exactly one shared instance per
# live session" role). compute(time, building_state, crowd_snapshot,
# evacuation_progress_snapshot, human_state_by_occupant_id) is the
# single entry point.
#
# LIVE EVIDENCE BOUNDARY (Phase 2) -- every input this engine reads is
# either already-canonical live state (LiveOccupantManager, BuildingState)
# or an already-computed sibling snapshot from an earlier pipeline stage
# (CrowdIntelligenceSnapshot, EvacuationProgressSnapshot) -- never a
# simulation-only source. Confirmed by direct investigation:
# decision_policy.rescue_policy/human_priority_policy, ground_truth.
# human_behavior.DynamicHumanState, and human_decision_engine/ are ALL
# simulation-only (GroundTruth/Scenario/BehaviourProfile-derived) and are
# never imported here (mechanically enforced -- see
# tests/test_emergency_response_architecture_guards.py).
#
# human_state_by_occupant_id is the ONE deliberately narrow exception:
# perception.models.human_observation.HumanState (FALLEN/CRAWLING/
# BEING_ASSISTED) does NOT reach live_occupants.occupant.LiveOccupant at
# all (investigated directly -- that type carries no HumanClassification/
# HumanState field, only RecognizedBehavior) -- so this parameter is
# OPTIONAL and the CALLER'S OWN responsibility to populate correctly
# (i.e. only if the caller's own perception source assigns
# HumanObservation.person_id using the SAME identity scheme as
# LiveOccupant.occupant_id.) Never assumed to exist, never silently
# reinterpreted from a mismatched identity space.
#
# Never makes an evacuation decision, never broadcasts, never executes
# a control, never dispatches personnel -- purely a deterministic
# reporting layer (Phase 19/25).
# =====================================================


_HAZARD_SEVERITY_SCORE = {
    HazardSeverity.NONE: 0.0,
    HazardSeverity.LOW: 0.25,
    HazardSeverity.MODERATE: 0.5,
    HazardSeverity.HIGH: 0.75,
    HazardSeverity.CRITICAL: 1.0,
}

# Phase 5's own explicit, never-to-be-undone safety boundary --
# RecognizedBehavior.POSSIBLY_FALLEN is a hedged geometric heuristic
# (see behavior_recognition's own docstring on the false-positive causes
# this heuristic has); it may only ever produce a POSSIBLE assistance
# signal here, never CONFIRMED.
_POSSIBLE_ASSISTANCE_BEHAVIORS = frozenset({RecognizedBehavior.POSSIBLY_FALLEN})

# The ONLY HumanState values treated as CONFIRMED assistance evidence --
# mirrors perception.human_inference._POSSIBLE_INJURY_STATES' own
# selection exactly (FALLEN/CRAWLING/BEING_ASSISTED), reused as the same
# disclosed judgment, one layer over -- but reported here as CONFIRMED
# (never merely "possible") since, unlike POSSIBLY_FALLEN, these
# HumanState values are not hedged heuristics; they are already the
# perception source's own asserted observation.
_CONFIRMED_ASSISTANCE_STATES = frozenset({HumanState.FALLEN, HumanState.CRAWLING, HumanState.BEING_ASSISTED})


class EmergencyResponseIntelligenceEngine:

    def __init__(
        self,
        building,
        live_occupant_manager,
        weights: Optional[ResponseWeights] = None,
        thresholds: Optional[ResponsePriorityThresholds] = None,
    ):

        self.building = building
        self.live_occupant_manager = live_occupant_manager
        self.weights = weights if weights is not None else ResponseWeights()
        self.thresholds = thresholds if thresholds is not None else ResponsePriorityThresholds()

        self._zones = [(zone, floor.id) for floor in building.ordered_floors() for zone in floor.zones]
        self._floor_order = {floor.id: floor.display_order for floor in building.ordered_floors()}

    # =====================================================

    def compute(
        self, time: float, building_state, crowd_snapshot=None, evacuation_progress_snapshot=None,
        human_state_by_occupant_id: Optional[Mapping[str, HumanState]] = None,
    ) -> EmergencyResponseSnapshot:

        active_occupants = self.live_occupant_manager.active_occupants()
        human_state_by_occupant_id = human_state_by_occupant_id or {}

        occupants_by_zone: Dict[str, list] = {}
        for occupant in active_occupants:
            if occupant.current_zone_id is not None:
                occupants_by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

        zone_alarm_ids = self._zone_alarm_ids(building_state)

        zones = {}
        for zone, floor_id in self._zones:

            zones[zone.id] = self._compute_zone_priority(
                zone_id=zone.id, floor_id=floor_id,
                occupants=occupants_by_zone.get(zone.id, ()),
                building_state=building_state, crowd_snapshot=crowd_snapshot,
                evacuation_progress_snapshot=evacuation_progress_snapshot,
                human_state_by_occupant_id=human_state_by_occupant_id,
                alarm_active=zone.id in zone_alarm_ids,
                time=time,
            )

        floors = self._compute_floor_summaries(zones)
        order = self._order_zones(zones)

        return EmergencyResponseSnapshot(timestamp=time, zones=zones, floors=floors, response_priority_order=order)

    # =====================================================

    def _zone_alarm_ids(self, building_state) -> set:

        # Phase 6 item 8 -- FACP/detector alarm evidence, read directly
        # from BuildingState (the already-established live evidence
        # boundary), never a second, independent FACP/SensorManager
        # query. smoke_detector_states/heat_detector_states are keyed by
        # sensor_id, each carrying the SAME SensorStatus.zone_ids
        # convention every other perception-facing provider in this
        # codebase already reads (see live_perception.providers).

        if building_state is None or building_state.facp_status is None:
            return set()

        active_source_ids = set(building_state.facp_status.active_alarm_source_ids)

        zone_ids = set()
        for detector_states in (building_state.smoke_detector_states, building_state.heat_detector_states):
            for sensor_id, asset in detector_states.items():
                if sensor_id in active_source_ids:
                    zone_ids.update(asset.status.zone_ids)

        return zone_ids

    # =====================================================

    def _compute_zone_priority(
        self, *, zone_id, floor_id, occupants, building_state, crowd_snapshot, evacuation_progress_snapshot,
        human_state_by_occupant_id, alarm_active, time,
    ) -> ZoneResponsePriority:

        known_occupant_count = len(occupants)

        assistance_signals = [
            self._assistance_signal(occupant, human_state_by_occupant_id.get(occupant.occupant_id))
            for occupant in occupants
        ]
        possible_count = sum(1 for signal in assistance_signals if signal.possible and not signal.confirmed)
        confirmed_count = sum(1 for signal in assistance_signals if signal.confirmed)

        clearance_status = None
        evacuation_stalled = False
        observability_fraction = None

        if evacuation_progress_snapshot is not None:

            zone_clearance = evacuation_progress_snapshot.zone(zone_id)

            if zone_clearance is not None:

                clearance_status = zone_clearance.status
                evacuation_stalled = zone_clearance.status == ZoneClearanceStatus.STALLED
                observability_fraction = 1.0 if zone_clearance.observable else 0.0

        hazard_severity = None
        if building_state is not None:
            hazard_severity = building_state.zone_severity(zone_id)

        congestion_restricting = False
        if crowd_snapshot is not None:

            zone_density = crowd_snapshot.zone(zone_id)

            if zone_density is not None and zone_density.density_classification is not None:
                congestion_restricting = (
                    zone_density.density_classification.value >= IntensityLevel.HIGH.value and evacuation_stalled
                )

        score, reason_codes = self._score_zone(
            known_occupant_count=known_occupant_count, possible_count=possible_count, confirmed_count=confirmed_count,
            evacuation_stalled=evacuation_stalled, hazard_severity=hazard_severity,
            congestion_restricting=congestion_restricting, clearance_status=clearance_status, alarm_active=alarm_active,
        )

        priority_level = self.thresholds.classify(score)
        explanation = self._explain(zone_id, priority_level, reason_codes)

        return ZoneResponsePriority(
            zone_id=zone_id, floor_id=floor_id,
            priority_level=priority_level, priority_score=score,
            known_occupant_count=known_occupant_count,
            possible_assistance_count=possible_count, confirmed_assistance_count=confirmed_count,
            evacuation_stalled=evacuation_stalled,
            hazard_severity=hazard_severity.name if hazard_severity is not None else None,
            clearance_status=clearance_status,
            observability_fraction=observability_fraction,
            reason_codes=reason_codes,
            explanation=explanation,
            timestamp=time,
        )

    # =====================================================

    def _assistance_signal(self, occupant, human_state) -> OccupantAssistanceSignal:

        possible = occupant.behavior in _POSSIBLE_ASSISTANCE_BEHAVIORS
        confirmed = human_state in _CONFIRMED_ASSISTANCE_STATES

        return OccupantAssistanceSignal(
            occupant_id=occupant.occupant_id, zone_id=occupant.current_zone_id,
            possible=possible, confirmed=confirmed,
        )

    # =====================================================

    def _score_zone(
        self, *, known_occupant_count, possible_count, confirmed_count, evacuation_stalled,
        hazard_severity, congestion_restricting, clearance_status, alarm_active,
    ):

        # Phase 7's own explicit transparency requirement -- every
        # contribution below is a documented, disclosed weight (see
        # emergency_response.models.ResponseWeights), never an
        # unexplained constant, and every nonzero contribution adds its
        # own named reason code so the operator can see WHY.
        #
        # priority_score is None (ResponsePriorityLevel.UNKNOWN) ONLY
        # when there is genuinely zero evidence of any kind for this
        # zone -- no known occupants, no assistance signal, not stalled,
        # no hazard reading at all, no clearance data, no alarm. This is
        # rare in practice (evacuation_progress/crowd_intelligence both
        # index every zone in the building), but kept as an honest floor
        # rather than fabricating a LOW score from nothing.

        w = self.weights
        score = 0.0
        reasons = []

        has_any_evidence = (
            known_occupant_count > 0 or possible_count > 0 or confirmed_count > 0 or evacuation_stalled
            or hazard_severity is not None or clearance_status is not None or alarm_active
        )

        if not has_any_evidence:
            return None, ()

        if known_occupant_count > 0:

            score += w.occupants_weight * min(known_occupant_count / w.occupants_normalization_count, 1.0)
            reasons.append(ResponseReason.KNOWN_OCCUPANTS_PRESENT)

        if confirmed_count > 0:

            score += w.confirmed_assistance_weight
            reasons.append(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED)

        elif possible_count > 0:

            score += w.possible_assistance_weight
            reasons.append(ResponseReason.POSSIBLE_ASSISTANCE_REQUIRED)

        if evacuation_stalled:

            score += w.stalled_weight
            reasons.append(ResponseReason.EVACUATION_STALLED)

        if hazard_severity is not None and hazard_severity != HazardSeverity.NONE:

            score += w.hazard_weight * _HAZARD_SEVERITY_SCORE[hazard_severity]
            reasons.append(ResponseReason.HAZARD_PRESENT)

        if congestion_restricting:

            score += w.congestion_restricting_weight
            reasons.append(ResponseReason.HIGH_CONGESTION_RESTRICTING_EVACUATION)

        if clearance_status == ZoneClearanceStatus.UNKNOWN:

            # Phase 6/9's own critical requirement: poor observability
            # is a REAL, positive contribution to priority (a search/
            # verification concern), never a silent default to LOW.
            score += w.uncertainty_weight
            reasons.append(ResponseReason.UNCERTAIN_OCCUPANCY)

        elif clearance_status == ZoneClearanceStatus.OBSERVED_CLEAR and known_occupant_count == 0:

            reasons.append(ResponseReason.OBSERVED_CLEAR)

        if alarm_active:

            score += w.facp_alarm_weight
            reasons.append(ResponseReason.FACP_ALARM_ACTIVE)

        return score, tuple(reasons)

    # =====================================================

    def _explain(self, zone_id: str, priority_level: str, reason_codes) -> str:

        if not reason_codes:
            return f"Zone {zone_id}: no evidence currently available."

        readable = ", ".join(code.replace("_", " ").title() for code in reason_codes)

        return f"Zone {zone_id} -- {priority_level}. Evidence: {readable}."

    # =====================================================

    def _compute_floor_summaries(self, zones: Mapping[str, ZoneResponsePriority]) -> Dict[str, FloorResponseSummary]:

        by_floor: Dict[str, List[ZoneResponsePriority]] = {}
        for priority in zones.values():

            floor_id = priority.floor_id or ""
            by_floor.setdefault(floor_id, []).append(priority)

        summaries = {}
        for floor_id, priorities in by_floor.items():

            summaries[floor_id] = FloorResponseSummary(
                floor_id=floor_id,
                critical_zone_count=sum(1 for p in priorities if p.priority_level == ResponsePriorityLevel.CRITICAL),
                high_zone_count=sum(1 for p in priorities if p.priority_level == ResponsePriorityLevel.HIGH),
                moderate_zone_count=sum(1 for p in priorities if p.priority_level == ResponsePriorityLevel.MODERATE),
                known_occupants_remaining=sum(p.known_occupant_count for p in priorities),
                possible_assistance_count=sum(p.possible_assistance_count + p.confirmed_assistance_count for p in priorities),
            )

        return summaries

    # =====================================================

    def _order_zones(self, zones: Mapping[str, ZoneResponsePriority]):

        # Phase 11's own required deterministic ordering: priority_score
        # (descending, None treated as lowest) -> floor display_order
        # (ascending) -> zone_id (ascending, alphabetical) -- a fixed,
        # disclosed tie-break, never dependent on dict/insertion order.

        def sort_key(zone_id):

            priority = zones[zone_id]
            score = priority.priority_score if priority.priority_score is not None else -1.0
            floor_order = self._floor_order.get(priority.floor_id, 0)

            return (-score, floor_order, zone_id)

        return tuple(sorted(zones.keys(), key=sort_key))
