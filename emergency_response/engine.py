from typing import Dict, List, Mapping, Optional

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from hazard.severity import HazardSeverity

from crowd_intelligence.models import IntensityLevel

from evacuation_progress.models import ZoneClearanceStatus

from emergency_response.models import (
    AlarmSourceEvidence, EmergencyResponseSnapshot, FloorResponseSummary, OccupantAssistanceSignal,
    OccupantEvidenceSummary, ResponsePriorityLevel, ResponsePriorityThresholds, ResponseReason, ResponseWeights,
    ZoneResponsePriority,
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
# human_state_by_occupant_id was, before the Live Human State &
# Assistance Perception Bridge milestone, the ONE way any HumanState
# evidence could ever reach this engine (live_occupants.occupant.
# LiveOccupant carried no HumanClassification/HumanState field at all,
# only RecognizedBehavior) -- it remains supported, and remains the
# CALLER'S OWN responsibility to populate correctly (i.e. only if the
# caller's own perception source assigns HumanObservation.person_id
# using the SAME identity scheme as LiveOccupant.occupant_id), but is
# now consulted ONLY as a fallback: LiveOccupant.human_state (populated
# by human_evidence.reconciliation via live_occupants.manager.
# LiveOccupantManager, see that package's own docstring) is the
# PRIMARY, canonical live-sourced signal whenever it is genuinely known
# -- see _assistance_signal() below for the exact precedence.
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

# The HumanState values treated as CONFIRMED assistance evidence --
# mirrors perception.human_inference._POSSIBLE_INJURY_STATES' own
# selection (FALLEN/CRAWLING/BEING_ASSISTED), but reported here as
# CONFIRMED (never merely "possible") since, unlike POSSIBLY_FALLEN,
# these HumanState values are not hedged heuristics; they are already
# the perception source's own asserted observation.
#
# Live Human State & Assistance Perception Bridge milestone -- split
# into two DISTINCT tiers (Phase 12/23 test 33's own "BEING_ASSISTED
# distinguishable from unassisted FALLEN" requirement): FALLEN/CRAWLING
# describe someone who needs help and has NOT yet received it;
# BEING_ASSISTED describes someone already being helped -- still
# genuinely elevated priority, but a materially different operational
# picture, never conflated into one undifferentiated count.
_CONFIRMED_ASSISTANCE_STATES = frozenset({HumanState.FALLEN, HumanState.CRAWLING})
_BEING_ASSISTED_STATES = frozenset({HumanState.BEING_ASSISTED})

# Phase 12's own explicit, conservative "assistance-awareness, not
# fabricated incapacity" boundary -- these classifications contribute
# only the small, disclosed ResponseWeights.vulnerable_classification_
# weight (Sec models.py), never treated as, or conflated with, an
# assistance/incapacity claim. FIREFIGHTER/FIRE_WARDEN are deliberately
# excluded (they denote response PERSONNEL already on scene, not
# occupants needing rescue) and ADULT/UNKNOWN carry no such connotation
# either.
_VULNERABLE_CLASSIFICATIONS = frozenset(
    {HumanClassification.CHILD, HumanClassification.ELDERLY, HumanClassification.WHEELCHAIR_USER}
)


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
        trajectory_snapshot=None,
    ) -> EmergencyResponseSnapshot:

        # trajectory_snapshot -- Live Occupant Trajectory, Movement
        # Anomaly & Route-Deviation Intelligence milestone, Phase 21.
        # Typed Any/untyped-object here deliberately (a trajectory_
        # intelligence.models.TrajectoryIntelligenceSnapshot, consulted
        # ONLY through its own public .occupant(occupant_id) accessor)
        # -- the same sibling-snapshot pattern crowd_snapshot/
        # evacuation_progress_snapshot already establish. Kept last and
        # optional so every existing positional caller keeps working
        # unchanged.

        # Canonical Live Occupancy Source of Truth milestone -- zone
        # MEMBERSHIP is decided once, by LiveOccupantManager.
        # canonical_occupancy() (the SAME grouping crowd_intelligence/
        # evacuation_progress/live_perception all read -- docs/
        # architecture/canonical_live_occupancy.md), never independently
        # re-decided here. This engine still needs the full LiveOccupant
        # objects (assistance-signal/classification evidence is
        # genuinely this engine's own concern, not a duplicated
        # headcount), so canonical occupant ids are resolved back to
        # objects via a lookup built from active_occupants() -- one
        # fetch, not a second independent grouping decision.
        facts = self.live_occupant_manager.canonical_occupancy(time)
        active_occupants = self.live_occupant_manager.active_occupants()
        human_state_by_occupant_id = human_state_by_occupant_id or {}

        occupants_by_id = {occupant.occupant_id: occupant for occupant in active_occupants}

        occupants_by_zone: Dict[str, list] = {
            zone_id: [occupants_by_id[oid] for oid in occupant_ids if oid in occupants_by_id]
            for zone_id, occupant_ids in facts.occupant_ids_by_zone.items()
        }

        alarm_sources_by_zone = self._alarm_sources_by_zone(building_state)

        zones = {}
        for zone, floor_id in self._zones:

            zones[zone.id] = self._compute_zone_priority(
                zone_id=zone.id, floor_id=floor_id,
                occupants=occupants_by_zone.get(zone.id, ()),
                building_state=building_state, crowd_snapshot=crowd_snapshot,
                evacuation_progress_snapshot=evacuation_progress_snapshot,
                human_state_by_occupant_id=human_state_by_occupant_id,
                alarm_sources=alarm_sources_by_zone.get(zone.id, ()),
                trajectory_snapshot=trajectory_snapshot,
                time=time,
            )

        floors = self._compute_floor_summaries(zones)
        order = self._order_zones(zones)

        return EmergencyResponseSnapshot(timestamp=time, zones=zones, floors=floors, response_priority_order=order)

    # =====================================================

    def _alarm_sources_by_zone(self, building_state) -> Dict[str, tuple]:

        # Phase 6 item 8 (original) / Manual Call Point -> Live
        # Emergency Response Integration milestone, Phase 3 (extended)
        # -- FACP/detector alarm evidence, read directly from
        # BuildingState (the already-established live evidence
        # boundary), never a second, independent FACP/SensorManager
        # query. smoke_detector_states/heat_detector_states/
        # manual_call_point_states are each keyed by sensor_id, all
        # three carrying the SAME SensorStatus.zone_ids convention
        # every other perception-facing provider in this codebase
        # already reads (see live_perception.providers). Manual Call
        # Point is now included as a THIRD, structurally-distinguished
        # source type -- previously silently excluded (an MCP-only
        # alarm was invisible to this method, and therefore to
        # Emergency Response entirely, despite being a genuine active
        # FACP alarm source; see docs/architecture/
        # designer_asset_connectivity_audit.md's own finding).
        #
        # Returns {zone_id: (AlarmSourceEvidence, ...)} -- every zone a
        # currently-alarming source is assigned to gets its own
        # structured entry per source, never a collapsed boolean.
        # source_type comes straight from SensorStatus.sensor_type, so
        # a caller (or this engine's own scoring below) never needs to
        # parse a reason-code string to tell Smoke/Heat/MCP apart.

        if building_state is None or building_state.facp_status is None:
            return {}

        active_source_ids = set(building_state.facp_status.active_alarm_source_ids)

        by_zone: Dict[str, list] = {}

        for detector_states in (
            building_state.smoke_detector_states,
            building_state.heat_detector_states,
            building_state.manual_call_point_states,
        ):
            for sensor_id, asset in detector_states.items():

                if sensor_id not in active_source_ids:
                    continue

                evidence = AlarmSourceEvidence(
                    source_id=sensor_id, source_type=asset.status.sensor_type, zone_ids=asset.status.zone_ids,
                )

                for zone_id in asset.status.zone_ids:
                    by_zone.setdefault(zone_id, []).append(evidence)

        return {zone_id: tuple(sources) for zone_id, sources in by_zone.items()}

    # =====================================================

    def _compute_zone_priority(
        self, *, zone_id, floor_id, occupants, building_state, crowd_snapshot, evacuation_progress_snapshot,
        human_state_by_occupant_id, alarm_sources, trajectory_snapshot, time,
    ) -> ZoneResponsePriority:

        # Manual Call Point -> Live Emergency Response Integration
        # milestone -- automatic detector evidence and manual human-
        # reported evidence are kept as two INDEPENDENT booleans, never
        # collapsed into one. A zone with both an automatic Smoke
        # Detector alarm AND an MCP activation gets BOTH contributions
        # (Phase 7's own explicit "MCP evidence must not overwrite
        # detector evidence; detector evidence must not overwrite MCP
        # evidence" requirement) -- neither can ever silently mask the
        # other.
        automatic_alarm_active = any(source.source_type != "ManualCallPoint" for source in alarm_sources)
        manual_emergency_reported = any(source.source_type == "ManualCallPoint" for source in alarm_sources)

        known_occupant_count = len(occupants)

        assistance_signals = [
            self._assistance_signal(occupant, human_state_by_occupant_id.get(occupant.occupant_id))
            for occupant in occupants
        ]
        possible_count = sum(1 for signal in assistance_signals if signal.possible and not signal.confirmed and not signal.being_assisted)
        confirmed_count = sum(1 for signal in assistance_signals if signal.confirmed)
        being_assisted_count = sum(1 for signal in assistance_signals if signal.being_assisted)

        vulnerable_person_observed = any(
            occupant.human_classification in _VULNERABLE_CLASSIFICATIONS for occupant in occupants
        )

        occupant_evidence = tuple(
            OccupantEvidenceSummary(
                occupant_id=occupant.occupant_id,
                human_state=occupant.human_state.name if occupant.human_state is not None else None,
                classification=(
                    occupant.human_classification.name
                    if occupant.human_classification != HumanClassification.UNKNOWN else None
                ),
                possible_assistance=occupant.behavior in _POSSIBLE_ASSISTANCE_BEHAVIORS,
            )
            for occupant in occupants
            if (
                occupant.human_state is not None
                or occupant.human_classification != HumanClassification.UNKNOWN
                or occupant.behavior in _POSSIBLE_ASSISTANCE_BEHAVIORS
            )
        )

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

        severe_route_anomaly = self._severe_route_anomaly(occupants, trajectory_snapshot)

        score, reason_codes = self._score_zone(
            known_occupant_count=known_occupant_count, possible_count=possible_count, confirmed_count=confirmed_count,
            being_assisted_count=being_assisted_count, vulnerable_person_observed=vulnerable_person_observed,
            evacuation_stalled=evacuation_stalled, hazard_severity=hazard_severity,
            congestion_restricting=congestion_restricting, clearance_status=clearance_status,
            automatic_alarm_active=automatic_alarm_active, manual_emergency_reported=manual_emergency_reported,
            severe_route_anomaly=severe_route_anomaly,
        )

        priority_level = self.thresholds.classify(score)
        explanation = self._explain(zone_id, priority_level, reason_codes, alarm_sources)

        return ZoneResponsePriority(
            zone_id=zone_id, floor_id=floor_id,
            priority_level=priority_level, priority_score=score,
            known_occupant_count=known_occupant_count,
            possible_assistance_count=possible_count, confirmed_assistance_count=confirmed_count,
            being_assisted_count=being_assisted_count, vulnerable_person_observed=vulnerable_person_observed,
            evacuation_stalled=evacuation_stalled,
            hazard_severity=hazard_severity.name if hazard_severity is not None else None,
            clearance_status=clearance_status,
            observability_fraction=observability_fraction,
            reason_codes=reason_codes,
            occupant_evidence=occupant_evidence,
            alarm_sources=alarm_sources,
            manual_emergency_reported=manual_emergency_reported,
            explanation=explanation,
            timestamp=time,
        )

    # =====================================================

    def _assistance_signal(self, occupant, human_state_override) -> OccupantAssistanceSignal:

        # Live Human State & Assistance Perception Bridge milestone --
        # LiveOccupant.human_state (the reconciled, live-pipeline-
        # sourced field) is now PRIMARY: it is consulted whenever
        # genuinely known. The caller-supplied human_state_by_occupant_id
        # override remains available and is consulted ONLY when the
        # occupant itself carries no live-sourced state -- e.g. a
        # deployment with an external correlation source but no live
        # HumanDetector state evidence configured yet. Never merged or
        # allowed to silently overrule a genuine live reading.
        human_state = occupant.human_state if occupant.human_state is not None else human_state_override

        possible = occupant.behavior in _POSSIBLE_ASSISTANCE_BEHAVIORS
        confirmed = human_state in _CONFIRMED_ASSISTANCE_STATES
        being_assisted = human_state in _BEING_ASSISTED_STATES

        return OccupantAssistanceSignal(
            occupant_id=occupant.occupant_id, zone_id=occupant.current_zone_id,
            possible=possible, confirmed=confirmed, being_assisted=being_assisted,
        )

    # =====================================================

    def _severe_route_anomaly(self, occupants, trajectory_snapshot) -> bool:

        # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
        # Intelligence milestone, Phase 21 -- consulted ONLY through
        # trajectory_snapshot's own public .occupant(occupant_id)
        # accessor (never a direct import of trajectory_intelligence.
        # engine/anomaly/route_progress -- see this package's own
        # architecture guard test's allow-list). Examples this is meant
        # to catch: confirmed FALLEN + MOVEMENT_STALLED, an occupant
        # remaining in a hazardous zone, persistent moving-away from
        # every safe exit, or NO_SAFE_ROUTE -- all already folded into
        # trajectory_intelligence's own anomaly_severity, so this is
        # simply "does any occupant here carry a HIGH/CRITICAL reading."

        if trajectory_snapshot is None:
            return False

        for occupant in occupants:

            result = trajectory_snapshot.occupant(occupant.occupant_id)

            if result is not None and result.anomaly_severity in ("HIGH", "CRITICAL"):
                return True

        return False

    # =====================================================

    def _score_zone(
        self, *, known_occupant_count, possible_count, confirmed_count, being_assisted_count,
        vulnerable_person_observed, evacuation_stalled, hazard_severity, congestion_restricting,
        clearance_status, automatic_alarm_active, manual_emergency_reported, severe_route_anomaly,
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
            known_occupant_count > 0 or possible_count > 0 or confirmed_count > 0 or being_assisted_count > 0
            or vulnerable_person_observed or evacuation_stalled
            or hazard_severity is not None or clearance_status is not None
            or automatic_alarm_active or manual_emergency_reported or severe_route_anomaly
        )

        if not has_any_evidence:
            return None, ()

        if known_occupant_count > 0:

            score += w.occupants_weight * min(known_occupant_count / w.occupants_normalization_count, 1.0)
            reasons.append(ResponseReason.KNOWN_OCCUPANTS_PRESENT)

        # Live Human State & Assistance Perception Bridge milestone --
        # confirmed (FALLEN/CRAWLING) and being_assisted (BEING_ASSISTED)
        # are mutually exclusive per-occupant tiers (Sec models.py), but
        # a single zone can genuinely hold BOTH kinds of occupant at
        # once -- both contributions apply independently, never one
        # replacing the other at the zone level.
        if confirmed_count > 0:

            score += w.confirmed_assistance_weight
            reasons.append(ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED)

        elif possible_count > 0:

            score += w.possible_assistance_weight
            reasons.append(ResponseReason.POSSIBLE_ASSISTANCE_REQUIRED)

        if being_assisted_count > 0:

            score += w.being_assisted_weight
            reasons.append(ResponseReason.ASSISTANCE_IN_PROGRESS)

        if vulnerable_person_observed:

            score += w.vulnerable_classification_weight
            reasons.append(ResponseReason.VULNERABLE_PERSON_OBSERVED)

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

        if automatic_alarm_active:

            score += w.facp_alarm_weight
            reasons.append(ResponseReason.FACP_ALARM_ACTIVE)

        # Manual Call Point -> Live Emergency Response Integration
        # milestone -- an INDEPENDENT contribution from automatic_alarm_
        # active above (Phase 7's own "neither may overwrite the other"
        # requirement): a zone with both an automatic detector alarm
        # AND an MCP activation receives both FACP_ALARM_ACTIVE and
        # MANUAL_EMERGENCY_REPORTED, each with its own disclosed weight.
        if manual_emergency_reported:

            score += w.manual_report_weight
            reasons.append(ResponseReason.MANUAL_EMERGENCY_REPORTED)

        if severe_route_anomaly:

            score += w.severe_route_anomaly_weight
            reasons.append(ResponseReason.SEVERE_ROUTE_ANOMALY)

        return score, tuple(reasons)

    # =====================================================

    def _explain(self, zone_id: str, priority_level: str, reason_codes, alarm_sources=()) -> str:

        if not reason_codes:
            return f"Zone {zone_id}: no evidence currently available."

        readable = ", ".join(code.replace("_", " ").title() for code in reason_codes)

        explanation = f"Zone {zone_id} -- {priority_level}. Evidence: {readable}."

        # Manual Call Point -> Live Emergency Response Integration
        # milestone, Phase 5 -- names the specific source id, matching
        # the milestone's own worked example ("Manual emergency report
        # received from MCP-1 in Zone Z3") rather than leaving an
        # operator to infer which device from a generic reason code
        # alone. Never claims what the person who activated it
        # observed -- purely "a report was received from this device."
        manual_sources = tuple(source.source_id for source in alarm_sources if source.source_type == "ManualCallPoint")

        if manual_sources:
            names = ", ".join(sorted(manual_sources))
            explanation += f" Manual emergency report received from {names} in Zone {zone_id}."

        return explanation

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
                being_assisted_count=sum(p.being_assisted_count for p in priorities),
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
