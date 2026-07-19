from typing import Dict, List, Mapping, Optional, Tuple

from perception.models.building_observation import (
    BuildingObservation,
    ObservationState,
    ObservedEdgeState,
    ObservedNodeState,
    ObservedOccupancy,
    PerceptionSeverity,
    PerceptionSystemStatus,
)
from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.human_observation import HumanObservation
from perception.models.smoke_detector_observation import SmokeDetectorReading

from perception.fusion.occupancy_estimation import EstimatedOccupancy


# Local labels for "which kind of detector reported this" -- mirror
# models.detector.Detector.DETECTOR_TYPES's "Smoke"/"Heat" strings
# without importing models.detector itself (this module's allowed
# dependencies are perception.* only; it has no Building Model
# geometry to resolve and no need for the Detector class at all).
_SMOKE_SOURCE_LABEL = "Smoke"
_HEAT_SOURCE_LABEL = "Heat"

# A small, fixed, controlled vocabulary -- the only visibility
# categories this module knows how to rank when more than one camera
# assigned to the same zone disagrees. Matches the examples already
# named in the frozen architecture (docs/architecture/
# perception_layer_review_2.md SS1.2). A category outside this
# vocabulary is never ranked against another -- see
# _worst_visibility() below.
_VISIBILITY_SEVERITY_ORDER = {"clear": 0, "reduced": 1, "heavy": 2}


class SensorFusion:

    # The final assembly stage of the Perception pipeline -- combines
    # this cycle's EstimatedOccupancy (Occupancy Estimation's own
    # output, already zone-keyed) with raw CameraFrameObservation
    # (visibility only -- occupancy was already resolved upstream),
    # SmokeDetectorReading, and HeatDetectorReading into one canonical
    # BuildingObservation. This is deliberately the *only* place in
    # this package that constructs a BuildingObservation -- everything
    # upstream (GroundTruthCameraProvider/GroundTruthSmokeDetector
    # Provider/GroundTruthHeatDetectorProvider, OccupancyEstimator)
    # produces narrower, single-purpose types; this class is where
    # they are finally merged into the one shape a Rule-Based Engine
    # or future RL Agent is meant to consume.
    #
    # "Fusion" here means combining already-collected, same-instant
    # observations -- an OR across detector alarm bits, a worst-case
    # pick across disagreeing visibility readings, a fixed severity
    # classification of those same discrete facts. It is not physics:
    # this class never predicts a future value, never fills a gap
    # with a modeled estimate, and never reads Ground Truth. Fire
    # growth and smoke spread are Ground-Truth-side concerns this
    # class has no access to and no opinion about.
    #
    # smoke_detector_zone_assignments/heat_detector_zone_assignments/
    # camera_zone_assignments are already-resolved detector_id/
    # camera_id -> zone_id topology, supplied by the caller -- the
    # same pattern OccupancyEstimator already uses, for the same
    # reason: this class has no Building Model geometry dependency
    # (see the module-level note on why models.detector is not
    # imported), and takes topology as a given fact rather than
    # deriving it. A detector/camera absent from its assignment
    # mapping contributes nothing -- never guessed at, never assigned
    # to a default zone.
    #
    # edge_zone_endpoints is optional and, if omitted, edge_observations
    # is always empty -- this module has no edge topology of its own
    # (Navigation is explicitly outside its allowed dependencies), so
    # ObservedEdgeState.blocked_estimate can only ever be derived when
    # the caller supplies which two zones an edge connects. Populating
    # edge_observations without that input would mean fabricating
    # topology this class was never given -- left empty rather than
    # guessed at.

    def __init__(
        self,
        smoke_detector_zone_assignments: Mapping[str, str],
        heat_detector_zone_assignments: Mapping[str, str],
        camera_zone_assignments: Mapping[str, str],
        edge_zone_endpoints: Optional[Mapping[str, Tuple[str, str]]] = None,
    ):

        self._smoke_detector_zone_assignments = dict(smoke_detector_zone_assignments)
        self._heat_detector_zone_assignments = dict(heat_detector_zone_assignments)
        self._camera_zone_assignments = dict(camera_zone_assignments)
        self._edge_zone_endpoints = dict(edge_zone_endpoints or {})

    # =====================================================

    def fuse(
        self,
        timestamp: float,
        occupancy_estimates: Mapping[str, EstimatedOccupancy],
        camera_observations: Optional[List[CameraFrameObservation]] = None,
        smoke_detector_readings: Optional[List[SmokeDetectorReading]] = None,
        heat_detector_readings: Optional[List[HeatDetectorReading]] = None,
        human_observations: Optional[Mapping[str, HumanObservation]] = None,
    ) -> BuildingObservation:

        # human_observations -- Human Perception Integration, additive:
        # a keyword-only-in-practice parameter appended after every
        # existing one, defaulting to None/empty, so every existing
        # caller of fuse() is unaffected. Unlike the four raw-reading
        # lists above, this is not fused here: a HumanObservation is
        # already a complete, per-person record by the time it reaches
        # this method (produced by whatever HumanObservationProvider
        # the caller composed -- see perception.providers.
        # human_observation_provider), not a handful of same-instant
        # raw readings this class would otherwise need to combine.
        # Fusing per-person tracks *across* sources (e.g. reconciling
        # two cameras' independent tracks of the same physical person)
        # is future work this method does not attempt.

        camera_observations = camera_observations or []
        smoke_detector_readings = smoke_detector_readings or []
        heat_detector_readings = heat_detector_readings or []
        human_observations = human_observations or {}

        node_observations = self._build_node_observations(
            camera_observations, smoke_detector_readings, heat_detector_readings,
        )

        return BuildingObservation(
            timestamp=timestamp,
            node_observations=node_observations,
            occupancy_observations=self._build_occupancy_observations(occupancy_estimates),
            edge_observations=self._build_edge_observations(node_observations),
            human_observations=human_observations,
            system_status=self._build_system_status(
                camera_observations, smoke_detector_readings, heat_detector_readings,
            ),
        )

    # =====================================================
    # Occupancy -- a direct pass-through of Occupancy Estimation's own
    # output, never recomputed here. Only OBSERVED zones get an entry;
    # an UNOBSERVED EstimatedOccupancy is omitted entirely rather than
    # written through as an empty ObservedOccupancy() -- both mean the
    # same thing to a reader of BuildingObservation (its own total
    # accessor already defaults a missing zone to ObservedOccupancy()),
    # but omitting keeps node/occupancy/edge_observations sparse and
    # honest about what was actually observed this cycle, the same
    # "only touched ids get an entry" convention HazardEvolutionEngine
    # already uses for HazardSnapshot.
    # =====================================================

    def _build_occupancy_observations(
        self, occupancy_estimates: Mapping[str, EstimatedOccupancy],
    ) -> Dict[str, ObservedOccupancy]:

        return {
            zone_id: ObservedOccupancy(
                estimated_count=estimate.estimated_count, confidence=estimate.confidence,
            )
            for zone_id, estimate in occupancy_estimates.items()
            if estimate.observation_state == ObservationState.OBSERVED
        }

    # =====================================================
    # Hazard side -- one ObservedNodeState per zone touched by at
    # least one detector reading or one camera visibility reading this
    # cycle. A zone touched by neither is omitted (UNOBSERVED default).
    # =====================================================

    def _build_node_observations(
        self,
        camera_observations: List[CameraFrameObservation],
        smoke_detector_readings: List[SmokeDetectorReading],
        heat_detector_readings: List[HeatDetectorReading],
    ) -> Dict[str, ObservedNodeState]:

        # zone_id -> list of (source_label, alarm_active, timestamp)
        detector_contributions: Dict[str, List[Tuple[str, bool, float]]] = {}

        for reading in smoke_detector_readings:
            zone_id = self._smoke_detector_zone_assignments.get(reading.detector_id)
            if zone_id is None:
                continue
            detector_contributions.setdefault(zone_id, []).append(
                (_SMOKE_SOURCE_LABEL, reading.alarm_active, reading.timestamp),
            )

        for reading in heat_detector_readings:
            zone_id = self._heat_detector_zone_assignments.get(reading.detector_id)
            if zone_id is None:
                continue
            detector_contributions.setdefault(zone_id, []).append(
                (_HEAT_SOURCE_LABEL, reading.alarm_active, reading.timestamp),
            )

        # zone_id -> list of (visibility_estimate, timestamp)
        visibility_contributions: Dict[str, List[Tuple[str, float]]] = {}

        for observation in camera_observations:
            zone_id = self._camera_zone_assignments.get(observation.camera_id)
            if zone_id is None or observation.visibility_estimate is None:
                continue
            visibility_contributions.setdefault(zone_id, []).append(
                (observation.visibility_estimate, observation.timestamp),
            )

        touched_zone_ids = set(detector_contributions) | set(visibility_contributions)

        return {
            zone_id: self._fuse_zone(
                detector_contributions.get(zone_id, []),
                visibility_contributions.get(zone_id, []),
            )
            for zone_id in touched_zone_ids
        }

    # =====================================================

    def _fuse_zone(
        self,
        detector_readings: List[Tuple[str, bool, float]],
        visibility_readings: List[Tuple[str, float]],
    ) -> ObservedNodeState:

        alarm_active: Optional[bool] = None
        alarm_source_types: List[str] = []
        latest_time: Optional[float] = None

        if detector_readings:

            # A zone is in alarm if *any* detector assigned to it says
            # so -- an OR across devices, the same "any source's
            # opinion is enough" fusion rule DefaultHazardMergeStrategy
            # already applies on the Ground Truth side, just for a
            # discrete bool instead of a continuous score. Never a
            # fabricated in-between state.
            alarm_active = any(triggered for _, triggered, _ in detector_readings)

            # Only the source types that actually triggered -- matches
            # ObservedNodeState.alarm_source_types's own documented
            # convention (empty, not every device type present, when
            # observed-but-no-alarm).
            alarm_source_types = sorted({
                label for label, triggered, _ in detector_readings if triggered
            })

        visibility_estimate = self._worst_visibility(
            [visibility for visibility, _ in visibility_readings],
        )

        all_timestamps = (
            [timestamp for _, _, timestamp in detector_readings]
            + [timestamp for _, timestamp in visibility_readings]
        )
        if all_timestamps:
            latest_time = max(all_timestamps)

        return ObservedNodeState(
            observation_state=ObservationState.OBSERVED,
            alarm_active=alarm_active,
            alarm_source_types=alarm_source_types,
            visibility_estimate=visibility_estimate,
            estimated_severity=self._classify_severity(
                alarm_active, alarm_source_types, visibility_estimate,
            ),
            last_observed_time=latest_time,
        )

    # =====================================================
    # Worst-case-wins across disagreeing cameras -- the same
    # conservative "don't understate a hazard" reasoning
    # DefaultHazardMergeStrategy already applies on the Ground Truth
    # side. A category outside the known vocabulary is never compared
    # against a known one (there is no principled way to rank an
    # unrecognized label); the first reading encountered wins in that
    # case, purely as a documented, deterministic fallback -- not a
    # claim that it is actually the worst.
    # =====================================================

    def _worst_visibility(self, visibility_estimates: List[str]) -> Optional[str]:

        if not visibility_estimates:
            return None

        ranked = [
            estimate for estimate in visibility_estimates
            if estimate in _VISIBILITY_SEVERITY_ORDER
        ]

        if not ranked:
            return visibility_estimates[0]

        return max(ranked, key=lambda estimate: _VISIBILITY_SEVERITY_ORDER[estimate])

    # =====================================================
    # A fixed, documented classification of already-known discrete
    # facts -- not a hazard estimate. This never reads a continuous
    # value and never predicts anything; it only labels the alarm/
    # visibility signals already fused above with the one ordinal
    # scale every consumer already shares (PerceptionSeverity).
    # =====================================================

    def _classify_severity(
        self,
        alarm_active: Optional[bool],
        alarm_source_types: List[str],
        visibility_estimate: Optional[str],
    ) -> PerceptionSeverity:

        if alarm_active:
            return (
                PerceptionSeverity.CRITICAL if len(alarm_source_types) >= 2
                else PerceptionSeverity.HIGH
            )

        if visibility_estimate == "heavy":
            return PerceptionSeverity.HIGH

        if visibility_estimate == "reduced":
            return PerceptionSeverity.MODERATE

        return PerceptionSeverity.NONE

    # =====================================================
    # Edges -- only derived when the caller supplied
    # edge_zone_endpoints (see class docstring). blocked_estimate is
    # never asserted False on partial information: it is True as soon
    # as either endpoint shows a confirmed HIGH/CRITICAL severity
    # (one confirmed hazard is enough to flag a connector), False only
    # when *both* endpoints are OBSERVED and neither is HIGH/CRITICAL,
    # and None (no opinion) whenever either endpoint is UNOBSERVED and
    # neither shows a confirmed hazard -- "unknown" is never collapsed
    # into "assumed passable".
    # =====================================================

    def _build_edge_observations(
        self, node_observations: Dict[str, ObservedNodeState],
    ) -> Dict[str, ObservedEdgeState]:

        if not self._edge_zone_endpoints:
            return {}

        edge_observations = {}

        for edge_id, (zone_a_id, zone_b_id) in self._edge_zone_endpoints.items():

            state_a = node_observations.get(zone_a_id, ObservedNodeState())
            state_b = node_observations.get(zone_b_id, ObservedNodeState())

            edge_observations[edge_id] = ObservedEdgeState(
                blocked_estimate=self._blocked_estimate(state_a, state_b),
            )

        return edge_observations

    # =====================================================

    def _blocked_estimate(
        self, state_a: ObservedNodeState, state_b: ObservedNodeState,
    ) -> Optional[bool]:

        confirmed_hazard_severities = (PerceptionSeverity.HIGH, PerceptionSeverity.CRITICAL)

        if (
            state_a.estimated_severity in confirmed_hazard_severities
            or state_b.estimated_severity in confirmed_hazard_severities
        ):
            return True

        both_observed = (
            state_a.observation_state == ObservationState.OBSERVED
            and state_b.observation_state == ObservationState.OBSERVED
        )

        if both_observed:
            return False

        return None

    # =====================================================
    # System status -- direct counts of what actually reported this
    # cycle, never a claim about device inventory or health this class
    # has no way to know (it never sees Detector.active/Camera.active
    # -- that is Building Model data, resolved upstream by whichever
    # Ground Truth adapter produced these readings in the first
    # place). panel_communication_ok keeps Phase 1's documented
    # default (True -- "no failure reported") unchanged: no panel/FACP
    # feed is an input to this class, so there is nothing to base a
    # different value on.
    # =====================================================

    def _build_system_status(
        self,
        camera_observations: List[CameraFrameObservation],
        smoke_detector_readings: List[SmokeDetectorReading],
        heat_detector_readings: List[HeatDetectorReading],
    ) -> PerceptionSystemStatus:

        active_camera_ids = {observation.camera_id for observation in camera_observations}
        active_detector_ids = (
            {reading.detector_id for reading in smoke_detector_readings}
            | {reading.detector_id for reading in heat_detector_readings}
        )

        return PerceptionSystemStatus(
            active_camera_count=len(active_camera_ids),
            active_detector_count=len(active_detector_ids),
        )
