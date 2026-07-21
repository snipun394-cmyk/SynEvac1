from typing import Sequence

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from sensor_fusion.observation import FusedObservation, ObservationKind


# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 4 -- the PRODUCTION version of the small, ad-hoc bridge the Sensor
# Fusion Engine milestone proved only inside a test
# (tests/test_sensor_fusion_building_state_integration.py). Converts
# FusedObservations into the exact HazardSnapshot/OccupancySnapshot
# shapes building_state.estimator.BuildingStateEstimator.estimate()
# already accepts -- BuildingStateEstimator/BuildingState remain
# completely UNMODIFIED (Phase 4/8's own explicit requirement).
#
# CRITICAL, re-verified line by line against every field's own existing
# default semantics (never invented here):
#   - HazardNodeState.hazard_score defaults to 0.0 (its OWN dataclass
#     default, not something this module invents) -- a zone is only
#     ever GIVEN an entry in node_states when at least one real fused
#     SMOKE/HEAT/FIRE/TEMPERATURE/VISIBILITY observation exists for it;
#     a zone with ZERO such observations gets NO entry at all, and
#     BuildingState.zone_severity()/HazardSummary.severity_for() already
#     honestly resolve an absent zone to HazardSeverity.NONE via their
#     own existing .get(zone_id, HazardSeverity.NONE) total accessor --
#     never fabricated here, only preserved.
#   - occupancy.observation.OccupancyObservation.occupant_count defaults
#     to None ("no reading available, never zero people" -- that
#     type's own docstring) -- a zone is only given an entry in
#     OccupancySnapshot.observations when a real fused OCCUPANCY
#     observation exists for it; OccupancySnapshot.observation_at()
#     already honestly returns OccupancyObservation() (occupant_count=
#     None) for an absent zone.
# Missing smoke never becomes 0 smoke; missing occupancy never becomes
# 0 occupants -- both are simply absent, exactly as their own existing
# types already define "no data" to mean.
#
# ALARM-kind FusedObservations are DELIBERATELY NOT translated into
# anything here (Phase 4 lists it as one possible example, not a
# requirement) -- BuildingState.facp_status/building_alarm_status are
# already correctly derived from FACPSnapshot/detector readings via
# BuildingStateEstimator's own existing, unmodified logic; inventing a
# SECOND, indirect path from fused ALARM observations back into
# building_alarm_status would risk silently disagreeing with FACP's
# own more authoritative view and duplicates a concern that already has
# a single source of truth. Fused ALARM observations remain available
# on live_perception.snapshot.FusedPerceptionSnapshot.fused_observations
# for any future diagnostic consumer, just not force-fit into
# BuildingStateEstimator's own parameters.

_HAZARD_KINDS = (
    ObservationKind.SMOKE, ObservationKind.HEAT, ObservationKind.FIRE,
    ObservationKind.TEMPERATURE, ObservationKind.VISIBILITY,
)


class BuildingStateInputAdapter:

    def __init__(
        self,
        hazard_score_when_alarming: float = 1.0,
        smoke_level_when_alarming: float = 1.0,
        temperature_when_alarming: float = 100.0,
    ):

        # Named, configurable constants -- never inline magic numbers.
        # These apply ONLY to the boolean SMOKE/HEAT/FIRE kinds (a real
        # detector's own native output is a threshold-crossing bit, per
        # perception.models.smoke_detector_observation.SmokeDetectorReading's
        # own docstring -- there is no honest continuous smoke_level/
        # temperature reading to pass through, so an alarming boolean
        # reading is translated into the SAME conservative, worst-case
        # "fully alarming" hazard picture HazardNodeState's own fields
        # already use for a confirmed hazard). TEMPERATURE/VISIBILITY-kind
        # FusedObservations (a genuinely continuous future sensor) pass
        # their own real measurement through directly, never through
        # these constants.
        self.hazard_score_when_alarming = hazard_score_when_alarming
        self.smoke_level_when_alarming = smoke_level_when_alarming
        self.temperature_when_alarming = temperature_when_alarming

    # =====================================================

    def to_hazard_snapshot(self, fused_observations: Sequence[FusedObservation], timestamp: float) -> HazardSnapshot:

        node_states = {}

        for fused in fused_observations:

            if fused.kind not in _HAZARD_KINDS:
                continue

            existing = node_states.get(fused.location, HazardNodeState())

            node_states[fused.location] = self._merge_node_state(existing, fused)

        return HazardSnapshot(timestamp=timestamp, node_states=node_states)

    # =====================================================

    def _merge_node_state(self, existing: HazardNodeState, fused: FusedObservation) -> HazardNodeState:

        smoke_level = existing.smoke_level
        temperature = existing.temperature
        visibility = existing.visibility
        hazard_score = existing.hazard_score

        if fused.kind in (ObservationKind.SMOKE, ObservationKind.FIRE):

            alarming = bool(fused.measurement)
            smoke_level = self.smoke_level_when_alarming if alarming else 0.0
            hazard_score = max(hazard_score, fused.confidence if alarming else 0.0)

        elif fused.kind == ObservationKind.HEAT:

            alarming = bool(fused.measurement)
            temperature = self.temperature_when_alarming if alarming else temperature
            hazard_score = max(hazard_score, fused.confidence if alarming else 0.0)

        elif fused.kind == ObservationKind.TEMPERATURE:

            # A genuine continuous reading (a future sensor kind) --
            # passed through directly, never forced through the
            # boolean-alarm constants above.
            temperature = fused.measurement

        elif fused.kind == ObservationKind.VISIBILITY:

            visibility = fused.measurement

        return HazardNodeState(
            smoke_level=smoke_level, temperature=temperature, visibility=visibility, hazard_score=hazard_score,
        )

    # =====================================================

    def to_occupancy_snapshot(self, fused_observations: Sequence[FusedObservation], timestamp: float) -> OccupancySnapshot:

        observations = {
            fused.location: OccupancyObservation(occupant_count=fused.measurement, confidence=fused.confidence)
            for fused in fused_observations
            if fused.kind == ObservationKind.OCCUPANCY
        }

        return OccupancySnapshot(timestamp=timestamp, observations=observations)
