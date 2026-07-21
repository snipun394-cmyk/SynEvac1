from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple


class ObservationKind(Enum):

    # Sensor Fusion Engine milestone, Phase 3 -- the complete,
    # exhaustive-for-now vocabulary this package's Observation carries,
    # mirroring live_system.event_bus.EventType's own "a fixed enum,
    # not a free string, extending this list is the natural extension
    # point for a future kind" convention. Every value here is a KIND
    # of measurement, deliberately NOT a location or a source --
    # Observation itself carries those as separate fields (Phase 3's
    # own "every observation contains source/timestamp/confidence/
    # location/measurement" shape).

    OCCUPANCY = auto()
    BEHAVIOR = auto()
    SMOKE = auto()
    HEAT = auto()
    FIRE = auto()
    TEMPERATURE = auto()
    VISIBILITY = auto()
    ALARM = auto()

    # Future sensors (Phase 3's own "Future sensors" requirement --
    # BLE/WiFi/UWB positioning, firefighter reports, future detector
    # types) plug in as NEW ObservationKind members here, exactly the
    # way this enum's own construction already accommodates -- nothing
    # about Observation/FusedObservation/ObservationProvider/
    # SensorFusionEngine needs to change to add one. A kind with no
    # dedicated merge rule in sensor_fusion.merge still fuses honestly
    # via that module's own documented generic fallback (Sec 5 of
    # docs/architecture/sensor_fusion.md).


@dataclass(frozen=True)
class Observation:

    # One source's report of one measurement, at one location, at one
    # instant -- Phase 3's own required shape exactly. Deliberately a
    # NEW, UNIFIED type, not a reuse of any existing per-package
    # observation model (perception.models.human_observation.
    # HumanObservation, occupancy.observation.OccupancyObservation,
    # hazard.node_state.HazardNodeState, perception.models.
    # smoke_detector_observation.SmokeDetectorReading, ...) -- none of
    # those is cross-KIND (each belongs to exactly one package's own
    # pipeline), and Phase 1's own investigation confirmed no existing
    # type already serves this unified role (see docs/architecture/
    # sensor_fusion.md Sec 2).
    #
    # location is a zone_id string -- the same shared coordinate key
    # hazard.snapshot.HazardSnapshot/occupancy.snapshot.
    # OccupancySnapshot/models.zone.Zone already use, chosen for the
    # same reason those types already share it (see occupancy.snapshot.
    # OccupancySnapshot's own docstring: "the only thing that lets a
    # future AI Decision Engine line up" two different snapshot types
    # for the same place).
    #
    # measurement is deliberately typed Any -- same "typed Any, not
    # imported for its own sake" convention live_system.event_bus.
    # Event.payload already uses, since different kinds carry
    # genuinely different shapes (a float occupant_count/smoke_level/
    # temperature, a bool alarm_active, a behavior_recognition.
    # observation.RecognizedBehavior, or a future kind's own shape) --
    # this package never needs to import any of those concrete types
    # to define Observation/FusedObservation themselves.

    source: str
    kind: ObservationKind
    location: str
    timestamp: float
    confidence: float
    measurement: Any

    def __post_init__(self):

        if not (0.0 <= self.confidence <= 1.0):

            raise ValueError(
                f"Observation.confidence must be in [0.0, 1.0], got {self.confidence!r}."
            )


@dataclass(frozen=True)
class FusedObservation:

    # Phase 5's required output -- one (location, kind) group's
    # reconciled result, as of one SensorFusionEngine.fuse() call.
    # contributing_sources is sorted, deterministic, and NEVER empty
    # (a FusedObservation is only ever produced for a group that had at
    # least one real Observation -- "missing data" for a (location,
    # kind) pair honestly means no FusedObservation at all, never one
    # with an empty contributing_sources and a fabricated measurement).

    kind: ObservationKind
    location: str
    timestamp: float
    measurement: Any
    confidence: float
    contributing_sources: Tuple[str, ...]

    # True exactly when detect_conflict() (sensor_fusion/conflict.py)
    # found the contributing sources genuinely disagreed -- Phase 7's
    # own "never fabricate certainty" requirement made inspectable
    # rather than silently absorbed into a lower confidence number
    # alone.
    conflict: bool = False

    def __post_init__(self):

        if not (0.0 <= self.confidence <= 1.0):

            raise ValueError(
                f"FusedObservation.confidence must be in [0.0, 1.0], got {self.confidence!r}."
            )

        if not self.contributing_sources:

            raise ValueError("FusedObservation.contributing_sources must never be empty.")
