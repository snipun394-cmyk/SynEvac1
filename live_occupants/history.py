import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState


# Live Occupant Digital Twin milestone, Phase 5 -- a deliberate,
# documented ADDITION beyond Phase 2's suggested file list (__init__,
# occupant, manager, state, lifecycle, events): "occupant history" has
# no obvious home among those five (occupant.py is the point-in-time
# model, lifecycle.py is status-transition POLICY, not storage) -- the
# same kind of small, motivated deviation this codebase's own prior
# milestones already made (e.g. human_detection/video_source.py, not in
# that milestone's own suggested list either) when the actual shape of
# the problem called for it.

DEFAULT_MAX_HISTORY_LENGTH = 30


@dataclass(frozen=True)
class CameraTransitionRecord:
    timestamp: float
    from_camera_id: Optional[str]
    to_camera_id: Optional[str]


@dataclass(frozen=True)
class ZoneTransitionRecord:
    timestamp: float
    from_zone_id: Optional[str]
    to_zone_id: Optional[str]


@dataclass(frozen=True)
class BehaviorChangeRecord:
    timestamp: float
    from_behavior: Optional[RecognizedBehavior]
    to_behavior: Optional[RecognizedBehavior]


@dataclass(frozen=True)
class ClassificationChangeRecord:
    timestamp: float
    from_classification: HumanClassification
    to_classification: HumanClassification


@dataclass(frozen=True)
class StateChangeRecord:
    timestamp: float
    from_state: Optional[HumanState]
    to_state: Optional[HumanState]


@dataclass(frozen=True)
class PositionSample:
    timestamp: float
    world_position: Tuple[float, float]


@dataclass(frozen=True)
class VelocitySample:
    timestamp: float
    world_velocity: float


@dataclass(frozen=True)
class OccupantHistory:

    # A bounded, IMMUTABLE record of one occupant's own past transitions
    # -- frozen, exactly like live_occupants.occupant.LiveOccupant
    # itself (matching this codebase's own "immutable value objects,
    # a mutable owning registry" convention throughout -- BuildingState/
    # FusedTrack/Detection/TrackedHuman/BehaviorObservation are all
    # frozen too). Every with_*() method below returns a NEW
    # OccupantHistory instance with one record appended and the oldest
    # dropped once max_length is exceeded -- "no unlimited growth,
    # configurable history length" (Phase 5's own requirement),
    # implemented via plain tuple slicing rather than a mutable deque,
    # to stay consistent with this type's own immutability.

    camera_transitions: Tuple[CameraTransitionRecord, ...] = field(default_factory=tuple)
    zone_transitions: Tuple[ZoneTransitionRecord, ...] = field(default_factory=tuple)
    behavior_changes: Tuple[BehaviorChangeRecord, ...] = field(default_factory=tuple)
    position_samples: Tuple[PositionSample, ...] = field(default_factory=tuple)
    velocity_samples: Tuple[VelocitySample, ...] = field(default_factory=tuple)

    # Live Human State & Assistance Perception Bridge milestone --
    # mirrors behavior_changes exactly: one record per GENUINE value
    # change only (LiveOccupantManager only ever calls with_*_change()
    # below when the reconciled value actually differs from the last
    # recorded one), never one entry per cycle regardless of whether
    # anything changed (Phase 10's own "do not store duplicate identical
    # states every cycle" requirement).
    classification_changes: Tuple[ClassificationChangeRecord, ...] = field(default_factory=tuple)
    state_changes: Tuple[StateChangeRecord, ...] = field(default_factory=tuple)

    max_length: int = DEFAULT_MAX_HISTORY_LENGTH

    # =====================================================

    def with_camera_transition(self, timestamp: float, from_camera_id: Optional[str], to_camera_id: Optional[str]) -> "OccupantHistory":

        record = CameraTransitionRecord(timestamp, from_camera_id, to_camera_id)
        transitions = (*self.camera_transitions, record)[-self.max_length:]

        return dataclasses.replace(self, camera_transitions=transitions)

    # =====================================================

    def with_zone_transition(self, timestamp: float, from_zone_id: Optional[str], to_zone_id: Optional[str]) -> "OccupantHistory":

        record = ZoneTransitionRecord(timestamp, from_zone_id, to_zone_id)
        transitions = (*self.zone_transitions, record)[-self.max_length:]

        return dataclasses.replace(self, zone_transitions=transitions)

    # =====================================================

    def with_behavior_change(
        self, timestamp: float, from_behavior: Optional[RecognizedBehavior], to_behavior: Optional[RecognizedBehavior],
    ) -> "OccupantHistory":

        record = BehaviorChangeRecord(timestamp, from_behavior, to_behavior)
        changes = (*self.behavior_changes, record)[-self.max_length:]

        return dataclasses.replace(self, behavior_changes=changes)

    # =====================================================

    def with_position_sample(self, timestamp: float, world_position: Tuple[float, float]) -> "OccupantHistory":

        sample = PositionSample(timestamp, world_position)
        samples = (*self.position_samples, sample)[-self.max_length:]

        return dataclasses.replace(self, position_samples=samples)

    # =====================================================

    def with_velocity_sample(self, timestamp: float, world_velocity: float) -> "OccupantHistory":

        sample = VelocitySample(timestamp, world_velocity)
        samples = (*self.velocity_samples, sample)[-self.max_length:]

        return dataclasses.replace(self, velocity_samples=samples)

    # =====================================================

    def with_classification_change(
        self, timestamp: float, from_classification: HumanClassification, to_classification: HumanClassification,
    ) -> "OccupantHistory":

        record = ClassificationChangeRecord(timestamp, from_classification, to_classification)
        changes = (*self.classification_changes, record)[-self.max_length:]

        return dataclasses.replace(self, classification_changes=changes)

    # =====================================================

    def with_state_change(
        self, timestamp: float, from_state: Optional[HumanState], to_state: Optional[HumanState],
    ) -> "OccupantHistory":

        record = StateChangeRecord(timestamp, from_state, to_state)
        changes = (*self.state_changes, record)[-self.max_length:]

        return dataclasses.replace(self, state_changes=changes)
