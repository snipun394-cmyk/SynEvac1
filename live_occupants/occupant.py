from dataclasses import dataclass, field
from typing import Optional, Tuple

from behavior_recognition.observation import RecognizedBehavior

from live_occupants.history import OccupantHistory
from live_occupants.state import OccupantStatus


@dataclass(frozen=True)
class LiveOccupant:

    # Live Occupant Digital Twin milestone, Phase 3 -- the ONE canonical
    # runtime representation of one occupant, gathering what today is
    # scattered across tracking.tracked_human.TrackedHuman (per-camera,
    # ephemeral), behavior_recognition.observation.BehaviorObservation
    # (per-camera, ephemeral), cross_camera_identity's own
    # GlobalIdentityRecord (global, but thin -- only last_camera_id/
    # last_track_id/last_timestamp, no zone/behavior/position at all),
    # and virtual_camera.detection.Detection (per-camera, ephemeral,
    # recomputed fresh every cycle, thrown away once
    # MultiCameraFusionEngine consumes it). None of those objects is
    # both (a) keyed by the GLOBAL occupant_id and (b) persists across
    # cycles with first_seen/last_seen/lifecycle status -- this type is
    # the first one that is.
    #
    # Deliberately does NOT duplicate BuildingState or Detection (Phase
    # 3's own explicit instruction): no `classification`/
    # `is_false_positive`/`floor_severity`/anything BuildingState's own
    # FusedTrack/hazard_summary already derives independently. Every
    # field here is either (a) genuinely new information no existing
    # frozen, per-cycle type can honestly hold (first_seen, current_
    # status, history) because it requires persisting ACROSS cycles, or
    # (b) the CURRENT single-cycle snapshot of what's already flowing
    # through the pipeline (current_camera_id/current_track_id/
    # current_zone/current_floor/world_position/world_velocity/
    # behavior/confidence), reused as-is rather than re-derived.
    #
    # Frozen, like every other value object in this codebase's pipeline
    # (TrackedHuman, BehaviorObservation, Detection, FusedTrack,
    # BuildingState) -- live_occupants.manager.LiveOccupantManager is
    # the one place that owns MUTATION, by replacing its own stored
    # reference with a new LiveOccupant instance (dataclasses.replace())
    # each update, exactly the same "immutable value + mutable owning
    # registry" pattern cross_camera_identity.identity_registry.
    # IdentityRegistry already established for GlobalIdentityRecord.

    occupant_id: str

    current_camera_id: Optional[str]
    current_track_id: Optional[str]

    current_zone_id: Optional[str]
    current_floor_id: Optional[str]

    world_position: Optional[Tuple[float, float]]
    world_velocity: Optional[float]

    behavior: Optional[RecognizedBehavior]

    confidence: float

    first_seen: float
    last_seen: float

    status: OccupantStatus

    history: OccupantHistory = field(default_factory=OccupantHistory)
