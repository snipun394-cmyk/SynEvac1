from dataclasses import dataclass
from enum import Enum, auto

from scenario.serialization import enum_from_str, enum_to_str


# Six immutable records holding only the *generated Scenario state* of
# an engineering object -- architecture doc §3.2/§4.2/§7. Each is keyed
# by the referenced object's id only (door_id/exit_id/stair_id/
# obstacle_id/camera_id/detector_id) -- never by a reference to the
# Designer object itself (models.Door/Exit/Staircase/Obstacle/Camera/
# Detector). Resolving an id to its geometry/engineering properties is
# the Validator's/Simulator's job against the Building, not something
# this package stores or computes.
#
# Value-type naming note: the frozen Definition schema (§3.2) names
# Door's three-way value "DoorState", and separately describes both
# Stairs' two-way value and Cameras/Detectors' two-way value as
# "AvailabilityState" -- but with two different member sets (Stairs:
# AVAILABLE/CLOSED: Cameras/Detectors: AVAILABLE/FAILED). Reusing one
# type name for two different enums is not representable in Python, so
# this module gives the stairs enum and the camera/detector enum
# distinct names (StairAvailability, DeviceAvailability) while keeping
# their member sets exactly as the architecture doc states. This is a
# naming-precision choice at the "how" level the architecture document
# explicitly leaves to implementation elsewhere (§4's own "where, not
# how" restraint) -- not a redesign of what either enum means.


class DoorState(Enum):

    OPEN = auto()
    CLOSED = auto()
    LOCKED = auto()

    @property
    def is_traversable(self) -> bool:

        # Single source of truth for "can an occupant get through this
        # door" -- OPEN and CLOSED (unlocked) both are; only LOCKED
        # isn't. Every consumer that needs a traversability verdict from
        # a DoorState (rather than from a live Building Door's own
        # active/locked fields, which Edge.traversable already reads)
        # must go through this property instead of re-deriving its own
        # OPEN/CLOSED/LOCKED -> bool mapping.
        return self != DoorState.LOCKED


class StairAvailability(Enum):

    AVAILABLE = auto()
    CLOSED = auto()


class PresenceState(Enum):

    ACTIVE = auto()
    INACTIVE = auto()


class DeviceAvailability(Enum):

    # Shared by Cameras and Detectors -- §3.2 states Camera's
    # AVAILABLE/FAILED pair explicitly and describes Detector's as the
    # same shape without restating it.

    AVAILABLE = auto()
    FAILED = auto()


# =====================================================


@dataclass(frozen=True)
class ScenarioDoorState:

    door_id: str
    state: DoorState

    def to_dict(self) -> dict:

        return {"door_id": self.door_id, "state": enum_to_str(self.state)}

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioDoorState":

        return cls(door_id=data["door_id"], state=enum_from_str(DoorState, data["state"]))


# =====================================================


@dataclass(frozen=True)
class ScenarioExitState:

    # Exits are bool-shaped in the frozen Definition schema
    # (exit_state_distribution: Dict[exit_id, Distribution[bool]]) --
    # no named enum exists for them there, unlike doors/stairs/
    # obstacles/cameras/detectors, so this record follows suit with a
    # plain is_open flag rather than inventing a two-member enum the
    # architecture doc never named.

    exit_id: str
    is_open: bool

    def to_dict(self) -> dict:

        return {"exit_id": self.exit_id, "is_open": self.is_open}

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioExitState":

        return cls(exit_id=data["exit_id"], is_open=data["is_open"])


# =====================================================


@dataclass(frozen=True)
class ScenarioStairState:

    stair_id: str
    availability: StairAvailability

    def to_dict(self) -> dict:

        return {
            "stair_id": self.stair_id,
            "availability": enum_to_str(self.availability),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioStairState":

        return cls(
            stair_id=data["stair_id"],
            availability=enum_from_str(StairAvailability, data["availability"]),
        )


# =====================================================


@dataclass(frozen=True)
class ScenarioObstacleState:

    obstacle_id: str
    presence: PresenceState

    def to_dict(self) -> dict:

        return {
            "obstacle_id": self.obstacle_id,
            "presence": enum_to_str(self.presence),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioObstacleState":

        return cls(
            obstacle_id=data["obstacle_id"],
            presence=enum_from_str(PresenceState, data["presence"]),
        )


# =====================================================


@dataclass(frozen=True)
class ScenarioCameraState:

    camera_id: str
    availability: DeviceAvailability

    def to_dict(self) -> dict:

        return {
            "camera_id": self.camera_id,
            "availability": enum_to_str(self.availability),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioCameraState":

        return cls(
            camera_id=data["camera_id"],
            availability=enum_from_str(DeviceAvailability, data["availability"]),
        )


# =====================================================


@dataclass(frozen=True)
class ScenarioDetectorState:

    detector_id: str
    availability: DeviceAvailability

    def to_dict(self) -> dict:

        return {
            "detector_id": self.detector_id,
            "availability": enum_to_str(self.availability),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioDetectorState":

        return cls(
            detector_id=data["detector_id"],
            availability=enum_from_str(DeviceAvailability, data["availability"]),
        )
