from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping

from scenario_definition.distributions import (
    Distribution,
    distribution_mapping_from_dict,
    distribution_mapping_to_dict,
)


# Value-type vocabulary for the engineering-state distributions below --
# architecture doc §3.2. These exist purely as a discoverable,
# typo-safe reference vocabulary for authoring a WeightedOptions/
# FixedValue's option labels (e.g. WeightedOptions({DoorState.OPEN.name:
# 0.7, ...})); this package never uses them as a Distribution's actual
# key *type* (distributions.WeightedOptions restricts weight keys to
# already-primitive values, never an enum member, so it never needs to
# know these exist to serialize a distribution -- see distributions.py).
#
# Naming note: §3.2 names both Stairs' two-way value and Cameras'/
# Detectors' two-way value "AvailabilityState", but with two different
# member sets (Stairs: AVAILABLE/CLOSED; Cameras/Detectors: AVAILABLE/
# FAILED) -- not representable as one Python type. Split into
# StairAvailability and DeviceAvailability (shared by Camera and
# Detector, since §3.2 describes Detector's shape as identical to
# Camera's without restating it) -- same distinction already made the
# same way in scenario/engineering_state.py for the resolved side.


class DoorState(Enum):

    OPEN = auto()
    CLOSED = auto()
    LOCKED = auto()


class StairAvailability(Enum):

    AVAILABLE = auto()
    CLOSED = auto()


class PresenceState(Enum):

    ACTIVE = auto()
    INACTIVE = auto()


class DeviceAvailability(Enum):

    AVAILABLE = auto()
    FAILED = auto()


# =====================================================


@dataclass(frozen=True)
class EngineeringConstraints:

    # Pure configuration for Door/Exit/Stair/Obstacle/Camera/Detector
    # state -- architecture doc §3.2's Exits/Doors/Stairs/Obstacles/
    # Cameras/Detectors rows. "Always Open", "Always Closed", "Random",
    # and "Probability distributions" (this implementation phase's own
    # worked examples) are all just different Distribution values
    # inside the same per-category Dict[id, Distribution] field --
    # FixedValue(...) for "always X", WeightedOptions(...) for
    # "random"/weighted -- never separate always_open_ids/
    # always_closed_ids-style sibling fields, the exact collapse §3.1
    # already applied to door/exit/occupancy state.
    #
    # min_open_exits is a plain int, not a Distribution -- a joint rule
    # over the *result* of every exit's independent rule, not itself a
    # value to draw (§3.1: confirmed already compliant, unchanged).
    # This module does not check it against exit_state_distribution's
    # feasibility -- that is validation.py's job (§3.4).

    door_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)

    exit_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)
    min_open_exits: int = 0

    stair_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)
    obstacle_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)
    camera_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)
    detector_state_distribution: Mapping[str, Distribution] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        for attr in (
            "door_state_distribution",
            "exit_state_distribution",
            "stair_state_distribution",
            "obstacle_state_distribution",
            "camera_state_distribution",
            "detector_state_distribution",
        ):
            object.__setattr__(self, attr, MappingProxyType(dict(getattr(self, attr))))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "door_state_distribution": distribution_mapping_to_dict(self.door_state_distribution),
            "exit_state_distribution": distribution_mapping_to_dict(self.exit_state_distribution),
            "min_open_exits": self.min_open_exits,
            "stair_state_distribution": distribution_mapping_to_dict(
                self.stair_state_distribution,
            ),
            "obstacle_state_distribution": distribution_mapping_to_dict(
                self.obstacle_state_distribution,
            ),
            "camera_state_distribution": distribution_mapping_to_dict(
                self.camera_state_distribution,
            ),
            "detector_state_distribution": distribution_mapping_to_dict(
                self.detector_state_distribution,
            ),
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "EngineeringConstraints":

        return cls(
            door_state_distribution=distribution_mapping_from_dict(
                data.get("door_state_distribution", {}),
            ),
            exit_state_distribution=distribution_mapping_from_dict(
                data.get("exit_state_distribution", {}),
            ),
            min_open_exits=data.get("min_open_exits", 0),
            stair_state_distribution=distribution_mapping_from_dict(
                data.get("stair_state_distribution", {}),
            ),
            obstacle_state_distribution=distribution_mapping_from_dict(
                data.get("obstacle_state_distribution", {}),
            ),
            camera_state_distribution=distribution_mapping_from_dict(
                data.get("camera_state_distribution", {}),
            ),
            detector_state_distribution=distribution_mapping_from_dict(
                data.get("detector_state_distribution", {}),
            ),
        )
