from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
from gymnasium import spaces

from dataset_builder.schema import (
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
    ordered_zones,
)

from hazard.severity import HazardSeverity

from models.building import Building

from simulation_interactive import SimulationSnapshot


# =====================================================
# Honesty discipline: this module reads ONLY SimulationSnapshot (an
# InteractiveSimulation StepResult's own honest, per-tick output) plus
# the env's own record of which recommendations/broadcasts it has
# itself already issued. Nothing here ever imports ground_truth or
# decision_policy -- see tests/test_rl_training.py::
# ObservationHonestyTests, which asserts exactly that.
#
# Markov-property discipline (added alongside the honesty discipline
# above, not a replacement for it): reward_function.py's
# RecommendationDisciplineComponent penalizes a redundant/contradictory
# recommendation based on the EXACT exit/stair id last recommended and
# how many steps ago that changed -- information this encoder must
# also expose, or the policy is penalized on state it structurally
# cannot see (a POMDP with respect to its own reward). exit:{id}:
# is_active_recommendation_target / stair:{id}:is_active_recommendation_
# target and global:steps_since_recommendation_change close that gap;
# see environment.py::_encode() for where the values come from (the
# same ActionState reward_function.py itself just updated this step).
# =====================================================


_SEVERITY_ORDINAL = {
    HazardSeverity.NONE: 0.0,
    HazardSeverity.LOW: 1.0,
    HazardSeverity.MODERATE: 2.0,
    HazardSeverity.HIGH: 3.0,
    HazardSeverity.CRITICAL: 4.0,
}

_DOOR_STATE_ORDINAL = {"OPEN": 0.0, "CLOSED": 1.0, "LOCKED": 2.0}
_EXIT_STATE_ORDINAL = {"OPEN": 0.0, "CLOSED": 1.0}
_STAIR_STATE_ORDINAL = {"AVAILABLE": 0.0, "CLOSED": 1.0}
_DEVICE_STATE_ORDINAL = {"AVAILABLE": 0.0, "FAILED": 1.0}

# zone-level "what has this environment already recommended here"
RECOMMENDATION_NONE = "NONE"
RECOMMENDATION_EXIT = "EXIT"
RECOMMENDATION_STAIR = "STAIR"
_ZONE_RECOMMENDATION_ORDINAL = {
    RECOMMENDATION_NONE: 0.0, RECOMMENDATION_EXIT: 1.0, RECOMMENDATION_STAIR: 2.0,
}

# global "what has this environment already broadcast"
BROADCAST_NONE = "NONE"
BROADCAST_EVACUATE = "EVACUATE"
BROADCAST_SHELTER = "SHELTER"
_BROADCAST_ORDINAL = {BROADCAST_NONE: 0.0, BROADCAST_EVACUATE: 1.0, BROADCAST_SHELTER: 2.0}

# Upper bound applied to the raw "steps since the active recommendation
# last changed" count before it is encoded. reward_function.py's own
# RecommendationDisciplineComponent only ever asks "is this within
# contradiction_window_steps" (a small window, default 5) -- once the
# count is well past any plausible window, further magnitude carries
# no additional meaning, so it is capped rather than encoded as a raw,
# unbounded value (ActionState's own "no recommendation yet" sentinel
# is 10**9, which would otherwise dominate this feature's scale).
_STEPS_SINCE_RECOMMENDATION_CHANGE_CAP = 50


@dataclass(frozen=True)
class ObservationSchema:

    zone_ids: Tuple[str, ...]
    door_ids: Tuple[str, ...]
    exit_ids: Tuple[str, ...]
    stair_ids: Tuple[str, ...]
    detector_ids: Tuple[str, ...]
    camera_ids: Tuple[str, ...]
    feature_names: Tuple[str, ...]

    def to_dict(self) -> dict:

        return {
            "zone_ids": list(self.zone_ids),
            "door_ids": list(self.door_ids),
            "exit_ids": list(self.exit_ids),
            "stair_ids": list(self.stair_ids),
            "detector_ids": list(self.detector_ids),
            "camera_ids": list(self.camera_ids),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationSchema":

        return cls(
            zone_ids=tuple(data["zone_ids"]),
            door_ids=tuple(data["door_ids"]),
            exit_ids=tuple(data["exit_ids"]),
            stair_ids=tuple(data["stair_ids"]),
            detector_ids=tuple(data["detector_ids"]),
            camera_ids=tuple(data["camera_ids"]),
            feature_names=tuple(data["feature_names"]),
        )


class ObservationEncoder:

    # Fixed-shape encoding, determined ONLY by building topology (zone/
    # door/exit/stair/detector/camera counts), never by occupant count --
    # occupant counts vary per generated scenario, but Gymnasium requires
    # a stable observation_space for the lifetime of an env instance.
    # Mirrors dataset_builder/schema.py's own "Zone_i_Occupancy column,
    # not one row per occupant" convention.

    def __init__(self, building: Building):

        zone_ids = tuple(zone.id for zone in ordered_zones(building))
        door_ids = tuple(door.id for door in ordered_doors(building))
        exit_ids = tuple(exit_obj.id for exit_obj in ordered_exits(building))
        stair_ids = tuple(stair.id for stair in ordered_stairs(building))
        detector_ids = tuple(detector.id for detector in ordered_detectors(building))
        camera_ids = tuple(camera.id for camera in ordered_cameras(building))

        feature_names = []

        for zone_id in zone_ids:
            feature_names.append(f"zone:{zone_id}:occupancy")
            feature_names.append(f"zone:{zone_id}:hazard_severity")
            feature_names.append(f"zone:{zone_id}:active_recommendation")

        for door_id in door_ids:
            feature_names.append(f"door:{door_id}:state")
            feature_names.append(f"door:{door_id}:congestion")
            feature_names.append(f"door:{door_id}:queue_length")

        for exit_id in exit_ids:
            feature_names.append(f"exit:{exit_id}:state")
            feature_names.append(f"exit:{exit_id}:congestion")
            feature_names.append(f"exit:{exit_id}:queue_length")
            feature_names.append(f"exit:{exit_id}:is_active_recommendation_target")

        for stair_id in stair_ids:
            feature_names.append(f"stair:{stair_id}:state")
            feature_names.append(f"stair:{stair_id}:congestion")
            feature_names.append(f"stair:{stair_id}:queue_length")
            feature_names.append(f"stair:{stair_id}:is_active_recommendation_target")

        for detector_id in detector_ids:
            feature_names.append(f"detector:{detector_id}:state")

        for camera_id in camera_ids:
            feature_names.append(f"camera:{camera_id}:state")

        feature_names.extend([
            "global:elapsed_time_fraction",
            "global:people_evacuated",
            "global:people_remaining",
            "global:unreachable_count",
            "global:active_broadcast",
            "global:steps_since_recommendation_change",
            # Human Population & Assisted Evacuation Modeling Phase 8 --
            # additive, global aggregate counts only (never per-occupant
            # -- this class's own docstring already fixes observation
            # shape to building topology alone, since occupant/
            # firefighter counts vary per episode). "present" means
            # currently in the building and not yet ARRIVED, the same
            # OccupantState.ARRIVED-based accounting global:
            # people_remaining above already uses.
            "global:children_present",
            "global:elderly_present",
            "global:wheelchair_users_present",
            "global:firefighters_present",
        ])

        self._schema = ObservationSchema(
            zone_ids=zone_ids, door_ids=door_ids, exit_ids=exit_ids, stair_ids=stair_ids,
            detector_ids=detector_ids, camera_ids=camera_ids,
            feature_names=tuple(feature_names),
        )

    # =====================================================

    @property
    def schema(self) -> ObservationSchema:

        return self._schema

    # =====================================================

    @property
    def space(self) -> spaces.Box:

        size = len(self._schema.feature_names)

        return spaces.Box(low=0.0, high=np.inf, shape=(size,), dtype=np.float32)

    # =====================================================

    def encode(
        self,
        snapshot: SimulationSnapshot,
        active_recommendation_by_zone: Mapping[str, str],
        active_broadcast: Optional[str],
        elapsed_time: float,
        max_time: float,
        active_recommendation_target: Optional[str] = None,
        steps_since_recommendation_change: int = 0,
    ) -> np.ndarray:

        # active_recommendation_target/steps_since_recommendation_change
        # default to "no active target"/0 so any existing caller that
        # predates this pair of arguments keeps working unchanged --
        # environment.py is the one production caller, and it always
        # passes both explicitly (see its own _encode()).

        values = []

        for zone_id in self._schema.zone_ids:
            values.append(float(snapshot.node_occupancy.get(zone_id, 0)))
            severity = snapshot.hazard_snapshot.node_state(zone_id).severity
            values.append(_SEVERITY_ORDINAL[severity])
            recommendation = active_recommendation_by_zone.get(zone_id, RECOMMENDATION_NONE)
            values.append(_ZONE_RECOMMENDATION_ORDINAL[recommendation])

        for door_id in self._schema.door_ids:
            values.append(_DOOR_STATE_ORDINAL[snapshot.door_states[door_id]])
            values.append(float(snapshot.edge_congestion.get(door_id, 0)))
            values.append(float(snapshot.edge_queue_lengths.get(door_id, 0)))

        for exit_id in self._schema.exit_ids:
            values.append(_EXIT_STATE_ORDINAL[snapshot.exit_states[exit_id]])
            values.append(float(snapshot.edge_congestion.get(exit_id, 0)))
            values.append(float(snapshot.edge_queue_lengths.get(exit_id, 0)))
            values.append(1.0 if active_recommendation_target == exit_id else 0.0)

        for stair_id in self._schema.stair_ids:
            values.append(_STAIR_STATE_ORDINAL[snapshot.stair_states[stair_id]])
            values.append(float(snapshot.edge_congestion.get(stair_id, 0)))
            values.append(float(snapshot.edge_queue_lengths.get(stair_id, 0)))
            values.append(1.0 if active_recommendation_target == stair_id else 0.0)

        for detector_id in self._schema.detector_ids:
            values.append(_DEVICE_STATE_ORDINAL[snapshot.detector_states[detector_id]])

        for camera_id in self._schema.camera_ids:
            values.append(_DEVICE_STATE_ORDINAL[snapshot.camera_states[camera_id]])

        elapsed_fraction = (elapsed_time / max_time) if max_time > 0 else 0.0
        unreachable_count = sum(
            1 for occupant in snapshot.occupants if occupant.state == "UNREACHABLE"
        )

        values.append(float(elapsed_fraction))
        values.append(float(snapshot.people_evacuated))
        values.append(float(snapshot.people_remaining))
        values.append(float(unreachable_count))
        values.append(_BROADCAST_ORDINAL[active_broadcast or BROADCAST_NONE])
        values.append(
            float(min(steps_since_recommendation_change, _STEPS_SINCE_RECOMMENDATION_CHANGE_CAP))
        )

        present = [occupant for occupant in snapshot.occupants if occupant.state != "ARRIVED"]

        values.append(float(sum(1 for o in present if o.profile_category == "CHILD")))
        values.append(float(sum(1 for o in present if o.profile_category == "ELDERLY")))
        values.append(float(sum(1 for o in present if o.profile_category == "WHEELCHAIR_USER")))
        values.append(float(sum(1 for o in present if o.profile_category == "FIREFIGHTER")))

        return np.array(values, dtype=np.float32)
