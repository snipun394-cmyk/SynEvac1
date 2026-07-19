from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Tuple

from models.building import Building
from scenario.scenario import Scenario

from perception.models.human_observation import HumanObservation

from decision_policy import (
    announcement_policy, exit_policy, human_priority_policy, rescue_policy, stair_policy, zone_policy,
)


@dataclass(frozen=True)
class DecisionInputs:

    # Everything AI Decision Policy V2 needs from one completed
    # simulation -- nothing here is generated, validated, or simulated
    # by this package. timeline_rows is optional (defaults to empty):
    # it is used only for the most-current, already-id-keyed hazard
    # signal (current_fire_zones) a Timeline Dataset row already
    # carries -- see _current_fire_zone_ids() below. This package
    # never reverses Building-enumeration-indexed timeline columns
    # (Zone_N_Occupancy, Exit_N_State, ...) back to specific ids, since
    # that would require re-deriving the exact enumeration order
    # dataset_builder used when the rows were produced -- an
    # unnecessary coupling this package avoids by design.

    building: Building
    scenario: Scenario
    ground_truth: Any  # ground_truth.labels.GroundTruth
    timeline_rows: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # Human Perception Integration -- additive. Empty by default, so
    # every existing caller (which never supplies this) is completely
    # unaffected: generate_policy() below only computes
    # human_rescue_priorities when this is non-empty. Keyed by
    # HumanObservation.person_id, the same live_system/perception
    # tracking identity BuildingObservation.human_observations already
    # uses -- never re-keyed to a Building/Scenario id, since a
    # perception-tracked person has no such id at all.
    human_observations: Mapping[str, HumanObservation] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionPolicy:

    # One immutable record per completed simulation -- the deterministic
    # engineering decisions computed from it. Every list-shaped field is
    # a plain tuple of plain dicts, same convention ground_truth.GroundTruth
    # already established, so to_dict()/from_dict() never need anything
    # beyond tuple<->list conversion.

    scenario_id: str

    zone_decisions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    exit_decisions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    stair_decisions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    announcements: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    rescue_priorities: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    rescue_order: Tuple[str, ...] = field(default_factory=tuple)
    firefighter_deployment_priority: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # Human Perception Integration -- additive, empty whenever
    # DecisionInputs.human_observations was empty (see generate_policy()
    # below) -- every existing caller/serialized artifact that predates
    # this field simply round-trips it as (), never breaking
    # from_dict()/to_dict() for an old payload (see from_dict()'s own
    # data.get(..., []) default below).
    human_rescue_priorities: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "zone_decisions", tuple(self.zone_decisions))
        object.__setattr__(self, "exit_decisions", tuple(self.exit_decisions))
        object.__setattr__(self, "stair_decisions", tuple(self.stair_decisions))
        object.__setattr__(self, "announcements", tuple(self.announcements))
        object.__setattr__(self, "rescue_priorities", tuple(self.rescue_priorities))
        object.__setattr__(self, "rescue_order", tuple(self.rescue_order))
        object.__setattr__(
            self, "firefighter_deployment_priority", tuple(self.firefighter_deployment_priority),
        )
        object.__setattr__(self, "human_rescue_priorities", tuple(self.human_rescue_priorities))

    # =====================================================

    def to_dict(self) -> dict:

        return {
            "scenario_id": self.scenario_id,
            "zone_decisions": [dict(entry) for entry in self.zone_decisions],
            "exit_decisions": [dict(entry) for entry in self.exit_decisions],
            "stair_decisions": [dict(entry) for entry in self.stair_decisions],
            "announcements": [dict(entry) for entry in self.announcements],
            "rescue_priorities": [dict(entry) for entry in self.rescue_priorities],
            "rescue_order": list(self.rescue_order),
            "firefighter_deployment_priority": [
                dict(entry) for entry in self.firefighter_deployment_priority
            ],
            "human_rescue_priorities": [dict(entry) for entry in self.human_rescue_priorities],
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionPolicy":

        return cls(
            scenario_id=data["scenario_id"],
            zone_decisions=data.get("zone_decisions", []),
            exit_decisions=data.get("exit_decisions", []),
            stair_decisions=data.get("stair_decisions", []),
            announcements=data.get("announcements", []),
            rescue_priorities=data.get("rescue_priorities", []),
            rescue_order=data.get("rescue_order", []),
            firefighter_deployment_priority=data.get("firefighter_deployment_priority", []),
            human_rescue_priorities=data.get("human_rescue_priorities", []),
        )


# =====================================================
# Deterministic enumeration of a Building's zones/exits/stairs -- same
# convention dataset_builder/schema.py and ground_truth/analyzer.py
# already establish (walk Building.ordered_floors(), then each Floor's
# own collection in insertion order), reimplemented here rather than
# imported so this package has no dependency on either sibling.
# =====================================================


def ordered_zones(building: Building) -> List:

    return [zone for floor in building.ordered_floors() for zone in floor.zones]


def ordered_exits(building: Building) -> List:

    return [exit_obj for floor in building.ordered_floors() for exit_obj in floor.exits]


def ordered_stairs(building: Building) -> List:

    return [stair for floor in building.ordered_floors() for stair in floor.stairs]


# =====================================================


def generate_policy(inputs: DecisionInputs) -> DecisionPolicy:

    building = inputs.building
    scenario = inputs.scenario
    ground_truth = inputs.ground_truth

    zones = ordered_zones(building)
    exits = ordered_exits(building)
    stairs = ordered_stairs(building)

    current_fire_zone_ids = _current_fire_zone_ids(inputs.timeline_rows)

    zone_decisions = zone_policy.compute_zone_decisions(
        scenario=scenario, ground_truth=ground_truth, zones=zones,
        current_fire_zone_ids=current_fire_zone_ids,
    )

    exit_decisions = exit_policy.compute_exit_decisions(
        scenario=scenario, ground_truth=ground_truth, exits=exits,
        zone_decisions=zone_decisions,
    )

    stair_decisions = stair_policy.compute_stair_decisions(
        ground_truth=ground_truth, stairs=stairs,
        current_fire_zone_ids=current_fire_zone_ids,
    )

    announcements = announcement_policy.generate_announcements(zone_decisions)

    rescue_priorities, rescue_order = rescue_policy.compute_rescue_priorities(
        scenario=scenario, ground_truth=ground_truth, zones=zones,
    )

    firefighter_deployment_priority = rescue_policy.compute_firefighter_deployment_priority(
        rescue_priorities=rescue_priorities, rescue_order=rescue_order,
        stair_decisions=stair_decisions, exit_decisions=exit_decisions,
    )

    human_rescue_priorities = human_priority_policy.compute_human_rescue_priorities(
        inputs.human_observations,
    )

    return DecisionPolicy(
        scenario_id=ground_truth.scenario_id,
        zone_decisions=zone_decisions,
        exit_decisions=exit_decisions,
        stair_decisions=stair_decisions,
        announcements=announcements,
        rescue_priorities=rescue_priorities,
        rescue_order=rescue_order,
        firefighter_deployment_priority=firefighter_deployment_priority,
        human_rescue_priorities=human_rescue_priorities,
    )


# =====================================================


def _current_fire_zone_ids(timeline_rows) -> FrozenSet[str]:

    if not timeline_rows:
        return frozenset()

    latest_row = timeline_rows[-1]
    raw = latest_row.get("current_fire_zones") or ""

    return frozenset(zone_id for zone_id in raw.split(";") if zone_id)
