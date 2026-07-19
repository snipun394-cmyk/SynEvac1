from typing import Any, Dict

from behaviour_profile_resolver.category import OccupantCategory, occupant_category
from behaviour_profile_resolver.occupant_attributes import OccupantAttributes, derive_occupant_attributes
from behaviour_profile_resolver.occupant_grouping import assign_occupant_groups

from scenario.engineering_state import (
    DeviceAvailability,
    DoorState,
    PresenceState,
    StairAvailability,
)

from dataset_builder.schema import (
    camera_state_columns,
    detector_state_columns,
    door_state_columns,
    exit_state_columns,
    obstacle_state_columns,
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_obstacles,
    ordered_stairs,
    ordered_zones,
    stair_state_columns,
    zone_occupancy_columns,
)


# =====================================================
# Dataset 1: Scenario Features (one row per scenario).
#
# Every value below is read directly off Scenario/Building -- nothing
# here samples, simulates, or infers a value the Scenario Engine did
# not already resolve. Where the Scenario carries no explicit
# engineering-state override for a given object id, the Building
# model's own baseline field is read instead (Building is an equally
# valid input source per the Dataset Builder's contract, not a guess).
# =====================================================


def extract_scenario_features(run) -> Dict[str, Any]:

    scenario = run.scenario
    building = run.building

    row: Dict[str, Any] = {}

    row["scenario_id"] = scenario.metadata.scenario_id
    row["definition_id"] = scenario.metadata.definition_id
    row["seed"] = scenario.metadata.seed

    fire = scenario.fire
    row["ignition_zone"] = fire.ignition_zone_id if fire is not None else None
    row["ignition_floor"] = fire.ignition_floor_id if fire is not None else None
    row["fire_profile"] = fire.fire_profile if fire is not None else None
    row["growth_time"] = (
        fire.growth_parameters.get("growth_time") if fire is not None else None
    )

    row["total_occupants"] = len(scenario.occupants)

    # Human Population & Assisted Evacuation Modeling Phase 8 -- AI
    # training input features. Both are pure, pre-simulation Scenario
    # facts (never simulated/estimated, matching this function's own
    # "nothing here simulates" contract): profile category counts read
    # straight off scenario.occupants, Firefighter_Count off
    # scenario.firefighters. Simulation *outcomes* (Fallen_Count,
    # rescue counts, ...) belong on Dataset 2 (dataset_builder.labels.
    # extract_simulation_outcome()) only -- they cannot be known before
    # a simulation runs, so they are never duplicated here.
    row.update(_profile_category_feature_values(scenario))
    row["Firefighter_Count"] = len(scenario.firefighters)

    # Occupant Attributes Phase 4 -- Dataset Builder integration. Every
    # value is a pure, deterministic function of already-fixed Scenario
    # data (occupant_attributes.derive_occupant_attributes()/
    # occupant_grouping.assign_occupant_groups(), both keyed only by
    # scenario.metadata.seed/occupant_id/behaviour_profile_id/zone_id)
    # -- a genuine pre-simulation fact, landing on Dataset 1 (Scenario
    # Features) exactly like every other pre-simulation feature this
    # function already extracts, never Dataset 2 (Simulation Outcomes).
    # ai_training/training_dataset need no code changes to pick these
    # up -- both already ingest every column present in scenario_
    # features.csv generically (FeatureSchema.infer()/select_features()
    # default to "every present column").
    row.update(_occupant_attribute_feature_values(scenario))
    row.update(_group_feature_values(scenario))

    row.update(_zone_occupancy_values(scenario, building))
    row.update(_door_state_values(scenario, building))
    row.update(_exit_state_values(scenario, building))
    row.update(_stair_state_values(scenario, building))
    row.update(_obstacle_state_values(scenario, building))
    row.update(_detector_state_values(scenario, building))
    row.update(_camera_state_values(scenario, building))

    return row


# =====================================================

_PROFILE_CATEGORY_FEATURE_COLUMNS = {
    OccupantCategory.ADULT: "Adult_Count",
    OccupantCategory.CHILD: "Child_Count",
    OccupantCategory.ELDERLY: "Elderly_Count",
    OccupantCategory.WHEELCHAIR_USER: "Wheelchair_Count",
    OccupantCategory.VISITOR: "Visitor_Count",
}


def _profile_category_feature_values(scenario) -> Dict[str, int]:

    counts = {column: 0 for column in _PROFILE_CATEGORY_FEATURE_COLUMNS.values()}

    for occupant in scenario.occupants:

        column = _PROFILE_CATEGORY_FEATURE_COLUMNS.get(
            occupant_category(occupant.behaviour_profile_id),
        )

        if column is not None:
            counts[column] += 1

    return counts


# =====================================================

_ATTRIBUTE_FEATURE_COLUMNS = {
    "walking_speed_multiplier": "Mean_Walking_Speed_Multiplier",
    "reaction_speed": "Mean_Reaction_Speed",
    "stamina": "Mean_Stamina",
    "smoke_tolerance": "Mean_Smoke_Tolerance",
    "visibility_tolerance": "Mean_Visibility_Tolerance",
    "fatigue_resistance": "Mean_Fatigue_Resistance",
    "mobility_factor": "Mean_Mobility_Factor",
    "leadership": "Mean_Leadership",
    "risk_aversion": "Mean_Risk_Aversion",
    "route_familiarity": "Mean_Route_Familiarity",
    "compliance": "Mean_Compliance",
    "helping_likelihood": "Mean_Helping_Likelihood",
    "panic_susceptibility": "Mean_Panic_Susceptibility",
    "crowd_following_tendency": "Mean_Crowd_Following_Tendency",
}


def _occupant_attribute_feature_values(scenario) -> Dict[str, Any]:

    if not scenario.occupants:
        return {column: None for column in _ATTRIBUTE_FEATURE_COLUMNS.values()}

    totals: Dict[str, float] = {field_name: 0.0 for field_name in _ATTRIBUTE_FEATURE_COLUMNS}

    for occupant in scenario.occupants:

        attributes: OccupantAttributes = derive_occupant_attributes(
            scenario.metadata.seed, occupant.occupant_id, occupant.behaviour_profile_id,
        )

        for field_name in _ATTRIBUTE_FEATURE_COLUMNS:
            totals[field_name] += getattr(attributes, field_name)

    count = len(scenario.occupants)

    return {
        column: totals[field_name] / count
        for field_name, column in _ATTRIBUTE_FEATURE_COLUMNS.items()
    }


# =====================================================


def _group_feature_values(scenario) -> Dict[str, Any]:

    assignments = assign_occupant_groups(scenario)

    group_sizes: Dict[str, int] = {}
    for assignment in assignments.values():
        group_sizes[assignment.group_id] = group_sizes.get(assignment.group_id, 0) + 1

    return {
        "Group_Count": len(group_sizes),
        "Grouped_Occupant_Count": len(assignments),
        "Mean_Group_Size": (
            sum(group_sizes.values()) / len(group_sizes) if group_sizes else None
        ),
    }


# =====================================================


def _zone_occupancy_values(scenario, building) -> Dict[str, Any]:

    zones = ordered_zones(building)
    columns = zone_occupancy_columns(building)

    counts = {}
    for occupant in scenario.occupants:
        counts[occupant.zone_id] = counts.get(occupant.zone_id, 0) + 1

    return {
        column: counts.get(zone.id, 0)
        for column, zone in zip(columns, zones)
    }


# =====================================================


def _door_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {state.door_id: state.state for state in scenario.door_states}

    def baseline(door):

        if door.locked:
            return DoorState.LOCKED

        if not door.active:
            return DoorState.CLOSED

        return DoorState.OPEN

    return {
        column: overrides.get(door.id, baseline(door)).name
        for column, door in zip(door_state_columns(building), ordered_doors(building))
    }


# =====================================================


def _exit_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {state.exit_id: state.is_open for state in scenario.exit_states}

    def is_open(exit_obj):

        return overrides.get(exit_obj.id, not exit_obj.is_blocked)

    return {
        column: "OPEN" if is_open(exit_obj) else "CLOSED"
        for column, exit_obj in zip(exit_state_columns(building), ordered_exits(building))
    }


# =====================================================


def _stair_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {state.stair_id: state.availability for state in scenario.stair_states}

    # Staircase carries no availability flag of its own in the Building
    # model (see models/staircase.py) -- AVAILABLE is the only baseline
    # there is to fall back to.
    return {
        column: overrides.get(stair.id, StairAvailability.AVAILABLE).name
        for column, stair in zip(stair_state_columns(building), ordered_stairs(building))
    }


# =====================================================


def _obstacle_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {
        state.obstacle_id: state.presence for state in scenario.obstacle_states
    }

    def baseline(obstacle):

        return PresenceState.ACTIVE if obstacle.active else PresenceState.INACTIVE

    return {
        column: overrides.get(obstacle.id, baseline(obstacle)).name
        for column, obstacle in zip(
            obstacle_state_columns(building), ordered_obstacles(building),
        )
    }


# =====================================================


def _detector_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {
        state.detector_id: state.availability for state in scenario.detector_states
    }

    def baseline(detector):

        return DeviceAvailability.AVAILABLE if detector.active else DeviceAvailability.FAILED

    return {
        column: overrides.get(detector.id, baseline(detector)).name
        for column, detector in zip(
            detector_state_columns(building), ordered_detectors(building),
        )
    }


# =====================================================


def _camera_state_values(scenario, building) -> Dict[str, Any]:

    overrides = {
        state.camera_id: state.availability for state in scenario.camera_states
    }

    def baseline(camera):

        return DeviceAvailability.AVAILABLE if camera.active else DeviceAvailability.FAILED

    return {
        column: overrides.get(camera.id, baseline(camera)).name
        for column, camera in zip(camera_state_columns(building), ordered_cameras(building))
    }
