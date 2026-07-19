from typing import List

from models.building import Building


# =====================================================
# Deterministic enumeration of a Building's engineering objects.
#
# Column numbering (Zone_1, Zone_2, ... / Door_1, Door_2, ...) must be
# stable across calls for the same Building, so every enumeration walks
# Building.ordered_floors() (display_order, not list position -- the
# Building model's own source of truth) and then each Floor's own
# collection in insertion order. Never re-sorted by name/id, since
# neither is guaranteed stable or meaningful across objects.
# =====================================================


def ordered_zones(building: Building) -> List:

    return [
        zone
        for floor in building.ordered_floors()
        for zone in floor.zones
    ]


def ordered_doors(building: Building) -> List:

    return [
        door
        for floor in building.ordered_floors()
        for door in floor.doors
    ]


def ordered_exits(building: Building) -> List:

    return [
        exit_obj
        for floor in building.ordered_floors()
        for exit_obj in floor.exits
    ]


def ordered_stairs(building: Building) -> List:

    return [
        stair
        for floor in building.ordered_floors()
        for stair in floor.stairs
    ]


def ordered_obstacles(building: Building) -> List:

    return [
        obstacle
        for floor in building.ordered_floors()
        for obstacle in floor.obstacles
    ]


def ordered_detectors(building: Building) -> List:

    return [
        detector
        for floor in building.ordered_floors()
        for detector in floor.detectors
    ]


def ordered_cameras(building: Building) -> List:

    return [
        camera
        for floor in building.ordered_floors()
        for camera in floor.cameras
    ]


# =====================================================
# Column-name generators for Dataset 1 (Scenario Features). Each
# returns names in the same order as the matching ordered_*() above,
# so a row's dynamic columns and their values always line up.
# =====================================================


def zone_occupancy_columns(building: Building) -> List[str]:

    return [
        f"Zone_{index}_Occupancy"
        for index in range(1, len(ordered_zones(building)) + 1)
    ]


def door_state_columns(building: Building) -> List[str]:

    return [
        f"Door_{index}_State"
        for index in range(1, len(ordered_doors(building)) + 1)
    ]


def exit_state_columns(building: Building) -> List[str]:

    return [
        f"Exit_{index}_State"
        for index in range(1, len(ordered_exits(building)) + 1)
    ]


def stair_state_columns(building: Building) -> List[str]:

    return [
        f"Stair_{index}_State"
        for index in range(1, len(ordered_stairs(building)) + 1)
    ]


def obstacle_state_columns(building: Building) -> List[str]:

    return [
        f"Obstacle_{index}_State"
        for index in range(1, len(ordered_obstacles(building)) + 1)
    ]


def detector_state_columns(building: Building) -> List[str]:

    return [
        f"Detector_{index}_State"
        for index in range(1, len(ordered_detectors(building)) + 1)
    ]


def camera_state_columns(building: Building) -> List[str]:

    return [
        f"Camera_{index}_State"
        for index in range(1, len(ordered_cameras(building)) + 1)
    ]


# =====================================================
# Fixed-shape column tuples for Dataset 2 (Simulation Outcomes) and
# Dataset 3 (Zone-Level Results). Unlike Dataset 1, these never depend
# on a particular Building, so the column list is a plain constant.
# Append-only: a future version may add columns to the end, but must
# never reorder or remove one a consumer may already depend on.
# =====================================================

SCENARIO_METADATA_COLUMNS = (
    "scenario_id",
    "definition_id",
    "seed",
)

SIMULATION_OUTCOME_COLUMNS = (
    "scenario_id",
    "total_evacuation_time",
    "average_evacuation_time",
    "last_occupant_exit_time",
    "people_evacuated",
    "people_trapped",
    "reachable_occupants",
    "unreachable_occupants",
    "building_cleared",
    "simulation_finished",
    "maximum_queue_length",
    "maximum_density",
    "maximum_congestion",
    "most_congested_exit",
    "most_congested_stair",
    # Human Population & Assisted Evacuation Modeling Phase 6 --
    # appended, never inserted/reordered, per this tuple's own
    # "append-only" contract above.
    "Adult_Count",
    "Child_Count",
    "Elderly_Count",
    "Wheelchair_Count",
    "Visitor_Count",
    "Helping_Group_Count",
    "Assisted_Occupant_Count",
    "Fallen_Count",
    "Possible_Injury_Count",
    "Firefighter_Count",
    "Firefighter_Rescues",
    # Human Decision Engine Phase 6 -- appended, never inserted/
    # reordered, per this tuple's own "append-only" contract above.
    "Help_Decision_Count",
    "Help_Rejected_Count",
    "Firefighter_Task_Count",
    "Group_Formed_Count",
    "Group_Dissolved_Count",
    "Rescue_Initiated_Count",
    "Rescue_Completed_Count",
)

ZONE_RESULT_COLUMNS = (
    "scenario_id",
    "zone_id",
    "initial_occupants",
    "final_occupants",
    "evacuated",
    "trapped",
    "average_delay",
    "average_travel_time",
    "average_waiting_time",
    "exit_used",
    "route_length",
)
