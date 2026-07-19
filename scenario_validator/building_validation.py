from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Building Validation -- architecture doc §5.3, module 1. A
# precondition for every other module: every other module's checks
# presuppose that the ids they operate on actually resolve against the
# Building, so this module runs first and a failure here short-circuits
# the rest (see validator.py). A defense-in-depth re-check, not a
# redundant one -- the Generator should only ever sample ids the
# Definition already declared (themselves checked to exist by
# scenario_definition's own self-validation), so a failure here is
# evidence of a Generator bug, not a Definition-authoring mistake, even
# though the two look identical from outside.
#
# "A valid Project exists" folds into this, not a separate check
# (§5.3): a Project is a thin wrapper holding exactly one Building
# (models/project.py) -- there is no Project-level validity concern
# this module needs beyond the Building itself being present, which is
# checked once, up front, below.


def _all_object_ids(building):

    ids = {
        "zone": set(), "door": set(), "exit": set(), "stair": set(),
        "obstacle": set(), "camera": set(), "detector": set(), "floor": set(),
    }

    for floor in building.floors:

        ids["floor"].add(floor.id)

        for zone in floor.zones:
            ids["zone"].add(zone.id)

        for door in floor.doors:
            ids["door"].add(door.id)

        for exit_obj in floor.exits:
            ids["exit"].add(exit_obj.id)

        for obstacle in floor.obstacles:
            ids["obstacle"].add(obstacle.id)

        for camera in floor.cameras:
            ids["camera"].add(camera.id)

        for detector in floor.detectors:
            ids["detector"].add(detector.id)

        for stair in floor.stairs:
            ids["stair"].add(stair.id)

    return ids


def validate_building(candidate, building) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    if building is None:

        report.add(
            FailureCategory.STRUCTURAL, ScenarioValidationReport.ERROR,
            "MISSING_BUILDING", "No Building was supplied to validate the candidate against.",
        )
        return report

    ids = _all_object_ids(building)

    def check(id_value, category, location):

        if id_value and id_value not in ids[category]:

            report.add(
                FailureCategory.STRUCTURAL, ScenarioValidationReport.ERROR,
                "UNKNOWN_ID",
                f"{location}: {category} id {id_value!r} does not exist on the "
                f"referenced Building.",
                object_id=id_value,
            )

    for occupant in candidate.occupants:
        check(occupant.zone_id, "zone", "occupant.zone_id")

    if candidate.fire is not None:
        check(candidate.fire.ignition_zone_id, "zone", "fire.ignition_zone_id")
        check(candidate.fire.ignition_floor_id, "floor", "fire.ignition_floor_id")

    for state in candidate.door_states:
        check(state.door_id, "door", "door_states")

    for state in candidate.exit_states:
        check(state.exit_id, "exit", "exit_states")

    for state in candidate.stair_states:
        check(state.stair_id, "stair", "stair_states")

    for state in candidate.obstacle_states:
        check(state.obstacle_id, "obstacle", "obstacle_states")

    for state in candidate.camera_states:
        check(state.camera_id, "camera", "camera_states")

    for state in candidate.detector_states:
        check(state.detector_id, "detector", "detector_states")

    target_category_by_type = {
        "zone": "zone", "door": "door", "exit": "exit", "stair": "stair",
        "obstacle": "obstacle", "camera": "camera", "detector": "detector",
    }

    for event in candidate.events:

        category = target_category_by_type.get(event.target_type)

        if category is not None:
            check(event.target_id, category, "events.target_id")

    return report
