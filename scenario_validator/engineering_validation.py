from scenario_definition.distributions import FixedValue

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Door / Exit / Stair (+ Obstacle/Camera/Detector) Validation --
# architecture doc §5.3, module 4, tagged STRUCTURAL. Deliberately
# scoped to state and count only (§5.3): whether that state actually
# *enables* evacuation is exclusively navigation_validation.py's
# concern -- the two modules are non-overlapping by design.


def _pinned_value_matches(resolved_value, fixed_value):

    # resolved_value is either a bool (Exit) or an Enum member (Door/
    # Stair/Obstacle/Camera/Detector, from scenario.engineering_state)
    # -- FixedValue.value is whatever plain primitive
    # scenario_definition authored it as (a bool for exits, an enum
    # .name string for everything else, per scenario_definition/
    # engineering_constraints.py's own documented convention).

    if isinstance(resolved_value, bool):
        return resolved_value == fixed_value.value

    return resolved_value.name == fixed_value.value


def _check_category(report, resolved_states, distribution_map, id_field, value_field, category_name):

    for state in resolved_states:

        object_id = getattr(state, id_field)
        distribution = distribution_map.get(object_id)

        if not isinstance(distribution, FixedValue):
            continue

        resolved_value = getattr(state, value_field)

        if not _pinned_value_matches(resolved_value, distribution):

            report.add(
                FailureCategory.STRUCTURAL, ScenarioValidationReport.ERROR,
                "PINNED_STATE_MISMATCH",
                f"{category_name} {object_id!r} was pinned to {distribution.value!r} "
                f"but resolved to {resolved_value!r}.",
                object_id=object_id,
            )


def validate_engineering(candidate, definition) -> ScenarioValidationReport:

    report = ScenarioValidationReport()
    engineering = definition.engineering

    open_exit_count = sum(1 for state in candidate.exit_states if state.is_open)

    if open_exit_count < engineering.min_open_exits:

        report.add(
            FailureCategory.STRUCTURAL, ScenarioValidationReport.ERROR,
            "MIN_OPEN_EXITS_UNSATISFIED",
            f"Only {open_exit_count} exit(s) are open; min_open_exits requires "
            f"{engineering.min_open_exits}.",
        )

    _check_category(
        report, candidate.door_states, engineering.door_state_distribution,
        "door_id", "state", "Door",
    )
    _check_category(
        report, candidate.exit_states, engineering.exit_state_distribution,
        "exit_id", "is_open", "Exit",
    )
    _check_category(
        report, candidate.stair_states, engineering.stair_state_distribution,
        "stair_id", "availability", "Stair",
    )
    _check_category(
        report, candidate.obstacle_states, engineering.obstacle_state_distribution,
        "obstacle_id", "presence", "Obstacle",
    )
    _check_category(
        report, candidate.camera_states, engineering.camera_state_distribution,
        "camera_id", "availability", "Camera",
    )
    _check_category(
        report, candidate.detector_states, engineering.detector_state_distribution,
        "detector_id", "availability", "Detector",
    )

    return report
