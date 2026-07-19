from dataclasses import dataclass

from scenario_definition.distributions import FixedValue, UniformRange, WeightedOptions


# DefinitionValidationIssue/DefinitionValidationReport -- structurally
# identical to navigation.validation.ValidationIssue/ValidationReport
# (errors/warnings list, add()/is_valid/errors/warnings/by_code(), no
# behavior beyond accumulation), but locally defined rather than
# imported -- architecture doc §3.1 finding 3: importing
# navigation/validation.py's shape would mean scenario_definition/
# importing navigation, a package this document elsewhere insists
# Definitions must never contain state from. Any resemblance to
# ValidationReport's shape is convention-following, not a dependency.


@dataclass
class DefinitionValidationIssue:

    code: str
    severity: str
    message: str

    object_id: str = ""
    floor_id: str = ""


class DefinitionValidationReport:

    ERROR = "error"
    WARNING = "warning"

    def __init__(self):

        self.issues = []

    # =====================================================

    def add(self, code, message, severity="error", object_id="", floor_id=""):

        self.issues.append(
            DefinitionValidationIssue(
                code=code,
                severity=severity,
                message=message,
                object_id=object_id,
                floor_id=floor_id,
            )
        )

    # =====================================================

    @property
    def is_valid(self):

        return not any(issue.severity == self.ERROR for issue in self.issues)

    # =====================================================

    @property
    def errors(self):

        return [issue for issue in self.issues if issue.severity == self.ERROR]

    # =====================================================

    @property
    def warnings(self):

        return [issue for issue in self.issues if issue.severity == self.WARNING]

    # =====================================================

    def by_code(self, code):

        return [issue for issue in self.issues if issue.code == code]

    # =====================================================

    def __len__(self):

        return len(self.issues)

    def __iter__(self):

        return iter(self.issues)

    # =====================================================

    def __repr__(self):

        return f"DefinitionValidationReport(errors={len(self.errors)}, warnings={len(self.warnings)})"


# =====================================================
# Self-validation -- architecture doc §3.4. Checks structural
# well-formedness of the rulebook only: internally consistent, and
# refers to things that exist. Never samples anything, never checks
# whether any scenario satisfying the Definition is reachable,
# winnable, or "makes sense" as a fire scenario -- that is exclusively
# a Scenario Validator's job (§5), against a concrete sampled
# candidate, and stays completely out of scope here.
#
# `building` is accepted as an explicit parameter (the frozen schema's
# `ScenarioDefinition.validate() -> DefinitionValidationReport` line
# does not show one, and §3.2's field table has no building_id/
# Building-reference field for a Definition to carry on its own) --
# the only way to perform "ids referenced anywhere actually exist on
# the referenced Building" without ScenarioDefinition owning a
# Building reference nowhere else in the frozen schema. Building-
# dependent checks are skipped (not failed) when building is None, the
# same defensive default designer/validation.py's
# validate_building_authoring() already uses for a missing building.
# =====================================================


def validate_definition(definition, building=None) -> DefinitionValidationReport:

    report = DefinitionValidationReport()

    _check_distribution_shapes(report, definition)
    _check_occupancy_counts(report, definition)
    _check_event_templates(report, definition)
    _check_ignition_allow_forbid_disjoint(report, definition)

    if building is not None:
        _check_ids_exist_on_building(report, definition, building)
        _check_min_open_exits_feasible(report, definition, building)
        _check_allowed_floor_has_a_reachable_zone(report, definition, building)

    return report


# =====================================================
# Generic Distribution-shape checks -- apply uniformly to every
# Distribution anywhere in the Definition, regardless of which field
# it lives in.
# =====================================================


def _iter_distributions(definition):

    fire = definition.fire

    yield "fire.growth_parameter_distribution", fire.growth_parameter_distribution

    if fire.ignition_zone_preference is not None:
        yield "fire.ignition_zone_preference", fire.ignition_zone_preference

    for zone_id, distribution in definition.occupant.occupancy_distribution.items():
        yield f"occupant.occupancy_distribution[{zone_id}]", distribution

    for zone_id, distribution in definition.occupant.behaviour_profile_distribution.items():
        yield f"occupant.behaviour_profile_distribution[{zone_id}]", distribution

    engineering = definition.engineering

    for category in (
        "door_state_distribution",
        "exit_state_distribution",
        "stair_state_distribution",
        "obstacle_state_distribution",
        "camera_state_distribution",
        "detector_state_distribution",
    ):
        for object_id, distribution in getattr(engineering, category).items():
            yield f"engineering.{category}[{object_id}]", distribution

    for index, template in enumerate(definition.event_templates):
        yield f"event_templates[{index}].occurs", template.occurs
        yield f"event_templates[{index}].time", template.time


def _check_distribution_shapes(report, definition):

    for location, distribution in _iter_distributions(definition):

        if isinstance(distribution, UniformRange):

            if distribution.low > distribution.high:
                report.add(
                    "invalid_range",
                    f"{location}: UniformRange.low ({distribution.low}) exceeds "
                    f"UniformRange.high ({distribution.high}).",
                    severity=DefinitionValidationReport.ERROR,
                )

        elif isinstance(distribution, WeightedOptions):

            if not distribution.weights:
                report.add(
                    "empty_weighted_options",
                    f"{location}: WeightedOptions has no options.",
                    severity=DefinitionValidationReport.ERROR,
                )
                continue

            if any(weight < 0 for weight in distribution.weights.values()):
                report.add(
                    "negative_weight",
                    f"{location}: WeightedOptions has a negative weight.",
                    severity=DefinitionValidationReport.ERROR,
                )

            if sum(distribution.weights.values()) <= 0:
                report.add(
                    "non_positive_weight_total",
                    f"{location}: WeightedOptions weights sum to <= 0.",
                    severity=DefinitionValidationReport.ERROR,
                )


# =====================================================
# Occupants -- negative occupant counts.
# =====================================================


def _check_occupancy_counts(report, definition):

    for zone_id, distribution in definition.occupant.occupancy_distribution.items():

        if isinstance(distribution, FixedValue):

            if isinstance(distribution.value, (int, float)) and distribution.value < 0:
                report.add(
                    "negative_occupant_count",
                    f"occupant.occupancy_distribution[{zone_id}]: fixed occupant count "
                    f"{distribution.value} is negative.",
                    severity=DefinitionValidationReport.ERROR,
                    object_id=zone_id,
                )

        elif isinstance(distribution, UniformRange):

            if distribution.low < 0 or distribution.high < 0:
                report.add(
                    "negative_occupant_count",
                    f"occupant.occupancy_distribution[{zone_id}]: occupant count range "
                    f"[{distribution.low}, {distribution.high}] includes a negative value.",
                    severity=DefinitionValidationReport.ERROR,
                    object_id=zone_id,
                )


# =====================================================
# Events -- missing required fields (an empty target/event type is
# structurally present but semantically absent, since Python's
# dataclass mechanics already prevent a field from being truly missing
# post-construction).
# =====================================================


def _check_event_templates(report, definition):

    for index, template in enumerate(definition.event_templates):

        for field_name in ("target_type", "target_id", "event_type"):

            if not getattr(template, field_name):
                report.add(
                    "missing_required_field",
                    f"event_templates[{index}].{field_name} is required and must not "
                    f"be empty.",
                    severity=DefinitionValidationReport.ERROR,
                )


# =====================================================
# Fire allow/forbid consistency -- building-independent half.
# =====================================================


def _check_ignition_allow_forbid_disjoint(report, definition):

    fire = definition.fire

    shared = fire.allowed_ignition_zone_ids & fire.forbidden_ignition_zone_ids

    if shared:
        report.add(
            "ignition_allow_forbid_overlap",
            f"fire: zone id(s) {sorted(shared)} appear in both "
            f"allowed_ignition_zone_ids and forbidden_ignition_zone_ids.",
            severity=DefinitionValidationReport.ERROR,
        )


# =====================================================
# Building-dependent checks.
# =====================================================


def _all_object_ids(building):

    ids = {"zone": set(), "door": set(), "exit": set(), "stair": set(), "obstacle": set(),
           "camera": set(), "detector": set(), "floor": set()}

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

        # A Staircase is one physical connector spanning exactly two
        # floors, rendered on both (models/staircase.py) -- appears in
        # floor.stairs on each of its two floors, so the set naturally
        # dedupes by id regardless.
        for staircase in floor.stairs:
            ids["stair"].add(staircase.id)

    return ids


def _check_ids_exist_on_building(report, definition, building):

    ids = _all_object_ids(building)

    def check_membership(id_value, category, location):

        if id_value and id_value not in ids[category]:
            report.add(
                "unknown_id",
                f"{location}: {category} id {id_value!r} does not exist on the "
                f"referenced Building.",
                severity=DefinitionValidationReport.ERROR,
                object_id=id_value,
            )

    fire = definition.fire

    for zone_id in fire.allowed_ignition_zone_ids:
        check_membership(zone_id, "zone", "fire.allowed_ignition_zone_ids")

    for zone_id in fire.forbidden_ignition_zone_ids:
        check_membership(zone_id, "zone", "fire.forbidden_ignition_zone_ids")

    for floor_id in fire.allowed_ignition_floor_ids:
        check_membership(floor_id, "floor", "fire.allowed_ignition_floor_ids")

    for zone_id in definition.occupant.occupancy_distribution:
        check_membership(zone_id, "zone", "occupant.occupancy_distribution")

    for zone_id in definition.occupant.behaviour_profile_distribution:
        check_membership(zone_id, "zone", "occupant.behaviour_profile_distribution")

    engineering = definition.engineering
    category_by_field = {
        "door_state_distribution": "door",
        "exit_state_distribution": "exit",
        "stair_state_distribution": "stair",
        "obstacle_state_distribution": "obstacle",
        "camera_state_distribution": "camera",
        "detector_state_distribution": "detector",
    }

    for field_name, category in category_by_field.items():
        for object_id in getattr(engineering, field_name):
            check_membership(object_id, category, f"engineering.{field_name}")

    for index, template in enumerate(definition.event_templates):

        category = category_by_field.get(f"{template.target_type}_state_distribution")

        if category is not None:
            check_membership(
                template.target_id, category, f"event_templates[{index}].target_id",
            )


def _resolvable_open(distribution):

    # "Could possibly resolve OPEN at all" (§3.4) -- optimistic in
    # every case except a hard FixedValue(False)/pinned-closed
    # commitment, which never counts toward feasibility even
    # optimistically.

    if isinstance(distribution, FixedValue):
        return bool(distribution.value)

    if isinstance(distribution, WeightedOptions):
        return distribution.weights.get(True, 0) > 0

    return True


def _check_min_open_exits_feasible(report, definition, building):

    ids = _all_object_ids(building)
    exit_state_distribution = definition.engineering.exit_state_distribution

    possibly_open_count = 0

    for exit_id in ids["exit"]:

        distribution = exit_state_distribution.get(exit_id)

        # Absent from the dict means "default rule applies" (§3.2) --
        # this module has no opinion on what that default resolves to
        # beyond the one named exception (a hard CLOSED pin never
        # counts), so an absent entry is counted optimistically, same
        # as every other undetermined case.
        if distribution is None or _resolvable_open(distribution):
            possibly_open_count += 1

    if definition.engineering.min_open_exits > possibly_open_count:
        report.add(
            "min_open_exits_infeasible",
            f"engineering.min_open_exits ({definition.engineering.min_open_exits}) "
            f"exceeds the number of exits that could possibly resolve OPEN "
            f"({possibly_open_count}).",
            severity=DefinitionValidationReport.ERROR,
        )


def _check_allowed_floor_has_a_reachable_zone(report, definition, building):

    fire = definition.fire

    if not fire.allowed_ignition_floor_ids:
        return

    zones_by_floor = {}

    for floor in building.floors:
        zones_by_floor[floor.id] = {zone.id for zone in floor.zones}

    for floor_id in fire.allowed_ignition_floor_ids:

        zone_ids = zones_by_floor.get(floor_id, set())

        eligible = zone_ids - fire.forbidden_ignition_zone_ids

        if fire.allowed_ignition_zone_ids:
            eligible &= fire.allowed_ignition_zone_ids

        if not eligible:
            report.add(
                "allowed_floor_has_no_eligible_zone",
                f"fire.allowed_ignition_floor_ids: floor {floor_id!r} has no zone "
                f"that is both allowed and not forbidden.",
                severity=DefinitionValidationReport.ERROR,
                floor_id=floor_id,
            )
