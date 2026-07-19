from scenario_definition.distributions import FixedValue, UniformRange, WeightedOptions

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Fire Validation -- architecture doc §5.3, module 3. Re-checks
# independently rather than trusting the Generator got the allow/
# forbid rules and §4.7's default-sampling policy right -- the same
# Generator-bug-vs-Definition-mistake reasoning as Building Validation.


def _value_matches_distribution(value, distribution):

    if isinstance(distribution, FixedValue):
        return value == distribution.value

    if isinstance(distribution, UniformRange):
        return distribution.low <= value <= distribution.high

    if isinstance(distribution, WeightedOptions):
        return distribution.weights.get(value, 0) > 0

    return True


def validate_fire(candidate, definition) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    if candidate.fire is None:

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "MISSING_FIRE", "The candidate has no fire state at all.",
        )
        return report

    fire = candidate.fire
    fire_def = definition.fire

    if fire_def.allowed_ignition_zone_ids and fire.ignition_zone_id not in fire_def.allowed_ignition_zone_ids:

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "IGNITION_ZONE_NOT_ALLOWED",
            f"Ignition zone {fire.ignition_zone_id!r} is not in "
            f"allowed_ignition_zone_ids.",
            object_id=fire.ignition_zone_id,
        )

    if fire.ignition_zone_id in fire_def.forbidden_ignition_zone_ids:

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "IGNITION_ZONE_FORBIDDEN",
            f"Ignition zone {fire.ignition_zone_id!r} is in "
            f"forbidden_ignition_zone_ids.",
            object_id=fire.ignition_zone_id,
        )

    if (
        fire_def.allowed_ignition_floor_ids
        and fire.ignition_floor_id not in fire_def.allowed_ignition_floor_ids
    ):

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "IGNITION_FLOOR_NOT_ALLOWED",
            f"Ignition floor {fire.ignition_floor_id!r} is not in "
            f"allowed_ignition_floor_ids.",
            object_id=fire.ignition_floor_id,
        )

    if fire_def.allowed_fire_profiles and fire.fire_profile not in fire_def.allowed_fire_profiles:

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "FIRE_PROFILE_NOT_ALLOWED",
            f"Fire profile {fire.fire_profile!r} is not in allowed_fire_profiles.",
            object_id=fire.fire_profile,
        )

    growth_time = fire.growth_parameters.get("growth_time")

    if growth_time is not None and not _value_matches_distribution(
        growth_time, fire_def.growth_parameter_distribution,
    ):

        report.add(
            FailureCategory.FIRE, ScenarioValidationReport.ERROR,
            "GROWTH_PARAMETERS_OUT_OF_SUPPORT",
            f"Fire growth_time {growth_time!r} is outside the support of "
            f"growth_parameter_distribution.",
        )

    return report
