from collections import defaultdict

from scenario_definition.distributions import FixedValue, UniformRange, WeightedOptions

from scenario_validator.issue import FailureCategory, ScenarioValidationIssue
from scenario_validator.report import ScenarioValidationReport


# Occupant Validation -- architecture doc §5.3, module 2. Splits across
# OCCUPANCY (count/proportion/profile-id-presence) and GEOMETRY
# (polygon containment), per §5.5's explicit exception to the otherwise
# 1:1 module-to-category mapping.

# A soft, documented placeholder, not a validated life-safety density
# threshold -- same honesty convention HazardSeverity.from_score's own
# cutoffs already use. Occupants per square meter of a zone's bounding
# box (Zone.contains()'s own area, since that is what the Generator
# actually samples positions within).
HIGH_DENSITY_THRESHOLD_OCCUPANTS_PER_SQM = 3.0

# The smallest zone-level occupant count worth running a proportion-
# skew check against at all -- below this, sampling noise alone easily
# produces a "missing" option even under a correctly-authored
# distribution, and flagging it would be a false positive, not a real
# signal.
MIN_OCCUPANTS_FOR_PROPORTION_CHECK = 10

# An option this common in the declared distribution "should" appear
# at least once among MIN_OCCUPANTS_FOR_PROPORTION_CHECK+ independent
# draws; if it never does, that's worth a WARNING, never an ERROR --
# proportion-closeness is inherently a statistical, soft property, not
# a hard pass/fail one.
PROPORTION_SKEW_WEIGHT_THRESHOLD = 0.15


def _zone_lookup(building):

    lookup = {}

    for floor in building.floors:
        for zone in floor.zones:
            lookup[zone.id] = zone

    return lookup


def _count_matches_distribution(count, distribution):

    if isinstance(distribution, FixedValue):
        return count == distribution.value

    if isinstance(distribution, UniformRange):
        return distribution.low <= count <= distribution.high

    if isinstance(distribution, WeightedOptions):
        return distribution.weights.get(count, 0) > 0

    return True


def validate_occupants(candidate, definition, building) -> ScenarioValidationReport:

    report = ScenarioValidationReport()

    zones = _zone_lookup(building)
    occupancy_distribution = definition.occupant.occupancy_distribution
    profile_distribution = definition.occupant.behaviour_profile_distribution

    counts_by_zone = defaultdict(int)
    profiles_by_zone = defaultdict(list)

    for occupant in candidate.occupants:

        counts_by_zone[occupant.zone_id] += 1
        profiles_by_zone[occupant.zone_id].append(occupant.behaviour_profile_id)

        if not occupant.zone_id:

            report.add(
                FailureCategory.OCCUPANCY, ScenarioValidationReport.ERROR,
                "OCCUPANT_MISSING_ZONE",
                f"Occupant {occupant.occupant_id!r} does not belong to any zone.",
                object_id=occupant.occupant_id,
            )
            continue

        zone = zones.get(occupant.zone_id)

        if zone is not None and not zone.contains(*occupant.position):

            report.add(
                FailureCategory.GEOMETRY, ScenarioValidationReport.ERROR,
                "OCCUPANT_OUTSIDE_ZONE",
                f"Occupant {occupant.occupant_id!r} at {occupant.position} lies "
                f"outside zone {occupant.zone_id!r}.",
                object_id=occupant.occupant_id,
            )

        if not occupant.behaviour_profile_id:

            report.add(
                FailureCategory.OCCUPANCY, ScenarioValidationReport.ERROR,
                "MISSING_BEHAVIOUR_PROFILE_ID",
                f"Occupant {occupant.occupant_id!r} has no behaviour_profile_id -- "
                f"the Definition states no rule for zone {occupant.zone_id!r} "
                f"(§8: an opaque identifier is required, format only, never a "
                f"registry lookup).",
                object_id=occupant.occupant_id,
            )

    for zone_id, count in counts_by_zone.items():

        distribution = occupancy_distribution.get(zone_id)

        if distribution is not None and not _count_matches_distribution(count, distribution):

            report.add(
                FailureCategory.OCCUPANCY, ScenarioValidationReport.ERROR,
                "OCCUPANT_COUNT_OUT_OF_SUPPORT",
                f"Zone {zone_id!r} has {count} occupant(s), outside the support of "
                f"its declared occupancy_distribution.",
                object_id=zone_id,
            )

        zone = zones.get(zone_id)

        if zone is not None and zone.width > 0 and zone.height > 0:

            density = count / (zone.width * zone.height)

            if density > HIGH_DENSITY_THRESHOLD_OCCUPANTS_PER_SQM:

                report.add(
                    FailureCategory.OCCUPANCY, ScenarioValidationReport.WARNING,
                    "HIGH_OCCUPANCY_DENSITY",
                    f"Zone {zone_id!r} has a high occupancy density "
                    f"({density:.2f} occupants/m^2) -- scenario still valid.",
                    object_id=zone_id,
                )

    for zone_id, profile_ids in profiles_by_zone.items():

        distribution = profile_distribution.get(zone_id)

        if not isinstance(distribution, WeightedOptions):
            continue

        for profile_id in profile_ids:

            if profile_id and distribution.weights.get(profile_id, 0) <= 0:

                report.add(
                    FailureCategory.OCCUPANCY, ScenarioValidationReport.ERROR,
                    "BEHAVIOUR_PROFILE_NOT_IN_SUPPORT",
                    f"Occupant behaviour_profile_id {profile_id!r} in zone "
                    f"{zone_id!r} is not a member of the declared "
                    f"behaviour_profile_distribution -- this checks membership "
                    f"only (a format/support check), never whether the id is a "
                    f"real registered Behaviour Profile (§3.4/§8).",
                    object_id=zone_id,
                )

        if len(profile_ids) >= MIN_OCCUPANTS_FOR_PROPORTION_CHECK:

            observed = set(profile_ids)

            for option, weight in distribution.weights.items():

                if weight >= PROPORTION_SKEW_WEIGHT_THRESHOLD and option not in observed:

                    report.add(
                        FailureCategory.OCCUPANCY, ScenarioValidationReport.WARNING,
                        "BEHAVIOUR_PROFILE_PROPORTION_SKEW",
                        f"Zone {zone_id!r} sampled {len(profile_ids)} occupants but "
                        f"never drew {option!r} (declared weight {weight}) -- "
                        f"scenario still valid; proportions look skewed relative to "
                        f"the declared distribution.",
                        object_id=zone_id,
                    )

    return report
