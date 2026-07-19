from typing import Any, Dict

from behaviour_profile_resolver.occupant_attributes import derive_occupant_attributes
from behaviour_profile_resolver.occupant_grouping import assign_occupant_groups

from simulator.occupant import OccupantState


# =====================================================
# Occupant Attributes Phase 4 -- Ground Truth outcome summaries.
# Mirrors ground_truth/profile_outcomes.py's own shape exactly: cross
# an already-computed classification (here, occupant_attributes.
# derive_occupant_attributes()/occupant_grouping.assign_occupant_groups()
# -- both pure, deterministic functions of already-fixed Scenario data,
# same as behaviour_profile_resolver.category.occupant_category() that
# profile_outcomes.py itself reads) against the same authoritative "did
# they reach OccupantState.ARRIVED" signal every other evacuation-
# outcome computation in this package already uses. Nothing here
# samples, simulates, or infers a value -- every mean/rate below is a
# plain arithmetic summary over values the Scenario's own seed already
# fixed before this simulation ever ran.
#
# A bounded, meaningful set of summaries, not an exhaustive one: three
# representative attributes (walking_speed_multiplier, risk_aversion,
# panic_susceptibility) split evacuated vs. trapped, plus Phase 3's own
# group cohesion split (grouped vs. ungrouped evacuation rate).
# =====================================================


def compute_occupant_attribute_outcomes(scenario, movement_result) -> Dict[str, Any]:

    timelines = movement_result.occupants
    group_assignments = assign_occupant_groups(scenario)

    walking_speed_evacuated, walking_speed_trapped = [], []
    risk_aversion_evacuated, risk_aversion_trapped = [], []
    panic_evacuated, panic_trapped = [], []

    grouped_evacuated_count = grouped_total_count = 0
    ungrouped_evacuated_count = ungrouped_total_count = 0

    for occupant in scenario.occupants:

        attributes = derive_occupant_attributes(
            scenario.metadata.seed, occupant.occupant_id, occupant.behaviour_profile_id,
        )

        timeline = timelines.get(occupant.occupant_id)
        evacuated = timeline is not None and timeline.state == OccupantState.ARRIVED

        (walking_speed_evacuated if evacuated else walking_speed_trapped).append(
            attributes.walking_speed_multiplier,
        )
        (risk_aversion_evacuated if evacuated else risk_aversion_trapped).append(
            attributes.risk_aversion,
        )
        (panic_evacuated if evacuated else panic_trapped).append(attributes.panic_susceptibility)

        if occupant.occupant_id in group_assignments:

            grouped_total_count += 1
            if evacuated:
                grouped_evacuated_count += 1

        else:

            ungrouped_total_count += 1
            if evacuated:
                ungrouped_evacuated_count += 1

    group_sizes: Dict[str, int] = {}
    for assignment in group_assignments.values():
        group_sizes[assignment.group_id] = group_sizes.get(assignment.group_id, 0) + 1

    return {
        "mean_walking_speed_multiplier_evacuated": _mean(walking_speed_evacuated),
        "mean_walking_speed_multiplier_trapped": _mean(walking_speed_trapped),
        "mean_risk_aversion_evacuated": _mean(risk_aversion_evacuated),
        "mean_risk_aversion_trapped": _mean(risk_aversion_trapped),
        "mean_panic_susceptibility_evacuated": _mean(panic_evacuated),
        "mean_panic_susceptibility_trapped": _mean(panic_trapped),
        "group_count": len(group_sizes),
        "grouped_occupant_count": len(group_assignments),
        "mean_group_size": _mean(list(group_sizes.values())),
        "grouped_evacuation_rate": _rate(grouped_evacuated_count, grouped_total_count),
        "ungrouped_evacuation_rate": _rate(ungrouped_evacuated_count, ungrouped_total_count),
    }


def _mean(values):

    return sum(values) / len(values) if values else None


def _rate(count, total):

    return count / total if total else None
