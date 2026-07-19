from typing import Any, Dict

from behaviour_profile_resolver.category import OccupantCategory, occupant_category

from simulator.occupant import OccupantState


# =====================================================
# Human Population & Assisted Evacuation Modeling Phase 7 -- profile-
# aware outcome counts. behaviour_profile_resolver.category is already
# the one place in this platform allowed to interpret what a
# behaviour_profile_id represents (see that module's own docstring);
# this module only crosses that classification against the same
# authoritative "did they reach OccupantState.ARRIVED" signal every
# other evacuation-outcome computation in this package already reads
# (ground_truth.evacuation_metrics.compute_evacuation_metrics(), this
# module's own sibling).
# =====================================================


def compute_profile_outcomes(scenario, movement_result) -> Dict[str, Any]:

    timelines = movement_result.occupants

    wheelchair_evacuated = 0
    wheelchair_trapped = 0
    children_evacuated = 0
    children_trapped = 0

    for occupant in scenario.occupants:

        category = occupant_category(occupant.behaviour_profile_id)

        if category not in (OccupantCategory.WHEELCHAIR_USER, OccupantCategory.CHILD):
            continue

        timeline = timelines.get(occupant.occupant_id)
        evacuated = timeline is not None and timeline.state == OccupantState.ARRIVED

        if category == OccupantCategory.WHEELCHAIR_USER:

            if evacuated:
                wheelchair_evacuated += 1
            else:
                wheelchair_trapped += 1

        elif category == OccupantCategory.CHILD:

            if evacuated:
                children_evacuated += 1
            else:
                children_trapped += 1

    return {
        "wheelchair_evacuated": wheelchair_evacuated,
        "wheelchair_trapped": wheelchair_trapped,
        "children_evacuated": children_evacuated,
        "children_trapped": children_trapped,
    }
