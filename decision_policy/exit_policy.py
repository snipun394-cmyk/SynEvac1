from typing import Any, Dict, List


# =====================================================
# Per-exit engineering decisions. Baseline open/closed state is read
# the same way dataset_builder/ground_truth already do (Scenario
# override for this exit id, else the Exit model's own is_blocked
# baseline) -- never inferred, never re-simulated.
# =====================================================

KEEP_OPEN = "KEEP_OPEN"
OPEN = "OPEN"
CLOSE = "CLOSE"
HIGH_CONGESTION = "HIGH_CONGESTION"


def compute_exit_decisions(scenario, ground_truth, exits, zone_decisions) -> List[Dict[str, Any]]:

    baseline_open = _exit_open_lookup(scenario, exits)

    recommended_by_any_zone = {
        decision["recommended_exit"]
        for decision in zone_decisions
        if decision.get("recommended_exit")
    }

    hazardous_zone_ids = set(ground_truth.hazard_spread_order)
    if ground_truth.maximum_hazard_zone:
        hazardous_zone_ids.add(ground_truth.maximum_hazard_zone)

    congested_ids = set(ground_truth.exits_exceeding_capacity)
    if ground_truth.worst_exit:
        congested_ids.add(ground_truth.worst_exit)

    decisions = []

    for exit_obj in exits:

        status = _exit_status(
            exit_obj=exit_obj,
            is_open=baseline_open.get(exit_obj.id, True),
            hazardous_zone_ids=hazardous_zone_ids,
            congested_ids=congested_ids,
            recommended_by_any_zone=exit_obj.id in recommended_by_any_zone,
            people_trapped=ground_truth.people_trapped,
        )

        decisions.append({"exit_id": exit_obj.id, "status": status})

    return decisions


# =====================================================


def _exit_open_lookup(scenario, exits) -> Dict[str, bool]:

    lookup = {exit_obj.id: not exit_obj.is_blocked for exit_obj in exits}

    for state in scenario.exit_states:
        lookup[state.exit_id] = state.is_open

    return lookup


def _exit_status(
    exit_obj,
    is_open: bool,
    hazardous_zone_ids: set,
    congested_ids: set,
    recommended_by_any_zone: bool,
    people_trapped: int,
) -> str:

    # The zone an exit is staged in (Exit.zone_id) being hazardous
    # means occupants can't safely reach the exit itself, regardless
    # of the exit's own open/closed state -- routing anyone there is
    # unsafe.
    if exit_obj.zone_id in hazardous_zone_ids:
        return CLOSE

    if exit_obj.id in congested_ids:
        return HIGH_CONGESTION

    if not is_open:

        if recommended_by_any_zone or people_trapped > 0:
            return OPEN

        return CLOSE

    return KEEP_OPEN
