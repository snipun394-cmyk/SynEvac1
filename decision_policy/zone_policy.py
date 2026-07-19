from typing import Any, Dict, FrozenSet, List, Optional


# =====================================================
# Per-zone engineering decisions -- deterministic rules over already-
# computed GroundTruth fields (zone_route_stats, zone_risk_scores,
# hazard fields) plus Scenario's own occupant placements. No AI, no
# ML, no simulation: every branch below is a fixed, disclosed
# threshold, the same "policy constant, not a fabricated data value"
# convention ground_truth/recommendations.py already established.
# =====================================================

SHELTER_IN_PLACE = "SHELTER_IN_PLACE"
EVACUATE_IMMEDIATELY = "EVACUATE_IMMEDIATELY"
WAIT = "WAIT"

# A zone this hazardous is unsafe to evacuate through at all --
# sheltering beats attempting a doomed evacuation.
CRITICAL_RISK_THRESHOLD = 0.75

# A zone at or above this risk should evacuate now, before conditions
# worsen further.
ELEVATED_RISK_THRESHOLD = 0.35


def compute_zone_decisions(
    scenario,
    ground_truth,
    zones,
    current_fire_zone_ids: FrozenSet[str] = frozenset(),
) -> List[Dict[str, Any]]:

    route_stats_by_zone = {
        entry["zone_id"]: entry for entry in ground_truth.zone_route_stats
    }
    risk_by_zone = {
        entry["zone_id"]: entry["risk_score"] for entry in ground_truth.zone_risk_scores
    }

    hazardous_zone_ids = _hazardous_zone_ids(ground_truth, current_fire_zone_ids)
    congested_exit_ids = _congested_exit_ids(ground_truth)
    occupant_counts = _zone_occupant_counts(scenario, zones)

    decisions = []

    for zone in zones:

        route_stats = route_stats_by_zone.get(zone.id, {})
        recommended_exit = route_stats.get("preferred_exit")
        recommended_stair = route_stats.get("preferred_stair")
        risk_score = risk_by_zone.get(zone.id)

        action = _zone_action(
            zone_id=zone.id,
            risk_score=risk_score,
            recommended_exit=recommended_exit,
            recommended_stair=recommended_stair,
            hazardous_zone_ids=hazardous_zone_ids,
            congested_exit_ids=congested_exit_ids,
            occupant_count=occupant_counts.get(zone.id, 0),
        )

        decisions.append(
            {
                "zone_id": zone.id,
                "recommended_exit": recommended_exit,
                "recommended_stair": recommended_stair,
                "action": action,
            }
        )

    return decisions


# =====================================================


def _hazardous_zone_ids(ground_truth, current_fire_zone_ids) -> set:

    hazardous = set(ground_truth.hazard_spread_order) | set(current_fire_zone_ids)

    if ground_truth.maximum_hazard_zone:
        hazardous.add(ground_truth.maximum_hazard_zone)

    if ground_truth.first_hazardous_zone:
        hazardous.add(ground_truth.first_hazardous_zone)

    return hazardous


def _congested_exit_ids(ground_truth) -> set:

    congested = set(ground_truth.exits_exceeding_capacity)

    if ground_truth.worst_exit:
        congested.add(ground_truth.worst_exit)

    return congested


def _zone_occupant_counts(scenario, zones) -> Dict[str, int]:

    counts = {zone.id: 0 for zone in zones}

    for occupant in scenario.occupants:
        if occupant.zone_id in counts:
            counts[occupant.zone_id] += 1

    return counts


def _zone_action(
    zone_id: str,
    risk_score: Optional[float],
    recommended_exit: Optional[str],
    recommended_stair: Optional[str],
    hazardous_zone_ids: set,
    congested_exit_ids: set,
    occupant_count: int,
) -> str:

    no_viable_route = recommended_exit is None and recommended_stair is None

    # Nowhere to go and someone is actually there -- staying put beats
    # attempting a route that doesn't exist.
    if occupant_count > 0 and no_viable_route:
        return SHELTER_IN_PLACE

    if risk_score is not None and risk_score >= CRITICAL_RISK_THRESHOLD:
        return SHELTER_IN_PLACE

    if zone_id in hazardous_zone_ids:
        return EVACUATE_IMMEDIATELY

    if risk_score is not None and risk_score >= ELEVATED_RISK_THRESHOLD:
        return EVACUATE_IMMEDIATELY

    # No hazard urgency -- but if the only route out is already jammed,
    # holding briefly avoids adding to that jam.
    if recommended_exit is not None and recommended_exit in congested_exit_ids:
        return WAIT

    return EVACUATE_IMMEDIATELY
