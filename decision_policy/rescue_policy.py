from typing import Any, Dict, List, Tuple

from decision_policy.exit_policy import CLOSE
from decision_policy.stair_policy import AVOID


# =====================================================
# Rescue priority/order and firefighter deployment priority --
# deterministic ranking over GroundTruth's own zone_risk_scores and
# Scenario's own occupant placements. impact_score = risk x headcount
# is a disclosed, fixed formula (severity times how many people it
# affects), not a fabricated value.
# =====================================================

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MODERATE = "MODERATE"
LOW = "LOW"
NONE_LEVEL = "NONE"

CRITICAL_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.5
MODERATE_THRESHOLD = 0.25


def compute_rescue_priorities(scenario, ground_truth, zones) -> Tuple[List[Dict[str, Any]], Tuple[str, ...]]:

    occupant_counts = {zone.id: 0 for zone in zones}
    for occupant in scenario.occupants:
        if occupant.zone_id in occupant_counts:
            occupant_counts[occupant.zone_id] += 1

    risk_by_zone = {
        entry["zone_id"]: entry["risk_score"] for entry in ground_truth.zone_risk_scores
    }

    priorities = []

    for zone in zones:

        occupant_count = occupant_counts.get(zone.id, 0)
        risk_score = risk_by_zone.get(zone.id)

        if ground_truth.building_cleared or occupant_count == 0 or risk_score is None:
            level = NONE_LEVEL
            impact_score = 0.0
        else:
            impact_score = risk_score * occupant_count
            level = _priority_level(risk_score)

        priorities.append(
            {
                "zone_id": zone.id,
                "rescue_priority": level,
                "impact_score": impact_score,
                "occupant_count": occupant_count,
            }
        )

    ranked = sorted(
        (entry for entry in priorities if entry["rescue_priority"] != NONE_LEVEL),
        key=lambda entry: entry["impact_score"],
        reverse=True,
    )

    rescue_order = tuple(entry["zone_id"] for entry in ranked)

    return priorities, rescue_order


# =====================================================


def _priority_level(risk_score: float) -> str:

    if risk_score >= CRITICAL_THRESHOLD:
        return CRITICAL

    if risk_score >= HIGH_THRESHOLD:
        return HIGH

    if risk_score >= MODERATE_THRESHOLD:
        return MODERATE

    return LOW


# =====================================================


def compute_firefighter_deployment_priority(
    rescue_priorities, rescue_order, stair_decisions, exit_decisions,
) -> List[Dict[str, Any]]:

    # Fixed category order -- rescue zones (people first), then
    # unsafe/bottlenecked stairs, then exits that lead through hazard --
    # each category internally already ranked by its own upstream
    # ordering (rescue_order's own impact_score sort; stair/exit
    # decisions in their own Building-enumeration order).

    priority_by_zone = {entry["zone_id"]: entry for entry in rescue_priorities}

    entries = []
    rank = 1

    for zone_id in rescue_order:

        entries.append(
            {
                "rank": rank,
                "target_type": "zone",
                "target_id": zone_id,
                "reason": f"Rescue priority {priority_by_zone[zone_id]['rescue_priority']}",
            }
        )
        rank += 1

    for decision in stair_decisions:

        if decision["status"] == AVOID:

            entries.append(
                {
                    "rank": rank,
                    "target_type": "stair",
                    "target_id": decision["stair_id"],
                    "reason": "Stair marked AVOID -- unsafe or severely bottlenecked",
                }
            )
            rank += 1

    for decision in exit_decisions:

        if decision["status"] == CLOSE:

            entries.append(
                {
                    "rank": rank,
                    "target_type": "exit",
                    "target_id": decision["exit_id"],
                    "reason": "Exit marked CLOSE -- leads through a hazardous zone",
                }
            )
            rank += 1

    return entries
