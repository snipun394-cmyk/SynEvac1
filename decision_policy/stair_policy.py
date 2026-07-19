from typing import Any, Dict, FrozenSet, List


# =====================================================
# Per-stair engineering decisions. Staircase carries no modeled
# capacity anywhere in this codebase (see ground_truth/bottleneck.py's
# identical note), so stair_risk_scores is already the fraction of the
# stair's own crossings that had to queue -- reused as-is here rather
# than inventing a second stair risk measure.
# =====================================================

USE = "USE"
AVOID = "AVOID"
CONGESTED = "CONGESTED"

# A stair this often forcing occupants to queue -- or one directly
# connecting to a currently/eventually hazardous zone -- is unsafe to
# route anyone toward.
AVOID_RISK_THRESHOLD = 0.6

# Below AVOID but still frequently causing queueing -- usable, but
# should be flagged as slow.
CONGESTED_RISK_THRESHOLD = 0.3


def compute_stair_decisions(
    ground_truth,
    stairs,
    current_fire_zone_ids: FrozenSet[str] = frozenset(),
) -> List[Dict[str, Any]]:

    risk_by_stair = {
        entry["stair_id"]: entry["risk_score"] for entry in ground_truth.stair_risk_scores
    }

    congested_ids = set(ground_truth.stairs_exceeding_capacity)
    if ground_truth.worst_stair:
        congested_ids.add(ground_truth.worst_stair)

    hazardous_zone_ids = set(ground_truth.hazard_spread_order) | set(current_fire_zone_ids)
    if ground_truth.maximum_hazard_zone:
        hazardous_zone_ids.add(ground_truth.maximum_hazard_zone)

    decisions = []

    for stair in stairs:

        risk_score = risk_by_stair.get(stair.id)

        connects_to_hazard = (
            stair.from_zone_id in hazardous_zone_ids or stair.to_zone_id in hazardous_zone_ids
        )

        if connects_to_hazard or (risk_score is not None and risk_score >= AVOID_RISK_THRESHOLD):
            status = AVOID
        elif stair.id in congested_ids or (
            risk_score is not None and risk_score >= CONGESTED_RISK_THRESHOLD
        ):
            status = CONGESTED
        else:
            status = USE

        decisions.append({"stair_id": stair.id, "status": status})

    return decisions
