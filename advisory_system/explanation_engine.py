from typing import List, Optional, Tuple

from advisory_system.recommendation_models import Explanation


# =====================================================
# Explanation Engine. Every reason string produced here is built
# exclusively from fields an upstream package already computed
# (GroundTruth's own risk/congestion/hazard fields, DecisionPolicy's
# own zone/exit/stair decisions) -- this module contains no threshold,
# no judgment, and no new arithmetic beyond plain differences between
# two already-known numbers (see estimate_rset_improvement() below).
# If a fact isn't already available on GroundTruth/DecisionPolicy, it
# is simply omitted from the explanation -- never invented to fill out
# a sentence.
# =====================================================


def compute_redirect_target(ground_truth, zone_id: str) -> Optional[str]:

    # A genuine alternative only exists when this zone's own preferred
    # exit is itself flagged overloaded and at least one other exit is
    # underutilized -- exactly the same two signals ground_truth.
    # recommendations._redirect_recommendations() already requires
    # before proposing a redirect, independently re-checked here (not
    # imported: this module reads ground_truth.recommendations' output
    # for its per-item reason text elsewhere, but the redirect *target*
    # itself is embedded only in that module's own free-text action
    # string, not a separate field, so recomputing from the same two
    # underlying GroundTruth fields is more honest than parsing text).
    # None whenever no redirect condition holds -- a zone already using
    # its own uncongested default route has no "alternative" to report.

    route_stats_by_zone = {entry["zone_id"]: entry for entry in ground_truth.zone_route_stats}
    entry = route_stats_by_zone.get(zone_id)

    if entry is None or entry.get("preferred_exit") is None:
        return None

    preferred_exit = entry["preferred_exit"]

    overloaded = set(ground_truth.exits_exceeding_capacity)
    if ground_truth.worst_exit:
        overloaded.add(ground_truth.worst_exit)

    if preferred_exit not in overloaded:
        return None

    underutilized = [exit_id for exit_id in ground_truth.exits_underutilized if exit_id != preferred_exit]

    if not underutilized:
        return None

    return underutilized[0]


def estimate_rset_improvement(ground_truth, zone_id: str, alternative_exit_id: Optional[str]) -> Optional[float]:

    # Only ever called with a genuine redirect target (see
    # compute_redirect_target() above / advisory_engine._resolve_zone_action())
    # -- comparing against a zone's own already-recommended exit would
    # be a category error (there is no "improvement" from staying on
    # the same route). Proxy, disclosed: GroundTruth.zone_route_stats
    # holds one entry per zone (that zone's own actually-taken/preferred
    # route), not one entry per (zone, candidate exit) pair -- there is
    # no re-simulated "how long would this zone take via the
    # alternative exit" figure anywhere in this platform. The honest
    # approximation available is another zone's own average_travel_time
    # whose *own* preferred_exit already is the candidate alternative --
    # i.e. "occupants who actually used that exit took about this
    # long," reused as a stand-in. None (never a guessed number) when
    # no such comparison point exists.

    if alternative_exit_id is None:
        return None

    route_stats = {entry["zone_id"]: entry for entry in ground_truth.zone_route_stats}

    current = route_stats.get(zone_id)
    if current is None or current.get("average_travel_time") is None:
        return None

    alternative_time = None
    for entry in ground_truth.zone_route_stats:

        if entry["zone_id"] == zone_id:
            continue

        if entry.get("preferred_exit") == alternative_exit_id and entry.get("average_travel_time") is not None:
            alternative_time = entry["average_travel_time"]
            break

    if alternative_time is None:
        return None

    return current["average_travel_time"] - alternative_time


def explain_zone_recommendation(
    *,
    zone_id: str,
    action: str,
    recommended_exit: Optional[str],
    ground_truth,
    confidence: Optional[float],
    rset_improvement: Optional[float],
) -> Explanation:

    reasons: List[str] = []

    risk_by_zone = {entry["zone_id"]: entry["risk_score"] for entry in ground_truth.zone_risk_scores}
    risk_score = risk_by_zone.get(zone_id)

    if zone_id == ground_truth.maximum_hazard_zone:
        reasons.append(f"Zone {zone_id} is the most hazardous zone in this incident.")
    elif zone_id in ground_truth.hazard_spread_order:
        reasons.append(f"Zone {zone_id} lies on the fire's own observed spread path.")

    if risk_score is not None:
        reasons.append(f"Zone {zone_id} risk score is {risk_score:.2f}.")

    congested_exits = set(ground_truth.exits_exceeding_capacity)
    if ground_truth.worst_exit:
        congested_exits.add(ground_truth.worst_exit)

    if recommended_exit and recommended_exit in congested_exits:
        reasons.append(f"Exit {recommended_exit} is at or over its modeled capacity.")

    if rset_improvement is not None and rset_improvement > 0:
        reasons.append(
            f"Recommended route reduces predicted evacuation time by {rset_improvement:.0f} seconds "
            f"relative to Zone {zone_id}'s own default route.",
        )

    if not reasons:
        reasons.append(f"Zone {zone_id}'s own Decision Policy action is {action}.")

    summary = f"{action.replace('_', ' ').title()} for Zone {zone_id}"

    return Explanation(summary=summary, reasons=tuple(reasons), confidence=confidence)


def explain_building_recommendation(
    *, action: str, reason: str, confidence: Optional[float],
) -> Explanation:

    # Building recommendations (ground_truth.recommendations /
    # advisory_engine's own additional engineering suggestions) already
    # carry one disclosed, deterministic reason string each -- this
    # just wraps it in the same Explanation shape civilian
    # recommendations use, never adding a second, independently-derived
    # reason.

    return Explanation(summary=action, reasons=(reason,), confidence=confidence)
