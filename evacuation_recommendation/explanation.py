from evacuation_recommendation.models import RecommendationReason, RecommendationStatus


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 5 --
# every recommendation must explain itself in plain language, built
# ENTIRELY from the same disclosed reason codes scoring.py already
# produced -- no opaque score is ever the only explanation. Mirrors
# emergency_response.engine._explain()'s own convention exactly.
# =====================================================


_REASON_TEXT = {
    RecommendationReason.SHORTEST_SAFE_ROUTE: "Shortest safe route",
    RecommendationReason.LOW_CONGESTION: "Low congestion",
    RecommendationReason.HIGH_CONGESTION: "High congestion",
    RecommendationReason.QUEUE_PRESENT: "Queue present",
    RecommendationReason.HIGH_THROUGHPUT: "High throughput",
    RecommendationReason.LOW_THROUGHPUT: "Low throughput",
    RecommendationReason.TRAJECTORY_PROGRESSING_NORMALLY: "Trajectory progressing normally",
    RecommendationReason.TRAJECTORY_MOVING_AWAY: "Trajectory moving away from this exit",
    RecommendationReason.EMERGENCY_RESPONSE_ZONE_ELEVATED: "Elevated emergency response priority near this exit",
    RecommendationReason.AI_BOTTLENECK_RISK_LOW: "AI bottleneck risk LOW",
    RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED: "AI bottleneck risk ELEVATED",
    RecommendationReason.GOOD_COVERAGE: "Good observation coverage",
    RecommendationReason.POOR_COVERAGE: "Poor observation coverage",
    RecommendationReason.NO_SAFE_EXIT_REACHABLE: "No safe exit is currently reachable from this zone",
}


def explain_recommendation(zone_id, status, recommended_exit_id, reason_codes, coverage_fraction) -> str:

    if status == RecommendationStatus.NO_SAFE_EXIT_AVAILABLE:
        return f"Zone {zone_id}: no safe exit currently reachable."

    readable = ", ".join(_REASON_TEXT.get(code, code) for code in reason_codes)
    coverage_note = f" Coverage {coverage_fraction:.0%}." if coverage_fraction is not None else ""

    if not readable:
        return f"Zone {zone_id} -- recommend Exit {recommended_exit_id}.{coverage_note}"

    return f"Zone {zone_id} -- recommend Exit {recommended_exit_id}. Reasons: {readable}.{coverage_note}"
