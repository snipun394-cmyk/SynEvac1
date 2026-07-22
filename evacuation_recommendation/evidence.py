# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone -- small,
# internal evidence-gathering helpers used by engine.py. NOT to be
# confused with advisory_system.evacuation_recommendation_evidence
# (the ADVISORY-facing plain-value evidence object one layer up) --
# this module stays entirely internal to this package.
# =====================================================


def zone_coverage_fraction(zone_id, crowd_snapshot):

    if crowd_snapshot is None:
        return None

    metrics = crowd_snapshot.zone(zone_id)

    return metrics.position_coverage_fraction if metrics is not None else None


def hazard_evidence_available(building_state) -> bool:

    return (
        building_state is not None
        and building_state.hazard_summary is not None
        and len(building_state.hazard_summary.zone_severities) > 0
    )


def ai_bottleneck_probability(ai_prediction_snapshot):

    # Duck-typed against live_system.live_ai_gateway.LiveAIPredictionSnapshot
    # -- deliberately no import of that type (or of advisory_system.
    # ai_evidence.AIDecisionEvidence) anywhere in this package, mirroring
    # emergency_response.engine's own duck-typed consumption of sibling
    # snapshots. A single, BUILDING-WIDE value (never per-exit -- see
    # advisory_system.ai_evidence's own documented boundary); scoring.py
    # applies it uniformly across every candidate for a zone, so it can
    # never change which exit ranks higher (Phase 7/13's own "AI only
    # supports" requirement).

    if ai_prediction_snapshot is None:
        return None

    bottleneck = getattr(ai_prediction_snapshot, "bottleneck", None)

    if bottleneck is None:
        return None

    return getattr(bottleneck, "probability", None)
