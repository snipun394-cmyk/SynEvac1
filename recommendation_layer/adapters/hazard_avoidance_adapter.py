from typing import Tuple

from evacuation_recommendation.models import RecommendationStatus as _UpstreamStatus
from emergency_response.models import ResponseReason

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# Hazard Avoidance -- three independent, thin emission paths, one per
# provider: evacuation_recommendation's own NO_SAFE_EXIT_AVAILABLE
# status (a zone whose deterministic safety evaluation already
# excluded every candidate exit), emergency_response's own HAZARD_
# PRESENT reason code, and advisory_system's commander_dashboard.
# critical_zones (present only when a caller wired an AdvisoryReport --
# see docs/architecture/recommendation_layer.md's disclosed live-
# wiring gap). This adapter never merges these three itself --
# RecommendationManager collapses same-zone hits into one
# Recommendation with full provenance.
# =====================================================


def adapt(evacuation_recommendation_snapshot, emergency_response_snapshot, advisory_report) -> Tuple[Recommendation, ...]:

    candidates = []

    if evacuation_recommendation_snapshot is not None:

        for zone in evacuation_recommendation_snapshot.zones.values():

            if zone.status != _UpstreamStatus.NO_SAFE_EXIT_AVAILABLE:
                continue

            candidates.append(Recommendation(
                type=RecommendationType.HAZARD_AVOIDANCE,
                priority=RecommendationPriority.CRITICAL,
                trigger_condition=TriggerCondition.ZONE_NO_SAFE_EXIT,
                affected_zones=(zone.zone_id,),
                confidence=zone.confidence,
                explanation=zone.explanation,
                technical_reason=f"reason_codes={list(zone.reason_codes)}",
                supporting_evidence={"occupant_count": zone.occupant_count},
                recommended_action=f"Do not route occupants toward zone {zone.zone_id} -- no safe exit is currently reachable.",
                primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
            ))

    if emergency_response_snapshot is not None:

        for zone in emergency_response_snapshot.zones.values():

            if ResponseReason.HAZARD_PRESENT not in zone.reason_codes:
                continue

            candidates.append(Recommendation(
                type=RecommendationType.HAZARD_AVOIDANCE,
                priority=RecommendationPriority.HIGH,
                severity=zone.hazard_severity,
                trigger_condition=TriggerCondition.ZONE_HAZARD_PRESENT,
                affected_zones=(zone.zone_id,),
                explanation=zone.explanation,
                technical_reason=f"reason_codes={list(zone.reason_codes)}",
                supporting_evidence={"priority_score": zone.priority_score},
                recommended_action=f"Avoid routing occupants through zone {zone.zone_id} -- an active hazard is present.",
                primary_source=RecommendationSource.EMERGENCY_RESPONSE,
            ))

    if advisory_report is not None:

        for zone_id in advisory_report.commander_dashboard.critical_zones:

            candidates.append(Recommendation(
                type=RecommendationType.HAZARD_AVOIDANCE,
                priority=RecommendationPriority.HIGH,
                trigger_condition=TriggerCondition.ZONE_HAZARD_PRESENT,
                affected_zones=(zone_id,),
                technical_reason="advisory_system.commander_dashboard.critical_zones",
                recommended_action=f"Avoid routing occupants through zone {zone_id} -- flagged critical by the commander dashboard.",
                primary_source=RecommendationSource.ADVISORY_SYSTEM,
            ))

    return tuple(candidates)
