from typing import Tuple

from evacuation_recommendation.models import RecommendationReason, RecommendationStatus as _UpstreamStatus

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# System Warning -- about the health of the recommendation pipeline
# itself, not the incident. `ai_prediction_snapshot` is accepted for
# signature completeness/future use, but AI_BOTTLENECK_RISK_ELEVATED
# is already folded into evacuation_recommendation's own reason_codes
# (its engine already applies AI support before this layer ever sees
# it) -- reading it there, rather than recomputing anything from the
# raw prediction, keeps this adapter a pure passthrough.
# =====================================================


LOW_CONFIDENCE_THRESHOLD = 0.4
LOW_COVERAGE_THRESHOLD = 0.5
BUILDING_WIDESPREAD_NO_SAFE_EXIT_ZONE_COUNT = 3


def adapt(evacuation_recommendation_snapshot, evacuation_guidance_snapshot, ai_prediction_snapshot=None) -> Tuple[Recommendation, ...]:

    candidates = []

    no_safe_exit_zone_ids = []

    if evacuation_recommendation_snapshot is not None:

        for zone in evacuation_recommendation_snapshot.zones.values():

            if zone.status == _UpstreamStatus.NO_SAFE_EXIT_AVAILABLE:
                no_safe_exit_zone_ids.append(zone.zone_id)

            low_confidence = zone.confidence is not None and zone.confidence < LOW_CONFIDENCE_THRESHOLD
            low_coverage = zone.coverage_fraction is not None and zone.coverage_fraction < LOW_COVERAGE_THRESHOLD
            poor_coverage_flagged = RecommendationReason.POOR_COVERAGE in zone.reason_codes

            if low_confidence or low_coverage or poor_coverage_flagged:

                candidates.append(Recommendation(
                    type=RecommendationType.SYSTEM_WARNING,
                    priority=RecommendationPriority.MEDIUM,
                    trigger_condition=TriggerCondition.RECOMMENDATION_LOW_CONFIDENCE,
                    affected_zones=(zone.zone_id,),
                    confidence=zone.confidence,
                    technical_reason=f"confidence={zone.confidence} coverage_fraction={zone.coverage_fraction}",
                    supporting_evidence={"confidence": zone.confidence, "coverage_fraction": zone.coverage_fraction},
                    recommended_action=f"Treat zone {zone.zone_id}'s recommendation with caution -- confidence/coverage is low.",
                    primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
                ))

            if RecommendationReason.AI_BOTTLENECK_RISK_ELEVATED in zone.reason_codes:

                candidates.append(Recommendation(
                    type=RecommendationType.SYSTEM_WARNING,
                    priority=RecommendationPriority.LOW,
                    trigger_condition=TriggerCondition.RECOMMENDATION_AI_BOTTLENECK_RISK,
                    affected_zones=(zone.zone_id,),
                    technical_reason="reason_codes contains AI_BOTTLENECK_RISK_ELEVATED",
                    recommended_action=f"Predictive AI flags elevated bottleneck risk near zone {zone.zone_id} (support-only, not decision-driving).",
                    primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
                ))

        if len(no_safe_exit_zone_ids) >= BUILDING_WIDESPREAD_NO_SAFE_EXIT_ZONE_COUNT:

            candidates.append(Recommendation(
                type=RecommendationType.SYSTEM_WARNING,
                priority=RecommendationPriority.CRITICAL,
                trigger_condition=TriggerCondition.BUILDING_NO_SAFE_EXIT_WIDESPREAD,
                affected_zones=tuple(no_safe_exit_zone_ids),
                technical_reason=f"{len(no_safe_exit_zone_ids)} occupied zones have no safe exit available",
                recommended_action="Widespread loss of safe exits across the building -- escalate immediately.",
                primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
            ))

    if evacuation_guidance_snapshot is not None:

        for zone_id, plan in evacuation_guidance_snapshot.zones.items():

            if not plan.inconsistencies:
                continue

            candidates.append(Recommendation(
                type=RecommendationType.SYSTEM_WARNING,
                priority=RecommendationPriority.MEDIUM,
                trigger_condition=TriggerCondition.GUIDANCE_INCONSISTENCY,
                affected_zones=(zone_id,),
                confidence=plan.confidence,
                technical_reason=f"inconsistencies={list(plan.inconsistencies)}",
                supporting_evidence={"inconsistencies": list(plan.inconsistencies)},
                recommended_action=f"Evacuation guidance for zone {zone_id} has an unresolved inconsistency.",
                primary_source=RecommendationSource.EVACUATION_GUIDANCE,
            ))

    return tuple(candidates)
