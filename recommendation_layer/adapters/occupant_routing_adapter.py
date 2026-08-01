from typing import Tuple

from evacuation_recommendation.models import RecommendationReason, RecommendationStatus as _UpstreamStatus

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# Occupant Routing -- a thin, near-verbatim adaptation of evacuation_
# recommendation's own per-zone exit recommendation. This adapter
# introduces no new judgment: priority is a direct passthrough of the
# provider's own reason_codes, never a recomputed score.
# =====================================================


_ELEVATED_REASON_CODES = frozenset((
    RecommendationReason.HIGH_CONGESTION,
    RecommendationReason.QUEUE_PRESENT,
    RecommendationReason.EMERGENCY_RESPONSE_ZONE_ELEVATED,
))


def adapt(evacuation_recommendation_snapshot) -> Tuple[Recommendation, ...]:

    if evacuation_recommendation_snapshot is None:
        return ()

    candidates = []

    for zone in evacuation_recommendation_snapshot.zones.values():

        if zone.status != _UpstreamStatus.RECOMMENDED:
            continue

        priority = (
            RecommendationPriority.MEDIUM if set(zone.reason_codes) & _ELEVATED_REASON_CODES
            else RecommendationPriority.LOW
        )

        candidates.append(Recommendation(
            type=RecommendationType.OCCUPANT_ROUTING,
            priority=priority,
            trigger_condition=TriggerCondition.ZONE_EXIT_RECOMMENDED,
            affected_zones=(zone.zone_id,),
            affected_exits=(zone.recommended_exit_id,) if zone.recommended_exit_id else (),
            confidence=zone.confidence,
            explanation=zone.explanation,
            technical_reason=f"reason_codes={list(zone.reason_codes)}",
            supporting_evidence={
                "ranked_exit_ids": list(zone.ranked_exit_ids),
                "alternative_exit_ids": list(zone.alternative_exit_ids),
                "coverage_fraction": zone.coverage_fraction,
                "occupant_count": zone.occupant_count,
            },
            recommended_action=(
                f"Route zone {zone.zone_id} occupants to exit {zone.recommended_exit_id}."
                if zone.recommended_exit_id else ""
            ),
            primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
        ))

    return tuple(candidates)
