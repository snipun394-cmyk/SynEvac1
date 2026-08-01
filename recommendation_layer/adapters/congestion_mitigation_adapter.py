from typing import Tuple

from evacuation_recommendation.models import RecommendationReason

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# Congestion Mitigation -- primary source is evacuation_recommendation's
# own congestion-flavored reason codes (always available live);
# crowd_intelligence's building-wide congested-asset lists and
# advisory_system's congestion-flavored BuildingRecommendation entries
# are independent, thin, enrichment-only emission paths -- never the
# sole trigger, matching this codebase's own "AI/Advisory support,
# never drive" discipline. RecommendationManager, not this adapter,
# collapses same-zone/exit hits from multiple sources into one
# Recommendation.
# =====================================================


_CONGESTION_REASON_CODES = frozenset((
    RecommendationReason.HIGH_CONGESTION,
    RecommendationReason.QUEUE_PRESENT,
    RecommendationReason.LOW_THROUGHPUT,
))

_HIGH_CONGESTION_ASSET_LEVELS = frozenset(("HIGH", "VERY_HIGH", "CRITICAL"))


def adapt(evacuation_recommendation_snapshot, crowd_intelligence_snapshot, advisory_report) -> Tuple[Recommendation, ...]:

    candidates = []

    if evacuation_recommendation_snapshot is not None:

        for zone in evacuation_recommendation_snapshot.zones.values():

            zone_reason_hit = set(zone.reason_codes) & _CONGESTION_REASON_CODES

            for candidate in zone.candidates:

                candidate_hit = (
                    set(candidate.reason_codes) & _CONGESTION_REASON_CODES
                    or (candidate.congestion_level in _HIGH_CONGESTION_ASSET_LEVELS)
                )

                if not (zone_reason_hit or candidate_hit):
                    continue

                priority = (
                    RecommendationPriority.HIGH if RecommendationReason.QUEUE_PRESENT in candidate.reason_codes
                    else RecommendationPriority.MEDIUM
                )

                candidates.append(Recommendation(
                    type=RecommendationType.CONGESTION_MITIGATION,
                    priority=priority,
                    trigger_condition=TriggerCondition.ZONE_HIGH_CONGESTION,
                    affected_zones=(zone.zone_id,),
                    affected_exits=(candidate.exit_id,),
                    confidence=zone.confidence,
                    technical_reason=f"reason_codes={list(candidate.reason_codes)}",
                    supporting_evidence={
                        "congestion_level": candidate.congestion_level,
                        "queue_candidate_count": candidate.queue_candidate_count,
                        "throughput_per_minute": candidate.throughput_per_minute,
                    },
                    recommended_action=f"Consider redistributing zone {zone.zone_id} away from exit {candidate.exit_id}.",
                    primary_source=RecommendationSource.EVACUATION_RECOMMENDATION,
                ))

    if crowd_intelligence_snapshot is not None:

        summary = crowd_intelligence_snapshot.building_summary

        for exit_id in summary.congested_exits:

            candidates.append(Recommendation(
                type=RecommendationType.CONGESTION_MITIGATION,
                priority=RecommendationPriority.MEDIUM,
                trigger_condition=TriggerCondition.ZONE_HIGH_CONGESTION,
                affected_exits=(exit_id,),
                technical_reason="crowd_intelligence.building_summary.congested_exits",
                recommended_action=f"Exit {exit_id} is reporting building-wide congestion.",
                primary_source=RecommendationSource.CROWD_INTELLIGENCE,
            ))

    if advisory_report is not None:

        for entry in advisory_report.building_recommendations:

            is_congestion = "congest" in entry.action.lower() or "congest" in entry.reason.lower()

            if not is_congestion or entry.target_id is None:
                continue

            affected_exits = (entry.target_id,) if entry.target_type in ("exit", "door", "stair") else ()
            affected_zones = () if affected_exits else (entry.target_id,)

            candidates.append(Recommendation(
                type=RecommendationType.CONGESTION_MITIGATION,
                priority=RecommendationPriority.MEDIUM,
                trigger_condition=TriggerCondition.ZONE_HIGH_CONGESTION,
                affected_zones=affected_zones,
                affected_exits=affected_exits,
                confidence=entry.confidence,
                explanation=entry.reason,
                technical_reason=entry.action,
                recommended_action=entry.action,
                primary_source=RecommendationSource.ADVISORY_SYSTEM,
            ))

    return tuple(candidates)
