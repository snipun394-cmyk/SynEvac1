from typing import Tuple

from emergency_response.models import ResponsePriorityLevel

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# Warden Dispatch -- primary, always-available source is emergency_
# response's own zone priority levels and assistance counts (this
# category is the one most often left advisory-absent live, per the
# pre-implementation dependency analysis, so it must work from
# emergency_response_snapshot alone). advisory_system's firefighter_
# intelligence live_priority_zone_ids/live_possible_assistance_zone_ids
# are an independent, thin, enrichment-only emission path -- never the
# sole trigger. Renamed from the brief's "Warden Recommendations" to
# read as an instruction to a human warden, not an automated action --
# this package never dispatches anyone itself.
# =====================================================


_ELEVATED_LEVELS = frozenset((ResponsePriorityLevel.HIGH, ResponsePriorityLevel.CRITICAL))

_LEVEL_TO_PRIORITY = {
    ResponsePriorityLevel.CRITICAL: RecommendationPriority.CRITICAL,
    ResponsePriorityLevel.HIGH: RecommendationPriority.HIGH,
}


def adapt(emergency_response_snapshot, advisory_report) -> Tuple[Recommendation, ...]:

    candidates = []

    if emergency_response_snapshot is not None:

        for zone in emergency_response_snapshot.zones.values():

            if zone.priority_level in _ELEVATED_LEVELS:

                candidates.append(Recommendation(
                    type=RecommendationType.WARDEN_DISPATCH,
                    priority=_LEVEL_TO_PRIORITY[zone.priority_level],
                    severity=zone.priority_level,
                    trigger_condition=TriggerCondition.ZONE_RESPONSE_ELEVATED,
                    affected_zones=(zone.zone_id,),
                    confidence=zone.priority_score,
                    explanation=zone.explanation,
                    technical_reason=f"reason_codes={list(zone.reason_codes)}",
                    supporting_evidence={"priority_score": zone.priority_score, "known_occupant_count": zone.known_occupant_count},
                    recommended_action=f"Dispatch a warden to zone {zone.zone_id} (response priority {zone.priority_level}).",
                    primary_source=RecommendationSource.EMERGENCY_RESPONSE,
                ))

            assistance_count = zone.possible_assistance_count + zone.confirmed_assistance_count + zone.being_assisted_count

            if assistance_count > 0:

                priority = RecommendationPriority.HIGH if zone.confirmed_assistance_count > 0 else RecommendationPriority.MEDIUM

                candidates.append(Recommendation(
                    type=RecommendationType.WARDEN_DISPATCH,
                    priority=priority,
                    trigger_condition=TriggerCondition.ZONE_ASSISTANCE_REQUIRED,
                    affected_zones=(zone.zone_id,),
                    technical_reason=(
                        f"possible={zone.possible_assistance_count} "
                        f"confirmed={zone.confirmed_assistance_count} "
                        f"being_assisted={zone.being_assisted_count}"
                    ),
                    supporting_evidence={
                        "possible_assistance_count": zone.possible_assistance_count,
                        "confirmed_assistance_count": zone.confirmed_assistance_count,
                        "being_assisted_count": zone.being_assisted_count,
                    },
                    recommended_action=f"Send a warden to assist occupant(s) in zone {zone.zone_id}.",
                    primary_source=RecommendationSource.EMERGENCY_RESPONSE,
                ))

    if advisory_report is not None:

        firefighter_intelligence = advisory_report.firefighter_intelligence

        for zone_id in firefighter_intelligence.live_priority_zone_ids:

            candidates.append(Recommendation(
                type=RecommendationType.WARDEN_DISPATCH,
                priority=RecommendationPriority.HIGH,
                trigger_condition=TriggerCondition.ZONE_RESPONSE_ELEVATED,
                affected_zones=(zone_id,),
                technical_reason="advisory_system.firefighter_intelligence.live_priority_zone_ids",
                recommended_action=f"Dispatch a warden to zone {zone_id} (flagged by firefighter intelligence).",
                primary_source=RecommendationSource.ADVISORY_SYSTEM,
            ))

        for zone_id in firefighter_intelligence.live_possible_assistance_zone_ids:

            candidates.append(Recommendation(
                type=RecommendationType.WARDEN_DISPATCH,
                priority=RecommendationPriority.MEDIUM,
                trigger_condition=TriggerCondition.ZONE_ASSISTANCE_REQUIRED,
                affected_zones=(zone_id,),
                technical_reason="advisory_system.firefighter_intelligence.live_possible_assistance_zone_ids",
                recommended_action=f"Send a warden to assist occupant(s) in zone {zone_id}.",
                primary_source=RecommendationSource.ADVISORY_SYSTEM,
            ))

    return tuple(candidates)
