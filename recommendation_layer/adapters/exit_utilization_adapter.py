from collections import Counter
from typing import Tuple

from recommendation_layer.models import (
    Recommendation, RecommendationPriority, RecommendationSource, RecommendationType, TriggerCondition,
)


# =====================================================
# Exit Utilization -- the ONE category with no upstream provider of
# its own: no existing package aggregates ACROSS zones to notice a
# building-wide redistribution opportunity. Deliberately kept to a
# single, named, documented threshold pair below (never a tunable
# scoring formula) -- this is presentation-threshold tuning, not
# recommendation intelligence. Never re-ranks a zone's own
# recommendation; it only flags an opportunity for an operator to
# consider on top of evacuation_recommendation's own, unmodified,
# per-zone ranking.
# =====================================================


EXIT_OVERUTILIZATION_RATIO = 2.0
EXIT_OVERUTILIZATION_MIN_ZONES = 3


def adapt(evacuation_recommendation_snapshot) -> Tuple[Recommendation, ...]:

    if evacuation_recommendation_snapshot is None:
        return ()

    zones_by_exit = {}
    alternatives_by_exit = {}

    for zone in evacuation_recommendation_snapshot.zones.values():

        if not zone.recommended_exit_id:
            continue

        zones_by_exit.setdefault(zone.recommended_exit_id, []).append(zone.zone_id)
        alternatives_by_exit.setdefault(zone.recommended_exit_id, set()).update(zone.alternative_exit_ids)

    load_by_exit = Counter({exit_id: len(zone_ids) for exit_id, zone_ids in zones_by_exit.items()})

    candidates = []
    already_flagged = set()

    for exit_id, load in load_by_exit.items():

        if load < EXIT_OVERUTILIZATION_MIN_ZONES or exit_id in already_flagged:
            continue

        for alternative_id in alternatives_by_exit.get(exit_id, ()):

            if alternative_id == exit_id:
                continue

            alternative_load = load_by_exit.get(alternative_id, 0)

            if alternative_load > 0 and load < EXIT_OVERUTILIZATION_RATIO * alternative_load:
                continue

            already_flagged.add(exit_id)

            candidates.append(Recommendation(
                type=RecommendationType.EXIT_UTILIZATION,
                priority=RecommendationPriority.MEDIUM,
                trigger_condition=TriggerCondition.EXIT_OVERUTILIZED,
                affected_zones=tuple(zones_by_exit[exit_id]),
                affected_exits=(exit_id,),
                technical_reason=f"{load} zones routed to exit {exit_id} vs {alternative_load} to safe alternative {alternative_id}",
                supporting_evidence={"load": load, "alternative_exit_id": alternative_id, "alternative_load": alternative_load},
                recommended_action=f"Consider redistributing some of exit {exit_id}'s occupant load toward exit {alternative_id}.",
                primary_source=RecommendationSource.RECOMMENDATION_LAYER,
            ))

            candidates.append(Recommendation(
                type=RecommendationType.EXIT_UTILIZATION,
                priority=RecommendationPriority.INFO,
                trigger_condition=TriggerCondition.EXIT_UNDERUTILIZED_ALTERNATIVE,
                affected_exits=(alternative_id,),
                technical_reason=f"{alternative_load} zones routed to exit {alternative_id} vs {load} to overutilized exit {exit_id}",
                supporting_evidence={"load": alternative_load, "overutilized_exit_id": exit_id, "overutilized_load": load},
                recommended_action=f"Exit {alternative_id} has spare capacity relative to exit {exit_id}.",
                primary_source=RecommendationSource.RECOMMENDATION_LAYER,
            ))

            break

    return tuple(candidates)
