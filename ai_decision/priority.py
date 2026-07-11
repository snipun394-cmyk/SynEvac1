from typing import List

from hazard.severity import HazardSeverity

from ai_decision.recommendation import PriorityEvacuationEntry, ZoneRecommendation

# A single, explicit severity ordering -- deliberately not relying on
# HazardSeverity's own auto()-assigned enum values (an implementation
# detail of declaration order, not a documented contract), so this
# stays correct even if HazardSeverity's declaration order ever
# changes. The one place both the unsafe-zone threshold check
# (ai_decision/engine.py) and priority ranking below compare
# severities against each other.
SEVERITY_RANK = {
    HazardSeverity.NONE: 0,
    HazardSeverity.LOW: 1,
    HazardSeverity.MODERATE: 2,
    HazardSeverity.HIGH: 3,
    HazardSeverity.CRITICAL: 4,
}


class EvacuationPriorityRule:

    # The one interface AIDecisionEngine delegates to for "in what
    # order should these zones evacuate" -- mirrors TenabilityScorer/
    # FireGrowthCurve's role: the engine that consumes this embeds no
    # ranking policy of its own, so a future RL-based ranker plugs in
    # without touching AIDecisionEngine at all (the "augment without
    # replacing" swap point the milestone asks for).
    #
    # Must be a pure function of its input: the same list of
    # ZoneRecommendation objects always produces the same order, no
    # side effects, no dependency on anything but the zones handed in.

    def rank(self, zone_recommendations: List[ZoneRecommendation]) -> List[PriorityEvacuationEntry]:

        raise NotImplementedError


class SeverityOccupancyPriorityRule(EvacuationPriorityRule):

    # V1's whole ranking policy: highest severity first, then highest
    # known occupant count, then zone_id for full determinism when
    # both are tied. A documented placeholder -- not a validated
    # life-safety prioritization model -- same honesty as
    # HazardSeverity's own score cutoffs. occupant_count=None (no
    # sensor reading) sorts as if it were 0, never crashes the
    # comparison and never outranks a zone with a known headcount.

    def rank(self, zone_recommendations):

        ranked = sorted(
            zone_recommendations,
            key=lambda rec: (
                -SEVERITY_RANK[rec.severity],
                -(rec.occupant_count or 0.0),
                rec.zone_id,
            ),
        )

        return [
            PriorityEvacuationEntry(
                rank=index + 1,
                zone_id=rec.zone_id,
                reason=self._reason(rec),
            )
            for index, rec in enumerate(ranked)
        ]

    # =====================================================

    def _reason(self, rec: ZoneRecommendation) -> str:

        occupant_clause = (
            f"{int(rec.occupant_count)} occupant(s)"
            if rec.occupant_count is not None
            else "occupancy unknown"
        )

        reachability_clause = (
            "" if rec.is_reachable else "; no safe route to an exit"
        )

        return f"{rec.severity.name} hazard, {occupant_clause}{reachability_clause}"
