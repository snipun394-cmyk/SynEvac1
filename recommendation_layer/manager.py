import dataclasses
from typing import Dict, Optional, Sequence
from uuid import uuid4

from recommendation_layer.models import (
    PRIORITY_ORDINAL, PROVIDER_PRIORITY_ORDER, Recommendation, RecommendationSet, RecommendationStatus,
)


# =====================================================
# RecommendationManager -- the ONE place in this package that performs
# cross-provider merging, provenance bookkeeping, and lifecycle
# (create/update/expire) for Recommendation objects. Adapters (see
# recommendation_layer/adapters/) never do any of this themselves --
# each adapter call independently emits one candidate per (trigger_
# condition, zone/exit, contributing provider), and this class is the
# only place that decides which candidate "wins" a same-cycle
# collision, records provenance, and tracks identity across cycles.
#
# A Recommendation's own recommendation_id is stable for its entire
# active lifetime -- it is minted exactly once, the first cycle its
# dedup key is seen, and never re-minted while that key keeps
# reappearing (including through the grace-period window below).
# =====================================================


DEFAULT_GRACE_PERIOD_SECONDS = 5.0


def dedup_key(candidate: Recommendation) -> str:

    return "|".join([
        candidate.type,
        candidate.trigger_condition,
        ",".join(sorted(candidate.affected_zones)),
        ",".join(sorted(candidate.affected_exits)),
    ])


def _provider_rank(source: str) -> int:

    if source in PROVIDER_PRIORITY_ORDER:
        return PROVIDER_PRIORITY_ORDER.index(source)

    return len(PROVIDER_PRIORITY_ORDER)


def _merge_group(group: Sequence[Recommendation]) -> Recommendation:

    # This cycle's own same-key collision resolution -- the ONLY place
    # supporting_sources/evidence_origin ever get computed. `group` is
    # never empty (callers only ever build one per non-empty dedup-key
    # bucket).

    ordered = sorted(group, key=lambda c: _provider_rank(c.primary_source))

    winner = ordered[0]

    supporting_sources = tuple(
        candidate.primary_source for candidate in ordered[1:]
        if candidate.primary_source != winner.primary_source
    )

    merged_evidence = dict(winner.supporting_evidence)
    merged_origin = {key: winner.primary_source for key in winner.supporting_evidence}

    for candidate in ordered[1:]:

        for key, value in candidate.supporting_evidence.items():

            if key not in merged_evidence:

                merged_evidence[key] = value
                merged_origin[key] = candidate.primary_source

    return dataclasses.replace(
        winner, supporting_sources=supporting_sources,
        supporting_evidence=merged_evidence, evidence_origin=merged_origin,
    )


def sort_key(recommendation: Recommendation):

    return (
        -PRIORITY_ORDINAL.get(recommendation.priority, 0),
        -recommendation.updated_at,
        recommendation.recommendation_id,
    )


class RecommendationManager:

    def __init__(self, grace_period_seconds: float = DEFAULT_GRACE_PERIOD_SECONDS):

        self.grace_period_seconds = grace_period_seconds

        self._by_dedup_key: Dict[str, Recommendation] = {}

    # =====================================================

    def ingest(self, candidates: Sequence[Recommendation], time: float) -> RecommendationSet:

        grouped: Dict[str, list] = {}

        for candidate in candidates:

            grouped.setdefault(dedup_key(candidate), []).append(candidate)

        merged_by_key = {key: _merge_group(group) for key, group in grouped.items()}

        for key, merged in merged_by_key.items():

            existing = self._by_dedup_key.get(key)

            if existing is not None:

                self._by_dedup_key[key] = dataclasses.replace(
                    merged, recommendation_id=existing.recommendation_id, created_at=existing.created_at,
                    updated_at=time, status=RecommendationStatus.ACTIVE, expires_at=None,
                )

            else:

                self._by_dedup_key[key] = dataclasses.replace(
                    merged, recommendation_id=uuid4().hex[:12], created_at=time, updated_at=time,
                    status=RecommendationStatus.ACTIVE, expires_at=None,
                )

        # Everything previously tracked but absent from this cycle's own
        # candidates moves along the expire clock -- never dropped
        # abruptly, and never blanked without first being returned once
        # more under EXPIRED so a consumer can see the transition.
        for key in list(self._by_dedup_key.keys()):

            if key in merged_by_key:
                continue

            existing = self._by_dedup_key[key]

            if existing.status == RecommendationStatus.EXPIRED:

                del self._by_dedup_key[key]

            elif existing.expires_at is None:

                self._by_dedup_key[key] = dataclasses.replace(existing, expires_at=time + self.grace_period_seconds)

            elif time >= existing.expires_at:

                self._by_dedup_key[key] = dataclasses.replace(
                    existing, status=RecommendationStatus.EXPIRED, updated_at=time,
                )

        recommendations = tuple(sorted(self._by_dedup_key.values(), key=sort_key))

        return RecommendationSet(timestamp=time, recommendations=recommendations)
