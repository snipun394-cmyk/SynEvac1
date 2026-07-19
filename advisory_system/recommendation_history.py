from typing import Dict, List, Optional, Tuple

from advisory_system.recommendation_models import RecommendationChange


# =====================================================
# Recommendation History. An append-only, in-memory log of every zone-
# level recommendation the Advisory System has issued this session,
# plus a derived change-log entry whenever a new recommendation for an
# already-seen zone differs from its own immediately-previous one --
# Phase 7's own worked example ("10:03 Zone 2 Use Exit 1 -> 10:05
# Changed to Exit 3, Reason: Smoke spread detected"). This class only
# stores what generate_report() actually produced; it never persists or
# reloads a session (a caller wanting that wraps this class, the same
# "data structure, not a storage engine" scope every other in-memory
# record collection in this codebase already keeps).
# =====================================================


class RecommendationHistory:

    def __init__(self):

        self._last_by_zone: Dict[str, Tuple[str, Optional[float]]] = {}
        self._changes: List[RecommendationChange] = []

    # =====================================================

    def record(
        self,
        *,
        timestamp: float,
        zone_id: str,
        recommendation: str,
        confidence: Optional[float],
        reason_for_change: str = "",
    ) -> Optional[RecommendationChange]:

        # Returns the RecommendationChange this call produced, or None
        # if this is the first recommendation ever seen for this zone,
        # or if it is identical to the zone's own immediately-previous
        # recommendation (not a change -- reissuing the same guidance
        # is not logged as one, keeping the change-log genuinely "what
        # changed," not "every announcement ever made").

        previous = self._last_by_zone.get(zone_id)
        self._last_by_zone[zone_id] = (recommendation, confidence)

        if previous is None or previous[0] == recommendation:
            return None

        change = RecommendationChange(
            timestamp=timestamp,
            zone_id=zone_id,
            previous_recommendation=previous[0],
            new_recommendation=recommendation,
            reason_for_change=reason_for_change,
            confidence_before=previous[1],
            confidence_after=confidence,
        )

        self._changes.append(change)

        return change

    # =====================================================

    @property
    def changes(self) -> Tuple[RecommendationChange, ...]:

        return tuple(self._changes)

    # =====================================================

    def changes_for(self, zone_id: str) -> Tuple[RecommendationChange, ...]:

        return tuple(change for change in self._changes if change.zone_id == zone_id)

    # =====================================================

    def current_recommendation(self, zone_id: str) -> Optional[str]:

        entry = self._last_by_zone.get(zone_id)

        return entry[0] if entry is not None else None
