from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sensor_fusion.confidence import (
    DEFAULT_AGREEMENT_BONUS, DEFAULT_CONFLICT_PENALTY, DEFAULT_SOURCE_WEIGHT,
    DEFAULT_STALENESS_HALF_LIFE_SECONDS, decay_for_staleness, fuse_confidence, weighted_confidence,
)
from sensor_fusion.conflict import detect_conflict
from sensor_fusion.merge import merge_measurements
from sensor_fusion.observation import FusedObservation, Observation, ObservationKind
from sensor_fusion.provider import ObservationProvider


class SensorFusionEngine:

    # Sensor Fusion Engine milestone, Phase 5 -- a PURE fusion engine:
    # collect Observations, group by (location, kind), merge measurements,
    # resolve conflicts, calculate confidence, produce FusedObservations.
    # Deliberately does NOT build BuildingState (Phase 5's own explicit
    # instruction) or anything BuildingState-shaped -- that remains
    # building_state.estimator.BuildingStateEstimator's own job, entirely
    # untouched by this package (see docs/architecture/sensor_fusion.md
    # Sec 7 for how a caller bridges FusedObservations into
    # BuildingStateEstimator's own existing parameters, OUTSIDE this
    # package).
    #
    # Grouping is PER (location, kind) -- never across kinds (Phase 7's
    # own "camera says occupied, smoke detector silent" example: these
    # are two independent FusedObservations, an OCCUPANCY one and a
    # SMOKE one, never compared against each other as if they could
    # agree or conflict).

    def __init__(
        self,
        source_weights: Optional[Mapping[str, float]] = None,
        staleness_half_life_seconds: float = DEFAULT_STALENESS_HALF_LIFE_SECONDS,
        agreement_bonus: float = DEFAULT_AGREEMENT_BONUS,
        conflict_penalty: float = DEFAULT_CONFLICT_PENALTY,
        numeric_conflict_tolerance: Optional[Mapping[ObservationKind, float]] = None,
    ):

        self.source_weights = dict(source_weights or {})
        self.staleness_half_life_seconds = staleness_half_life_seconds
        self.agreement_bonus = agreement_bonus
        self.conflict_penalty = conflict_penalty
        self.numeric_conflict_tolerance = dict(numeric_conflict_tolerance or {})

    # =====================================================
    # Collection (Phase 5's "collect observations")
    # =====================================================

    def collect(self, providers: Sequence[ObservationProvider], time: float) -> Tuple[Observation, ...]:

        observations: List[Observation] = []

        for provider in providers:

            try:
                observations.extend(provider.collect(time))
            except Exception:
                # A single failing provider must never prevent every
                # other provider's evidence from being fused (Phase 9's
                # own "provider failures" test category) -- an honest
                # "this provider contributed nothing this cycle", never
                # a crash.
                continue

        return tuple(observations)

    # =====================================================
    # Fusion (Phase 5's "group by location", "merge", "resolve
    # conflicts", "calculate confidence", "produce fused observations")
    # =====================================================

    def fuse(self, observations: Sequence[Observation], time: float) -> Tuple[FusedObservation, ...]:

        groups: Dict[Tuple[str, ObservationKind], List[Observation]] = {}

        for observation in observations:
            groups.setdefault((observation.location, observation.kind), []).append(observation)

        fused = [
            self._fuse_group(location, kind, group, time)
            for (location, kind), group in groups.items()
        ]

        # Deterministic output order regardless of dict/collection
        # iteration order (Phase 9's own "no randomness" requirement).
        return tuple(sorted(fused, key=lambda f: (f.location, f.kind.name)))

    # =====================================================

    def _fuse_group(self, location: str, kind: ObservationKind, group: List[Observation], time: float) -> FusedObservation:

        # Duplicate observations (Phase 9) -- the SAME source
        # reporting twice for the SAME (location, kind) this cycle is
        # still only ONE source's opinion, never independent
        # corroboration; keep only that source's own best (highest-
        # confidence, then most-recent) entry before any agreement/
        # conflict reasoning runs.
        deduplicated = self._dedupe_by_source(group)

        weighted_confidences = []

        for observation in deduplicated:

            age_seconds = max(0.0, time - observation.timestamp)
            decayed = decay_for_staleness(observation.confidence, age_seconds, self.staleness_half_life_seconds)
            weight = self.source_weights.get(observation.source, DEFAULT_SOURCE_WEIGHT)
            weighted_confidences.append(weighted_confidence(decayed, weight))

        measurements = [observation.measurement for observation in deduplicated]
        tolerance = self.numeric_conflict_tolerance.get(kind)

        conflict = detect_conflict(kind, measurements, tolerance)
        agreement = len(deduplicated) >= 2 and not conflict

        merged_measurement = merge_measurements(kind, deduplicated)
        fused_confidence = fuse_confidence(
            weighted_confidences, agreement, conflict, self.agreement_bonus, self.conflict_penalty,
        )

        return FusedObservation(
            kind=kind,
            location=location,
            timestamp=time,
            measurement=merged_measurement,
            confidence=fused_confidence,
            contributing_sources=tuple(sorted(observation.source for observation in deduplicated)),
            conflict=conflict,
        )

    # =====================================================

    def _dedupe_by_source(self, group: List[Observation]) -> List[Observation]:

        best_by_source: Dict[str, Observation] = {}

        for observation in group:

            existing = best_by_source.get(observation.source)

            if (
                existing is None
                or observation.confidence > existing.confidence
                or (observation.confidence == existing.confidence and observation.timestamp > existing.timestamp)
            ):
                best_by_source[observation.source] = observation

        return list(best_by_source.values())
