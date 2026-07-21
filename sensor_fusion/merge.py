from collections import Counter
from typing import Sequence

from sensor_fusion.observation import Observation, ObservationKind


_WORST_CASE_MAX_KINDS = (ObservationKind.OCCUPANCY, ObservationKind.TEMPERATURE)
_WORST_CASE_MIN_KINDS = (ObservationKind.VISIBILITY,)
_BOOLEAN_ANY_KINDS = (ObservationKind.SMOKE, ObservationKind.HEAT, ObservationKind.FIRE, ObservationKind.ALARM)


def merge_measurements(kind: ObservationKind, observations: Sequence[Observation]):

    # Sensor Fusion Engine milestone, Phase 5 -- deterministic,
    # life-safety-appropriate merge rules, matching hazard_evolution.
    # merge_strategy.DefaultHazardMergeStrategy's own "worst-case-wins"
    # philosophy applied consistently here rather than reinvented:
    # OCCUPANCY/TEMPERATURE take the MAX across sources (never
    # undercount how many people might be present, or understate how
    # hot a zone is), VISIBILITY takes the MIN (never overstate how far
    # someone could see). SMOKE/HEAT/FIRE/ALARM are booleans -- ANY
    # source alarming means the fused result alarms (a life-safety
    # system must never let one quiet detector silently override
    # another's genuine alarm). BEHAVIOR is categorical -- resolved by
    # majority vote among contributing sources, deterministic ties
    # broken by sorted value.
    #
    # A kind with no rule above (a genuinely unknown, future
    # ObservationKind -- Phase 3's own "Future sensors" requirement)
    # falls back to the single highest-confidence contributing
    # observation's own measurement -- an honest, generic default that
    # never guesses a numeric/categorical merge policy this package has
    # no basis to assume.

    measurements = [observation.measurement for observation in observations]

    if kind in _WORST_CASE_MAX_KINDS:
        return max(measurements)

    if kind in _WORST_CASE_MIN_KINDS:
        return min(measurements)

    if kind in _BOOLEAN_ANY_KINDS:
        return any(bool(measurement) for measurement in measurements)

    if kind == ObservationKind.BEHAVIOR:
        return _majority_vote(measurements)

    return _highest_confidence_value(observations)


def _majority_vote(measurements: Sequence):

    counts = Counter(measurements)
    max_count = max(counts.values())

    tied_candidates = sorted(
        (measurement for measurement, count in counts.items() if count == max_count),
        key=str,
    )

    return tied_candidates[0]


def _highest_confidence_value(observations: Sequence[Observation]):

    best = max(observations, key=lambda observation: (observation.confidence, observation.source))

    return best.measurement
