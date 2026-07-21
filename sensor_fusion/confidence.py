from typing import Sequence


# Sensor Fusion Engine milestone, Phase 6 -- named defaults, never
# inline magic numbers. Entirely deterministic arithmetic -- no ML,
# no learned weighting, exactly Phase 6's own "No ML" requirement.
DEFAULT_SOURCE_WEIGHT = 1.0
DEFAULT_STALENESS_HALF_LIFE_SECONDS = 30.0
DEFAULT_AGREEMENT_BONUS = 0.15
DEFAULT_CONFLICT_PENALTY = 0.3


def decay_for_staleness(confidence: float, age_seconds: float, half_life_seconds: float) -> float:

    # Exponential decay -- confidence halves every half_life_seconds
    # of age, never goes negative, never needs a hard cutoff. age_seconds
    # <= 0 (a same-instant or even slightly-in-the-future observation --
    # see engine.py's own clamping for "late observations") means no
    # decay at all, full confidence retained.

    if half_life_seconds <= 0 or age_seconds <= 0:
        return confidence

    return confidence * (0.5 ** (age_seconds / half_life_seconds))


def weighted_confidence(confidence: float, source_weight: float) -> float:

    # source_weight (Phase 6's "source weighting") lets a caller trust
    # one provider more than another (e.g. a calibrated smoke detector
    # over an unverified manual report) -- clamped to [0.0, 1.0] since
    # a weight above 1.0 or a decayed confidence combination could
    # otherwise push the result out of Observation's own valid range.

    return max(0.0, min(1.0, confidence * source_weight))


def fuse_confidence(
    weighted_confidences: Sequence[float],
    agreement: bool,
    conflict: bool,
    agreement_bonus: float = DEFAULT_AGREEMENT_BONUS,
    conflict_penalty: float = DEFAULT_CONFLICT_PENALTY,
) -> float:

    # The plain average of every contributing (staleness-decayed,
    # source-weighted) confidence, then EITHER an agreement bonus
    # (>= 2 sources, and they did not conflict) OR a conflict penalty
    # (sources disagreed) -- never both, and neither when there is
    # only a single contributing source: Phase 7's own "heat detector
    # alarm, camera unavailable -> retain confidence honestly" example
    # is exactly this single-source case, where the average is already
    # the whole, honest answer -- no bonus for corroboration that never
    # happened, no penalty for a disagreement that never happened
    # either.

    if not weighted_confidences:
        return 0.0

    base = sum(weighted_confidences) / len(weighted_confidences)

    if conflict:
        return max(0.0, base - conflict_penalty)

    if agreement:
        return min(1.0, base + agreement_bonus)

    return base
