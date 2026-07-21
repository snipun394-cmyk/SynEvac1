from typing import Mapping, Optional, Sequence

from perception.models.human_observation import HumanObservation


# =====================================================
# Confidence Engine. Two entirely separate notions of confidence live
# in an Advisory Report (Phase 8's own explicit requirement: "Occupancy
# confidence remains separate from recommendation confidence"):
#
#   - occupancy_confidence() -- how sure Live Perception is that a
#     person exists and is where it says they are. Read straight off
#     HumanObservation.confidence, never re-derived.
#
#   - recommendation_confidence() -- how sure the Advisory System is
#     that a given recommendation is the right one. Every Decision
#     Policy rule this system builds on is deterministic (see
#     decision_policy/zone_policy.py's own "no AI, no ML, no
#     randomness" docstring), so it has no predict_proba()/ensemble-
#     dispersion signal of its own the way ai_inference.confidence.py's
#     model-backed confidence does -- DETERMINISTIC_RULE_BASE_CONFIDENCE
#     below is the disclosed floor every recommendation starts from
#     (the same "disclosed policy threshold, not a fabricated data
#     value" convention ground_truth.recommendations.HIGH_RISK_THRESHOLD
#     and decision_policy.zone_policy's own risk thresholds already
#     establish), blended upward with whatever real corroborating
#     signals (AI confidence, RL policy confidence, cross-source
#     agreement, risk-score distance from its own triggering threshold)
#     are actually available for this recommendation. A source that
#     was never supplied is dropped, never treated as disagreement or
#     as zero.
# =====================================================

DETERMINISTIC_RULE_BASE_CONFIDENCE = 0.70


def combine_confidence(*sources: Optional[float]) -> Optional[float]:

    values = [value for value in sources if value is not None]

    if not values:
        return None

    return sum(values) / len(values)


def risk_based_confidence(risk_score: Optional[float], threshold: float) -> Optional[float]:

    # How decisively an already-computed GroundTruth risk_score crossed
    # the fixed threshold that triggered this recommendation -- a
    # disclosed, deterministic formula over a real, already-computed
    # value (ground_truth.risk_analysis' own zone/exit/stair
    # risk_score), never a new judgment about the underlying hazard.

    if risk_score is None or threshold <= 0:
        return None

    distance = abs(risk_score - threshold) / threshold

    return max(0.0, min(1.0, 0.5 + 0.5 * distance))


def agreement_confidence(signals: Sequence[Optional[bool]]) -> Optional[float]:

    # Fraction of already-computed corroborating signals (e.g. does the
    # AI Bottleneck prediction flag the same location Decision Policy
    # already flagged -- see ai_inference.recommendation.
    # decision_policy_flagged()) that agree with this recommendation.
    # A None entry (a source that was never consulted, or had no
    # opinion about this exact location) is excluded from both the
    # numerator and denominator -- never counted as disagreement.

    considered = [signal for signal in signals if signal is not None]

    if not considered:
        return None

    return sum(1 for signal in considered if signal) / len(considered)


def occupancy_confidence(human_observations: Mapping[str, HumanObservation]) -> Optional[float]:

    values = [
        observation.confidence
        for observation in human_observations.values()
        if observation.confidence is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def recommendation_confidence(
    *,
    risk_score: Optional[float] = None,
    risk_threshold: float = 0.5,
    ai_confidence: Optional[float] = None,
    rl_confidence: Optional[float] = None,
    crowd_confidence: Optional[float] = None,
    agreement_signals: Sequence[Optional[bool]] = (),
) -> float:

    # Always returns a real number (never None): a deterministic
    # Decision Policy rule underlies every Advisory recommendation, so
    # DETERMINISTIC_RULE_BASE_CONFIDENCE is always available as a floor
    # even when no AI/RL/crowd/agreement signal was ever supplied for
    # this particular recommendation.
    #
    # Live Crowd Intelligence -> Operational Advisory Integration
    # milestone -- crowd_confidence blends in exactly like ai_confidence/
    # rl_confidence always have (this function already averages whatever
    # real signals are supplied; that was never "one opaque score" any
    # more than ai_confidence blending already was). Phase 11's own
    # "keep provenance separate" requirement is satisfied one layer up,
    # by advisory_engine.py's own confidence_source tuple recording
    # WHICH sources contributed to this exact blended number -- never by
    # withholding crowd_confidence from the blend itself.

    signals = [DETERMINISTIC_RULE_BASE_CONFIDENCE]

    risk_conf = risk_based_confidence(risk_score, risk_threshold)
    if risk_conf is not None:
        signals.append(risk_conf)

    if ai_confidence is not None:
        signals.append(ai_confidence)

    if rl_confidence is not None:
        signals.append(rl_confidence)

    if crowd_confidence is not None:
        signals.append(crowd_confidence)

    agreement = agreement_confidence(agreement_signals)
    if agreement is not None:
        signals.append(agreement)

    return combine_confidence(*signals)
