from dataclasses import dataclass
from typing import Optional, Tuple

from perception.models.human_observation import HumanClassification, HumanState


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 7/8
# -- the ONE place multi-camera evidence conflict and temporal
# staleness are resolved, deterministically, for both HumanClassification
# and HumanState. Pure functions only: no mutation, no randomness, no
# dependency on iteration/call order (Phase 23 tests 17/18) -- every
# function here is a pure reduction of (existing evidence, new evidence)
# to (next evidence), callable in any order and always converging to the
# same final answer for a fixed set of readings (proven in
# tests/test_human_evidence.py::DeterministicReconciliationTests).
#
# THE RULE (disclosed, documented, never fabricated):
#   1. A new reading of UNKNOWN/None (no genuine evidence this cycle)
#      NEVER overwrites existing, known evidence -- staleness (below) is
#      the only mechanism that ever reverts a stale value, never an
#      immediate overwrite by an empty reading (Phase 8).
#   2. Known beats unknown: if there is currently no known evidence at
#      all, ANY genuine new reading is simply adopted.
#   3. Agreement: a new reading that matches the existing value only
#      refreshes recency/confidence -- never "resets" anything.
#   4. Genuine conflict between two DIFFERENT known values is resolved
#      by comparing (timestamp, confidence, source) as a strict total
#      order: a STRICTLY LATER timestamp always wins outright (temporal
#      recency dominates for a real, sequential re-observation -- this
#      is what lets HumanState update readily, WALKING -> RUNNING ->
#      FALLEN -> BEING_ASSISTED, Phase 6/23 tests 22-24); at EQUAL
#      timestamps (a genuine same-cycle conflict between two cameras),
#      the reading with the higher confidence wins, and on a further
#      tie (or either confidence unavailable), the LEXICOGRAPHICALLY
#      SMALLER camera_id/source wins -- a fixed, content-addressed
#      tie-break, never dependent on which detection happened to be
#      processed first (Phase 23 tests 17/18).
# =====================================================


@dataclass(frozen=True)
class HumanEvidenceConfig:

    # Documented, configurable project assumptions -- same disclosure
    # discipline as crowd_intelligence.models.DensityThresholds/
    # emergency_response.models.ResponseWeights. Classification changes
    # slowly in reality (a person's demographic/equipment classification
    # does not change cycle to cycle) so it is tolerated for much longer
    # without fresh confirmation than HumanState, which describes what a
    # person is doing RIGHT NOW and must not be trusted long after the
    # last genuine sighting (Phase 8's own "treat them differently"
    # requirement).

    classification_staleness_seconds: float = 300.0
    state_staleness_seconds: float = 10.0


def _effective_confidence(confidence: Optional[float]) -> float:

    # An unavailable confidence is treated as the lowest possible value
    # for comparison purposes ONLY (never surfaced as a fabricated
    # number itself) -- "known-with-confidence beats known-without-
    # confidence" when two genuinely different values conflict.

    return confidence if confidence is not None else -1.0


def _new_evidence_wins(
    *,
    new_timestamp: float, new_confidence: Optional[float], new_source: Optional[str],
    existing_timestamp: Optional[float], existing_confidence: Optional[float], existing_source: Optional[str],
) -> bool:

    if existing_timestamp is None:
        return True

    if new_timestamp > existing_timestamp:
        return True

    if new_timestamp < existing_timestamp:
        return False

    new_conf = _effective_confidence(new_confidence)
    existing_conf = _effective_confidence(existing_confidence)

    if new_conf != existing_conf:
        return new_conf > existing_conf

    if new_source is None or existing_source is None:
        # No comparable, distinguishing source to break the tie
        # honestly -- keep whatever is already held rather than
        # guessing.
        return False

    return new_source < existing_source


# =====================================================


def reconcile_classification(
    *,
    existing_classification: HumanClassification,
    existing_confidence: Optional[float],
    existing_source: Optional[str],
    existing_last_observed_at: Optional[float],
    new_classification: HumanClassification,
    new_confidence: Optional[float],
    new_source: Optional[str],
    timestamp: float,
) -> Tuple[HumanClassification, Optional[float], Optional[str], Optional[float]]:

    if new_classification == HumanClassification.UNKNOWN:
        return existing_classification, existing_confidence, existing_source, existing_last_observed_at

    if existing_classification == HumanClassification.UNKNOWN:
        return new_classification, new_confidence, new_source, timestamp

    if new_classification == existing_classification:
        return new_classification, new_confidence, new_source, timestamp

    if _new_evidence_wins(
        new_timestamp=timestamp, new_confidence=new_confidence, new_source=new_source,
        existing_timestamp=existing_last_observed_at, existing_confidence=existing_confidence,
        existing_source=existing_source,
    ):
        return new_classification, new_confidence, new_source, timestamp

    return existing_classification, existing_confidence, existing_source, existing_last_observed_at


def reconcile_state(
    *,
    existing_state: Optional[HumanState],
    existing_confidence: Optional[float],
    existing_source: Optional[str],
    existing_last_observed_at: Optional[float],
    new_state: Optional[HumanState],
    new_confidence: Optional[float],
    new_source: Optional[str],
    timestamp: float,
) -> Tuple[Optional[HumanState], Optional[float], Optional[str], Optional[float]]:

    if new_state is None:
        return existing_state, existing_confidence, existing_source, existing_last_observed_at

    if existing_state is None:
        return new_state, new_confidence, new_source, timestamp

    if new_state == existing_state:
        return new_state, new_confidence, new_source, timestamp

    if _new_evidence_wins(
        new_timestamp=timestamp, new_confidence=new_confidence, new_source=new_source,
        existing_timestamp=existing_last_observed_at, existing_confidence=existing_confidence,
        existing_source=existing_source,
    ):
        return new_state, new_confidence, new_source, timestamp

    return existing_state, existing_confidence, existing_source, existing_last_observed_at


# =====================================================
# Staleness / expiry (Phase 8) -- applied independently of reconcile_*()
# above, so an occupant who simply stops being reconfirmed (never
# contradicted, just no longer observed) still honestly reverts to
# UNKNOWN/None once too much real time has passed, rather than being
# treated as permanently known forever.
# =====================================================


def apply_classification_staleness(
    classification: HumanClassification, confidence: Optional[float], source: Optional[str],
    last_observed_at: Optional[float], *, now: float, config: HumanEvidenceConfig,
) -> Tuple[HumanClassification, Optional[float], Optional[str], Optional[float]]:

    if classification == HumanClassification.UNKNOWN or last_observed_at is None:
        return classification, confidence, source, last_observed_at

    if (now - last_observed_at) > config.classification_staleness_seconds:
        return HumanClassification.UNKNOWN, None, None, None

    return classification, confidence, source, last_observed_at


def apply_state_staleness(
    state: Optional[HumanState], confidence: Optional[float], source: Optional[str],
    last_observed_at: Optional[float], *, now: float, config: HumanEvidenceConfig,
) -> Tuple[Optional[HumanState], Optional[float], Optional[str], Optional[float]]:

    if state is None or last_observed_at is None:
        return state, confidence, source, last_observed_at

    if (now - last_observed_at) > config.state_staleness_seconds:
        return None, None, None, None

    return state, confidence, source, last_observed_at
